// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstdio>

#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_sem_model.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};
std::atomic<bool> g_back_busy{};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    if (!g_back_busy.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::fprintf(stderr, "GPU split owner check: %s on thread %lu while the back runs\n", where,
                 thread);
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        std::fprintf(stderr, "  frame %s+%#llx\n", base_name ? base_name + 1 : name,
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                                                     reinterpret_cast<uintptr_t>(module)));
    }
    std::fflush(stderr);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
#ifdef ARCHITECTURE_x86_64
    // Front macros run compiled (macro_turing_jit.h) only.
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
#endif
    // EDEN_NX2_READ_AHEAD=0: the front's macros drain the back for every DMA read.
    const char* read_ahead_mode = std::getenv("EDEN_NX2_READ_AHEAD");
    read_ahead = front_macros && (read_ahead_mode == nullptr || read_ahead_mode[0] != '0');
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}",
             front_macros ? ", macros on the first" : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    run_count = 0;
    run_data = false;
    run_flags = {};
    run_tags = 0;
    back_engine = nullptr;
    front_macros = false;
    read_ahead = false;
    ClearPendingWrites();
    last_unplaced_write = 0;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            SleepForBack(consumed_cache);
        }
    }
}

void GpuSplit::SleepForBack(u64 seen) {
    if (!wait_wake.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
        return;
    }
    // A yield hands the core to whatever else is ready (the guest's CPU threads) for the rest of
    // its time slice, and the back can finish and sit idle meanwhile. Sleep on `consumed` instead:
    // the back wakes this thread when it stores a new value (BackLoop).
    front_sleeping.store(true, std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) == seen) {
        consumed.wait(seen, std::memory_order_seq_cst);
    }
    front_sleeping.store(false, std::memory_order_relaxed);
}

void GpuSplit::Publish() {
    if (run_count != 0) {
        FlushRun();
    }
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::Drain() {
    if (run_count != 0) {
        FlushRun();
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position) {
        ClearPendingWrites();
        Nx2SemModel::OnDrained();
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = 8;
            ClearPendingWrites();
            Nx2SemModel::OnDrained();
            return;
        }
        if (spins < 16384) {
            Pause();
        } else {
            SleepForBack(consumed_cache);
        }
    }
}

void GpuSplit::WaitForWriters() {
    const u64 target = std::min(last_writer_position, write_position);
    if (early_submit.load(std::memory_order_relaxed) &&
        consumed.load(std::memory_order_acquire) < target) {
        submit_requested.store(true, std::memory_order_relaxed);
    }
    WaitFor(target);
}

void GpuSplit::RecordWrite(const Tegra::MemoryManager& memory, GPUVAddr address, size_t size,
                           const void* data) {
    const auto* bytes = static_cast<const u8*>(data);
    const u64 position = run_count != 0 ? InRun : write_position;
    while (size != 0) {
        // A piece per 4 KiB page: each maps contiguous device memory.
        const size_t piece = std::min<size_t>(size, 0x1000 - (address & 0xfff));
        if (const auto device = memory.GpuToCpuAddress(address)) {
            const u64 index = bytes != nullptr ? pending_data_base + pending_data.size() : NoData;
            if (bytes != nullptr) {
                pending_data.insert(pending_data.end(), bytes, bytes + piece);
            }
            // Constant buffer uploads come a word at a time: extend the last write. Its older
            // bytes count as written later, which only matters past work the front can't place.
            PendingWrite* const last =
                pending_head == pending_writes.size() ? nullptr : &pending_writes.back();
            if (last != nullptr && index != NoData && last->data != NoData &&
                last->address + last->size == *device && last->data + last->size == index &&
                last_unplaced_write <= last->position && last->size < 0x10000) {
                last->size += piece;
                last->position = position;
            } else {
                pending_writes.push_back({position, *device, piece, index});
            }
        }
        // Unmapped: the back skips the write too.
        address += piece;
        size -= piece;
        if (bytes != nullptr) {
            bytes += piece;
        }
    }
    if (pending_writes.size() >= prune_at || pending_data.size() >= prune_data_at) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        PrunePendingWrites();
        prune_at = pending_writes.size() + 4096;
        prune_data_at = pending_data.size() + 0x100000;
    }
}

void GpuSplit::PrunePendingWrites() {
    while (pending_head != pending_writes.size() &&
           pending_writes[pending_head].position <= consumed_cache) {
        ++pending_head;
    }
    if (pending_head == pending_writes.size()) {
        ClearPendingWrites();
        return;
    }
    // Drop what the back has written now and then.
    if (pending_head >= 1024 && pending_head * 2 >= pending_writes.size()) {
        pending_writes.erase(pending_writes.begin(),
                             pending_writes.begin() + static_cast<ptrdiff_t>(pending_head));
        pending_head = 0;
    }
    u64 first = pending_data_base + pending_data.size();
    for (size_t i = pending_head; i < pending_writes.size(); ++i) {
        if (pending_writes[i].data != NoData) {
            first = pending_writes[i].data;
            break;
        }
    }
    const u64 dead = first - pending_data_base;
    if (dead >= 0x10000 && dead * 2 >= pending_data.size()) {
        pending_data.erase(pending_data.begin(), pending_data.begin() + static_cast<ptrdiff_t>(dead));
        pending_data_base = first;
    }
}

void GpuSplit::NoteRelease(const Tegra::MemoryManager& memory, GPUVAddr address, u32 payload,
                           bool one_word) {
    // As the query cache writes payload reports.
    if (one_word) {
        NoteWrite(memory, address, sizeof(payload), &payload);
        return;
    }
    const u64 value = payload;
    NoteWrite(memory, address, sizeof(value), &value);
    NoteWrite(memory, address + sizeof(value), sizeof(u64), nullptr);
}

void GpuSplit::ClearPendingWrites() {
    pending_writes.clear();
    pending_head = 0;
    pending_data.clear();
    pending_data_base = 0;
    prune_at = 4096;
    prune_data_at = 0x100000;
}

bool GpuSplit::ReadAhead(Tegra::MemoryManager& memory, GPUVAddr address, u32* words, u32 count) {
    ++read_ahead_stats.reads;
    // Releases (and their payloads) reach memory as the back runs them only with the default
    // fence behavior: delayed ones land when the host GPU gets there.
    const bool delayed_releases = Settings::IsGPUFenceBehaviorDefault()
                                      ? Settings::IsGPULevelHigh()
                                      : Settings::IsGPUFenceBehaviorBalanced() ||
                                            Settings::IsGPUFenceBehaviorAccurate();
    const u64 size = u64{count} * sizeof(u32);
    const auto device = memory.GpuToCpuAddress(address);
    if (!read_ahead || delayed_releases || count == 0 || !device ||
        memory.GpuToCpuAddress(address + size - 1) != *device + size - 1) {
        ++read_ahead_stats.fallbacks;
        return false;
    }
    // GPU work writes these pages through the buffer cache: the words may be newer there.
    const auto gpu_written = [&] {
        for (DAddr page = *device & ~DAddr{0xfff}; page < *device + size; page += 0x1000) {
            if (Tegra::Nx2IsGpuWritten(page)) {
                return true;
            }
        }
        return false;
    };
    if (gpu_written()) {
        ++read_ahead_stats.fallbacks;
        return false;
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    PrunePendingWrites();
    u64 target = last_unplaced_write;
    for (size_t i = pending_head; i < pending_writes.size(); ++i) {
        const PendingWrite& write = pending_writes[i];
        if (write.data == NoData && write.address < *device + size &&
            *device < write.address + write.size) {
            target = std::max(target, write.position);
        }
    }
    if (target > consumed_cache) {
        const auto start = std::chrono::steady_clock::now();
        WaitFor(target);
        ++read_ahead_stats.waits;
        read_ahead_stats.wait_ns += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
                .count());
        PrunePendingWrites();
        if (gpu_written()) {
            ++read_ahead_stats.fallbacks;
            return false;
        }
    }
    for (u32 i = 0; i < count; ++i) {
        words[i] = memory.Read<u32>(address + u64{i} * sizeof(u32));
    }
    // What the back hasn't written yet, oldest first. The bytes it is writing right now are
    // among them.
    u8* const out = reinterpret_cast<u8*>(words);
    bool overlaid = false;
    for (size_t i = pending_head; i < pending_writes.size(); ++i) {
        const PendingWrite& write = pending_writes[i];
        const DAddr begin = std::max(write.address, *device);
        const DAddr end = std::min(write.address + write.size, *device + size);
        if (begin < end && write.data != NoData) {
            std::memcpy(out + (begin - *device),
                        pending_data.data() + (write.data - pending_data_base) + (begin - write.address),
                        end - begin);
            overlaid = true;
        }
    }
    read_ahead_stats.overlays += overlaid ? 1 : 0;
    return true;
}

void GpuSplit::WaitFor(u64 target) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return;
    }
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            return;
        }
        if (spins < 16384) {
            Pause();
        } else {
            SleepForBack(consumed_cache);
        }
    }
}

void GpuSplit::FlushRun() {
    const u32 count = run_count;
    const bool data = run_data;
    run_count = 0; // Reserve flushes while it is set
    run_data = false;
    if (count == 0) {
        return;
    }
    const size_t head = data ? 1 : 1 + RunFlagWords;
    const size_t size = head + count;
    u64* const entry = Reserve(size);
    entry[0] = static_cast<u64>(data ? Kind::Data : Kind::State) | (u64{run_tags} << 8) |
               (u64{count} << 32);
    if (!data) {
        std::memcpy(entry + 1, run_flags.data(), sizeof(run_flags));
        run_flags = {};
        run_tags = 0;
    }
    std::memcpy(entry + head, run_writes.data(), size_t{count} * sizeof(u64));
    Commit(size);
    // Writes recorded while the run was collected (constant buffer data for ReadAhead) are made
    // when the back runs this entry. Any other entry flushes the run first, so they are the last.
    for (size_t i = pending_writes.size();
         i > pending_head && pending_writes[i - 1].position == InRun; --i) {
        pending_writes[i - 1].position = write_position;
    }
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
}

size_t GpuSplit::Execute(const u64* entry) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        ++back_events;
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        back_events += (head & Quiet) == 0 ? 1 : 0;
        return 2 + payload;
    }
    case Kind::State: {
        const u32 count = static_cast<u32>(head >> 32);
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        back_engine->ApplyStateRun(*system, entry + 1, entry + 1 + RunFlagWords, count,
                                   static_cast<u8>(head >> 8));
        return 1 + RunFlagWords + count;
    }
    case Kind::Data: {
        const u32 count = static_cast<u32>(head >> 32);
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        back_engine->ApplyDataRun(*system, entry + 1, count);
        return 1 + count;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPUBack");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
    on_back_thread = true;
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    t_back_read_position = &read_position;
    Tegra::GPU& gpu = system->GPU();
    // Guest CPU read flushes run at the next entry boundary instead of after the work queued
    // behind them (GPU::HasEarlySyncRequests); checked every 1024 words and after each batch.
    const auto run_early_syncs = [&gpu] {
        if (gpu.HasEarlySyncRequests()) [[unlikely]] {
            gpu.TickWork();
        }
    };
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position; ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                if (available == read_position && !stop_token.stop_requested()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        u64 last_sync_check = read_position;
        g_back_busy.store(true, std::memory_order_relaxed);
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            const size_t size = Execute(&ring[index]);
            read_position += size != 0 ? size : RingWords - index;
            if (submit_requested.load(std::memory_order_relaxed)) [[unlikely]] {
                submit_requested.store(false, std::memory_order_relaxed);
                if (submit_hook != nullptr &&
                    read_position < wait_target.load(std::memory_order_relaxed)) {
                    submit_hook(submit_context);
                }
            }
            if (read_position - last_sync_check >= 1024) {
                last_sync_check = read_position;
                run_early_syncs();
            }
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                StoreConsumed(read_position);
                last_store = read_position;
            }
        }
        run_early_syncs();
        if (back_idle_hook != nullptr &&
            published.load(std::memory_order_acquire) == read_position) {
            back_idle_hook(back_idle_context);
        }
        g_back_busy.store(false, std::memory_order_release);
        StoreConsumed(read_position);
    }
}

} // namespace VideoCommon
