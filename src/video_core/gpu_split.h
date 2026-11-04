// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
    }
    /// A state write (not in the engine's execution_mask) whose effect the front worked out:
    /// `value` is what the register holds after it (shadow RAM applied), `flag_a`/`flag_b` the
    /// dirty flags it sets and `tags` the engine's own bits for it (EngineInterface::ApplyStateRun),
    /// none when `changed` is false (it keeps the register's value). Consecutive ones go to the
    /// back as one entry, which stores the registers and sets what the whole run sets in one pass.
    void StateWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 value, u8 flag_a,
                    u8 flag_b, u8 tags, bool changed) {
        if (engine != front_engine || run_data || run_count == MaxRunWrites) [[unlikely]] {
            FlushRun();
            SelectEngine(engine);
        }
        run_writes[run_count++] = u64{method} | (u64{value} << 32);
        if (changed) {
            run_flags[flag_a >> 6] |= u64{1} << (flag_a & 63);
            run_flags[flag_b >> 6] |= u64{1} << (flag_b & 63);
            run_tags |= tags;
        }
    }
    /// A word of a data method the engine takes many of in a row (Maxwell3D's constant buffer
    /// data, one write to guest memory each): consecutive ones go to the back as one entry
    /// (EngineInterface::ApplyDataRun), `value` as in StateWrite.
    void DataWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 value) {
        if (engine != front_engine || !run_data || run_count == MaxRunWrites) [[unlikely]] {
            FlushRun();
            SelectEngine(engine);
            run_data = true;
        }
        run_writes[run_count++] = u64{method} | (u64{value} << 32);
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    template <typename Func>
    void Defer(const Func& func) {
        DeferImpl(func, 0);
    }
    /// Defer for work of the 3D engine itself that changes state only through its usual paths
    /// (method sink, registers): not a back event (BackEvents).
    template <typename Func>
    void DeferEngineWork(const Func& func) {
        DeferImpl(func, Quiet);
    }
    template <typename Func>
    void DeferImpl(const Func& func, u64 flags) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
    }
    void DeferFunction(std::function<void()>&& func);

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();

    /// Past their spin, the front's waits for the back (Drain, WaitFor, WaitForSpace) sleep until
    /// the back stores `consumed` instead of yielding. EDEN_NX2_WAIT_WAKE=0 turns that off
    /// (comparisons); it can flip at run time.
    static inline std::atomic<bool> wait_wake{[] {
        const char* value = std::getenv("EDEN_NX2_WAIT_WAKE");
        return value == nullptr || value[0] != '0';
    }()};
    /// Waits until the back executed everything sent so far.
    void Drain();

    /// Back thread: counts the entries other than 3D engine methods it ran (engine switches and
    /// deferred calls), the work between two draws that their setup can't see in 3D state.
    [[nodiscard]] u64 BackEvents() const noexcept {
        return back_events;
    }
    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() noexcept {
        return on_back_thread;
    }

    /// Compute or copy work was just sent: what can write GPU-written command lists.
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();

    /// On the back, while it runs an entry: whether more work was sent after it (false on any
    /// other thread, or without the split).
    bool BackHasMoreWork() const {
        return t_back_read_position != nullptr &&
               published.load(std::memory_order_relaxed) - *t_back_read_position > 64;
    }

    /// Runs on the back after it ran everything sent so far, before the front can see it done
    /// (the back still owns the rasterizer then): work the back put off until it idles.
    void SetBackIdleHook(void (*hook)(void*), void* context) {
        back_idle_context = context;
        back_idle_hook = hook;
    }
    /// Runs on the back when the front starts waiting for the writers of a GPU-written command
    /// list (WaitForWriters) and the back is still short of them: submits the work recorded so
    /// far, so the host GPU runs it while the back records the rest up to the writers, and the
    /// writers' copy the front then waits for is done sooner.
    void SetSubmitHook(void (*hook)(void*), void* context) {
        submit_context = context;
        submit_hook = hook;
    }
    /// EDEN_NX2_EARLY_SUBMIT=0 turns the submit above off (comparisons); it can flip at run time.
    static inline std::atomic<bool> early_submit{[] {
        const char* value = std::getenv("EDEN_NX2_EARLY_SUBMIT");
        return value == nullptr || value[0] != '0';
    }()};
    // Reads ahead of the back (the 3D engine's MME DMA reads): the front keeps a record of the
    // guest memory writes it queued for the back that it can predict (constant buffer and inline
    // uploads, payload releases) and of the last queued work that writes where it can't tell
    // (compute work, copies). A read then waits only for the latter, and only when there is any,
    // instead of for everything queued. Front thread only.

    [[nodiscard]] bool ReadsAhead() const {
        return read_ahead;
    }
    /// The entry just sent writes `size` bytes at `address` when the back runs it; `data`: the
    /// bytes, nullptr when the front can't know them (a timestamp, a counter).
    void NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, size_t size,
                   const void* data) {
        if (read_ahead) {
            RecordWrite(memory, address, size, data);
        }
    }
    /// A queued payload release: the payload, or (not `one_word`) the payload as a 64-bit value
    /// and a timestamp.
    void NoteRelease(const Tegra::MemoryManager& memory, GPUVAddr address, u32 payload,
                     bool one_word);
    /// The entry just sent may write guest memory the front can't place.
    void NoteUnplacedWrite() {
        last_unplaced_write = write_position;
    }
    /// Reads `count` words at `address` as the back will see them once it has run everything
    /// sent so far. False when the front can't tell (pages the GPU writes, whose newest words may
    /// be in the back's caches; releases applied late): the caller drains and reads itself.
    bool ReadAhead(Tegra::MemoryManager& memory, GPUVAddr address, u32* words, u32 count);

    struct ReadAheadStats {
        u64 reads;
        u64 fallbacks; ///< ReadAhead returned false
        u64 waits;     ///< Waited for queued writes it couldn't predict
        u64 wait_ns;
        u64 overlays;  ///< Reads that took words from queued writes
    };
    ReadAheadStats read_ahead_stats{};

private:
    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        State,  ///< Bits 8-15 tags, 32-63 count; words 1-4 dirty flags, then a method | value << 32 each
        Data,   ///< Bits 32-63 count; then a method | value << 32 each
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)
    static constexpr u64 Quiet = 1 << 7;  ///< Call: 3D engine work, not a back event

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    static constexpr u32 MaxRunWrites = 512;
    static constexpr size_t RunFlagWords = 4; ///< 256 dirty flags (Maxwell3D::DirtyState)

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        // Every other entry comes after the state writes collected so far.
        if (run_count != 0) {
            FlushRun();
        }
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    /// Sends the writes collected so far (StateWrite, DataWrite) as one entry.
    void FlushRun();
    void Commit(size_t words) {
        write_position += words;
        if (write_position - published_local >= publish_every) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);

    /// A guest memory write queued for the back (NoteWrite).
    struct PendingWrite {
        u64 position; ///< The back has made it once it consumed this far
        DAddr address;
        u64 size;
        u64 data; ///< Index of the bytes in pending_data, or NoData
    };
    static constexpr u64 NoData = ~u64{0};
    /// Position of writes recorded while a run (StateWrite, DataWrite) is still being collected:
    /// the back makes them once it runs that entry, whose end FlushRun fills in.
    static constexpr u64 InRun = ~u64{0};
    void RecordWrite(const Tegra::MemoryManager& memory, GPUVAddr address, size_t size,
                     const void* data);
    void PrunePendingWrites();
    void ClearPendingWrites();
    /// Waits until the back consumed `target`.
    void WaitFor(u64 target);
    /// Front, past the spin of a wait for `consumed` (last seen as `seen`).
    void SleepForBack(u64 seen);
    /// Back: hands `consumed` to the front and wakes it if it sleeps on it.
    void StoreConsumed(u64 position) {
        consumed.store(position, std::memory_order_seq_cst);
        if (front_sleeping.load(std::memory_order_seq_cst)) {
            consumed.notify_all();
        }
    }

    void BackLoop(std::stop_token stop_token);
    size_t Execute(const u64* entry);

    Core::System* system{};
    bool active{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};
    /// Queued guest memory writes (ReadAhead), oldest first from pending_head, and their bytes.
    bool read_ahead{};
    std::vector<PendingWrite> pending_writes;
    size_t pending_head{};
    std::vector<u8> pending_data;
    u64 pending_data_base{}; ///< Index of pending_data[0]
    size_t prune_at{};
    size_t prune_data_at{};
    u64 last_unplaced_write{};
    /// Writes for front_engine not sent yet (StateWrite, or DataWrite when run_data), and the
    /// dirty flags the state writes set.
    u32 run_count{};
    bool run_data{};
    u8 run_tags{};
    std::array<u64, RunFlagWords> run_flags{};
    std::array<u64, MaxRunWrites> run_writes{};

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    void (*back_idle_hook)(void*){};
    void* back_idle_context{};
    void (*submit_hook)(void*){};
    void* submit_context{};
    /// The back thread's read position (BackHasMoreWork); null on other threads.
    static inline thread_local const u64* t_back_read_position{};
    u64 back_events{};
    static inline thread_local bool on_back_thread{};

    static GpuSplit instance;

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// A position the front waits for (WaitForWriters): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    /// WaitForWriters asks the back to submit what it recorded (submit_hook).
    alignas(128) std::atomic<bool> submit_requested{};
    alignas(128) std::atomic<bool> back_waiting{};
    /// The front sleeps on `consumed` (SleepForBack): the back wakes it when it stores one.
    alignas(128) std::atomic<bool> front_sleeping{};
    std::atomic<u32> wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
