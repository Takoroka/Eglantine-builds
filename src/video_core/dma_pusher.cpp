// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <span>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/nx2_profile.h"
#include "video_core/dma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;

namespace {

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

// WaitForEntryWriters: waits that found a page's entries unwritten before it is a known CPU list
// page.
constexpr u32 CpuListPageAfter = 8;

// Every 10 s: what the front did with command lists and how long it waited for the back (nx2.h,
// Nx2ListStats), per second.
void ReportListStats() {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point window_start = Clock::now();
    const Clock::time_point now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    if (seconds < 10.0) {
        return;
    }
    window_start = now;
    auto& stats = g_nx2_list_stats;
    const auto rate = [seconds](std::atomic<u64>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) / seconds;
    };
    const auto ms = [&](std::atomic<u64>& counter) { return rate(counter) / 1e6; };
    const double entries = rate(stats.entries);
    const double copied = rate(stats.copied);
    const double synced = rate(stats.synced);
    const double exact = rate(stats.exact);
    const double drains = rate(stats.drains);
    const double drain_ms = ms(stats.drain_ns);
    const double flush_ms = ms(stats.flush_ns);
    const double flushes = rate(stats.flushes);
    const double batch_flushes = rate(stats.batch_flushes);
    const double batch_pages = rate(stats.batch_pages);
    const double draw_page_reads = rate(stats.draw_page_reads);
    const double would_exact = rate(stats.would_exact);
    const double writer_ms = ms(stats.writer_ns);
    const double copy_ms = ms(stats.copy_ns);
    const double matched = rate(stats.matched);
    const double mismatched = rate(stats.mismatched);
    const double no_copy = rate(stats.no_copy);
    const double draw_writes = rate(stats.draw_writes);
    const double draw_kib = rate(stats.draw_bytes) / 1024.0;
    const double entry_waits = rate(stats.entry_waits);
    const double entry_wait_ms = ms(stats.entry_wait_ns);
    const double entry_needed = rate(stats.entry_needed);
    const double entry_skips = rate(stats.entry_skips);
    const double entry_known_needed = rate(stats.entry_known_needed);
    const double entry_known_turned = rate(stats.entry_known_turned);
    const double split_drains = rate(stats.split_drains);
    const double split_drain_ms = ms(stats.split_drain_ns);
    const double split_waits = rate(stats.split_waits);
    const double split_wait_ms = ms(stats.split_wait_ns);
    const double acquire_drains = rate(stats.acquire_drains);
    const double acquire_drain_ms = ms(stats.acquire_drain_ns);
    const double resume_drains = rate(stats.resume_drains);
    const double resume_drain_ms = ms(stats.resume_drain_ns);
    const double execute_drains = rate(stats.execute_drains);
    const double execute_drain_ms = ms(stats.execute_drain_ns);
    const double execute_on_back = rate(stats.execute_on_back);
    const double macro_read_drains = rate(stats.macro_read_drains);
    const double macro_read_drain_ms = ms(stats.macro_read_drain_ns);
    const double sync_drains = rate(stats.sync_drains);
    const double sync_drain_ms = ms(stats.sync_drain_ns);
    const double run_drains = rate(stats.run_drains);
    const double run_drain_ms = ms(stats.run_drain_ns);
    const double acquire_gequal = rate(stats.acquire_gequal);
    const double acquire_may_drain = rate(stats.acquire_may_drain);
    const double acquire_met = rate(stats.acquire_met);
    const double indirect_draws = rate(stats.indirect_draws);
    const double indirect_checked = rate(stats.indirect_checked);
    const double indirect_odd = rate(stats.indirect_odd);
    const double indirect_barriers = rate(stats.indirect_barriers);
    const double extended_draws = rate(stats.extended_draws);
    const double extended_rechecks = rate(stats.extended_rechecks);
    const double extended_compared = rate(stats.extended_compared);
    const double extended_mismatched = rate(stats.extended_mismatched);
    const double extended_fallbacks = rate(stats.extended_fallbacks);
    const double barriers_skipped = rate(stats.barriers_skipped);
    if (entries == 0 && draw_writes == 0 && entry_waits == 0 && entry_skips == 0 &&
        split_drains == 0 && split_waits == 0 && indirect_draws == 0 && extended_draws == 0 &&
        extended_rechecks == 0) {
        return;
    }
    using Common::Nx2Ab::On;
    LOG_INFO(HW_GPU,
             "Command lists per second (bit 48 {}, 49 {}, 50 {}, 51 {}, 52 {}, 55 {}): "
             "indirect draw macros {:.0f} drawn from the GPU ({:.0f} barriers), {:.0f} checked "
             "({:.0f} odd); extended {:.0f} drawn on the GPU, {:.0f} re-drawn after the front's "
             "read ({:.0f} compared, {:.0f} mismatched), {:.0f} fallbacks, {:.0f} barriers "
             "skipped (writes elsewhere) | acquire drains {:.0f} GEQUAL, {:.0f} with nothing "
             "else to parse, {:.0f} met after it | {:.0f} entries on "
             "GPU-written pages | read from copies {:.0f}, synced {:.0f}, past the draws' ranges "
             "{:.0f} | list drains {:.0f}: {:.1f} ms for the back, {:.1f} ms flushing ({:.0f} "
             "flushes, {:.0f} batched taking {:.0f} other pages); {:.0f} on pages draws "
             "write ({:.0f} outside their ranges) | waits for writers {:.1f} ms, for copies "
             "{:.1f} ms | drained words vs guest memory/copy: {:.0f} same, {:.0f} different, "
             "{:.0f} neither current | draw writes to list pages {:.0f} ({:.0f} KiB) | entries on "
             "unwritten pages: {:.0f} waits ({:.1f} ms), {:.0f} found GPU-written, {:.0f} skipped "
             "as CPU list pages, {:.0f} known ones needed the wait, {:.0f} turned GPU-written | "
             "all front waits for the back: {:.0f} drains ({:.1f} ms), {:.0f} position waits "
             "({:.1f} ms) | drain calls: acquires {:.0f} ({:.1f} ms), resumed acquires {:.0f} "
             "({:.1f} ms), SEM_EXECUTE {:.0f} ({:.1f} ms) + {:.0f} checked on the back, macro "
             "reads {:.0f} ({:.1f} ms), syncs {:.0f} ({:.1f} ms), runs {:.0f} ({:.1f} ms)",
             On(Common::Nx2Ab::ListDrawRanges) ? "on" : "off",
             On(Common::Nx2Ab::ListWriterWaitSkip) ? "on" : "off",
             On(Common::Nx2Ab::SemAcquireOnBack) ? "on" : "off",
             On(Common::Nx2Ab::ListFlushBatch) ? "on" : "off",
             Nx2IndirectDrawMacro() ? "on" : "off",
             On(Common::Nx2Ab::IndirectDrawMacroExtended) ? "on" : "off", indirect_draws,
             indirect_barriers, indirect_checked, indirect_odd, extended_draws, extended_rechecks,
             extended_compared, extended_mismatched, extended_fallbacks, barriers_skipped,
             acquire_gequal, acquire_may_drain,
             acquire_met, entries, copied, synced, exact,
             drains, drain_ms, flush_ms, flushes, batch_flushes, batch_pages, draw_page_reads,
             would_exact, writer_ms, copy_ms,
             matched, mismatched, no_copy, draw_writes, draw_kib, entry_waits, entry_wait_ms,
             entry_needed, entry_skips, entry_known_needed, entry_known_turned, split_drains,
             split_drain_ms, split_waits, split_wait_ms, acquire_drains, acquire_drain_ms,
             resume_drains, resume_drain_ms, execute_drains, execute_drain_ms, execute_on_back,
             macro_read_drains, macro_read_drain_ms, sync_drains, sync_drain_ms, run_drains,
             run_drain_ms);
}

} // Anonymous namespace

namespace {
// Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads. The
// register writes that set them up only store a value: marking those too made every list fetch
// after one wait for all the work queued before it (draws included).
bool WritesMemory(Engines::EngineInterface* engine, Engines::EngineTypes type, u32 method,
                  u32 count) {
    if (type == Engines::EngineTypes::KeplerCompute) {
        const auto* const compute = static_cast<const Engines::KeplerCompute*>(engine);
        for (u32 i = 0; i < count; ++i) {
            if (compute->MayWriteMemory(method + i)) {
                return true;
            }
        }
        return false;
    }
    if (type == Engines::EngineTypes::MaxwellDMA) {
        for (u32 i = 0; i < count; ++i) {
            const u32 current = method + i;
            if (current >= MacroRegistersStart ||
                (current < engine->execution_mask.size() && engine->execution_mask[current])) {
                return true;
            }
        }
    }
    return false;
}
} // Anonymous namespace

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    // The sub-index is kept: a channel suspended on a semaphore resumes inside its command list.
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer([gpu = &system.GPU()] {
            gpu->FlushCommands();
            gpu->OnCommandListEnd();
        });
        split.Publish();
        return;
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }
    const u64 suspensions_before = suspensions.load(std::memory_order_relaxed);
    if (!puller.ResumeAcquire(*this)) {
        return false;
    }
    // Lists queued while the channel waited were read from memory, not as submitted: the game
    // writes the commands a list runs before it releases the acquire, so the words at submit time
    // could be older. Reading them as late as the parse is too late in Hyrule Warriors: Age of
    // Imprisonment fights: a command ring the game had already refilled for a later frame, read
    // as this frame's commands (methods on unbound subchannels, garbage QMDs, then a channel
    // waiting forever on a release it never got). The acquire is released now, so the commands
    // are written and the memory not yet reused: copy them here. EDEN_NX2_RESNAPSHOT=0: off.
    if (ResnapshotEnabled() && suspensions.load(std::memory_order_relaxed) != suspensions_before) {
        const u64 now = suspensions.load(std::memory_order_acquire);
        for (CommandList& list : dma_pushbuffer) {
            if (!list.submitted_offsets.empty()) {
                RefreshSnapshot(list, now);
            }
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset +=
            ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        dma_pushbuffer.pop_front();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced; });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    split.Defer([engine_state, memory = &memory_manager, address = GPUVAddr{dma_state.dma_get},
                                 bytes = words * sizeof(u32)] {
                        engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                    });
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        if (command_resume_offset == 0) {
            // A method's data continues across GP entries, whatever their level: title 4 ends a
            // main entry after a macro's first word and supplies the remaining parameters in a
            // 5-word subroutine entry. EDEN_NX2_PARK_SPLIT_METHODS restores the old guess (an
            // entry elsewhere is a call that starts clean; resume where the split one ended).
            static const bool park_split_methods = std::getenv("EDEN_NX2_PARK_SPLIT_METHODS") != nullptr;
            if (g_nx2_ampere) {
                if (static u32 split_logs{}; dma_state.method_count != 0 && split_logs < 64) {
                    ++split_logs;
                    LOG_DEBUG(HW_GPU, "Split method {:#x} ({} left) on ch {}: next entry {:#x} size {} raw {:#x} (level bit {}, priv bit {}), expected {:#x}, parked {}",
                             dma_state.method, dma_state.method_count, channel_state.bind_id,
                             u64{header.addr}, u64{header.size}, header.raw, u64{header.is_push_buffer},
                             u64{header.allow_flush}, split_method_end, parked_methods.size());
                    std::array<u32, 8> next_words{};
                    std::array<u32, 4> prev_words{};
                    memory_manager.ReadBlockUnsafe(header.addr, next_words.data(),
                                             std::min<size_t>(header.size, 8) * sizeof(u32));
                    memory_manager.ReadBlockUnsafe(split_method_end - sizeof(prev_words), prev_words.data(),
                                             sizeof(prev_words));
                    LOG_DEBUG(HW_GPU, "  previous entry ends {:08x}; next entry starts {:08x}",
                              fmt::join(prev_words, " "), fmt::join(next_words, " "));
                }
                if (park_split_methods) {
                    ResolveSplitMethod(header.addr);
                }
            }
            if (dma_pushbuffer_subindex < command_list.submit_hashes.size()) {
                CheckSubmitHash(header, command_list.submit_hashes[dma_pushbuffer_subindex]);
            }
        }
        // A resumed entry is read again: the words after the acquire may be new by now.
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // The split's back marks the pages GPU work writes as it runs that work: a page first
        // written by a dispatch or copy still queued for it reads as unwritten, and its words as
        // stale (Fast Fusion's culling fills in instanced draws' arguments; stale ones asked for
        // billions of instances and lost the device loading a race).
        const size_t entry_bytes = size_t{header.size} * sizeof(u32);
        bool gpu_written = IsGpuWrittenCommands(dma_state.dma_get, entry_bytes);
        if (gpu_written) {
            Nx2CountList(g_nx2_list_stats.entries);
            if (!cpu_list_pages.empty()) {
                ForgetCpuListPages(dma_state.dma_get, entry_bytes);
            }
        } else if (VideoCommon::GpuSplit::Get().Active() &&
                   VideoCommon::GpuSplit::Get().WritersQueued()) {
            gpu_written = WaitForEntryWriters(dma_state.dma_get, entry_bytes);
            if (gpu_written) {
                Nx2CountList(g_nx2_list_stats.entries);
            }
        }
        ReportListStats();
        // The words as submitted, unless this is a resumed entry or the GPU has newer ones the
        // safe read below would flush (with the split, the front can't tell: it reads memory
        // unflushed anyway). Entries on pages the GPU writes (culling) always take the flushing
        // read: their words at submit time are an older frame's.
        const bool submitted = command_resume_offset == 0 &&
                               dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                               (command_list.snapshot_suspensions & 1) == 0 &&
                               command_list.snapshot_suspensions ==
                                   suspensions.load(std::memory_order_acquire) &&
                               !std::ranges::binary_search(
                                   command_list.deferred_entries,
                                   static_cast<u32>(dma_pushbuffer_subindex)) &&
                               !gpu_written &&
                               !(use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                                          header.size * sizeof(u32)));
        if (!submitted && dma_pushbuffer_subindex < command_list.submitted_offsets.size()) {
            // An entry parsed from memory instead of its submit-time copy is the way a command
            // ring the game already refilled gets read as this frame's commands (garbage methods,
            // unbound subchannels, render targets of absurd size). Count how often the words
            // differ, by the reason the copy was skipped. Bounded: the first entries only.
            enum Reason : u32 { Resumed, SuspendedCopy, SuspensionsChanged, GpuWritten, SafeDirty, Count };
            static std::array<std::atomic<u64>, Count> checked{};
            static std::array<std::atomic<u64>, Count> differing{};
            static std::atomic<u64> total_checked{};
            static constexpr std::array<const char*, Count> names{
                "resumed entry", "copied during a suspension", "suspension count changed",
                "GPU-written page", "safe read of dirty memory"};
            if (total_checked.load(std::memory_order_relaxed) < 50000) {
                const Reason reason = command_resume_offset != 0                      ? Resumed
                                      : (command_list.snapshot_suspensions & 1) != 0  ? SuspendedCopy
                                      : command_list.snapshot_suspensions !=
                                              suspensions.load(std::memory_order_acquire)
                                          ? SuspensionsChanged
                                      : gpu_written ? GpuWritten
                                                    : SafeDirty;
                std::vector<u32> live(header.size);
                memory_manager.ReadBlockUnsafe(dma_state.dma_get, live.data(),
                                               live.size() * sizeof(u32));
                const CommandHeader* const copy = command_list.submitted_words.data() +
                                                  command_list.submitted_offsets[dma_pushbuffer_subindex];
                u32 first = ~0U, words = 0;
                for (u32 i = 0; i < header.size; ++i) {
                    if (copy[i].argument != live[i]) {
                        first = std::min(first, i);
                        ++words;
                    }
                }
                ++checked[reason];
                const u64 n = total_checked.fetch_add(1) + 1;
                if (words != 0) {
                    const u64 seen = differing[reason].fetch_add(1);
                    if (seen < 4) {
                        LOG_WARNING(HW_GPU,
                                    "Entry {:#x}+{:#x} parsed from memory ({}) differs from its "
                                    "submit-time copy in {} words, first at word {}",
                                    u64{header.addr}, u64{header.size}, names[reason], words, first);
                    }
                }
                if ((n % 10000) == 0) {
                    for (u32 r = 0; r < Count; ++r) {
                        LOG_INFO(HW_GPU, "Entries parsed from memory ({}): {} checked, {} differ from the submit-time copy",
                                 names[r], checked[r].load(), differing[r].load());
                    }
                }
            }
        }
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            RecordParsedEntry(header, commands);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (gpu_written && IndirectDrawMacro(header)) {
            // An indirect draw's arguments: the back draws from them on the GPU.
        } else if (ReadGpuWrittenCommands(dma_state.dma_get, static_cast<u32>(header.size))) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
        if (puller.HasPendingAcquire()) {
            return false;
        }
        command_resume_offset = 0;
        if (dma_state.method_count != 0) {
            split_method_end = header.addr + header.size * sizeof(u32);
        }
    }

    if (++dma_pushbuffer_subindex >= command_list_size) {
        // A split method resumes within the submission that split it; ring addresses repeat
        // every frame, so a stale one could otherwise swallow the headers of a later entry.
        for (const auto& parked : parked_methods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Submission ended with parked method {:#x} (sub {}, {} words left, "
                            "resume {:#x})",
                            parked.method, parked.subchannel, parked.method_count,
                            parked.resume_address);
            }
        }
        parked_methods.clear();
        dma_pushbuffer.pop_front();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        // SignalFence invalidates the GPU caches, which can record copies into the scheduler: the
        // back's state, which a drained back still submits from in its idle hook.
        const Nx2CountedTime counted{g_nx2_list_stats.sync_drains,
                                     g_nx2_list_stats.sync_drain_ns};
        VideoCommon::GpuSplit::Get().Run([this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        });
    }

    return true;
}

// EDEN_NX2_SEM_WATCH=<gpu address>: log every method whose argument is the low word of an
// address in the 16 bytes there (finds the engine and method that release a semaphore).
static const u32 g_semaphore_watch = [] {
    const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
    return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
}();

// EDEN_NX2_PB_CHECK also keeps the last parsed entries, for DumpCurrentEntry.
static const bool g_record_parsed_entries = std::getenv("EDEN_NX2_PB_CHECK") != nullptr;

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (g_nx2_ampere && dma_state.method < non_puller_methods) {
                max_write = 1; // a semaphore acquire may suspend the channel after any word
            }
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            // Fast-path batching for consecutive non-execution methods in increasing mode
            if (!dma_state.non_incrementing && !dma_increment_once &&
                dma_state.method >= non_puller_methods &&
                !VideoCommon::GpuSplit::Get().Active() &&
                g_semaphore_watch == 0) {
                auto subchannel = subchannels[dma_state.subchannel];
                if (subchannel != nullptr) {
                    const size_t available = std::min<size_t>(dma_state.method_count, commands.size() - index);
                    u32 batch_count = 0;
                    while (batch_count < available &&
                           (dma_state.method + batch_count) < subchannel->execution_mask.size() &&
                           !subchannel->execution_mask[dma_state.method + batch_count]) {
                        ++batch_count;
                    }
                    if (batch_count > 1) {
                        subchannel->method_sink.reserve(subchannel->method_sink.size() + batch_count);
                        for (u32 i = 0; i < batch_count; ++i) {
                            subchannel->method_sink.emplace_back(dma_state.method + i, commands[index + i].argument);
                        }
                        dma_state.method += batch_count;
                        dma_state.method_count -= batch_count;
                        index += batch_count;
                        continue;
                    }
                }
            }

            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingAcquire()) {
            return index;
        }
    }
    return commands.size();
}

// NVN on Switch 2 ends GPFIFO entries in the middle of a method's data and calls another command
// buffer (e.g. a compute dispatch) before the entry that carries the rest, which starts right
// where the first one ended. An entry starting anywhere else is such a call: run it with a clean
// method state and resume the unfinished method when the stream continues at that address.
void DmaPusher::ResolveSplitMethod(GPUVAddr entry_address) {
    const auto continued = std::ranges::find(parked_methods, entry_address,
                                             &ParkedMethod::resume_address);
    std::optional<ParkedMethod> resume;
    if (continued != parked_methods.end()) {
        resume = *continued;
        parked_methods.erase(continued);
    }
    if (dma_state.method_count != 0 && entry_address != split_method_end) {
        if (parked_methods.size() == MaxParkedMethods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 8) {
                const auto& oldest = parked_methods.front();
                LOG_WARNING(HW_GPU,
                            "Dropping parked method {:#x} (sub {}, {} words left, resume {:#x})",
                            oldest.method, oldest.subchannel, oldest.method_count,
                            oldest.resume_address);
            }
            parked_methods.erase(parked_methods.begin());
        }
        parked_methods.push_back(ParkedMethod{
            .method = dma_state.method,
            .subchannel = dma_state.subchannel,
            .method_count = dma_state.method_count,
            .non_incrementing = dma_state.non_incrementing,
            .increment_once = dma_increment_once,
            .resume_address = split_method_end,
        });
        dma_state.method_count = 0;
        dma_increment_once = false;
    }
    if (resume && dma_state.method_count == 0) {
        dma_state.method = resume->method;
        dma_state.subchannel = resume->subchannel;
        dma_state.method_count = resume->method_count;
        dma_state.non_incrementing = resume->non_incrementing;
        dma_increment_once = resume->increment_once;
    }
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}


void DmaPusher::WatchArgument(u32 method, u32 argument) const {
    const u32 watch = g_semaphore_watch;
    if ((argument & ~0xfU) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} subchannel {} method {:#x} argument {:#x}",
                 channel_state.bind_id, dma_state.subchannel, method, argument);
    }
}

// A method sent straight to the split's back that may write guest memory where the front can't
// tell (copies, inline-to-memory uploads, 2D blits; compute and 3D work without their front
// state): reads ahead of the back wait for it (GpuSplit::NoteUnplacedWrite).
static bool WritesUnplaced(Engines::EngineTypes type, u32 method) {
    switch (type) {
    case Engines::EngineTypes::MaxwellDMA:
        return method == 0xc0; // LAUNCH_DMA
    case Engines::EngineTypes::KeplerMemory:
        return method == 0x6d; // LOAD_INLINE_DATA
    case Engines::EngineTypes::Fermi2D:
        return method == FERMI2D_REG_INDEX(pixels_from_memory.src_y0) + 1;
    case Engines::EngineTypes::KeplerCompute:
    case Engines::EngineTypes::Maxwell3D:
        return true;
    default:
        return false;
    }
}

bool DmaPusher::WantsSegment() const {
    // Only macro parameters and compute uploads read the engine's current_dma_segment.
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        WatchArgument(dma_state.method, argument);
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const bool writer = split.Active() &&
                            WritesMemory(subchannel, subchannel_type[dma_state.subchannel],
                                         dma_state.method, 1);
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        return;
                    }
                }
            }
            split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                         WantsSegment() ? &segment : nullptr);
            if (WritesUnplaced(subchannel_type[dma_state.subchannel], dma_state.method)) {
                split.NoteUnplacedWrite();
            }
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        for (u32 i = 0; i < num_methods; ++i) {
            WatchArgument(dma_state.method + (dma_state.non_incrementing ? 0 : i), base_start[i]);
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const bool writer = split.Active() &&
                            WritesMemory(subchannel, subchannel_type[dma_state.subchannel],
                                         dma_state.method, dma_state.non_incrementing ? 1 : num_methods);
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter();
            }
        };
        if (split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        return;
                    }
                }
            }
            split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                              dma_state.method_count, WantsSegment() ? &segment : nullptr);
            if (WritesUnplaced(subchannel_type[dma_state.subchannel], dma_state.method)) {
                split.NoteUnplacedWrite();
            }
            return;
        }
        VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::RefreshSnapshot(CommandList& list, u64 now) {
    list.snapshot_suspensions = now;
    for (size_t i = 0; i < list.command_lists.size() && i < list.submitted_offsets.size(); ++i) {
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
    }
    list.deferred_entries.clear();
}

void DmaPusher::RefreshDeferredEntries(CommandList& list) {
    for (const u32 i : list.deferred_entries) {
        if (i >= list.command_lists.size() || i >= list.submitted_offsets.size()) {
            continue;
        }
        const CommandListHeader& entry = list.command_lists[i];
        const u32 offset = list.submitted_offsets[i];
        if (offset + entry.size > list.submitted_words.size()) {
            continue;
        }
        memory_manager.ReadBlockUnsafe(entry.addr, list.submitted_words.data() + offset,
                                       entry.size * sizeof(u32));
    }
    list.deferred_entries.clear();
}

bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            return true;
        }
    }
    return false;
}

bool DmaPusher::WaitForEntryWriters(GPUVAddr address, size_t size) {
    // The wait (Step) is for a dispatch or copy still queued that writes these pages first. Pages
    // the game's CPU writes its command lists on never see one: once entries there were found
    // unwritten after the wait CpuListPageAfter times, A/B bit ListWriterWaitSkip reads them
    // without waiting (~50 waits a second in Donkey Kong Bananza, 3 ms each at the median).
    auto& stats = g_nx2_list_stats;
    const bool known = KnownCpuListPages(address, size);
    if (known && Common::Nx2Ab::On(Common::Nx2Ab::ListWriterWaitSkip)) {
        Nx2CountList(stats.entry_skips);
        return false;
    }
    const auto wait_start = std::chrono::steady_clock::now();
    if (!VideoCommon::GpuSplit::Get().WaitForWriters(false)) {
        return false;
    }
    Nx2CountList(stats.entry_waits);
    Nx2CountList(stats.entry_wait_ns, NanosecondsSince(wait_start));
    if (IsGpuWrittenCommands(address, size)) {
        Nx2CountList(stats.entry_needed);
        if (known) {
            Nx2CountList(stats.entry_known_needed);
        }
        if (!cpu_list_pages.empty()) {
            ForgetCpuListPages(address, size);
        }
        return true;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page)) {
            u32& waits = cpu_list_pages[*device >> Nx2WrittenPageBits];
            waits = (std::min)(waits + 1, CpuListPageAfter);
        }
    }
    return false;
}

bool DmaPusher::IndirectDrawMacro(const CommandListHeader& header) {
    // The entry is exactly the rest of a macro's parameters (sent non-incrementing, as macro
    // calls are), and the 3D engine runs macros on the split's front.
    if (header.size != 5 || dma_state.method_count != 5 || !dma_state.non_incrementing ||
        command_resume_offset != 0 || dma_state.method < MacroRegistersStart ||
        subchannel_type[dma_state.subchannel] != Engines::EngineTypes::Maxwell3D) {
        // The first few short GPU-written entries this turns down, and why.
        static std::atomic<u32> logged{};
        if (header.size <= 8 && logged.fetch_add(1) < 24) {
            LOG_INFO(HW_GPU,
                     "Indirect draw macro: entry {:#x} ({} words) not taken: method {:#x}, {} "
                     "words left, non-incrementing {}, resume offset {}, engine {}",
                     u64{header.addr}, u64{header.size}, dma_state.method,
                     dma_state.method_count, dma_state.non_incrementing, command_resume_offset,
                     static_cast<u32>(subchannel_type[dma_state.subchannel]));
        }
        return false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannels[dma_state.subchannel]);
    if (!split.Active() || !split.FrontMacros() || maxwell3d == nullptr ||
        !maxwell3d->GetNativeMacroEngine().native_turing) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1) < 4) {
            LOG_INFO(HW_GPU, "Indirect draw macro: not taken, macros don't run on the front");
        }
        return false;
    }
    if (!maxwell3d->FrontIndirectDrawMacro(system, split, dma_state.method, dma_state.dma_get)) {
        return false;
    }
    dma_state.method_count = 0;
    dma_state.is_last_call = true;
    return true;
}

bool DmaPusher::KnownCpuListPages(GPUVAddr address, size_t size) {
    if (size == 0 || cpu_list_pages.empty()) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        const auto device = memory_manager.GpuToCpuAddress(page);
        if (!device) {
            return false;
        }
        const auto it = cpu_list_pages.find(*device >> Nx2WrittenPageBits);
        if (it == cpu_list_pages.end() || it->second < CpuListPageAfter) {
            return false;
        }
    }
    return true;
}

void DmaPusher::ForgetCpuListPages(GPUVAddr address, size_t size) {
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        const auto device = memory_manager.GpuToCpuAddress(page);
        if (!device) {
            continue;
        }
        const auto it = cpu_list_pages.find(*device >> Nx2WrittenPageBits);
        if (it == cpu_list_pages.end()) {
            continue;
        }
        if (it->second >= CpuListPageAfter) {
            // GPU work wrote a page its entries were read from without waiting for it: a list
            // there may have been read before that work ran.
            Nx2CountList(g_nx2_list_stats.entry_known_turned);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Command list page {:#x} the CPU wrote the lists of is GPU-written now",
                            *device & ~u64{0xfff});
            }
        }
        cpu_list_pages.erase(it);
    }
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, u32 count) {
    // The pages are marked as command list pages: from the next write on, the buffer cache copies
    // them to host memory right after the dispatch or copy that writes them and submits. The
    // front then waits only for the back to run that work and for the GPU to finish the copy, not
    // for everything queued and a cache flush (FlushGpuWrittenCommands: ~400 GPU round trips a
    // second in a Mario Kart World race, during which the back had nothing to do). Pages
    // FlushGpuWrittenCommands flushed whole and no GPU work wrote since are read from guest
    // memory.
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || count == 0 ||
        rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            Tegra::Nx2MarkCommandListPage(*device);
            written = true;
        }
    }
    if (!written) {
        return false;
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        const auto wait_start = std::chrono::steady_clock::now();
        if (split.WaitForWriters(true)) {
            Nx2CountList(g_nx2_list_stats.writer_ns, NanosecondsSince(wait_start));
        }
    }
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (Tegra::Nx2IsGpuWritten(*device)) {
            where = rasterizer->ReadGpuWrittenCommandList(*device, part);
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    return true;
}

void DmaPusher::FlushGpuWrittenCommands(GPUVAddr address, size_t size) {
    // Title 1 runs small command lists its culling dispatches fill in each frame. Accurate GPU
    // mode reads every command list through the caches (safe reads); here only the ones on pages
    // the GPU has written are, or their draws run with an older frame's words (trees vanish).
    // These are the lists ReadGpuWrittenCommands can't take from a copy or guest memory (a page's
    // first list, or pages draws write too).
    if (!IsGpuWrittenCommands(address, size)) {
        return;
    }
    // The caches belong to the split's back: it first runs what the front queued (the dispatch
    // that writes these words among it). The whole pages are flushed: their next lists read
    // guest memory until GPU work writes them again.
    // On pages draws write too, the words are compared after the flush with what a read without
    // the drain finds, guest memory as it is now or the copy (nx2.h, Nx2ListDrawRanges).
    list_words_before.resize_destructive(size);
    memory_manager.ReadBlockUnsafe(address, list_words_before.data(), size);
    Nx2CountList(g_nx2_list_stats.drains);
    const auto drain_start = std::chrono::steady_clock::now();
    VideoCommon::GpuSplit::Get().Drain();
    Nx2CountList(g_nx2_list_stats.drain_ns, NanosecondsSince(drain_start));
    const auto flush_start = std::chrono::steady_clock::now();
    const GPUVAddr first_page = address & ~GPUVAddr{0xfff};
    const GPUVAddr end_page = (address + size + 0xfff) & ~GPUVAddr{0xfff};
    if (memory_manager.IsMemoryDirty(first_page, end_page - first_page,
                                     VideoCommon::CacheType::BufferCache)) {
        Nx2CountList(g_nx2_list_stats.flushes);
        // Along with every other command list page draws wrote (nx2.h, Nx2ListFlushBatch).
        boost::container::small_vector<DAddr, 4> pages;
        bool batched = rasterizer != nullptr && Nx2ListFlushBatch();
        for (GPUVAddr page = first_page; batched && page < end_page; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            batched = device.has_value();
            if (batched) {
                pages.push_back(*device);
            }
        }
        // The downloads record into the Vulkan scheduler, which the split's back owns: a drained
        // back still submits from its idle hook, so the front must not flush these pages itself.
        VideoCommon::GpuSplit::Get().Run([this, batched, page_data = pages.data(),
                                          page_count = pages.size(), first_page,
                                          size = end_page - first_page] {
            if (!batched || !rasterizer->FlushGpuWrittenCommandListPages(
                                std::span<const DAddr>{page_data, page_count})) {
                memory_manager.FlushRegion(first_page, size, VideoCommon::CacheType::BufferCache);
            }
        });
    }
    Nx2CountList(g_nx2_list_stats.flush_ns, NanosecondsSince(flush_start));
    if (rasterizer == nullptr) {
        return;
    }
    std::array<u8, 0x1000> words;
    const GPUVAddr end = address + size;
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        if (const auto device = memory_manager.GpuToCpuAddress(chunk);
            device && Tegra::Nx2IsCommandListPage(*device)) {
            const size_t piece = static_cast<size_t>(chunk_end - chunk);
            memory_manager.ReadBlockUnsafe(chunk, words.data(), piece);
            rasterizer->VerifyGpuWrittenCommandList(
                *device, std::span<const u8>{words.data(), piece},
                std::span<const u8>{list_words_before.data() + (chunk - address), piece});
        }
        chunk = chunk_end;
    }
    for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsCommandListPage(*device)) {
            rasterizer->NoteGpuWrittenCommandListSynced(*device);
        }
    }
}

void DmaPusher::ReportUnboundSubchannel() {
    // Methods sent to a subchannel with no engine (no SetObject yet) are dropped.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 8) {
        LOG_WARNING(HW_GPU, "Dropping method {:#x} on unbound subchannel {}", dma_state.method,
                    dma_state.subchannel);
    }
}

void DmaPusher::CheckSubmitHash(const CommandListHeader& header, u64 submit_hash) {
    std::vector<u32> words(header.size);
    memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
    static std::atomic<u64> checked{};
    static std::atomic<u64> changed{};
    ++checked;
    if (HashPushbufferWords(words) != submit_hash) {
        if (changed.fetch_add(1) < 16) {
            LOG_WARNING(HW_GPU, "pushbuffer entry {:#x}+{:#x} changed between submit and execution",
                        header.addr.Value(), header.size.Value());
        }
    }
    if ((checked % 20000) == 0) {
        LOG_WARNING(HW_GPU, "pushbuffer check: {} of {} entries changed after submit",
                    changed.load(), checked.load());
    }
}

void DmaPusher::RecordParsedEntry(const CommandListHeader& header,
                                  std::span<const CommandHeader> commands) {
    if (!g_record_parsed_entries) {
        return;
    }
    u64 hash = 0xcbf29ce484222325ULL;
    for (const CommandHeader& command : commands) {
        hash = (hash ^ command.argument) * 0x100000001b3ULL;
    }
    parsed_entries[parsed_entry_count++ % parsed_entries.size()] = {header.addr, static_cast<u32>(header.size), hash};
}

void DmaPusher::DumpCurrentEntry(const char* reason) {
    // Entries whose memory changed after they were parsed (the CPU rewrote them meanwhile).
    for (size_t i = 0; i < std::min(parsed_entry_count, parsed_entries.size()); ++i) {
        const auto& entry = parsed_entries[(parsed_entry_count - 1 - i) % parsed_entries.size()];
        std::vector<u32> words(entry.size);
        memory_manager.ReadBlockUnsafe(entry.address, words.data(), words.size() * sizeof(u32));
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        LOG_WARNING(HW_GPU, "  parsed -{}: {:#x}+{:#x} {}", i, entry.address, entry.size,
                    hash == entry.hash ? "unchanged" : "CHANGED since parsed");
    }
    // EDEN_NX2_PB_DUMP=<file>: also write the whole command list (u64 address, u32 size and the
    // current words of every entry, then the current entry index) for offline replay.
    if (static const char* const path = std::getenv("EDEN_NX2_PB_DUMP");
        path != nullptr && !dma_pushbuffer.empty()) {
        if (std::FILE* file = std::fopen(path, "wb")) {
            const auto& lists = dma_pushbuffer.front().command_lists;
            const u64 current = dma_pushbuffer_subindex;
            const u64 count = lists.size();
            std::fwrite(&count, sizeof(count), 1, file);
            std::fwrite(&current, sizeof(current), 1, file);
            for (const CommandListHeader& header : lists) {
                const u64 address = header.addr;
                const u32 size = static_cast<u32>(header.size);
                std::vector<u32> words(size);
                memory_manager.ReadBlockUnsafe(address, words.data(), words.size() * sizeof(u32));
                std::fwrite(&address, sizeof(address), 1, file);
                std::fwrite(&size, sizeof(size), 1, file);
                std::fwrite(words.data(), sizeof(u32), words.size(), file);
            }
            std::fclose(file);
        }
    }
    LOG_WARNING(HW_GPU, "pushbuffer dump ({}): method={:#x} sub={} count={} non_inc={} get={:#x}",
                reason, dma_state.method, dma_state.subchannel, dma_state.method_count,
                dma_state.non_incrementing, dma_state.dma_get);
    if (dma_pushbuffer.empty()) {
        return;
    }
    const auto& lists = dma_pushbuffer.front().command_lists;
    const size_t first = dma_pushbuffer_subindex >= 2 ? dma_pushbuffer_subindex - 2 : 0;
    for (size_t i = first; i <= dma_pushbuffer_subindex && i < lists.size(); ++i) {
        const CommandListHeader& header = lists[i];
        LOG_WARNING(HW_GPU, "  entry {} of {}: addr={:#x} size={:#x}", i, lists.size(),
                    header.addr.Value(), header.size.Value());
        for (u64 word = 0; word < std::min<u64>(header.size, 0x80); word += 8) {
            std::string gpu_line;
            std::string cpu_line;
            for (u64 j = word; j < std::min<u64>(header.size, word + 8); ++j) {
                const GPUVAddr address = header.addr + j * sizeof(u32);
                const u32 gpu_value = memory_manager.Read<u32>(address);
                const u32 cpu_value = system.ApplicationMemory().Read32(address);
                gpu_line += fmt::format(" {:08x}", gpu_value);
                cpu_line += gpu_value == cpu_value ? std::string(" ........")
                                                   : fmt::format(" {:08x}", cpu_value);
            }
            LOG_WARNING(HW_GPU, "    +{:03x}:{}", word * 4, gpu_line);
            LOG_WARNING(HW_GPU, "    cpu :{}", cpu_line);
        }
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
