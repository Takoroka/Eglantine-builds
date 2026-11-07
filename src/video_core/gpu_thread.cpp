// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/nx2_pending_wake.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"
#ifdef _WIN32
#include "common/windows/timer_resolution.h"
#endif

namespace VideoCommon::GPUThread {

namespace {

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

/// A/B bit PendingWake: with channels parked and nothing queued, wait for a wake
/// (VideoCommon::PendingWake) instead of yielding for 2 ms and then sleeping timer ticks.
bool UsePendingWake() {
    static const bool available = VideoCommon::PendingWake::Available();
    return available && Common::Nx2Ab::On(Common::Nx2Ab::PendingWake);
}

/// The longest wait for a wake: a release nobody notifies (the game's CPU writes the word) shows
/// at most this late. EDEN_NX2_PENDING_WAKE_US=<10-2000> sets it (default 200).
std::chrono::nanoseconds PendingWakeCap() {
    static const std::chrono::nanoseconds cap = [] {
        u64 us = 200;
        if (const char* value = std::getenv("EDEN_NX2_PENDING_WAKE_US"); value != nullptr) {
            us = std::clamp<u64>(std::strtoull(value, nullptr, 10), 10, 2000);
        }
        return std::chrono::nanoseconds{std::chrono::microseconds{us}};
    }();
    return cap;
}

void LogPendingWakeChoice() {
    if (!VideoCommon::PendingWake::Available()) {
        LOG_INFO(HW_GPU, "GPU thread: parked channels are polled (yield 2 ms, then timer ticks): "
                         "no monitored wait (MONITORX or UMWAIT) on this CPU");
    } else if (!Common::Nx2Ab::On(Common::Nx2Ab::PendingWake)) {
        LOG_INFO(HW_GPU, "GPU thread: parked channels are polled (yield 2 ms, then timer ticks): "
                         "A/B bit PendingWake is off");
    } else {
        LOG_INFO(HW_GPU, "GPU thread: parked channels wait for a wake, at most {} us (A/B bit "
                         "PendingWake)",
                 std::chrono::duration_cast<std::chrono::microseconds>(PendingWakeCap()).count());
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPU");
        VideoCommon::GpuSplit::MarkGpuThread();
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the polling below).
        auto last_progress = std::chrono::steady_clock::now();
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        auto& wake = VideoCommon::PendingWake::Get();
        auto& stats = Tegra::g_nx2_list_stats;
        // How the last wait for the parked channels ended, for the counters of what resumed
        // right after it.
        enum class LastWait { None, Woken, Capped, Polled };
        LastWait last_wait = LastWait::None;
        LogPendingWakeChoice();
        Tegra::Engines::Puller::may_drain = false;
        while (!stop_token.stop_requested()) {
            Profile::Report();
            split.MaybeStart(system);
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores the CPU releases: poll them meanwhile.
                // Armed before the channels are looked at, so a release after the look ends the
                // wait below at once.
                const bool use_wake = UsePendingWake();
                const u32 seen = use_wake ? wake.Arm() : 0;
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                if (split.Active() && scheduler.HasPendingChannels() && state.queue.Empty()) {
                    // Nothing else to parse: the parked channels may now wait for the back
                    // (Puller::may_drain); while command lists arrive they only look at memory.
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    Tegra::Engines::Puller::may_drain = true;
                    resumed |= scheduler.ResumePending(system.GPU());
                    Tegra::Engines::Puller::may_drain = false;
                }
                if (resumed) {
                    switch (last_wait) {
                    case LastWait::Woken:
                        Tegra::Nx2CountList(stats.pending_woken_resumes);
                        break;
                    case LastWait::Capped:
                        Tegra::Nx2CountList(stats.pending_cap_resumes);
                        break;
                    case LastWait::Polled:
                        Tegra::Nx2CountList(stats.pending_poll_resumes);
                        break;
                    case LastWait::None:
                        break;
                    }
                }
                last_wait = LastWait::None;
                if (!state.queue.TryPop(next)) {
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    const auto wait_start = std::chrono::steady_clock::now();
                    if (use_wake) {
                        // Until a semaphore write, the end of a batch on the back or a command
                        // (PendingWake), or the cap.
                        const bool woken = wake.Wait(seen, PendingWakeCap());
                        wake.Disarm();
                        Tegra::Nx2CountList(stats.pending_waits);
                        Tegra::Nx2CountList(stats.pending_wait_ns, NanosecondsSince(wait_start));
                        last_wait = woken ? LastWait::Woken : LastWait::Capped;
                        if (woken) {
                            Tegra::Nx2CountList(stats.pending_woken);
                            const u32 causes = wake.TakeCauses();
                            if ((causes & VideoCommon::PendingWake::SemaphoreWrite) != 0) {
                                Tegra::Nx2CountList(stats.pending_by_semaphore);
                            }
                            if ((causes & VideoCommon::PendingWake::BackBatch) != 0) {
                                Tegra::Nx2CountList(stats.pending_by_back);
                            }
                            if ((causes & VideoCommon::PendingWake::Command) != 0) {
                                Tegra::Nx2CountList(stats.pending_by_command);
                            }
                        }
                        continue;
                    }
                    // The CPU usually releases the semaphore soon after: poll for two
                    // milliseconds before sleeping, then sleep a timer tick at a time. Both GPU
                    // threads wait meanwhile, so every bit a sleep runs past the release is
                    // lost (title 1: ~9% of its race time is spent here; a 1 ms sleep_for
                    // lasts up to ~1.5 ms).
                    if (wait_start - last_progress < std::chrono::milliseconds{2}) {
                        std::this_thread::yield();
                    } else {
#ifdef _WIN32
                        Common::Windows::SleepForOneTick();
#else
                        std::this_thread::sleep_for(std::chrono::microseconds{250});
#endif
                    }
                    Tegra::Nx2CountList(stats.pending_polls);
                    Tegra::Nx2CountList(stats.pending_poll_ns, NanosecondsSince(wait_start));
                    last_wait = LastWait::Polled;
                    continue;
                }
                if (use_wake) {
                    wake.Disarm();
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            last_progress = std::chrono::steady_clock::now();
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    Profile::Scope scope{Profile::Zone::Push};
                    // The guest wrote what the list reads before submitting it.
                    split.Defer([rasterizer_ptr] { rasterizer_ptr->OnGuestSyncPoint(); });
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); });
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
                        const Tegra::Nx2CountedTime counted{
                            Tegra::g_nx2_list_stats.run_drains,
                            Tegra::g_nx2_list_stats.run_drain_ns};
                        split.DeferFunction(std::move(run->func));
                        split.Drain();
                    } else {
                        split.DeferFunction(std::move(run->func));
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr, size = flush->size] {
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                        rasterizer_ptr->OnCacheInvalidation(addr, size);
                    });
                } else {
                    ASSERT(false);
                }
                split.Defer([state_ptr = &state, fence = next.fence, block = next.block] {
                    SignalCommand(*state_ptr, fence, block);
                });
                split.Publish();
                continue;
            }
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Push};
                // The guest wrote what the list reads before submitting it.
                renderer.ReadRasterizer()->OnGuestSyncPoint();
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads) {
    PushCommand(RunCommand{std::move(func), before_later_reads}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    // The GPU thread may wait for a parked channel's release (A/B bit PendingWake).
    VideoCommon::PendingWake::Get().Notify(VideoCommon::PendingWake::Command);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
    if (rasterizer) {
        rasterizer->OnGpuShutdown();
    }
}

} // namespace VideoCommon::GPUThread
