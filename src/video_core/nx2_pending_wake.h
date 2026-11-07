// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "common/common_types.h"

namespace VideoCommon {

/// Wakes the GPU thread while its channels are parked on semaphore acquires and nothing else is
/// queued (GPUThread::ThreadManager, A/B bit PendingWake). Without it the thread yields in a loop
/// for 2 ms, then sleeps a timer tick at a time, keeping a logical CPU busy next to GPUBack.
/// What can meet a parked acquire or bring new work notifies it: a GPU semaphore write
/// (Nx2SemaphoreWaiters::OnWrite), the end of a batch on the split's back thread, a command pushed
/// to the GPU thread. The wake is advisory: the GPU thread looks at its channels again at least
/// every cap (a release the game's CPU writes notifies nothing and shows then).
/// Idea: SerkiGohan's pending wake (Eglantine #681) and MFS-code/Eglantine-steamdeck's latch wake.
class PendingWake {
public:
    /// Why a wait ended early (TakeCauses).
    enum Cause : u32 {
        SemaphoreWrite = 1U << 0,
        BackBatch = 1U << 1,
        Command = 1U << 2,
    };

    static PendingWake& Get() {
        static PendingWake wake;
        return wake;
    }

    /// Whether Wait waits precisely on this machine: a monitored wait (MONITORX/MWAITX or
    /// UMONITOR/UMWAIT) on Windows, a condition variable elsewhere. Windows' own waits end on
    /// timer ticks (0.5-1 ms), longer than the cap: there the GPU thread polls as before.
    [[nodiscard]] static bool Available();

    /// Any thread, after its change is visible (the semaphore word, the queued command). Cheap
    /// while the GPU thread doesn't wait: a fence and a load.
    void Notify(Cause cause) {
        // Orders the caller's change before the look at `armed` (Arm's mirror image): either
        // the GPU thread's last look at its channels sees the change, or this sees it armed.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!armed.load(std::memory_order_relaxed)) {
            return;
        }
        causes.fetch_or(cause, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_seq_cst);
        WakeWaiter();
    }

    /// GPU thread, before its last look at the channels and the queue: the value Wait waits on.
    [[nodiscard]] u32 Arm() {
        armed.store(true, std::memory_order_seq_cst);
        return epoch.load(std::memory_order_seq_cst);
    }

    /// GPU thread, when it goes back to work (notifications cost nothing then).
    void Disarm() {
        armed.store(false, std::memory_order_relaxed);
    }

    /// GPU thread: waits until a Notify since the Arm that returned `seen`, or for `cap`.
    /// Returns whether a Notify ended it. Only where Available.
    bool Wait(u32 seen, std::chrono::nanoseconds cap);

    /// The causes notified since the last call (Cause bits).
    [[nodiscard]] u32 TakeCauses() {
        return causes.exchange(0, std::memory_order_relaxed);
    }

private:
    void WakeWaiter() {
#ifndef _WIN32
        // The waiter checks `epoch` under the mutex before it blocks: taking it here means it
        // either saw the new value or is blocked and gets the notification.
        { std::scoped_lock lk{mutex}; }
        cv.notify_one();
#endif
        // On Windows the waiter monitors `epoch`'s cache line: the write above ends the wait.
    }

    alignas(64) std::atomic<u32> epoch{};
    alignas(64) std::atomic<bool> armed{};
    std::atomic<u32> causes{};
#ifndef _WIN32
    std::mutex mutex;
    std::condition_variable cv;
#endif
};

} // namespace VideoCommon
