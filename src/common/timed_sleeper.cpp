// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/timed_sleeper.h"

#ifdef _WIN32
#include <array>
#include <windows.h>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

namespace Common {

#ifdef _WIN32

TimedSleeper::TimedSleeper() {
    timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                   TIMER_ALL_ACCESS);
    wake_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
}

TimedSleeper::~TimedSleeper() {
    if (timer != nullptr) {
        CloseHandle(timer);
    }
    if (wake_event != nullptr) {
        CloseHandle(wake_event);
    }
}

bool TimedSleeper::IsAvailable() const noexcept {
    return timer != nullptr && wake_event != nullptr;
}

TimedSleeper::Result TimedSleeper::SleepFor(std::chrono::nanoseconds duration) {
    if (!IsAvailable()) {
        return Result::Failed;
    }
    // Relative due time (negative), in 100 ns units.
    LARGE_INTEGER due{};
    due.QuadPart = -std::max<LONGLONG>(static_cast<LONGLONG>(duration.count() / 100), 1);
    if (!SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
        return Result::Failed;
    }
    // The wake event first: when both are signaled, the wake wins and is consumed.
    const std::array<HANDLE, 2> handles{wake_event, timer};
    switch (WaitForMultipleObjects(static_cast<DWORD>(handles.size()), handles.data(), FALSE,
                                   INFINITE)) {
    case WAIT_OBJECT_0:
        CancelWaitableTimer(timer);
        return Result::Woken;
    case WAIT_OBJECT_0 + 1:
        return Result::TimedOut;
    default:
        CancelWaitableTimer(timer);
        return Result::Failed;
    }
}

void TimedSleeper::Wake() {
    if (wake_event != nullptr) {
        SetEvent(wake_event);
    }
}

#else

TimedSleeper::TimedSleeper() = default;
TimedSleeper::~TimedSleeper() = default;

bool TimedSleeper::IsAvailable() const noexcept {
    return true;
}

TimedSleeper::Result TimedSleeper::SleepFor(std::chrono::nanoseconds duration) {
    std::unique_lock lock{mutex};
    const bool woken_now = condvar.wait_for(lock, duration, [this] { return woken; });
    woken = false;
    return woken_now ? Result::Woken : Result::TimedOut;
}

void TimedSleeper::Wake() {
    {
        std::scoped_lock lock{mutex};
        woken = true;
    }
    condvar.notify_one();
}

#endif

} // namespace Common
