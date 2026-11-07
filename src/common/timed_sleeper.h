// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "common/common_types.h"

namespace Common {

/// A sleep another thread can end early, for waits that finish with a precise wait of their own
/// (CoreTiming's cheap timed wait). Windows: a high-resolution waitable timer
/// (CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, Windows 10 1803 and later) and an auto-reset event;
/// elsewhere a condition variable. A Wake while nobody sleeps ends the next sleep at once, so a
/// wake is never lost (at worst a later sleep ends early).
class TimedSleeper {
public:
    enum class Result {
        TimedOut, ///< The duration passed
        Woken,    ///< Wake ended it
        Failed,   ///< The timer could not be armed or waited on (no sleep happened)
    };

    TimedSleeper();
    ~TimedSleeper();

    TimedSleeper(const TimedSleeper&) = delete;
    TimedSleeper& operator=(const TimedSleeper&) = delete;

    /// False when this host has no precise sleep (Windows without high-resolution timers).
    [[nodiscard]] bool IsAvailable() const noexcept;

    /// Sleeps for the duration or until Wake.
    Result SleepFor(std::chrono::nanoseconds duration);

    /// Ends the current sleep, or the next one if nobody sleeps.
    void Wake();

private:
#ifdef _WIN32
    void* timer{};
    void* wake_event{};
#else
    std::mutex mutex;
    std::condition_variable condvar;
    bool woken{};
#endif
};

/// How long before a deadline a TimedSleeper sleep has to end so that the sleep's own lateness
/// (the timer firing late, the thread being scheduled late) stays ahead of the deadline: the
/// largest lateness of the recent sleeps, plus a guard. Until enough sleeps were seen it is a
/// cautious 1 ms. Not thread-safe (one waiting thread owns it).
class SleepMargin {
public:
    static constexpr s64 GUARD_NS = 50'000;
    static constexpr s64 INITIAL_NS = 1'000'000;
    static constexpr s64 MAX_NS = 4'000'000;
    static constexpr u32 WINDOW = 256;  ///< Sleeps per window; the margin covers two windows
    static constexpr u32 WARMUP = 16;   ///< Sleeps before the measured margin is used

    [[nodiscard]] s64 Margin() const noexcept {
        if (samples < WARMUP) {
            return INITIAL_NS;
        }
        return std::clamp((std::max)(previous_max, current_max) + GUARD_NS, GUARD_NS, MAX_NS);
    }

    /// One sleep that timed out: how long after its requested end it returned.
    void AddOvershoot(s64 overshoot_ns) noexcept {
        current_max = (std::max)(current_max, std::max<s64>(overshoot_ns, 0));
        ++samples;
        if (++in_window == WINDOW) {
            previous_max = current_max;
            current_max = 0;
            in_window = 0;
        }
    }

private:
    s64 previous_max{};
    s64 current_max{};
    u32 in_window{};
    u64 samples{};
};

} // namespace Common
