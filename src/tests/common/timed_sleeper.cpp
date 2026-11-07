// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/timed_sleeper.h"

using namespace std::chrono_literals;
using Common::SleepMargin;
using Common::TimedSleeper;

TEST_CASE("SleepMargin starts cautious and follows the recent timer lateness",
          "[common][timed_sleeper]") {
    SleepMargin margin;
    CHECK(margin.Margin() == SleepMargin::INITIAL_NS);
    for (u32 i = 0; i + 1 < SleepMargin::WARMUP; ++i) {
        margin.AddOvershoot(100'000);
    }
    CHECK(margin.Margin() == SleepMargin::INITIAL_NS);
    margin.AddOvershoot(100'000);
    CHECK(margin.Margin() == 100'000 + SleepMargin::GUARD_NS);

    // One late sleep raises it at once.
    margin.AddOvershoot(700'000);
    CHECK(margin.Margin() == 700'000 + SleepMargin::GUARD_NS);
    // It is remembered for the rest of this window and all of the next one...
    for (u32 i = 0; i < SleepMargin::WINDOW; ++i) {
        margin.AddOvershoot(20'000);
    }
    CHECK(margin.Margin() == 700'000 + SleepMargin::GUARD_NS);
    // ...then forgotten.
    for (u32 i = 0; i < SleepMargin::WINDOW; ++i) {
        margin.AddOvershoot(20'000);
    }
    CHECK(margin.Margin() == 20'000 + SleepMargin::GUARD_NS);

    // Early returns count as on time; the margin keeps its guard and its cap.
    SleepMargin early;
    for (u32 i = 0; i < SleepMargin::WARMUP; ++i) {
        early.AddOvershoot(-5'000);
    }
    CHECK(early.Margin() == SleepMargin::GUARD_NS);
    early.AddOvershoot(60'000'000);
    CHECK(early.Margin() == SleepMargin::MAX_NS);
}

TEST_CASE("TimedSleeper sleeps for the duration", "[common][timed_sleeper]") {
    TimedSleeper sleeper;
    REQUIRE(sleeper.IsAvailable());
    const auto start = std::chrono::steady_clock::now();
    CHECK(sleeper.SleepFor(2ms) == TimedSleeper::Result::TimedOut);
    // The timer must not fire early (a little slack for clock granularity).
    CHECK(std::chrono::steady_clock::now() - start >= 1900us);
}

TEST_CASE("TimedSleeper keeps a wake that came before the sleep", "[common][timed_sleeper]") {
    TimedSleeper sleeper;
    sleeper.Wake();
    const auto start = std::chrono::steady_clock::now();
    CHECK(sleeper.SleepFor(5s) == TimedSleeper::Result::Woken);
    CHECK(std::chrono::steady_clock::now() - start < 1s);
    // The wake was taken: the next sleep runs its course.
    CHECK(sleeper.SleepFor(1ms) == TimedSleeper::Result::TimedOut);
}

TEST_CASE("TimedSleeper wakes from another thread", "[common][timed_sleeper]") {
    TimedSleeper sleeper;
    const auto start = std::chrono::steady_clock::now();
    std::thread waker([&sleeper] {
        std::this_thread::sleep_for(20ms);
        sleeper.Wake();
    });
    const auto result = sleeper.SleepFor(10s);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    waker.join();
    CHECK(result == TimedSleeper::Result::Woken);
    CHECK(elapsed < 5s);
}
