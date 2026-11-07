// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/nx2_wfe_park.h"

using Core::WfeSpinTracker;

TEST_CASE("WFE spins park only after SPIN_NS of one spin", "[core][wfe_park]") {
    WfeSpinTracker tracker;
    constexpr u64 pc = 0x80001234;
    constexpr s64 step = 1'000; // 1 us between WFEs: one spin
    s64 now = 5'000'000;
    // The first SPIN_NS of the spin keep returning at once.
    for (s64 elapsed = 0; elapsed < WfeSpinTracker::SPIN_NS; elapsed += step) {
        CAPTURE(elapsed);
        CHECK_FALSE(tracker.ShouldPark(pc, now + elapsed));
    }
    // From then on it parks, and a park's end keeps the spin going.
    CHECK(tracker.ShouldPark(pc, now + WfeSpinTracker::SPIN_NS));
    tracker.ParkEnded(now + WfeSpinTracker::SPIN_NS + 20'000);
    CHECK(tracker.ShouldPark(pc, now + WfeSpinTracker::SPIN_NS + 21'000));
}

TEST_CASE("WFE spins end at a gap or another PC", "[core][wfe_park]") {
    WfeSpinTracker tracker;
    constexpr u64 pc = 0x80001234;
    s64 now = 1'000'000;
    for (int i = 0; i <= 120; ++i) {
        tracker.ShouldPark(pc, now);
        now += 1'000;
    }
    CHECK(tracker.ShouldPark(pc, now));
    // A gap longer than STREAK_GAP_NS: a new spin, which starts over.
    now += WfeSpinTracker::STREAK_GAP_NS + 1;
    CHECK_FALSE(tracker.ShouldPark(pc, now));
    // A WFE at another PC: a new spin too.
    for (int i = 0; i <= 120; ++i) {
        now += 1'000;
        tracker.ShouldPark(pc, now);
    }
    CHECK(tracker.ShouldPark(pc, now + 1'000));
    CHECK_FALSE(tracker.ShouldPark(pc + 4, now + 2'000));
}
