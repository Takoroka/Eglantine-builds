// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "video_core/nx2_pending_wake.h"

using VideoCommon::PendingWake;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

TEST_CASE("A notification before the GPU thread arms wakes nothing", "[video_core][pending_wake]") {
    PendingWake wake;
    wake.Notify(PendingWake::Command);
    const u32 seen = wake.Arm();
    REQUIRE(wake.TakeCauses() == 0);
    if (PendingWake::Available()) {
        const auto start = Clock::now();
        REQUIRE_FALSE(wake.Wait(seen, 2ms));
        // The cap is the floor: no wait ends early without a notification.
        REQUIRE(Clock::now() - start >= 2ms);
    }
}

TEST_CASE("A notification after Arm ends the wait at once", "[video_core][pending_wake]") {
    PendingWake wake;
    const u32 seen = wake.Arm();
    wake.Notify(PendingWake::SemaphoreWrite);
    const auto start = Clock::now();
    REQUIRE(wake.Wait(seen, 1s));
    REQUIRE(Clock::now() - start < 100ms);
    REQUIRE(wake.TakeCauses() == PendingWake::SemaphoreWrite);
    REQUIRE(wake.TakeCauses() == 0);
}

TEST_CASE("Disarmed, notifications change nothing", "[video_core][pending_wake]") {
    PendingWake wake;
    const u32 seen = wake.Arm();
    wake.Disarm();
    wake.Notify(PendingWake::BackBatch);
    wake.Notify(PendingWake::Command);
    REQUIRE(wake.Arm() == seen);
    REQUIRE(wake.TakeCauses() == 0);
}

TEST_CASE("Causes of several notifications add up", "[video_core][pending_wake]") {
    PendingWake wake;
    const u32 seen = wake.Arm();
    wake.Notify(PendingWake::BackBatch);
    wake.Notify(PendingWake::Command);
    REQUIRE(wake.Wait(seen, 1s));
    REQUIRE(wake.TakeCauses() == (PendingWake::BackBatch | PendingWake::Command));
}

TEST_CASE("Another thread's notification ends a wait long before the cap",
          "[video_core][pending_wake]") {
    if (!PendingWake::Available()) {
        WARN("No precise wait on this machine: the GPU thread polls as before");
        return;
    }
    PendingWake wake;
    for (int round = 0; round < 20; ++round) {
        const u32 seen = wake.Arm();
        std::thread notifier{[&wake] {
            std::this_thread::sleep_for(5ms);
            wake.Notify(PendingWake::BackBatch);
        }};
        const auto start = Clock::now();
        const bool woken = wake.Wait(seen, 10s);
        const auto waited = Clock::now() - start;
        notifier.join();
        wake.Disarm();
        REQUIRE(woken);
        REQUIRE(waited < 2s);
        REQUIRE(wake.TakeCauses() == PendingWake::BackBatch);
    }
}

TEST_CASE("Waits without notifications end at the cap", "[video_core][pending_wake]") {
    if (!PendingWake::Available()) {
        WARN("No precise wait on this machine: the GPU thread polls as before");
        return;
    }
    PendingWake wake;
    // 50 waits of 200 us take about 10 ms (timer-tick waits would take 25-50 ms; the bound is
    // loose for loaded machines).
    const u32 seen = wake.Arm();
    const auto start = Clock::now();
    for (int i = 0; i < 50; ++i) {
        REQUIRE_FALSE(wake.Wait(seen, 200us));
    }
    const auto waited = Clock::now() - start;
    REQUIRE(waited >= 10ms);
    REQUIRE(waited < 500ms);
}
