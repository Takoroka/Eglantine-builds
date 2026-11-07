// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/service/nx2_present_deadline.h"
#include "video_core/gpu_build_clock.h"

using Service::Nx2::PresentDeadline;
using VideoCommon::GpuBuildClock;
using Clock = PresentDeadline::Clock;
using namespace std::chrono_literals;

namespace {
// Waits in 50 ms slices from `from` for `length`, the GPU thread building for `built` of each
// slice, as WaitForFrame does.
Clock::time_point Wait(PresentDeadline& deadline, Clock::time_point from, Clock::duration length,
                       Clock::duration built) {
    Clock::time_point now = from;
    while (now < from + length) {
        const Clock::time_point next = now + 50ms;
        deadline.AfterSlice(now, next, built);
        now = next;
    }
    return now;
}
} // Anonymous namespace

TEST_CASE("Without the bit the deadline is 4 s", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, false};
    Wait(deadline, start, 3s, 50ms);
    REQUIRE_FALSE(deadline.Passed(start + 3999ms));
    REQUIRE(deadline.Passed(start + 4s));
    REQUIRE(deadline.Extended() == Clock::duration::zero());
}

TEST_CASE("No builds keep the 4 s deadline", "[core][nx2_present_deadline]") {
    // Kirby Air Riders, 11-07: the back busy for seconds without building; holding the present
    // for that stalled the game. Only build time moves the deadline.
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    Wait(deadline, start, 4s, 0ms);
    REQUIRE(deadline.Passed(start + 4s));
    REQUIRE(deadline.Extended() == Clock::duration::zero());
}

TEST_CASE("Build time doesn't count toward the deadline", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    // 11-07's Switch Sports case: ~5 s of first-run pipeline builds, then the fence comes.
    const Clock::time_point after_builds = Wait(deadline, start, 5s, 50ms);
    REQUIRE(deadline.Extended() == 5s);
    REQUIRE_FALSE(deadline.Passed(after_builds));
    REQUIRE_FALSE(deadline.Passed(start + 8999ms));
    // Then no builds: 4 s of that give up as before.
    Wait(deadline, after_builds, 4s, 0ms);
    REQUIRE(deadline.Passed(start + 9s));
}

TEST_CASE("Only the built part of a slice counts", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    // 20 ms of every 50 ms built for 5 s: 2 s more.
    Wait(deadline, start, 5s, 20ms);
    REQUIRE(deadline.Extended() == 2s);
    REQUIRE_FALSE(deadline.Passed(start + 5999ms));
    REQUIRE(deadline.Passed(start + 6s));
}

TEST_CASE("More build time than the slice counts as the slice", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    deadline.AfterSlice(start, start + 50ms, 400ms);
    REQUIRE(deadline.Extended() == 50ms);
}

TEST_CASE("Builds move the deadline to 30 s at most", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    Wait(deadline, start, 60s, 50ms);
    REQUIRE_FALSE(deadline.Passed(start + 29999ms));
    REQUIRE(deadline.Passed(start + 30s));
    REQUIRE(deadline.Extended() == 26s);
}

TEST_CASE("Empty and backwards slices move nothing", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    deadline.AfterSlice(start + 1s, start + 1s, 50ms);
    deadline.AfterSlice(start + 2s, start + 1s, 50ms);
    deadline.AfterSlice(start, start + 50ms, -10ms);
    REQUIRE(deadline.Extended() == Clock::duration::zero());
    REQUIRE(deadline.Passed(start + 4s));
}

TEST_CASE("The build clock counts finished and running builds", "[core][gpu_build_clock]") {
    GpuBuildClock clock;
    const Clock::time_point t0 = Clock::time_point{} + 1000s;
    REQUIRE(clock.Total(t0) == Clock::duration::zero());
    clock.Begin(t0);
    REQUIRE(clock.Total(t0 + 30ms) == 30ms);
    // Nested scopes (a pending wait inside a build) count once.
    clock.Begin(t0 + 10ms);
    clock.End(t0 + 20ms);
    REQUIRE(clock.Total(t0 + 40ms) == 40ms);
    clock.End(t0 + 50ms);
    REQUIRE(clock.Total(t0 + 80ms) == 50ms);
    clock.Begin(t0 + 100ms);
    REQUIRE(clock.Total(t0 + 125ms) == 75ms);
    clock.End(t0 + 150ms);
    REQUIRE(clock.Total(t0 + 1s) == 100ms);
    // An unmatched end changes nothing.
    clock.End(t0 + 2s);
    REQUIRE(clock.Total(t0 + 3s) == 100ms);
}

TEST_CASE("A wait slice takes the build time inside it", "[core][gpu_build_clock]") {
    GpuBuildClock clock;
    PresentDeadline deadline{Clock::time_point{} + 1000s, true};
    const Clock::time_point slice = Clock::time_point{} + 1000s;
    clock.Begin(slice - 1s); // A build that began before the wait
    const auto before = clock.Total(slice);
    clock.End(slice + 30ms);
    deadline.AfterSlice(slice, slice + 50ms, clock.Total(slice + 50ms) - before);
    REQUIRE(deadline.Extended() == 30ms);
}
