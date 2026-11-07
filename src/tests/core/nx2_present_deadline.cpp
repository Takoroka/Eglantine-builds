// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/service/nx2_present_deadline.h"

using Service::Nx2::PresentDeadline;
using Clock = PresentDeadline::Clock;
using namespace std::chrono_literals;

namespace {
// Waits in 50 ms slices from `from` for `length`, the back busy or not, as WaitForFrame does.
Clock::time_point Wait(PresentDeadline& deadline, Clock::time_point from, Clock::duration length,
                       bool back_busy) {
    Clock::time_point now = from;
    while (now < from + length) {
        const Clock::time_point next = now + 50ms;
        deadline.AfterSlice(now, next, back_busy);
        now = next;
    }
    return now;
}
} // Anonymous namespace

TEST_CASE("Without the bit the deadline is 4 s", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, false};
    Wait(deadline, start, 3s, true);
    REQUIRE_FALSE(deadline.Passed(start + 3999ms));
    REQUIRE(deadline.Passed(start + 4s));
    REQUIRE(deadline.Extended() == Clock::duration::zero());
}

TEST_CASE("An idle back keeps the 4 s deadline", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    Wait(deadline, start, 4s, false);
    REQUIRE(deadline.Passed(start + 4s));
    REQUIRE(deadline.Extended() == Clock::duration::zero());
}

TEST_CASE("The back's work doesn't count toward the deadline", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    // 11-07's Switch Sports case: ~5 s of first-run pipeline builds, then the fence comes.
    const Clock::time_point after_builds = Wait(deadline, start, 5s, true);
    REQUIRE(deadline.Extended() == 5s);
    REQUIRE_FALSE(deadline.Passed(after_builds));
    REQUIRE_FALSE(deadline.Passed(start + 8999ms));
    // Then an idle back: 4 s of it give up as before.
    Wait(deadline, after_builds, 4s, false);
    REQUIRE(deadline.Passed(start + 9s));
}

TEST_CASE("The back's work moves the deadline to 30 s at most", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    Wait(deadline, start, 60s, true);
    REQUIRE_FALSE(deadline.Passed(start + 29999ms));
    REQUIRE(deadline.Passed(start + 30s));
    REQUIRE(deadline.Extended() == 26s);
}

TEST_CASE("Empty and backwards slices move nothing", "[core][nx2_present_deadline]") {
    const Clock::time_point start{};
    PresentDeadline deadline{start, true};
    deadline.AfterSlice(start + 1s, start + 1s, true);
    deadline.AfterSlice(start + 2s, start + 1s, true);
    REQUIRE(deadline.Extended() == Clock::duration::zero());
    REQUIRE(deadline.Passed(start + 4s));
}
