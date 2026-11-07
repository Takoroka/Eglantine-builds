// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "video_core/engines/instance_loop_hold.h"

using Tegra::Engines::InstanceLoopHold;
using Tegra::Engines::MacroDrawsMatchOneDraw;
using Tegra::Engines::RecordedDraw;

TEST_CASE("A loop of several instances holds its first for the batch", "[video_core][instance_loop_hold]") {
    InstanceLoopHold hold;
    hold.Expect(3);
    REQUIRE(hold.HoldFirst(0));
    REQUIRE(hold.Held());
    // A second End in General mode in the same call draws as before.
    REQUIRE_FALSE(hold.HoldFirst(0));
    // The batch of the other instances covers it: one draw saved.
    REQUIRE(hold.OnBatch());
    REQUIRE_FALSE(hold.Held());
    REQUIRE_FALSE(hold.OnBatch());
}

TEST_CASE("Single-instance calls and continued loops draw as before", "[video_core][instance_loop_hold]") {
    InstanceLoopHold hold;
    hold.Expect(1);
    REQUIRE_FALSE(hold.HoldFirst(0));
    hold.Expect(2);
    // A batch already drew instances of this loop: its first is not the loop's first.
    REQUIRE_FALSE(hold.HoldFirst(4));
    REQUIRE_FALSE(hold.Held());
    // No expectation at all.
    InstanceLoopHold none;
    REQUIRE_FALSE(none.HoldFirst(0));
}

TEST_CASE("A held instance with no batch after it is drawn alone", "[video_core][instance_loop_hold]") {
    InstanceLoopHold hold;
    hold.Expect(2);
    REQUIRE(hold.HoldFirst(0));
    REQUIRE(hold.OnFlushWithoutBatch());
    REQUIRE_FALSE(hold.Held());
    REQUIRE_FALSE(hold.OnFlushWithoutBatch());
}

TEST_CASE("An expectation ends with its call", "[video_core][instance_loop_hold]") {
    InstanceLoopHold hold;
    hold.Expect(4);
    hold.Forget();
    REQUIRE_FALSE(hold.HoldFirst(0));
    // Used once, then gone.
    hold.Expect(4);
    REQUIRE(hold.HoldFirst(0));
    REQUIRE(hold.OnBatch());
    REQUIRE_FALSE(hold.HoldFirst(0));
}

TEST_CASE("The macro path's draws are the one draw plus the first instance", "[video_core][instance_loop_hold]") {
    // Topology triangles (4), 36 indices, 5 instances from index 12, base vertex 100, base 7.
    const std::array<u32, 6> parameters{4, 36, 5, 12, 100, 7};
    const RecordedDraw first{true, 4, 12, 36, 100, 7, 1};
    const RecordedDraw all{true, 4, 12, 36, 100, 7, 5};
    REQUIRE(MacroDrawsMatchOneDraw(std::array{first, all}, parameters));
    // Topology bits above the field are not the topology.
    const std::array<u32, 6> high_bits{0x10004, 36, 5, 12, 100, 7};
    REQUIRE(MacroDrawsMatchOneDraw(std::array{first, all}, high_bits));
    // Anything else is a difference.
    REQUIRE_FALSE(MacroDrawsMatchOneDraw(std::array{all}, parameters));
    REQUIRE_FALSE(MacroDrawsMatchOneDraw(std::array{first, all, all}, parameters));
    RecordedDraw other_base = all;
    other_base.base_instance = 8;
    REQUIRE_FALSE(MacroDrawsMatchOneDraw(std::array{first, other_base}, parameters));
    RecordedDraw fewer = all;
    fewer.instances = 4;
    REQUIRE_FALSE(MacroDrawsMatchOneDraw(std::array{first, fewer}, parameters));
    RecordedDraw arrays = first;
    arrays.indexed = false;
    REQUIRE_FALSE(MacroDrawsMatchOneDraw(std::array{arrays, all}, parameters));
}
