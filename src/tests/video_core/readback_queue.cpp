// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/vulkan_present_queue.h"

using Vulkan::ChoosePresentQueue;
using Vulkan::ChooseReadbackQueue;

TEST_CASE("Download copies get the queue after the presents' one", "[video_core][readback_queue]") {
    // NVIDIA: 16 graphics queues; presents take queue 1, download copies queue 2.
    const auto present = ChoosePresentQueue(0, 0, 16, true);
    const auto readback = ChooseReadbackQueue(present, 16, true);
    REQUIRE(readback.available);
    REQUIRE(readback.readback_index == 2);
    REQUIRE(readback.graphics_queue_count == 3);
    REQUIRE(readback.readback_index != present.present_index);
}

TEST_CASE("Download copies never share the presents' queue", "[video_core][readback_queue]") {
    // Two graphics queues: the second one presents, nothing is left for download copies.
    const auto present = ChoosePresentQueue(0, 0, 2, true);
    REQUIRE(present.present_index == 1);
    const auto readback = ChooseReadbackQueue(present, 2, true);
    REQUIRE_FALSE(readback.available);
    REQUIRE(readback.graphics_queue_count == 2);
}

TEST_CASE("Download copies take queue 1 when presents stay off the graphics family's extra "
          "queues",
          "[video_core][readback_queue]") {
    // Presents on queue 0 (A/B bit PresentOwnQueue off).
    const auto on_graphics = ChoosePresentQueue(0, 0, 16, false);
    const auto first = ChooseReadbackQueue(on_graphics, 16, true);
    REQUIRE(first.available);
    REQUIRE(first.readback_index == 1);
    REQUIRE(first.graphics_queue_count == 2);
    // Presents on another family's queue.
    const auto other_family = ChoosePresentQueue(0, 2, 2, true);
    const auto second = ChooseReadbackQueue(other_family, 2, true);
    REQUIRE(second.available);
    REQUIRE(second.readback_index == 1);
    REQUIRE(second.graphics_queue_count == 2);
}

TEST_CASE("No readback queue on a single-queue graphics family or with the bit off",
          "[video_core][readback_queue]") {
    // AMD and Intel: one graphics queue.
    const auto single = ChoosePresentQueue(0, 0, 1, true);
    const auto none = ChooseReadbackQueue(single, 1, true);
    REQUIRE_FALSE(none.available);
    REQUIRE(none.graphics_queue_count == 1);
    // A/B bit ReadbackQueue off: the device creates the queues it did before.
    for (const bool present_own : {false, true}) {
        const auto present = ChoosePresentQueue(0, 0, 16, present_own);
        const auto off = ChooseReadbackQueue(present, 16, false);
        REQUIRE_FALSE(off.available);
        REQUIRE(off.graphics_queue_count == present.graphics_queue_count);
    }
    static_assert(ChooseReadbackQueue(ChoosePresentQueue(0, 0, 16, true), 16, true).readback_index ==
                  2);
}
