// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "video_core/vulkan_common/vulkan_present_queue.h"

TEST_CASE("Presents use a second graphics queue when the family has one", "[video_core][present_queue]") {
    using Vulkan::ChoosePresentQueue;
    // NVIDIA: 16 queues in family 0, which presents.
    const auto nvidia = ChoosePresentQueue(0, 0, 16, true);
    REQUIRE(nvidia.graphics_queue_count == 2);
    REQUIRE(nvidia.present_index == 1);
    REQUIRE(nvidia.own_queue);
    // Exactly two queues.
    const auto two = ChoosePresentQueue(0, 0, 2, true);
    REQUIRE(two.graphics_queue_count == 2);
    REQUIRE(two.present_index == 1);
    REQUIRE(two.own_queue);
}

TEST_CASE("A graphics family with one queue presents on the graphics queue", "[video_core][present_queue]") {
    using Vulkan::ChoosePresentQueue;
    const auto single = ChoosePresentQueue(0, 0, 1, true);
    REQUIRE(single.graphics_queue_count == 1);
    REQUIRE(single.present_index == 0);
    REQUIRE_FALSE(single.own_queue);
}

TEST_CASE("A/B bit off keeps today's present queue", "[video_core][present_queue]") {
    using Vulkan::ChoosePresentQueue;
    // Same family or another one: queue 0 of the present family, under the submit lock.
    for (const u32 present_family : {0U, 2U}) {
        for (const u32 queues : {1U, 2U, 16U}) {
            const auto off = ChoosePresentQueue(0, present_family, queues, false);
            REQUIRE(off.graphics_queue_count == 1);
            REQUIRE(off.present_index == 0);
            REQUIRE_FALSE(off.own_queue);
        }
    }
}

TEST_CASE("Another present family's queue is apart from the graphics queue", "[video_core][present_queue]") {
    using Vulkan::ChoosePresentQueue;
    const auto other = ChoosePresentQueue(0, 2, 16, true);
    REQUIRE(other.graphics_queue_count == 1);
    REQUIRE(other.present_index == 0);
    REQUIRE(other.own_queue);
    // Even when the graphics family has a single queue.
    const auto single = ChoosePresentQueue(1, 0, 1, true);
    REQUIRE(single.graphics_queue_count == 1);
    REQUIRE(single.present_index == 0);
    REQUIRE(single.own_queue);
}

TEST_CASE("Present queue choices are compile-time constants", "[video_core][present_queue]") {
    static_assert(Vulkan::ChoosePresentQueue(0, 0, 16, true).present_index == 1);
    static_assert(!Vulkan::ChoosePresentQueue(0, 0, 1, true).own_queue);
    SUCCEED();
}
