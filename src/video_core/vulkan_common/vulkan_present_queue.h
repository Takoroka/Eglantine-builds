// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// The queue the swapchain presents on (Device, Swapchain::Present).
struct PresentQueueChoice {
    u32 graphics_queue_count; ///< Queues the device creates in the graphics family
    u32 present_index;        ///< The present queue's index in the present family
    bool own_queue;           ///< Not the graphics queue: presents needn't hold the submit lock
};

/// The graphics queue is queue 0 of its family, and every submit and sparse bind takes the
/// scheduler's submit lock for it. A present on it holds that lock as well, for as long as the
/// present blocks (a vblank under FIFO). A present on another queue doesn't: queue 0 of another
/// present family, or a second queue of the graphics family when it has one. `separate`: whether
/// presents may leave the submit lock at all (A/B bit PresentOwnQueue); without it they stay on
/// queue 0 of the present family under that lock, as before.
[[nodiscard]] constexpr PresentQueueChoice ChoosePresentQueue(u32 graphics_family,
                                                              u32 present_family,
                                                              u32 graphics_family_queues,
                                                              bool separate) {
    if (!separate) {
        return {.graphics_queue_count = 1, .present_index = 0, .own_queue = false};
    }
    if (present_family != graphics_family) {
        return {.graphics_queue_count = 1, .present_index = 0, .own_queue = true};
    }
    if (graphics_family_queues >= 2) {
        return {.graphics_queue_count = 2, .present_index = 1, .own_queue = true};
    }
    return {.graphics_queue_count = 1, .present_index = 0, .own_queue = false};
}

/// The queue download copies go to (A/B bit 11 ReadbackQueue, BufferCacheRuntime::ReadbackCopies).
struct ReadbackQueueChoice {
    u32 graphics_queue_count; ///< Queues the device creates in the graphics family, presents' included
    u32 readback_index;       ///< Its index in the graphics family (valid only with `available`)
    bool available;
};

/// A graphics-family queue of its own, after the queues `present` takes (present_index 1 when
/// presents use a second one). Not the present queue: a present blocks for a vblank under FIFO
/// while holding that queue's lock. None when the family has no free queue left or when not
/// `wanted`; the downloads then wait for the graphics queue as before.
[[nodiscard]] constexpr ReadbackQueueChoice ChooseReadbackQueue(const PresentQueueChoice& present,
                                                                u32 graphics_family_queues,
                                                                bool wanted) {
    const u32 used = present.graphics_queue_count;
    if (!wanted || graphics_family_queues <= used) {
        return {.graphics_queue_count = used, .readback_index = 0, .available = false};
    }
    return {.graphics_queue_count = used + 1, .readback_index = used, .available = true};
}

} // namespace Vulkan
