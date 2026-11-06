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

} // namespace Vulkan
