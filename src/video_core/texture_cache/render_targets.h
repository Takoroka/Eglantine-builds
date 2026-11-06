// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>

#include <numeric>
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

/// Framebuffer properties used to lookup a framebuffer
struct RenderTargets {
    constexpr bool operator==(const RenderTargets&) const noexcept = default;

    constexpr bool Contains(std::span<const ImageViewId> elements) const noexcept {
        const auto contains = [elements](ImageViewId item) {
            return std::ranges::find(elements, item) != elements.end();
        };
        return std::ranges::any_of(color_buffer_ids, contains) || contains(depth_buffer_id);
    }

    std::array<ImageViewId, NUM_RT> color_buffer_ids{};
    ImageViewId depth_buffer_id{};
    std::array<u8, NUM_RT> draw_buffers{};
    Extent2D size{};
    bool is_rescaled{};
};

/// Whether one of `images` (a draw's sampled and storage images) is the image of an active
/// render target. rt_active_mask: bit i for color target i, bit NUM_RT for depth/stencil.
[[nodiscard]] constexpr bool SamplesRenderTargetImages(std::span<const ImageId> images,
                                                       u32 rt_active_mask,
                                                       const std::array<ImageId, NUM_RT>& color_images,
                                                       ImageId depth_image) noexcept {
    if (rt_active_mask == 0) {
        return false;
    }
    for (const ImageId image : images) {
        if (!image) {
            continue;
        }
        if ((rt_active_mask & (1u << NUM_RT)) != 0 && image == depth_image) {
            return true;
        }
        for (size_t index = 0; index < NUM_RT; ++index) {
            if ((rt_active_mask & (1u << index)) != 0 && image == color_images[index]) {
                return true;
            }
        }
    }
    return false;
}

} // namespace VideoCommon

namespace std {

template <>
struct hash<VideoCommon::RenderTargets> {
    size_t operator()(const VideoCommon::RenderTargets& rt) const noexcept {
        using VideoCommon::ImageViewId;
        size_t value = std::hash<ImageViewId>{}(rt.depth_buffer_id);
        for (const ImageViewId color_buffer_id : rt.color_buffer_ids) {
            value ^= std::hash<ImageViewId>{}(color_buffer_id);
        }
        value ^= std::bit_cast<u64>(rt.draw_buffers);
        value ^= std::bit_cast<u64>(rt.size);
        return value;
    }
};

} // namespace std
