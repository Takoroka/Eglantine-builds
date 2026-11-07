// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCommon {

/// VK_EXT_memory_priority levels (A/B bit VramPriority). When another process takes VRAM and the
/// driver has to move device memory to system memory (read over PCIe from then on), it moves
/// the lowest priority first: a cold texture rather than the frame's render targets.
namespace MemoryPriority {
/// Texture cache images nothing used for ResidencyColdFrames (needs pageable device memory).
inline constexpr float Cold = 0.25f;
/// Vulkan's default: textures, sparse texture pages, host memory.
inline constexpr float Normal = 0.5f;
/// Device-local buffers: the guest's vertex, index, uniform and storage data, command lists.
inline constexpr float Buffers = 0.75f;
/// Render targets and depth buffers, the stream buffer (uniforms and small uploads).
inline constexpr float Frame = 1.0f;
} // namespace MemoryPriority

/// Where a texture cache image with its own device memory stands.
enum class ResidencyTier : u8 {
    Normal, ///< Sampled.
    Frame,  ///< Written by the GPU: a render target, depth buffer or storage image.
    Cold,   ///< Unused for ResidencyColdFrames.
};

/// Frames without a use before the cold scan lowers an image (~10 s at 60 fps).
inline constexpr u64 ResidencyColdFrames = 600;
/// Frames between two cold scans.
inline constexpr u64 ResidencyScanFrames = 60;
/// Priority changes a cold scan makes at most (the rest wait for the next scan).
inline constexpr u32 ResidencyMaxLoweredPerScan = 256;

[[nodiscard]] constexpr float ResidencyPriority(ResidencyTier tier) {
    switch (tier) {
    case ResidencyTier::Frame:
        return MemoryPriority::Frame;
    case ResidencyTier::Cold:
        return MemoryPriority::Cold;
    case ResidencyTier::Normal:
        break;
    }
    return MemoryPriority::Normal;
}

/// The tier an image moves to when a draw or dispatch uses it: GPU writes make it Frame, a use
/// brings a cold image back to Normal, and a render target that is only sampled stays Frame
/// (until the cold scan finds it unused).
[[nodiscard]] constexpr ResidencyTier ResidencyTierOnUse(ResidencyTier current,
                                                         bool is_modification) {
    if (is_modification) {
        return ResidencyTier::Frame;
    }
    return current == ResidencyTier::Cold ? ResidencyTier::Normal : current;
}

/// The newest frame tick an image can have been used in and still be lowered by a cold scan at
/// frame_tick (LeastRecentlyUsedCache::ForEachItemBelow's bound), or nothing that early.
[[nodiscard]] constexpr bool ResidencyColdBound(u64 frame_tick, u64 cold_frames, u64& bound) {
    if (frame_tick <= cold_frames) {
        return false;
    }
    bound = frame_tick - cold_frames - 1;
    return true;
}

} // namespace VideoCommon
