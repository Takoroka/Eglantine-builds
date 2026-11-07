// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

namespace Vulkan {

// A/B bit FrontPipelineKey: the split GPU thread's front makes the graphics pipeline state of each
// draw (FixedPipelineState::RefreshFront, from its copy of the 3D engine's registers) and sends the
// part the pipeline lookup compares (Size() bytes) in a ring entry right before the draw's method
// (GpuSplit::DeferBeforeMethod). The back takes it in place of FixedPipelineState::Refresh when
// its lookup runs inside that very method's entry (PipelineCache::TakeFrontKey).
// Idea 22 of the play-dkb plan (front-side pipeline key).

/// The entry sizes: most drivers keep the state up to the vertex layout (dynamic vertex input),
/// some up to the transform feedback state, a few (transform feedback on) all of it.
constexpr size_t FrontKeyWords(size_t bytes) noexcept {
    return (bytes + sizeof(u64) - 1) / sizeof(u64);
}
constexpr size_t FrontKeySmallWords = FrontKeyWords(offsetof(FixedPipelineState, attributes));
constexpr size_t FrontKeyMediumWords = FrontKeyWords(offsetof(FixedPipelineState, xfb_state));
constexpr size_t FrontKeyFullWords = FrontKeyWords(sizeof(FixedPipelineState));

/// The smallest entry for a state of `size` bytes (FixedPipelineState::Size()).
constexpr size_t FrontKeyEntryWords(size_t size) noexcept {
    const size_t words = FrontKeyWords(size);
    return words <= FrontKeySmallWords    ? FrontKeySmallWords
           : words <= FrontKeyMediumWords ? FrontKeyMediumWords
                                          : FrontKeyFullWords;
}

/// Keys the front sent (counted there, reported by the back).
inline std::atomic<u64> g_front_keys_sent{};

class PipelineCache;

/// The ring entry (trivially copyable: GpuSplit copies it into the ring). Runs on the back:
/// PipelineCache::SetFrontKey with the position of the method's entry right after it
/// (operator() is defined in vk_pipeline_cache.cpp, which sends these).
template <size_t Words>
struct FrontKeyEntry {
    PipelineCache* cache;
    u32 size;
    std::array<u64, Words> words;

    void operator()() const;
};

} // namespace Vulkan
