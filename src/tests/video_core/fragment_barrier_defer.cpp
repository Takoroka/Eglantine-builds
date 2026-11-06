// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>
#include "video_core/renderer_vulkan/fragment_barrier_state.h"
#include "video_core/texture_cache/render_targets.h"

using Vulkan::FragmentBarrierState;

TEST_CASE("Deferred fragment barriers survive unrelated draws", "[video_core][fragment_barrier]") {
    FragmentBarrierState state;
    REQUIRE_FALSE(state.Pending());
    REQUIRE_FALSE(state.EndsPassForDraw(true));
    REQUIRE_FALSE(state.EndsPassOnRequest(false));

    state.Defer();
    REQUIRE(state.Pending());
    // Draws that sample no render target don't end the pass and don't use the barrier up: the
    // next draw that samples one still needs it (a first port lost it here).
    for (int draw = 0; draw < 3; ++draw) {
        REQUIRE_FALSE(state.EndsPassForDraw(false));
        REQUIRE_FALSE(state.EndsPassOnRequest(true));
        REQUIRE(state.Pending());
    }
    REQUIRE(state.EndsPassForDraw(true));
    state.OnPassEnd();
    REQUIRE_FALSE(state.Pending());
    REQUIRE_FALSE(state.EndsPassForDraw(true));
}

TEST_CASE("Render pass requests that may sample end a deferred barrier's pass",
          "[video_core][fragment_barrier]") {
    FragmentBarrierState state;
    state.Defer();
    // Setup-checked draws and clears keep it; blits and texture draws end the pass first.
    REQUIRE_FALSE(state.EndsPassOnRequest(true));
    REQUIRE(state.EndsPassOnRequest(false));
    // A second barrier before the pass ends is the same pending barrier.
    state.Defer();
    REQUIRE(state.Pending());
    state.OnPassEnd();
    REQUIRE_FALSE(state.EndsPassOnRequest(false));
}

TEST_CASE("Draws sampling render target images are found", "[video_core][fragment_barrier]") {
    using VideoCommon::ImageId;
    using VideoCommon::NUM_RT;
    using VideoCommon::SamplesRenderTargetImages;
    const auto id = [](u32 index) { return ImageId{index}; };
    std::array<ImageId, NUM_RT> colors{};
    colors[0] = id(4);
    colors[2] = id(7);
    const ImageId depth = id(9);
    const u32 color0 = 1u << 0;
    const u32 color2 = 1u << 2;
    const u32 depth_bit = 1u << NUM_RT;

    const std::array<ImageId, 3> unrelated{id(1), ImageId{}, id(5)};
    REQUIRE_FALSE(SamplesRenderTargetImages(unrelated, color0 | color2 | depth_bit, colors, depth));

    const std::array<ImageId, 2> samples_color2{id(1), id(7)};
    REQUIRE(SamplesRenderTargetImages(samples_color2, color0 | color2, colors, depth));
    // Only active targets count: target 2 off.
    REQUIRE_FALSE(SamplesRenderTargetImages(samples_color2, color0 | depth_bit, colors, depth));

    const std::array<ImageId, 1> samples_depth{id(9)};
    REQUIRE(SamplesRenderTargetImages(samples_depth, depth_bit, colors, depth));
    REQUIRE_FALSE(SamplesRenderTargetImages(samples_depth, color0 | color2, colors, depth));

    // Null views and an unset depth target match nothing.
    const std::array<ImageId, 2> nulls{ImageId{}, ImageId{}};
    REQUIRE_FALSE(SamplesRenderTargetImages(nulls, color0 | color2 | depth_bit, colors, ImageId{}));
    REQUIRE_FALSE(SamplesRenderTargetImages(samples_color2, 0, colors, depth));
}
