// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <memory>
#include <random>
#include <catch2/catch_test_macros.hpp>
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_front_pipeline_key.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace {

using Vulkan::DynamicFeatures;
using Vulkan::FixedPipelineState;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Hint = Tegra::Engines::Maxwell3D::EngineHint;
using Flags = Tegra::Engines::Maxwell3D::DirtyState::Flags;

DynamicFeatures Features(bool eds, bool eds2_logic, bool eds3_blend, bool dynamic_vi,
                         bool repack) {
    DynamicFeatures f{};
    f.driver_id = 4;
    f.driver_version = 0x12345678;
    f.has_extended_dynamic_state = eds;
    f.has_extended_dynamic_state_2 = true; // FrontRefreshSupported
    f.has_extended_dynamic_state_2_logic_op = eds2_logic;
    f.has_extended_dynamic_state_3_blend = eds3_blend;
    f.has_extended_dynamic_state_3_enables = eds3_blend;
    f.has_dynamic_state3_depth_clamp_enable = eds3_blend;
    f.has_dynamic_state3_logic_op_enable = eds3_blend;
    f.has_dynamic_state3_line_stipple_enable = false;
    f.has_dynamic_vertex_input = dynamic_vi;
    f.has_static_vertex_repack = repack;
    f.has_color_write_enable = eds3_blend;
    f.has_provoking_vertex = true;
    f.has_provoking_vertex_first_mode = true;
    f.has_provoking_vertex_last_mode = true;
    f.has_provoking_vertex_tf_preserve = false;
    return f;
}

// NVIDIA-like, RADV-like, without dynamic vertex input (static repack), and the plain path.
const std::array kFeatureSets{
    Features(true, true, true, true, false),
    Features(true, true, false, true, false),
    Features(true, true, true, false, true),
    Features(false, false, false, false, false),
};

std::unique_ptr<Maxwell> RandomRegs(u32 seed, bool blend) {
    auto regs = std::make_unique<Maxwell>();
    std::mt19937 rng{seed};
    for (u32& word : regs->reg_array) {
        word = static_cast<u32>(rng());
    }
    // Blend factors and equations the packers take (random words are out of their range).
    const auto valid = [&](auto& b) {
        b.color_op = Maxwell::Blend::Equation::Add_D3D;
        b.alpha_op = Maxwell::Blend::Equation::Max_D3D;
        b.color_source = Maxwell::Blend::Factor::SourceAlpha_D3D;
        b.color_dest = Maxwell::Blend::Factor::OneMinusSourceAlpha_D3D;
        b.alpha_source = Maxwell::Blend::Factor::One_D3D;
        b.alpha_dest = Maxwell::Blend::Factor::Zero_D3D;
    };
    valid(regs->blend);
    for (auto& target : regs->blend_per_target) {
        valid(target);
    }
    for (u32& enable : regs->blend.enable) {
        enable = blend ? (rng() & 1) : 0;
    }
    return regs;
}

Flags AllFlags() {
    Flags flags;
    flags.set();
    return flags;
}

FixedPipelineState Back(const Maxwell& regs, Maxwell::PrimitiveTopology topology, Hint hint,
                        const DynamicFeatures& features, Flags& flags) {
    FixedPipelineState state{};
    state.RefreshWith(regs, topology, hint, features, flags);
    return state;
}

FixedPipelineState Front(const Maxwell& regs, Maxwell::PrimitiveTopology topology, Hint hint,
                         const DynamicFeatures& features) {
    FixedPipelineState state{};
    state.RefreshFront(regs, features);
    state.PatchBackFields(topology, hint, features);
    return state;
}

bool SameKey(const FixedPipelineState& a, const FixedPipelineState& b) {
    return a.Size() == b.Size() && std::memcmp(&a, &b, a.Size()) == 0;
}

} // Anonymous namespace

TEST_CASE("The front's state is the back's Refresh", "[video_core][front_pipeline_key]") {
    constexpr std::array topologies{Maxwell::PrimitiveTopology::Triangles,
                                    Maxwell::PrimitiveTopology::TriangleStrip,
                                    Maxwell::PrimitiveTopology::Lines,
                                    Maxwell::PrimitiveTopology::Patches};
    for (const DynamicFeatures& features : kFeatureSets) {
        for (u32 seed = 1; seed <= 24; ++seed) {
            const auto regs = RandomRegs(seed, (seed & 1) != 0);
            const auto topology = topologies[seed % topologies.size()];
            const Hint hint = (seed % 5) == 0 ? Hint::OnHLEMacro : Hint::None;
            Flags flags = AllFlags();
            const FixedPipelineState back = Back(*regs, topology, hint, features, flags);
            const FixedPipelineState front = Front(*regs, topology, hint, features);
            INFO("seed " << seed << ", size " << back.Size());
            REQUIRE(SameKey(front, back));
        }
    }
}

TEST_CASE("The back's state kept per dirty flag stays the front's", "[video_core][front_pipeline_key]") {
    for (const DynamicFeatures& features : kFeatureSets) {
        auto regs = RandomRegs(77, true);
        FixedPipelineState back{};
        Flags flags = AllFlags();
        back.RefreshWith(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features, flags);
        // Registers outside the flagged parts change, no flag: the back redoes them anyway.
        regs->alpha_test_enabled = 1;
        regs->alpha_test_func = Maxwell::ComparisonOp::Greater_GL;
        regs->point_size = 3.0f;
        regs->line_stipple_params.factor.Assign(2);
        Flags none;
        back.RefreshWith(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features, none);
        REQUIRE(SameKey(Front(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features),
                        back));
        // A viewport swizzle changes with its flag set, as the 3D engine sets it.
        regs->viewport_transform[3].swizzle.raw ^= 0x5;
        Flags swizzle;
        swizzle[Vulkan::Dirty::ViewportSwizzles] = true;
        back.RefreshWith(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features,
                         swizzle);
        REQUIRE(SameKey(Front(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features),
                        back));
    }
}

TEST_CASE("Taking a front state clears the flags Refresh clears", "[video_core][front_pipeline_key]") {
    const auto regs = RandomRegs(5, true);
    for (const DynamicFeatures& features : kFeatureSets) {
        Flags refreshed = AllFlags();
        (void)Back(*regs, Maxwell::PrimitiveTopology::Triangles, Hint::None, features, refreshed);
        Flags consumed = AllFlags();
        FixedPipelineState::ConsumeRefreshFlags(consumed, features);
        REQUIRE(refreshed == consumed);
    }
}

TEST_CASE("Front key entries fit every state size", "[video_core][front_pipeline_key]") {
    using namespace Vulkan;
    STATIC_REQUIRE(FrontKeySmallWords <= FrontKeyMediumWords);
    STATIC_REQUIRE(FrontKeyMediumWords <= FrontKeyFullWords);
    const auto regs = RandomRegs(9, false);
    for (const DynamicFeatures& features : kFeatureSets) {
        for (const u32 xfb : {0U, 1U}) {
            auto copy = std::make_unique<Maxwell>(*regs);
            copy->transform_feedback_enabled = xfb;
            const FixedPipelineState state =
                Front(*copy, Maxwell::PrimitiveTopology::Triangles, Hint::None, features);
            const size_t words = FrontKeyEntryWords(state.Size());
            REQUIRE(words * sizeof(u64) >= state.Size());
            REQUIRE(words <= FrontKeyFullWords);
            if (features.has_dynamic_vertex_input && xfb == 0) {
                // The common case takes the small entry.
                REQUIRE(words == FrontKeySmallWords);
            }
        }
    }
}

TEST_CASE("Front keys need extended dynamic state 2", "[video_core][front_pipeline_key]") {
    DynamicFeatures features = kFeatureSets[0];
    REQUIRE(FixedPipelineState::FrontRefreshSupported(features));
    features.has_extended_dynamic_state_2 = false;
    REQUIRE_FALSE(FixedPipelineState::FrontRefreshSupported(features));
}
