// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {

constexpr u32 NUM_TEXTURE_SCALING_WORDS = 4;
constexpr u32 NUM_IMAGE_SCALING_WORDS = 2;
constexpr u32 NUM_TEXTURE_AND_IMAGE_SCALING_WORDS =
    NUM_TEXTURE_SCALING_WORDS + NUM_IMAGE_SCALING_WORDS;

struct RescalingLayout {
    alignas(16) std::array<u32, NUM_TEXTURE_SCALING_WORDS> rescaling_textures;
    alignas(16) std::array<u32, NUM_IMAGE_SCALING_WORDS> rescaling_images;
    u32 down_factor;
};
struct RenderAreaLayout {
    std::array<f32, 4> render_area;
};
constexpr u32 RESCALING_LAYOUT_WORDS_OFFSET = offsetof(RescalingLayout, rescaling_textures);
constexpr u32 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET = offsetof(RescalingLayout, down_factor);
constexpr u32 RENDERAREA_LAYOUT_OFFSET = offsetof(RenderAreaLayout, render_area);

// Bindless textures (EDEN_NX2_BINDLESS, graphics pipelines). Textures are sampled from the global
// descriptor heap, set BINDLESS_HEAP_SET: binding t (t = TextureType) is an array of
// BINDLESS_IMAGE_CAPACITY sampled images of that type (variables of another sampled type or depth
// may alias it), binding BINDLESS_SAMPLER_BINDING an array of BINDLESS_SAMPLER_CAPACITY samplers.
// The stage's other descriptors move to set BINDLESS_PER_DRAW_SET, bindings numbered as before.
// Each texture element takes one push constant word: the image slot in the low
// BINDLESS_IMAGE_SLOT_BITS bits, the sampler slot above them. A stage's words start at
// RuntimeInfo::bindless_word_base, in Info::texture_descriptors order (a descriptor of count n takes
// n words, indexed like the array). A bindless stage has one push constant block: RescalingLayout at
// 0, the render area at BINDLESS_RENDER_AREA_OFFSET, the words at BINDLESS_WORDS_OFFSET.
constexpr u32 BINDLESS_HEAP_SET = 0;
constexpr u32 BINDLESS_PER_DRAW_SET = 1;
constexpr u32 BINDLESS_IMAGE_CAPACITY = 1u << 16;
constexpr u32 BINDLESS_SAMPLER_CAPACITY = 1u << 12;
constexpr u32 BINDLESS_SAMPLER_BINDING = Shader::NUM_TEXTURE_TYPES;
constexpr u32 BINDLESS_IMAGE_SLOT_BITS = 20;
constexpr u32 BINDLESS_PUSH_CONSTANT_SIZE = 256;
constexpr u32 BINDLESS_RENDER_AREA_OFFSET = static_cast<u32>(sizeof(RescalingLayout));
constexpr u32 BINDLESS_WORDS_OFFSET =
    BINDLESS_RENDER_AREA_OFFSET + static_cast<u32>(sizeof(RenderAreaLayout));
constexpr u32 NUM_BINDLESS_WORDS =
    (BINDLESS_PUSH_CONSTANT_SIZE - BINDLESS_WORDS_OFFSET) / static_cast<u32>(sizeof(u32));

/// A texture element's push constant word.
[[nodiscard]] constexpr u32 BindlessWord(u32 image_slot, u32 sampler_slot) noexcept {
    return image_slot | (sampler_slot << BINDLESS_IMAGE_SLOT_BITS);
}

[[nodiscard]] std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program, Bindings& bindings);
[[nodiscard]] inline std::vector<u32> EmitSPIRV(const Profile& profile, IR::Program& program) {
    Bindings binding;
    return EmitSPIRV(profile, {}, program, binding);
}

} // namespace Shader::Backend::SPIRV
