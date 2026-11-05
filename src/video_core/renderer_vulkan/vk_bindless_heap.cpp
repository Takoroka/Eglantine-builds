// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/vk_bindless_heap.h"
#include "video_core/renderer_vulkan/vk_bindless_table.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
using Shader::Backend::SPIRV::BINDLESS_IMAGE_CAPACITY;
using Shader::Backend::SPIRV::BINDLESS_SAMPLER_BINDING;
using Shader::Backend::SPIRV::BINDLESS_SAMPLER_CAPACITY;

constexpr u32 NUM_BINDINGS = Shader::NUM_TEXTURE_TYPES + 1;

VkWriteDescriptorSet MakeWrite(VkDescriptorSet set, u32 binding, u32 slot, VkDescriptorType type,
                               const VkDescriptorImageInfo* info) {
    return {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = binding,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = info,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
}
} // Anonymous namespace

BindlessHeap::BindlessHeap(const Device& device_) : device{device_} {
    if (!device.IsBindlessSupported()) {
        return;
    }
    const vk::Device& dev = device.GetLogical();
    std::array<VkDescriptorSetLayoutBinding, NUM_BINDINGS> bindings;
    for (u32 binding = 0; binding < NUM_BINDINGS; ++binding) {
        const bool is_sampler = binding == BINDLESS_SAMPLER_BINDING;
        bindings[binding] = {
            .binding = binding,
            .descriptorType = is_sampler ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = is_sampler ? BINDLESS_SAMPLER_CAPACITY : BINDLESS_IMAGE_CAPACITY,
            .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
            .pImmutableSamplers = nullptr,
        };
    }
    std::array<VkDescriptorBindingFlags, NUM_BINDINGS> binding_flags;
    binding_flags.fill(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                       VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                       VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
    const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = NUM_BINDINGS,
        .pBindingFlags = binding_flags.data(),
    };
    layout = dev.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = NUM_BINDINGS,
        .pBindings = bindings.data(),
    });
    const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, Shader::NUM_TEXTURE_TYPES * BINDLESS_IMAGE_CAPACITY},
        {VK_DESCRIPTOR_TYPE_SAMPLER, BINDLESS_SAMPLER_CAPACITY},
    }};
    pool = dev.CreateDescriptorPool({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
                 VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
    const VkDescriptorSetLayout set_layout = *layout;
    sets = pool.Allocate({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    });
    if (sets.IsOutOfPoolMemory()) {
        LOG_ERROR(Render_Vulkan, "Bindless heap: no descriptor set, bindless textures off");
        return;
    }
    default_sampler = dev.CreateSampler({
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
    set = sets[0];
    LOG_INFO(Render_Vulkan, "Bindless heap: {} x {} sampled images, {} samplers",
             Shader::NUM_TEXTURE_TYPES, BINDLESS_IMAGE_CAPACITY, BINDLESS_SAMPLER_CAPACITY);
}

BindlessHeap::~BindlessHeap() = default;

void BindlessHeap::WriteNullSlots(std::span<const VkImageView> null_views) {
    if (!Enabled()) {
        return;
    }
    infos.clear();
    writes.clear();
    // The writes point into infos: no reallocation once they are taken.
    infos.reserve(null_views.size() + 1);
    for (u32 type = 0; type < null_views.size(); ++type) {
        if (null_views[type] == VK_NULL_HANDLE) {
            continue;
        }
        infos.push_back({VK_NULL_HANDLE, null_views[type], VK_IMAGE_LAYOUT_GENERAL});
        writes.push_back(MakeWrite(set, type, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &infos.back()));
    }
    infos.push_back({*default_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    writes.push_back(
        MakeWrite(set, BINDLESS_SAMPLER_BINDING, 0, VK_DESCRIPTOR_TYPE_SAMPLER, &infos.back()));
    device.GetLogical().UpdateDescriptorSets(writes, {});
}

void BindlessHeap::Apply() {
    if (!table->HasWrites()) {
        return;
    }
    const auto image_writes = table->ImageWrites();
    const auto sampler_writes = table->SamplerWrites();
    infos.resize(image_writes.size() + sampler_writes.size());
    writes.clear();
    writes.reserve(infos.size());
    // In queue order: a slot freed and allocated again in between is written twice, the last wins.
    size_t index = 0;
    for (const BindlessTable::ImageWrite& write : image_writes) {
        infos[index] = {VK_NULL_HANDLE, write.view, VK_IMAGE_LAYOUT_GENERAL};
        writes.push_back(MakeWrite(set, static_cast<u32>(write.type), write.slot,
                                   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &infos[index]));
        ++index;
    }
    for (const BindlessTable::SamplerWrite& write : sampler_writes) {
        infos[index] = {write.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        writes.push_back(MakeWrite(set, BINDLESS_SAMPLER_BINDING, write.slot,
                                   VK_DESCRIPTOR_TYPE_SAMPLER, &infos[index]));
        ++index;
    }
    device.GetLogical().UpdateDescriptorSets(writes, {});
    table->ClearWrites();
}

} // namespace Vulkan
