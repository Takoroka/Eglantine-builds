// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class BindlessTable;
class Device;

/// Bindless textures (EDEN_NX2_BINDLESS): the global descriptor set every bindless graphics
/// pipeline binds at Shader::Backend::SPIRV::BINDLESS_HEAP_SET. Binding t (a Shader::TextureType)
/// holds BINDLESS_IMAGE_CAPACITY sampled images, binding BINDLESS_SAMPLER_BINDING the samplers;
/// BindlessTable hands out the slots. Slot 0 of each binding is reserved: the null image view of
/// that type, a default sampler. The set is updated after binding: a slot is written before the
/// first draw that reads it is recorded, and BindlessTable only reuses a view's slot after the view
/// left the delayed destruction ring, when no pending command buffer reads it any more.
class BindlessHeap {
public:
    explicit BindlessHeap(const Device& device);
    ~BindlessHeap();

    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    /// Whether bindless pipelines can be made (Device::IsBindlessSupported).
    [[nodiscard]] bool Enabled() const noexcept {
        return set != VK_NULL_HANDLE;
    }

    [[nodiscard]] VkDescriptorSetLayout Layout() const noexcept {
        return *layout;
    }

    [[nodiscard]] VkDescriptorSet Set() const noexcept {
        return set;
    }

    /// The slot table whose writes Apply makes (the texture cache runtime's).
    void Attach(BindlessTable& table_) noexcept {
        table = &table_;
    }

    [[nodiscard]] BindlessTable& Table() const noexcept {
        return *table;
    }

    /// Writes slot 0: null_views[t] in image binding t (skipped when null) and the default sampler.
    void WriteNullSlots(std::span<const VkImageView> null_views);

    /// Writes the slots the table queued since the last call, in order, and clears its queue. Back
    /// thread, before the draws that read them are recorded.
    void Apply();

private:
    const Device& device;
    BindlessTable* table{};
    vk::DescriptorSetLayout layout;
    vk::DescriptorPool pool;
    vk::DescriptorSets sets;
    VkDescriptorSet set{};
    vk::Sampler default_sampler;
    std::vector<VkDescriptorImageInfo> infos;
    std::vector<VkWriteDescriptorSet> writes;
};

} // namespace Vulkan
