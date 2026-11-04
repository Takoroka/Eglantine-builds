// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Bindless textures: stable slots in the global descriptor arrays for the image views and
/// samplers that draws use, so a draw passes slot indices instead of writing its descriptors.
///
/// An image slot belongs to one image view and one texture type (the shader's arrays are typed);
/// a sampler slot to one sampler variant (Vulkan samplers depend on the view they are paired with,
/// see Sampler::HandleFor). A slot is allocated the first time a draw uses it and its descriptor
/// write is queued once; the descriptor heap applies the queued writes before the draws that use
/// them. Image slots are freed when their view is destroyed: the texture cache keeps erased views
/// TICKS_TO_DESTROY frames, past the GPU's last use, so a freed slot is no longer read when it is
/// reused. Sampler variants live as long as the texture cache. Back thread only.
class BindlessTable {
public:
    static constexpr u32 INVALID_SLOT = ~0u;
    static constexpr u32 DEFAULT_IMAGE_CAPACITY = 1u << 16;
    static constexpr u32 DEFAULT_SAMPLER_CAPACITY = 1u << 12;

    struct ImageWrite {
        u32 slot;
        Shader::TextureType type;
        VkImageView view;
    };

    struct SamplerWrite {
        u32 slot;
        VkSampler sampler;
    };

    /// Array sizes of the heap: slots are below these. Set before the first slot is allocated.
    void SetCapacity(u32 images, u32 samplers) noexcept {
        image_capacity = images;
        sampler_capacity = samplers;
    }

    [[nodiscard]] u32 ImageCapacity() const noexcept {
        return image_capacity;
    }

    [[nodiscard]] u32 SamplerCapacity() const noexcept {
        return sampler_capacity;
    }

    /// A new image slot for `view` of `type`, its write queued; INVALID_SLOT when all are in use.
    [[nodiscard]] u32 AllocateImage(Shader::TextureType type, VkImageView view) {
        u32 slot;
        if (!free_images.empty()) {
            slot = free_images.back();
            free_images.pop_back();
        } else if (next_image < image_capacity) {
            slot = next_image++;
        } else {
            return INVALID_SLOT;
        }
        image_writes.push_back({slot, type, view});
        ++images_in_use;
        return slot;
    }

    /// A new sampler slot, its write queued; INVALID_SLOT when all are in use.
    [[nodiscard]] u32 AllocateSampler(VkSampler sampler) {
        if (next_sampler >= sampler_capacity) {
            return INVALID_SLOT;
        }
        const u32 slot = next_sampler++;
        sampler_writes.push_back({slot, sampler});
        return slot;
    }

    /// Returns an image slot whose view was destroyed.
    void FreeImage(u32 slot) {
        free_images.push_back(slot);
        --images_in_use;
    }

    /// Descriptor writes queued since the last ClearWrites, oldest first (a slot freed and
    /// allocated again in between appears twice: apply them in order).
    [[nodiscard]] std::span<const ImageWrite> ImageWrites() const noexcept {
        return image_writes;
    }

    [[nodiscard]] std::span<const SamplerWrite> SamplerWrites() const noexcept {
        return sampler_writes;
    }

    [[nodiscard]] bool HasWrites() const noexcept {
        return !image_writes.empty() || !sampler_writes.empty();
    }

    void ClearWrites() noexcept {
        image_writes.clear();
        sampler_writes.clear();
    }

    [[nodiscard]] u32 ImagesInUse() const noexcept {
        return images_in_use;
    }

    [[nodiscard]] u32 SamplersInUse() const noexcept {
        return next_sampler - 1;
    }

private:
    u32 image_capacity = DEFAULT_IMAGE_CAPACITY;
    u32 sampler_capacity = DEFAULT_SAMPLER_CAPACITY;
    // Slot 0 of every array is the heap's null view / default sampler, what a draw reads for a
    // texture without a slot: allocation starts at 1.
    u32 next_image = 1;
    u32 next_sampler = 1;
    u32 images_in_use = 0;
    std::vector<u32> free_images;
    std::vector<ImageWrite> image_writes;
    std::vector<SamplerWrite> sampler_writes;
};

/// An image view's bindless slots, one per texture type; freed with the view. Moving it moves the
/// slots (the texture cache moves erased views into its delayed destruction ring).
class BindlessViewSlots {
public:
    BindlessViewSlots() noexcept {
        slots.fill(BindlessTable::INVALID_SLOT);
    }

    ~BindlessViewSlots() {
        Release();
    }

    BindlessViewSlots(const BindlessViewSlots&) = delete;
    BindlessViewSlots& operator=(const BindlessViewSlots&) = delete;

    BindlessViewSlots(BindlessViewSlots&& rhs) noexcept
        : table{std::exchange(rhs.table, nullptr)}, slots{rhs.slots} {
        rhs.slots.fill(BindlessTable::INVALID_SLOT);
    }

    BindlessViewSlots& operator=(BindlessViewSlots&& rhs) noexcept {
        if (this != &rhs) {
            Release();
            table = std::exchange(rhs.table, nullptr);
            slots = rhs.slots;
            rhs.slots.fill(BindlessTable::INVALID_SLOT);
        }
        return *this;
    }

    /// The slot of `type`, allocated from `table_` for `view` on first use; INVALID_SLOT when the
    /// table is full.
    [[nodiscard]] u32 Get(BindlessTable& table_, Shader::TextureType type, VkImageView view) {
        u32& slot = slots[static_cast<size_t>(type)];
        if (slot == BindlessTable::INVALID_SLOT) [[unlikely]] {
            table = &table_;
            slot = table_.AllocateImage(type, view);
        }
        return slot;
    }

private:
    void Release() noexcept {
        if (table == nullptr) {
            return;
        }
        for (u32& slot : slots) {
            if (slot != BindlessTable::INVALID_SLOT) {
                table->FreeImage(slot);
                slot = BindlessTable::INVALID_SLOT;
            }
        }
        table = nullptr;
    }

    BindlessTable* table = nullptr;
    std::array<u32, Shader::NUM_TEXTURE_TYPES> slots;
};

/// One texture element of a draw for the shader: the slots it samples with.
struct BindlessTexture {
    u32 image_slot;
    u32 sampler_slot;
};

} // namespace Vulkan
