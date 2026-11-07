// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Sparse memory binds collected for one vkQueueBindSparse (A/B bit SparseBindAsync: the binds of
// a submission go to the queue in one call that signals a timeline semaphore the submission
// waits for, instead of one call and a fence wait each).

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan {

class SparseBindBatch {
public:
    void AddImage(VkImage image, std::span<const VkSparseImageMemoryBind> binds) {
        Add(images, image_binds, image, binds);
    }

    void AddImageOpaque(VkImage image, std::span<const VkSparseMemoryBind> binds) {
        Add(opaques, opaque_binds, image, binds);
    }

    void AddBuffer(VkBuffer buffer, std::span<const VkSparseMemoryBind> binds) {
        Add(buffers, buffer_binds, buffer, binds);
    }

    /// Moves another batch's binds to the end of this one.
    void Append(SparseBindBatch&& other) {
        for (const auto& range : other.images) {
            AddImage(range.handle, std::span{other.image_binds}.subspan(range.first, range.count));
        }
        for (const auto& range : other.opaques) {
            AddImageOpaque(range.handle,
                           std::span{other.opaque_binds}.subspan(range.first, range.count));
        }
        for (const auto& range : other.buffers) {
            AddBuffer(range.handle,
                      std::span{other.buffer_binds}.subspan(range.first, range.count));
        }
        requests += other.requests;
        other.Clear();
    }

    /// One request (a bind call of an image or buffer): counted, nothing else.
    void CountRequest() noexcept {
        ++requests;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return image_binds.empty() && opaque_binds.empty() && buffer_binds.empty();
    }

    [[nodiscard]] u32 Requests() const noexcept {
        return requests;
    }

    /// Memory binds (blocks, mip tails, buffer ranges) in the batch.
    [[nodiscard]] size_t Binds() const noexcept {
        return image_binds.size() + opaque_binds.size() + buffer_binds.size();
    }

    [[nodiscard]] size_t ImageInfoCount() const noexcept {
        return images.size();
    }

    /// The batch as one VkBindSparseInfo (no semaphores). It points into the batch, which must
    /// stay unchanged while the info is used.
    [[nodiscard]] VkBindSparseInfo Info() {
        image_infos.clear();
        opaque_infos.clear();
        buffer_infos.clear();
        for (const auto& range : images) {
            image_infos.push_back(VkSparseImageMemoryBindInfo{
                .image = range.handle,
                .bindCount = range.count,
                .pBinds = image_binds.data() + range.first,
            });
        }
        for (const auto& range : opaques) {
            opaque_infos.push_back(VkSparseImageOpaqueMemoryBindInfo{
                .image = range.handle,
                .bindCount = range.count,
                .pBinds = opaque_binds.data() + range.first,
            });
        }
        for (const auto& range : buffers) {
            buffer_infos.push_back(VkSparseBufferMemoryBindInfo{
                .buffer = range.handle,
                .bindCount = range.count,
                .pBinds = buffer_binds.data() + range.first,
            });
        }
        return VkBindSparseInfo{
            .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .bufferBindCount = static_cast<u32>(buffer_infos.size()),
            .pBufferBinds = buffer_infos.empty() ? nullptr : buffer_infos.data(),
            .imageOpaqueBindCount = static_cast<u32>(opaque_infos.size()),
            .pImageOpaqueBinds = opaque_infos.empty() ? nullptr : opaque_infos.data(),
            .imageBindCount = static_cast<u32>(image_infos.size()),
            .pImageBinds = image_infos.empty() ? nullptr : image_infos.data(),
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
    }

    void Clear() noexcept {
        images.clear();
        opaques.clear();
        buffers.clear();
        image_binds.clear();
        opaque_binds.clear();
        buffer_binds.clear();
        requests = 0;
    }

private:
    template <typename Handle>
    struct Range {
        Handle handle;
        u32 first;
        u32 count;
    };

    /// Binds of the resource the last range is for extend that range.
    template <typename Handle, typename Bind>
    static void Add(std::vector<Range<Handle>>& ranges, std::vector<Bind>& all, Handle handle,
                    std::span<const Bind> binds) {
        if (binds.empty()) {
            return;
        }
        if (ranges.empty() || ranges.back().handle != handle) {
            ranges.push_back({handle, static_cast<u32>(all.size()), 0});
        }
        all.insert(all.end(), binds.begin(), binds.end());
        ranges.back().count += static_cast<u32>(binds.size());
    }

    std::vector<Range<VkImage>> images;
    std::vector<Range<VkImage>> opaques;
    std::vector<Range<VkBuffer>> buffers;
    std::vector<VkSparseImageMemoryBind> image_binds;
    std::vector<VkSparseMemoryBind> opaque_binds;
    std::vector<VkSparseMemoryBind> buffer_binds;
    std::vector<VkSparseImageMemoryBindInfo> image_infos;
    std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
    std::vector<VkSparseBufferMemoryBindInfo> buffer_infos;
    u32 requests = 0;
};

/// A queue family for sparse binds apart from the graphics queue (A/B bit SparseBindAsync):
/// one with sparse binding that is neither the graphics nor the present family, transfer-only
/// first, then compute. Nothing when there is none: binds stay on the graphics queue.
[[nodiscard]] inline std::optional<u32> ChooseSparseQueueFamily(
    std::span<const VkQueueFamilyProperties> families, u32 graphics_family, u32 present_family) {
    std::optional<u32> compute;
    for (u32 index = 0; index < static_cast<u32>(families.size()); ++index) {
        const VkQueueFamilyProperties& family = families[index];
        if (index == graphics_family || index == present_family || family.queueCount == 0 ||
            (family.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) == 0 ||
            (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            continue;
        }
        if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
            return index;
        }
        if (!compute) {
            compute = index;
        }
    }
    return compute;
}

} // namespace Vulkan
