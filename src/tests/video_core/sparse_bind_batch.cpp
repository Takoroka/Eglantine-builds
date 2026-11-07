// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/sparse_bind_batch.h"

using Vulkan::SparseBindBatch;

namespace {
template <typename Handle>
Handle FakeHandle(std::uintptr_t value) {
    return reinterpret_cast<Handle>(value);
}

VkSparseImageMemoryBind Block(u32 x, u32 level) {
    return VkSparseImageMemoryBind{
        .subresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = level, .arrayLayer = 0},
        .offset{static_cast<s32>(x), 0, 0},
        .extent{128, 128, 1},
        .memory = FakeHandle<VkDeviceMemory>(0x1000),
        .memoryOffset = x * 65536,
        .flags = 0,
    };
}

VkSparseMemoryBind Range(VkDeviceSize offset) {
    return VkSparseMemoryBind{
        .resourceOffset = offset,
        .size = 65536,
        .memory = FakeHandle<VkDeviceMemory>(0x2000),
        .memoryOffset = offset,
        .flags = 0,
    };
}

VkQueueFamilyProperties Family(VkQueueFlags flags, u32 count = 1) {
    return VkQueueFamilyProperties{
        .queueFlags = flags,
        .queueCount = count,
        .timestampValidBits = 64,
        .minImageTransferGranularity{1, 1, 1},
    };
}
} // Anonymous namespace

TEST_CASE("Sparse bind batch: an empty batch has nothing to bind", "[sparse_bind]") {
    SparseBindBatch batch;
    REQUIRE(batch.Empty());
    batch.AddImage(FakeHandle<VkImage>(0x10), {});
    batch.AddBuffer(FakeHandle<VkBuffer>(0x20), {});
    REQUIRE(batch.Empty());
    const VkBindSparseInfo info = batch.Info();
    REQUIRE(info.sType == VK_STRUCTURE_TYPE_BIND_SPARSE_INFO);
    REQUIRE(info.imageBindCount == 0);
    REQUIRE(info.pImageBinds == nullptr);
    REQUIRE(info.imageOpaqueBindCount == 0);
    REQUIRE(info.bufferBindCount == 0);
    REQUIRE(info.waitSemaphoreCount == 0);
    REQUIRE(info.signalSemaphoreCount == 0);
}

TEST_CASE("Sparse bind batch: one info per resource run, binds in order", "[sparse_bind]") {
    const VkImage image_a = FakeHandle<VkImage>(0x10);
    const VkImage image_b = FakeHandle<VkImage>(0x11);
    const VkBuffer buffer = FakeHandle<VkBuffer>(0x20);
    SparseBindBatch batch;
    // Two requests for image A in a row extend one info; B then A again starts new ones.
    batch.AddImage(image_a, std::array{Block(0, 0), Block(128, 0)});
    batch.AddImage(image_a, std::array{Block(256, 0)});
    batch.AddImage(image_b, std::array{Block(0, 1)});
    batch.AddImage(image_a, std::array{Block(384, 0)});
    batch.AddImageOpaque(image_a, std::array{Range(0x10000)});
    batch.AddBuffer(buffer, std::array{Range(0), Range(65536)});
    REQUIRE_FALSE(batch.Empty());
    REQUIRE(batch.Binds() == 8);
    REQUIRE(batch.ImageInfoCount() == 3);

    const VkBindSparseInfo info = batch.Info();
    REQUIRE(info.imageBindCount == 3);
    REQUIRE(info.pImageBinds[0].image == image_a);
    REQUIRE(info.pImageBinds[0].bindCount == 3);
    REQUIRE(info.pImageBinds[0].pBinds[2].offset.x == 256);
    REQUIRE(info.pImageBinds[1].image == image_b);
    REQUIRE(info.pImageBinds[1].bindCount == 1);
    REQUIRE(info.pImageBinds[1].pBinds[0].subresource.mipLevel == 1);
    REQUIRE(info.pImageBinds[2].image == image_a);
    REQUIRE(info.pImageBinds[2].pBinds[0].offset.x == 384);
    REQUIRE(info.imageOpaqueBindCount == 1);
    REQUIRE(info.pImageOpaqueBinds[0].image == image_a);
    REQUIRE(info.pImageOpaqueBinds[0].pBinds[0].resourceOffset == 0x10000);
    REQUIRE(info.bufferBindCount == 1);
    REQUIRE(info.pBufferBinds[0].buffer == buffer);
    REQUIRE(info.pBufferBinds[0].bindCount == 2);
    REQUIRE(info.pBufferBinds[0].pBinds[1].resourceOffset == 65536);
}

TEST_CASE("Sparse bind batch: appending moves the binds and the request count", "[sparse_bind]") {
    const VkImage image = FakeHandle<VkImage>(0x10);
    SparseBindBatch pending;
    pending.AddImage(image, std::array{Block(0, 0)});
    pending.CountRequest();

    SparseBindBatch request;
    request.AddImage(image, std::array{Block(128, 0)});
    request.AddImageOpaque(image, std::array{Range(0x8000)});
    request.CountRequest();

    pending.Append(std::move(request));
    REQUIRE(request.Empty());
    REQUIRE(request.Requests() == 0);
    REQUIRE(pending.Requests() == 2);
    REQUIRE(pending.Binds() == 3);
    // The same image right after: still one image info.
    const VkBindSparseInfo info = pending.Info();
    REQUIRE(info.imageBindCount == 1);
    REQUIRE(info.pImageBinds[0].bindCount == 2);
    REQUIRE(info.pImageBinds[0].pBinds[1].offset.x == 128);
    REQUIRE(info.imageOpaqueBindCount == 1);

    pending.Clear();
    REQUIRE(pending.Empty());
    REQUIRE(pending.Requests() == 0);
    REQUIRE(pending.Info().imageBindCount == 0);
}

TEST_CASE("Sparse queue family: transfer-only first, never graphics or present",
          "[sparse_bind]") {
    using Vulkan::ChooseSparseQueueFamily;
    constexpr VkQueueFlags graphics = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                                      VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    constexpr VkQueueFlags transfer = VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    constexpr VkQueueFlags compute =
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;

    // NVIDIA: graphics (16), transfer (2), compute (8), video decode.
    const std::vector nvidia{Family(graphics, 16), Family(transfer, 2), Family(compute, 8),
                             Family(VK_QUEUE_VIDEO_DECODE_BIT_KHR)};
    REQUIRE(ChooseSparseQueueFamily(nvidia, 0, 0) == 1u);

    // Compute families only: the first of them.
    const std::vector compute_only{Family(graphics), Family(VK_QUEUE_COMPUTE_BIT),
                                   Family(compute, 4), Family(compute, 4)};
    REQUIRE(ChooseSparseQueueFamily(compute_only, 0, 0) == 2u);

    // A transfer family without sparse binding, or with no queues, doesn't count.
    const std::vector no_sparse{Family(graphics), Family(VK_QUEUE_TRANSFER_BIT),
                                Family(transfer, 0)};
    REQUIRE_FALSE(ChooseSparseQueueFamily(no_sparse, 0, 0).has_value());

    // The present family is left to presents, a second graphics family to graphics.
    const std::vector present_apart{Family(graphics), Family(transfer), Family(graphics)};
    REQUIRE_FALSE(ChooseSparseQueueFamily(present_apart, 0, 1).has_value());
    REQUIRE(ChooseSparseQueueFamily(present_apart, 0, 0) == 1u);

    // Graphics family not first.
    const std::vector graphics_second{Family(transfer), Family(graphics)};
    REQUIRE(ChooseSparseQueueFamily(graphics_second, 1, 1) == 0u);
}
