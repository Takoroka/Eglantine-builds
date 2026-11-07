// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <memory>

#include <boost/container/small_vector.hpp>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_readback_queue.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

std::unique_ptr<ReadbackQueue> ReadbackQueue::Create(const Device& device,
                                                     const MasterSemaphore& master_semaphore) {
    const VkSemaphore timeline = master_semaphore.Timeline();
    if (!device.HasReadbackQueue() || timeline == VK_NULL_HANDLE) {
        if (device.HasReadbackQueue()) {
            LOG_INFO(Render_Vulkan, "Download copies stay on the graphics queue: no timeline "
                                    "semaphores to wait on");
        }
        return nullptr;
    }
    return std::make_unique<ReadbackQueue>(device, timeline);
}

ReadbackQueue::ReadbackQueue(const Device& device_, VkSemaphore timeline_)
    : device{device_}, timeline{timeline_}, queue{device.GetReadbackQueue()} {
    const auto& logical = device.GetLogical();
    pool = logical.CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    cmdbufs = pool.Allocate(1);
    fence = logical.CreateFence({
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
}

ReadbackQueue::~ReadbackQueue() = default;

u64 ReadbackQueue::CopyAndWait(VkBuffer dst, std::span<const Region> regions, u64 wait_tick) {
    std::scoped_lock lock{mutex};
    const vk::CommandBuffer cmdbuf{cmdbufs[0], device.GetDispatchLoader()};
    cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    // Consecutive regions of one source go in one command.
    boost::container::small_vector<VkBufferCopy, 16> copies;
    for (size_t i = 0; i < regions.size();) {
        const VkBuffer src = regions[i].src;
        copies.clear();
        for (; i < regions.size() && regions[i].src == src; ++i) {
            copies.push_back(regions[i].copy);
        }
        cmdbuf.CopyBuffer(src, dst, copies);
    }
    // The copies' writes made available to the host before the fence signals.
    static constexpr VkMemoryBarrier HOST_READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           HOST_READ_BARRIER);
    cmdbuf.End();

    // The semaphore wait makes the submission of wait_tick, and everything before it on the
    // graphics queue, happen before the copies (and its writes visible to them).
    const VkCommandBuffer handle = *cmdbuf;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkTimelineSemaphoreSubmitInfo timeline_si{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_tick,
        .signalSemaphoreValueCount = 0,
        .pSignalSemaphoreValues = nullptr,
    };
    const bool waits = wait_tick != 0;
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = waits ? &timeline_si : nullptr,
        .waitSemaphoreCount = waits ? 1U : 0U,
        .pWaitSemaphores = waits ? &timeline : nullptr,
        .pWaitDstStageMask = waits ? &wait_stage : nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &handle,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    const auto start = std::chrono::steady_clock::now();
    vk::Check(queue.Submit(submit_info, *fence));
    if (const VkResult result = fence.Wait(); result != VK_SUCCESS) {
        // As a failed tick wait: say why, then the throw ends the process.
        LOG_CRITICAL(Render_Vulkan, "Waiting for a download copy (tick {}) failed: {}", wait_tick,
                     static_cast<int>(result));
        vk::Check(result);
    }
    const u64 waited = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count());
    fence.Reset();
    return waited;
}

} // namespace Vulkan
