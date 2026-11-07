// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <span>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// Download copies on a graphics-family queue of their own (A/B bit 11 ReadbackQueue,
/// Device::GetReadbackQueue). A copy waits on the master timeline semaphore for the tick of the
/// last submission that used its sources, so it doesn't sit behind the graphics queue's later
/// work, and its caller waits for that copy alone. The same queue family: no ownership transfer.
/// The semaphore wait orders the copy after that submission's writes; the caller blocks until
/// the copy is done, so nothing it records later can rewrite the sources first.
class ReadbackQueue {
public:
    /// One copy: from `src` into the destination buffer.
    struct Region {
        VkBuffer src;
        VkBufferCopy copy;
    };

    /// Null (Create returns none) without a readback queue or timeline semaphores.
    static std::unique_ptr<ReadbackQueue> Create(const Device& device,
                                                 const MasterSemaphore& master_semaphore);

    ReadbackQueue(const Device& device, VkSemaphore timeline);
    ~ReadbackQueue();

    /// Copies `regions` into `dst` once the GPU reached `wait_tick` (0: no wait), then makes the
    /// writes visible to the host and waits for them. Returns the host's wait in nanoseconds.
    u64 CopyAndWait(VkBuffer dst, std::span<const Region> regions, u64 wait_tick);

private:
    const Device& device;
    VkSemaphore timeline;
    vk::Queue queue;
    vk::CommandPool pool;
    vk::CommandBuffers cmdbufs;
    vk::Fence fence;
    std::mutex mutex; ///< The queue, the command buffer and the fence
};

} // namespace Vulkan
