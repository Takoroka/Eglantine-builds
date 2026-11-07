// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_gpu_timestamps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

std::unique_ptr<GpuTimestamps> GpuTimestamps::Create(const Device& device) {
    const char* const value = std::getenv("EDEN_NX2_GPU_TIMESTAMPS");
    if (value == nullptr || value[0] != '1') {
        return nullptr;
    }
    const auto families = device.GetPhysical().GetQueueFamilyProperties();
    const u32 family = device.GetGraphicsFamily();
    const u32 valid_bits = family < families.size() ? families[family].timestampValidBits : 0;
    const float period = device.GetPhysical().GetProperties().limits.timestampPeriod;
    if (valid_bits == 0 || period <= 0.0f || !device.IsHostQueryResetSupported()) {
        LOG_WARNING(Render_Vulkan,
                    "GPU timestamps (EDEN_NX2_GPU_TIMESTAMPS): unavailable on this device "
                    "({} valid bits, period {} ns, host query reset {})",
                    valid_bits, period, device.IsHostQueryResetSupported());
        return nullptr;
    }
    LOG_INFO(Render_Vulkan,
             "GPU timestamps (EDEN_NX2_GPU_TIMESTAMPS): on, {} valid bits, {} ns per tick; a "
             "\"GPU timestamps:\" line every 30 s",
             valid_bits, period);
    return std::make_unique<GpuTimestamps>(device, valid_bits, static_cast<double>(period));
}

GpuTimestamps::GpuTimestamps(const Device& device_, u32 valid_bits_, double period_ns_)
    : device{device_}, valid_bits{valid_bits_}, period_ns{period_ns_},
      next_log{std::chrono::steady_clock::now() + std::chrono::seconds{30}} {
    for (Slot& slot : slots) {
        slot.pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = MaxSegments * 2,
            .pipelineStatistics = 0,
        });
        device.GetLogical().ResetQueryPool(*slot.pool, 0, MaxSegments * 2);
        slot.segments.reserve(256);
    }
}

GpuTimestamps::~GpuTimestamps() = default;

std::optional<GpuTimestamps::Reserved> GpuTimestamps::Reserve(u64 key) {
    Slot& slot = slots[current];
    if (slot.pending || slot.segments.size() >= MaxSegments) {
        stats.DropSegment();
        return std::nullopt;
    }
    const u32 segment = static_cast<u32>(slot.segments.size());
    slot.segments.push_back({key, 0});
    return Reserved{*slot.pool, segment * 2, segment};
}

void GpuTimestamps::SetWork(u32 segment, u32 work) {
    Slot& slot = slots[current];
    if (!slot.pending && segment < slot.segments.size()) {
        slot.segments[segment].work = work;
    }
}

void GpuTimestamps::EndFrame(Scheduler& scheduler) {
    Slot& slot = slots[current];
    if (slot.pending) {
        // Still waiting for the GPU from SlotCount frames ago: this frame recorded nothing.
        stats.SkipFrame();
    } else {
        slot.end_tick = scheduler.CurrentTick();
        slot.pending = true;
        current = (current + 1) % SlotCount;
    }
    // Oldest first; ticks only grow along the ring.
    for (size_t step = 0; step < SlotCount; ++step) {
        Slot& done = slots[(current + step) % SlotCount];
        if (!done.pending) {
            continue;
        }
        if (!scheduler.IsFree(done.end_tick)) {
            break;
        }
        Read(done);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_log) {
        next_log = now + std::chrono::seconds{30};
        if (stats.Frames() != 0) {
            LOG_INFO(Render_Vulkan, "GPU timestamps: {}", stats.Report(12));
        }
        stats.Reset();
    }
}

void GpuTimestamps::Read(Slot& slot) {
    const u32 queries = static_cast<u32>(slot.segments.size()) * 2;
    if (queries != 0) {
        // Value and availability of each query.
        results.assign(static_cast<size_t>(queries) * 2, 0);
        const VkResult result = device.GetLogical().GetQueryResults(
            *slot.pool, 0, queries, results.size() * sizeof(u64), results.data(),
            sizeof(u64) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_SUCCESS || result == VK_NOT_READY) {
            for (size_t index = 0; index < slot.segments.size(); ++index) {
                const u64* const pair = results.data() + index * 4;
                if (pair[1] == 0 || pair[3] == 0) {
                    stats.DropSegment();
                    continue;
                }
                const u64 ticks = VideoCommon::TimestampTicks(pair[0], pair[2], valid_bits);
                stats.Add(slot.segments[index].key, static_cast<double>(ticks) * period_ns,
                          slot.segments[index].work);
            }
        }
        device.GetLogical().ResetQueryPool(*slot.pool, 0, queries);
    }
    stats.EndFrame();
    slot.segments.clear();
    slot.pending = false;
}

} // namespace Vulkan
