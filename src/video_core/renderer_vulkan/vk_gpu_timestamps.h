// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_timestamp_stats.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// EDEN_NX2_GPU_TIMESTAMPS=1 (measurement aid, off by default): the Scheduler writes a timestamp
/// pair around each render pass, dispatch batch, texture upload batch, upload command buffer and
/// the present; a frame's pairs are read once its last submission is done (a frame or two
/// later), and every 30 s one log line ("GPU timestamps:") gives the GPU ms per frame of the
/// segments that took the most.
class GpuTimestamps {
public:
    /// Null when the variable is unset or the device can't time its graphics queue.
    static std::unique_ptr<GpuTimestamps> Create(const Device& device);

    GpuTimestamps(const Device& device, u32 valid_bits, double period_ns);
    ~GpuTimestamps();

    GpuTimestamps(const GpuTimestamps&) = delete;
    GpuTimestamps& operator=(const GpuTimestamps&) = delete;

    /// Queries of one segment in the current frame's pool: begin, then begin + 1 for its end.
    struct Reserved {
        VkQueryPool pool;
        u32 begin_query;
        u32 segment;
    };

    [[nodiscard]] bool Known(u64 key) const {
        return stats.Known(key);
    }

    void Name(u64 key, VideoCommon::GpuSegmentKind kind, std::string label) {
        stats.Name(key, kind, std::move(label));
    }

    /// A segment of a named key in the current frame; nothing when the frame's pool is full or
    /// still waiting for the GPU.
    std::optional<Reserved> Reserve(u64 key);

    /// The draws or dispatches of a segment of the current frame, set when it ends.
    void SetWork(u32 segment, u32 work);

    /// The frame's segments are all closed: they are read once the GPU is done with the
    /// scheduler's current tick. Reads finished frames, and logs every 30 s.
    void EndFrame(Scheduler& scheduler);

private:
    static constexpr size_t SlotCount = 8;
    static constexpr u32 MaxSegments = 1024;

    struct Segment {
        u64 key;
        u32 work;
    };

    struct Slot {
        vk::QueryPool pool;
        std::vector<Segment> segments;
        u64 end_tick{};
        bool pending{}; ///< Closed, waiting for the GPU.
    };

    void Read(Slot& slot);

    const Device& device;
    const u32 valid_bits;
    const double period_ns;
    std::array<Slot, SlotCount> slots;
    size_t current{};
    std::vector<u64> results;
    VideoCommon::GpuTimestampStats stats;
    std::chrono::steady_clock::time_point next_log;
};

} // namespace Vulkan
