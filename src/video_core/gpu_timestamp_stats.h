// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// EDEN_NX2_GPU_TIMESTAMPS=1 (measurement aid, off by default): the GPU time of each render pass,
// dispatch batch, the uploads and the present, summed per segment key and reported as ms per
// frame (Vulkan::GpuTimestamps records and reads the timestamps).

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"

namespace VideoCommon {

/// Ticks from begin to end on a queue whose timestamps have valid_bits valid bits (the counter
/// wraps around at 2^valid_bits).
[[nodiscard]] constexpr u64 TimestampTicks(u64 begin, u64 end, u32 valid_bits) {
    const u64 mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    return (end - begin) & mask;
}

enum class GpuSegmentKind : u8 {
    Pass,    ///< A render pass (work = draws).
    Compute, ///< Consecutive dispatches of one program, and what was recorded between them.
    Upload,  ///< The upload command buffer of a submission (copies run before its draws).
    Present, ///< Presentation: post-processing, upscaling and the blit to the swapchain.
};

class GpuTimestampStats {
public:
    struct Entry {
        u64 key{};
        GpuSegmentKind kind{};
        std::string label;
        double ns{};
        u64 instances{};
        u64 work{};
    };

    [[nodiscard]] bool Known(u64 key) const {
        return entries.contains(key);
    }

    /// Names a key (once; the label is only made for a new key).
    void Name(u64 key, GpuSegmentKind kind, std::string label) {
        auto [it, inserted] = entries.try_emplace(key);
        if (inserted) {
            it->second.key = key;
            it->second.kind = kind;
            it->second.label = std::move(label);
        }
    }

    /// One timed instance of a named key.
    void Add(u64 key, double ns, u64 work) {
        const auto it = entries.find(key);
        if (it == entries.end()) {
            return;
        }
        it->second.ns += ns;
        ++it->second.instances;
        it->second.work += work;
    }

    void EndFrame() {
        ++frames;
    }

    void SkipFrame() {
        ++skipped_frames;
    }

    void DropSegment() {
        ++dropped_segments;
    }

    [[nodiscard]] u64 Frames() const {
        return frames;
    }

    /// Total GPU ms per frame of a kind (all kinds: the sum).
    [[nodiscard]] double MsPerFrame(const GpuSegmentKind* kind = nullptr) const {
        if (frames == 0) {
            return 0.0;
        }
        double ns = 0.0;
        for (const auto& [key, entry] : entries) {
            if (kind == nullptr || entry.kind == *kind) {
                ns += entry.ns;
            }
        }
        return ns / 1e6 / static_cast<double>(frames);
    }

    /// The keys with the most GPU time first (ties: lower key first), at most `count`.
    [[nodiscard]] std::vector<const Entry*> Top(size_t count) const {
        std::vector<const Entry*> sorted;
        sorted.reserve(entries.size());
        for (const auto& [key, entry] : entries) {
            if (entry.instances != 0) {
                sorted.push_back(&entry);
            }
        }
        std::ranges::sort(sorted, [](const Entry* lhs, const Entry* rhs) {
            return lhs->ns != rhs->ns ? lhs->ns > rhs->ns : lhs->key < rhs->key;
        });
        if (sorted.size() > count) {
            sorted.resize(count);
        }
        return sorted;
    }

    /// One line: frames, ms per frame by kind, then the top keys.
    [[nodiscard]] std::string Report(size_t count) const {
        static constexpr GpuSegmentKind pass = GpuSegmentKind::Pass;
        static constexpr GpuSegmentKind compute = GpuSegmentKind::Compute;
        static constexpr GpuSegmentKind upload = GpuSegmentKind::Upload;
        static constexpr GpuSegmentKind present = GpuSegmentKind::Present;
        std::string line = fmt::format(
            "{} frames ({} skipped, {} segments dropped), {:.2f} ms/frame timed: passes {:.2f}, "
            "compute {:.2f}, uploads {:.2f}, present {:.2f}; top:",
            frames, skipped_frames, dropped_segments, MsPerFrame(), MsPerFrame(&pass),
            MsPerFrame(&compute), MsPerFrame(&upload), MsPerFrame(&present));
        const double per_frame = frames != 0 ? 1.0 / static_cast<double>(frames) : 0.0;
        size_t rank = 0;
        for (const Entry* entry : Top(count)) {
            const char* const unit = entry->kind == GpuSegmentKind::Pass      ? "draws"
                                     : entry->kind == GpuSegmentKind::Compute ? "dispatches"
                                                                              : "";
            line += fmt::format(" {}) {} {:.2f} ms/frame ({:.1f}/frame", ++rank, entry->label,
                                entry->ns / 1e6 * per_frame,
                                static_cast<double>(entry->instances) * per_frame);
            if (*unit != '\0') {
                line += fmt::format(", {:.0f} {} each",
                                    static_cast<double>(entry->work) /
                                        static_cast<double>(entry->instances),
                                    unit);
            }
            line += ')';
        }
        return line;
    }

    /// Starts a new window; labels stay.
    void Reset() {
        for (auto& [key, entry] : entries) {
            entry.ns = 0.0;
            entry.instances = 0;
            entry.work = 0;
        }
        frames = 0;
        skipped_frames = 0;
        dropped_segments = 0;
    }

private:
    std::unordered_map<u64, Entry> entries;
    u64 frames{};
    u64 skipped_frames{};
    u64 dropped_segments{};
};

} // namespace VideoCommon
