// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/lru_cache.h"
#include "video_core/gpu_timestamp_stats.h"
#include "video_core/texture_cache/memory_priority.h"

using VideoCommon::GpuSegmentKind;
using VideoCommon::GpuTimestampStats;
using VideoCommon::ResidencyTier;
namespace Priority = VideoCommon::MemoryPriority;

TEST_CASE("VRAM priority: levels are ordered and Normal is Vulkan's default",
          "[vram_priority]") {
    STATIC_REQUIRE(Priority::Normal == 0.5f);
    STATIC_REQUIRE(0.0f <= Priority::Cold);
    STATIC_REQUIRE(Priority::Cold < Priority::Normal);
    STATIC_REQUIRE(Priority::Normal < Priority::Buffers);
    STATIC_REQUIRE(Priority::Buffers < Priority::Frame);
    STATIC_REQUIRE(Priority::Frame <= 1.0f);
}

TEST_CASE("VRAM priority: tiers map to their levels", "[vram_priority]") {
    STATIC_REQUIRE(VideoCommon::ResidencyPriority(ResidencyTier::Normal) == Priority::Normal);
    STATIC_REQUIRE(VideoCommon::ResidencyPriority(ResidencyTier::Frame) == Priority::Frame);
    STATIC_REQUIRE(VideoCommon::ResidencyPriority(ResidencyTier::Cold) == Priority::Cold);
}

TEST_CASE("VRAM priority: GPU writes raise, uses bring cold images back", "[vram_priority]") {
    using VideoCommon::ResidencyTierOnUse;
    // A write makes any image the frame's.
    for (const ResidencyTier tier :
         {ResidencyTier::Normal, ResidencyTier::Frame, ResidencyTier::Cold}) {
        REQUIRE(ResidencyTierOnUse(tier, true) == ResidencyTier::Frame);
    }
    // A sample brings a cold image back to normal and leaves the others.
    REQUIRE(ResidencyTierOnUse(ResidencyTier::Cold, false) == ResidencyTier::Normal);
    REQUIRE(ResidencyTierOnUse(ResidencyTier::Normal, false) == ResidencyTier::Normal);
    REQUIRE(ResidencyTierOnUse(ResidencyTier::Frame, false) == ResidencyTier::Frame);
}

namespace {
struct LruTraits {
    using ObjectType = int;
    using TickType = u64;
};
} // Anonymous namespace

TEST_CASE("VRAM priority: the cold scan visits images unused for the cold frames",
          "[vram_priority]") {
    constexpr u64 cold_frames = VideoCommon::ResidencyColdFrames;
    u64 bound{};
    // Nothing can be cold before the first cold_frames frames have passed.
    REQUIRE_FALSE(VideoCommon::ResidencyColdBound(0, cold_frames, bound));
    REQUIRE_FALSE(VideoCommon::ResidencyColdBound(cold_frames, cold_frames, bound));
    REQUIRE(VideoCommon::ResidencyColdBound(cold_frames + 1, cold_frames, bound));
    REQUIRE(bound == 0);

    // Images last used at frames 0, 100, 399, 400, 401 and 900; the scan at frame 1000.
    Common::LeastRecentlyUsedCache<LruTraits> lru;
    const std::vector<u64> last_use{0, 100, 399, 400, 401, 900};
    for (size_t image = 0; image < last_use.size(); ++image) {
        lru.Insert(static_cast<int>(image), last_use[image]);
    }
    constexpr u64 frame_tick = 1000;
    REQUIRE(VideoCommon::ResidencyColdBound(frame_tick, cold_frames, bound));
    std::vector<int> visited;
    lru.ForEachItemBelow(bound, [&](int image) { visited.push_back(image); });
    // Unused for more than 600 frames: last used before frame 400.
    REQUIRE(visited == std::vector<int>{0, 1, 2});
}

TEST_CASE("GPU timestamps: ticks wrap at the valid bits", "[gpu_timestamps]") {
    using VideoCommon::TimestampTicks;
    STATIC_REQUIRE(TimestampTicks(100, 350, 64) == 250);
    STATIC_REQUIRE(TimestampTicks(100, 350, 36) == 250);
    // A 36-bit counter that wrapped between the two marks.
    constexpr u64 top = (u64{1} << 36) - 10;
    STATIC_REQUIRE(TimestampTicks(top, 5, 36) == 15);
    // 64 valid bits wrap at 2^64.
    STATIC_REQUIRE(TimestampTicks(~u64{0} - 4, 5, 64) == 10);
}

TEST_CASE("GPU timestamps: segments add up per key and frame", "[gpu_timestamps]") {
    GpuTimestampStats stats;
    REQUIRE_FALSE(stats.Known(1));
    stats.Name(1, GpuSegmentKind::Pass, "pass 1920x1080");
    stats.Name(1, GpuSegmentKind::Compute, "renamed"); // the first name stays
    stats.Name(2, GpuSegmentKind::Compute, "compute 00000000deadbeef");
    stats.Name(3, GpuSegmentKind::Upload, "texture uploads");
    REQUIRE(stats.Known(1));

    // Two frames: the pass 3 ms each (600 draws), compute 0.5 ms twice a frame, uploads 1 ms once.
    for (int frame = 0; frame < 2; ++frame) {
        stats.Add(1, 3'000'000.0, 600);
        stats.Add(2, 500'000.0, 4);
        stats.Add(2, 500'000.0, 4);
        stats.EndFrame();
    }
    stats.Add(3, 2'000'000.0, 1);
    stats.Add(99, 5'000'000.0, 1); // never named: ignored
    REQUIRE(stats.Frames() == 2);

    constexpr GpuSegmentKind pass = GpuSegmentKind::Pass;
    constexpr GpuSegmentKind compute = GpuSegmentKind::Compute;
    constexpr GpuSegmentKind upload = GpuSegmentKind::Upload;
    REQUIRE(stats.MsPerFrame() == 5.0);
    REQUIRE(stats.MsPerFrame(&pass) == 3.0);
    REQUIRE(stats.MsPerFrame(&compute) == 1.0);
    REQUIRE(stats.MsPerFrame(&upload) == 1.0);

    const auto top = stats.Top(2);
    REQUIRE(top.size() == 2);
    REQUIRE(top[0]->key == 1);
    REQUIRE(top[0]->label == "pass 1920x1080");
    REQUIRE(top[0]->instances == 2);
    REQUIRE(top[0]->work == 1200);
    // Compute and the uploads both took 2 ms: the lower key first.
    REQUIRE(top[1]->key == 2);

    const std::string report = stats.Report(12);
    REQUIRE(report.starts_with("2 frames (0 skipped, 0 segments dropped), 5.00 ms/frame timed: "
                               "passes 3.00, compute 1.00, uploads 1.00, present 0.00; top:"));
    REQUIRE(report.find(" 1) pass 1920x1080 3.00 ms/frame (1.0/frame, 600 draws each)") !=
            std::string::npos);
    REQUIRE(report.find(" 2) compute 00000000deadbeef 1.00 ms/frame (2.0/frame, 4 dispatches "
                        "each)") != std::string::npos);
    REQUIRE(report.find(" 3) texture uploads 1.00 ms/frame (0.5/frame)") != std::string::npos);
}

TEST_CASE("GPU timestamps: a new window keeps the names", "[gpu_timestamps]") {
    GpuTimestampStats stats;
    stats.Name(7, GpuSegmentKind::Present, "present");
    stats.Add(7, 1'000'000.0, 0);
    stats.EndFrame();
    stats.SkipFrame();
    stats.DropSegment();
    REQUIRE(stats.Report(4).starts_with("1 frames (1 skipped, 1 segments dropped)"));

    stats.Reset();
    REQUIRE(stats.Frames() == 0);
    REQUIRE(stats.Known(7));
    REQUIRE(stats.Top(4).empty()); // no instances in the new window
    REQUIRE(stats.MsPerFrame() == 0.0);
    stats.Add(7, 2'000'000.0, 0);
    stats.EndFrame();
    REQUIRE(stats.MsPerFrame() == 2.0);
}
