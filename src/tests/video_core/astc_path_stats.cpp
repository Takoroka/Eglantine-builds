// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/astc_path_stats.h"

using VideoCommon::AstcCpuReason;
using VideoCommon::ClassifyCpuAstc;
using VideoCommon::ImageType;

TEST_CASE("ASTC path stats: why an upload missed the GPU transcode", "[astc_path_stats]") {
    // 3D first, whatever the setting; then slices; then the decode setting; else a fallback.
    STATIC_REQUIRE(ClassifyCpuAstc(ImageType::e3D, 16, true) == AstcCpuReason::Volume);
    STATIC_REQUIRE(ClassifyCpuAstc(ImageType::e3D, 1, false) == AstcCpuReason::Volume);
    STATIC_REQUIRE(ClassifyCpuAstc(ImageType::e2D, 4, true) == AstcCpuReason::Depth);
    STATIC_REQUIRE(ClassifyCpuAstc(ImageType::e2D, 1, false) == AstcCpuReason::Setting);
    STATIC_REQUIRE(ClassifyCpuAstc(ImageType::e2D, 1, true) == AstcCpuReason::Other);
    STATIC_REQUIRE(VideoCommon::AstcCopyTexels(256, 128, 1, 6) == 196608);
}

TEST_CASE("ASTC path stats: one line per 30 s window, nothing without uploads",
          "[astc_path_stats]") {
    VideoCommon::AstcPathStats stats;
    const auto start = std::chrono::steady_clock::time_point{} + std::chrono::hours{1};
    // The first call starts the window.
    REQUIRE_FALSE(stats.TakeReport(start).has_value());
    stats.CountGpu(true, 2'000'000);
    stats.CountGpu(true, 500'000);
    stats.CountGpu(false, 1'000'000);
    stats.CountCpu(AstcCpuReason::Volume, 300'000, 4'000'000);
    stats.CountCpu(AstcCpuReason::Setting, 100'000, 1'500'000);
    REQUIRE_FALSE(stats.TakeReport(start + std::chrono::seconds{29}).has_value());
    const auto report = stats.TakeReport(start + std::chrono::seconds{30});
    REQUIRE(report.has_value());
    REQUIRE(*report == "ASTC uploads last 30 s: GPU decode + BC encode 2 (2.5 Mtexels), GPU decode "
                       "only 1 (1.0 Mtexels), CPU 2 (0.4 Mtexels, 5.5 ms): 3D 1, depth > 1 0, "
                       "decode setting 1, other 0");
    // A new window: nothing counted, nothing logged.
    REQUIRE_FALSE(stats.TakeReport(start + std::chrono::seconds{61}).has_value());
    REQUIRE_FALSE(stats.Report(false).has_value());
}
