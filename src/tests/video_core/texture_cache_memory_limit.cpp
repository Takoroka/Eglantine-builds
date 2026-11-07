// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "video_core/texture_cache/memory_limit.h"

using namespace Common::Literals;
using VideoCommon::ResolveTextureCacheMemoryLimit;

TEST_CASE("Texture cache memory limit: Auto is three quarters of the budget",
          "[texture_cache_memory_limit]") {
    // 8 GB card with a 7.3 GiB budget.
    const auto limit = ResolveTextureCacheMemoryLimit(0, 7475_MiB, 8_GiB);
    REQUIRE(limit.automatic);
    REQUIRE(limit.limit == 7475_MiB / 4 * 3);
    REQUIRE(limit.recommended == limit.limit);
    REQUIRE_FALSE(limit.BelowRecommended());
    REQUIRE_FALSE(limit.raised);
    REQUIRE_FALSE(limit.clamped);
}

TEST_CASE("Texture cache memory limit: Auto follows the budget", "[texture_cache_memory_limit]") {
    const auto full = ResolveTextureCacheMemoryLimit(0, 11_GiB, 12_GiB);
    const auto shrunk = ResolveTextureCacheMemoryLimit(0, 8_GiB, 12_GiB);
    REQUIRE(full.limit == 11_GiB / 4 * 3);
    REQUIRE(shrunk.limit == 6_GiB);
}

TEST_CASE("Texture cache memory limit: Auto never goes under 1 GiB",
          "[texture_cache_memory_limit]") {
    const auto limit = ResolveTextureCacheMemoryLimit(0, 1_GiB, 2_GiB);
    REQUIRE(limit.limit == 1_GiB);
    REQUIRE(limit.recommended == 1_GiB);
}

TEST_CASE("Texture cache memory limit: values under 1 GiB are raised",
          "[texture_cache_memory_limit]") {
    const auto limit = ResolveTextureCacheMemoryLimit(512, 7_GiB, 8_GiB);
    REQUIRE_FALSE(limit.automatic);
    REQUIRE(limit.raised);
    REQUIRE(limit.limit == 1_GiB);
    REQUIRE(limit.BelowRecommended());
}

TEST_CASE("Texture cache memory limit: values over the device memory are lowered",
          "[texture_cache_memory_limit]") {
    const auto limit = ResolveTextureCacheMemoryLimit(32768, 7_GiB, 8_GiB);
    REQUIRE(limit.clamped);
    REQUIRE(limit.limit == 8_GiB);
    REQUIRE_FALSE(limit.BelowRecommended());
}

TEST_CASE("Texture cache memory limit: an explicit value in range is kept",
          "[texture_cache_memory_limit]") {
    const auto low = ResolveTextureCacheMemoryLimit(4096, 7_GiB, 8_GiB);
    REQUIRE(low.limit == 4_GiB);
    REQUIRE(low.BelowRecommended()); // 4 GiB < 5.25 GiB
    const auto high = ResolveTextureCacheMemoryLimit(6144, 7_GiB, 8_GiB);
    REQUIRE(high.limit == 6_GiB);
    REQUIRE_FALSE(high.BelowRecommended());
}

TEST_CASE("Texture cache memory limit: unknown device memory does not clamp",
          "[texture_cache_memory_limit]") {
    const auto limit = ResolveTextureCacheMemoryLimit(20480, 7_GiB, 0);
    REQUIRE(limit.limit == 20_GiB);
    REQUIRE_FALSE(limit.clamped);
}

TEST_CASE("Texture cache memory limit: Auto only raises the fixed cap",
          "[texture_cache_memory_limit]") {
    using VideoCommon::TextureCacheCriticalCap;
    // This PC's 8 GB card (7179 MiB budget): 5384 MiB instead of 5120.
    REQUIRE(TextureCacheCriticalCap(ResolveTextureCacheMemoryLimit(0, 7179_MiB, 8_GiB), 5_GiB) ==
            7179_MiB / 4 * 3);
    // An integrated GPU's 4 GiB share: three quarters is 3 GiB, the fixed cap stays.
    REQUIRE(TextureCacheCriticalCap(ResolveTextureCacheMemoryLimit(0, 4_GiB, 16_GiB), 5_GiB) ==
            5_GiB);
    // An explicit value replaces the cap either way.
    REQUIRE(TextureCacheCriticalCap(ResolveTextureCacheMemoryLimit(3072, 7_GiB, 8_GiB), 5_GiB) ==
            3_GiB);
    REQUIRE(TextureCacheCriticalCap(ResolveTextureCacheMemoryLimit(7168, 7_GiB, 8_GiB), 5_GiB) ==
            7_GiB);
    // Nothing resolved (no device memory information): the fixed cap.
    REQUIRE(TextureCacheCriticalCap(VideoCommon::TextureCacheMemoryLimit{}, 5_GiB) == 5_GiB);
}

TEST_CASE("Texture cache memory limit: only explicit values cap the budget thresholds",
          "[texture_cache_memory_limit]") {
    using VideoCommon::TextureCacheLimitCapsBudget;
    REQUIRE_FALSE(TextureCacheLimitCapsBudget(ResolveTextureCacheMemoryLimit(0, 7_GiB, 8_GiB)));
    REQUIRE(TextureCacheLimitCapsBudget(ResolveTextureCacheMemoryLimit(4096, 7_GiB, 8_GiB)));
    REQUIRE_FALSE(TextureCacheLimitCapsBudget(VideoCommon::TextureCacheMemoryLimit{}));
}
