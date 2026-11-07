// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "common/literals.h"

namespace VideoCommon {

using namespace Common::Literals;

/// The smallest texture cache memory limit: anything under it is raised to it.
constexpr u64 MinTextureCacheMemoryLimit = 1_GiB;

/// The collector's fixed cap before the limit (TextureCache::MaxCriticalMemory), and Auto's floor.
constexpr u64 FixedTextureCacheCriticalCap = 5_GiB;

/// What the Texture Cache Memory Limit setting resolves to on one device.
struct TextureCacheMemoryLimit {
    u64 limit;       ///< Bytes the collector's critical threshold stays under.
    u64 recommended; ///< Auto's value: three quarters of the budget, at least the minimum.
    bool automatic;  ///< The setting is 0 (Auto).
    bool raised;     ///< An explicit value under the minimum was raised to it.
    bool clamped;    ///< An explicit value over the device memory was lowered to it.

    [[nodiscard]] constexpr bool BelowRecommended() const {
        return limit < recommended;
    }
};

/**
 * Resolves the Texture Cache Memory Limit setting.
 * @param setting_mib   The setting in MiB, 0 for Auto.
 * @param budget        The device's memory budget (VK_EXT_memory_budget), or its device-local
 *                      memory without the extension.
 * @param device_memory The device-local heap size (the card's memory), 0 if unknown.
 */
[[nodiscard]] constexpr TextureCacheMemoryLimit ResolveTextureCacheMemoryLimit(u32 setting_mib,
                                                                               u64 budget,
                                                                               u64 device_memory) {
    TextureCacheMemoryLimit result{};
    const u64 ceiling = device_memory >= MinTextureCacheMemoryLimit ? device_memory : ~u64{0};
    result.recommended =
        (std::min)((std::max)(budget / 4 * 3, MinTextureCacheMemoryLimit), ceiling);
    result.automatic = setting_mib == 0;
    if (result.automatic) {
        result.limit = result.recommended;
        return result;
    }
    u64 limit = u64{setting_mib} << 20;
    if (limit < MinTextureCacheMemoryLimit) {
        limit = MinTextureCacheMemoryLimit;
        result.raised = true;
    }
    if (limit > ceiling) {
        limit = ceiling;
        result.clamped = true;
    }
    result.limit = limit;
    return result;
}

/**
 * The cap of the collector's critical threshold where it used a fixed one (`fixed_cap`): an
 * explicit limit replaces it; Auto raises it to three quarters of the budget on cards with more,
 * and never lowers it (8 GB card: 5.3 GiB instead of 5 GiB).
 */
[[nodiscard]] constexpr u64 TextureCacheCriticalCap(const TextureCacheMemoryLimit& limit,
                                                    u64 fixed_cap) {
    if (limit.limit == 0) {
        return fixed_cap;
    }
    return limit.automatic ? (std::max)(limit.limit, fixed_cap) : limit.limit;
}

/**
 * Whether the limit also caps the thresholds a discrete GPU's collector takes from the driver's
 * budget (budget minus margins). Only an explicit limit does: three quarters of an 8 GB card's
 * budget (5.3 GiB) would put the expected threshold at 4.4 GiB, at Mario Kart World's race working
 * set (4.0-4.3 GB), and bring back the evictions and re-decodes the budget thresholds removed.
 */
[[nodiscard]] constexpr bool TextureCacheLimitCapsBudget(const TextureCacheMemoryLimit& limit) {
    return limit.limit != 0 && !limit.automatic;
}

} // namespace VideoCommon
