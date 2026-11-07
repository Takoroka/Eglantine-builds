// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEMP (measurement aid): helpers for the counter log lines the play build writes every 30 s
// ("nx2 timing", "nx2 SVC mix", "nx2 CPU hints"). The counters themselves are relaxed atomics
// next to the code they count; these only turn a window of counts into text.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"

namespace Common::Nx2Counters {

/// Length of a counter window: one log line per counter set per this many nanoseconds at most.
constexpr s64 WINDOW_NS = 30'000'000'000;

/// Rate per second of a count over a window (0 for an empty window).
constexpr double PerSecond(u64 count, s64 window_ns) {
    return window_ns > 0 ? static_cast<double>(count) * 1e9 / static_cast<double>(window_ns)
                         : 0.0;
}

/// Share of a window in percent (0 for an empty window).
constexpr double Percent(s64 part_ns, s64 window_ns) {
    return window_ns > 0 ? static_cast<double>(part_ns) * 100.0 / static_cast<double>(window_ns)
                         : 0.0;
}

/// A rate in a few characters: 0, 7, 950, 12.3k, 123k, 1.20M.
inline std::string FormatRate(double per_second) {
    if (per_second < 999.5) {
        return fmt::format("{:.0f}", std::max(per_second, 0.0));
    }
    if (per_second < 99'950.0) {
        return fmt::format("{:.1f}k", per_second / 1e3);
    }
    if (per_second < 999'500.0) {
        return fmt::format("{:.0f}k", per_second / 1e3);
    }
    return fmt::format("{:.2f}M", per_second / 1e6);
}

/// Indices of the largest non-zero values, largest first (equal values: lower index first),
/// at most max_count of them.
inline std::vector<size_t> TopIndices(std::span<const u64> values, size_t max_count) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != 0) {
            indices.push_back(i);
        }
    }
    const size_t count = std::min(max_count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count),
                      indices.end(), [values](size_t a, size_t b) {
                          return values[a] != values[b] ? values[a] > values[b] : a < b;
                      });
    indices.resize(count);
    return indices;
}

/// Counts of nanosecond durations in buckets about 19% wide (4 per power of two; exact below
/// 4 ns), for percentiles of a window of values. Negative values count as 0.
class LatencyHistogram {
public:
    static constexpr size_t NUM_BUCKETS = 4 * 40; ///< Up to 2^41 ns; larger values count there

    void Add(s64 ns) noexcept {
        ++counts[Bucket(ns > 0 ? static_cast<u64>(ns) : 0)];
        ++total;
    }

    [[nodiscard]] u64 Count() const noexcept {
        return total;
    }

    /// The duration that the fraction p (0 to 1) of the values are at most: the upper end of the
    /// bucket the p-th value fell in (0 when empty).
    [[nodiscard]] s64 Percentile(double p) const noexcept {
        if (total == 0) {
            return 0;
        }
        const double wanted = std::clamp(p, 0.0, 1.0) * static_cast<double>(total);
        const u64 rank = std::max<u64>(static_cast<u64>(wanted + 0.999999), 1);
        u64 seen = 0;
        for (size_t bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            seen += counts[bucket];
            if (seen >= rank) {
                return static_cast<s64>(UpperEnd(bucket));
            }
        }
        return static_cast<s64>(UpperEnd(NUM_BUCKETS - 1));
    }

    void Clear() noexcept {
        counts = {};
        total = 0;
    }

    static constexpr size_t Bucket(u64 ns) noexcept {
        if (ns < 4) {
            return static_cast<size_t>(ns);
        }
        const u32 exponent = static_cast<u32>(std::bit_width(ns)) - 1;
        const u64 mantissa = (ns >> (exponent - 2)) & 3;
        return std::min<size_t>(4 * (exponent - 1) + static_cast<size_t>(mantissa),
                                NUM_BUCKETS - 1);
    }

    /// The largest duration a bucket holds.
    static constexpr u64 UpperEnd(size_t bucket) noexcept {
        if (bucket < 4) {
            return bucket;
        }
        const u32 exponent = static_cast<u32>(bucket / 4) + 1;
        const u64 mantissa = bucket % 4;
        return ((5 + mantissa) << (exponent - 2)) - 1;
    }

private:
    std::array<u32, NUM_BUCKETS> counts{};
    u64 total{};
};

/// Per-core rates joined by '/': "12.3k/950/0/7".
inline std::string FormatPerCore(std::span<const u64> per_core, s64 window_ns) {
    std::string text;
    for (size_t core = 0; core < per_core.size(); ++core) {
        if (core != 0) {
            text += '/';
        }
        text += FormatRate(PerSecond(per_core[core], window_ns));
    }
    return text;
}

} // namespace Common::Nx2Counters
