// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/nx2_counters.h"

using namespace Common::Nx2Counters;

TEST_CASE("Nx2Counters rates and shares", "[common][nx2_counters]") {
    CHECK(PerSecond(300, 30'000'000'000) == 10.0);
    CHECK(PerSecond(5, 500'000'000) == 10.0);
    CHECK(PerSecond(7, 0) == 0.0);
    CHECK(PerSecond(7, -1) == 0.0);
    CHECK(Percent(15'000'000'000, 30'000'000'000) == 50.0);
    CHECK(Percent(1, 0) == 0.0);
}

TEST_CASE("Nx2Counters rate text", "[common][nx2_counters]") {
    CHECK(FormatRate(0.0) == "0");
    CHECK(FormatRate(-0.2) == "0");
    CHECK(FormatRate(0.4) == "0");
    CHECK(FormatRate(7.0) == "7");
    CHECK(FormatRate(950.0) == "950");
    CHECK(FormatRate(999.4) == "999");
    CHECK(FormatRate(999.5) == "1.0k");
    CHECK(FormatRate(12'345.0) == "12.3k");
    CHECK(FormatRate(99'949.0) == "99.9k");
    CHECK(FormatRate(99'950.0) == "100k");
    CHECK(FormatRate(123'456.0) == "123k");
    CHECK(FormatRate(1'204'000.0) == "1.20M");
}

TEST_CASE("Nx2Counters top indices", "[common][nx2_counters]") {
    const std::array<u64, 8> values{0, 5, 9, 0, 9, 1, 3, 0};
    CHECK(TopIndices(values, 3) == std::vector<size_t>{2, 4, 1});
    CHECK(TopIndices(values, 12) == std::vector<size_t>{2, 4, 1, 6, 5});
    CHECK(TopIndices(values, 0).empty());
    const std::array<u64, 3> zeros{};
    CHECK(TopIndices(zeros, 12).empty());
}

TEST_CASE("Nx2Counters latency histogram buckets", "[common][nx2_counters]") {
    using H = LatencyHistogram;
    // Exact below 4 ns, then 4 buckets per power of two; every value lies within its bucket.
    for (u64 ns : {0ULL, 1ULL, 3ULL, 4ULL, 7ULL, 8ULL, 9ULL, 1000ULL, 49'152ULL, 57'343ULL,
                   57'344ULL, 1'000'000ULL, (1ULL << 40) + 12345ULL}) {
        const size_t bucket = H::Bucket(ns);
        CAPTURE(ns, bucket);
        CHECK(ns <= H::UpperEnd(bucket));
        if (bucket > 0) {
            CHECK(ns > H::UpperEnd(bucket - 1));
        }
    }
    CHECK(H::Bucket(50'000) == H::Bucket(49'152));
    CHECK(H::UpperEnd(H::Bucket(50'000)) == 57'343);
    // Buckets are at most a quarter of their lower end wide (about 19-25%).
    for (size_t bucket = 5; bucket < H::NUM_BUCKETS; ++bucket) {
        const u64 lower = H::UpperEnd(bucket - 1) + 1;
        CHECK(H::UpperEnd(bucket) - lower + 1 <= lower / 4 + 1);
    }
    // Past the last bucket, values count in it.
    CHECK(H::Bucket(~0ULL) == H::NUM_BUCKETS - 1);
}

TEST_CASE("Nx2Counters latency percentiles", "[common][nx2_counters]") {
    LatencyHistogram histogram;
    CHECK(histogram.Percentile(0.5) == 0);
    for (s64 us = 1; us <= 100; ++us) {
        histogram.Add(us * 1000);
    }
    CHECK(histogram.Count() == 100);
    const s64 p50 = histogram.Percentile(0.5);
    const s64 p99 = histogram.Percentile(0.99);
    CHECK(p50 >= 50'000);
    CHECK(p50 <= 50'000 * 5 / 4);
    CHECK(p99 >= 99'000);
    CHECK(p99 <= 99'000 * 5 / 4);
    CHECK(histogram.Percentile(1.0) >= 100'000);
    CHECK(histogram.Percentile(0.0) <= 1'000 * 5 / 4);
    // Negative durations count as 0.
    histogram.Clear();
    histogram.Add(-7);
    CHECK(histogram.Percentile(0.5) == 0);
    // One outlier moves p99 only when it is more than 1% of the values.
    histogram.Clear();
    for (int i = 0; i < 200; ++i) {
        histogram.Add(10'000);
    }
    histogram.Add(5'000'000);
    CHECK(histogram.Percentile(0.99) <= 10'000 * 5 / 4);
    CHECK(histogram.Percentile(1.0) >= 5'000'000);
}

TEST_CASE("Nx2Counters per-core text", "[common][nx2_counters]") {
    const std::array<u64, 6> per_core{369'000, 28'500, 0, 210, 30, 3'600'000};
    CHECK(FormatPerCore(per_core, 30'000'000'000) == "12.3k/950/0/7/1/120k");
    CHECK(FormatPerCore(std::array<u64, 1>{60}, 30'000'000'000) == "2");
}
