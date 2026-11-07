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

TEST_CASE("Nx2Counters per-core text", "[common][nx2_counters]") {
    const std::array<u64, 6> per_core{369'000, 28'500, 0, 210, 30, 3'600'000};
    CHECK(FormatPerCore(per_core, 30'000'000'000) == "12.3k/950/0/7/1/120k");
    CHECK(FormatPerCore(std::array<u64, 1>{60}, 30'000'000'000) == "2");
}
