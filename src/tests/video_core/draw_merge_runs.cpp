// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/draw_merge_runs.h"

using VideoCommon::DrawMergeRuns;

namespace {
// A draw with the full setup (it may start runs), a repeat, a uniform-only refresh, an instanced
// or indirect draw.
void Full(DrawMergeRuns& runs) {
    runs.Note(true, false, false);
}
void Repeat(DrawMergeRuns& runs) {
    runs.Note(true, true, false);
}
void Uniforms(DrawMergeRuns& runs) {
    runs.Note(true, false, true);
}
void Instanced(DrawMergeRuns& runs) {
    runs.Note(false, true, true);
}
} // Anonymous namespace

TEST_CASE("DrawMergeRuns: a run starts at a draw and grows with uniform-only changes",
          "[video_core]") {
    DrawMergeRuns runs;
    Full(runs);
    Uniforms(runs);
    Uniforms(runs);
    Repeat(runs);
    // Open: counted once it ends.
    REQUIRE(runs.Get().patch_runs == 0);
    Full(runs); // ends the run of 4, starts another
    const auto& totals = runs.Get();
    REQUIRE(totals.draws == 5);
    REQUIRE(totals.patch_runs == 1);
    REQUIRE(totals.patch_draws == 4);
    REQUIRE(totals.patch_lengths[1] == 1); // 3-4
    // The plain run: the uniform refreshes broke it; only the last two (Uniforms, Repeat) are one.
    REQUIRE(totals.plain_runs == 1);
    REQUIRE(totals.plain_draws == 2);
}

TEST_CASE("DrawMergeRuns: single draws and instanced draws make no runs", "[video_core]") {
    DrawMergeRuns runs;
    Full(runs);
    Full(runs);
    Instanced(runs); // ends nothing of 2: neither full draw was followed by a joiner
    Repeat(runs);    // an instanced draw starts no run, so this repeat starts one alone
    Full(runs);
    const auto& totals = runs.Get();
    REQUIRE(totals.draws == 5);
    REQUIRE(totals.patch_runs == 0);
    REQUIRE(totals.plain_runs == 0);
}

TEST_CASE("DrawMergeRuns: run lengths go to their buckets", "[video_core]") {
    DrawMergeRuns runs;
    const auto run = [&](u32 length) {
        Full(runs);
        for (u32 i = 1; i < length; ++i) {
            Repeat(runs);
        }
    };
    run(2);
    run(3);
    run(8);
    run(16);
    run(40);
    Full(runs); // ends the last
    const auto& totals = runs.Get();
    REQUIRE(totals.patch_runs == 5);
    REQUIRE(totals.patch_lengths[0] == 1);
    REQUIRE(totals.patch_lengths[1] == 1);
    REQUIRE(totals.patch_lengths[2] == 1);
    REQUIRE(totals.patch_lengths[3] == 1);
    REQUIRE(totals.patch_lengths[4] == 1);
    REQUIRE(totals.patch_draws == 2 + 3 + 8 + 16 + 40);
    // All repeats: the plain runs are the same.
    REQUIRE(totals.plain_runs == 5);
    REQUIRE(totals.plain_draws == totals.patch_draws);
}
