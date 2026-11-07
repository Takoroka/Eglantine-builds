// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"

namespace VideoCommon {

/// Runs of draws one multi-draw could replace (play-dkb8 idea 6, the counts that decide A/B bit
/// 19 DrawMerge): consecutive direct single-instance draws, each differing from the one before
/// only in uniform buffers or the index range. A run starts at any such draw; the next ones join
/// it while they qualify. Two kinds:
/// - patch runs: repeats and uniform-only refreshes. Merging them needs the shaders to read
///   their uniform buffers through a per-draw table (gl_DrawID).
/// - plain runs: all draws after the first are repeats (same buffers). A multi-draw needs no
///   shader change.
class DrawMergeRuns {
public:
    struct Totals {
        u64 draws{};
        u64 patch_runs{};
        u64 patch_draws{};
        u64 plain_runs{};
        u64 plain_draws{};
        std::array<u64, 5> patch_lengths{}; ///< Patch runs of 2, 3-4, 5-8, 9-16, 17+ draws
    };

    /// One draw. `single`: direct and one instance (it may start or join a run). `repeats`: it
    /// repeats the previous draw's setup. `uniforms_only`: only uniform buffers or the index
    /// range changed since the previous draw.
    void Note(bool single, bool repeats, bool uniforms_only) noexcept {
        ++totals.draws;
        const bool joins_patch = single && (repeats || uniforms_only) && patch_length != 0;
        const bool joins_plain = single && repeats && plain_length != 0;
        if (joins_patch) {
            ++patch_length;
        } else {
            EndPatchRun();
            patch_length = single ? 1 : 0;
        }
        if (joins_plain) {
            ++plain_length;
        } else {
            EndPlainRun();
            plain_length = single ? 1 : 0;
        }
    }

    /// The totals of the runs that ended (an open one counts once it ends).
    [[nodiscard]] const Totals& Get() const noexcept {
        return totals;
    }

private:
    static void AddPatchRun(Totals& to, u32 length) noexcept {
        if (length < 2) {
            return;
        }
        ++to.patch_runs;
        to.patch_draws += length;
        const size_t bucket = length <= 2 ? 0 : length <= 4 ? 1 : length <= 8 ? 2 : length <= 16 ? 3 : 4;
        ++to.patch_lengths[bucket];
    }

    static void AddPlainRun(Totals& to, u32 length) noexcept {
        if (length < 2) {
            return;
        }
        ++to.plain_runs;
        to.plain_draws += length;
    }

    void EndPatchRun() noexcept {
        AddPatchRun(totals, patch_length);
        patch_length = 0;
    }

    void EndPlainRun() noexcept {
        AddPlainRun(totals, plain_length);
        plain_length = 0;
    }

    Totals totals{};
    u32 patch_length = 0;
    u32 plain_length = 0;
};

} // namespace VideoCommon
