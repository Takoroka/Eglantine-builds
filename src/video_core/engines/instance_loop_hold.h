// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>

#include "common/common_types.h"

namespace Tegra::Engines {

/// A/B bit DrawLoopOneDraw. NVN2's draw-loop macros (0xe50, 0x1e7) draw one indexed draw of N
/// instances as Begin(First)/End and N - 1 Begin(Subsequent)/End pairs. The draw manager draws
/// the first instance at its End, then all N together when the loop ends (DrawDeferred: "the
/// first draw of a batch also covers the loop's first instance"), so instance 0 was drawn twice.
/// When the macro's parameters say N >= 2, the call expects the loop (Expect), the first End holds
/// its instance (HoldFirst) and the batch covers it (OnBatch); a loop that turns out to have one
/// instance draws the held one as its End would have (OnFlushWithoutBatch).
class InstanceLoopHold {
public:
    /// Before the macro runs.
    void Expect(u32 instances) noexcept {
        expected = instances;
    }

    /// After the macro: what is still held was drawn by its end (DrawDeferred) already.
    void Forget() noexcept {
        expected = 0;
    }

    /// DrawEnd in General mode, not forced: true when it holds the instance instead of drawing it.
    /// `drawn_instances`: the instances the current loop's batches drew (0 for its first End).
    [[nodiscard]] bool HoldFirst(u32 drawn_instances) noexcept {
        if (expected < 2 || drawn_instances != 0 || held) {
            return false;
        }
        expected = 0;
        held = true;
        return true;
    }

    /// Whether an instance waits to be drawn (part of HasDeferredDraw).
    [[nodiscard]] bool Held() const noexcept {
        return held;
    }

    /// DrawDeferred with instances waiting: their batch covers the held one. True when one was
    /// held (a draw saved).
    bool OnBatch() noexcept {
        const bool was_held = held;
        held = false;
        return was_held;
    }

    /// DrawDeferred with no instance after the first: true when the held one must be drawn now,
    /// as its End would have drawn it.
    bool OnFlushWithoutBatch() noexcept {
        const bool was_held = held;
        held = false;
        return was_held;
    }

private:
    u32 expected{};
    bool held{};
};

/// A draw as ProcessDraw issues it (EDEN_NX2_DRAW_LOOP_VERIFY's record).
struct RecordedDraw {
    bool indexed;
    u32 topology;
    u32 first_index;
    u32 index_count;
    u32 base_vertex;
    u32 base_instance;
    u32 instances;
};

/// EDEN_NX2_DRAW_LOOP_VERIFY: whether the draws the macro path made for a call with `parameters`
/// (topology, index count, instances, first index, base vertex, base instance) are what the
/// one-draw path keeps (its last draw, all instances) plus the first instance on its own.
[[nodiscard]] inline bool MacroDrawsMatchOneDraw(std::span<const RecordedDraw> draws,
                                                 std::span<const u32, 6> parameters) noexcept {
    if (draws.size() != 2) {
        return false;
    }
    const auto same_draw = [&](const RecordedDraw& draw) {
        // The begin method's topology field is its low 16 bits.
        return draw.indexed && draw.topology == (parameters[0] & 0xffffU) &&
               draw.index_count == parameters[1] &&
               draw.first_index == parameters[3] && draw.base_vertex == parameters[4] &&
               draw.base_instance == parameters[5];
    };
    return same_draw(draws[0]) && draws[0].instances == 1 && same_draw(draws[1]) &&
           draws[1].instances == parameters[2];
}

} // namespace Tegra::Engines
