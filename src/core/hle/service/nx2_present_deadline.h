// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>

namespace Service::Nx2 {

/// nx2:vi2's wait for a presented frame's GPU fence (WaitForFrame) gives up after 4 s, so a GPU
/// thread stuck on the game can't hang it. A/B bit PresentWaitBusyBack: time in which the GPU
/// thread built pipelines (VideoCommon::GpuBuildClock: translation, compiles, waits for builder
/// threads) doesn't count toward those 4 s, up to 30 s of waiting in all. Giving up while it still
/// builds let the game run 4+ s ahead and rewrite command memory the GPU thread had not read yet
/// (11-07: Switch Sports' first-run pipeline builds, ~5 s; issue #50's class).
///
/// Only build time counts. Counting any time the split's back wasn't idle also held presents
/// while the back waited on work the game does after the present returns: Kirby Air Riders'
/// presents then waited up to 11.6 s and it ran at half its presents (11-07 A/B).
class PresentDeadline {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr Clock::duration Base = std::chrono::milliseconds{4000};
    static constexpr Clock::duration Cap = std::chrono::seconds{30};

    PresentDeadline(Clock::time_point start, bool count_builds) noexcept
        : deadline{start + Base}, cap{start + Cap}, extend{count_builds} {}

    /// After waiting from `from` to `to`, in which the GPU thread built pipelines for `built`:
    /// the deadline moves on by that much (at most the slice), up to the cap.
    void AfterSlice(Clock::time_point from, Clock::time_point to,
                    Clock::duration built) noexcept {
        if (!extend || to <= from || built <= Clock::duration::zero()) {
            return;
        }
        const Clock::duration counted = (std::min)(built, to - from);
        const Clock::time_point moved = (std::min)(deadline + counted, cap);
        extended += moved - deadline;
        deadline = moved;
    }

    [[nodiscard]] bool Passed(Clock::time_point now) const noexcept {
        return now >= deadline;
    }

    /// How far the deadline moved for the GPU thread's builds.
    [[nodiscard]] Clock::duration Extended() const noexcept {
        return extended;
    }

private:
    Clock::time_point deadline;
    Clock::time_point cap;
    Clock::duration extended{};
    bool extend;
};

} // namespace Service::Nx2
