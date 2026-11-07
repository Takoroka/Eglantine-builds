// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>

namespace Service::Nx2 {

/// nx2:vi2's wait for a presented frame's GPU fence (WaitForFrame) gives up after 4 s, so a GPU
/// thread stuck on the game can't hang it. A/B bit PresentWaitBusyBack: time in which the split
/// GPU thread's back was working (building pipelines, translating shaders, waiting for the host
/// GPU) doesn't count toward those 4 s, up to 30 s of waiting in all. Giving up while the back
/// still works let the game run 4+ s ahead of it and rewrite command memory the GPU thread had
/// not read yet (11-07: Switch Sports' first-run pipeline builds, ~5 s; issue #50's class).
class PresentDeadline {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr Clock::duration Base = std::chrono::milliseconds{4000};
    static constexpr Clock::duration Cap = std::chrono::seconds{30};

    PresentDeadline(Clock::time_point start, bool count_busy_back) noexcept
        : deadline{start + Base}, cap{start + Cap}, extend{count_busy_back} {}

    /// After waiting from `from` to `to`: `back_busy`, whether the back was working at `to`.
    void AfterSlice(Clock::time_point from, Clock::time_point to, bool back_busy) noexcept {
        if (!extend || !back_busy || to <= from) {
            return;
        }
        const Clock::time_point moved = (std::min)(deadline + (to - from), cap);
        extended += moved - deadline;
        deadline = moved;
    }

    [[nodiscard]] bool Passed(Clock::time_point now) const noexcept {
        return now >= deadline;
    }

    /// How far the deadline moved for the back's work.
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
