// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "common/common_types.h"

namespace VideoCommon {

/// Time the GPU thread has spent building pipelines: translating and compiling them, or waiting
/// for a builder thread to finish one a draw or dispatch needs. nx2:vi2's present wait doesn't
/// count it toward its 4 s deadline (A/B bit PresentWaitBusyBack). Only builds count: the split's
/// back can also be busy waiting on work the game does after the present returns, and holding the
/// present for that stalled Kirby Air Riders for seconds (11-07).
///
/// Written by the one thread that builds for the GPU (GpuBuildScope), read by any.
class GpuBuildClock {
public:
    using Clock = std::chrono::steady_clock;

    [[nodiscard]] static GpuBuildClock& Get() noexcept {
        static GpuBuildClock clock;
        return clock;
    }

    /// All the build time so far, with the part of a build still running.
    [[nodiscard]] Clock::duration Total(Clock::time_point now) const noexcept {
        // A build ending between the two loads is missed for that read only: it is in the next.
        const Clock::rep begun{begin.load(std::memory_order_acquire)};
        const Clock::rep done{total.load(std::memory_order_acquire)};
        const Clock::rep running{begun != 0 ? (std::max)(Clock::rep{0}, Ticks(now) - begun)
                                            : Clock::rep{0}};
        return Clock::duration{done + running};
    }

    /// Nested builds count once, from the outer one's beginning to its end.
    void Begin(Clock::time_point now) noexcept {
        if (depth++ == 0) {
            begin.store((std::max)(Ticks(now), Clock::rep{1}), std::memory_order_release);
        }
    }
    void End(Clock::time_point now) noexcept {
        if (depth == 0 || --depth != 0) {
            return;
        }
        const Clock::rep begun{begin.exchange(0, std::memory_order_acq_rel)};
        if (begun != 0) {
            total.fetch_add((std::max)(Clock::rep{0}, Ticks(now) - begun),
                            std::memory_order_acq_rel);
        }
    }

private:
    [[nodiscard]] static Clock::rep Ticks(Clock::time_point time) noexcept {
        return time.time_since_epoch().count();
    }

    std::atomic<Clock::rep> begin{0}; ///< Start of the running outer build, 0 when none
    std::atomic<Clock::rep> total{0}; ///< Finished builds
    u32 depth{};                      ///< Touched only by the building thread
};

/// A GPU-thread build for GpuBuildClock.
class GpuBuildScope {
public:
    GpuBuildScope() noexcept {
        GpuBuildClock::Get().Begin(GpuBuildClock::Clock::now());
    }
    ~GpuBuildScope() {
        GpuBuildClock::Get().End(GpuBuildClock::Clock::now());
    }
    GpuBuildScope(const GpuBuildScope&) = delete;
    GpuBuildScope& operator=(const GpuBuildScope&) = delete;
};

} // namespace VideoCommon
