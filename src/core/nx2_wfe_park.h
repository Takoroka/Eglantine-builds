// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A/B bit WfePark (play-dkb8 idea 27 step 2): a guest core spinning on WFE parks briefly instead
// of running the spin at full speed. On the console WFE sleeps until another core's SEV, an
// interrupt, or a store that clears the core's exclusive monitor. The last one (how guest spin
// locks are usually released) can't be seen here, so a park is short (PARK_NS) and only starts
// once the spin has gone on for SPIN_NS: short waits keep today's behaviour, long ones re-check
// at least every PARK_NS while the host core rests.

#include <atomic>

#include "common/common_types.h"
#include "common/thread.h"

namespace Core {

/// Which WFEs belong to one spin, and when a spin has gone on long enough to park.
/// Pure, for the guest core's own thread.
class WfeSpinTracker {
public:
    static constexpr s64 STREAK_GAP_NS = 10'000; ///< WFEs at one PC closer than this: one spin
    static constexpr s64 SPIN_NS = 100'000;      ///< A spin parks once it has gone on this long

    /// A WFE at pc at time now: whether it should park.
    bool ShouldPark(u64 pc, s64 now) noexcept {
        if (pc != last_pc || now - last_ns > STREAK_GAP_NS) {
            streak_start = now;
        }
        last_pc = pc;
        last_ns = now;
        return now - streak_start >= SPIN_NS;
    }

    /// The park of the WFE at pc ended at now (the spin goes on from there).
    void ParkEnded(s64 now) noexcept {
        last_ns = now;
    }

private:
    u64 last_pc{~u64{0}};
    s64 last_ns{};
    s64 streak_start{};
};

/// One guest core's event register and park.
class WfePark {
public:
    static constexpr s64 PARK_NS = 20'000;

    /// WFE on this core (the core's host thread, inside the JIT). interrupted: the core's
    /// pending interrupt flag, checked after the park is marked. Returns when the guest may go on.
    void WaitForEvent(u64 pc, const std::atomic<bool>& interrupted, size_t core_index);

    /// SEVL on this core: the next WFE returns at once.
    void SendEventLocal() noexcept {
        event_register.store(true);
    }

    /// SEV from any core, or an interrupt for this one: sets the event register (SEV only) and
    /// ends a park.
    void SendEvent() {
        event_register.store(true);
        Wake();
    }
    void Wake() {
        if (parked.load()) {
            park_event.Set();
        }
    }

    /// Whether this host can park precisely (MWAITX/UMWAIT or a condition variable; not the
    /// timer-tick sleep, whose 0.5 ms steps are too coarse).
    static bool Supported();

private:
    std::atomic<bool> event_register{false};
    std::atomic<bool> parked{false};
    Common::Event park_event;
    WfeSpinTracker tracker; ///< The core's own thread only
};

} // namespace Core
