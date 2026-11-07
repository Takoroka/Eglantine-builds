// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nx2_pending_wake.h"

#if defined(_WIN32) && defined(ARCHITECTURE_x86_64)
#include "common/cpu_features.h"
#include "common/event_wait.h"
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "common/x64/rdtsc.h"
#endif

namespace VideoCommon {

#ifdef _WIN32
#ifdef ARCHITECTURE_x86_64

namespace {

std::chrono::nanoseconds Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
}

// The monitored waits of Common::Event::WaitFor (common/thread.cpp), on a counter instead of a
// flag: a write to the monitored line, the timer, or an interrupt ends one; the loop checks again.
#ifdef __clang__
__attribute__((target("mwaitx")))
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("mwaitx")
#endif
bool WaitMonitorx(const std::atomic<u32>& epoch, u32 seen, std::chrono::nanoseconds deadline,
                  u64 ticks_per_second) {
    for (;;) {
        // clang's _mm_monitorx takes a non-const pointer; the monitor only reads.
        _mm_monitorx(const_cast<std::atomic<u32>*>(&epoch), 0, 0);
        if (epoch.load(std::memory_order_acquire) != seen) {
            return true;
        }
        const u32 cycles = Common::MwaitxTimeoutCycles(deadline - Now(), ticks_per_second);
        if (cycles == 0) {
            return false;
        }
        // EAX bit 1: the timer in EBX (TSC clocks) ends the wait. Hint 0: C1.
        _mm_mwaitx(1U << 1, 0U, cycles);
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

#ifdef __clang__
__attribute__((target("waitpkg")))
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("waitpkg")
#endif
bool WaitUmwait(const std::atomic<u32>& epoch, u32 seen, std::chrono::nanoseconds deadline,
                u64 ticks_per_second) {
    for (;;) {
        _umonitor(const_cast<std::atomic<u32>*>(&epoch));
        if (epoch.load(std::memory_order_acquire) != seen) {
            return true;
        }
        const u32 cycles = Common::MwaitxTimeoutCycles(deadline - Now(), ticks_per_second);
        if (cycles == 0) {
            return false;
        }
        // UMWAIT also ends at the OS limit (IA32_UMWAIT_CONTROL): the loop waits again. State 0:
        // C0.2. The deadline is absolute TSC.
        _umwait(0, Common::X64::FencedRDTSC() + cycles);
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

} // Anonymous namespace

bool PendingWake::Available() {
    return Common::g_cpu_caps.monitorx || Common::g_cpu_caps.waitpkg;
}

bool PendingWake::Wait(u32 seen, std::chrono::nanoseconds cap) {
    if (epoch.load(std::memory_order_acquire) != seen) {
        return true;
    }
    static const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
    const auto deadline = Common::EventWaitDeadline(Now(), cap);
    if (Common::g_cpu_caps.monitorx) {
        return WaitMonitorx(epoch, seen, deadline, ticks_per_second);
    }
    if (Common::g_cpu_caps.waitpkg) {
        return WaitUmwait(epoch, seen, deadline, ticks_per_second);
    }
    return false;
}

#else // ARCHITECTURE_x86_64

bool PendingWake::Available() {
    return false;
}

bool PendingWake::Wait(u32 seen, std::chrono::nanoseconds) {
    return epoch.load(std::memory_order_acquire) != seen;
}

#endif // ARCHITECTURE_x86_64
#else  // _WIN32

bool PendingWake::Available() {
    return true;
}

bool PendingWake::Wait(u32 seen, std::chrono::nanoseconds cap) {
    std::unique_lock lk{mutex};
    return cv.wait_for(lk, cap, [&] { return epoch.load(std::memory_order_acquire) != seen; });
}

#endif // _WIN32

} // namespace VideoCommon
