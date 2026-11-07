// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEMP (measurement aid): what the guest cores ask of the kernel and how often they spin.
// Per guest core: SVC ids, and the hint instructions (WFE, WFI, YIELD, SEV, SEVL) the JIT
// reports. Relaxed counters; the host timing thread logs and resets them every 30 s
// ("nx2 SVC mix" and "nx2 CPU hints" lines, Report).

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"
#include "core/hardware_properties.h"

namespace Core::Nx2CpuStats {

/// Hint instructions, in the order the "nx2 CPU hints" line prints them.
enum class Hint : u32 {
    WaitForEvent,
    WaitForInterrupt,
    Yield,
    SendEvent,
    SendEventLocal,
    Count,
};

/// The address arbiter's paths (A/B bit KernelWakeFast), for the "nx2 arbiter" line.
enum class Arbiter : u32 {
    SignalFast,       ///< Signal with no waiter in the address's bucket: no scheduler lock
    SignalSlow,       ///< Signal under the scheduler lock (waiters in the bucket, other types, bit off)
    SignalNoneWoken,  ///< ... of which woke nobody (a bucket neighbour, or a waiter just left)
    ThreadsWoken,     ///< Threads the signals woke
    WaitFast,         ///< Wait whose value already didn't match: no scheduler lock
    WaitSlow,         ///< Wait under the scheduler lock
    WaitSlept,        ///< ... of which slept
    VerifyMismatch,   ///< EDEN_NX2_KERNEL_WAKE_VERIFY: a fast signal found a waiter (a bug)
    Count,
};

/// WFE parking (A/B bit WfePark), for the "nx2 CPU hints" line.
enum class Wfe : u32 {
    EventSet,   ///< WFE found the event register set (SEV/SEVL): returned at once, as on the console
    Spin,       ///< WFE early in a spin: returned at once, as before
    Parked,     ///< WFE late in a long spin: parked
    ParkWoken,  ///< ... parks a SEV or an interrupt ended
    ParkNs,     ///< Time parked (ns, summed)
    Count,
};

constexpr size_t NUM_SVC_IDS = 256; ///< SVC numbers above 0xff count as 0xff

/// One guest core's counters, on cache lines of their own.
struct alignas(64) CoreCounters {
    std::array<std::atomic<u32>, NUM_SVC_IDS> svcs{};
    std::array<std::atomic<u32>, static_cast<size_t>(Hint::Count)> hints{};
    std::array<std::atomic<u32>, static_cast<size_t>(Arbiter::Count)> arbiter{};
    std::array<std::atomic<u64>, static_cast<size_t>(Wfe::Count)> wfe{};
};

inline std::array<CoreCounters, Hardware::NUM_CPU_CORES> g_cores{};

inline void CountSvc(size_t core, u32 svc) {
    if (core < g_cores.size()) {
        g_cores[core].svcs[svc < NUM_SVC_IDS ? svc : NUM_SVC_IDS - 1].fetch_add(
            1, std::memory_order_relaxed);
    }
}

inline void CountHint(size_t core, Hint hint) {
    if (core < g_cores.size()) {
        g_cores[core].hints[static_cast<size_t>(hint)].fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountArbiter(size_t core, Arbiter what, u32 amount = 1) {
    if (core < g_cores.size()) {
        g_cores[core].arbiter[static_cast<size_t>(what)].fetch_add(amount,
                                                                   std::memory_order_relaxed);
    }
}

inline void CountWfe(size_t core, Wfe what, u64 amount = 1) {
    if (core < g_cores.size()) {
        g_cores[core].wfe[static_cast<size_t>(what)].fetch_add(amount, std::memory_order_relaxed);
    }
}

/// Logs the lines for the counts since the previous call and resets them; window_ns is the time
/// they cover (pauses left out).
void Report(s64 window_ns);

} // namespace Core::Nx2CpuStats
