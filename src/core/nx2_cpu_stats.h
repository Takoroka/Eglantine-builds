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

constexpr size_t NUM_SVC_IDS = 256; ///< SVC numbers above 0xff count as 0xff

/// One guest core's counters, on cache lines of their own.
struct alignas(64) CoreCounters {
    std::array<std::atomic<u32>, NUM_SVC_IDS> svcs{};
    std::array<std::atomic<u32>, static_cast<size_t>(Hint::Count)> hints{};
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

/// Logs the two lines for the counts since the previous call and resets them; window_ns is
/// the time they cover (pauses left out).
void Report(s64 window_ns);

} // namespace Core::Nx2CpuStats
