// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <string_view>

#include "common/cpu_features.h"
#include "common/nx2_ab.h"
#include "core/nx2_cpu_stats.h"
#include "core/nx2_wfe_park.h"

namespace Core {

bool WfePark::Supported() {
    static const bool supported = [] {
        const std::string_view method = Common::EventWaitMethod();
        return method != "sleep";
    }();
    return supported;
}

void WfePark::WaitForEvent(u64 pc, const std::atomic<bool>& interrupted, size_t core_index) {
    using Nx2CpuStats::Wfe;
    // A set event register (SEVL before the loop, another core's SEV) is consumed and WFE goes
    // on at once, as on the console.
    if (event_register.exchange(false)) {
        Nx2CpuStats::CountWfe(core_index, Wfe::EventSet);
        return;
    }
    const s64 now = Common::g_wall_clock.GetTimeNS().count();
    if (!tracker.ShouldPark(pc, now) || !Supported() ||
        !Common::Nx2Ab::On(Common::Nx2Ab::WfePark)) {
        Nx2CpuStats::CountWfe(core_index, Wfe::Spin);
        return;
    }
    // Mark the park, then look at what ends it: a waker sets its flag, then looks at the mark
    // (both sequentially consistent), so a SEV or an interrupt is never missed.
    parked.store(true);
    const bool ready = event_register.load() || interrupted.load();
    const bool woken = ready || park_event.WaitFor(std::chrono::nanoseconds{PARK_NS});
    parked.store(false);
    // A SEV during the park is consumed by it.
    event_register.store(false);
    if (ready) {
        Nx2CpuStats::CountWfe(core_index, Wfe::EventSet);
        return;
    }
    const s64 end = Common::g_wall_clock.GetTimeNS().count();
    tracker.ParkEnded(end);
    Nx2CpuStats::CountWfe(core_index, Wfe::Parked);
    Nx2CpuStats::CountWfe(core_index, Wfe::ParkNs, static_cast<u64>(end - now));
    if (woken) {
        Nx2CpuStats::CountWfe(core_index, Wfe::ParkWoken);
    }
}

} // namespace Core
