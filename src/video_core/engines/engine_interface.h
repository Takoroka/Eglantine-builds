// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <limits>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {

enum class EngineTypes : u32 {
    Nv01Timer,
    KeplerCompute,
    Maxwell3D,
    Fermi2D,
    MaxwellDMA,
    KeplerMemory,
};

class EngineInterface {
public:
    virtual ~EngineInterface() = default;

    /// Write the value to the register identified by method.
    virtual void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) = 0;

    /// Write multiple values to the register identified by method.
    virtual void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) = 0;

    void ConsumeSink(Core::System& system) {
        if (!method_sink.empty()) {
            ConsumeSinkImpl(system);
        }
    }

    /// Split GPU thread (gpu_split.h): consecutive state writes the front collected, each a
    /// method | value << 32 with the value the register holds after it, the dirty flags they
    /// set together (256 bits) and the engine's own tags for them (GpuSplit::StateWrite). By
    /// default they run as the methods they are.
    virtual void ApplyStateRun(Core::System& system, [[maybe_unused]] const u64* flags,
                               const u64* writes, u32 count, [[maybe_unused]] u8 tags) {
        ConsumeSink(system);
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, static_cast<u32>(writes[i] & 0xffff), static_cast<u32>(writes[i] >> 32),
                       true);
        }
    }
    /// The same for consecutive words of a data method (GpuSplit::DataWrite), no flags.
    virtual void ApplyDataRun(Core::System& system, const u64* writes, u32 count) {
        EngineInterface::ApplyStateRun(system, nullptr, writes, count, 0);
    }

    std::bitset<(std::numeric_limits<u16>::max)()> execution_mask{};
    std::vector<std::pair<u32, u32>> method_sink{};
    GPUVAddr current_dma_segment;
    bool current_dirty{};
protected:
    virtual void ConsumeSinkImpl(Core::System& system) {
        for (auto [method, value] : method_sink) {
            CallMethod(system, method, value, true);
        }
        method_sink.clear();
    }
};

} // namespace Tegra::Engines
