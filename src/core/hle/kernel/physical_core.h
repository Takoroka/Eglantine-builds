// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>

#include "core/arm/arm_interface.h"
#include "core/nx2_wfe_park.h"

namespace Kernel {
class KernelCore;
} // namespace Kernel

namespace Core {
class ExclusiveMonitor;
class System;
} // namespace Core

namespace Kernel {

class PhysicalCore {
public:
    PhysicalCore(KernelCore& kernel, std::size_t core_index);
    ~PhysicalCore();

    YUZU_NON_COPYABLE(PhysicalCore);
    YUZU_NON_MOVEABLE(PhysicalCore);

    // Execute guest code running on the given thread.
    void RunThread(KernelCore& kernel, KThread* thread);

    // Copy context from thread to current core.
    void LoadContext(const KThread* thread);
    void LoadSvcArguments(const KProcess& process, std::span<const uint64_t, 8> args);

    // Copy context from current core to thread.
    void SaveContext(KThread* thread) const;
    void SaveSvcArguments(KProcess& process, std::span<uint64_t, 8> args) const;

    // Copy floating point status registers to the target thread.
    void CloneFpuStatus(KThread* dst) const;

    // Log backtrace of current processor state.
    void LogBacktrace(KernelCore& kernel);

    // Wait for an interrupt.
    void Idle(KernelCore& kernel);

    // Interrupt this core.
    void Interrupt();

    // Clear this core's interrupt.
    void ClearInterrupt();

    // Check if this core is interrupted.
    bool IsInterrupted() const;

    std::size_t CoreIndex() const {
        return m_core_index;
    }

    /// WFE on this core (its host thread, inside the JIT): A/B bit WfePark.
    void WaitForEvent(u64 pc) {
        m_wfe.WaitForEvent(pc, m_is_interrupted, m_core_index);
    }
    /// SEVL on this core.
    void SendEventLocal() noexcept {
        m_wfe.SendEventLocal();
    }
    /// SEV from any core: sets this core's event register and ends its park.
    void SendEvent() {
        m_wfe.SendEvent();
    }

private:
    const std::size_t m_core_index;
    std::mutex m_guard;
    std::condition_variable m_on_interrupt;
    Core::ArmInterface* m_arm_interface{};
    KThread* m_current_thread{};
    std::atomic<bool> m_is_interrupted{};
    bool m_is_single_core{};
    Core::WfePark m_wfe;
};

} // namespace Kernel
