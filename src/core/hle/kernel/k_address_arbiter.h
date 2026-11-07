// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/assert.h"
#include "common/common_types.h"
#include "core/hle/kernel/k_condition_variable.h"
#include "core/hle/kernel/svc_types.h"

union Result;

namespace Core {
class System;
}

namespace Kernel {

class KernelCore;

class KAddressArbiter {
public:
    using ThreadTree = KConditionVariable::ThreadTree;

    explicit KAddressArbiter(Core::System& system);
    ~KAddressArbiter();

    Result SignalToAddress(uint64_t addr, Svc::SignalType type, s32 value, s32 count) {
        switch (type) {
        case Svc::SignalType::Signal:
            R_RETURN(this->Signal(addr, count));
        case Svc::SignalType::SignalAndIncrementIfEqual:
            R_RETURN(this->SignalAndIncrementIfEqual(addr, value, count));
        case Svc::SignalType::SignalAndModifyByWaitingCountIfEqual:
            R_RETURN(this->SignalAndModifyByWaitingCountIfEqual(addr, value, count));
        default:
            UNREACHABLE();
        }
    }

    Result WaitForAddress(uint64_t addr, Svc::ArbitrationType type, s64 value, s64 timeout) {
        switch (type) {
        case Svc::ArbitrationType::WaitIfLessThan:
            R_RETURN(WaitIfLessThan(addr, static_cast<s32>(value), false, timeout));
        case Svc::ArbitrationType::DecrementAndWaitIfLessThan:
            R_RETURN(WaitIfLessThan(addr, static_cast<s32>(value), true, timeout));
        case Svc::ArbitrationType::WaitIfEqual:
            R_RETURN(WaitIfEqual(addr, static_cast<s32>(value), timeout, false));
        case Svc::ArbitrationType::WaitIfEqual64:
            R_RETURN(WaitIfEqual(addr, value, timeout, true));
        default:
            UNREACHABLE();
        }
    }

    /// TEMP (A/B bit KernelWakeFast): threads waiting on an address of the bucket, counted from
    /// just before a wait reads the user value until the thread leaves m_tree (kept with the bit
    /// off too, so the bit can flip in a run). Zero: no thread waits on any address of the bucket.
    static constexpr size_t WAITER_BUCKETS = 256;
    static constexpr size_t WaiterBucket(u64 addr) noexcept {
        return static_cast<size_t>(((addr >> 2) * 0x9E3779B97F4A7C15ULL) >> 56);
    }

private:
    Result Signal(uint64_t addr, s32 count);
    Result SignalAndIncrementIfEqual(uint64_t addr, s32 value, s32 count);
    Result SignalAndModifyByWaitingCountIfEqual(uint64_t addr, s32 value, s32 count);
    Result WaitIfLessThan(uint64_t addr, s32 value, bool decrement, s64 timeout);
    Result WaitIfEqual(uint64_t addr, s64 value, s64 timeout, bool is_64_bit);

    /// Ends the waits of up to count threads waiting on addr (all of them with count <= 0),
    /// starting at it; under the scheduler lock. Returns how many it woke.
    s32 EndWaits(ThreadTree::iterator it, uint64_t addr, s32 count);

    std::atomic<u32>& WaiterHint(u64 addr) noexcept {
        return m_waiter_hint[WaiterBucket(addr)];
    }

private:
    ThreadTree m_tree;
    Core::System& system;
    std::array<std::atomic<u32>, WAITER_BUCKETS> m_waiter_hint{};
};

} // namespace Kernel
