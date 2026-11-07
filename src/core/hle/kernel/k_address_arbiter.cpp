// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common/logging.h"
#include "common/nx2_ab.h"
#include "core/arm/exclusive_monitor.h"
#include "core/core.h"
#include "core/hle/kernel/k_address_arbiter.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/k_typed_address.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_results.h"
#include "core/memory.h"
#include "core/nx2_cpu_stats.h"

namespace Kernel {

KAddressArbiter::KAddressArbiter(Core::System& system_)
    : system{system_}
{}
KAddressArbiter::~KAddressArbiter() = default;

namespace {

bool ReadFromUser(KernelCore& kernel, s32* out, KProcessAddress address) {
    *out = GetCurrentMemory(kernel).Read32(GetInteger(address));
    return true;
}

bool DecrementIfLessThan(KernelCore& kernel, s32* out, KProcessAddress address, s32 value) {
    auto& monitor = GetCurrentProcess(kernel).GetExclusiveMonitor();
    const auto current_core = kernel.CurrentPhysicalCoreIndex();

    // NOTE: If scheduler lock is not held here, interrupt disable is required.
    // KScopedInterruptDisable di;

    // TODO(bunnei): We should call CanAccessAtomic(..) here.

    s32 current_value{};

    while (true) {
        // Load the value from the address.
        current_value =
            static_cast<s32>(monitor.ExclusiveRead32(current_core, GetInteger(address)));

        // Compare it to the desired one.
        if (current_value < value) {
            // If less than, we want to try to decrement.
            const s32 decrement_value = current_value - 1;

            // Decrement and try to store.
            if (monitor.ExclusiveWrite32(current_core, GetInteger(address),
                                         static_cast<u32>(decrement_value))) {
                break;
            }

            // If we failed to store, try again.
        } else {
            // Otherwise, clear our exclusive hold and finish
            monitor.ClearExclusive(current_core);
            break;
        }
    }

    // We're done.
    *out = current_value;
    return true;
}

bool UpdateIfEqual(KernelCore& kernel, s32* out, KProcessAddress address, s32 value,
                   s32 new_value) {
    auto& monitor = GetCurrentProcess(kernel).GetExclusiveMonitor();
    const auto current_core = kernel.CurrentPhysicalCoreIndex();

    // NOTE: If scheduler lock is not held here, interrupt disable is required.
    // KScopedInterruptDisable di;

    // TODO(bunnei): We should call CanAccessAtomic(..) here.

    s32 current_value{};

    // Load the value from the address.
    while (true) {
        current_value =
            static_cast<s32>(monitor.ExclusiveRead32(current_core, GetInteger(address)));

        // Compare it to the desired one.
        if (current_value == value) {
            // If equal, we want to try to write the new value.

            // Try to store.
            if (monitor.ExclusiveWrite32(current_core, GetInteger(address),
                                         static_cast<u32>(new_value))) {
                break;
            }

            // If we failed to store, try again.
        } else {
            // Otherwise, clear our exclusive hold and finish.
            monitor.ClearExclusive(current_core);
            break;
        }
    }

    // We're done.
    *out = current_value;
    return true;
}

class ThreadQueueImplForKAddressArbiter final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKAddressArbiter(KernelCore& kernel, KAddressArbiter::ThreadTree* t,
                                               std::atomic<u32>* waiter_hint)
        : KThreadQueue(kernel), m_tree(t), m_waiter_hint(waiter_hint) {}

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // If the thread is waiting on an address arbiter, remove it from the tree.
        if (waiting_thread->IsWaitingForAddressArbiter()) {
            m_tree->erase(m_tree->iterator_to(*waiting_thread));
            waiting_thread->ClearAddressArbiter();
            m_waiter_hint->fetch_sub(1, std::memory_order_seq_cst);
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    KAddressArbiter::ThreadTree* m_tree{};
    std::atomic<u32>* m_waiter_hint{};
};

/// EDEN_NX2_KERNEL_WAKE_VERIFY=1: every fast signal (A/B bit KernelWakeFast) also looks for a
/// waiter in the tree under the scheduler lock; one found is a bug, logged and woken.
bool KernelWakeVerify() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_KERNEL_WAKE_VERIFY");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void CountArbiter(KernelCore& kernel, Core::Nx2CpuStats::Arbiter what, u32 amount = 1) {
    Core::Nx2CpuStats::CountArbiter(kernel.CurrentPhysicalCoreIndex(), what, amount);
}

} // namespace

s32 KAddressArbiter::EndWaits(ThreadTree::iterator it, uint64_t addr, s32 count) {
    s32 num_waiters{};
    while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
           (it->GetAddressArbiterKey() == addr)) {
        // End the thread's wait.
        KThread* target_thread = std::addressof(*it);
        target_thread->EndWait(system.Kernel(), ResultSuccess);

        ASSERT(target_thread->IsWaitingForAddressArbiter());
        target_thread->ClearAddressArbiter();

        it = m_tree.erase(it);
        WaiterHint(addr).fetch_sub(1, std::memory_order_seq_cst);
        ++num_waiters;
    }
    return num_waiters;
}

Result KAddressArbiter::Signal(uint64_t addr, s32 count) {
    using Core::Nx2CpuStats::Arbiter;
    auto& kernel = system.Kernel();
    if (Common::Nx2Ab::On(Common::Nx2Ab::KernelWakeFast)) {
        // A wait counts itself in the bucket (a full barrier) before it reads the value under the
        // scheduler lock; the guest changed the value before this call. With this fence between
        // that store and the count read, either the count shows the waiter, or the waiter reads
        // the new value and doesn't sleep. So no waiter in the bucket = nothing to wake.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (WaiterHint(addr).load(std::memory_order_seq_cst) == 0) {
            if (!KernelWakeVerify()) {
                CountArbiter(kernel, Arbiter::SignalFast);
                R_SUCCEED();
            }
            KScopedSchedulerLock sl(kernel);
            const s32 woken = EndWaits(m_tree.nfind_key({addr, -1}), addr, count);
            if (woken != 0) {
                CountArbiter(kernel, Arbiter::VerifyMismatch);
                LOG_WARNING(Kernel, "nx2 arbiter verify: a fast signal of {:#x} found {} waiters",
                            addr, woken);
            }
            CountArbiter(kernel, Arbiter::SignalFast);
            R_SUCCEED();
        }
    }

    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(kernel);
        num_waiters = EndWaits(m_tree.nfind_key({addr, -1}), addr, count);
    }
    CountArbiter(kernel, Arbiter::SignalSlow);
    if (num_waiters == 0) {
        CountArbiter(kernel, Arbiter::SignalNoneWoken);
    }
    CountArbiter(kernel, Arbiter::ThreadsWoken, static_cast<u32>(num_waiters));
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndIncrementIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        // Check the userspace value.
        s32 user_value{};
        R_UNLESS(UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, value + 1),
                 ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        num_waiters = EndWaits(m_tree.nfind_key({addr, -1}), addr, count);
    }
    CountArbiter(system.Kernel(), Core::Nx2CpuStats::Arbiter::SignalSlow);
    CountArbiter(system.Kernel(), Core::Nx2CpuStats::Arbiter::ThreadsWoken,
                 static_cast<u32>(num_waiters));
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndModifyByWaitingCountIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        auto it = m_tree.nfind_key({addr, -1});
        // Determine the updated value.
        s32 new_value{};
        if (count <= 0) {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                new_value = value - 2;
            } else {
                new_value = value + 1;
            }
        } else {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                auto tmp_it = it;
                s32 tmp_num_waiters{};
                while (++tmp_it != m_tree.end() && tmp_it->GetAddressArbiterKey() == addr) {
                    if (tmp_num_waiters++ >= count) {
                        break;
                    }
                }

                if (tmp_num_waiters < count) {
                    new_value = value - 1;
                } else {
                    new_value = value;
                }
            } else {
                new_value = value + 1;
            }
        }

        // Check the userspace value.
        s32 user_value{};
        bool succeeded{};
        if (value != new_value) {
            succeeded = UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, new_value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        R_UNLESS(succeeded, ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        num_waiters = EndWaits(it, addr, count);
    }
    CountArbiter(system.Kernel(), Core::Nx2CpuStats::Arbiter::SignalSlow);
    CountArbiter(system.Kernel(), Core::Nx2CpuStats::Arbiter::ThreadsWoken,
                 static_cast<u32>(num_waiters));
    R_SUCCEED();
}

Result KAddressArbiter::WaitIfLessThan(uint64_t addr, s32 value, bool decrement, s64 timeout) {
    using Core::Nx2CpuStats::Arbiter;
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());

    // A/B bit KernelWakeFast: a value that already isn't less is the same InvalidState the
    // locked check gives (with decrement, DecrementIfLessThan leaves such a value alone too), as
    // if the call had looked a moment earlier. Termination keeps its precedence: slow path.
    if (Common::Nx2Ab::On(Common::Nx2Ab::KernelWakeFast) && !cur_thread->IsTerminationRequested()) {
        s32 user_value{};
        if (ReadFromUser(system.Kernel(), std::addressof(user_value), addr) &&
            user_value >= value) {
            CountArbiter(system.Kernel(), Arbiter::WaitFast);
            R_THROW(ResultInvalidState);
        }
    }
    CountArbiter(system.Kernel(), Arbiter::WaitSlow);

    KHardwareTimer* timer{};
    std::atomic<u32>& waiter_hint = WaiterHint(addr);
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), std::addressof(m_tree),
                                                 std::addressof(waiter_hint));

    {
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Count this thread as a waiter before reading the value (see Signal), until it leaves
        // the tree; the early returns below take it back.
        waiter_hint.fetch_add(1, std::memory_order_seq_cst);

        // Read the value from userspace.
        s32 user_value{};
        bool succeeded{};
        if (decrement) {
            succeeded = DecrementIfLessThan(system.Kernel(), std::addressof(user_value), addr, value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        if (!succeeded) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is less than the specified one.
        if (user_value >= value) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }
    CountArbiter(system.Kernel(), Arbiter::WaitSlept);

    // Get the result.
    return cur_thread->GetWaitResult();
}

Result KAddressArbiter::WaitIfEqual(uint64_t addr, s64 value, s64 timeout, bool is_64_bit) {
    using Core::Nx2CpuStats::Arbiter;
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());

    // A/B bit KernelWakeFast: a value that already differs is the locked check's InvalidState,
    // as if the call had looked a moment earlier (memory errors and termination: slow path).
    if (Common::Nx2Ab::On(Common::Nx2Ab::KernelWakeFast) && !cur_thread->IsTerminationRequested()) {
        auto& memory = GetCurrentMemory(system.Kernel());
        bool readable{};
        s64 user_value{};
        if (is_64_bit) {
            readable = memory.IsValidVirtualAddressRange(addr, sizeof(s64));
            if (readable) {
                user_value = static_cast<s64>(memory.Read64(addr));
            }
        } else {
            s32 user_value32{};
            readable = ReadFromUser(system.Kernel(), std::addressof(user_value32), addr);
            user_value = user_value32;
        }
        if (readable && user_value != value) {
            CountArbiter(system.Kernel(), Arbiter::WaitFast);
            R_THROW(ResultInvalidState);
        }
    }
    CountArbiter(system.Kernel(), Arbiter::WaitSlow);

    KHardwareTimer* timer{};
    std::atomic<u32>& waiter_hint = WaiterHint(addr);
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), std::addressof(m_tree),
                                                 std::addressof(waiter_hint));

    {
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Count this thread as a waiter before reading the value (see Signal), until it leaves
        // the tree; the early returns below take it back.
        waiter_hint.fetch_add(1, std::memory_order_seq_cst);

        // Read the value from userspace.
        s64 user_value{};
        bool succeeded{};
        if (is_64_bit) {
            auto& memory = GetCurrentMemory(system.Kernel());
            succeeded = memory.IsValidVirtualAddressRange(addr, sizeof(s64));
            if (succeeded) {
                user_value = static_cast<s64>(memory.Read64(addr));
            }
        } else {
            s32 user_value32{};
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value32), addr);
            user_value = user_value32;
        }
        if (!succeeded) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is equal.
        if (value != user_value) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            waiter_hint.fetch_sub(1, std::memory_order_seq_cst);
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }
    CountArbiter(system.Kernel(), Arbiter::WaitSlept);

    // Get the result.
    return cur_thread->GetWaitResult();
}

} // namespace Kernel
