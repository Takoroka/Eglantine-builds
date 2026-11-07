// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex> // Kernel headers that use std::mutex get it through this one

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/hle/kernel/k_scoped_lock.h"

namespace Kernel {

class KSpinLock {
public:
    explicit KSpinLock() = default;

    YUZU_NON_COPYABLE(KSpinLock);
    YUZU_NON_MOVEABLE(KSpinLock);

    void Lock();
    void Unlock();
    bool TryLock();

private:
    // 0 free, 1 held, 2 held with (possible) sleepers. The scheduler lock is taken by every
    // core thousands of times a frame and held for microseconds: a contended Lock spins briefly
    // before sleeping, instead of putting the host thread to sleep at once and paying its
    // wake-up latency (std::mutex did).
    std::atomic<u32> m_state{0};
};

// A spin lock alone on its cache line: the global scheduler lock is one (every SVC of a job system
// takes it), and the other cores spin on its word while the holder writes the lock count, the
// owner and the priority queue's per-core bitsets, which shared the word's line before.
// Idea: MythosOfs/Orchid 452d0ace50.
class alignas(64) KAlignedSpinLock : public KSpinLock {
public:
    explicit KAlignedSpinLock() = default;
};
static_assert(sizeof(KAlignedSpinLock) == 64);

// TODO(bunnei): Alias for now, in case we want to implement these accurately in the future.
using KNotAlignedSpinLock = KSpinLock;

using KScopedSpinLock = KScopedLock<KSpinLock>;
using KScopedAlignedSpinLock = KScopedLock<KAlignedSpinLock>;
using KScopedNotAlignedSpinLock = KScopedLock<KNotAlignedSpinLock>;

} // namespace Kernel
