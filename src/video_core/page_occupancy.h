// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>

#include "common/common_types.h"

namespace VideoCommon {

/// Invalidations skip caches whose PageOccupancy says a range is empty, and a range inside one
/// mapped big page skips the page walk (MemoryManager::GetSubmappedRangeImpl).
/// EDEN_NX2_INVAL_FILTER=0 turns both off (comparisons); flipping it at run time is safe.
inline std::atomic<bool> g_invalidation_filter{[] {
    const char* value = std::getenv("EDEN_NX2_INVAL_FILTER");
    return value == nullptr || value[0] != '0';
}()};

[[nodiscard]] inline bool InvalidationFilterOn() noexcept {
    return g_invalidation_filter.load(std::memory_order_relaxed);
}

/// Counts a cache's entries per page where any thread can read the counts without the cache's
/// lock: a range whose pages all count zero holds none of its entries, and the cache can skip an
/// invalidation of it. Pages share counters (page modulo NumCounters), which only makes a range
/// look occupied when it is not.
template <u32 PageBits, size_t NumCounters = 4096>
class PageOccupancy {
public:
    void Add(u64 page) noexcept {
        counters[page % NumCounters].fetch_add(1, std::memory_order_release);
    }

    void Remove(u64 page, s32 count = 1) noexcept {
        counters[page % NumCounters].fetch_sub(count, std::memory_order_release);
    }

    [[nodiscard]] bool MayHave(u64 addr, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 first = addr >> PageBits;
        const u64 last = (addr + size - 1) >> PageBits;
        if (last - first >= NumCounters) {
            return true;
        }
        for (u64 page = first; page <= last; ++page) {
            if (counters[page % NumCounters].load(std::memory_order_acquire) != 0) {
                return true;
            }
        }
        return false;
    }

private:
    std::array<std::atomic<s32>, NumCounters> counters{};
};

} // namespace VideoCommon
