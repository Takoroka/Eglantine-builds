// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

// Macro DMA reads of report memory (occlusion predicates): below High GPU accuracy a counter
// report reaches guest memory only once a later fence has popped, so a read before that returns
// an older count (Ocarina of Time's white flashes and missing vegetation). Idea: zortos293/lichii
// 7ae3e7401.
namespace VideoCommon::QueryReads {

/// Writes the reports in [address, address + size) to guest memory, waiting for those still
/// waiting on host results (RasterizerInterface::FlushQueryResults). Runs where the rasterizer's
/// caches are owned: the back of a split GPU thread.
inline void Flush(const Tegra::MemoryManager& memory, VideoCore::RasterizerInterface& rasterizer,
                  GPUVAddr address, size_t size) {
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = address + size;
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = (std::min)((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            rasterizer.FlushQueryResults(*device, page_end - begin);
        }
        begin = page_end;
    }
}

struct Stats {
    std::atomic<u64> waits{};
    std::atomic<u64> timeouts{};
    std::atomic<u64> wait_us{};
    std::atomic<u64> ahead_fallbacks{};
    std::atomic<s64> last_log_ns{};
};
inline Stats g_stats;

/// "nx2 macro query waits": the counts at the first event and then at most every 30 s (only
/// while there are events).
inline void MaybeLog() {
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = g_stats.last_log_ns.load(std::memory_order_relaxed);
    if ((last != 0 && now - last < 30'000'000'000) ||
        !g_stats.last_log_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 waits = g_stats.waits.exchange(0, std::memory_order_relaxed);
    const u64 wait_us = g_stats.wait_us.exchange(0, std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "nx2 macro query waits since the last line: {} ({} us total, {} past 100 ms), reads "
             "ahead sent to the drain on report pages: {}",
             waits, wait_us, g_stats.timeouts.exchange(0, std::memory_order_relaxed),
             g_stats.ahead_fallbacks.exchange(0, std::memory_order_relaxed));
}

/// A read waited for pending reports for `us`; `written`: they landed within the limit.
inline void CountWait(bool written, u64 us) {
    g_stats.waits.fetch_add(1, std::memory_order_relaxed);
    g_stats.wait_us.fetch_add(us, std::memory_order_relaxed);
    if (!written) {
        g_stats.timeouts.fetch_add(1, std::memory_order_relaxed);
    }
    MaybeLog();
}

/// A read the front could have taken ahead of the back drained it: it touches report pages.
inline void CountAheadFallback() {
    g_stats.ahead_fallbacks.fetch_add(1, std::memory_order_relaxed);
    MaybeLog();
}

} // namespace VideoCommon::QueryReads
