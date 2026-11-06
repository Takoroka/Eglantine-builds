// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEMP (measurement aid): A/B of optimizations, for the lichii idea branches on the play build.
// EDEN_NX2_AB=<hex mask> picks the optimizations (bits, see Feature) that take part; they turn on
// and off together every EDEN_NX2_AB_PERIOD_MS (default 4000) and the present path logs the mean
// frame time of each phase. EDEN_NX2_AB_OFF=<hex mask> keeps those optimizations off for the
// whole run (an arm "off" against a run without it). Optimizations outside both masks (and all of
// them without the variables) stay on. An optimization that is off behaves like the code before
// it and drops what it cached (it sees Epoch() change).

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace Common::Nx2Ab {

// The registry of bits in use is C:\dev\nx2work\rounds\AB-BITS.md (bits 32 and up: 1ull << n).
enum Feature : u64 {
    // fences (11-05): direct draws with no vertices or no instances skip the draw setup.
    ZeroDrawSkip = 1ull << 40,
    // vkoff (11-05): draws with none of the dynamic states' dirty flags set skip their updates.
    DynamicStateSkip = 1ull << 41,
    // hosttime (11-06): writes of values only dynamic states read leave the next draw the repeat
    // path (Maxwell3D::SameStateDynamic).
    DynamicDeltas = 1ull << 44,
    // hosttime (11-06): vertex streams bind a range resolved before without resolving it again
    // while the mappings and buffers are as then (BufferCache::vertex_lookups).
    VertexLookupCache = 1ull << 45,
    ListDrawRanges = 1ull << 48,     ///< Lists on pages draws write: past the draws' ranges (nx2.h)
    ListWriterWaitSkip = 1ull << 49, ///< Known CPU list pages skip the wait for writers (dma_pusher)
    SemAcquireOnBack = 1ull << 50,   ///< SEM_EXECUTE acquires checked on the back, no drain (puller)
    ListFlushBatch = 1ull << 51,     ///< List drains download all draw-written list pages at once
};

inline const u64 g_mask = [] {
    const char* value = std::getenv("EDEN_NX2_AB");
    return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 16)) : u64{0};
}();

inline const u64 g_forced_off = [] {
    const char* value = std::getenv("EDEN_NX2_AB_OFF");
    return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 16)) : u64{0};
}();

inline const u32 g_period_ms = [] {
    const char* value = std::getenv("EDEN_NX2_AB_PERIOD_MS");
    return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 4000u;
}();

/// Phase: true while the optimizations in the mask are off.
inline std::atomic<bool> g_off{false};
/// Bumped at every phase change.
inline std::atomic<u32> g_epoch{0};

inline bool On(Feature feature) {
    return (g_forced_off & feature) == 0 &&
           ((g_mask & feature) == 0 || !g_off.load(std::memory_order_relaxed));
}

inline u32 Epoch() {
    return g_epoch.load(std::memory_order_relaxed);
}

/// Called by the present path once per presented frame with that frame's time.
inline void OnPresent(long long frame_us) {
    if (g_mask == 0) {
        return;
    }
    struct Bucket {
        long long us{};
        long long frames{};
    };
    static Bucket buckets[2]{};
    static Bucket totals[2]{};
    static u32 since_flip{};
    static auto start = std::chrono::steady_clock::now();
    static auto last_log = start;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    const bool off = ((ms / g_period_ms) & 1) != 0;
    const bool was_off = g_off.load(std::memory_order_relaxed);
    if (off != was_off) {
        g_off.store(off, std::memory_order_relaxed);
        g_epoch.fetch_add(1, std::memory_order_relaxed);
        since_flip = 0;
        return;
    }
    // The first frames after a flip were prepared partly in the other phase; hitches are loads.
    if (++since_flip <= 6 || frame_us > 100'000 || frame_us <= 0) {
        return;
    }
    buckets[was_off ? 1 : 0].us += frame_us;
    ++buckets[was_off ? 1 : 0].frames;
    if (now - last_log < std::chrono::seconds{30}) {
        return;
    }
    last_log = now;
    const auto mean = [](const Bucket& b) {
        return b.frames != 0 ? static_cast<double>(b.us) / static_cast<double>(b.frames) : 0.0;
    };
    for (int i = 0; i < 2; ++i) {
        totals[i].us += buckets[i].us;
        totals[i].frames += buckets[i].frames;
    }
    LOG_INFO(Service,
             "nx2 A/B mask {:#x}: last 30 s on {:.0f} us ({} frames), off {:.0f} us ({} frames), "
             "on/off {:.3f} | total on {:.0f} us ({} frames), off {:.0f} us ({} frames), on/off {:.3f}",
             g_mask, mean(buckets[0]), buckets[0].frames, mean(buckets[1]), buckets[1].frames,
             mean(buckets[1]) != 0.0 ? mean(buckets[0]) / mean(buckets[1]) : 0.0,
             mean(totals[0]), totals[0].frames, mean(totals[1]), totals[1].frames,
             mean(totals[1]) != 0.0 ? mean(totals[0]) / mean(totals[1]) : 0.0);
    buckets[0] = {};
    buckets[1] = {};
}

} // namespace Common::Nx2Ab
