// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "common/logging.h"

// Counters of the native ARMv8.1 LSE atomics (UserConfig::native_lse_atomics): operations done
// with one host atomic in JIT code (counted per JIT in A64JitState::lse_native_ops) and on the
// slow path (the memory callbacks); and of the exclusive stores on the lockless monitor path
// (UserConfig::lockless_exclusive_monitor, A64JitState::exclusive_stores/_failures). One
// "LSE atomics" log line per 30 s at most.
namespace Dynarmic::Backend::X64::LseStats {

inline std::atomic<std::uint64_t> g_native{0};
inline std::atomic<std::uint64_t> g_fallback{0};
inline std::atomic<std::uint64_t> g_exclusive_stores{0};
inline std::atomic<std::uint64_t> g_exclusive_store_failures{0};

inline void CountFallback() {
    g_fallback.fetch_add(1, std::memory_order_relaxed);
}

/// A JIT's counter totals (its jit state's fields).
struct Totals {
    std::uint64_t native = 0;
    std::uint64_t exclusive_stores = 0;
    std::uint64_t exclusive_store_failures = 0;
};

/// Adds a JIT's counts since its previous call (`flushed` remembers them) and writes the log
/// line when 30 s have passed since the previous one.
inline void Flush(const Totals& totals, Totals& flushed) {
    const auto add = [](std::atomic<std::uint64_t>& global, std::uint64_t total, std::uint64_t& done) {
        if (total != done) {
            global.fetch_add(total - done, std::memory_order_relaxed);
            done = total;
        }
    };
    add(g_native, totals.native, flushed.native);
    add(g_exclusive_stores, totals.exclusive_stores, flushed.exclusive_stores);
    add(g_exclusive_store_failures, totals.exclusive_store_failures, flushed.exclusive_store_failures);

    static std::atomic<std::int64_t> last_line_ms{0};
    static std::atomic<std::uint64_t> native_at_last_line{0};
    static std::atomic<std::uint64_t> fallback_at_last_line{0};
    static std::atomic<std::uint64_t> stores_at_last_line{0};
    static std::atomic<std::uint64_t> failures_at_last_line{0};
    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
    std::int64_t last_ms = last_line_ms.load(std::memory_order_relaxed);
    if (last_ms == 0) {
        // The first call starts the first window.
        last_line_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed);
        return;
    }
    if (now_ms - last_ms < 30'000 ||
        !last_line_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        return;
    }
    const auto delta = [](const std::atomic<std::uint64_t>& global, std::atomic<std::uint64_t>& at_last) {
        const std::uint64_t now = global.load(std::memory_order_relaxed);
        return now - at_last.exchange(now, std::memory_order_relaxed);
    };
    const std::uint64_t native = delta(g_native, native_at_last_line);
    const std::uint64_t fallback = delta(g_fallback, fallback_at_last_line);
    const std::uint64_t stores = delta(g_exclusive_stores, stores_at_last_line);
    const std::uint64_t failures = delta(g_exclusive_store_failures, failures_at_last_line);
    if (native == 0 && fallback == 0 && stores == 0) {
        return;
    }
    const double seconds = static_cast<double>(now_ms - last_ms) / 1000.0;
    const auto per_second = [seconds](std::uint64_t count) { return static_cast<double>(count) / seconds; };
    LOG_INFO(Core_ARM,
             "LSE atomics (A/B bit 15): native {:.0f}/s, fallback {:.0f}/s (pages that fault in "
             "fastmem, misaligned, no fastmem) | lockless exclusive stores (A/B bit 25) {:.0f}/s, "
             "failed {:.0f}/s",
             per_second(native), per_second(fallback), per_second(stores), per_second(failures));
}

} // namespace Dynarmic::Backend::X64::LseStats
