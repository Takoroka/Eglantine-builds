// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "common/logging.h"

// Counters of the native ARMv8.1 LSE atomics (UserConfig::native_lse_atomics): operations done
// with one host atomic in JIT code (counted per JIT in A64JitState::lse_native_ops) and on the
// slow path (the memory callbacks). One "LSE atomics" log line per 30 s at most.
namespace Dynarmic::Backend::X64::LseStats {

inline std::atomic<std::uint64_t> g_native{0};
inline std::atomic<std::uint64_t> g_fallback{0};

inline void CountFallback() {
    g_fallback.fetch_add(1, std::memory_order_relaxed);
}

/// Adds a JIT's native operations since its previous call (`flushed` remembers them) and writes
/// the log line when 30 s have passed since the previous one.
inline void Flush(std::uint64_t native_total, std::uint64_t& flushed) {
    if (native_total != flushed) {
        g_native.fetch_add(native_total - flushed, std::memory_order_relaxed);
        flushed = native_total;
    }

    static std::atomic<std::int64_t> last_line_ms{0};
    static std::atomic<std::uint64_t> native_at_last_line{0};
    static std::atomic<std::uint64_t> fallback_at_last_line{0};
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
    const std::uint64_t native = g_native.load(std::memory_order_relaxed);
    const std::uint64_t fallback = g_fallback.load(std::memory_order_relaxed);
    const std::uint64_t native_delta =
        native - native_at_last_line.exchange(native, std::memory_order_relaxed);
    const std::uint64_t fallback_delta =
        fallback - fallback_at_last_line.exchange(fallback, std::memory_order_relaxed);
    if (native_delta == 0 && fallback_delta == 0) {
        return;
    }
    const double seconds = static_cast<double>(now_ms - last_ms) / 1000.0;
    LOG_INFO(Core_ARM,
             "LSE atomics (A/B bit 15): native {:.0f}/s, fallback {:.0f}/s (pages that fault in "
             "fastmem, misaligned, no fastmem); totals native {} fallback {}",
             static_cast<double>(native_delta) / seconds,
             static_cast<double>(fallback_delta) / seconds, native, fallback);
}

} // namespace Dynarmic::Backend::X64::LseStats
