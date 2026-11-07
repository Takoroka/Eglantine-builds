// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "common/logging.h"

namespace Dynarmic::Backend::X64 {

/// Full code-cache flushes of all JIT instances: a JIT whose code cache ran out of space drops all
/// of its compiled code, and every block runs through the compiler again (a hitch, then a slow
/// stretch). Before this counter nothing said that it happened.
inline std::atomic<std::uint64_t> g_full_cache_flushes{0};

/// Counts a full flush of a JIT's `cache_size`-byte code cache that ran out of space `filled_in`
/// after its previous full clear (or its creation) and logs it as "JIT code cache full", at most
/// one line per 30 s.
inline void NoteFullCacheFlush(const char* isa, unsigned processor_id, std::size_t cache_size,
                               std::chrono::steady_clock::duration filled_in) {
    static std::atomic<std::int64_t> last_line_ms{std::numeric_limits<std::int64_t>::min() / 2};
    static std::atomic<std::uint64_t> total_at_last_line{0};

    const std::uint64_t total = g_full_cache_flushes.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
    std::int64_t last_ms = last_line_ms.load(std::memory_order_relaxed);
    if (now_ms - last_ms < 30'000 ||
        !last_line_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        return;
    }
    const std::uint64_t since_last_line =
        total - total_at_last_line.exchange(total, std::memory_order_relaxed);
    LOG_WARNING(Core_ARM,
                "JIT code cache full ({}, core {}, {} MiB): dropped all compiled code {:.1f} s "
                "after the previous clear, recompiling; full flushes {} total, {} since the last line",
                isa, processor_id, cache_size >> 20,
                std::chrono::duration<double>(filled_in).count(), total, since_last_line);
}

} // namespace Dynarmic::Backend::X64
