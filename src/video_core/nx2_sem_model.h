// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Split GPU thread: the front's model of the semaphore words the back will have written once it
// has run everything queued so far. The front queues every release itself (puller semaphores,
// the 3D and compute engines' report semaphores, compute QMD releases through a shadow of the
// QMD scheduler), in the order the back runs them; with the default fence behavior the back
// applies each at once. A semaphore acquire the model says is met then needs no drain: the
// releases run before anything queued after the acquire. The back checks each skipped acquire
// at its place in the stream; a word the model got wrong drains from then on.
// EDEN_NX2_SEM_MODEL=0: off (every unmet acquire drains, as before). =check: the acquires still
// drain and the log counts how the model did.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/memory_manager.h"

namespace VideoCommon::Nx2SemModel {

enum class Mode { Off, Check, Skip };

inline const Mode g_mode = [] {
    const char* value = std::getenv("EDEN_NX2_SEM_MODEL");
    if (value == nullptr) {
        return Mode::Skip;
    }
    return value[0] == 'c' ? Mode::Check : value[0] == '0' ? Mode::Off : Mode::Skip;
}();

enum class Op { Set, Add, Max, Unknown };

struct Entry {
    u32 value{};
    u64 epoch{};
    bool known{};
};

/// Front thread only.
inline std::unordered_map<DAddr, Entry> g_entries;
/// Bumped by GpuSplit::Drain once the back has run everything: every word is in memory then.
inline u64 g_epoch = 1;

inline bool Active() {
    return g_mode != Mode::Off;
}

/// A release the front queues now: op on the word at gpu_addr (bytes: 4, or 16 for a release
/// with a timestamp, which also writes the three words after it).
inline void Apply(const Tegra::MemoryManager& memory, GPUVAddr gpu_addr, Op op, u32 payload,
                  u32 bytes = 4) {
    if (!Active()) {
        return;
    }
    const auto device = memory.GpuToCpuAddress(gpu_addr);
    if (!device) {
        return;
    }
    Entry& entry = g_entries[*device];
    if (entry.epoch != g_epoch) {
        // No release of this word was queued since the back was last idle: memory holds it.
        entry.value = memory.Read<u32>(gpu_addr);
        entry.known = true;
        entry.epoch = g_epoch;
    }
    switch (op) {
    case Op::Set:
        entry.value = payload;
        break;
    case Op::Add:
        entry.value += payload;
        break;
    case Op::Max:
        entry.value = (std::max)(entry.value, payload);
        break;
    case Op::Unknown:
        entry.known = false;
        break;
    }
    // EDEN_NX2_SEM_WATCH=<gpu address> (as the back's semaphore watch): what the front models.
    static const GPUVAddr watch = [] {
        const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
        return text ? GPUVAddr{std::strtoull(text, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (gpu_addr & ~GPUVAddr{0xf}) == (watch & ~GPUVAddr{0xf})) {
        LOG_INFO(HW_GPU, "Semaphore model watch: {:#x} op {} payload {:#x} bytes {} -> {:#x} (known {})",
                 gpu_addr, static_cast<int>(op), payload, bytes, entry.value, entry.known);
    }
    for (u32 offset = 4; offset < bytes; offset += 4) {
        Entry& next = g_entries[*device + offset];
        next.epoch = g_epoch;
        next.known = offset == 4 && op == Op::Set; // the payload's upper half
        next.value = 0;
    }
}

/// The value the word will hold once the back ran everything queued, if the front knows it.
inline std::optional<u32> Predict(const Tegra::MemoryManager& memory, GPUVAddr gpu_addr) {
    const auto device = memory.GpuToCpuAddress(gpu_addr);
    if (!device) {
        return std::nullopt;
    }
    const auto it = g_entries.find(*device);
    if (it == g_entries.end() || it->second.epoch != g_epoch || !it->second.known) {
        return std::nullopt;
    }
    return it->second.value;
}

/// Called once the back is idle (a full drain).
inline void OnDrained() {
    ++g_epoch;
    if (g_entries.size() > 65536) {
        g_entries.clear();
    }
}

inline bool Skips() {
    return g_mode == Mode::Skip;
}

/// Words the model once got wrong (a writer it doesn't see): their acquires always drain.
/// Written by the back (skip mode's check) and the front (check mode), read by the front.
inline std::mutex g_suspect_mutex;
inline std::unordered_set<DAddr> g_suspects;
inline std::atomic<u32> g_suspect_count{};

inline bool IsSuspect(DAddr device) {
    if (g_suspect_count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::scoped_lock lock{g_suspect_mutex};
    return g_suspects.contains(device);
}

inline void MarkSuspect(DAddr device, GPUVAddr address, u32 value, u32 wanted) {
    std::scoped_lock lock{g_suspect_mutex};
    if (g_suspects.insert(device).second) {
        g_suspect_count.fetch_add(1, std::memory_order_relaxed);
        LOG_WARNING(HW_GPU, "Semaphore model: {:#x} was {:#x} where the model said >= {:#x}; its "
                    "acquires wait for the back from now on", address, value, wanted);
    }
}

struct Counts {
    u64 unmet{};       ///< GEQUAL acquires not met in memory
    u64 predicted{};   ///< Of those, ones the model says are met
    u64 right{};       ///< Predicted, and met after the drain
    u64 wrong{};       ///< Predicted, but not met after the drain
    u64 value_off{};   ///< Predicted value differed from the word after the drain
    u64 skipped{};     ///< Drains skipped (skip mode)
    u64 suspect{};     ///< Predicted, but on a suspect word (drained)
};
inline Counts g_counts{};
/// Skip mode: skipped acquires the back found not met at their point of the stream.
inline std::atomic<u64> g_skip_misses{};

inline void LogCounts() {
    const auto& c = g_counts;
    LOG_INFO(HW_GPU,
             "Semaphore model: {} unmet GEQUAL acquires, model says met {} (right {}, wrong {}, "
             "value off {}, on suspect words {}), drains skipped {} (back found {} not met), "
             "suspect words {}",
             c.unmet, c.predicted, c.right, c.wrong, c.value_off, c.suspect, c.skipped,
             g_skip_misses.load(std::memory_order_relaxed),
             g_suspect_count.load(std::memory_order_relaxed));
}

} // namespace VideoCommon::Nx2SemModel
