// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEMP (measurement aid, 11-07): where audio goes missing. Relaxed counters on the sink streams
// (the device callback), the ADSP render loop, the render manager's command generation and the
// game's renderer updates; one "nx2 audio" log line every 10 s (MaybeReport, from the device
// callback).

#include <array>
#include <atomic>
#include <chrono>
#include <string_view>

#include "common/common_types.h"

namespace AudioCore::Nx2AudioStats {

/// Monotonic nanoseconds (steady clock).
inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void AddMax(std::atomic<u64>& max, u64 value) {
    u64 current = max.load(std::memory_order_relaxed);
    while (value > current &&
           !max.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

inline void AddMin(std::atomic<u64>& min, u64 value) {
    u64 current = min.load(std::memory_order_relaxed);
    while (value < current &&
           !min.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

/// One sink stream (the device side): its callbacks and the buffers the renderer appends.
struct StreamCounters {
    std::array<char, 32> name{};
    std::atomic<bool> used{false};
    std::atomic<u64> callbacks{};
    std::atomic<u64> frames_requested{};
    std::atomic<u64> frames_requested_max{};
    std::atomic<u64> queue_sum{};             ///< Buffers queued at each callback's start
    std::atomic<u64> queue_min{~u64{0}};
    std::atomic<u64> underrun_callbacks{};    ///< Callbacks that ran out of queued buffers
    std::atomic<u64> underrun_frames{};       ///< Frames filled with the last frame instead
    std::atomic<u64> short_pop_frames{};      ///< Queued frames whose samples weren't in the ring
    std::atomic<u64> appended{};              ///< Buffers appended (rendered)
    std::atomic<u64> dropped_samples{};       ///< Samples the full ring didn't take
    std::atomic<u64> paused_callbacks{};      ///< Callbacks while the system was paused
};

constexpr size_t MAX_STREAMS = 6;
constexpr size_t MAX_SESSIONS = 2;

/// The ADSP render loop, the render manager and the game's renderer sessions.
struct RenderCounters {
    std::atomic<u64> passes{};             ///< Render messages the ADSP handled
    std::atomic<u64> pass_ns{};            ///< Their processing time (sessions, after the wait)
    std::atomic<u64> pass_ns_max{};
    std::atomic<u64> slow_passes{};        ///< Passes over 5 ms (one buffer's worth)
    std::atomic<u64> gap_ns_max{};         ///< Longest time between two render messages
    std::atomic<u64> wait_ns{};            ///< Time in WaitFreeSpace
    std::atomic<u64> full_after_wait{};    ///< WaitFreeSpace returned with the queue still full
    std::atomic<u64> empty_at_start{};     ///< Passes that started with session 0's queue empty
    std::atomic<u64> unfinished_lists{};   ///< Command lists left with commands (adsp behind)
    std::atomic<u64> unpaced_passes{};     ///< Session 1 rendered with no session 0 (no wait)
    std::array<std::atomic<u64>, MAX_SESSIONS> updates{};      ///< Game RequestUpdate calls
    std::array<std::atomic<u64>, MAX_SESSIONS> update_ns{};
    std::array<std::atomic<u64>, MAX_SESSIONS> update_ns_max{};
    std::array<std::atomic<u64>, MAX_SESSIONS> generated{};    ///< Command lists generated
    std::atomic<u64> lock_wait_ns_max{};   ///< SendCommandToDsp waiting for the session lock
    std::atomic<u64> generate_ns_max{};    ///< SendCommandToDsp's whole time
    std::atomic<u64> voices_starved{};     ///< Voice decodes that ran out of wave buffers
    /// Voices that starved and got data again within 200 ms (a gap in a continuing stream, not
    /// a sound that ended), the gaps' summed and longest length in passes (5 ms each), and the
    /// sample format of the voices that starved (index = SampleFormat).
    std::atomic<u64> starved_resumed{};
    std::atomic<u64> resumed_gap_passes{};
    std::atomic<u64> resumed_gap_passes_max{};
    std::array<std::atomic<u64>, 8> starved_by_format{};
    std::atomic<u64> pass_index{};         ///< Monotonic ADSP pass count (never reset)
};

inline RenderCounters g_render{};

/// The counters of the stream with this name (one slot per name; shared if all are taken).
StreamCounters& Stream(std::string_view name);

/// Logs the line when 10 s have passed since the last one (any thread; one wins).
void MaybeReport();

/// A voice decode ended (the ADSP thread): starved or not, for the resumed-gap counters.
/// voice: the voice channel's state (its identity), format: the SampleFormat value.
void CountVoiceDecode(const void* voice, u32 format, bool starved);

/// EDEN_NX2_AUDIO_DUMP=<dir>: appends a render session's final mix (before volume, so a muted
/// run still has it) to <dir>/<stream name>.pcm as interleaved s16. The ADSP thread only.
void DumpMix(std::string_view stream_name, const s16* samples, size_t count, u32 channels);

} // namespace AudioCore::Nx2AudioStats
