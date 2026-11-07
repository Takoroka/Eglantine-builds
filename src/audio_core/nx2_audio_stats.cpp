// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include "audio_core/nx2_audio_stats.h"
#include "common/logging.h"

namespace AudioCore::Nx2AudioStats {

namespace {

constexpr s64 WINDOW_NS = 10'000'000'000;

std::array<StreamCounters, MAX_STREAMS> g_streams{};
std::mutex g_streams_mutex;
std::atomic<s64> g_window_start{0};

u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

double Ms(u64 ns) {
    return static_cast<double>(ns) / 1e6;
}

double PerSecond(u64 count, s64 window_ns) {
    return window_ns > 0 ? static_cast<double>(count) * 1e9 / static_cast<double>(window_ns)
                         : 0.0;
}

} // Anonymous namespace

StreamCounters& Stream(std::string_view name) {
    std::scoped_lock lock{g_streams_mutex};
    const std::string_view shown = name.substr(0, sizeof(StreamCounters::name) - 1);
    for (StreamCounters& stream : g_streams) {
        if (stream.used.load(std::memory_order_relaxed) &&
            shown == std::string_view{stream.name.data()}) {
            return stream;
        }
    }
    for (StreamCounters& stream : g_streams) {
        if (!stream.used.load(std::memory_order_relaxed)) {
            std::memcpy(stream.name.data(), shown.data(), shown.size());
            stream.name[shown.size()] = '\0';
            stream.used.store(true, std::memory_order_release);
            return stream;
        }
    }
    return g_streams.back();
}

void MaybeReport() {
    const s64 now = NowNs();
    s64 start = g_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
        return;
    }
    if (now - start < WINDOW_NS ||
        !g_window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    const s64 window = now - start;

    std::string streams;
    for (StreamCounters& stream : g_streams) {
        if (!stream.used.load(std::memory_order_acquire)) {
            continue;
        }
        const u64 callbacks = Take(stream.callbacks);
        const u64 frames = Take(stream.frames_requested);
        const u64 frames_max = Take(stream.frames_requested_max);
        const u64 queue_sum = Take(stream.queue_sum);
        const u64 queue_min = stream.queue_min.exchange(~u64{0}, std::memory_order_relaxed);
        const u64 underruns = Take(stream.underrun_callbacks);
        const u64 underrun_frames = Take(stream.underrun_frames);
        const u64 short_pops = Take(stream.short_pop_frames);
        const u64 appended = Take(stream.appended);
        const u64 dropped = Take(stream.dropped_samples);
        const u64 paused = Take(stream.paused_callbacks);
        if (callbacks + appended == 0) {
            continue;
        }
        streams += fmt::format(
            "{}{}: callbacks {:.0f}/s, frames/callback avg {:.0f} max {}, queue min {} avg {:.1f} "
            "buffers, underruns {} ({:.1f} ms held), short pops {:.1f} ms, buffers in {:.0f}/s, "
            "dropped {:.1f} ms, paused callbacks {}",
            streams.empty() ? "" : " ; ", stream.name.data(), PerSecond(callbacks, window),
            callbacks != 0 ? static_cast<double>(frames) / static_cast<double>(callbacks) : 0.0,
            frames_max, callbacks != 0 ? queue_min : 0,
            callbacks != 0 ? static_cast<double>(queue_sum) / static_cast<double>(callbacks) : 0.0,
            underruns, static_cast<double>(underrun_frames) / 48.0,
            static_cast<double>(short_pops) / 48.0, PerSecond(appended, window),
            // 2 channels at 48 kHz: samples / 96 = ms
            static_cast<double>(dropped) / 96.0, paused);
    }

    RenderCounters& r = g_render;
    const u64 passes = Take(r.passes);
    const u64 pass_ns = Take(r.pass_ns);
    const u64 wait_ns = Take(r.wait_ns);
    std::string sessions;
    for (size_t session = 0; session < MAX_SESSIONS; ++session) {
        const u64 updates = Take(r.updates[session]);
        const u64 update_ns = Take(r.update_ns[session]);
        sessions += fmt::format(
            "{}session {}: game updates {:.1f}/s (avg {:.2f} ms max {:.2f} ms), lists {:.0f}/s",
            session == 0 ? "" : ", ", session, PerSecond(updates, window),
            updates != 0 ? Ms(update_ns) / static_cast<double>(updates) : 0.0,
            Ms(Take(r.update_ns_max[session])), PerSecond(Take(r.generated[session]), window));
    }
    LOG_INFO(Audio,
             "nx2 audio (last {:.1f} s) | {} | render: {:.0f} passes/s, pass avg {:.2f} ms max "
             "{:.2f} ms ({} over 5 ms), longest gap {:.1f} ms, waited for space avg {:.2f} ms, "
             "{} still full after the wait, {} started with the sink empty, {} unfinished lists, "
             "{} unpaced (session 1 without session 0) | {}, lock wait max {:.2f} ms, command "
             "generation max {:.2f} ms | voices starved {:.1f}/s",
             static_cast<double>(window) / 1e9, streams.empty() ? "no streams" : streams,
             PerSecond(passes, window),
             passes != 0 ? Ms(pass_ns) / static_cast<double>(passes) : 0.0,
             Ms(Take(r.pass_ns_max)), Take(r.slow_passes), Ms(Take(r.gap_ns_max)),
             passes != 0 ? Ms(wait_ns) / static_cast<double>(passes) : 0.0,
             Take(r.full_after_wait), Take(r.empty_at_start), Take(r.unfinished_lists),
             Take(r.unpaced_passes), sessions,
             Ms(Take(r.lock_wait_ns_max)), Ms(Take(r.generate_ns_max)),
             PerSecond(Take(r.voices_starved), window));
}

} // namespace AudioCore::Nx2AudioStats
