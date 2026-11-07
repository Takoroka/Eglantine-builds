// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

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

/// Per voice channel: the pass it last starved in (the ADSP thread only).
std::unordered_map<const void*, u64>& StarvedAt() {
    static std::unordered_map<const void*, u64> starved_at;
    return starved_at;
}

} // Anonymous namespace

void CountVoiceDecode(const void* voice, u32 format, bool starved) {
    constexpr u64 RESUME_WINDOW_PASSES = 40; // 200 ms
    RenderCounters& r = g_render;
    const u64 pass = r.pass_index.load(std::memory_order_relaxed);
    auto& starved_at = StarvedAt();
    if (starved) {
        r.starved_by_format[(std::min<size_t>)(format, r.starved_by_format.size() - 1)].fetch_add(
            1, std::memory_order_relaxed);
        starved_at.try_emplace(voice, pass);
        return;
    }
    const auto it = starved_at.find(voice);
    if (it == starved_at.end()) {
        return;
    }
    const u64 gap = pass - it->second;
    starved_at.erase(it);
    if (gap <= RESUME_WINDOW_PASSES) {
        r.starved_resumed.fetch_add(1, std::memory_order_relaxed);
        r.resumed_gap_passes.fetch_add(gap, std::memory_order_relaxed);
        AddMax(r.resumed_gap_passes_max, gap);
    }
    // Keep the map small: voices that starved long ago and never came back ended.
    if (starved_at.size() > 512) {
        std::erase_if(starved_at, [pass](const auto& entry) {
            return pass - entry.second > RESUME_WINDOW_PASSES;
        });
    }
}

void DumpMix(std::string_view stream_name, const s16* samples, size_t count, u32 channels) {
    static const char* const directory = std::getenv("EDEN_NX2_AUDIO_DUMP");
    if (directory == nullptr) {
        return;
    }
    static std::unordered_map<std::string, std::FILE*> files;
    const std::string name{stream_name};
    auto it = files.find(name);
    if (it == files.end()) {
        const std::string path = fmt::format("{}/{}.pcm", directory, name);
        std::FILE* file = std::fopen(path.c_str(), "wb");
        LOG_INFO(Audio, "nx2 audio dump: {} ({} channels, 48000 Hz, s16) {}", path, channels,
                 file != nullptr ? "open" : "FAILED");
        it = files.emplace(name, file).first;
    }
    if (it->second != nullptr) {
        std::fwrite(samples, sizeof(s16), count, it->second);
    }
}

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
        const u64 target = Take(stream.queue_target_max);
        if (callbacks + appended == 0) {
            continue;
        }
        streams += fmt::format(
            "{}{}: callbacks {:.0f}/s, frames/callback avg {:.0f} max {}, queue min {} avg {:.1f} "
            "buffers (target max {}), underruns {} ({:.1f} ms held), short pops {:.1f} ms, "
            "buffers in {:.0f}/s, dropped {:.1f} ms, paused callbacks {}",
            streams.empty() ? "" : " ; ", stream.name.data(), PerSecond(callbacks, window),
            callbacks != 0 ? static_cast<double>(frames) / static_cast<double>(callbacks) : 0.0,
            frames_max, callbacks != 0 ? queue_min : 0,
            callbacks != 0 ? static_cast<double>(queue_sum) / static_cast<double>(callbacks) : 0.0,
            target, underruns, static_cast<double>(underrun_frames) / 48.0,
            static_cast<double>(short_pops) / 48.0, PerSecond(appended, window),
            // 2 channels at 48 kHz: samples / 96 = ms
            static_cast<double>(dropped) / 96.0, paused);
    }

    RenderCounters& r = g_render;
    // Starved voices by SampleFormat (0 Invalid, 1 PcmInt8, 2 PcmInt16, 3 PcmInt24, 4 PcmInt32,
    // 5 PcmFloat, 6 Adpcm), only the formats that starved.
    std::string by_format;
    for (size_t format = 0; format < r.starved_by_format.size(); ++format) {
        if (const u64 count = Take(r.starved_by_format[format]); count != 0) {
            by_format += fmt::format("{}{}:{}", by_format.empty() ? "" : " ", format, count);
        }
    }
    if (by_format.empty()) {
        by_format = "none";
    }
    const u64 resumed = Take(r.starved_resumed);
    const u64 gap_passes = Take(r.resumed_gap_passes);
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
             "generation max {:.2f} ms | voices starved {:.1f}/s (by format {}), back within "
             "200 ms {:.2f}/s (gap avg {:.1f} ms max {:.0f} ms)",
             static_cast<double>(window) / 1e9, streams.empty() ? "no streams" : streams,
             PerSecond(passes, window),
             passes != 0 ? Ms(pass_ns) / static_cast<double>(passes) : 0.0,
             Ms(Take(r.pass_ns_max)), Take(r.slow_passes), Ms(Take(r.gap_ns_max)),
             passes != 0 ? Ms(wait_ns) / static_cast<double>(passes) : 0.0,
             Take(r.full_after_wait), Take(r.empty_at_start), Take(r.unfinished_lists),
             Take(r.unpaced_passes), sessions,
             Ms(Take(r.lock_wait_ns_max)), Ms(Take(r.generate_ns_max)),
             PerSecond(Take(r.voices_starved), window), by_format,
             PerSecond(resumed, window),
             resumed != 0 ? static_cast<double>(gap_passes) * 5.0 / static_cast<double>(resumed)
                          : 0.0,
             static_cast<double>(Take(r.resumed_gap_passes_max)) * 5.0);
}

} // namespace AudioCore::Nx2AudioStats
