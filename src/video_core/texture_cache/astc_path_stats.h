// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Where ASTC texture uploads go on GPUs without ASTC: decoded and re-compressed to BC1/BC3 on the
// GPU (TranscodeAstcToBcn), only decoded on the GPU, or decoded (and re-compressed) on the CPU,
// and why those missed the GPU transcode. One log line every 30 s while there are ASTC uploads
// (grep "ASTC uploads last 30 s"). Idea 9's question: is anything left for a GPU path to take?

#include <array>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

/// Why an ASTC upload took the CPU decode path.
enum class AstcCpuReason : u8 {
    Volume,  ///< A 3D image (the GPU paths take 2D images only).
    Depth,   ///< Not 3D, but more than one slice deep.
    Setting, ///< ASTC decoding is set to the CPU (synchronous or asynchronous).
    Other,   ///< The GPU paths would have taken it: a fallback.
};
inline constexpr size_t AstcCpuReasonCount = 4;

[[nodiscard]] constexpr AstcCpuReason ClassifyCpuAstc(ImageType type, u32 depth,
                                                      bool gpu_decode_setting) {
    if (type == ImageType::e3D) {
        return AstcCpuReason::Volume;
    }
    if (depth > 1) {
        return AstcCpuReason::Depth;
    }
    return gpu_decode_setting ? AstcCpuReason::Other : AstcCpuReason::Setting;
}

class AstcPathStats {
public:
    /// An upload decoded on the GPU, re-compressed there (transcode) or not.
    void CountGpu(bool transcode, u64 texels) noexcept {
        const size_t path = transcode ? 0 : 1;
        uploads[path].fetch_add(1, std::memory_order_relaxed);
        texel_counts[path].fetch_add(texels, std::memory_order_relaxed);
    }

    /// An upload decoded on the CPU, and the time the decode (and re-compression) took.
    void CountCpu(AstcCpuReason reason, u64 texels, u64 ns) noexcept {
        uploads[2].fetch_add(1, std::memory_order_relaxed);
        texel_counts[2].fetch_add(texels, std::memory_order_relaxed);
        reasons[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
        cpu_ns.fetch_add(ns, std::memory_order_relaxed);
    }

    /// The window's line once 30 s have passed since the last one (nothing when there were no
    /// ASTC uploads), and a new window. One thread calls it.
    [[nodiscard]] std::optional<std::string> TakeReport(std::chrono::steady_clock::time_point now) {
        if (next_report == std::chrono::steady_clock::time_point{}) {
            next_report = now + std::chrono::seconds{30};
            return std::nullopt;
        }
        if (now < next_report) {
            return std::nullopt;
        }
        next_report = now + std::chrono::seconds{30};
        return Report(true);
    }

    /// The counts so far as one line (and, with reset, a new window). Nothing without uploads.
    [[nodiscard]] std::optional<std::string> Report(bool reset) {
        std::array<u64, 3> n{};
        std::array<u64, 3> texels{};
        std::array<u64, AstcCpuReasonCount> why{};
        for (size_t i = 0; i < n.size(); ++i) {
            n[i] = Load(uploads[i], reset);
            texels[i] = Load(texel_counts[i], reset);
        }
        for (size_t i = 0; i < why.size(); ++i) {
            why[i] = Load(reasons[i], reset);
        }
        const u64 ns = Load(cpu_ns, reset);
        if (n[0] + n[1] + n[2] == 0) {
            return std::nullopt;
        }
        const auto mtexels = [](u64 value) { return static_cast<double>(value) / 1e6; };
        return fmt::format(
            "ASTC uploads last 30 s: GPU decode + BC encode {} ({:.1f} Mtexels), GPU decode "
            "only {} ({:.1f} Mtexels), CPU {} ({:.1f} Mtexels, {:.1f} ms): 3D {}, depth > 1 {}, "
            "decode setting {}, other {}",
            n[0], mtexels(texels[0]), n[1], mtexels(texels[1]), n[2], mtexels(texels[2]),
            static_cast<double>(ns) / 1e6, why[0], why[1], why[2], why[3]);
    }

private:
    static u64 Load(std::atomic<u64>& value, bool reset) noexcept {
        return reset ? value.exchange(0, std::memory_order_relaxed)
                     : value.load(std::memory_order_relaxed);
    }

    std::array<std::atomic<u64>, 3> uploads{};
    std::array<std::atomic<u64>, 3> texel_counts{};
    std::array<std::atomic<u64>, AstcCpuReasonCount> reasons{};
    std::atomic<u64> cpu_ns{};
    std::chrono::steady_clock::time_point next_report{};
};

/// The process's counters (uploads come from the GPU thread and the asynchronous decoders).
inline AstcPathStats& GetAstcPathStats() {
    static AstcPathStats stats;
    return stats;
}

/// Texels of one ASTC level copy: width x height x depth x layers.
[[nodiscard]] constexpr u64 AstcCopyTexels(u32 width, u32 height, u32 depth, u32 layers) {
    return u64{width} * height * depth * layers;
}

} // namespace VideoCommon
