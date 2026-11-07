// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "video_core/rasterizer_download_area.h"

namespace VideoCore {

/// Bumped when an image becomes GPU-modified (TextureCache). Guest reads of its pages must ask
/// the texture cache again, and read traps (armed by the buffer cache only) don't tell the cores
/// that. The guest read area tables (A/B bit 3 GuestReadAreas) are stamped with it.
inline std::atomic<u64> g_guest_read_epoch{0};

inline void BumpGuestReadEpoch() noexcept {
    g_guest_read_epoch.fetch_add(1, std::memory_order_release);
}

/// A guest core's remembered flush areas (A/B bit 3 GuestReadAreas, Core::Memory's
/// HandleRasterizerDownload): GPU::OnCPURead's answers since the stamp last changed. A read
/// inside one needs no flush, by the rule the single remembered area had before: nothing that
/// changes those answers happened since. Only its core uses it.
class GuestReadAreaTable {
public:
    static constexpr u32 Capacity = 8;

    /// What changes OnCPURead's answers: the read trap generation (the buffer cache's GPU writes),
    /// the image epoch (g_guest_read_epoch) and the A/B phase epoch.
    struct Stamp {
        u64 trap_generation{};
        u64 image_epoch{};
        u64 ab_epoch{};
        bool operator==(const Stamp&) const = default;
    };

    enum class Hit : u8 {
        None,
        Newest, ///< the area of the last miss: the single remembered area would have hit too
        Older,  ///< an older area: a hit only the table gives
    };

    /// Drops the areas when the stamp is not theirs.
    void Restamp(const Stamp& now) noexcept {
        if (now != stamp) {
            stamp = now;
            count = 0;
        }
    }

    /// Whether [start, end) is inside a remembered area. The newest is checked first.
    [[nodiscard]] Hit Find(u64 start, u64 end) const noexcept {
        for (u32 k = 0; k < count; ++k) {
            const u32 index = (newest + Capacity - k) % Capacity;
            const RasterizerDownloadArea& area = areas[index];
            if (area.start_address <= start && end <= area.end_address) {
                return k == 0 ? Hit::Newest : Hit::Older;
            }
        }
        return Hit::None;
    }

    /// Remembers a miss's area, in place of the oldest when full.
    void Insert(const RasterizerDownloadArea& area) noexcept {
        newest = count < Capacity ? count++ : (newest + 1) % Capacity;
        areas[newest] = area;
    }

    [[nodiscard]] u32 Count() const noexcept {
        return count;
    }

private:
    std::array<RasterizerDownloadArea, Capacity> areas{};
    Stamp stamp{};
    u32 count = 0;
    u32 newest = 0;
};

} // namespace VideoCore
