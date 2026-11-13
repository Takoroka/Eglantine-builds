// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"

namespace VideoCommon {

class UsageTracker {
    // PAGE_SHIFT is a macro on FreeBSD
    static constexpr size_t BUFFER_BYTES_PER_BITSHIFT = 6;
    static constexpr size_t BUFFER_PAGE_SHIFT = 6 + BUFFER_BYTES_PER_BITSHIFT;
    static constexpr size_t BUFFER_PAGE_BYTES = 1 << BUFFER_PAGE_SHIFT;
public:
    explicit UsageTracker(size_t size) {
        const size_t num_pages = (size >> BUFFER_PAGE_SHIFT) + 1;
        pages.resize(num_pages, 0ULL);
        full_pages.resize(Common::DivCeil(num_pages, size_t{64}), 0ULL);
    }

    void Reset() noexcept {
        // Every buffer the GPU is done with is reset each frame (BufferCacheRuntime::TickFrame),
        // used or not: one not tracked since the last reset is skipped, the others clear only the
        // pages tracked since then instead of every page.
        if (!tracked) {
            return;
        }
        tracked = false;
        if (touched_begin < touched_end) {
            std::fill(pages.begin() + touched_begin, pages.begin() + touched_end, 0ULL);
            std::fill(full_pages.begin() + touched_begin / 64,
                      full_pages.begin() + Common::DivCeil(touched_end, size_t{64}), 0ULL);
        }
        touched_begin = ~size_t{0};
        touched_end = 0;
        if (recent_cursor != 0) {
            recent.fill({~u64{0}, 0});
            recent_cursor = 0;
        }
    }

    void Track(u64 offset, u64 size) noexcept {
        // Tracking is idempotent until Reset: skip ranges tracked just before. Draws bind the
        // same (often multi-megabyte) storage buffers again and again, and each Track wrote a
        // word per 4 KiB of the range.
        for (const auto& [recent_offset, recent_size] : recent) {
            if (recent_offset == offset && recent_size == size) {
                return;
            }
        }
        tracked = true;
        recent[recent_cursor++ % recent.size()] = {offset, size};
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return;
        }
        touched_begin = (std::min)(touched_begin, page);
        touched_end = (std::max)(touched_end, page_end + 1);
        TrackPage(page, offset, size);
        if (page == page_end) {
            return;
        }
        // The pages in between are used whole: a bit each in full_pages, not a word each in
        // pages. Vertex buffer bindings span megabytes, and filling their words was thousands of
        // writes per draw.
        SetFullPages(page + 1, page_end);
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        TrackPage(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

    [[nodiscard]] bool IsUsed(u64 offset, u64 size) const noexcept {
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return false;
        }
        if (IsPageUsed(page, offset, size)) {
            return true;
        }
        if (page + 1 < page_end && AnyFullPage(page + 1, page_end)) {
            return true;
        }
        for (size_t i = page + 1; i < page_end; i++) {
            if (pages[i] != 0) {
                return true;
            }
        }
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        return IsPageUsed(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

private:
    void TrackPage(u64 page, u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const size_t offset_in_page = offset % BUFFER_PAGE_BYTES;
        const size_t first_bit = offset_in_page >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t bytes = std::min<size_t>(size, BUFFER_PAGE_BYTES - offset_in_page);
        const size_t last_bit = (offset_in_page + bytes - 1) >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t num_bits = last_bit - first_bit + 1;
        const size_t mask = ~u64{0} >> (64 - num_bits);
        pages[page] |= (~u64{0} & mask) << first_bit;
    }

    bool IsPageUsed(u64 page, u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        if (((full_pages[page / 64] >> (page % 64)) & 1) != 0) {
            return true;
        }
        const size_t offset_in_page = offset % BUFFER_PAGE_BYTES;
        const size_t first_bit = offset_in_page >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t bytes = std::min<size_t>(size, BUFFER_PAGE_BYTES - offset_in_page);
        const size_t last_bit = (offset_in_page + bytes - 1) >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t num_bits = last_bit - first_bit + 1;
        const size_t mask = ~u64{0} >> (64 - num_bits);
        const size_t mask2 = (~u64{0} & mask) << first_bit;
        return (pages[page] & mask2) != 0;
    }

    /// Calls func(word index, mask) for the full_pages bits of pages [begin, end).
    template <typename Func>
    static bool ForEachFullPageWord(size_t begin, size_t end, Func&& func) noexcept {
        while (begin < end) {
            const size_t bit = begin % 64;
            const size_t count = (std::min)(end - begin, 64 - bit);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if (func(begin / 64, mask)) {
                return true;
            }
            begin += count;
        }
        return false;
    }

    void SetFullPages(size_t begin, size_t end) noexcept {
        ForEachFullPageWord(begin, end, [this](size_t word, u64 mask) {
            full_pages[word] |= mask;
            return false;
        });
    }

    bool AnyFullPage(size_t begin, size_t end) const noexcept {
        return ForEachFullPageWord(begin, end, [this](size_t word, u64 mask) {
            return (full_pages[word] & mask) != 0;
        });
    }

private:
    std::vector<u64> pages;
    /// A bit per page, set where a tracked range covers the whole page.
    std::vector<u64> full_pages;
    /// Ranges tracked last, (offset, size).
    std::array<std::pair<u64, u64>, 4> recent{{{~u64{0}, 0}, {~u64{0}, 0}, {~u64{0}, 0}, {~u64{0}, 0}}};
    size_t recent_cursor{}; ///< 0: no ranges in `recent` since the last reset
    bool tracked{};         ///< Something was tracked since the last reset
    /// Pages [touched_begin, touched_end) hold every page tracked since the last reset.
    size_t touched_begin{~size_t{0}};
    size_t touched_end{};
};

} // namespace VideoCommon
