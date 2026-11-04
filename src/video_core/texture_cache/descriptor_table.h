// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

/// Cheaper per-draw descriptor lookups: entries compared where they are, only the read bits in
/// use cleared, views kept across table invalidations (TextureCache::VisitImageView).
/// EDEN_NX2_TEX_LOOKUP=0 turns them off (comparisons); flipping it at run time is safe.
inline std::atomic<bool> g_descriptor_lookup_reuse{[] {
    const char* value = std::getenv("EDEN_NX2_TEX_LOOKUP");
    return value == nullptr || value[0] != '0';
}()};

/// How a table entry compares with what was last read from it (DescriptorTable::ReadEntry).
enum class DescriptorRead : u8 {
    Same,       ///< The words read last time, and nothing invalidated the entry since
    UnreadSame, ///< Not read since the table was invalidated or moved, but the words it holds
                ///< (zeros when never read)
    New,        ///< Other words than the table holds
};

/// Texture handles and TIC/TSC entries read through GuestPageCache: one address translation per
/// 4 KiB page a draw reads instead of one per handle and entry. EDEN_NX2_TEX_PAGES=0 turns it off
/// (comparisons); flipping it at run time is safe.
inline std::atomic<bool> g_descriptor_page_reuse{[] {
    const char* value = std::getenv("EDEN_NX2_TEX_PAGES");
    return value == nullptr || value[0] != '0';
}()};

/// Host pointers of the last guest pages a draw read. Mappings can change between draws: the
/// texture cache resets its tables' caches at each SynchronizeDescriptors, and the pipelines keep
/// theirs for one draw.
class GuestPageCache {
public:
    /// Host pointer of `addr` when its `size` bytes lie in one page and `lookup(page)` has a
    /// pointer for that page, else nullptr.
    template <typename Lookup>
    [[nodiscard]] const u8* Get(GPUVAddr addr, size_t size, Lookup&& lookup) {
        const GPUVAddr offset = addr & Core::Memory::YUZU_PAGEMASK;
        if (offset + size > Core::Memory::YUZU_PAGESIZE) {
            return nullptr;
        }
        const GPUVAddr page = addr - offset;
        for (size_t i = 0; i < Ways; ++i) {
            if (pages[i] == page) {
                return hosts[i] + offset;
            }
        }
        const u8* const host = lookup(page);
        if (host == nullptr) {
            return nullptr;
        }
        pages[next] = page;
        hosts[next] = host;
        next = (next + 1) % Ways;
        return host + offset;
    }

    void Reset() noexcept {
        pages.fill(~GPUVAddr{0});
    }

private:
    static constexpr size_t Ways = 4;
    std::array<GPUVAddr, Ways> pages{~GPUVAddr{0}, ~GPUVAddr{0}, ~GPUVAddr{0}, ~GPUVAddr{0}};
    std::array<const u8*, Ways> hosts{};
    size_t next{};
};

/// Reads a texture handle word as MemoryManager::Read<u32> does, through `pages`.
template <class MemoryManager>
[[nodiscard]] u32 ReadHandleWord(GuestPageCache& pages, const MemoryManager& gpu_memory,
                                 GPUVAddr addr) {
    if (g_descriptor_page_reuse.load(std::memory_order_relaxed)) {
        if (const u8* const host = pages.Get(addr, sizeof(u32), [&](GPUVAddr page) {
                return gpu_memory.GetWordPage(page);
            })) {
            u32 value;
            std::memcpy(&value, host, sizeof(value));
            return value;
        }
    }
    return gpu_memory.template Read<u32>(addr);
}

template <typename T, typename Id = u32>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        // Cleared on the next read: the texture cache invalidates every channel's tables for
        // each image it deletes, dozens of times a frame while its collector runs.
        invalidated = true;
    }

    template <class MemoryManager>
    [[nodiscard]] std::pair<T, bool> Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            MarkRead(index);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    /// Reads an entry like Read, comparing it where it is: descriptors[index] holds its words
    /// afterwards, and an entry not read since an invalidation reports whether its words changed.
    template <class MemoryManager>
    [[nodiscard]] DescriptorRead ReadEntry(const MemoryManager& gpu_memory, u32 index) noexcept {
        if (!g_descriptor_lookup_reuse.load(std::memory_order_relaxed)) {
            return Read(gpu_memory, index).second ? DescriptorRead::New : DescriptorRead::Same;
        }
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        const bool was_read = (read_descriptors[index / 64] & (1ULL << (index % 64))) != 0;
        if (!was_read) {
            MarkRead(index);
        }
        T& last = descriptors[index];
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        const u8* host = nullptr;
        if (one_page) {
            host = g_descriptor_page_reuse.load(std::memory_order_relaxed)
                       ? pages.Get(gpu_addr, sizeof(T),
                                   [&](GPUVAddr page) { return gpu_memory.GetPointer(page); })
                       : gpu_memory.GetPointer(gpu_addr);
        }
        bool same;
        if (host != nullptr) {
            // Most entries a draw reads are the words of the last draw again: compared where
            // they are, copied only when they changed.
            same = SameWords(host, &last);
            if (!same) {
                std::memcpy(&last, host, sizeof(T));
            }
        } else {
            T value;
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(value), sizeof(T));
            same = SameWords(&value, &last);
            if (!same) {
                last = value;
            }
        }
        if (!same) {
            return DescriptorRead::New;
        }
        return was_read ? DescriptorRead::Same : DescriptorRead::UnreadSame;
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        pages.Reset();
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors, Id{0});
    }

    /// Forgets the pages ReadEntry translated (at each draw: mappings may change between draws).
    void ResetPages() noexcept {
        pages.Reset();
    }

    /// The erase count of bound_ids[index]'s slot when the entry found it (image tables, see
    /// TextureCache::VisitImageView); ~0 when none was stored.
    [[nodiscard]] u32 BoundErasures(u32 index) const noexcept {
        return index < bound_erasures.size() ? bound_erasures[index] : ~u32{0};
    }

    void SetBoundErasures(u32 index, u32 erasures) {
        if (index >= bound_erasures.size()) {
            bound_erasures.resize((std::max)(descriptors.size(), size_t{index} + 1), ~u32{0});
        }
        bound_erasures[index] = erasures;
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    std::vector<u32> bound_erasures;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read

private:
    static bool SameWords(const void* lhs, const void* rhs) noexcept {
        static_assert(sizeof(T) % sizeof(u64) == 0);
        u64 diff = 0;
        for (size_t offset = 0; offset < sizeof(T); offset += sizeof(u64)) {
            u64 a, b;
            std::memcpy(&a, static_cast<const u8*>(lhs) + offset, sizeof(u64));
            std::memcpy(&b, static_cast<const u8*>(rhs) + offset, sizeof(u64));
            diff |= a ^ b;
        }
        return diff == 0;
    }

    void MarkRead(u32 index) noexcept {
        read_descriptors[index / 64] |= 1ULL << (index % 64);
        read_words_begin = (std::min)(read_words_begin, index / 64);
        read_words_end = (std::max)(read_words_end, index / 64 + 1);
    }

    void ClearReadDescriptors() noexcept {
        if (g_descriptor_lookup_reuse.load(std::memory_order_relaxed)) {
            // Only the words with bits set: the limit can be far above the entries in use, and
            // this runs after every image deletion. bound_ids stay: an entry whose words are
            // the same keeps its view while the view lives (TextureCache::VisitImageView).
            const size_t end = (std::min)(read_descriptors.size(), size_t{read_words_end});
            if (read_words_begin < end) {
                std::fill(read_descriptors.begin() + read_words_begin,
                          read_descriptors.begin() + end, u64{0});
            }
            read_words_begin = (std::numeric_limits<u32>::max)();
            read_words_end = 0;
            return;
        }
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
        // Bits past the limit (set under a larger one) stay.
        read_words_begin = (std::max)(read_words_begin, static_cast<u32>(words));
        if (read_words_begin >= read_words_end) {
            read_words_begin = (std::numeric_limits<u32>::max)();
            read_words_end = 0;
        }
    }

    GuestPageCache pages;
    /// Words of read_descriptors that may have bits set: [begin, end)
    u32 read_words_begin = (std::numeric_limits<u32>::max)();
    u32 read_words_end = 0;
};

} // namespace VideoCommon
