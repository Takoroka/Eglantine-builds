// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

/// Cheaper per-draw descriptor lookups: entries compared where they are, the read bits in use
/// cleared, views kept across table invalidations (TextureCache::VisitImageView).
/// EDEN_NX2_TEX_LOOKUP=0 turns them off (comparisons); flipping it at run time is safe.
inline std::atomic<bool> g_descriptor_lookup_reuse{[] {
    const char* value = std::getenv("EDEN_NX2_TEX_LOOKUP");
    return value == nullptr || value[0] != '0';
}()};

/// How a table entry compares with what was last read from it.
enum class DescriptorRead : u8 {
    Same,       ///< The words read last time, and nothing invalidated the entry since
    UnreadSame, ///< Not read since the table was invalidated or moved, but the words read last
    New,        ///< Other words than the last read, or never read
};

template <typename T>
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

    /// Reads an entry: descriptors[index] holds its words afterwards.
    template <class MemoryManager>
    [[nodiscard]] DescriptorRead Read(const MemoryManager& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        if (invalidated) {
            ClearReadDescriptors();
            invalidated = false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        T& last = descriptors[index];
        u64& read_word = read_descriptors[index / 64];
        const u64 read_bit = 1ULL << (index % 64);
        const bool was_read = (read_word & read_bit) != 0;
        if (!was_read) {
            read_word |= read_bit;
            read_words_begin = (std::min)(read_words_begin, index / 64);
            read_words_end = (std::max)(read_words_end, index / 64 + 1);
        }
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr;
        bool same;
        if (host != nullptr && g_descriptor_lookup_reuse.load(std::memory_order_relaxed)) {
            // Most entries a draw reads are the words of the last draw again: compared where
            // they are, copied only when they changed.
            same = SameWords(host, &last);
            if (!same) {
                std::memcpy(&last, host, sizeof(T));
            }
        } else {
            T value;
            if (host != nullptr) {
                std::memcpy(std::addressof(value), host, sizeof(T));
            } else {
                gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(value), sizeof(T));
            }
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
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        ClearReadDescriptors();
        invalidated = false;
        //
        descriptors.resize(num_descriptors);
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
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

    void ClearReadDescriptors() noexcept {
        if (g_descriptor_lookup_reuse.load(std::memory_order_relaxed)) {
            // Only the words with bits set: the limit can be far above the entries in use, and
            // this runs after every image deletion.
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
        // Bits past the limit (set under a larger one) stay.
        read_words_begin = (std::max)(read_words_begin, static_cast<u32>(words));
        if (read_words_begin >= read_words_end) {
            read_words_begin = (std::numeric_limits<u32>::max)();
            read_words_end = 0;
        }
    }

    /// Words of read_descriptors that may have bits set: [begin, end)
    u32 read_words_begin = (std::numeric_limits<u32>::max)();
    u32 read_words_end = 0;
};

} // namespace VideoCommon
