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
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

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

    /// True when entry `index` was read since the table was last invalidated or refreshed and
    /// its words in guest memory still equal the copy Read kept, i.e. Read would report it
    /// unchanged. Changes nothing but the page cache.
    template <class MemoryManager>
    [[nodiscard]] bool Matches(const MemoryManager& gpu_memory, u32 index) noexcept {
        if (invalidated || index > current_limit ||
            (read_descriptors[index / 64] & (1ULL << (index % 64))) == 0) {
            return false;
        }
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        if (const u8* const host = EntryPointer(gpu_memory, gpu_addr)) {
            return std::memcmp(host, std::addressof(descriptors[index]), sizeof(T)) == 0;
        }
        T entry;
        gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(entry), sizeof(T));
        return entry == descriptors[index];
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
        if (const u8* const host = EntryPointer(gpu_memory, gpu_addr)) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
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

    /// Forgets the pages Read translated (at each draw: mappings may change between draws).
    void ResetPages() noexcept {
        pages.Reset();
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    std::vector<Id> bound_ids;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    bool invalidated{}; ///< read_descriptors is to be cleared before the next read

private:
    /// Host pointer of the entry at `gpu_addr` when it lies in one page, else nullptr.
    template <class MemoryManager>
    const u8* EntryPointer(const MemoryManager& gpu_memory, GPUVAddr gpu_addr) {
        if ((gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) > Core::Memory::YUZU_PAGESIZE) {
            return nullptr;
        }
        if (g_descriptor_page_reuse.load(std::memory_order_relaxed)) {
            return pages.Get(gpu_addr, sizeof(T),
                             [&](GPUVAddr page) { return gpu_memory.GetPointer(page); });
        }
        return gpu_memory.GetPointer(gpu_addr);
    }

    void ClearReadDescriptors() noexcept {
        // Reserved capacity is not addressable. A later limit increase clears its new range.
        const size_t words = (std::min)(read_descriptors.size(), size_t{current_limit} / 64 + 1);
        std::fill_n(read_descriptors.begin(), words, u64{0});
        const size_t ids = (std::min)(bound_ids.size(), size_t{current_limit} + 1);
        std::fill_n(bound_ids.begin(), ids, Id{0});
    }

    GuestPageCache pages;
};

} // namespace VideoCommon
