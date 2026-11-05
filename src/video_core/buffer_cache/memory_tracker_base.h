// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <deque>
#include <limits>
#include <type_traits>
#include "common/container/unordered_set.h"
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "video_core/buffer_cache/word_manager.h"

namespace VideoCommon {

template <typename DeviceTracker>
class MemoryTrackerBase {
    static constexpr size_t MAX_CPU_PAGE_BITS = 36; // device addresses (MaxwellDeviceTraits)
    static constexpr size_t HIGHER_PAGE_BITS = 22;
    static constexpr size_t HIGHER_PAGE_SIZE = 1ULL << HIGHER_PAGE_BITS;
    static constexpr size_t HIGHER_PAGE_MASK = HIGHER_PAGE_SIZE - 1ULL;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;
    static constexpr size_t WORDS_STACK_NEEDED = HIGHER_PAGE_SIZE / BYTES_PER_WORD;
    using Manager = WordManager<DeviceTracker, WORDS_STACK_NEEDED, HIGHER_PAGE_SIZE>;

public:
    MemoryTrackerBase(DeviceTracker& device_tracker_) : device_tracker{&device_tracker_} {}
    ~MemoryTrackerBase() = default;

    /// Returns the inclusive CPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedCpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<true>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::CPU, offset, size);
        });
    }

    /// Returns the inclusive GPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedGpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::GPU, offset, size);
        });
    }

    /// Returns true if a region has been modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::CPU, offset, size);
        });
    }

    /// Returns true if an upload of the region (ForEachUploadRange) would find nothing to upload
    /// and change no tracking: no page is CPU modified or untracked. A region not created yet is
    /// not clean (it starts all CPU modified).
    [[nodiscard]] bool IsRegionCpuClean(VAddr query_cpu_addr, u64 query_size) const noexcept {
        // Hot pages only need an upload while CPU modified (they stay untracked on purpose).
        size_t index{};
        u64 mask{};
        if (const Manager* const manager = SingleWord(query_cpu_addr, query_size, index, mask)) {
            return manager->IsWordCpuClean(index, mask, true);
        }
        std::size_t remaining_size{query_size};
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            const Manager* const manager{top_tier[page_index]};
            if (manager == nullptr ||
                !manager->IsRegionCpuClean(page_offset, copy_amount, true)) {
                return false;
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return true;
    }

    /// Returns true if a region has been modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        size_t index{};
        u64 mask{};
        if (const Manager* const manager = SingleWord(query_cpu_addr, query_size, index, mask)) {
            return manager->IsWordModified(Type::GPU, index, mask);
        }
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::GPU, offset, size);
        });
    }

    /// Returns true if a region has been marked as Preflushable
    [[nodiscard]] bool IsRegionPreflushable(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::Preflushable, offset, size);
        });
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        // The GPU thread's own writes (constant buffer data, before nearly every draw) mostly land
        // on pages that are CPU modified and untracked already: nothing to walk or unprotect.
        size_t index{};
        u64 mask{};
        Manager* const manager = SingleWord(dirty_cpu_addr, query_size, index, mask);
        if (manager != nullptr && manager->MarkWordCpuModifiedNoChange(index, mask)) {
            return;
        }
        IteratePages<true>(dirty_cpu_addr, query_size, [this](Manager* manager, u64 offset, size_t size) {
            manager->hot_pages_enabled = hot_pages_enabled;
            manager->ChangeRegionState(Type::CPU, true, manager->cpu_addr + offset, size);
            ListManager(manager);
        });
    }

    /// Unmark region as CPU modified, notifying the device_tracker about this change
    void UnmarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<true>(dirty_cpu_addr, query_size, [this](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::CPU, false, manager->cpu_addr + offset, size);
            ListManager(manager);
        });
    }

    /// Mark region as modified from the host GPU
    void MarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::GPU, true, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the host GPU
    void MarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, true, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::GPU, false, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, false, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the CPU
    /// but don't mark it as modified until FlusHCachedWrites is called.
    void CachedCpuWrite(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<true>(dirty_cpu_addr, query_size, [this](Manager* manager, u64 offset, size_t size) {
            const VAddr cpu_address = manager->cpu_addr + offset;
            manager->ChangeRegionState(Type::CachedCPU, true, cpu_address, size);
            cached_pages.insert(u32(cpu_address >> HIGHER_PAGE_BITS));
        });
    }

    /// Flushes cached CPU writes, and notify the device_tracker about the deltas
    void FlushCachedWrites(VAddr query_cpu_addr, u64 query_size) noexcept {
        IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, [[maybe_unused]] u64 offset, [[maybe_unused]] size_t size) {
            manager->FlushCachedWrites();
        });
    }

    void FlushCachedWrites() noexcept {
        for (auto id : cached_pages) {
            top_tier[id]->FlushCachedWrites();
        }
        cached_pages.clear();
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    template <typename Func>
    void ForEachUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<true>(query_cpu_range, query_size, [this, &func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + offset, size, func);
            ListManager(manager);
        });
    }

    /// Call 'func' for each CPU modified range, leaving those pages marked
    template <typename Func>
    void ForEachCpuModifiedRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<true>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::CPU, false, manager->cpu_addr + offset, size, func);
        });
    }

    /// The guest may have written memory the work that follows reads: hot pages (pages the guest
    /// rewrites after every upload, left without write protection) must be uploaded again before
    /// their next use.
    void ReArmHotPages() noexcept {
        // Hot pages lose their CPU modified mark only to uploads, which list their manager: the
        // others have nothing to re-arm (this runs at every list submit and met acquire).
        for (Manager* const manager : rearm_managers) {
            manager->rearm_pending = false;
            manager->in_rearm_list = false;
            manager->ReArmHotPages();
        }
        rearm_managers.clear();
    }

    /// Once a frame: ages the uploads that make pages hot. Every DEMOTE_FRAMES frames, and while
    /// hot pages are off, hot pages are protected again at their next upload; the ones the guest
    /// still rewrites soon after it become hot again.
    void TickFrame(bool enable_hot_pages) noexcept {
        std::erase_if(recent_managers, [](Manager* manager) {
            manager->in_recent_list = manager->AgeRecentUploads();
            return !manager->in_recent_list;
        });
        for (Manager* const manager : hot_managers) {
            manager->ClearHotUploads();
        }
        hot_pages_enabled = enable_hot_pages;
        if (++frames_since_demote >= DEMOTE_FRAMES || !hot_pages_enabled) {
            frames_since_demote = 0;
            for (Manager* const manager : hot_managers) {
                manager->DemoteHotPages();
                manager->in_hot_list = false;
            }
            hot_managers.clear();
        }
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <typename Func>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, bool clear, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func, clear](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, clear, manager->cpu_addr + offset, size, func);
        });
    }

    template <typename Func>
    void ForEachDownloadRangeAndClear(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, true, manager->cpu_addr + offset, size, func);
        });
    }

private:
    static constexpr u32 DEMOTE_FRAMES = 300;

    void ListManager(Manager* manager) {
        if (manager->has_hot) {
            manager->has_hot = false;
            if (!manager->in_hot_list) {
                manager->in_hot_list = true;
                hot_managers.push_back(manager);
            }
        }
        if (manager->has_recent) {
            manager->has_recent = false;
            if (!manager->in_recent_list) {
                manager->in_recent_list = true;
                recent_managers.push_back(manager);
            }
        }
        if (manager->rearm_pending && !manager->in_rearm_list) {
            manager->in_rearm_list = true;
            rearm_managers.push_back(manager);
        }
    }

    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, Manager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    template <bool create_region_on_fail, typename Func>
    std::pair<u64, u64> IteratePairs(VAddr cpu_address, size_t size, Func&& func) {
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        u64 begin = (std::numeric_limits<u64>::max)();
        u64 end = 0;
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            const auto execute = [&] {
                auto [new_begin, new_end] = func(manager, page_offset, copy_amount);
                if (new_begin != 0 || new_end != 0) {
                    const u64 base_address = page_index << HIGHER_PAGE_BITS;
                    begin = (std::min)(new_begin + base_address, begin);
                    end = (std::max)(new_end + base_address, end);
                }
            };
            if (manager) {
                execute();
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                execute();
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        if (begin < end) {
            return std::make_pair(begin, end);
        } else {
            return std::make_pair(0ULL, 0ULL);
        }
    }

    /// The manager holding all of a range that lies inside one of its words (a draw's ranges and the
    /// constant buffer writes nearly always do), that word's index and the range's bits in it.
    /// nullptr when the range spans words or its manager doesn't exist yet.
    Manager* SingleWord(VAddr address, u64 size, size_t& index, u64& mask) const noexcept {
        const VAddr last = address + size - 1;
        if (size == 0 || address / BYTES_PER_WORD != last / BYTES_PER_WORD) {
            return nullptr;
        }
        index = (address & HIGHER_PAGE_MASK) / BYTES_PER_WORD;
        const u64 first_page = (address % BYTES_PER_WORD) / BYTES_PER_PAGE;
        const u64 last_page = (last % BYTES_PER_WORD) / BYTES_PER_PAGE;
        mask = (~u64{0} >> (PAGES_PER_WORD - 1 - last_page)) & (~u64{0} << first_page);
        return top_tier[address >> HIGHER_PAGE_BITS];
    }

    void CreateRegion(std::size_t page_index) {
        const VAddr base_cpu_addr = page_index << HIGHER_PAGE_BITS;
        top_tier[page_index] = GetNewManager(base_cpu_addr);
    }

    Manager* GetNewManager(VAddr base_cpu_address) {
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                new (&last_pool[i]) Manager(0, *device_tracker);
                free_managers.push_back(&last_pool[i]);
            }
        }
        Manager* new_manager = free_managers.front();
        new_manager->cpu_addr = base_cpu_address;
        free_managers.pop_front();
        return new_manager;
    }

    std::array<Manager*, NUM_HIGH_PAGES> top_tier{};
    std::deque<std::array<Manager, MANAGER_POOL_SIZE>> manager_pool;
    std::deque<Manager*> free_managers;
    ::Common::unordered_set<u32> cached_pages;
    std::vector<Manager*> hot_managers;
    std::vector<Manager*> recent_managers;
    std::vector<Manager*> rearm_managers; ///< Hot pages uploaded since the last ReArmHotPages
    u32 frames_since_demote = 0;
    bool hot_pages_enabled = false; ///< Until TickFrame turns them on
    DeviceTracker* device_tracker = nullptr;
};

} // namespace VideoCommon
