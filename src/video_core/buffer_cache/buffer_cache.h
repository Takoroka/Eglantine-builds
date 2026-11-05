// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/nx2_ab.h"
#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_profile.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() = default;

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    // Hot pages (MemoryTrackerBase::ReArmHotPages) for Switch 2 titles; EDEN_NX2_HOT_PAGES=0 turns
    // them off (comparisons).
    static const bool hot_pages_allowed = [] {
        const char* value = std::getenv("EDEN_NX2_HOT_PAGES");
        return value == nullptr || value[0] != '0';
    }();
    memory_tracker.TickFrame(hot_pages_allowed && Tegra::g_nx2_ampere.load(std::memory_order_relaxed));
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    std::copy_n(channel_state->uniform_cache_hits.begin(),
                channel_state->uniform_cache_hits.size() - 1,
                channel_state->uniform_cache_hits.begin() + 1);
    std::copy_n(channel_state->uniform_cache_shots.begin(),
                channel_state->uniform_cache_shots.size() - 1,
                channel_state->uniform_cache_shots.begin() + 1);
    channel_state->uniform_cache_hits[0] = 0;
    channel_state->uniform_cache_shots[0] = 0;

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        if (buffer_bytes >= 512_MiB) {
            RunGarbageCollector(buffer_bytes >= 1_GiB);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data) {
    if (!keep_page_gpu_data) {
        // The memory is going away: what its pages held says nothing about the next mapping.
        native_tail_unknown.Subtract(device_addr, size);
        native_tail_output.Subtract(device_addr, size);
        native_tail_guest.Subtract(device_addr, size);
    }
    Nx2NoteListHostWrite(device_addr, size);
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    // Only the scheduler's owner may download (it records and submits): unmaps and other writes
    // reach this from CPU and service threads too, and submitting there raced the GPU thread's
    // own submissions (crashes in Scheduler::SubmitExecution). Those keep the old behaviour;
    // guest CPU writes come through TryCacheInvalidation, which sends such pages to the GPU thread.
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && VideoCommon::GpuSplit::OwnsScheduler() &&
        gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
void BufferCache<P>::WriteMemories(std::span<const std::pair<DAddr, std::size_t>> writes) {
    // Guest writes to GPU-written pages arrive here together, in pieces of 64-byte blocks.
    // WriteMemory one piece at a time downloaded the GPU-written rest of the piece's pages, and
    // with it the cache's older copy of the pieces still to come on them: those writes were lost
    // (Splatoon Raiders' characters, where native compute bindings mark the CPU's data
    // GPU-written). So every piece leaves the GPU-written ranges first, and the pages download
    // after, in one batch.
    for (const auto& [device_addr, size] : writes) {
        if (device_addr == 0 || size == 0) {
            continue;
        }
        Nx2NoteListHostWrite(device_addr, size);
        if (IsRegionGpuModified(device_addr, size)) {
            ClearDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
    Common::RangeSet<DAddr> pages;
    for (const auto& [device_addr, size] : writes) {
        if (device_addr == 0 || size == 0) {
            continue;
        }
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        if (gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
            pages.Add(page_begin, page_end - page_begin);
        }
    }
    DownloadRanges(pages);
    for (const auto& [device_addr, size] : writes) {
        if (device_addr == 0 || size == 0) {
            continue;
        }
        memory_tracker.MarkRegionAsCpuModified(device_addr, size);
        MarkVertexBuffersStale(device_addr, size);
    }
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    if (native_tail_unknown.Intersects(device_addr_start_aligned,
                                       device_addr_end_aligned - device_addr_start_aligned)) {
        LearnNativeTail(device_addr_start_aligned, device_addr_end_aligned);
    }
    if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
        area->preemtive = true;
        return area;
    };
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // Mark only data that the GPU wrote. A mark stays for the session and sends later GPU writes
    // in the range to a download at each fence.
    if (!area->preemtive) {
        // Each such read waits for a GPU-thread round trip, and the area used to be the read's own
        // pages: a guest walking data the GPU had just written (title 1 at its title load, 1,400
        // reads) paid one per 4 KiB page. The flush takes the read-trapped run around the read, up
        // to EDEN_NX2_READ_AREA_KIB (2048; 0 = the read's pages only), a quarter of it before.
        static const u64 max_area = [] {
            const char* value = std::getenv("EDEN_NX2_READ_AREA_KIB");
            return (value != nullptr ? std::strtoull(value, nullptr, 10) : 2048ULL) * 1024ULL;
        }();
        if (max_area != 0) {
            constexpr DAddr window = DAddr{64} << 20;
            const DAddr search = device_addr_start_aligned > window
                                     ? device_addr_start_aligned - window
                                     : 0;
            DAddr trapped_lo = 0;
            DAddr trapped_hi = 0;
            read_trapped_ranges.ForEachInRange(search, 2 * window, [&](DAddr lower, DAddr upper) {
                if (lower <= device_addr && device_addr < upper) {
                    trapped_lo = lower;
                    trapped_hi = upper;
                }
            });
            if (trapped_hi > trapped_lo) {
                const DAddr before = Common::AlignDown(max_area / 4, Core::DEVICE_PAGESIZE);
                const DAddr lo = device_addr_start_aligned > before
                                     ? device_addr_start_aligned - before
                                     : 0;
                area->start_address =
                    (std::min)(device_addr_start_aligned, (std::max)(lo, trapped_lo));
                area->end_address =
                    (std::max)(device_addr_end_aligned, (std::min)(lo + max_area, trapped_hi));
            }
        }
        memory_tracker.MarkRegionAsPreflushable(area->start_address,
                                                area->end_address - area->start_address);
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        // Each buffer downloads its own part only: given the whole range, the first buffer took
        // the GPU-written ranges of the buffers after it as its own, copying from offsets past
        // its end (VK_ERROR_DEVICE_LOST) and clearing them for their own buffers. Flush areas
        // used to be a page or two; CPU read flushes now span the read-trapped run around the read.
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        if (DownloadsAtFence(new_base_address, size)) {
            uncommitted_gpu_modified_ranges.Add(new_base_address, size);
        } else {
            ArmReadTraps(new_base_address, size);
        }
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        AddGpuModifiedRange(pair.first, pair.second);
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
            Nx2NoteListWrite(buffer_b, pair.first, pair.second);
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, amount);
    if (!dest_dirty) {
        return false;
    }

    const size_t size = amount * sizeof(u32);
    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    Binding& current = channel_state->uniform_buffers[stage][index];
    u64& generation = channel_state->uniform_buffer_map_generations[stage][index];
    // Before translating: a map that lands meanwhile leaves the binding to be translated again.
    const u64 map_generation = gpu_memory->MapGeneration();
    // EDEN_NX2_CB_BIND_SKIP=0 translates every bind (comparisons).
    static const bool skip_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CB_BIND_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    // Drivers bind most slots again for every draw, at the same address: while the page tables
    // stay the same, the binding (and the buffer it resolved to) still holds.
    if (skip_enabled && current.gpu_addr == gpu_addr && current.size == size &&
        current.device_addr != 0 && generation == map_generation) {
        return;
    }
    generation = map_generation;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        current = NULL_BINDING;
        return;
    }
    current = Binding{
        .device_addr = *device_addr,
        .gpu_addr = gpu_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
bool BufferCache<P>::RefreshGraphicsBuffers(bool is_indexed) {
    channel_state->has_deleted_buffers = false;
    if (is_indexed) {
        UpdateIndexBuffer();
    }
    UpdateVertexBuffers();
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        UpdateUniformBuffers(stage);
    }
    return !channel_state->has_deleted_buffers;
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    BindHostGraphicsUniformBuffers(stage);
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written) {
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= 1U << ssbo_index;
    channel_state->written_storage_buffers[stage] |= (is_written ? 1U : 0U) << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    const GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written);
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= 1U << ssbo_index;
    channel_state->written_compute_storage_buffers |= (is_written ? 1U : 0U) << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    const GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    // The storage buffer's address and size as the shader reads them: the same words under the same
    // page tables give the binding of last time.
    std::array<u64, 2> words;
    gpu_memory->ReadBlockUnsafe(ssbo_addr, words.data(), sizeof(words));
    const size_t as_id = gpu_memory->GetID();
    const u64 map_generation = gpu_memory->MapGeneration();
    const bool native_pointer = kepler_compute->IsNativeMode();
    const bool dlss_dispatch = kepler_compute->IsDlssDispatch();
    StorageBindingSource& source =
        compute_storage_binding_sources[((ssbo_addr >> 4) ^ (as_id << 5)) %
                                        compute_storage_binding_sources.size()];
    if (!source.valid || source.ssbo_addr != ssbo_addr || source.as_id != as_id ||
        source.words != words || source.map_generation != map_generation ||
        source.is_written != is_written || source.native_pointer != native_pointer ||
        source.dlss_dispatch != dlss_dispatch) {
        source = StorageBindingSource{
            .as_id = as_id,
            .ssbo_addr = ssbo_addr,
            .words = words,
            .map_generation = map_generation,
            .is_written = is_written,
            .native_pointer = native_pointer,
            .dlss_dispatch = dlss_dispatch,
            .valid = true,
            .binding = words[0] == 0
                           ? NULL_BINDING
                           : StorageBufferBinding(words[0], words[1], cbuf_index, is_written,
                                                  native_pointer, dlss_dispatch),
        };
    }
    channel_state->compute_storage_buffers[ssbo_index] = source.binding;
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        // Only what no write replaced since the copy was taken: the rest is older than guest
        // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
        ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
            device_memory.WriteBlockUnsafe(start, &read_mapped_memory[start - device_addr],
                                           end - start);
            gpu_modified_ranges.Subtract(start, end - start);
        });
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    if (size == 0) {
        return false;
    }
    // Even that lookup walks a tree of every GPU-written range, before nearly every draw (the
    // constant buffer writes FlushCaching invalidates), mostly in the same few areas: an area
    // known to hold none answers until a GPU-written range is added there.
    const DAddr end = addr + size;
    for (const auto& [gap_begin, gap_end] : gpu_unmodified_gaps) {
        if (gap_begin <= addr && end <= gap_end) {
            return false;
        }
    }
    DAddr gap_begin{};
    DAddr gap_end{};
    if (!gpu_modified_ranges.FreeGap(addr, size, gap_begin, gap_end)) {
        return true;
    }
    gpu_unmodified_gaps[next_gpu_unmodified_gap] = {gap_begin, gap_end};
    next_gpu_unmodified_gap = (next_gpu_unmodified_gap + 1) % gpu_unmodified_gaps.size();
    return false;
}

template <class P>
void BufferCache<P>::AddGpuModifiedRange(DAddr device_addr, u64 size) {
    ++gpu_modified_serial;
    ++buffer_check_stats.adds;
    // Draws and dispatches write the same ranges over and over: one the set covers already changes
    // neither the set nor the known gaps (they lie outside it).
    if (Common::Nx2Ab::On(Common::Nx2Ab::BufferCheckSkip) &&
        gpu_modified_ranges.Covers(device_addr, size)) {
        ++buffer_check_stats.covered;
        return;
    }
    gpu_modified_ranges.Add(device_addr, size);
    // Ranges only leave the set otherwise: known gaps stay gaps until a range is added in them.
    for (auto& gap : gpu_unmodified_gaps) {
        if (device_addr < gap.second && gap.first < device_addr + size) {
            gap = {};
        }
    }
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            if (!SynchronizeBuffer(buffer, binding.device_addr, binding.size)) {
                // New data under a repacked binding (hot pages are uploaded again without the
                // write trap that marks it stale): assemble the copy again.
                if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
                    if (runtime.IsVertexRepacked(index)) {
                        flags[Dirty::VertexBuffer0 + index] = true;
                    }
                }
            }
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
        // The same bytes as at the previous draw, which streamed them (uniform_stream_reuse.h).
        if (binding.buffer_id != NULL_BUFFER_ID &&
            VideoCommon::KeepSmallUniform(runtime, device_memory, stage, index, device_addr, size)) {
            channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
            channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
            return;
        }
    }
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !memory_tracker.IsRegionGpuModified(device_addr, size));
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                streamed = VideoCommon::BindSmallUniformReusing(
                    runtime, device_memory, stage, index, binding_index, device_addr, size);
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        }
        return;
    }
    // Classic cached path
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        // A range found in one piece stays so while the page tables do (unmaps reach the virtual
        // range cache from the page table operations): most storage buffers are, and asking the
        // cache for each of every draw and dispatch cost ~1.5% of the GPU thread.
        const size_t as_id = gpu_memory->GetID();
        const u64 map_generation = gpu_memory->MapGeneration();
        SingleRange& known = single_ranges[((binding.gpu_addr >> 6) ^ binding.size ^ (as_id << 7)) %
                                           single_ranges.size()];
        if (known.gpu_addr == binding.gpu_addr && known.size == binding.size &&
            known.as_id == as_id && known.map_generation == map_generation) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            known = SingleRange{
                .as_id = as_id,
                .gpu_addr = binding.gpu_addr,
                .size = binding.size,
                .map_generation = map_generation,
            };
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first + binding.segment_count > pool.size()) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            if (SynchronizeBuffer(buffer, segment.device_addr, segment.size)) {
                runtime.InvalidateMultiRange(key);
            }
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written) {
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, segment.size);
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return runtime.BindMultiRangeStorageBuffer(key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, compute_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        cpu_written_tail.clear();
        if (is_written && binding.learn_tail && binding.written_size < size) {
            // The synchronization below uploads them and forgets which they were.
            memory_tracker.ForEachCpuModifiedRange(
                binding.device_addr + binding.written_size, size - binding.written_size,
                [this](u64 device_addr, u64 range_size) {
                    cpu_written_tail.emplace_back(device_addr, device_addr + range_size);
                });
        }
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            MarkWrittenComputeStorage(binding, cpu_written_tail);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            slot_buffers.erase(inline_buffer_id);
            ++buffer_set_serial;
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    channel_state->index_buffer = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = FindBuffer(*device_addr, size, false),
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    // A range resolved before, while the address space, its mappings and the buffers are as then,
    // binds what it bound (no translation, continuity check or buffer search). Not with accurate
    // fences, where finding the buffer also synchronizes its writes.
    // EDEN_NX2_VB_CACHE_VERIFY=1 resolves hits anyway and counts the differences.
    static const bool verify = std::getenv("EDEN_NX2_VB_CACHE_VERIFY") != nullptr;
    VertexLookup* const lookup =
        array.enable != 0 && gpu_addr_end > gpu_addr_begin &&
                Common::Nx2Ab::On(Common::Nx2Ab::VertexLookupCache) &&
                !Settings::IsGPUFenceBehaviorAccurate()
            ? &vertex_lookups[((gpu_addr_begin ^ (gpu_addr_end << 7)) * 0x9E3779B97F4A7C15ULL) >>
                              (64 - VertexLookupBits)]
            : nullptr;
    const u64 map_generation = gpu_memory->MapGeneration();
    const bool same_entry = lookup != nullptr && lookup->begin == gpu_addr_begin &&
                            lookup->end == gpu_addr_end &&
                            lookup->address_space == gpu_memory->GetID();
    const bool hit = same_entry && lookup->map_generation == map_generation &&
                     lookup->buffer_set_serial == buffer_set_serial;
    if (hit && !verify) {
        channel_state->vertex_buffers[index] = lookup->binding;
        UpdateVertexBufferSlot(index, lookup->binding);
        CountVertexLookup(VertexLookupOutcome::Hit);
        return;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0 || !device_addr) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    if (lookup != nullptr) {
        if (hit) {
            const bool differs = lookup->binding.device_addr != binding.device_addr ||
                                 lookup->binding.size != binding.size ||
                                 lookup->binding.buffer_id != binding.buffer_id;
            CountVertexLookup(differs ? VertexLookupOutcome::HitMismatch : VertexLookupOutcome::Hit);
        } else {
            CountVertexLookup(!same_entry                                ? VertexLookupOutcome::MissEntry
                              : lookup->map_generation != map_generation ? VertexLookupOutcome::MissMappings
                                                                         : VertexLookupOutcome::MissBuffers);
        }
        // The buffer set as it is now: FindBuffer may have created the buffer.
        *lookup = {gpu_addr_begin, gpu_addr_end, gpu_memory->GetID(), map_generation,
                   buffer_set_serial, binding};
    }
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
void BufferCache<P>::CountVertexLookup(VertexLookupOutcome outcome) {
    // EDEN_NX2_VB_CACHE_STATS=1 (or the verify): hits, misses by reason and mismatches every 10 s.
    static const bool enabled = std::getenv("EDEN_NX2_VB_CACHE_STATS") != nullptr ||
                                std::getenv("EDEN_NX2_VB_CACHE_VERIFY") != nullptr;
    if (!enabled) {
        return;
    }
    static std::array<u64, 5> counts{};
    static auto last_log = std::chrono::steady_clock::now();
    ++counts[static_cast<size_t>(outcome)];
    if (const auto now = std::chrono::steady_clock::now(); now - last_log >= std::chrono::seconds{10}) {
        LOG_INFO(HW_GPU,
                 "nx2 vertex lookups: last {:.0f} s {} hits ({} differing from a full lookup), misses: "
                 "{} other range in the entry, {} remapped, {} buffers created or deleted",
                 std::chrono::duration<double>(now - last_log).count(), counts[0] + counts[1],
                 counts[1], counts[2], counts[3], counts[4]);
        counts = {};
        last_log = now;
    }
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        const BufferId buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        binding.buffer_id = buffer_id;
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.Address());
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf.size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync) {
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    AddGpuModifiedRange(device_addr, size);
    if (DownloadsAtFence(device_addr, size)) {
        if (!Common::Nx2Ab::On(Common::Nx2Ab::BufferCheckSkip) ||
            !uncommitted_gpu_modified_ranges.Covers(device_addr, size)) {
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
        }
    } else {
        ArmReadTraps(device_addr, size);
    }
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
        Nx2NoteListWrite(buffer_id, device_addr, size);
    }
}

template <class P>
void BufferCache<P>::MarkWrittenComputeStorage(
    const Binding& binding, std::span<const std::pair<DAddr, DAddr>> cpu_written) {
    if (binding.written_size != 0) {
        MarkWrittenBuffer(binding.buffer_id, binding.device_addr, binding.written_size, true);
    }
    if (!binding.learn_tail) {
        return;
    }
    // Past written_size, a native binding runs on over whatever its mapping holds: GPU output,
    // but also the guest's own data (Splatoon Raiders binds 48-byte outputs that sit in 7.6 MiB
    // of it). Marked GPU-written, the guest's writes there are applied late, and until then the
    // cache uses its older copy (DMA sources, uploads): gray and black squares on Raiders'
    // characters. So these pages are marked only once the guest has read one back without
    // writing it (GPU output, such as Raiders' ink); until it's known what a page holds, its
    // first read is trapped (LearnNativeTail).
    const DAddr tail_begin =
        Common::AlignUp(binding.device_addr + binding.written_size, DEVICE_PAGESIZE);
    const DAddr tail_end = Common::AlignDown(binding.device_addr + binding.size, DEVICE_PAGESIZE);
    if (tail_begin >= tail_end) {
        return;
    }
    const u64 tail_size = tail_end - tail_begin;
    // Known pages the guest wrote since the last bind are its own data. (On a page's first bind,
    // that is just its first upload.)
    for (const auto& [cpu_begin, cpu_end] : cpu_written) {
        const DAddr begin = (std::max)(cpu_begin, tail_begin);
        const DAddr end = (std::min)(cpu_end, tail_end);
        if (begin >= end) {
            continue;
        }
        Common::RangeSet<DAddr> known;
        native_tail_unknown.ForEachInRange(begin, end - begin,
                                           [&](DAddr b, DAddr e) { known.Add(b, e - b); });
        native_tail_output.ForEachInRange(begin, end - begin,
                                          [&](DAddr b, DAddr e) { known.Add(b, e - b); });
        known.ForEach([this](DAddr b, DAddr e) {
            native_tail_guest.Add(b, e - b);
            native_tail_unknown.Subtract(b, e - b);
            native_tail_output.Subtract(b, e - b);
        });
    }
    native_tail_output.ForEachInRange(tail_begin, tail_size, [&](DAddr b, DAddr e) {
        MarkWrittenBuffer(binding.buffer_id, b, static_cast<u32>(e - b), true);
    });
    Common::RangeSet<DAddr> fresh;
    fresh.Add(tail_begin, tail_size);
    const auto known = [&](DAddr b, DAddr e) { fresh.Subtract(b, e - b); };
    native_tail_unknown.ForEachInRange(tail_begin, tail_size, known);
    native_tail_output.ForEachInRange(tail_begin, tail_size, known);
    native_tail_guest.ForEachInRange(tail_begin, tail_size, known);
    fresh.ForEach([this](DAddr b, DAddr e) {
        native_tail_unknown.Add(b, e - b);
        ArmReadTraps(b, e - b);
    });
}

template <class P>
void BufferCache<P>::LearnNativeTail(DAddr page_begin, DAddr page_end) {
    Common::RangeSet<DAddr> pages;
    native_tail_unknown.ForEachInRange(page_begin, page_end - page_begin,
                                       [&](DAddr b, DAddr e) { pages.Add(b, e - b); });
    pages.ForEach([this](DAddr begin, DAddr end) {
        native_tail_unknown.Subtract(begin, end - begin);
        for (DAddr page = begin; page < end; page += DEVICE_PAGESIZE) {
            if (memory_tracker.IsRegionCpuModified(page, DEVICE_PAGESIZE)) {
                native_tail_guest.Add(page, DEVICE_PAGESIZE);
                continue;
            }
            // Not written by the guest since its last upload: the buffer holds that data plus
            // what the GPU wrote. Download it now, and at fences from now on.
            native_tail_output.Add(page, DEVICE_PAGESIZE);
            memory_tracker.MarkRegionAsGpuModified(page, DEVICE_PAGESIZE);
            gpu_modified_ranges.Add(page, DEVICE_PAGESIZE);
        }
    });
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    if (!nx2_collect_list_copies) {
                        const u64 chunks = Nx2PageChunks(page_addr, device_addr, size);
                        ++entry.draw_writes;
                        entry.draw_chunks |= chunks;
                        const DAddr end = (std::min)(page_addr + (DAddr{1} << page_bits),
                                                     device_addr + size);
                        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.draw_writes);
                        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.draw_bytes,
                                            end - (std::max)(page_addr, device_addr));
                        if (!entry.draw_written) {
                            entry.draw_written = true;
                            LOG_INFO(HW_GPU,
                                     "Command list page {:#x} is written by draws too ({:#x} "
                                     "bytes at {:#x}, pieces {:#018x}): its lists drain the split",
                                     page_addr, size, device_addr, chunks);
                        }
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
            }
            ++page;
        }
    }
}

template <class P>
u32* BufferCache<P>::Nx2CopyMarker(const Nx2Staging& copy) {
    return reinterpret_cast<u32*>(copy.mapped_span.data() +
                                  (DAddr{1} << Tegra::Nx2WrittenPageBits));
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        // The word after each copy's page: the GPU sets it once the copy is done.
        boost::container::small_vector<std::pair<decltype(Nx2Staging::buffer), u64>, 8> markers;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size + sizeof(u32), true);
            *Nx2CopyMarker(staging) = 0;
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            markers.emplace_back(staging.buffer, staging.offset + page_size);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_draw_writes = entry.draw_writes;
            entry.copy_tick = tick;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        runtime.Nx2WriteCopyMarkers(std::span{markers.data(), markers.size()});
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out) -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        // How the list's words are read without draining: as the page's (no draw writes it), or
        // past the draws that do (nx2.h, Nx2ListDrawRanges).
        enum class Use { None, Page, Exact };
        const u64 list_chunks = Nx2PageChunks(page, device_addr, out.size());
        const auto past_draws = [&](const Nx2ListPage& entry) {
            return Tegra::Nx2ListDrawRanges() && (entry.draw_chunks & list_chunks) == 0
                       ? Use::Exact
                       : Use::None;
        };
        const auto copy_use = [&](const Nx2ListPage& entry) {
            if (entry.draw_written) {
                return entry.CopyPastDraws(offset, out.size()) ? past_draws(entry) : Use::None;
            }
            return entry.copy && entry.copy_writes == entry.writes &&
                           offset >= entry.copy_begin && offset + out.size() <= entry.copy_end
                       ? Use::Page
                       : Use::None;
        };
        const auto count = [](Use use, bool copy) {
            auto& stats = Tegra::g_nx2_list_stats;
            Tegra::Nx2CountList(use == Use::Exact ? stats.exact
                                : copy            ? stats.copied
                                                  : stats.synced);
        };
        Use use{};
        u64 tick{};
        u64 writes{};
        const u32* marker{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            Nx2ListPage& entry = it->second;
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            bool synced = false;
            if (!entry.draw_written) {
                synced = entry.synced_writes == entry.writes;
                use = synced ? Use::Page : copy_use(entry);
            } else if (entry.SyncedPastDraws()) {
                synced = true;
                use = past_draws(entry);
            } else {
                use = copy_use(entry);
            }
            if (use == Use::None) {
                return Nx2ListRead::Unknown;
            }
            if (synced) {
                count(use, false);
                return Nx2ListRead::Synced;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            marker = Nx2CopyMarker(*entry.copy);
            if (runtime.IsFree(entry.copy_tick) || runtime.Nx2ListCopyMarked(marker)) {
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                count(use, true);
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes). The
        // copy stays mapped if the back replaces it meanwhile; the checks below catch that.
        const auto wait_start = std::chrono::steady_clock::now();
        runtime.Nx2WaitListCopy(marker, tick);
        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.copy_ns,
                            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - wait_start)
                                                 .count()));
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || copy_use(it->second) == Use::None ||
            it->second.copy_tick != tick || it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        count(use, true);
        return Nx2ListRead::Copied;
    }
}

template <class P>
u64 BufferCache<P>::Nx2PageChunks(DAddr page, DAddr begin, u64 size) {
    constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
    static_assert(page_size / 64 == 64, "one bit per 64 bytes of a page");
    const DAddr first = (std::max)(page, begin);
    const DAddr end = (std::min)(page + page_size, begin + size);
    if (first >= end) {
        return 0;
    }
    const u64 low = (first - page) / 64;
    const u64 pieces = (end - page - 1) / 64 - low + 1;
    return (pieces == 64 ? ~u64{0} : (u64{1} << pieces) - 1) << low;
}

template <class P>
void BufferCache<P>::Nx2VerifyListRead(DAddr device_addr, std::span<const u8> words,
                                       std::span<const u8> before) {
    if constexpr (!IS_OPENGL) {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (words.empty() || offset + words.size() > page_size || before.size() != words.size()) {
            return;
        }
        auto& stats = Tegra::g_nx2_list_stats;
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !it->second.draw_written) {
            return;
        }
        Nx2ListPage& entry = it->second;
        Tegra::Nx2CountList(stats.draw_page_reads);
        if ((entry.draw_chunks & Nx2PageChunks(page, device_addr, words.size())) == 0) {
            Tegra::Nx2CountList(stats.would_exact);
        }
        // What Nx2ReadListCopy reads past the draws: guest memory as it was before the drain, or
        // the copy. The split is drained and the page flushed, so the GPU is done with the copy.
        const u8* read = nullptr;
        const char* what = nullptr;
        bool compared = false;
        const auto compare = [&](const u8* data, const char* name) {
            compared = true;
            if (read == nullptr && std::memcmp(data, words.data(), words.size()) != 0) {
                read = data;
                what = name;
            }
        };
        if (entry.SyncedPastDraws()) {
            compare(before.data(), "guest memory");
        }
        if (entry.CopyPastDraws(offset, words.size()) &&
            (runtime.IsFree(entry.copy_tick) ||
             runtime.Nx2ListCopyMarked(Nx2CopyMarker(*entry.copy)))) {
            compare(entry.copy->mapped_span.data() + offset, "the copy");
        }
        if (!compared) {
            Tegra::Nx2CountList(stats.no_copy);
            return;
        }
        if (read == nullptr) {
            Tegra::Nx2CountList(stats.matched);
            return;
        }
        Tegra::Nx2CountList(stats.mismatched);
        size_t first = 0;
        while (read[first] == words[first]) {
            ++first;
        }
        if (!entry.mismatched) {
            entry.mismatched = true;
            LOG_INFO(HW_GPU,
                     "Command list page {:#x}: draws changed list words ({:#x} bytes at +{:#x}, "
                     "first at +{:#x}, {} was stale, draw pieces {:#018x})",
                     page, words.size(), offset, offset + first, what, entry.draw_chunks);
        }
        // What the draws write into the lists (the first 32 changes, up to 8 words each).
        static u32 dumped{};
        if (dumped < 32) {
            ++dumped;
            std::array<u32, 8> was{};
            std::array<u32, 8> now{};
            const size_t count = (std::min)(words.size() / sizeof(u32), was.size());
            std::memcpy(was.data(), read, count * sizeof(u32));
            std::memcpy(now.data(), words.data(), count * sizeof(u32));
            LOG_INFO(HW_GPU, "Command list at {:#x} ({:#x} bytes) was {:08x}, now {:08x}",
                     device_addr, words.size(), fmt::join(was.begin(), was.begin() + count, " "),
                     fmt::join(now.begin(), now.begin() + count, " "));
        }
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    if constexpr (!IS_OPENGL) {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        Common::RangeSet<DAddr> ranges;
        for (const DAddr page : pages) {
            ranges.Add(page, page_size);
        }
        u64 others = 0;
        {
            std::scoped_lock lock{nx2_list_mutex};
            for (const auto& [page, entry] : nx2_list_pages) {
                if (entry.draw_written && IsRegionGpuModified(page, page_size)) {
                    ranges.Add(page, page_size);
                    ++others;
                }
            }
        }
        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.batch_flushes);
        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.batch_pages, others);
        DownloadRanges(ranges);
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
        entry.synced_other_writes = entry.writes - entry.draw_writes;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page << page_bits);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            const bool synced_past_draws = entry.SyncedPastDraws();
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
            if (synced_past_draws) {
                entry.synced_other_writes = entry.writes - entry.draw_writes;
            }
            // The 64-byte pieces this write covers whole hold its bytes now, not the draws'
            // (until a draw writes them again).
            const DAddr page_addr = page << page_bits;
            const DAddr whole_begin =
                Common::AlignUp((std::max)(device_addr, page_addr), DAddr{64});
            const DAddr whole_end = Common::AlignDown(
                (std::min)(device_addr + size, page_addr + (DAddr{1} << page_bits)), DAddr{64});
            if (whole_begin < whole_end) {
                entry.draw_chunks &=
                    ~Nx2PageChunks(page_addr, whole_begin, whole_end - whole_begin);
            }
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
void BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    runtime.CopyBuffer(new_buffer, overlap, copies, true);
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    ++buffer_set_serial;
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    // Draws bind the same buffers many times a frame: touch the LRU item (a lookup in its pool)
    // only the first time in each frame.
    if (buffer_id != NULL_BUFFER_ID && buffer.TakeLRUTouch(frame_tick)) {
        lru_cache.Touch(buffer.getLRUID(), frame_tick);
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    // Most binds find nothing to upload. Look at the range's tracking bits first: walking it for
    // upload ranges (and collecting pages to track again) was ~2% of Mario Kart World's split GPU
    // back thread even where nothing changed.
    if (IsRangeCpuClean(device_addr, size)) {
        return true;
    }
    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
bool BufferCache<P>::IsRangeCpuClean(DAddr device_addr, u32 size) {
    // A range inside one tracking word is a single word check already.
    if (size == 0 || device_addr / BYTES_PER_WORD == (device_addr + size - 1) / BYTES_PER_WORD) {
        return memory_tracker.IsRegionCpuClean(device_addr, size);
    }
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_BUFCHECK_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    static const bool stats = [] {
        const char* value = std::getenv("EDEN_NX2_BUFCHECK_STATS");
        return verify || (value != nullptr && value[0] == '1');
    }();
    BufferCheckStats& counts = buffer_check_stats;
    ++counts.checks;
    if (stats && (++counts.calls & 4095) == 0) {
        static auto last_log = std::chrono::steady_clock::now();
        static BufferCheckStats last{};
        if (const auto now = std::chrono::steady_clock::now();
            now - last_log >= std::chrono::seconds{30}) {
            LOG_INFO(HW_GPU,
                     "nx2 bufcheck (skip {}): last 30 s {} multi-word clean checks, {} from the "
                     "table, {} verify mismatches; {} written ranges, {} already covered",
                     Common::Nx2Ab::On(Common::Nx2Ab::BufferCheckSkip) ? "on" : "off",
                     counts.checks - last.checks, counts.skipped - last.skipped,
                     counts.mismatches - last.mismatches, counts.adds - last.adds,
                     counts.covered - last.covered);
            last = counts;
            last_log = now;
        }
    }
    if (!Common::Nx2Ab::On(Common::Nx2Ab::BufferCheckSkip)) {
        return memory_tracker.IsRegionCpuClean(device_addr, size);
    }
    if (const u32 epoch = Common::Nx2Ab::Epoch(); epoch != clean_ranges_epoch) {
        clean_ranges_epoch = epoch;
        clean_ranges.fill({});
    }
    CleanRange& entry =
        clean_ranges[((device_addr >> 8) ^ (device_addr >> 20) ^ size) % clean_ranges.size()];
    const u64 generation = memory_tracker.DirtyGeneration(device_addr, size);
    if (entry.device_addr == device_addr && entry.size == size && entry.generation == generation) {
        ++counts.skipped;
        if (verify && !memory_tracker.IsRegionCpuClean(device_addr, size)) {
            ++counts.mismatches;
            entry = {};
            return false;
        }
        return true;
    }
    if (!memory_tracker.IsRegionCpuClean(device_addr, size)) {
        return false;
    }
    entry = {device_addr, size, generation};
    return true;
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        device_memory.WriteBlockUnsafe(
            device_addr, mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadRanges(const Common::RangeSet<DAddr>& ranges) {
    if (ranges.Empty()) {
        return;
    }
    if constexpr (USE_MEMORY_MAPS) {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Download, 16> downloads;
        u64 total_size_bytes = 0;
        ranges.ForEach([&](DAddr range_begin, DAddr range_end) {
            ForEachBufferInRange(range_begin, range_end - range_begin,
                                 [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(range_begin, buffer_addr);
                const DAddr end = (std::min)(range_end, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr stop) {
                                downloads.push_back({buffer_id, BufferCopy{
                                                                    .src_offset = start - buffer_addr,
                                                                    .dst_offset = total_size_bytes,
                                                                    .size = stop - start,
                                                                }});
                                total_size_bytes += Common::AlignUp(stop - start, 64);
                            });
                        ClearDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        });
        if (total_size_bytes == 0) {
            return;
        }
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        for (Download& download : downloads) {
            download.copy.dst_offset += download_staging.offset;
            Buffer& buffer = slot_buffers[download.buffer_id];
            buffer.MarkUsage(download.copy.src_offset, download.copy.size);
            runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
        }
        runtime.Finish();
        const u8* const mapped_memory = download_staging.mapped_span.data();
        for (const Download& download : downloads) {
            const DAddr device_addr =
                slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
            device_memory.WriteBlockUnsafe(
                device_addr, mapped_memory + (download.copy.dst_offset - download_staging.offset),
                download.copy.size);
        }
    } else {
        ranges.ForEach([this](DAddr begin, DAddr end) { DownloadMemory(begin, end - begin); });
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }

    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            device_memory.WriteBlockUnsafe(copy_device_addr, copy_mapped_memory, copy.size);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            device_memory.WriteBlockUnsafe(copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    ++buffer_set_serial;
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index,
                                             bool is_written, bool native_pointer,
                                             bool dlss_dispatch) const {
    const GPUVAddr gpu_addr = gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }
    return StorageBufferBinding(gpu_addr, gpu_memory->Read<u64>(ssbo_addr + 8), cbuf_index,
                                is_written, native_pointer, dlss_dispatch);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr gpu_addr, u64 next_qword, u32 cbuf_index,
                                             bool is_written, bool native_pointer,
                                             bool dlss_dispatch) const {
    const u32 memory_layout_size = static_cast<u32>(gpu_memory->GetMemoryLayoutSize(gpu_addr));
    const u32 mapped_size = (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
    const u32 packed_size = static_cast<u32>(next_qword);
    const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                    packed_size != 0 && packed_size <= memory_layout_size;
    // A native (Switch 2) compute program's pointer is bare: the word after it may be another
    // parameter (a dimension, stride or texture handle), not an NVN buffer size. Taking it as one
    // bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest, so a DLSS
    // pass is bound to the mapping (up to 8 MiB). So is a buffer in its own mapping (at most
    // 2 MiB from the pointer): Splatoon Raiders' ink buffers have their output past the NVN-style
    // size. One in a large pool keeps the NVN-style size when the word looks like one: past it
    // is the guest's other data (Raiders' 48-byte outputs, 7.6 MiB of it), and binding all of it
    // slowed Hyrule Warriors: Age of Imprisonment's fights to a few fps.
    const bool own_mapping = memory_layout_size <= 2_MiB;
    const bool bind_mapping = native_pointer && (own_mapping || dlss_dispatch);
    u32 size = bind_mapping || !next_qword_is_size ? mapped_size : packed_size;
    // The mapping usually runs on past the buffer, so past the NVN-style size a written binding
    // in its own mapping marks only pages known to be GPU output (MarkWrittenComputeStorage). A
    // DLSS pass marks all of it.
    const u32 written_limit =
        native_pointer && !dlss_dispatch && next_qword_is_size ? packed_size : size;
    if (native_pointer && size != 0) {
        // Stop where the mapping or its device range does: the binding is one device range, and
        // a written one isn't split into separate ranges without sparse sources.
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
    }

    // Alignment only applies to the offset of the buffer
    const u32 alignment = runtime.GetStorageBufferAlignment();
    const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
    const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

    const std::optional<DAddr> aligned_device_addr = gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
    if (!aligned_device_addr || size == 0) {
        LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
        return NULL_BINDING;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
               cbuf_index);
    // The end address used for size calculation does not need to be aligned
    const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

    u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
    if (is_written) {
        binding_size = aligned_size;
    }
    const Binding binding{
        .device_addr = *aligned_device_addr,
        .gpu_addr = aligned_gpu_addr,
        .size = binding_size,
        .buffer_id = BufferId{},
        .written_size = (std::min)(binding_size, static_cast<u32>(gpu_addr - aligned_gpu_addr) +
                                                     (std::min)(written_limit, size)),
        .learn_tail = native_pointer && own_mapping && !dlss_dispatch,
    };
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
