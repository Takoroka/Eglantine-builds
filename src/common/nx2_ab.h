// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TEMP (measurement aid): A/B of optimizations, for the lichii idea branches on the play build.
// EDEN_NX2_AB=<hex mask> picks the optimizations (bits, see Feature) that take part; they turn on
// and off together every EDEN_NX2_AB_PERIOD_MS (default 4000) and the present path logs the mean
// frame time of each phase. EDEN_NX2_AB_OFF=<hex mask> keeps those optimizations off for the
// whole run (an arm "off" against a run without it). Optimizations outside both masks (and all of
// them without the variables) stay on. An optimization that is off behaves like the code before
// it and drops what it cached (it sees Epoch() change).

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace Common::Nx2Ab {

// The registry of bits in use is C:\dev\nx2work\rounds\AB-BITS.md (bits 32 and up: 1ull << n).
enum Feature : u64 {
    // texlookup (11-06): the Texture Cache Memory Limit setting caps the collector's thresholds
    // (TextureCache::ApplyThresholdOverrides) in place of MaxCriticalMemory.
    TcMemoryLimit = 1ull << 1,
    // frontprep (11-06): with channels parked and nothing queued, the GPU thread waits for a
    // wake (semaphore writes, back batch ends, commands) instead of a 2 ms yield loop; at most
    // 200 us (VideoCommon::PendingWake). Idea: SerkiGohan (#681), MFS-code latch wake.
    PendingWake = 1ull << 2,
    // backperf (11-06, play-dkb6 idea 30a): guest reads of cached pages remember 8 flush areas
    // per core (VideoCore::GuestReadAreaTable) in place of one, stamped also with the epoch of
    // images becoming GPU-modified, so repeat reads skip GetFlushArea's two cache mutexes.
    GuestReadAreas = 1ull << 3,
    // vkoff (11-06): in game, graphics pipelines nothing waits for (asynchronous shaders) build on
    // a few threads of their own; boot and the builds a draw or dispatch waits for keep all of the
    // pipeline workers (PipelineCache::async_workers).
    BuilderThreadsByPhase = 1ull << 4,
    // vkoff (11-06): the split's front hands the compute programs of launches it schedules to the
    // pipeline workers, which translate and build them early; the back adopts such a pipeline
    // only when its environment answers every translation query the same (PipelineCache
    // compute prebuilds).
    PipelinePrebuild = 1ull << 13,
    // texlookup (11-06): VK_EXT_memory_priority / VK_EXT_pageable_device_local_memory: render
    // targets, buffers and the stream buffer high, texture cache images by recency
    // (VideoCommon::MemoryPriority). Read once, at device creation.
    VramPriority = 1ull << 6,
    // frontprep (11-06, play-dkb7 idea 22): the split's front makes each draw's graphics pipeline
    // state from its copy of the registers and sends it before the draw; the back's pipeline
    // lookup takes it in place of FixedPipelineState::Refresh (vk_front_pipeline_key.h).
    FrontPipelineKey = 1ull << 10,
    // texlookup (11-06, play-dkb7 idea 21): sparse binds are queued and go to the queue in one
    // vkQueueBindSparse right before the next submission, which waits for them on a timeline
    // semaphore, instead of a bind and a fence wait each (Scheduler::BindSparse). The bind queue
    // (a transfer queue with sparse binding when there is one) is picked at device creation.
    SparseBindAsync = 1ull << 12,
    // codegen (11-06, play-dkb7 idea 16): ARMv8.1 LSE atomics (LDADD..SWP, CAS, CASP) as one host
    // atomic on the fastmem pointer instead of an exclusive load + store pair (Dynarmic
    // UserConfig::native_lse_atomics). Read when a core's JIT is made.
    LseAtomics = 1ull << 15,
    // codegen (11-06, play-dkb8 idea 8): Eglantine #717 (brandonbunnelle): dynarmic x64
    // exclusives without the global monitor's lock, the load before the reservation is
    // published, cross-core clears with lock cmpxchg (Dynarmic UserConfig::
    // lockless_exclusive_monitor). Read when a core's JIT is made. Changes little under
    // Auto CPU accuracy, which already skips the lock (Unsafe_IgnoreGlobalMonitor).
    LockFreeMonitor = 1ull << 25,
    // fences (11-05): direct draws with no vertices or no instances skip the draw setup.
    ZeroDrawSkip = 1ull << 40,
    // vkoff (11-05): draws with none of the dynamic states' dirty flags set skip their updates.
    DynamicStateSkip = 1ull << 41,
    // samestate (11-05): constant buffers unchanged since the previous draw keep its stream copy.
    UniformSlotKeep = 1ull << 43,
    // hosttime (11-06): writes of values only dynamic states read leave the next draw the repeat
    // path (Maxwell3D::SameStateDynamic).
    DynamicDeltas = 1ull << 44,
    // hosttime (11-06): vertex streams bind a range resolved before without resolving it again
    // while the mappings and buffers are as then (BufferCache::vertex_lookups).
    VertexLookupCache = 1ull << 45,
    // backperf (11-06): a buffer range found CPU clean stays clean without a new check while none of
    // its tracking regions turned a page dirty (BufferCache::IsRangeCpuClean); written ranges the
    // GPU-modified sets cover already aren't added again.
    BufferCheckSkip = 1ull << 46,
    // texlookup (11-06): alias walks return at once when no alias was modified after the image
    // (ImageBase::alias_tick_bound).
    AliasWalkSkip = 1ull << 47,
    ListDrawRanges = 1ull << 48,     ///< Lists on pages draws write: past the draws' ranges (nx2.h)
    ListWriterWaitSkip = 1ull << 49, ///< Known CPU list pages skip the wait for writers (dma_pusher)
    SemAcquireOnBack = 1ull << 50,   ///< SEM_EXECUTE acquires checked on the back, no drain (puller)
    ListFlushBatch = 1ull << 51,     ///< List drains download all draw-written list pages at once
    IndirectDrawMacro = 1ull << 52,  ///< GPU-written draw arguments of NVN2's indirect draw macro
                                     ///< drawn from the GPU (Maxwell3D::FrontIndirectDrawMacro)
    // backperf (11-06): a fragment barrier keeps the render pass going until a draw samples one
    // of its render targets (RasterizerVulkan::FragmentBarrier). Idea: zortos293/lichii
    // 60e565c44 + 162a9736f. (53 was AcquireParkEarly, dropped with its code.)
    FragmentBarrierDefer = 1ull << 53,
    IndirectDrawMacroExtended = 1ull << 55, ///< The same for the macro's variant that also writes
                                            ///< base vertex/instance to the constant buffer
    // backperf (11-06): pipelines whose only writes are storage buffers take the buffer refresh
    // path (never a full repeat): RasterizerVulkan::PrepareDraw.
    StorageWriteRefresh = 1ull << 56,
    // codegen (11-06): the host timing thread is woken only for an event that became the earliest
    // (CoreTiming::WakeTimerThread). Idea: MythosOfs/Orchid 6e14af2079.
    TimingWakeEarliest = 1ull << 57,
    // codegen (11-06): host threads made runnable under the scheduler lock are woken right after
    // its spin lock is released, not while it is held (KAbstractSchedulerLock::Unlock).
    // Idea: MythosOfs/Orchid 7c9a94c184.
    WakeAfterUnlock = 1ull << 58,
    // backperf (11-06): a cached write to pages without GPU-written data asks the GPU-modified
    // sets once (BufferCache::CachedWriteMemory), and a dirty memory gather with nothing
    // collected returns before the collectors' lock and the swap
    // (Core::GPUDirtyMemoryManager::Gather). Idea: zortos293/lichii 44bf47d37.
    CachedWriteOneLookup = 1ull << 59,
    // hosttime (11-06): on AMD drivers, draws that sample their own colour target sample a copy
    // taken before the draw (TextureCache::ReplaceColorFeedbackViews, Eglantine PR #720).
    ColorFeedbackCopy = 1ull << 42,
    // hosttime (11-06): buffer downloads the CPU reads end with a transfer -> host read barrier
    // (BufferCacheRuntime::HostReadBarrier). Idea: zortos293/lichii aa3173300.
    HostReadBarrier = 1ull << 54,
    // hosttime (11-06): on AMD's own drivers, software HMMA compute pipelines build where nothing
    // waits for them; their dispatches are skipped until built (PipelineCache::matrix_workers).
    // Idea: zortos293/lichii PR #300 (ctgl1987 54a45b5774).
    DeferredMatrixPipelines = 1ull << 0,
    WaitForIdleBarrier = 1ull << 60, ///< Waits for idle wait (a barrier), compute-to-draw barriers
                                     ///< (RasterizerVulkan::WaitForIdle, Scheduler)
    PresentOwnQueue = 1ull << 61,    ///< Presents on a queue of their own, not under the scheduler's
                                     ///< submit lock (Swapchain::Present)
    AcquireNoDrain = 1ull << 62,     ///< Semaphore acquires read ahead of the back, no drain (puller)
    // switchbrew (11-06): the query cache's IsRegionGpuModified stops at the first dirty query
    // (QueryCacheBase::IterateCache). Idea: Orchid d38d025192.
    QueryIterateEarlyStop = 1ull << 63,
    // switchbrew (11-06), Eglantine #712, fence-delayed releases only (Puller::CanResumeAcquire):
    // a forced release that didn't land flushes the scheduler first, and a passed AcquireEqual
    // target is met after 10 ms.
    SemaphoreRescue = 1ull << 7,
    // The same PR's ring rescue: an AcquireEqual 100 ms short of a target at most 0x40 ahead
    // writes the target itself.
    SemaphoreRingRescue = 1ull << 8,
    // switchbrew (11-06): macro DMA reads of report memory wait for the reports the query cache
    // still holds, also under the GPU split (VideoCommon::FlushQueryResults). Idea:
    // zortos293/lichii 7ae3e7401.
    MacroReadQueryWait = 1ull << 9,
    // hosttime (11-06, play-dkb7 idea 1b): where the timed wait keeps the thread on its CPU
    // (MWAITX/UMWAIT), the host timing thread sleeps on a high-resolution timer until a measured
    // margin before the next event and waits precisely only for the rest
    // (CoreTiming::WaitForNextEvent, Common::TimedSleeper). Idea: Eglantine PR #538 (pretend7979).
    TimingWaitCheap = 1ull << 14,
    // switchbrew (11-06): VK_EXT_host_image_copy for the first upload of CPU-produced textures:
    // the CPU writes them into the image (Image::HostUpload), no staging copy on the GPU.
    HostImageCopy = 1ull << 16,
    // backperf (11-06, play-dkb7 idea 32): buffer downloads copy on a graphics-family queue of
    // their own (Vulkan::ReadbackQueue) after the GPU tick of the source's last use, and wait for
    // that copy alone instead of Finish() (the whole graphics queue).
    ReadbackQueue = 1ull << 11,
    // hosttime (11-06, play-dkb8 idea 10): plain address-arbiter signals with no thread waiting
    // in the address's bucket, and waits whose value already doesn't match, return without the
    // global scheduler lock (KAddressArbiter::m_waiter_hint). EDEN_NX2_KERNEL_WAKE_VERIFY=1
    // checks every fast signal against the wait tree.
    KernelWakeFast = 1ull << 23,
    // hosttime (11-06, play-dkb8 idea 27 step 2): a WFE that has spun for a while parks its core
    // briefly (precise wait on the core's event), and SEV/SEVL and interrupts end it
    // (Core::WfePark). A set event register still returns at once.
    WfePark = 1ull << 24,
};

inline const u64 g_mask = [] {
    const char* value = std::getenv("EDEN_NX2_AB");
    return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 16)) : u64{0};
}();

inline const u64 g_forced_off = [] {
    const char* value = std::getenv("EDEN_NX2_AB_OFF");
    return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 16)) : u64{0};
}();

inline const u32 g_period_ms = [] {
    const char* value = std::getenv("EDEN_NX2_AB_PERIOD_MS");
    return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 4000u;
}();

/// Phase: true while the optimizations in the mask are off.
inline std::atomic<bool> g_off{false};
/// Bumped at every phase change.
inline std::atomic<u32> g_epoch{0};

inline bool On(Feature feature) {
    return (g_forced_off & feature) == 0 &&
           ((g_mask & feature) == 0 || !g_off.load(std::memory_order_relaxed));
}

inline u32 Epoch() {
    return g_epoch.load(std::memory_order_relaxed);
}

/// Called by the present path once per presented frame with that frame's time.
inline void OnPresent(long long frame_us) {
    if (g_mask == 0) {
        return;
    }
    struct Bucket {
        long long us{};
        long long frames{};
    };
    static Bucket buckets[2]{};
    static Bucket totals[2]{};
    static u32 since_flip{};
    static auto start = std::chrono::steady_clock::now();
    static auto last_log = start;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    const bool off = ((ms / g_period_ms) & 1) != 0;
    const bool was_off = g_off.load(std::memory_order_relaxed);
    if (off != was_off) {
        g_off.store(off, std::memory_order_relaxed);
        g_epoch.fetch_add(1, std::memory_order_relaxed);
        since_flip = 0;
        return;
    }
    // The first frames after a flip were prepared partly in the other phase; hitches are loads.
    if (++since_flip <= 6 || frame_us > 100'000 || frame_us <= 0) {
        return;
    }
    buckets[was_off ? 1 : 0].us += frame_us;
    ++buckets[was_off ? 1 : 0].frames;
    if (now - last_log < std::chrono::seconds{30}) {
        return;
    }
    last_log = now;
    const auto mean = [](const Bucket& b) {
        return b.frames != 0 ? static_cast<double>(b.us) / static_cast<double>(b.frames) : 0.0;
    };
    for (int i = 0; i < 2; ++i) {
        totals[i].us += buckets[i].us;
        totals[i].frames += buckets[i].frames;
    }
    LOG_INFO(Service,
             "nx2 A/B mask {:#x}: last 30 s on {:.0f} us ({} frames), off {:.0f} us ({} frames), "
             "on/off {:.3f} | total on {:.0f} us ({} frames), off {:.0f} us ({} frames), on/off {:.3f}",
             g_mask, mean(buckets[0]), buckets[0].frames, mean(buckets[1]), buckets[1].frames,
             mean(buckets[1]) != 0.0 ? mean(buckets[0]) / mean(buckets[1]) : 0.0,
             mean(totals[0]), totals[0].frames, mean(totals[1]), totals[1].frames,
             mean(totals[1]) != 0.0 ? mean(totals[0]) / mean(totals[1]) : 0.0);
    buckets[0] = {};
    buckets[1] = {};
}

} // namespace Common::Nx2Ab
