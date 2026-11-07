// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           BindlessHeap& bindless_heap, RenderPassCache& render_pass_cache,
                           BufferCache& buffer_cache, TextureCache& texture_cache,
                           VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();
    [[nodiscard]] bool IsIndirectResetShader() const;

    /// Why the last CurrentGraphicsPipeline() returned null (diagnostics).
    [[nodiscard]] const char* GraphicsMissReason() const noexcept {
        return graphics_miss;
    }

    void TickFrame();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// Waits for the translations running on workers and starts no more: guest GPU memory, which
    /// they read texture descriptors from, goes away at shutdown.
    void StopTranslation();

private:
    /// A graphics pipeline translated and built on a pipeline worker (asynchronous shaders).
    struct PendingGraphicsPipeline {
        enum class State : u8 {
            Queued,
            Running,
            Done,
            Cancelled, ///< Built on the GPU thread instead, before a worker took it
        };
        GraphicsPipelineCacheKey key;
        GraphicsEnvironments environments;
        std::unique_ptr<GraphicsPipeline> pipeline;
        /// Translation read engine state the environments did not copy.
        bool needs_gpu_thread{};
        std::atomic<State> state{State::Queued};
        std::chrono::steady_clock::time_point queued{};
    };

    /// The workers asynchronous graphics builds go to: async_workers in game while A/B bit
    /// BuilderThreadsByPhase is on, workers otherwise.
    [[nodiscard]] Common::ThreadWorker& AsyncWorkers();

    /// Logs the in-game build counters (at most every 30 s, on the GPU thread).
    void LogBuildCounters();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    /// Whether the current draw may be skipped while its pipeline is built.
    [[nodiscard]] bool CanSkipDraw() const noexcept;

    /// Queues translation and build of the current graphics pipeline on the workers.
    void QueueGraphicsPipeline();

    /// Runs on a worker.
    void BuildPendingPipeline(PendingGraphicsPipeline& job);

    /// The pipeline a worker built, checked on the GPU thread; null when the worker could not
    /// build it or what it read has changed since.
    std::unique_ptr<GraphicsPipeline> TakePendingPipeline(PendingGraphicsPipeline& job);

    /// Moves the pipelines workers have finished into the cache.
    void CollectPendingPipelines();

    /// Writes a new pipeline to the disk cache (and the driver's cache when it is time to).
    void StorePipeline(const GraphicsPipeline& pipeline, const GraphicsPipelineCacheKey& key,
                       GraphicsEnvironments& environments);

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                           const ShaderInfo* shader);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(ShaderPools& pools,
                                                           const ComputePipelineCacheKey& key,
                                                           Shader::Environment& env,
                                                           PipelineStatistics* statistics,
                                                           bool build_in_parallel);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    BindlessHeap& bindless_heap;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};
    mutable const char* graphics_miss{"no-pipeline"};

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since

    ::Common::unordered_map<ComputePipelineCacheKey, std::unique_ptr<ComputePipeline>> compute_cache;
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    /// Pipelines on the workers; their graphics_cache entries are null meanwhile.
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::shared_ptr<PendingGraphicsPipeline>>
        pending_graphics;
    std::shared_mutex translation_mutex; ///< Held shared by each translation on a worker
    bool translation_stopped{};

    ShaderPools main_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;
    /// Task and mesh pipelines, with the mesh state they are translated with.
    std::filesystem::path mesh_pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::filesystem::path used_pipelines_filename;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};

    /// In-game build counters, logged by LogBuildCounters (before the workers that update them).
    struct BuildCounters {
        std::atomic<u64> async_queued{};
        std::atomic<u64> async_started{};
        std::atomic<u64> async_wait_us{};
        std::atomic<u64> async_wait_max_us{};
        // GPU thread only:
        u64 backlog_max{};
        u64 gpu_thread_graphics{};
        u64 gpu_thread_compute{};
        u64 queued_logged{};
        u64 started_logged{};
        std::chrono::steady_clock::time_point window_start{};
    } build_counters;
    size_t worker_count{};
    size_t async_worker_count{};

    Common::ThreadWorker workers;
    /// In game, graphics pipelines nothing waits for (asynchronous shaders: their draws are
    /// skipped meanwhile) translate and build here, on a few threads, not on all of workers
    /// (A/B bit BuilderThreadsByPhase). Boot, and the builds a draw or dispatch waits for, keep
    /// all of workers, which no longer have that queue in front of them. Null when the bit is
    /// off at start or the driver can't compile in parallel.
    std::unique_ptr<Common::ThreadWorker> async_workers;
    Common::ThreadWorker serialization_thread;
    /// Builds software matrix compute pipelines on AMD's own compilers, which nothing waits for
    /// (A/B bit DeferredMatrixPipelines); null elsewhere. Joined before the pipelines it builds go.
    std::unique_ptr<Common::ThreadWorker> matrix_workers;
    DynamicFeatures dynamic_features;

    /// Cached graphics pipelines boot skipped (not in used_pipelines) are built on background
    /// threads once the game runs, so a draw that needs one finds it built instead of translating
    /// it on the GPU thread. EDEN_NX2_BACKGROUND_PIPELINES=0 turns it off.
    std::unique_ptr<GraphicsPipeline> TakeBackgroundPipeline(const GraphicsPipelineCacheKey& key);
    void StopBackgroundPipelines();
    std::mutex background_mutex;
    /// Built and not yet asked for by a draw.
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> background_ready;
    /// Keys the GPU thread built itself meanwhile: their background build is skipped or dropped.
    std::unordered_set<GraphicsPipelineCacheKey> background_claimed;
    std::atomic_bool background_active{};
    std::atomic_bool background_stop{};
    std::atomic<size_t> background_left{};
    size_t background_taken{}; ///< Under background_mutex
    std::chrono::steady_clock::time_point background_start{};
    /// Last: destroyed (its threads joined) before the state above.
    std::unique_ptr<Common::ThreadWorker> background_workers;
};

} // namespace Vulkan
