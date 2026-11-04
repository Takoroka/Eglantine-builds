// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <span>
#include <type_traits>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, Common::ThreadWorker* worker_thread,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos);

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0].is_task || stage_infos[0].is_mesh;
    }

    /// Whether a draw may reuse the setup of a previous draw with this pipeline (none of its
    /// stages writes memory).
    [[nodiscard]] bool CanRepeatDraws() const noexcept {
        return can_repeat_draws;
    }

    /// Constant buffer bytes the setup reads (texture handles, storage buffer addresses): byte
    /// range [begin, end) of `slot` in `stage`.
    struct SetupCbufRead {
        u32 stage;
        u32 slot;
        u32 begin;
        u32 end;
    };
    [[nodiscard]] std::span<const SetupCbufRead> SetupCbufReads() const noexcept {
        return setup_cbuf_reads;
    }
    /// Per stage, the constant buffer slots the setup reads.
    [[nodiscard]] const std::array<u32, NUM_STAGES>& SetupCbufSlots() const noexcept {
        return setup_cbuf_slots;
    }
    /// Whether texture handles all come from constant buffers (none computed from storage
    /// buffers), so RefreshBuffers may keep the last setup's image views.
    [[nodiscard]] bool CanRefreshBuffers() const noexcept {
        return can_refresh_buffers;
    }

    enum class ConfigureResult {
        Done,
        Failed, ///< skip the draw
        Retry,  ///< RefreshBuffers only: nothing was recorded, run the full setup
    };
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed, false) == ConfigureResult::Done;
    }

    /// Setup of a draw that differs from the last configured draw of this pipeline only in its
    /// uniform buffers (bindings and contents) and vertex and index buffers: binds those again and
    /// pushes the descriptors with the last setup's image views and samplers.
    ConfigureResult RefreshBuffers(bool is_indexed) {
        return configure_func(this, is_indexed, true);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transitions.clear();
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return is_built.load(std::memory_order::relaxed);
    }

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed, bool refresh) {
            return refresh ? pl->RefreshImpl<Spec>(is_indexed)
                           : (pl->ConfigureImpl<Spec>(is_indexed) ? ConfigureResult::Done
                                                                  : ConfigureResult::Failed);
        };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    template <typename Spec>
    bool ConfigureImpl(bool is_indexed);

    template <typename Spec>
    ConfigureResult RefreshImpl(bool is_indexed);

    /// What the descriptor setup does with the texture set: nothing, store this setup's texture
    /// descriptors in it, or push its stored descriptors instead of the views'.
    enum class TextureSetUse {
        None,
        Record,
        Replay,
    };

    /// The setup after the buffer updates: descriptors, render targets (full setups) and the
    /// pipeline and descriptor binds.
    template <typename Spec>
    bool ConfigureDescriptors(std::span<const VideoCommon::ImageViewInOut> views,
                              const VideoCommon::SamplerId* samplers, bool full_setup,
                              TextureSetUse texture_set_use);

    /// Whether the texture set still holds what these handles resolve to: same handles, nothing
    /// in the texture cache changed that the resolution read, the same TIC and TSC entries.
    [[nodiscard]] bool TextureSetCurrent(std::span<const std::pair<u32, u32>> handles,
                                         bool via_header_index);

    /// Starts a texture set from a full setup's handles, views and samplers; the descriptor setup
    /// completes it (TextureSetUse::Record).
    void RecordTextureSet(std::span<const std::pair<u32, u32>> handles,
                          std::span<const VideoCommon::ImageViewInOut> views,
                          std::span<const VideoCommon::SamplerId> samplers, bool via_header_index);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    void MakePipeline(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    ConfigureResult (*configure_func)(GraphicsPipeline*, bool, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;
    u64 last_use_frame{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    u32 num_textures{};
    bool fragment_has_color0_output{};
    bool can_repeat_draws{true};
    bool can_refresh_buffers{true};
    std::vector<SetupCbufRead> setup_cbuf_reads;
    std::array<u32, NUM_STAGES> setup_cbuf_slots{};
    /// The last full setup's image views and samplers (RefreshBuffers).
    std::vector<VideoCommon::ImageViewInOut> last_views;
    std::vector<VideoCommon::SamplerId> last_samplers;

    /// Texture set (EDEN_NX2_BINDLESS): what the textures of the last full setup that resolved
    /// them came to. Later setups with the same handles reuse it while it is current
    /// (TextureSetCurrent) instead of looking up views and samplers and preparing their images.
    struct TextureSet {
        u64 generation{}; ///< TextureCache::TextureSetGeneration when it was resolved
        bool via_header_index{};
        bool valid{};
        std::vector<std::pair<u32, u32>> handles; ///< (TIC, TSC) per texture element
        std::vector<VideoCommon::ImageViewInOut> views;
        std::vector<VideoCommon::SamplerId> samplers;
        std::vector<DescriptorUpdateEntry> descriptors; ///< per texture element, stage order
        std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling{};
    };
    TextureSet texture_set;
    /// Textures all read from constant buffer handles, no texture buffers or images.
    bool uses_texture_sets{true};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
};

} // namespace Vulkan
