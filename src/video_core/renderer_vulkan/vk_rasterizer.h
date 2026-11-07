// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <unordered_map>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_bindless_heap.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core {
class System;
}

namespace Core::Frontend {
class EmuWindow;
}

namespace Tegra {

namespace Engines {
class Maxwell3D;
}

} // namespace Tegra

namespace Vulkan {

struct FramebufferTextureInfo;

class StateTracker;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache,
                           Scheduler& scheduler);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;

    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;

    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;

    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& src,
                            const Tegra::DMA::ImageOperand& dst);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    Scheduler& scheduler;
};

class RasterizerVulkan final : public VideoCore::RasterizerInterface,
                               protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                              Tegra::MaxwellDeviceMemoryManager& device_memory_,
                              const Device& device_, MemoryAllocator& memory_allocator_,
                              StateTracker& state_tracker_, Scheduler& scheduler_);
    ~RasterizerVulkan() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    const std::string* DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    bool FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    void VerifyGpuWrittenCommandList(DAddr addr, std::span<const u8> words,
                                     std::span<const u8> before) override;
    bool ReadBackGpuWords(DAddr addr, std::span<u8> out) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void OnGuestSyncPoint() override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    bool HasDrawTransformFeedback() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    bool SupportsGameDlssNetwork() const override;
    bool UsesNvidiaDlss() const override;
    bool CanProbeGameDlssOutput() const override {
        return true;
    }
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth,
                                   std::array<f32, 2> jitter,
                                   std::array<f32, 2> motion_scale) override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void InitializeChannel(Tegra::Control::ChannelState& channel) override;

    void BindChannel(Tegra::Control::ChannelState& channel) override;

    void ReleaseChannel(s32 channel_id) override;

    void OnGpuShutdown() override;

    std::optional<FramebufferTextureInfo> AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                            VAddr framebuffer_addr,
                                                            u32 pixel_stride);

private:
    static constexpr const u64 NEEDS_D24[] = {
        0x01006A800016E000ULL, // SSBU
        0x0100E95004038000ULL, // XC2
        0x0100A6301214E000ULL, // FE:Engage
    };
    static constexpr size_t MAX_TEXTURES = 192;
    static constexpr size_t MAX_IMAGES = 48;
    static constexpr size_t MAX_IMAGE_VIEWS = MAX_TEXTURES + MAX_IMAGES;

    static constexpr VkDeviceSize DEFAULT_BUFFER_SIZE = 4 * sizeof(float);

    template <typename Func>
    void PrepareDraw(bool is_indexed, Func&&);
    bool TryNx2IndirectReset();

    void FlushWork();

    /// Records the copies of the command list pages the work just recorded wrote and submits, so
    /// the GPU makes them now (BufferCache::Nx2CopyListPages).
    void SubmitNx2ListCopies();

    void UpdateDynamicStates(GraphicsPipeline* pipeline);

    /// Why the draw at hand can't reuse the previous draw's setup as it is (0: it can). With only
    /// the buffer bits set, RefreshBuffers may redo just the buffer bindings.
    enum RepeatRefusalBits : u32 {
        RefusalRegisters = 1u << 0,
        RefusalBackEvents = 1u << 1,
        RefusalSetupEvents = 1u << 2,
        RefusalRecords = 1u << 3,
        RefusalExcluded = 1u << 4,
        RefusalNoPrevious = 1u << 5,
        RefusalUniforms = 1u << 6,    ///< constant buffer uploads, addresses or binds
        RefusalGeometry = 1u << 7,    ///< vertex or index buffer addresses
        RefusalBufferEvents = 1u << 8, ///< uploads' cache invalidations, uniform binds
        RefusalIndexRange = 1u << 9,  ///< more indices than the setup synchronized
        RefusalBuffers = RefusalUniforms | RefusalGeometry | RefusalBufferEvents | RefusalIndexRange,
    };
    [[nodiscard]] u32 RepeatRefusal(bool is_indexed) const;
    /// With only buffer refusals: whether the changes left what else the setup read alone (the
    /// constant buffer words with texture handles and storage buffer addresses, texture
    /// descriptors, images, shaders).
    [[nodiscard]] bool CanRefreshBuffers() const;
    /// After a draw's setup: what the next draw compares against.
    void RememberDrawSetup(GraphicsPipeline* pipeline, bool is_indexed, u64 setup_events_seen);
    /// After a draw's buffer refresh.
    void RememberBufferRefresh(bool is_indexed);
    /// EDEN_NX2_REPEAT_STATS=1 (counters only): the path every draw takes (repeat, buffer refresh,
    /// full setup) and why the others were refused, with the register methods behind register
    /// refusals; indirect draws as if DrawIndirect's own setup changes didn't count. Every 10 s.
    void CountRepeat(u32 refusal, bool refresh);
    /// Same-state verify (EDEN_NX2_SAME_STATE_VERIFY): what the fast path would have reused.
    struct SameStateExpectation {
        const GraphicsPipeline* pipeline{};
        GraphicsPipeline::SetupFingerprint fingerprint{};
        bool repeat{};  ///< repeat (everything reused) rather than buffer refresh
        bool active{};  ///< this draw is checked: it takes the full setup
    };
    /// After the full setup of a checked draw: compares it with the fast path's.
    void CheckSameState(const SameStateExpectation& expected, const GraphicsPipeline* pipeline);
    /// Something a draw's setup reads may have changed outside the 3D registers (rasterizer calls
    /// for other work, cache invalidations from guest threads, counted once they are done).
    void NoteSetupChange() noexcept {
        setup_events.fetch_add(1, std::memory_order_release);
    }

    void HandleTransformFeedback();

    void UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStipple(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs,
                                     const GraphicsPipeline* pipeline);
    void UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs,
                                const GraphicsPipeline* pipeline);
    void UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;

    const Device& device;
    MemoryAllocator& memory_allocator;
    StateTracker& state_tracker;
    s32 bound_channel_id = -1;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    DescriptorPool descriptor_pool;
    GuestDescriptorQueue guest_descriptor_queue;
    ComputePassDescriptorQueue compute_pass_descriptor_queue;
    DescriptorBufferRing descriptor_buffer_ring;
    BindlessHeap bindless_heap;
    BlitImageHelper blit_image;
    RenderPassCache render_pass_cache;

    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    QueryCacheRuntime query_cache_runtime;
    QueryCache query_cache;
    PipelineCache pipeline_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;

    vk::Event wfi_event; // Only used when EGLANTINE_SYNC_FIX=0

    boost::container::static_vector<u32, MAX_IMAGE_VIEWS> image_view_indices;
    std::array<VideoCommon::ImageViewId, MAX_IMAGE_VIEWS> image_view_ids;
    boost::container::static_vector<VkSampler, MAX_TEXTURES> sampler_handles;

    u32 draw_counter = 0;
    bool mesh_draw{}; ///< the draw being prepared uses a task/mesh pipeline

    std::atomic<u64> setup_events{};
    u64 buffer_events{}; ///< InnerInvalidation and uniform binds (split back thread)
    /// BufferCache::Nx2GpuWriteSerial at the last barrier before an indirect draw (DrawIndirect).
    u64 nx2_indirect_barrier_serial{~u64{0}};
    /// DrawIndirect with arguments GPU work may have written: whether a barrier must come first.
    bool Nx2IndirectBarrierNeeded(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams& params);
    /// The last draw whose setup the next draw may reuse (RepeatRefusal).
    struct RepeatableDraw {
        GraphicsPipeline* pipeline{};
        std::array<u64, Tegra::Engines::Maxwell3D::NumSameStateKinds> serials{};
        u64 back_events{};
        u64 setup_events{};
        u64 buffer_events{};
        u64 record_serial{};
        u64 written_images{};
        u64 invalidated_shader_pages{};
        u64 index_limit{}; ///< bytes of the index buffer the setup synchronized
        Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology{};
        bool is_indexed{};
        bool valid{};
        bool refresh_only{}; ///< The pipeline writes storage buffers: a refresh, never a repeat
        /// BufferCache::BufferSetSerial at the setup: a refresh keeps the storage bindings, so a
        /// storage-writing pipeline refreshes only while no buffer was created or deleted.
        u64 buffer_set_serial{};
        u8 invalid_reason{}; ///< Why not valid (repeat stats only)
    } repeat;
    u64 same_state_verify_count{};
    bool indirect_draw{};      ///< DrawIndirect's PrepareDraw is running
    bool last_draw_indirect{}; ///< The draw prepared before this one was indirect
    struct RepeatStats {
        u64 draws{};
        u64 indirect{};
        u64 repeats{};
        u64 refreshes{};
        u64 indirect_repeats{};   ///< ... of them indirect (would-be: DrawIndirect never repeats)
        u64 indirect_refreshes{};
        u64 writer_refreshes{};   ///< Refreshes of storage-writing pipelines (StorageWriteRefresh)
        u64 writer_refused{};     ///< ... refused with no refusal bit (views, cbuf words, buffer set)
        std::array<u64, 10> refused_first{}; ///< Full setups by the lowest refusal bit set
        std::array<u64, 10> refused_any{};   ///< ... by every refusal bit set
        std::array<u64, 8> invalid{};        ///< RefusalNoPrevious: why the last setup wasn't repeatable
        /// Register refusals by Maxwell3D::last_other_method (NUM_REGS: a state run; the last
        /// slot: an HLE/interpreted macro).
        std::vector<u64> other_methods;
        /// Back-event refusals by the other engine of the last engine switch (EngineLabel).
        std::unordered_map<const char*, u64> back_event_engines;
    } repeat_stats;
    /// The bound channel's engines and their names in the repeat stats (no RTTI in GCC builds).
    std::array<std::pair<const Tegra::Engines::EngineInterface*, const char*>, 5> channel_engines{};
    [[nodiscard]] const char* EngineLabel(const Tegra::Engines::EngineInterface* engine) const noexcept;

    /// Uniform slot keep (uniform_stream_reuse.h): what happened since the last draw that bound
    /// uniform buffers. Clean when nothing but the guest writes listed here did.
    struct UniformKeepWindow {
        std::array<std::pair<u64, u64>, 32> written{}; ///< InnerInvalidation ranges since
        u32 num_written{};
        bool overflow{};
        bool bound{}; ///< A draw bound uniform buffers and nothing ended the window since
        u64 setup_events{};  ///< At that draw's start (guest threads count once done)
        u64 foreign_events{};
        u64 gpu_modified{};
        u32 epoch{};
        u64 back_events{};   ///< After the last submitted draw
        u64 record_serial{};
        const Tegra::Engines::Maxwell3D* maxwell3d{};
    } uniform_keep;
    std::atomic<u64> uniform_keep_foreign{}; ///< InnerInvalidation calls off the back thread
    struct UniformKeepStats {
        u64 draws{};
        std::array<u64, 9> refused{}; ///< First failing check of BeginUniformKeepWindow
    } uniform_keep_stats;
    /// Starts a draw's uniform buffer binds (once per draw, before the refresh or full setup).
    void BeginUniformKeepWindow();
    void LogUniformKeepStats();
};

} // namespace Vulkan
