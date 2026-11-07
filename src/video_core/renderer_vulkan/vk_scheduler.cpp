// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/nx2_ab.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_gpu_timestamps.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)},
      timestamps{GpuTimestamps::Create(device_)} {

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    if (master_semaphore->IsTimeline() && device.IsSparseBindingSupported()) {
        // A/B bit SparseBindAsync: queued sparse binds signal this; submissions wait for it.
        static constexpr VkSemaphoreTypeCreateInfo semaphore_type_ci{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0,
        };
        sparse_timeline = device.GetLogical().CreateSemaphore(VkSemaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &semaphore_type_ci,
            .flags = 0,
        });
    }
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() = default;

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    return signal_value;
}

bool Scheduler::DeferFlush() {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        return value == nullptr ? s64{500} : static_cast<s64>(std::atoll(value));
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        return false;
    }
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred() {
    if (flush_pending) {
        Flush();
    }
}

bool Scheduler::WaitSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return true;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            LOG_WARNING(Render_Vulkan, "Waited {} s for the GPU thread to submit tick {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick);
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    AllocateNewContext();
}

void Scheduler::WaitWorker() {
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call (~2% of the GPU thread at one dispatch per 8
        // draws): skip it while the worker is still busy, it pops the queue before waiting.
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;
    if (timestamps) [[unlikely]] {
        TimePassBegin(framebuffer);
    }

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    const bool after_compute = std::exchange(compute_writes_pending, false);
    if (after_compute) {
        Tegra::Nx2CountList(Tegra::g_nx2_list_stats.compute_draw_barriers);
    }
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count,
            after_compute](vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        if (after_compute) {
            // Draws read compute output as indirect arguments, vertices and resources; nothing
            // else orders those reads after the dispatch.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, bool keeps_fragment_barrier) {
    if (fragment_barrier.EndsPassOnRequest(keeps_fragment_barrier)) {
        // Blits and texture draws sample a source that may be one of the pass's attachments.
        EndRenderPass();
    }
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (renderpass == state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext() {
    EndRenderPass();
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    EndPendingOperations();
    if (timestamps) [[unlikely]] {
        // Segments end with their command buffers.
        CloseTimed(timed_outside);
        CloseTimed(timed_upload);
        CloseTimed(timed_present);
    }
    InvalidateState();

    // The sparse binds queued since the last submission go to the queue right before it.
    std::unique_ptr<SparseBindBatch> sparse;
    if (!pending_sparse.Empty()) {
        sparse = std::make_unique<SparseBindBatch>();
        sparse->Append(std::move(pending_sparse));
        pending_sparse_serial.store(0, std::memory_order_relaxed);
        ++sparse_handed;
    }
    const u64 signal_value = master_semaphore->NextTick();
    RecordImpl([signal_semaphore, wait_semaphore, signal_value, this,
                sparse = std::move(sparse)](vk::CommandBuffer cmdbuf,
                                            vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        u64 sparse_value = 0;
        if (sparse) {
            sparse_value = SubmitSparse(*sparse);
            sparse_bound.fetch_add(1, std::memory_order_release);
        }
        if (sparse_timeline) {
            TickSparse();
        }
        switch (const VkResult result = master_semaphore->SubmitQueue(
                    cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value,
                    sparse_value != 0 ? *sparse_timeline : VK_NULL_HANDLE, sparse_value)) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    ++record_serial;
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        // The barrier after the pass (below) is what a deferred fragment barrier waits for.
        fragment_barrier.OnPassEnd();
        if (!state.renderpass) {
            return;
        }

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        if (timed_pass.open) [[unlikely]] {
            CloseTimed(timed_pass);
        }
        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::WriteTimestamp(VkQueryPool pool, u32 query, bool upload) {
    // Bottom of the pipe at both ends: each mark waits for what was recorded before it, so a
    // segment's time is what the GPU took to finish its commands after the earlier ones.
    if (upload) {
        RecordImpl([pool, query](vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
            upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query);
        });
    } else {
        RecordImpl([pool, query](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query);
        });
    }
}

void Scheduler::OpenTimed(TimedSegment& segment, u64 key, bool upload) {
    const auto reserved = timestamps->Reserve(key);
    if (!reserved) {
        return;
    }
    segment = TimedSegment{
        .open = true,
        .upload = upload,
        .pool = reserved->pool,
        .end_query = reserved->begin_query + 1,
        .index = reserved->segment,
        .key = key,
        .work = 0,
    };
    WriteTimestamp(reserved->pool, reserved->begin_query, upload);
}

void Scheduler::CloseTimed(TimedSegment& segment) {
    if (!segment.open) {
        return;
    }
    WriteTimestamp(segment.pool, segment.end_query, segment.upload);
    timestamps->SetWork(segment.index, segment.work);
    segment.open = false;
}

namespace {
/// Segment keys: the kind in the top bits, so that kinds never share a key.
u64 TimedKey(VideoCommon::GpuSegmentKind kind, u64 value) {
    return (static_cast<u64>(kind) << 60) ^ (value & ((u64{1} << 60) - 1));
}
} // Anonymous namespace

void Scheduler::TimePassBegin(const Framebuffer* framebuffer) {
    CloseTimed(timed_outside);
    CloseTimed(timed_pass);
    // A pass is known by its size and attachment formats: the same pass of every frame.
    const RenderPassKey& base = framebuffer->RenderPassKeyBase();
    const VkExtent2D area = framebuffer->RenderArea();
    std::array<u32, 12> fields{};
    fields[0] = area.width;
    fields[1] = area.height;
    for (size_t slot = 0; slot < base.color_formats.size(); ++slot) {
        fields[2 + slot] = static_cast<u32>(base.color_formats[slot]);
    }
    fields[10] = static_cast<u32>(base.depth_format);
    fields[11] = static_cast<u32>(base.samples);
    const u64 key = TimedKey(VideoCommon::GpuSegmentKind::Pass,
                             Common::CityHash64(reinterpret_cast<const char*>(fields.data()),
                                                sizeof(fields)));
    if (!timestamps->Known(key)) {
        std::string label = fmt::format("pass {}x{}", area.width, area.height);
        const char* separator = " c:";
        for (const auto format : base.color_formats) {
            if (format != VideoCore::Surface::PixelFormat::Invalid) {
                label += fmt::format("{}{}", separator, format);
                separator = ",";
            }
        }
        if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
            label += fmt::format(" d:{}", base.depth_format);
        }
        if (base.samples != VK_SAMPLE_COUNT_1_BIT) {
            label += fmt::format(" x{}", static_cast<u32>(base.samples));
        }
        timestamps->Name(key, VideoCommon::GpuSegmentKind::Pass, std::move(label));
    }
    timed_pass_key = key;
    OpenTimed(timed_pass, key, false);
}

void Scheduler::TimeOutside(u64 value, VideoCommon::GpuSegmentKind kind, const char* name) {
    const u64 key = TimedKey(kind, value);
    if (timed_outside.open && timed_outside.key == key) {
        ++timed_outside.work;
        return;
    }
    CloseTimed(timed_outside);
    if (!timestamps->Known(key)) {
        timestamps->Name(key, kind,
                         name != nullptr ? std::string{name}
                                         : fmt::format("compute {:016x}", value));
    }
    OpenTimed(timed_outside, key, false);
    timed_outside.work = 1;
}

void Scheduler::TimeInternal(TimedWork work) {
    const auto [kind, name] = [work]() -> std::pair<VideoCommon::GpuSegmentKind, const char*> {
        using Kind = VideoCommon::GpuSegmentKind;
        switch (work) {
        case TimedWork::TextureUpload:
            return {Kind::Upload, "texture uploads"};
        case TimedWork::VertexRepack:
            return {Kind::Compute, "vertex repack"};
        case TimedWork::Uint8Index:
            return {Kind::Compute, "8-bit index conversion"};
        case TimedWork::QuadIndex:
            return {Kind::Compute, "quad index conversion"};
        case TimedWork::ConditionalResolve:
            return {Kind::Compute, "conditional rendering resolve"};
        case TimedWork::QueryPrefixScan:
            return {Kind::Compute, "query prefix scan"};
        case TimedWork::AstcDecode:
            return {Kind::Compute, "ASTC decode"};
        case TimedWork::BcEncode:
            return {Kind::Compute, "BC encode"};
        case TimedWork::Unswizzle3D:
            return {Kind::Compute, "3D unswizzle"};
        }
        return {Kind::Compute, "internal"};
    }();
    TimeOutside(static_cast<u64>(work), kind, name);
}

void Scheduler::TimeUploadBuffer() {
    if (timed_upload.open) {
        return;
    }
    static const u64 key = TimedKey(VideoCommon::GpuSegmentKind::Upload, 1);
    if (!timestamps->Known(key)) {
        timestamps->Name(key, VideoCommon::GpuSegmentKind::Upload,
                         "buffer uploads (upload command buffer)");
    }
    OpenTimed(timed_upload, key, true);
}

void Scheduler::TimePresent(bool begin) {
    if (!begin) {
        CloseTimed(timed_present);
        return;
    }
    CloseTimed(timed_outside);
    static const u64 key = TimedKey(VideoCommon::GpuSegmentKind::Present, 1);
    if (!timestamps->Known(key)) {
        timestamps->Name(key, VideoCommon::GpuSegmentKind::Present,
                         "present (post-processing, upscaling, swapchain blit)");
    }
    OpenTimed(timed_present, key, false);
}

void Scheduler::TimeFrameEnd() {
    const bool pass_active = timed_pass.open;
    CloseTimed(timed_pass);
    CloseTimed(timed_outside);
    CloseTimed(timed_upload);
    CloseTimed(timed_present);
    timestamps->EndFrame(*this);
    if (pass_active && state.renderpass) {
        // The render pass goes on into the next frame: its rest is a segment of that frame.
        OpenTimed(timed_pass, timed_pass_key, false);
    }
}

bool Scheduler::BindSparse(SparseBindBatch&& binds, bool now) {
    if (binds.Empty()) {
        return true;
    }
    sparse_stats.binds.fetch_add(binds.Binds(), std::memory_order_relaxed);
    binds.CountRequest();
    if (!now && sparse_timeline && Common::Nx2Ab::On(Common::Nx2Ab::SparseBindAsync)) {
        sparse_stats.queued.fetch_add(1, std::memory_order_relaxed);
        if (pending_sparse.Empty()) {
            pending_sparse_serial.store(++sparse_serials, std::memory_order_relaxed);
        }
        pending_sparse.Append(std::move(binds));
        return true;
    }
    sparse_stats.now.fetch_add(1, std::memory_order_relaxed);
    return BindSparseNow(std::move(binds));
}

bool Scheduler::BindSparseNow(SparseBindBatch&& binds) {
    // What was queued goes with it (other blocks: each is bound once and never rebound), so that
    // nothing queued is left for a submission after memory the caller frees.
    SparseBindBatch batch;
    if (!pending_sparse.Empty()) {
        batch.Append(std::move(pending_sparse));
        pending_sparse_serial.store(0, std::memory_order_relaxed);
    }
    batch.Append(std::move(binds));
    if (batch.Empty()) {
        return true;
    }
    // Batches given to submissions the worker hasn't made yet go to the queue first; this bind
    // then waits for them (the caller may free memory they rebind blocks away from).
    if (sparse_handed != sparse_bound.load(std::memory_order_acquire)) {
        WaitWorker();
    }
    VkBindSparseInfo info = batch.Info();
    vk::Fence fence = device.GetLogical().CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    const auto start = std::chrono::steady_clock::now();
    VkResult result;
    {
        std::scoped_lock lock{submit_mutex};
        // With the bit off, the graphics queue as before.
        const vk::Queue bind_queue = Common::Nx2Ab::On(Common::Nx2Ab::SparseBindAsync)
                                         ? device.GetSparseQueue()
                                         : device.GetGraphicsQueue();
        const VkSemaphore semaphore = sparse_timeline ? *sparse_timeline : VK_NULL_HANDLE;
        const u64 wait_value = sparse_timeline_value;
        const VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &wait_value,
            .signalSemaphoreValueCount = 0,
            .pSignalSemaphoreValues = nullptr,
        };
        if (semaphore != VK_NULL_HANDLE && wait_value != 0) {
            info.pNext = &timeline_info;
            info.waitSemaphoreCount = 1;
            info.pWaitSemaphores = &semaphore;
        }
        result = bind_queue.BindSparse(info, *fence);
    }
    if (result != VK_SUCCESS) {
        sparse_stats.failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    fence.Wait();
    sparse_stats.waited_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count()),
        std::memory_order_relaxed);
    return true;
}

void Scheduler::FlushSparseBinds(u64 serial) {
    if (serial == 0 || pending_sparse_serial.load(std::memory_order_relaxed) != serial) {
        return;
    }
    if (!BindSparseNow(SparseBindBatch{})) {
        LOG_ERROR(Render_Vulkan, "Sparse binds queued for a destroyed resource failed to bind");
    }
}

u64 Scheduler::SubmitSparse(SparseBindBatch& batch) {
    const u64 value = sparse_timeline_value + 1;
    const u64 wait_value = sparse_timeline_value;
    const VkSemaphore semaphore = *sparse_timeline;
    // Each batch also waits for the one before it: batches stay in order.
    const VkTimelineSemaphoreSubmitInfo timeline_info{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = wait_value != 0 ? 1U : 0U,
        .pWaitSemaphoreValues = wait_value != 0 ? &wait_value : nullptr,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &value,
    };
    VkBindSparseInfo info = batch.Info();
    info.pNext = &timeline_info;
    info.waitSemaphoreCount = wait_value != 0 ? 1U : 0U;
    info.pWaitSemaphores = wait_value != 0 ? &semaphore : nullptr;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &semaphore;
    const VkResult result = device.GetSparseQueue().BindSparse(info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        // The submission runs without them: the blocks read as unbound.
        if (sparse_stats.failed.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Vulkan, "Sparse binds: {} requests ({} binds) failed to bind ({})",
                      batch.Requests(), batch.Binds(), static_cast<int>(result));
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            device.ReportLoss();
        }
        return 0;
    }
    sparse_timeline_value = value;
    sparse_stats.batches.fetch_add(1, std::memory_order_relaxed);
    sparse_in_flight.emplace_back(value, std::chrono::steady_clock::now());
    return value;
}

void Scheduler::TickSparse() {
    const auto now = std::chrono::steady_clock::now();
    if (!sparse_in_flight.empty()) {
        // What a fence wait per batch would have cost the recording thread, at most (seen done
        // at a later submission).
        const u64 done = sparse_timeline.GetCounter();
        while (!sparse_in_flight.empty() && sparse_in_flight.front().first <= done) {
            sparse_stats.unwaited_ns.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     now - sparse_in_flight.front().second)
                                     .count()),
                std::memory_order_relaxed);
            sparse_in_flight.pop_front();
        }
    }
    if (sparse_next_log == std::chrono::steady_clock::time_point{}) {
        sparse_next_log = now + std::chrono::seconds{30};
        return;
    }
    if (now < sparse_next_log) {
        return;
    }
    sparse_next_log = now + std::chrono::seconds{30};
    const u64 queued = sparse_stats.queued.exchange(0, std::memory_order_relaxed);
    const u64 at_once = sparse_stats.now.exchange(0, std::memory_order_relaxed);
    const u64 binds = sparse_stats.binds.exchange(0, std::memory_order_relaxed);
    const u64 batches = sparse_stats.batches.exchange(0, std::memory_order_relaxed);
    const u64 failed = sparse_stats.failed.exchange(0, std::memory_order_relaxed);
    const u64 waited_ns = sparse_stats.waited_ns.exchange(0, std::memory_order_relaxed);
    const u64 unwaited_ns = sparse_stats.unwaited_ns.exchange(0, std::memory_order_relaxed);
    if (queued + at_once + failed == 0) {
        return;
    }
    // grep "Sparse binds last 30 s"
    LOG_INFO(Render_Vulkan,
             "Sparse binds last 30 s (A/B bit SparseBindAsync {}, {} queue): {} requests ({:.1f}/s, "
             "{} memory binds): {} queued in {} batches ({:.1f}/s), {} bound at once (fence waits "
             "{:.1f} ms); queued batches' GPU time nothing waited for: {:.1f} ms (at most); {} failed",
             Common::Nx2Ab::On(Common::Nx2Ab::SparseBindAsync) ? "on" : "off",
             device.HasOwnSparseQueue() ? "own" : "graphics", queued + at_once,
             static_cast<double>(queued + at_once) / 30.0, binds, queued, batches,
             static_cast<double>(batches) / 30.0, at_once, static_cast<double>(waited_ns) / 1e6,
             static_cast<double>(unwaited_ns) / 1e6, failed);
}

void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
