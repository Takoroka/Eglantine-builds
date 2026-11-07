// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "common/nx2_ab.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vertex_repack.h"

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/texture_cache/util.h"

namespace Vulkan {
namespace {
VkBufferCopy MakeBufferCopy(const VideoCommon::BufferCopy& copy) {
    return VkBufferCopy{
        .srcOffset = copy.src_offset,
        .dstOffset = copy.dst_offset,
        .size = copy.size,
    };
}

// EDEN_NX2_READBACK_VERIFY=1: BufferCacheRuntime::VerifyReadback.
const bool readback_verify = [] {
    const char* const value = std::getenv("EDEN_NX2_READBACK_VERIFY");
    return value != nullptr && value[0] == '1';
}();

void Nx2CountReadback(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

VkIndexType IndexTypeFromNumElements(const Device& device, u32 num_elements) {
    if (num_elements <= 0xff && device.IsExtIndexTypeUint8Supported()) {
        return VK_INDEX_TYPE_UINT8_EXT;
    }
    if (num_elements <= 0xffff) {
        return VK_INDEX_TYPE_UINT16;
    }
    return VK_INDEX_TYPE_UINT32;
}

size_t BytesPerIndex(VkIndexType index_type) {
    switch (index_type) {
    case VK_INDEX_TYPE_UINT8_EXT:
        return 1;
    case VK_INDEX_TYPE_UINT16:
        return 2;
    case VK_INDEX_TYPE_UINT32:
        return 4;
    default:
        ASSERT_MSG(false, "Invalid index type={}", index_type);
        return 1;
    }
}

vk::Buffer CreateBuffer(const Device& device, const MemoryAllocator& memory_allocator, u64 size,
                        VkDeviceSize sparse_alignment) {
    VkBufferUsageFlags flags =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsExtConditionalRendering()) {
        flags |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    const VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = flags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (sparse_alignment > 1) {
        return memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal, sparse_alignment);
    }
    return memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);
}
} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase(null_params), scheduler{&runtime.scheduler}, tracker{4096} {
    if (runtime.device.HasNullDescriptor()) {
        return;
    }
    device = &runtime.device;
    buffer = runtime.CreateNullBuffer();
    is_null = true;
    if (device->IsBufferDeviceAddressSupported()) {
        device_address = device->GetLogical().GetBufferDeviceAddress(*buffer);
    }
}

Buffer::Buffer(BufferCacheRuntime& runtime, DAddr cpu_addr_, u64 size_bytes_,
               bool sparse_compatible_)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), device{&runtime.device},
      scheduler{&runtime.scheduler},
      buffer{CreateBuffer(*device, runtime.memory_allocator, SizeBytes(),
                          runtime.SparseAlignmentFor(sparse_compatible_))},
      tracker{SizeBytes()} {
    sparse_compatible = sparse_compatible_;
    if (runtime.device.HasDebuggingToolAttached()) {
        buffer.SetObjectNameEXT(fmt::format("Buffer {:#x}", CpuAddr()).c_str());
    }
    if (device->IsBufferDeviceAddressSupported()) {
        device_address = device->GetLogical().GetBufferDeviceAddress(*buffer);
    }
}

void Buffer::MarkUsage(u64 offset, u64 size) noexcept {
    tracker.Track(offset, size);
    last_usage_tick = scheduler->CurrentTick();
}

VkBufferView Buffer::View(u32 offset, u32 size, VideoCore::Surface::PixelFormat format) {
    if (!device) {
        // Null buffer supported, return a null descriptor
        return VK_NULL_HANDLE;
    } else if (is_null) {
        // Null buffer not supported, adjust offset and size
        offset = 0;
        size = 0;
    }
    const auto it{std::ranges::find_if(views, [offset, size, format](const BufferView& view) {
        return offset == view.offset && size == view.size && format == view.format;
    })};
    if (it != views.end()) {
        return *it->handle;
    }
    views.push_back({
        .offset = offset,
        .size = size,
        .format = format,
        .handle = device->GetLogical().CreateBufferView({
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .buffer = *buffer,
            .format = MaxwellToVK::SurfaceFormat(*device, FormatType::Buffer, false, format).format,
            .offset = offset,
            .range = size,
        }),
    });
    return *views.back().handle;
}

class QuadIndexBuffer {
public:
    QuadIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                    Scheduler& scheduler_, StagingBufferPool& staging_pool_)
        : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
          staging_pool{staging_pool_} {}

    virtual ~QuadIndexBuffer() = default;

    void UpdateBuffer(u32 num_indices_) {
        if (num_indices_ <= num_indices) {
            return;
        }

        scheduler.Finish();

        num_indices = num_indices_;
        index_type = IndexTypeFromNumElements(device, num_indices);

        const u32 num_quads = GetQuadsNum(num_indices);
        const u32 num_triangle_indices = num_quads * 6;
        const u32 num_first_offset_copies = 4;
        const size_t bytes_per_index = BytesPerIndex(index_type);
        const size_t size_bytes = num_triangle_indices * bytes_per_index * num_first_offset_copies;
        const VkBufferCreateInfo buffer_ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size_bytes,
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        buffer = memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);
        if (device.HasDebuggingToolAttached()) {
            buffer.SetObjectNameEXT("Quad LUT");
        }

        const bool host_visible = buffer.IsHostVisible();
        const StagingBufferRef staging = [&] {
            if (host_visible) {
                return StagingBufferRef{};
            }
            return staging_pool.Request(size_bytes, MemoryUsage::Upload);
        }();

        u8* staging_data = host_visible ? buffer.Mapped().data() : staging.mapped_span.data();
        const size_t quad_size = bytes_per_index * 6;

        for (u32 first = 0; first < num_first_offset_copies; ++first) {
            for (u32 quad = 0; quad < num_quads; ++quad) {
                MakeAndUpdateIndices(staging_data, quad_size, quad, first);
                staging_data += quad_size;
            }
        }

        if (!host_visible) {
            scheduler.RequestOutsideRenderPassOperationContext();
            scheduler.Record([src_buffer = staging.buffer, src_offset = staging.offset,
                              dst_buffer = *buffer, size_bytes](vk::CommandBuffer cmdbuf) {
                const VkBufferCopy copy{
                    .srcOffset = src_offset,
                    .dstOffset = 0,
                    .size = size_bytes,
                };
                const VkBufferMemoryBarrier write_barrier{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = dst_buffer,
                    .offset = 0,
                    .size = size_bytes,
                };
                cmdbuf.CopyBuffer(src_buffer, dst_buffer, copy);
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, write_barrier);
            });
        } else {
            buffer.Flush();
        }
    }

    void BindBuffer(u32 first) {
        const VkIndexType index_type_ = index_type;
        const size_t sub_first_offset = static_cast<size_t>(first % 4) * GetQuadsNum(num_indices);
        const size_t offset =
            (sub_first_offset + GetQuadsNum(first)) * 6ULL * BytesPerIndex(index_type);
        scheduler.Record([buffer_ = *buffer, index_type_, offset](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindIndexBuffer(buffer_, offset, index_type_);
        });
    }

protected:
    virtual u32 GetQuadsNum(u32 num_indices) const = 0;

    virtual void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) = 0;

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;

    vk::Buffer buffer{};
    MemoryCommit memory_commit{};
    VkIndexType index_type{};
    u32 num_indices = 0;
};

class QuadArrayIndexBuffer : public QuadIndexBuffer {
public:
    QuadArrayIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                         Scheduler& scheduler_, StagingBufferPool& staging_pool_)
        : QuadIndexBuffer(device_, memory_allocator_, scheduler_, staging_pool_) {}

    ~QuadArrayIndexBuffer() = default;

private:
    u32 GetQuadsNum(u32 num_indices_) const override {
        return num_indices_ / 4;
    }

    template <typename T>
    static std::array<T, 6> MakeIndices(u32 quad, u32 first) {
        std::array<T, 6> indices{0, 1, 2, 0, 2, 3};
        for (T& index : indices) {
            index = static_cast<T>(first + index + quad * 4);
        }
        return indices;
    }

    void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) override {
        switch (index_type) {
        case VK_INDEX_TYPE_UINT8_EXT:
            std::memcpy(staging_data, MakeIndices<u8>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT16:
            std::memcpy(staging_data, MakeIndices<u16>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT32:
            std::memcpy(staging_data, MakeIndices<u32>(quad, first).data(), quad_size);
            break;
        default:
            ASSERT(false);
            break;
        }
    }
};

class QuadStripIndexBuffer : public QuadIndexBuffer {
public:
    QuadStripIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                         Scheduler& scheduler_, StagingBufferPool& staging_pool_)
        : QuadIndexBuffer(device_, memory_allocator_, scheduler_, staging_pool_) {}

    ~QuadStripIndexBuffer() = default;

private:
    u32 GetQuadsNum(u32 num_indices_) const override {
        return num_indices_ >= 4 ? (num_indices_ - 2) / 2 : 0;
    }

    template <typename T>
    static std::array<T, 6> MakeIndices(u32 quad, u32 first) {
        std::array<T, 6> indices{0, 3, 1, 0, 2, 3};
        for (T& index : indices) {
            index = static_cast<T>(first + index + quad * 2);
        }
        return indices;
    }

    void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) override {
        switch (index_type) {
        case VK_INDEX_TYPE_UINT8_EXT:
            std::memcpy(staging_data, MakeIndices<u8>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT16:
            std::memcpy(staging_data, MakeIndices<u16>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT32:
            std::memcpy(staging_data, MakeIndices<u32>(quad, first).data(), quad_size);
            break;
        default:
            ASSERT(false);
            break;
        }
    }
};

BufferCacheRuntime::BufferCacheRuntime(const Device& device_, MemoryAllocator& memory_allocator_,
                                       Scheduler& scheduler_, StagingBufferPool& staging_pool_,
                                       GuestDescriptorQueue& guest_descriptor_queue_,
                                       ComputePassDescriptorQueue& compute_pass_descriptor_queue,
                                       DescriptorPool& descriptor_pool)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      staging_pool{staging_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      quad_index_pass(device, scheduler, descriptor_pool, staging_pool,
                      compute_pass_descriptor_queue),
      multi_range_buffers(device_) {
    readback_queue = ReadbackQueue::Create(device, scheduler.GetMasterSemaphore());
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool enable_vertex_repack = VertexRepackEnabled();
    LOG_INFO(Render_Vulkan, "69-byte vertex layout repacking {} ({})",
             enable_vertex_repack ? "enabled" : "disabled",
             "portable Vulkan layout fallback");
    static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());
    LOG_INFO(Render_Vulkan, "Vertex repacking without dynamic vertex input {}",
             static_vertex_repack ? "enabled" : "disabled");
    if (enable_vertex_repack) {
        vertex_repack_pass = std::make_unique<VertexRepackPass>(
            device, scheduler, descriptor_pool, staging_pool, compute_pass_descriptor_queue);
    }
    limit_dynamic_storage_buffers = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                    driver_id == VK_DRIVER_ID_ARM_PROPRIETARY;
    if (limit_dynamic_storage_buffers) {
        max_dynamic_storage_buffers = device.GetMaxDescriptorSetStorageBuffersDynamic();
    }    
    if (device.SupportsUint8Indices()) {
        uint8_pass = std::make_unique<Uint8Pass>(device, scheduler, descriptor_pool, staging_pool,
                                                 compute_pass_descriptor_queue);
    }
    quad_array_index_buffer = std::make_shared<QuadArrayIndexBuffer>(device_, memory_allocator_,
                                                                     scheduler_, staging_pool_);
    quad_strip_index_buffer = std::make_shared<QuadStripIndexBuffer>(device_, memory_allocator_,
                                                                     scheduler_, staging_pool_);
}

u32 BufferCacheRuntime::SetVertexRepackLayout(const Tegra::Engines::Maxwell3D::Regs& regs,
                                            bool dynamic_input) {
    const u32 old_mask = vertex_repack_mask;
    const bool input_supported = VertexRepackInputSupported(dynamic_input, static_vertex_repack);
    vertex_repack_mask = vertex_repack_pass && input_supported ? VertexRepackMask(regs) : 0;
    if (vertex_repack_mask != old_mask) {
        LOG_DEBUG(Render_Vulkan, "Vertex repack selection: mask={:#x}, dynamic_input={}",
                  vertex_repack_mask, dynamic_input);
    }
    return old_mask ^ vertex_repack_mask;
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, MemoryUsage::Upload);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Download, deferred);
}

VkFormat BufferCacheRuntime::TexelBufferFormat(VideoCore::Surface::PixelFormat format) const {
    return MaxwellToVK::SurfaceFormat(device, FormatType::Buffer, false, format).format;
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return device.GetDeviceLocalMemory();
}

u64 BufferCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetDeviceMemoryUsage();
}

bool BufferCacheRuntime::CanReportMemoryUsage() const {
    return device.CanReportMemoryUsage();
}

u32 BufferCacheRuntime::GetUniformBufferAlignment() const {
    return static_cast<u32>(device.GetUniformBufferAlignment());
}

u32 BufferCacheRuntime::GetStorageBufferAlignment() const {
    return static_cast<u32>(device.GetStorageBufferAlignment());
}

void BufferCacheRuntime::TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept {
    for (auto it = slot_buffers.begin(); it != slot_buffers.end(); it++) {
        if (scheduler.IsFree(it->LastUsageTick())) {
            it->ResetUsageTracking();
        }
    }
}

u64 BufferCacheRuntime::CurrentTick() {
    return scheduler.CurrentTick();
}

bool BufferCacheRuntime::IsFree(u64 tick) {
    return scheduler.IsFree(tick);
}

void BufferCacheRuntime::Wait(u64 tick) {
    scheduler.Wait(tick);
}

void BufferCacheRuntime::WaitSubmittedTick(u64 tick) {
    auto& master = scheduler.GetMasterSemaphore();
    if (master.IsFree(tick)) {
        return;
    }
    // The copy's submission may be deferred (Scheduler::DeferFlush): the split's front asks the
    // back for it rather than waiting for the back's next submission. If the back has stopped,
    // nothing else uses the scheduler and Wait submits it.
    if (!scheduler.IsSubmitted(tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread() && !scheduler.WaitSubmitted(tick)) {
        scheduler.Wait(tick);
        return;
    }
    master.Wait(tick);
}

void BufferCacheRuntime::Nx2WriteCopyMarkers(
    std::span<const std::pair<VkBuffer, VkDeviceSize>> markers) {
    if (markers.empty()) {
        return;
    }
    // The copies' writes land before the markers' (transfer to transfer), and both reach the host.
    static constexpr VkMemoryBarrier COPY_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    static constexpr VkMemoryBarrier HOST_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    boost::container::small_vector<std::pair<VkBuffer, VkDeviceSize>, 8> list(markers.begin(),
                                                                               markers.end());
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([list = std::move(list)](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                               COPY_BARRIER);
        for (const auto& [buffer, offset] : list) {
            cmdbuf.FillBuffer(buffer, offset, sizeof(u32), 1);
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               HOST_BARRIER);
    });
}

bool BufferCacheRuntime::Nx2ListCopyMarked(const u32* marker) {
    if (!nx2_list_copy_poll.load(std::memory_order_relaxed)) {
        return false;
    }
    const bool marked = *static_cast<const volatile u32*>(marker) != 0;
    std::atomic_thread_fence(std::memory_order_acquire);
    return marked;
}

void BufferCacheRuntime::Nx2WaitListCopy(const u32* marker, u64 tick) {
    // The copy is the last work of its submission, but the fence signals after the whole
    // submission and the wait wakes this thread through the OS: watch the GPU's marker for up to
    // 2 ms (Mario Kart World's races waited ~0.3 ms per copy), then wait for the fence.
    if (nx2_list_copy_poll.load(std::memory_order_relaxed)) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 spins = 1;; ++spins) {
            if (Nx2ListCopyMarked(marker)) {
                return;
            }
            if (spins % 256 == 0) {
                if (scheduler.IsFree(tick) ||
                    std::chrono::steady_clock::now() - start > std::chrono::milliseconds{2}) {
                    break;
                }
            }
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
            _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield");
#endif
        }
    }
    scheduler.GetMasterSemaphore().Wait(tick);
}

void BufferCacheRuntime::Finish() {
    scheduler.Finish();
}

bool BufferCacheRuntime::CanReorderUpload(const Buffer& buffer,
                                          std::span<const VideoCommon::BufferCopy> copies) {
    if (Settings::values.disable_buffer_reorder) {
        return false;
    }
    const bool can_use_upload_cmdbuf =
        std::ranges::all_of(copies, [&](const VideoCommon::BufferCopy& copy) {
            return !buffer.IsRegionUsed(copy.dst_offset, copy.size);
        });
    return can_use_upload_cmdbuf;
}

void BufferCacheRuntime::CopyBuffer(VkBuffer dst_buffer, VkBuffer src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    if (dst_buffer == VK_NULL_HANDLE || src_buffer == VK_NULL_HANDLE) {
        return;
    }
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    static constexpr VkMemoryBarrier WRITE_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };

    // Measuring a popular game, this number never exceeds the specified size once data is warmed up
    boost::container::small_vector<VkBufferCopy, 8> vk_copies(copies.size());
    std::ranges::transform(copies, vk_copies.begin(), MakeBufferCopy);
    if (src_buffer == staging_pool.StreamBuf() && can_reorder_upload) {
        scheduler.RecordWithUploadBuffer([src_buffer, dst_buffer, vk_copies](
                                             vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
            upload_cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
        });
        return;
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src_buffer, dst_buffer, vk_copies, barrier](vk::CommandBuffer cmdbuf) {
        if (barrier) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        }
        cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
        if (barrier) {
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE, 0, WRITE_BARRIER);
        }
    });
}

void BufferCacheRuntime::PreCopyBarrier() {
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, READ_BARRIER);
    });
}

void BufferCacheRuntime::PostCopyBarrier() {
    static constexpr VkMemoryBarrier WRITE_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                               0, WRITE_BARRIER);
    });
}

void BufferCacheRuntime::HostReadBarrier() {
    // Waiting for the fence doesn't make device writes visible to the host by itself: the copies'
    // writes need a dependency on the host stage first. The post-copy barriers only reach the
    // graphics and compute stages. Download memory is host coherent, so no invalidation follows.
    if (!Common::Nx2Ab::On(Common::Nx2Ab::HostReadBarrier)) {
        return;
    }
    static constexpr VkMemoryBarrier HOST_READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               HOST_READ_BARRIER);
    });
}

bool BufferCacheRuntime::ReadbackCopies(
    const StagingBufferRef& staging,
    std::span<const std::pair<Buffer*, VideoCommon::BufferCopy>> copies) {
    if (!Common::Nx2Ab::On(Common::Nx2Ab::ReadbackQueue)) {
        return false;
    }
    if (!readback_queue) {
        Nx2CountReadback(readback_stats.no_queue);
        return false;
    }
    // The last submission that used a source: its writes are what the copies must see. A source
    // used in the command buffer still recording can't be waited for apart from the rest of it.
    u64 wait_tick = 0;
    for (const auto& [buffer, copy] : copies) {
        wait_tick = (std::max)({wait_tick, buffer->LastUsageTick(), buffer->GpuWriteTick(),
                                buffer->getWriteTick()});
    }
    if (!scheduler.IsSubmitted(wait_tick)) {
        Nx2CountReadback(readback_stats.recording);
        return false;
    }
    boost::container::small_vector<ReadbackQueue::Region, 16> regions;
    u64 bytes = 0;
    for (const auto& [buffer, copy] : copies) {
        regions.push_back({buffer->Handle(), MakeBufferCopy(copy)});
        bytes += copy.size;
    }
    const u64 waited = readback_queue->CopyAndWait(staging.buffer, regions, wait_tick);
    Nx2CountReadback(readback_stats.copies);
    Nx2CountReadback(readback_stats.copy_ns, waited);
    Nx2CountReadback(readback_stats.bytes, bytes);
    if (readback_verify) [[unlikely]] {
        VerifyReadback(staging, copies);
    }
    MaybeLogReadbacks();
    return true;
}

void BufferCacheRuntime::FinishReadback() {
    const auto start = std::chrono::steady_clock::now();
    scheduler.Finish();
    Nx2CountReadback(readback_stats.finishes);
    Nx2CountReadback(readback_stats.finish_ns,
                     static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count()));
    MaybeLogReadbacks();
}

void BufferCacheRuntime::VerifyReadback(
    const StagingBufferRef& staging,
    std::span<const std::pair<Buffer*, VideoCommon::BufferCopy>> copies) {
    u64 end = 0;
    for (const auto& [buffer, copy] : copies) {
        end = (std::max)(end, copy.dst_offset + copy.size - staging.offset);
    }
    const StagingBufferRef check = staging_pool.Request(end, MemoryUsage::Download);
    for (const auto& [buffer, copy] : copies) {
        VideoCommon::BufferCopy moved = copy;
        moved.dst_offset = copy.dst_offset - staging.offset + check.offset;
        buffer->MarkUsage(copy.src_offset, copy.size);
        CopyBuffer(check.buffer, *buffer, std::span{&moved, 1}, true);
    }
    HostReadBarrier();
    scheduler.Finish();
    Nx2CountReadback(readback_stats.verified);
    bool mismatched = false;
    for (const auto& [buffer, copy] : copies) {
        const u64 offset = copy.dst_offset - staging.offset;
        u8* const fast = staging.mapped_span.data() + offset;
        const u8* const slow = check.mapped_span.data() + offset;
        if (std::memcmp(fast, slow, copy.size) == 0) {
            continue;
        }
        if (!mismatched) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_WARNING(Render_Vulkan,
                            "Readback queue verify: buffer {:#x} [{:#x}, +{:#x}) differs from the "
                            "Finish() copy (last use tick {}, GPU write tick {})",
                            buffer->CpuAddr(), copy.src_offset, copy.size,
                            buffer->LastUsageTick(), buffer->GpuWriteTick());
            }
        }
        mismatched = true;
        std::memcpy(fast, slow, copy.size);
    }
    if (mismatched) {
        Nx2CountReadback(readback_stats.mismatched);
    }
}

void BufferCacheRuntime::MaybeLogReadbacks() {
    using namespace std::chrono;
    const s64 now = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    s64 next = readback_stats.next_log.load(std::memory_order_relaxed);
    if (now < next || !readback_stats.next_log.compare_exchange_strong(
                          next, now + duration_cast<nanoseconds>(30s).count(),
                          std::memory_order_relaxed)) {
        return;
    }
    const auto load = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    const ReadbackTotals totals{
        .copies = load(readback_stats.copies),
        .copy_ns = load(readback_stats.copy_ns),
        .bytes = load(readback_stats.bytes),
        .finishes = load(readback_stats.finishes),
        .finish_ns = load(readback_stats.finish_ns),
        .recording = load(readback_stats.recording),
        .no_queue = load(readback_stats.no_queue),
        .verified = load(readback_stats.verified),
        .mismatched = load(readback_stats.mismatched),
    };
    const ReadbackTotals prev = std::exchange(readback_last, totals);
    const s64 start = std::exchange(readback_last_log, now);
    if (start == 0) {
        return; // The first window starts here.
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    const auto rate = [seconds](u64 value, u64 before) {
        return static_cast<double>(value - before) / seconds;
    };
    const auto mean_us = [](u64 ns, u64 ns_before, u64 count, u64 count_before) {
        const u64 n = count - count_before;
        return n == 0 ? 0.0 : static_cast<double>(ns - ns_before) / 1e3 / static_cast<double>(n);
    };
    std::string verify;
    if (readback_verify) {
        verify = fmt::format("; verify: {} compared, {} differed", totals.verified - prev.verified,
                             totals.mismatched - prev.mismatched);
    }
    LOG_INFO(Render_Vulkan,
             "Readback queue (bit 11 {}) {:.0f} s: {:.0f}/s copied on the readback queue (mean "
             "{:.0f} us wait, {:.0f} KiB/s), {:.0f}/s through Finish() (mean {:.0f} us; {:.0f}/s "
             "with a source in the recording command buffer, {:.0f}/s without a readback "
             "queue){}",
             Common::Nx2Ab::On(Common::Nx2Ab::ReadbackQueue) ? "on" : "off", seconds,
             rate(totals.copies, prev.copies),
             mean_us(totals.copy_ns, prev.copy_ns, totals.copies, prev.copies),
             rate(totals.bytes, prev.bytes) / 1024.0, rate(totals.finishes, prev.finishes),
             mean_us(totals.finish_ns, prev.finish_ns, totals.finishes, prev.finishes),
             rate(totals.recording, prev.recording), rate(totals.no_queue, prev.no_queue), verify);
}

void BufferCacheRuntime::ClearBuffer(VkBuffer dest_buffer, u32 offset, size_t size, u32 value) {
    if (dest_buffer == VK_NULL_HANDLE) {
        return;
    }
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    static constexpr VkMemoryBarrier WRITE_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([dest_buffer, offset, size, value](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, READ_BARRIER);
        cmdbuf.FillBuffer(dest_buffer, offset, size, value);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                               0, WRITE_BARRIER);
    });
}

bool BufferCacheRuntime::BindMultiRangeStorageBuffer(u64 key, bool is_written) {
    if (multi_range_sources.empty() || multi_range_total == 0) {
        return false;
    }
    const MultiRangeRef ref = multi_range_buffers.Get(device, scheduler, memory_allocator, key,
                                                     multi_range_sources, multi_range_total);
    if (ref.handle == VK_NULL_HANDLE) {
        return false;
    }
    if (is_written && !ref.sparse) {
        return false;
    }
    if (ref.needs_gather) {
        PreCopyBarrier();
        VkDeviceSize dst_offset = 0;
        for (const MultiRangeSource& source : multi_range_sources) {
            const std::array<VideoCommon::BufferCopy, 1> copy{VideoCommon::BufferCopy{
                .src_offset = u64(source.offset),
                .dst_offset = u64(dst_offset),
                .size = size_t(source.size),
            }};
            CopyBuffer(ref.handle, source.handle, copy, false);
            dst_offset += source.size;
        }
        PostCopyBarrier();
        multi_range_buffers.MarkGathered(key);
    }
    guest_descriptor_queue.AddBuffer(ref.handle, ref.address, 0, ref.size);
    return true;
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 base_vertex, u32 num_indices, VkBuffer buffer,
                                         u32 offset, [[maybe_unused]] u32 size) {
    VkIndexType vk_index_type = MaxwellToVK::IndexFormat(index_format);
    VkDeviceSize vk_offset = offset;
    VkBuffer vk_buffer = buffer;
    if (topology == PrimitiveTopology::Quads || topology == PrimitiveTopology::QuadStrip) {
        vk_index_type = VK_INDEX_TYPE_UINT32;
        std::tie(vk_buffer, vk_offset) =
            quad_index_pass.Assemble(index_format, num_indices, base_vertex, buffer, offset,
                                     topology == PrimitiveTopology::QuadStrip);
    } else if (vk_index_type == VK_INDEX_TYPE_UINT8_EXT && !device.IsExtIndexTypeUint8Supported()) {
        vk_index_type = VK_INDEX_TYPE_UINT16;
        if (uint8_pass) {
            std::tie(vk_buffer, vk_offset) = uint8_pass->Assemble(num_indices, buffer, offset);
        } else if (device.GetDriverID() == VK_DRIVER_ID_QUALCOMM_PROPRIETARY) {
            ReserveNullBuffer();
            vk_buffer = *null_buffer;
            vk_offset = 0;
        }
    }
    if (vk_buffer == VK_NULL_HANDLE) {
        // Vulkan doesn't support null index buffers. Replace it with our own null buffer.
        ReserveNullBuffer();
        vk_buffer = *null_buffer;
    }
    scheduler.Record([vk_buffer, vk_offset, vk_index_type](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindIndexBuffer(vk_buffer, vk_offset, vk_index_type);
    });
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    if (count == 0) {
        ReserveNullBuffer();
        scheduler.Record([this](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindIndexBuffer(*null_buffer, 0, VK_INDEX_TYPE_UINT32);
        });
        return;
    }

    if (topology == PrimitiveTopology::Quads) {
        quad_array_index_buffer->UpdateBuffer(first + count);
        quad_array_index_buffer->BindBuffer(first);
    } else if (topology == PrimitiveTopology::QuadStrip) {
        quad_strip_index_buffer->UpdateBuffer(first + count);
        quad_strip_index_buffer->BindBuffer(first);
    }
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride) {
    if (index >= device.GetMaxVertexInputBindings()) {
        return;
    }
    if (buffer != VK_NULL_HANDLE && size != 0 && IsVertexRepacked(index)) {
        const auto repacked =
            vertex_repack_pass->Assemble(buffer, offset, size, u64{offset} + size);
        if (!repacked) {
            return;
        }
        buffer = repacked->first;
        offset = static_cast<u32>(repacked->second);
        size = static_cast<u32>((u64{size} + 68) / 69 * 72);
        stride = 72;
    }
    if (device.IsExtExtendedDynamicStateSupported()) {
        scheduler.Record([index, buffer, offset, size, stride](vk::CommandBuffer cmdbuf) {
            const VkDeviceSize vk_offset = buffer != VK_NULL_HANDLE ? offset : 0;
            const VkDeviceSize vk_size = buffer != VK_NULL_HANDLE ? size : VK_WHOLE_SIZE;
            const VkDeviceSize vk_stride = stride;
            cmdbuf.BindVertexBuffers2EXT(index, 1, &buffer, &vk_offset, &vk_size, &vk_stride);
        });
    } else {
        if (!device.HasNullDescriptor() && buffer == VK_NULL_HANDLE) {
            ReserveNullBuffer();
            buffer = *null_buffer;
            offset = 0;
        }
        scheduler.Record([index, buffer, offset](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffer(index, buffer, offset);
        });
    }
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    boost::container::static_vector<VkBuffer, VideoCommon::NUM_VERTEX_BUFFERS> buffer_handles(bindings.buffers.size());
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        auto handle = bindings.buffers[i]->Handle();
        if (handle == VK_NULL_HANDLE) {
            bindings.offsets[i] = 0;
            bindings.sizes[i] = VK_WHOLE_SIZE;
            if (!device.HasNullDescriptor()) {
                ReserveNullBuffer();
                handle = *null_buffer;
            }
        }
        buffer_handles[i] = handle;
        const u32 binding = bindings.min_index + i;
        if (handle != VK_NULL_HANDLE && IsVertexRepacked(binding) && bindings.strides[i] == 69 &&
            bindings.sizes[i] > 0) {
            const auto repacked = vertex_repack_pass->Assemble(
                handle, static_cast<u32>(bindings.offsets[i]), static_cast<u32>(bindings.sizes[i]),
                bindings.buffers[i]->SizeBytes());
            if (!repacked) {
                continue;
            }
            buffer_handles[i] = repacked->first;
            bindings.offsets[i] = repacked->second;
            bindings.sizes[i] = (bindings.sizes[i] + 68) / 69 * 72;
            bindings.strides[i] = 72;
            static bool reported = false;
            if (!reported) {
                LOG_INFO(Render_Vulkan, "GPU vertex repacking active: 69 -> 72 bytes");
                reported = true;
            }
        }
    }
    const u32 device_max = device.GetMaxVertexInputBindings();
    const u32 min_binding = (std::min)(bindings.min_index, device_max);
    const u32 max_binding = (std::min)(bindings.max_index, device_max);
    const u32 binding_count = max_binding - min_binding;
    if (binding_count == 0) {
        return;
    }
    if (device.IsExtExtendedDynamicStateSupported()) {
        scheduler.Record([bindings_ = std::move(bindings), buffer_handles_ = std::move(buffer_handles), binding_count](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffers2EXT(bindings_.min_index, binding_count, buffer_handles_.data(), bindings_.offsets.data(), bindings_.sizes.data(), bindings_.strides.data());
        });
    } else {
        scheduler.Record([bindings_ = std::move(bindings), buffer_handles_ = std::move(buffer_handles), binding_count](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffers(bindings_.min_index, binding_count, buffer_handles_.data(), bindings_.offsets.data());
        });
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffer(u32 index, VkBuffer buffer, u32 offset,
                                                     u32 size) {
    if (!device.IsExtTransformFeedbackSupported()) {
        // Already logged in the rasterizer
        return;
    }
    if (buffer == VK_NULL_HANDLE) {
        // Vulkan doesn't support null transform feedback buffers.
        // Replace it with our own null buffer.
        ReserveNullBuffer();
        buffer = *null_buffer;
        offset = 0;
        size = 0;
    }
    scheduler.Record([index, buffer, offset, size](vk::CommandBuffer cmdbuf) {
        const VkDeviceSize vk_offset = offset;
        const VkDeviceSize vk_size = size;
        cmdbuf.BindTransformFeedbackBuffersEXT(index, 1, &buffer, &vk_offset, &vk_size);
    });
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    if (!device.IsExtTransformFeedbackSupported()) {
        // Already logged in the rasterizer
        return;
    }
    boost::container::static_vector<VkBuffer, VideoCommon::NUM_VERTEX_BUFFERS> buffer_handles(bindings.buffers.size());
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        auto handle = bindings.buffers[i]->Handle();
        if (handle == VK_NULL_HANDLE) {
            ReserveNullBuffer();
            handle = *null_buffer;
            bindings.offsets[i] = 0;
            bindings.sizes[i] = 0;
        }
        buffer_handles[i] = handle;
    }
    scheduler.Record([bindings_ = std::move(bindings), buffer_handles_ = std::move(buffer_handles)](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindTransformFeedbackBuffersEXT(0, u32(buffer_handles_.size()), buffer_handles_.data(), bindings_.offsets.data(), bindings_.sizes.data());
    });
}

void BufferCacheRuntime::ReserveNullBuffer() {
    if (!null_buffer) {
        null_buffer = CreateNullBuffer();
    }
}

vk::Buffer BufferCacheRuntime::CreateNullBuffer() {
    VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = 4,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        create_info.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    vk::Buffer ret = memory_allocator.CreateBuffer(create_info, MemoryUsage::DeviceLocal);
    if (device.HasDebuggingToolAttached()) {
        ret.SetObjectNameEXT("Null buffer");
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([buffer = *ret](vk::CommandBuffer cmdbuf) {
        cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, 0);
    });

    return ret;
}

} // namespace Vulkan
