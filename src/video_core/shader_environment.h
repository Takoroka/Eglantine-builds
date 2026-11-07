// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"
#include "shader_recompiler/environment.h"
#include "video_core/engines/const_buffer_info.h"
#include "video_core/engines/maxwell_3d.h"

namespace Tegra {
class Memorymanager;
}

namespace VideoCommon {

/// Thrown when a program translated away from the GPU thread reads engine state that was not
/// copied for it. The pipeline is then built on the GPU thread.
class EngineStateMissing : public std::exception {
public:
    const char* what() const noexcept override {
        return "Engine state not copied for translation";
    }
};

/// A native (Switch 2) program after capture and lowering to Maxwell code. The shader cache keeps
/// it so the environments built for the program's pipelines don't capture and lower it again.
struct LoweredNativeCode {
    u64 diagnostic_key{};
    /// Shader header, lowered instructions with Maxwell scheduling words, then the native
    /// address map (what the environment's `code` used to be filled with).
    std::vector<u64> code;
    /// Bytes of the native program that were read, header included.
    size_t source_size{};
    /// Constant buffer holding the texture handles if the program requires one, otherwise ~0U
    /// (the engine's register value is used).
    u32 texture_bound = ~0U;
};

class GenericEnvironment : public Shader::Environment {
public:
    explicit GenericEnvironment() = default;
    explicit GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_);

    ~GenericEnvironment() override;

    [[nodiscard]] u32 TextureBoundBuffer() const final;

    [[nodiscard]] u32 LocalMemorySize() const final;

    [[nodiscard]] u32 SharedMemorySize() const final;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const final;

    [[nodiscard]] u64 ReadInstruction(u32 address) final;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] std::optional<u64> Analyze();

    void SetCachedSize(size_t size_bytes);

    [[nodiscard]] size_t CachedSizeWords() const noexcept;

    [[nodiscard]] size_t CachedSizeBytes() const noexcept;

    [[nodiscard]] size_t ReadSizeBytes() const noexcept;

    [[nodiscard]] size_t SourceSizeBytes() const noexcept {
        return native_source_size;
    }

    [[nodiscard]] const std::shared_ptr<const LoweredNativeCode>& LoweredNative() const noexcept {
        return lowered_native;
    }

    [[nodiscard]] bool CanBeSerialized() const noexcept;

    [[nodiscard]] u64 CalculateHash() const;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    /// with_mesh_state: also the mesh registers and pass-through header of task and mesh
    /// programs (only the mesh pipeline file carries them).
    void Serialize(std::ofstream& file, bool with_mesh_state = false) const;

    bool HasHLEMacroState() const override {
        return has_hle_engine_state;
    }

protected:
    std::optional<u64> TryFindSize();

    /// Adopts an already lowered native program, or captures and lowers it when there is none.
    void InitializeNativeCode(std::shared_ptr<const LoweredNativeCode> lowered,
                              u32 source_header_size = 128, bool geometry = false);
    std::shared_ptr<const LoweredNativeCode> LowerNativeCode(u32 source_header_size,
                                                              bool geometry = false);
    size_t native_source_size{};
    /// Lower matrix multiplies too (KeplerCompute::EmulatesNativeTensorPrograms).
    bool native_tensor{};
    /// ... in the compact form (KeplerCompute::UsesCompactTensorPrograms).
    bool native_compact_tensor{};
    std::shared_ptr<const LoweredNativeCode> lowered_native;

    Tegra::Texture::TICEntry ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                             bool via_header_index, u32 raw, bool unsafe = false);

    Tegra::MemoryManager* gpu_memory{};
    GPUVAddr program_base{};

    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types; ///< IsCbufStorageBufferPointer answers
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;

    u32 local_memory_size{};
    u32 texture_bound{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    u32 read_lowest = (std::numeric_limits<u32>::max)();
    u32 read_highest = 0;

    u32 cached_lowest = (std::numeric_limits<u32>::max)();
    u32 cached_highest = 0;
    u32 initial_offset = 0;

    u32 viewport_transform_state = 1;

    bool has_unbound_instructions = false;
    bool has_hle_engine_state = false;
    /// Translated on a pipeline worker: guest memory is only read without flushing.
    bool detached = false;
};

class GraphicsEnvironment final : public GenericEnvironment {
public:
    explicit GraphicsEnvironment() = default;
    explicit GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                 Tegra::MemoryManager& gpu_memory_,
                                 Tegra::Engines::Maxwell3D::Regs::ShaderType program,
                                 GPUVAddr program_base_, u32 start_address_,
                                 std::shared_ptr<const LoweredNativeCode> lowered_native_ = {});

    ~GraphicsEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;

    /// Copies the engine state translation reads (texture descriptor table, viewport transform,
    /// macro replacements and the constant buffers it reads values from), so the program can be
    /// translated on a pipeline worker while the GPU thread goes on. A read outside the copy
    /// throws EngineStateMissing.
    void DetachFromEngine();

    /// Frees the copied constant buffers, once the program is translated.
    void ReleaseEngineCopy() noexcept;

    /// Whether the texture descriptors a detached translation read still give the same types
    /// and formats (read again, on the GPU thread).
    [[nodiscard]] bool DescriptorsUnchanged();

private:
    static constexpr size_t NumConstBuffers = Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers;

    struct EngineCopy {
        GPUVAddr tic_address{};
        u32 tic_limit{};
        bool via_header_index{};
        u32 copied_buffers{}; ///< Mask of the constant buffers with data below
        std::array<Tegra::Engines::ConstBufferInfo, NumConstBuffers> const_buffers{};
        std::array<std::vector<u8>, NumConstBuffers> const_buffer_data;
        ::Common::unordered_map<u64, Tegra::Engines::Maxwell3D::HLEReplacementAttributeType>
            replace_table;
    };

    Tegra::Texture::TICEntry ReadTextureDescriptor(u32 handle);

    Tegra::Engines::Maxwell3D* maxwell3d{};
    size_t stage_index{};
    /// Shared by copies: environments are copied, not moved, and the copy is not changed once
    /// made (but for ReleaseEngineCopy, after translation).
    std::shared_ptr<EngineCopy> engine_copy;
};

/// The compute engine state a native program's translation reads, copied from a launch (its QMD)
/// and the engine's registers, so the program can be translated away from the GPU thread
/// (PipelineCache compute prebuilds).
struct ComputeEngineState {
    struct ConstBuffer {
        GPUVAddr address{};
        u32 size{};
    };
    std::array<ConstBuffer, 8> const_buffers{};
    u32 const_buffer_enable_mask{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};
    GPUVAddr tic_address{};
    u32 tic_limit{};
    u32 tex_cb_index{};
    bool linked_tsc{};
    bool native_tensor{};
    bool native_compact_tensor{};
};

class ComputeEnvironment final : public GenericEnvironment {
public:
    explicit ComputeEnvironment() = default;
    explicit ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_,
                                std::shared_ptr<const LoweredNativeCode> lowered_native_ = {});
    /// A native program's environment from a copy of the engine state, for a pipeline worker:
    /// guest memory is only read without flushing, and the program is captured and lowered here.
    explicit ComputeEnvironment(const ComputeEngineState& state, Tegra::MemoryManager& gpu_memory_,
                                GPUVAddr program_base_);

    ~ComputeEnvironment() override = default;

    /// Whether `current`, built on the GPU thread for a dispatch, has this environment's program
    /// and sizes and answers every query this environment's translation made the same: then
    /// translating with it gives the same program.
    [[nodiscard]] bool SameAnswers(ComputeEnvironment& current);

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(
        [[maybe_unused]] u32 bank, [[maybe_unused]] u32 offset) override {
        return std::nullopt;
    }

private:
    /// Whether the launch enables this constant buffer.
    [[nodiscard]] bool ConstBufferEnabled(u32 cbuf_index) const;
    /// A texture descriptor of the copied state (unflushed); throws EngineStateMissing for a
    /// handle outside the copied table.
    [[nodiscard]] Tegra::Texture::TICEntry ReadCopiedTextureInfo(u32 handle);

    Tegra::Engines::KeplerCompute* kepler_compute{};
    /// The engine state of an environment built from a copy (kepler_compute is null then).
    std::optional<ComputeEngineState> state_copy;
};

class FileEnvironment final : public Shader::Environment {
public:
    FileEnvironment() = default;
    ~FileEnvironment() override = default;

    FileEnvironment& operator=(FileEnvironment&&) noexcept = default;
    FileEnvironment(FileEnvironment&&) noexcept = default;

    FileEnvironment& operator=(const FileEnvironment&) = delete;
    FileEnvironment(const FileEnvironment&) = delete;

    void Deserialize(std::ifstream& file, bool with_mesh_state = false);

    [[nodiscard]] u64 ReadInstruction(u32 address) override;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    [[nodiscard]] Shader::TextureType ReadTextureType(u32 handle) override;

    [[nodiscard]] Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    [[nodiscard]] bool IsTexturePixelFormatInteger(u32 handle) override;

    [[nodiscard]] u32 ReadViewportTransformState() override;

    [[nodiscard]] u32 LocalMemorySize() const override;

    [[nodiscard]] u32 SharedMemorySize() const override;

    [[nodiscard]] u32 TextureBoundBuffer() const override;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const override;

    [[nodiscard]] std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) override;

    [[nodiscard]] bool HasHLEMacroState() const override {
        return cbuf_replacements.size() != 0;
    }

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

private:
    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types;
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;
    std::array<u32, 3> workgroup_size{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 initial_offset{};
    u32 viewport_transform_state = 1;
};

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version,
                       bool with_mesh_state = false);

template <typename Key, typename Envs>
    requires std::is_trivially_copyable_v<Key>
        && std::has_unique_object_representations_v<Key>
void SerializePipeline(const Key& key, const Envs& envs, const std::filesystem::path& filename, u32 cache_version,
                       bool with_mesh_state = false) {
    SerializePipeline(std::span(reinterpret_cast<const char*>(&key), sizeof(key)), std::span(envs.data(), envs.size()), filename, cache_version,
                      with_mesh_state);
}

void LoadPipelines(std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics,
    bool with_mesh_state = false);

} // namespace VideoCommon
