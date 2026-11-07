// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/stage.h"
#include "video_core/shader_environment.h"

namespace {

constexpr std::array<char, 8> MAGIC{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

template <typename T>
void Append(std::vector<char>& out, const T& value) {
    const char* const bytes{reinterpret_cast<const char*>(&value)};
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

/// One environment as GenericEnvironment::Serialize writes it: one code word, no recorded answers.
void AppendEnvironment(std::vector<char>& out, Shader::Stage stage) {
    Append(out, u64{sizeof(u64)}); // code size
    for (int count = 0; count < 5; ++count) {
        Append(out, u64{0}); // texture types, pixel formats, cbuf values, pointers, replacements
    }
    Append(out, u32{0}); // local memory size
    Append(out, u32{0}); // texture bound buffer
    Append(out, u32{0}); // start address
    Append(out, u32{0}); // lowest read
    Append(out, u32{0}); // highest read
    Append(out, u32{1}); // viewport transform state
    Append(out, stage);
    Append(out, u64{0}); // code
    if (stage == Shader::Stage::Compute) {
        Append(out, std::array<u32, 3>{8, 8, 1});
        Append(out, u32{0}); // shared memory
    } else {
        Append(out, std::array<char, sizeof(Shader::ProgramHeader)>{});
    }
}

void AppendEntry(std::vector<char>& out, Shader::Stage stage, std::span<const char> key) {
    Append(out, u32{1});
    AppendEnvironment(out, stage);
    out.insert(out.end(), key.begin(), key.end());
}

std::vector<char> Header(u32 version) {
    std::vector<char> out(MAGIC.begin(), MAGIC.end());
    Append(out, version);
    return out;
}

std::vector<char> Key(size_t size, char fill) {
    return std::vector<char>(size, fill);
}

void WriteFile(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

u32 FileVersion(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::array<char, 8> magic{};
    u32 version{};
    file.read(magic.data(), magic.size()).read(reinterpret_cast<char*>(&version), sizeof(version));
    return version;
}

/// Version 99 keys: compute 8 bytes, graphics 4. Version 100: compute 16 (the old bytes, then
/// 8 bytes of 0x5a); graphics keys can't be converted.
bool ConvertCompute(std::span<const char> old_key, std::vector<char>& new_key) {
    new_key.assign(old_key.begin(), old_key.end());
    new_key.resize(16, static_cast<char>(0x5a));
    return true;
}

bool DropGraphics(std::span<const char>, std::vector<char>&) {
    return false;
}

constexpr std::array<VideoCommon::PipelineKeyMigration, 1> MIGRATIONS{{
    {.from_version = 99,
     .compute_key_size = 8,
     .graphics_key_size = 4,
     .convert_compute = ConvertCompute,
     .convert_graphics = DropGraphics},
}};

struct Loaded {
    std::vector<std::vector<char>> compute_keys;
    size_t graphics{};
};

Loaded Load(const std::filesystem::path& path,
            std::span<const VideoCommon::PipelineKeyMigration> migrations) {
    Loaded loaded;
    std::stop_source stop;
    VideoCommon::LoadPipelines(
        stop.get_token(), path, 100,
        [&](std::ifstream& file, VideoCommon::FileEnvironment) {
            std::vector<char> key(16);
            file.read(key.data(), static_cast<std::streamsize>(key.size()));
            loaded.compute_keys.push_back(std::move(key));
        },
        [&](std::ifstream& file, std::vector<VideoCommon::FileEnvironment>) {
            std::array<char, 4> key{};
            file.read(key.data(), key.size());
            ++loaded.graphics;
        },
        false, migrations);
    return loaded;
}

std::filesystem::path TestPath(const char* name) {
    return std::filesystem::temp_directory_path() /
           (std::string{"eden-pipeline-migration-"} + name + ".bin");
}

} // Anonymous namespace

TEST_CASE("Pipeline cache migration keeps entries and converts their keys",
          "[video_core][pipeline_cache]") {
    const auto path{TestPath("convert")};
    std::vector<char> bytes{Header(99)};
    AppendEntry(bytes, Shader::Stage::Compute, Key(8, 'a'));
    AppendEntry(bytes, Shader::Stage::VertexB, Key(4, 'b'));
    AppendEntry(bytes, Shader::Stage::Compute, Key(8, 'c'));
    WriteFile(path, bytes);

    const Loaded loaded{Load(path, MIGRATIONS)};
    REQUIRE(loaded.graphics == 0);
    REQUIRE(loaded.compute_keys.size() == 2);
    std::vector<char> expected_a{Key(8, 'a')};
    expected_a.resize(16, static_cast<char>(0x5a));
    std::vector<char> expected_c{Key(8, 'c')};
    expected_c.resize(16, static_cast<char>(0x5a));
    REQUIRE(loaded.compute_keys[0] == expected_a);
    REQUIRE(loaded.compute_keys[1] == expected_c);
    REQUIRE(FileVersion(path) == 100);

    // The rewritten file is an ordinary current one.
    const Loaded again{Load(path, {})};
    REQUIRE(again.compute_keys == loaded.compute_keys);
    std::filesystem::remove(path);
}

TEST_CASE("Pipeline cache of a version without a migration is deleted",
          "[video_core][pipeline_cache]") {
    const auto path{TestPath("delete")};
    std::vector<char> bytes{Header(98)};
    AppendEntry(bytes, Shader::Stage::Compute, Key(8, 'a'));
    WriteFile(path, bytes);

    const Loaded loaded{Load(path, MIGRATIONS)};
    REQUIRE(loaded.compute_keys.empty());
    REQUIRE(!std::filesystem::exists(path));
}

TEST_CASE("Pipeline cache migration drops a partly written last entry",
          "[video_core][pipeline_cache]") {
    const auto path{TestPath("truncated")};
    std::vector<char> bytes{Header(99)};
    AppendEntry(bytes, Shader::Stage::Compute, Key(8, 'a'));
    std::vector<char> last;
    AppendEntry(last, Shader::Stage::Compute, Key(8, 'c'));
    bytes.insert(bytes.end(), last.begin(),
                 last.begin() + static_cast<std::ptrdiff_t>(last.size() / 2));
    WriteFile(path, bytes);

    const Loaded loaded{Load(path, MIGRATIONS)};
    REQUIRE(loaded.compute_keys.size() == 1);
    REQUIRE(loaded.compute_keys[0][0] == 'a');
    REQUIRE(FileVersion(path) == 100);
    std::filesystem::remove(path);
}
