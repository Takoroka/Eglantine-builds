// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "common/game_presets.h"

namespace {
void SetConflictingPreferences(Settings::Values& settings, bool per_game) {
    const auto set = [per_game](auto& setting, auto value) {
        setting.SetGlobal(!per_game);
        setting.SetValue(value);
    };
    set(settings.renderer_backend, Settings::RendererBackend::Null);
    set(settings.resolution_setup, Settings::ResolutionSetup::Res2X);
    set(settings.gpu_accuracy, Settings::GpuAccuracy::High);
    set(settings.use_asynchronous_shaders, false);
    set(settings.astc_recompression, Settings::AstcRecompression::Uncompressed);
    set(settings.vram_usage_mode, Settings::VramUsageMode::Aggressive);
    set(settings.use_reactive_flushing, true);
    set(settings.vsync_mode, Settings::VSyncMode::Immediate);
    set(settings.scaling_filter, Settings::ScalingFilter::NearestNeighbor);
    set(settings.anti_aliasing, Settings::AntiAliasing::Smaa);
}

void CheckPreset(const Settings::Values& settings) {
    CHECK(settings.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
    CHECK(settings.resolution_setup.GetValue() == Settings::ResolutionSetup::Res1X);
    CHECK(settings.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(settings.use_asynchronous_shaders.GetValue());
    CHECK(settings.astc_recompression.GetValue() == Settings::AstcRecompression::Bc3);
    CHECK(settings.vram_usage_mode.GetValue() == Settings::VramUsageMode::Conservative);
    CHECK_FALSE(settings.use_reactive_flushing.GetValue());
    CHECK(settings.vsync_mode.GetValue() == Settings::VSyncMode::FifoRelaxed);
    CHECK(settings.async_presentation.GetValue());
    CHECK(settings.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked);
    CHECK(settings.anti_aliasing.GetValue() == Settings::AntiAliasing::None);
}

void CheckPreferences(const Settings::Values& settings, bool global = false) {
    CHECK(settings.renderer_backend.GetValue(global) == Settings::RendererBackend::Null);
    CHECK(settings.resolution_setup.GetValue(global) == Settings::ResolutionSetup::Res2X);
    CHECK(settings.gpu_accuracy.GetValue(global) == Settings::GpuAccuracy::High);
    CHECK_FALSE(settings.use_asynchronous_shaders.GetValue(global));
    CHECK(settings.astc_recompression.GetValue(global) ==
          Settings::AstcRecompression::Uncompressed);
    CHECK(settings.vram_usage_mode.GetValue(global) == Settings::VramUsageMode::Aggressive);
    CHECK(settings.use_reactive_flushing.GetValue(global));
    CHECK(settings.vsync_mode.GetValue(global) == Settings::VSyncMode::Immediate);
    CHECK(settings.scaling_filter.GetValue(global) == Settings::ScalingFilter::NearestNeighbor);
    CHECK(settings.anti_aliasing.GetValue(global) == Settings::AntiAliasing::Smaa);
}
} // namespace

TEST_CASE("Display defaults are docked without anti-aliasing", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
#ifndef __ANDROID__
    CHECK(settings->use_docked_mode.GetDefault() == Settings::ConsoleMode::Docked);
#endif
    CHECK(settings->anti_aliasing.GetDefault() == Settings::AntiAliasing::None);
    CHECK(settings->scaling_filter.GetDefault() == Settings::ScalingFilter::NearestNeighbor);
}

TEST_CASE("Recommended MKW settings can be made global", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    SetConflictingPreferences(*settings, false);
    Settings::SetMarioKartWorldRecommendedSettings(*settings, true);
    CheckPreset(*settings);
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->gpu_accuracy.UsingGlobal());
    CHECK(settings->renderer_backend.GetValue(true) == Settings::RendererBackend::Vulkan);
    CHECK(settings->gpu_accuracy.GetValue(true) == Settings::GpuAccuracy::Low);
}

TEST_CASE("MKW preset fills in default settings for the session", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    REQUIRE(Settings::ApplyMarioKartWorldPreset(*settings, Settings::MarioKartWorldProgramId));
    CheckPreset(*settings);
    for (const auto& restore : settings->linkage.restore_functions) {
        restore();
    }
    CHECK(settings->gpu_accuracy.UsingGlobal());
    CHECK(settings->gpu_accuracy.GetValue() == settings->gpu_accuracy.GetDefault());
    CHECK(settings->use_reactive_flushing.GetValue() == settings->use_reactive_flushing.GetDefault());
}

TEST_CASE("MKW preset keeps the user's own settings", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    for (bool per_game : {false, true}) {
        SetConflictingPreferences(*settings, per_game);
        Settings::ApplyMarioKartWorldPreset(*settings, Settings::MarioKartWorldProgramId);
        CheckPreferences(*settings, !per_game);
        CHECK(settings->renderer_backend.UsingGlobal() == !per_game);
        CHECK(settings->resolution_setup.UsingGlobal() == !per_game);
    }
}

TEST_CASE("MKW preset does not change unrelated games", "[game_presets]") {
    auto settings = std::make_unique<Settings::Values>();
    CHECK_FALSE(Settings::ApplyMarioKartWorldPreset(*settings, 0));
    CHECK_FALSE(Settings::ApplyMarioKartWorldPreset(*settings, Settings::MarioKartWorldProgramId + 1));
    CHECK(settings->renderer_backend.UsingGlobal());
    CHECK(settings->gpu_accuracy.GetValue() == settings->gpu_accuracy.GetDefault());
}
