// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/settings.h"

namespace Settings {

inline constexpr u64 MarioKartWorldProgramId = 0x0400C3F00006E000ULL;
inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000;

// Mario Kart World's recommended settings, passed one by one to `apply(setting, value)`. Docked at
// 1x without anti-aliasing: handheld or a lower resolution would only hide the emulator's cost.
template <typename Apply>
void ForEachMarioKartWorldRecommendation(Values& settings, Apply&& apply) {
    apply(settings.renderer_backend, RendererBackend::Vulkan);
    apply(settings.resolution_setup, ResolutionSetup::Res1X);
    apply(settings.gpu_accuracy, GpuAccuracy::Low);
    apply(settings.use_asynchronous_shaders, true);
    apply(settings.async_presentation, true);
    apply(settings.astc_recompression, AstcRecompression::Bc3);
    apply(settings.vram_usage_mode, VramUsageMode::Conservative);
    apply(settings.use_reactive_flushing, false);
    apply(settings.vsync_mode, VSyncMode::FifoRelaxed);
    apply(settings.use_docked_mode, ConsoleMode::Docked);
    apply(settings.anti_aliasing, AntiAliasing::None);
}

inline void SetMarioKartWorldRecommendedSettings(Values& settings, bool global) {
    ForEachMarioKartWorldRecommendation(settings, [global](auto& setting, auto value) {
        setting.SetGlobal(global);
        setting.SetValue(value);
    });
}

// Applies the recommended settings to Mario Kart World for the session, without asking, but only
// to settings the user left alone: no per-game value and the global value still at its default.
// Whatever the user chose stays. True when it changed anything.
inline bool ApplyMarioKartWorldPreset(Values& settings, u64 program_id) {
    if (program_id != MarioKartWorldProgramId) {
        return false;
    }
    bool applied = false;
    ForEachMarioKartWorldRecommendation(settings, [&applied](auto& setting, auto value) {
        if (!setting.UsingGlobal() || setting.GetValue(true) != setting.GetDefault() ||
            setting.GetValue(true) == value) {
            return;
        }
        setting.SetGlobal(false);
        setting.SetValue(value);
        applied = true;
    });
    return applied;
}

inline void SetOcarinaOfTimeRecommendedSettings(Values& settings, bool global) {
    const auto session = [global](auto& setting, auto value) {
        setting.SetGlobal(global);
        setting.SetValue(value);
    };
    session(settings.renderer_backend, RendererBackend::Vulkan);
    session(settings.resolution_setup, ResolutionSetup::Res1X);
    session(settings.gpu_accuracy, GpuAccuracy::Low);
    session(settings.use_asynchronous_shaders, true);
    session(settings.astc_recompression, AstcRecompression::Bc3);
    session(settings.vram_usage_mode, VramUsageMode::Conservative);
    session(settings.use_reactive_flushing, true);
    session(settings.vsync_mode, VSyncMode::FifoRelaxed);
    session(settings.use_docked_mode, ConsoleMode::Docked);
    session(settings.scaling_filter, ScalingFilter::ScaleForce);
    session(settings.anti_aliasing, AntiAliasing::Fxaa);
}

inline bool ApplyOcarinaOfTimePreset(Values& settings, u64 program_id, bool accepted) {
    if (!accepted || program_id != OcarinaOfTimeProgramId) {
        return false;
    }
    SetOcarinaOfTimeRecommendedSettings(settings, false);
    return true;
}

} // namespace Settings
