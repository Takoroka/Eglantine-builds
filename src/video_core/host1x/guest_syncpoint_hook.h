// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace Tegra::Host1x {

/// Called after every guest syncpoint increment, on the thread that made it (the GPU threads).
/// The Switch 2 gsv service sets it to answer syncpoint waits when a value moves instead of on its
/// next poll, so it must stay cheap.
inline std::atomic<void (*)(u32 syncpoint_id)> g_guest_increment_hook{};

} // namespace Tegra::Host1x
