// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "audio_core/nx2_audio_stats.h"

using AudioCore::Nx2AudioStats::QueueTargetFor;

TEST_CASE("Render queue target follows the device's largest request", "[audio_core][queue_target]") {
    constexpr u32 ring = 4;
    constexpr u32 buffer = 240; // 5 ms at 48 kHz
    // A device that asks 10 ms per callback keeps today's 4 buffers.
    CHECK(QueueTargetFor(ring, 480, buffer) == 4);
    CHECK(QueueTargetFor(ring, 0, buffer) == 4);
    CHECK(QueueTargetFor(ring, 240, buffer) == 4);
    // Larger requests: enough buffers to cover one, plus two.
    CHECK(QueueTargetFor(ring, 481, buffer) == 5);
    CHECK(QueueTargetFor(ring, 1056, buffer) == 7);
    CHECK(QueueTargetFor(ring, 2048, buffer) == 11);
    CHECK(QueueTargetFor(ring, 4800, buffer) == 22);
    // Never fewer than the ring size.
    CHECK(QueueTargetFor(8, 480, buffer) == 8);
}
