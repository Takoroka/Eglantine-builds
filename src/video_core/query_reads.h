// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

// Macro DMA reads of report memory (occlusion predicates): below High GPU accuracy a counter
// report reaches guest memory only once a later fence has popped, so a read before that returns
// an older count (Ocarina of Time's white flashes and missing vegetation).
namespace VideoCommon::QueryReads {

/// Writes the reports in [address, address + size) to guest memory, waiting for those still
/// waiting on host results (RasterizerInterface::FlushQueryResults). Runs where the rasterizer's
/// caches are owned: the back of a split GPU thread.
inline void Flush(const Tegra::MemoryManager& memory, VideoCore::RasterizerInterface& rasterizer,
                  GPUVAddr address, size_t size) {
    constexpr GPUVAddr page_size = 0x1000;
    const GPUVAddr end = address + size;
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = (std::min)((begin & ~(page_size - 1)) + page_size, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            rasterizer.FlushQueryResults(*device, page_end - begin);
        }
        begin = page_end;
    }
}

} // namespace VideoCommon::QueryReads
