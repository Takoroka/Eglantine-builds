// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "core/device_memory_manager.h"

namespace Tegra::Host1x {
class Host1x;
} // namespace Tegra::Host1x

namespace Service::Nvidia::NvCore {

class HeapMapper {
public:
    HeapMapper(VAddr start_vaddress, DAddr start_daddress, size_t size, Core::Asid asid,
               Tegra::Host1x::Host1x& host1x);
    ~HeapMapper();

    bool IsInBounds(VAddr start, size_t size) const {
        VAddr end = start + size;
        return start >= m_vaddress && end <= (m_vaddress + m_size);
    }

    /// mapped_before, if given, gets the parts of [start, +size) that were mapped already (their
    /// device pages are kept as they are).
    DAddr Map(VAddr start, size_t size,
              std::vector<std::pair<VAddr, size_t>>* mapped_before = nullptr);

    void Unmap(VAddr start, size_t size);

    DAddr GetRegionStart() const {
        return m_daddress;
    }

    size_t GetRegionSize() const {
        return m_size;
    }

private:
    struct HeapMapperInternal;
    VAddr m_vaddress;
    DAddr m_daddress;
    size_t m_size;
    Core::Asid m_asid;
    std::unique_ptr<HeapMapperInternal> m_internal;
};

} // namespace Service::Nvidia::NvCore
