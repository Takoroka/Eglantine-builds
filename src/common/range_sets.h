// SPDX-FileCopyrightText: 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "common/common_types.h"

namespace Common {

template <typename AddressType>
class RangeSet {
public:
    RangeSet();
    ~RangeSet();

    RangeSet(RangeSet const&) = delete;
    RangeSet& operator=(RangeSet const&) = delete;

    RangeSet(RangeSet&& other);
    RangeSet& operator=(RangeSet&& other);

    void Add(AddressType base_address, size_t size);
    void Subtract(AddressType base_address, size_t size);
    void Clear();
    bool Empty() const;

    /// Whether any range overlaps [base_address, base_address + size): one lookup.
    bool Intersects(AddressType base_address, size_t size) const;

    /// Whether one range holds all of [base_address, base_address + size): one lookup (adjacent
    /// ranges are joined, so a covered query lies in a single range).
    bool Covers(AddressType base_address, size_t size) const;

    /// When no range overlaps [base_address, base_address + size), the gap between the ranges
    /// around it as [gap_begin, gap_end), and true; false when a range overlaps it. One lookup.
    bool FreeGap(AddressType base_address, size_t size, AddressType& gap_begin,
                 AddressType& gap_end) const;

    template <typename Func>
    void ForEach(Func&& func) const;

    template <typename Func>
    void ForEachInRange(AddressType device_addr, size_t size, Func&& func) const;

private:
    struct RangeSetImpl;
    std::unique_ptr<RangeSetImpl> m_impl;
};

template <typename AddressType>
class OverlapRangeSet {
public:
    OverlapRangeSet();
    ~OverlapRangeSet();

    OverlapRangeSet(OverlapRangeSet const&) = delete;
    OverlapRangeSet& operator=(OverlapRangeSet const&) = delete;

    OverlapRangeSet(OverlapRangeSet&& other);
    OverlapRangeSet& operator=(OverlapRangeSet&& other);

    void Add(AddressType base_address, size_t size);
    void Subtract(AddressType base_address, size_t size);

    template <typename Func>
    void Subtract(AddressType base_address, size_t size, Func&& on_delete);

    void DeleteAll(AddressType base_address, size_t size);
    void Clear();
    bool Empty() const;

    template <typename Func>
    void ForEach(Func&& func) const;

    template <typename Func>
    void ForEachInRange(AddressType device_addr, size_t size, Func&& func) const;

private:
    struct OverlapRangeSetImpl;
    std::unique_ptr<OverlapRangeSetImpl> m_impl;
};

} // namespace Common
