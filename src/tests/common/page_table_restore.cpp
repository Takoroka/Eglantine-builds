// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"

namespace {

using Common::PageTable;
using Common::PageType;
using Entry = PageTable::PageEntryData;

constexpr uintptr_t HostPointer = 0x7f1234560000;

u64 Bits(const Entry& entry) {
    return std::bit_cast<u64>(entry.Raw());
}

// What Dynarmic does with the entry (page_table_marked_bit 0, page_table_write_marked_mask).
bool LoadTraps(const Entry& entry) {
    return (Bits(entry) & 1) != 0;
}
bool StoreTraps(const Entry& entry) {
    return (Bits(entry) & PageTable::WRITE_MARKED_MASK) != 0;
}

} // Anonymous namespace

TEST_CASE("PageEntryData: a cached page restored to direct reads keeps trapping stores",
          "[common]") {
    Entry entry;
    entry.Store(false, PageType::Memory, 0, HostPointer);
    REQUIRE_FALSE(LoadTraps(entry));
    REQUIRE_FALSE(StoreTraps(entry));

    entry.MarkRasterizerCached();
    REQUIRE(entry.Type() == PageType::RasterizerCachedMemory);
    REQUIRE(LoadTraps(entry));
    REQUIRE(StoreTraps(entry));
    REQUIRE(entry.Pointer() == 0); // the marked entry has no pointer for reads
    REQUIRE_FALSE(entry.CachedReadsUnmarked());

    REQUIRE(entry.UnmarkCachedReads());
    REQUIRE(entry.CachedReadsUnmarked());
    REQUIRE(entry.Type() == PageType::RasterizerCachedMemory);
    REQUIRE_FALSE(LoadTraps(entry));
    REQUIRE(StoreTraps(entry)); // the write tracking still sees stores
    REQUIRE(entry.Pointer() == HostPointer);

    // Only once.
    REQUIRE_FALSE(entry.UnmarkCachedReads());

    REQUIRE(entry.MarkCachedReads());
    REQUIRE(LoadTraps(entry));
    REQUIRE(StoreTraps(entry));
    REQUIRE_FALSE(entry.CachedReadsUnmarked());
    REQUIRE_FALSE(entry.MarkCachedReads());
}

TEST_CASE("PageEntryData: only marked cached pages can be restored", "[common]") {
    Entry memory;
    memory.Store(false, PageType::Memory, 0, HostPointer);
    REQUIRE_FALSE(memory.UnmarkCachedReads());
    REQUIRE_FALSE(memory.MarkCachedReads());
    REQUIRE_FALSE(StoreTraps(memory));

    Entry debug;
    debug.MarkDebug(HostPointer, 0);
    REQUIRE_FALSE(debug.UnmarkCachedReads());
    REQUIRE(LoadTraps(debug));
    REQUIRE(StoreTraps(debug));

    Entry unmapped;
    unmapped.Store(false, PageType::Unmapped, 0, 0);
    REQUIRE_FALSE(unmapped.UnmarkCachedReads());
}

TEST_CASE("PageEntryData: caching a restored page again marks it", "[common]") {
    Entry entry;
    entry.Store(false, PageType::Memory, 0, HostPointer);
    entry.MarkRasterizerCached();
    REQUIRE(entry.UnmarkCachedReads());
    // RasterizerMarkRegionCached(true) on it: MarkRasterizerCached's bits include the marked one.
    entry.MarkRasterizerCached();
    REQUIRE(LoadTraps(entry));
    REQUIRE_FALSE(entry.CachedReadsUnmarked());
}
