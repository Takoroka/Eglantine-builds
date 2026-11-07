// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/guest_read_areas.h"

namespace {

using VideoCore::GuestReadAreaTable;
using Hit = GuestReadAreaTable::Hit;

VideoCore::RasterizerDownloadArea Area(u64 start, u64 end) {
    return {.start_address = start, .end_address = end, .preemtive = true};
}

constexpr GuestReadAreaTable::Stamp StampOf(u64 traps, u64 images = 0, u64 ab = 0) {
    return {.trap_generation = traps, .image_epoch = images, .ab_epoch = ab};
}

} // Anonymous namespace

TEST_CASE("GuestReadAreaTable: hits inside remembered areas, newest first", "[video_core]") {
    GuestReadAreaTable table;
    table.Restamp(StampOf(1));
    REQUIRE(table.Find(0x1000, 0x1004) == Hit::None);

    table.Insert(Area(0x1000, 0x2000));
    REQUIRE(table.Find(0x1000, 0x1004) == Hit::Newest);
    REQUIRE(table.Find(0x1ffc, 0x2000) == Hit::Newest);
    // Partly outside: a miss, as with the single area.
    REQUIRE(table.Find(0x1ffc, 0x2004) == Hit::None);
    REQUIRE(table.Find(0xffc, 0x1004) == Hit::None);

    table.Insert(Area(0x8000, 0x9000));
    REQUIRE(table.Find(0x8100, 0x8108) == Hit::Newest);
    // The first area is still remembered: the single area had lost it.
    REQUIRE(table.Find(0x1100, 0x1108) == Hit::Older);
    REQUIRE(table.Count() == 2);
}

TEST_CASE("GuestReadAreaTable: a full table replaces its oldest area", "[video_core]") {
    GuestReadAreaTable table;
    table.Restamp(StampOf(7));
    const auto page = [](u64 i) { return Area(i * 0x10000, i * 0x10000 + 0x1000); };
    for (u64 i = 0; i < GuestReadAreaTable::Capacity; ++i) {
        table.Insert(page(i));
    }
    REQUIRE(table.Count() == GuestReadAreaTable::Capacity);
    for (u64 i = 0; i < GuestReadAreaTable::Capacity; ++i) {
        REQUIRE(table.Find(i * 0x10000, i * 0x10000 + 8) ==
                (i + 1 == GuestReadAreaTable::Capacity ? Hit::Newest : Hit::Older));
    }
    // One more: area 0 goes, the rest stay.
    table.Insert(page(GuestReadAreaTable::Capacity));
    REQUIRE(table.Count() == GuestReadAreaTable::Capacity);
    REQUIRE(table.Find(0, 8) == Hit::None);
    REQUIRE(table.Find(GuestReadAreaTable::Capacity * 0x10000,
                       GuestReadAreaTable::Capacity * 0x10000 + 8) == Hit::Newest);
    for (u64 i = 1; i < GuestReadAreaTable::Capacity; ++i) {
        REQUIRE(table.Find(i * 0x10000, i * 0x10000 + 8) == Hit::Older);
    }
    // And the next one replaces area 1.
    table.Insert(page(GuestReadAreaTable::Capacity + 1));
    REQUIRE(table.Find(0x10000, 0x10008) == Hit::None);
    REQUIRE(table.Find(0x20000, 0x20008) == Hit::Older);
}

TEST_CASE("GuestReadAreaTable: any stamp change drops every area", "[video_core]") {
    GuestReadAreaTable table;
    table.Restamp(StampOf(3, 5, 1));
    table.Insert(Area(0x1000, 0x2000));
    table.Insert(Area(0x4000, 0x5000));

    // The same stamp keeps them.
    table.Restamp(StampOf(3, 5, 1));
    REQUIRE(table.Count() == 2);
    REQUIRE(table.Find(0x1000, 0x1010) == Hit::Older);

    // A read trap armed (the buffer cache's GPU writes).
    table.Restamp(StampOf(4, 5, 1));
    REQUIRE(table.Count() == 0);
    REQUIRE(table.Find(0x1000, 0x1010) == Hit::None);
    REQUIRE(table.Find(0x4000, 0x4010) == Hit::None);

    // An image became GPU-modified.
    table.Insert(Area(0x1000, 0x2000));
    table.Restamp(StampOf(4, 6, 1));
    REQUIRE(table.Find(0x1000, 0x1010) == Hit::None);

    // The A/B phase changed.
    table.Insert(Area(0x1000, 0x2000));
    table.Restamp(StampOf(4, 6, 2));
    REQUIRE(table.Find(0x1000, 0x1010) == Hit::None);

    // Insertion after a drop starts over: the new area is the newest and the only one.
    table.Insert(Area(0x6000, 0x7000));
    REQUIRE(table.Count() == 1);
    REQUIRE(table.Find(0x6000, 0x6004) == Hit::Newest);
}
