// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// page_table_write_marked_mask (Eden A/B bit 18 FastmemRestore): with it, the page-table path
// sends a store to the callbacks when the entry has any of the mask's bits set, while a load
// still checks page_table_marked_bit alone. Eden's cached pages restored to direct reads have the
// marked bit clear and the high type bit set: their loads read memory, their stores must still
// reach the write tracking (the callbacks).

#include <array>
#include <cstring>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/tests/A64/testenv.h"

namespace {

constexpr u64 PageBits = 12;
constexpr u64 AddressSpaceBits = 24;
constexpr u64 DataPage = 0x10000;
constexpr u64 HostValue = 0x1111111111111111;
constexpr u64 StoreValue = 0x2222222222222222;
// What A64TestEnv's callbacks give for 8 bytes at DataPage (each byte its address).
constexpr u64 CallbackValue = 0x0706050403020100;

struct Result {
    u64 loaded;          ///< X1 after LDR X1, [X0]
    bool store_callback; ///< STR X2, [X0] went to the callbacks
    u64 host_after;      ///< The page's memory after the store
};

// Runs LDR X1, [X0]; STR X2, [X0] on DataPage with its page table entry flags.
Result Run(u64 entry_flags, std::optional<u64> write_marked_mask) {
    alignas(4096) static std::array<u8, 4096> host;
    std::memset(host.data(), 0x11, host.size());
    std::vector<u64> table(size_t{1} << (AddressSpaceBits - PageBits), 0);
    // Absolute offset entries: the host pointer is (entry & mask) + vaddr.
    table[DataPage >> PageBits] = (reinterpret_cast<u64>(host.data()) - DataPage) | entry_flags;

    A64TestEnv env;
    Dynarmic::A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.page_table = reinterpret_cast<void**>(table.data());
    conf.page_table_address_space_bits = AddressSpaceBits;
    conf.page_table_pointer_mask = ~u64{(u64{1} << PageBits) - 1};
    conf.page_table_marked_bit = 0;
    conf.page_table_write_marked_mask = write_marked_mask;
    conf.absolute_offset_page_table = true;
    conf.silently_mirror_page_table = false;
    Dynarmic::A64::Jit jit{conf};

    env.code_mem.emplace_back(0xF9400001); // LDR X1, [X0]
    env.code_mem.emplace_back(0xF9000002); // STR X2, [X0]
    env.code_mem.emplace_back(0x14000000); // B .
    jit.SetPC(0);
    jit.SetRegister(0, DataPage);
    jit.SetRegister(1, 0);
    jit.SetRegister(2, StoreValue);
    env.ticks_left = 3;
    jit.Run();

    u64 host_after;
    std::memcpy(&host_after, host.data(), sizeof(host_after));
    return {
        .loaded = jit.GetRegister(1),
        .store_callback = env.modified_memory.contains(DataPage),
        .host_after = host_after,
    };
}

// Eden's entry low bits: marked (bit 0), then the page type (bits 1-2).
constexpr u64 MemoryPage = 0b010;      // regular memory
constexpr u64 CachedPage = 0b111;      // rasterizer cached, marked
constexpr u64 RestoredPage = 0b110;    // rasterizer cached, reads restored
constexpr u64 WriteMarkedMask = 0b101; // Common::PageTable::WRITE_MARKED_MASK

} // Anonymous namespace

TEST_CASE("A64: stores to a page with write-marked bits go to the callbacks, loads don't",
          "[a64]") {
    const Result restored = Run(RestoredPage, WriteMarkedMask);
    CHECK(restored.loaded == HostValue);  // read directly
    CHECK(restored.store_callback);       // the write still trapped
    CHECK(restored.host_after == HostValue);
}

TEST_CASE("A64: a marked page sends loads and stores to the callbacks with the mask too",
          "[a64]") {
    const Result cached = Run(CachedPage, WriteMarkedMask);
    CHECK(cached.loaded == CallbackValue);
    CHECK(cached.store_callback);
    CHECK(cached.host_after == HostValue);
}

TEST_CASE("A64: regular memory stays direct for loads and stores with the mask", "[a64]") {
    const Result memory = Run(MemoryPage, WriteMarkedMask);
    CHECK(memory.loaded == HostValue);
    CHECK_FALSE(memory.store_callback);
    CHECK(memory.host_after == StoreValue);
}

TEST_CASE("A64: without the mask a store to such a page goes direct (why Eden needs it)",
          "[a64]") {
    const Result unmasked = Run(RestoredPage, std::nullopt);
    CHECK(unmasked.loaded == HostValue);
    CHECK_FALSE(unmasked.store_callback);
    CHECK(unmasked.host_after == StoreValue);
}
