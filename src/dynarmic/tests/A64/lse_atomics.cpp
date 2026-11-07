// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Native ARMv8.1 LSE atomics (UserConfig::native_lse_atomics) against the exclusive load + store
// lowering they replace, and against a model of the instructions:
// - every LDADD/LDCLR/LDEOR/LDSET/LD{S,U}{MAX,MIN}/SWP, CAS and CASP encoding (sizes, acquire/
//   release variants, register aliasing, XZR), at an aligned address of a normal page (native
//   path: no fallback may run), in a no-access page of the fastmem arena (the fault goes to the
//   callbacks; a second run uses the code recompiled without fastmem) and misaligned (slow path);
// - atomicity with four cores: LDADD and CAS counters, LDSET/LDCLR bit ownership, CASP pairs, and
//   LDADD mixed with LDAXR/STLXR on the same counter.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "common/common_types.h"
#include "dynarmic/backend/x64/lse_atomics_stats.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/optimization_flags.h"

using namespace Dynarmic;

namespace {

constexpr u64 page_size = 0x1000;
// Arena layout: code, one normal data page, one no-access page (served by the callbacks).
constexpr u64 code_base = 0x0;
constexpr u64 code_size = 0x10000;
constexpr u64 data_base = 0x10000;
constexpr u64 slow_base = 0x11000;
constexpr u64 arena_size = 0x20000;
constexpr size_t arena_bits = 17;

class Arena {
public:
    Arena() {
#ifdef _WIN32
        base = static_cast<u8*>(VirtualAlloc(nullptr, arena_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old_protect{};
        REQUIRE(base != nullptr);
        REQUIRE(VirtualProtect(base + slow_base, page_size, PAGE_NOACCESS, &old_protect));
#else
        void* p = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        REQUIRE(p != MAP_FAILED);
        base = static_cast<u8*>(p);
        REQUIRE(mprotect(base + slow_base, page_size, PROT_NONE) == 0);
#endif
    }
    ~Arena() {
#ifdef _WIN32
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, arena_size);
#endif
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    u8* base = nullptr;
};

class LseEnv final : public A64::UserCallbacks {
public:
    explicit LseEnv(Arena& arena_) : arena(arena_) {}

    Arena& arena;
    // Contents of the no-access page: the JIT only reaches it through the callbacks.
    std::array<u8, page_size> slow{};
    u64 ticks_left = 0;
    A64::Jit* jit = nullptr;

    static bool IsSlow(u64 vaddr) {
        return vaddr >= slow_base && vaddr < slow_base + page_size;
    }
    u8& Byte(u64 vaddr) {
        if (vaddr >= arena_size) {
            unmapped = 0;  // outside the arena: reads 0, writes are dropped
            return unmapped;
        }
        return IsSlow(vaddr) ? slow[vaddr - slow_base] : arena.base[vaddr];
    }
    u8 unmapped = 0;
    template<typename T>
    T Read(u64 vaddr) {
        T value{};
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(static_cast<T>(Byte(vaddr + i)) << (8 * i));
        }
        return value;
    }
    template<typename T>
    void Write(u64 vaddr, T value) {
        for (size_t i = 0; i < sizeof(T); ++i) {
            Byte(vaddr + i) = static_cast<u8>(value >> (8 * i));
        }
    }
    template<typename T>
    bool WriteExclusive(u64 vaddr, T value, T expected) {
        if (!IsSlow(vaddr) && vaddr % sizeof(T) == 0) {
            return std::atomic_ref<T>(*reinterpret_cast<T*>(arena.base + vaddr)).compare_exchange_strong(expected, value);
        }
        // Single-threaded uses only (the tests with several cores stay in the data page).
        if (Read<T>(vaddr) != expected) {
            return false;
        }
        Write<T>(vaddr, value);
        return true;
    }

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        return Read<u32>(vaddr);
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        switch (size) {
        case 1:
            return Read<u8>(vaddr);
        case 2:
            return Read<u16>(vaddr);
        case 4:
            return Read<u32>(vaddr);
        default:
            return Read<u64>(vaddr);
        }
    }
    A64::Vector MemoryRead128(u64 vaddr) override {
        return {Read<u64>(vaddr), Read<u64>(vaddr + 8)};
    }
    void MemoryWrite(u64 vaddr, u64 value, size_t size) override {
        switch (size) {
        case 1:
            return Write<u8>(vaddr, static_cast<u8>(value));
        case 2:
            return Write<u16>(vaddr, static_cast<u16>(value));
        case 4:
            return Write<u32>(vaddr, static_cast<u32>(value));
        default:
            return Write<u64>(vaddr, value);
        }
    }
    void MemoryWrite128(u64 vaddr, A64::Vector value) override {
        Write<u64>(vaddr, value[0]);
        Write<u64>(vaddr + 8, value[1]);
    }
    bool MemoryWriteExclusive8(u64 vaddr, u8 value, u8 expected) override {
        return WriteExclusive<u8>(vaddr, value, expected);
    }
    bool MemoryWriteExclusive16(u64 vaddr, u16 value, u16 expected) override {
        return WriteExclusive<u16>(vaddr, value, expected);
    }
    bool MemoryWriteExclusive32(u64 vaddr, u32 value, u32 expected) override {
        return WriteExclusive<u32>(vaddr, value, expected);
    }
    bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64 expected) override {
        return WriteExclusive<u64>(vaddr, value, expected);
    }
    bool MemoryWriteExclusive128(u64 vaddr, A64::Vector value, A64::Vector expected) override {
        if (MemoryRead128(vaddr) != expected) {
            return false;
        }
        MemoryWrite128(vaddr, value);
        return true;
    }

    void CallSVC(std::uint32_t) override {
        jit->HaltExecution();
    }
    void ExceptionRaised(u64 pc, A64::Exception exception) override {
        FAIL("ExceptionRaised at pc " << pc << " exception " << static_cast<int>(exception));
    }
    void AddTicks(std::uint64_t ticks) override {
        ticks_left = ticks > ticks_left ? 0 : ticks_left - ticks;
    }
    std::uint64_t GetTicksRemaining() override {
        return ticks_left;
    }
    std::uint64_t GetCNTPCT() override {
        return 0;
    }
};

// Eden's Auto CPU accuracy runs exclusives with Unsafe_IgnoreGlobalMonitor.
bool g_ignore_global_monitor = false;

A64::UserConfig MakeConfig(LseEnv& env, ExclusiveMonitor& monitor, bool native, size_t processor_id = 0) {
    A64::UserConfig conf{};
    if (g_ignore_global_monitor) {
        conf.optimizations |= OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
        conf.unsafe_optimizations = true;
    }
    conf.callbacks = &env;
    conf.fastmem_pointer = reinterpret_cast<uintptr_t>(env.arena.base);
    conf.fastmem_address_space_bits = arena_bits;
    conf.silently_mirror_fastmem = false;
    conf.fastmem_exclusive_access = true;
    conf.recompile_on_fastmem_failure = true;
    conf.recompile_on_exclusive_fastmem_failure = true;
    conf.global_monitor = &monitor;
    conf.processor_id = static_cast<std::uint8_t>(processor_id);
    conf.native_lse_atomics = native;
    return conf;
}

// Encodings
u32 Lse(u32 size, bool A, bool R, u32 Rs, u32 op, u32 Rn, u32 Rt) {
    return 0x38200000u | size << 30 | u32(A) << 23 | u32(R) << 22 | Rs << 16 | op << 12 | Rn << 5 | Rt;
}
u32 Cas(u32 size, bool L, u32 Rs, bool o0, u32 Rn, u32 Rt) {
    return 0x08A07C00u | size << 30 | u32(L) << 22 | Rs << 16 | u32(o0) << 15 | Rn << 5 | Rt;
}
u32 Casp(bool sz, bool L, u32 Rs, bool o0, u32 Rn, u32 Rt) {
    return 0x08207C00u | u32(sz) << 30 | u32(L) << 22 | Rs << 16 | u32(o0) << 15 | Rn << 5 | Rt;
}
constexpr u32 branch_self = 0x14000000u;  // B .
constexpr u32 svc0 = 0xD4000001u;
u32 BCond(u32 cond, size_t from, size_t to) {
    const s64 offset = (static_cast<s64>(to) - static_cast<s64>(from));
    return 0x54000000u | ((static_cast<u32>(offset) & 0x7FFFF) << 5) | cond;
}
u32 Cbnz32(u32 Rt, size_t from, size_t to) {
    const s64 offset = (static_cast<s64>(to) - static_cast<s64>(from));
    return 0x35000000u | ((static_cast<u32>(offset) & 0x7FFFF) << 5) | Rt;
}
constexpr u32 cond_ne = 1;

// Model of the instructions
u64 SizeMask(size_t bytes) {
    return bytes == 8 ? ~u64{0} : (u64{1} << (bytes * 8)) - 1;
}
s64 SignExtend(u64 value, size_t bytes) {
    const unsigned shift = static_cast<unsigned>(64 - bytes * 8);
    return static_cast<s64>(value << shift) >> shift;
}
u64 ModelApply(u32 op, u64 old, u64 s, size_t bytes) {
    const u64 mask = SizeMask(bytes);
    old &= mask;
    s &= mask;
    switch (op) {
    case 0:
        return (old + s) & mask;
    case 1:
        return old & ~s;
    case 2:
        return old ^ s;
    case 3:
        return old | s;
    case 4:
        return SignExtend(old, bytes) >= SignExtend(s, bytes) ? old : s;
    case 5:
        return SignExtend(old, bytes) <= SignExtend(s, bytes) ? old : s;
    case 6:
        return old >= s ? old : s;
    case 7:
        return old <= s ? old : s;
    default:
        return s;
    }
}

struct State {
    std::array<u64, 31> regs{};
    u64 pc = 0;
    std::vector<u8> data;
    std::vector<u8> slow;
    u64 fallbacks = 0;

    bool operator==(const State& other) const {
        return regs == other.regs && pc == other.pc && data == other.data && slow == other.slow;
    }
};

struct Machine {
    explicit Machine(bool native_)
            : native(native_), env(arena), monitor(1), jit(MakeConfig(env, monitor, native_)) {
        env.jit = &jit;
    }
    bool native;
    Arena arena;
    LseEnv env;
    ExclusiveMonitor monitor;
    A64::Jit jit;
};

struct Case {
    u32 insn;
    u64 addr;
    std::array<u64, 31> regs;
    std::vector<u8> data;
    std::vector<u8> slow;
    int runs;
};

State Execute(Machine& m, size_t slot, const Case& c) {
    const u64 pc = code_base + slot * 8;
    m.env.Write<u32>(pc, c.insn);
    m.env.Write<u32>(pc + 4, branch_self);
    std::memcpy(m.arena.base + data_base, c.data.data(), page_size);
    std::memcpy(m.env.slow.data(), c.slow.data(), page_size);
    m.jit.SetRegisters(c.regs);
    m.jit.SetSP(0x8000);
    m.jit.SetPstate(0);

    const u64 fallbacks_before = Backend::X64::LseStats::g_fallback.load();
    for (int run = 0; run < c.runs; ++run) {
        m.jit.SetRegisters(c.regs);  // each run from the same registers (memory carries over)
        m.jit.SetPC(pc);
        m.env.ticks_left = 4;
        m.jit.Run();
    }

    State s;
    s.regs = m.jit.GetRegisters();
    s.pc = m.jit.GetPC();
    s.data.assign(m.arena.base + data_base, m.arena.base + data_base + page_size);
    s.slow.assign(m.env.slow.begin(), m.env.slow.end());
    s.fallbacks = Backend::X64::LseStats::g_fallback.load() - fallbacks_before;
    return s;
}

// Reference result from the model (one run).
State Model(const Case& c, State s) {
    s.regs = c.regs;
    s.data = c.data;
    s.slow = c.slow;
    auto mem_byte = [&](u64 a) -> u8& {
        return LseEnv::IsSlow(a) ? s.slow[a - slow_base] : s.data[a - data_base];
    };
    auto read = [&](u64 a, size_t bytes) {
        u64 v = 0;
        for (size_t i = 0; i < bytes; ++i) {
            v |= u64{mem_byte(a + i)} << (8 * i);
        }
        return v;
    };
    auto write = [&](u64 a, u64 v, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) {
            mem_byte(a + i) = static_cast<u8>(v >> (8 * i));
        }
    };
    auto reg = [&](u32 r) { return r == 31 ? u64{0} : s.regs[r]; };
    auto set = [&](u32 r, u64 v) {
        if (r != 31) {
            s.regs[r] = v;
        }
    };

    const u32 insn = c.insn;
    const u32 Rt = insn & 31, Rn = (insn >> 5) & 31, Rs = (insn >> 16) & 31;
    const u64 addr = reg(Rn);
    if ((insn & 0x3F200C00u) == 0x38200000u) {  // LDADD..SWP
        const size_t bytes = size_t{1} << (insn >> 30);
        const u32 op = (insn >> 12) & 0xF;
        const u64 old = read(addr, bytes);
        const u64 operand = reg(Rs);
        write(addr, ModelApply(op, old, operand, bytes), bytes);
        set(Rt, old);
    } else if ((insn & 0x3FA07C00u) == 0x08A07C00u) {  // CAS
        const size_t bytes = size_t{1} << (insn >> 30);
        const u64 mask = SizeMask(bytes);
        const u64 old = read(addr, bytes);
        if (old == (reg(Rs) & mask)) {
            write(addr, reg(Rt) & mask, bytes);
        }
        set(Rs, old);
    } else {  // CASP
        const bool sz = (insn >> 30) & 1;
        const size_t bytes = sz ? 8 : 4;
        const u64 mask = SizeMask(bytes);
        const u64 old_lo = read(addr, bytes), old_hi = read(addr + bytes, bytes);
        if (old_lo == (reg(Rs) & mask) && old_hi == (reg(Rs + 1) & mask)) {
            write(addr, reg(Rt) & mask, bytes);
            write(addr + bytes, reg(Rt + 1) & mask, bytes);
        }
        set(Rs, old_lo);
        set(Rs + 1, old_hi);
    }
    return s;
}

struct Encoding {
    u32 insn;
    size_t bytes;        // access size (both words for CASP)
    size_t align;        // natural alignment
    bool is_cas;
    bool is_casp;
    u32 rs;
    u32 rt;
};

std::vector<Encoding> AllEncodings() {
    std::vector<Encoding> out;
    // LDADD..SWP: every op, size and acquire/release variant (Rs = X1, Rt = X2, Rn = X0), and for
    // AL: Rt = XZR (the ST* aliases), Rs == Rt, Rt == Rn.
    for (u32 op = 0; op <= 8; ++op) {
        for (u32 size = 0; size < 4; ++size) {
            const size_t bytes = size_t{1} << size;
            for (u32 ar = 0; ar < 4; ++ar) {
                out.push_back({Lse(size, ar & 1, ar >> 1, 1, op, 0, 2), bytes, bytes, false, false, 1, 2});
            }
            out.push_back({Lse(size, true, true, 1, op, 0, 31), bytes, bytes, false, false, 1, 31});
            out.push_back({Lse(size, true, true, 1, op, 0, 1), bytes, bytes, false, false, 1, 1});
            out.push_back({Lse(size, true, true, 1, op, 0, 0), bytes, bytes, false, false, 1, 0});
        }
    }
    // CAS: every size and variant (Rs = X1, Rt = X2), and Rs == Rt, Rs = XZR.
    for (u32 size = 0; size < 4; ++size) {
        const size_t bytes = size_t{1} << size;
        for (u32 lo = 0; lo < 4; ++lo) {
            out.push_back({Cas(size, lo & 1, 1, lo >> 1, 0, 2), bytes, bytes, true, false, 1, 2});
        }
        out.push_back({Cas(size, true, 1, true, 0, 1), bytes, bytes, true, false, 1, 1});
        out.push_back({Cas(size, true, 31, true, 0, 2), bytes, bytes, true, false, 31, 2});
    }
    // CASP: both sizes and variants (Rs = X2/X3, Rt = X4/X5), and Rs = X30 (the pair's second
    // register is XZR).
    for (u32 sz = 0; sz < 2; ++sz) {
        const size_t bytes = sz ? 16 : 8;
        for (u32 lo = 0; lo < 4; ++lo) {
            out.push_back({Casp(sz, lo & 1, 2, lo >> 1, 0, 4), bytes, bytes, false, true, 2, 4});
        }
        out.push_back({Casp(sz, true, 30, true, 0, 4), bytes, bytes, false, true, 30, 4});
    }
    return out;
}

u64 Interesting(std::mt19937_64& rng) {
    static constexpr std::array<u64, 12> values{
        0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF,
        0x7FFF'FFFF, 0x8000'0000, 0x7FFF'FFFF'FFFF'FFFF, 0x8000'0000'0000'0000,
    };
    switch (rng() % 4) {
    case 0:
        return values[rng() % values.size()];
    case 1:
        return ~values[rng() % values.size()];
    case 2:
        return values[rng() % values.size()] | (rng() << 32);
    default:
        return rng();
    }
}

}  // namespace

TEST_CASE("A64: LSE atomics native vs exclusive pair", "[a64][lse]") {
    Machine native{true};
    Machine reference{false};
    std::mt19937_64 rng{0x15E'A7'0111C5ull};
    const auto encodings = AllEncodings();

    enum class Where { Aligned, NoAccessPage, Misaligned };
    // LSE_TEST_TRACE=1 prints each case before it runs (to find one that crashes).
    const bool trace = std::getenv("LSE_TEST_TRACE") != nullptr;
    size_t slot = 0;
    size_t cases = 0;
    for (const Encoding& e : encodings) {
        for (const Where where : {Where::Aligned, Where::NoAccessPage, Where::Misaligned}) {
            if (where == Where::Misaligned && (e.align == 1 || e.bytes == 16)) {
                continue;  // bytes never misalign; a misaligned cmpxchg16b in the reference faults
            }
            for (int v = 0; v < 6; ++v) {
                Case c;
                c.insn = e.insn;
                const u64 base = where == Where::NoAccessPage ? slow_base : data_base;
                c.addr = base + 0x400 + 0x40 * static_cast<u64>(rng() % 16) + (where == Where::Misaligned ? e.align / 2 : 0);
                c.data.resize(page_size);
                c.slow.resize(page_size);
                for (auto& b : c.data) {
                    b = static_cast<u8>(rng());
                }
                for (auto& b : c.slow) {
                    b = static_cast<u8>(rng());
                }
                for (auto& r : c.regs) {
                    r = Interesting(rng);
                }
                c.regs[0] = c.addr;
                // Memory value at the address, and for CAS/CASP a matching compare value half
                // of the time (only the access size's low bits have to match).
                auto mem = [&](u64 a) -> u8& {
                    return LseEnv::IsSlow(a) ? c.slow[a - slow_base] : c.data[a - data_base];
                };
                const size_t word = e.is_casp ? e.bytes / 2 : e.bytes;
                for (size_t w = 0; w < (e.is_casp ? 2u : 1u); ++w) {
                    const u64 value = Interesting(rng);
                    for (size_t i = 0; i < word; ++i) {
                        mem(c.addr + w * word + i) = static_cast<u8>(value >> (8 * i));
                    }
                    if ((e.is_cas || e.is_casp) && (v & 1) == 0 && e.rs + w != 31) {
                        const u64 mask = SizeMask(word);
                        u64& rs = c.regs[e.rs + w];
                        rs = (rs & ~mask) | (value & mask);
                    }
                }
                c.runs = where == Where::NoAccessPage ? 2 : 1;

                INFO("insn " << std::hex << c.insn << " addr " << c.addr << " case " << std::dec << cases);
                if (trace) {
                    std::fprintf(stderr, "case %zu insn %08x addr %llx where %d: native\n", cases, c.insn,
                                 static_cast<unsigned long long>(c.addr), static_cast<int>(where));
                    std::fflush(stderr);
                }
                const State got = Execute(native, slot, c);
                if (trace) {
                    std::fprintf(stderr, "  reference\n");
                    std::fflush(stderr);
                }
                const State want = Execute(reference, slot, c);
                REQUIRE(got.regs == want.regs);
                REQUIRE(got.data == want.data);
                REQUIRE(got.slow == want.slow);
                REQUIRE(got.pc == want.pc);
                REQUIRE(want.fallbacks == 0);
                if (where == Where::Aligned) {
                    REQUIRE(got.fallbacks == 0);  // done by the host atomic
                } else {
                    REQUIRE(got.fallbacks == static_cast<u64>(c.runs));
                }
                if (c.runs == 1) {
                    const State model = Model(c, got);
                    REQUIRE(got.regs == model.regs);
                    REQUIRE(got.data == model.data);
                    REQUIRE(got.slow == model.slow);
                }

                ++slot;
                ++cases;
                REQUIRE(slot * 8 < code_size);
            }
        }
    }
    INFO("cases " << cases);
    REQUIRE(cases > 4000);
}

namespace {

// Four cores run `code` (at code_base) until SVC #0; `setup` sets each core's registers and
// `after` reads them when all are done.
template<typename Setup, typename After = void (*)(A64::Jit&, size_t)>
void RunCores(Arena& arena, bool native, const std::vector<u32>& code, Setup setup,
              After after = [](A64::Jit&, size_t) {}) {
    constexpr size_t cores = 4;
    ExclusiveMonitor monitor{cores};
    std::vector<std::unique_ptr<LseEnv>> envs;
    std::vector<std::unique_ptr<A64::Jit>> jits;
    for (size_t i = 0; i < code.size(); ++i) {
        std::memcpy(arena.base + code_base + i * 4, &code[i], 4);
    }
    for (size_t core = 0; core < cores; ++core) {
        envs.push_back(std::make_unique<LseEnv>(arena));
        jits.push_back(std::make_unique<A64::Jit>(MakeConfig(*envs.back(), monitor, native, core)));
        envs.back()->jit = jits.back().get();
        envs.back()->ticks_left = u64{1} << 60;  // the JIT treats the count as signed
        jits.back()->SetPC(code_base);
        jits.back()->SetSP(0x8000);
        setup(*jits.back(), core);
    }
    std::vector<std::thread> threads;
    for (size_t core = 0; core < cores; ++core) {
        threads.emplace_back([&, core] { jits[core]->Run(); });
    }
    for (auto& t : threads) {
        t.join();
    }
    for (size_t core = 0; core < cores; ++core) {
        after(*jits[core], core);
    }
}

}  // namespace

TEST_CASE("A64: LSE atomics with four cores", "[a64][lse]") {
    constexpr u64 iterations = 100'000;
    const u64 fallbacks_before = Backend::X64::LseStats::g_fallback.load();
    Arena arena;
    std::memset(arena.base + data_base, 0, 16);

    SECTION("LDADDAL counter") {
        const std::vector<u32> code{
            Lse(3, true, true, 1, 0, 0, 2),  // 0: LDADDAL X1, X2, [X0]
            0xF1000463u,                     // 1: SUBS X3, X3, #1
            BCond(cond_ne, 2, 0),            // 2: B.NE 0
            svc0,                            // 3
        };
        RunCores(arena, true, code, [](A64::Jit& jit, size_t) {
            jit.SetRegister(0, data_base);
            jit.SetRegister(1, 3);
            jit.SetRegister(3, iterations);
        });
        u64 total{};
        std::memcpy(&total, arena.base + data_base, 8);
        REQUIRE(total == 4 * 3 * iterations);
    }

    SECTION("CASAL counter") {
        const std::vector<u32> code{
            0xF9400002u,                  // 0: LDR X2, [X0]
            0x8B010044u,                  // 1: ADD X4, X2, X1
            0xAA0203E6u,                  // 2: MOV X6, X2
            Cas(3, true, 2, true, 0, 4),  // 3: CASAL X2, X4, [X0]
            0xEB06005Fu,                  // 4: CMP X2, X6
            BCond(cond_ne, 5, 1),         // 5: B.NE 1 (X2 is the current value)
            0xAA0403E2u,                  // 6: MOV X2, X4
            0xF1000463u,                  // 7: SUBS X3, X3, #1
            BCond(cond_ne, 8, 1),         // 8: B.NE 1
            svc0,                         // 9
        };
        RunCores(arena, true, code, [](A64::Jit& jit, size_t) {
            jit.SetRegister(0, data_base);
            jit.SetRegister(1, 1);
            jit.SetRegister(3, iterations);
        });
        u64 total{};
        std::memcpy(&total, arena.base + data_base, 8);
        REQUIRE(total == 4 * iterations);
    }

    SECTION("LDSETAL/LDCLRAL bit ownership") {
        // Each core owns one bit of a shared word: its LDSET must see the bit clear and its LDCLR
        // must see it set. A read-modify-write that isn't atomic loses other cores' updates and
        // breaks that for some core (counted in X5).
        const std::vector<u32> code{
            Lse(2, true, true, 1, 3, 0, 2),  // 0: LDSETAL W1, W2, [X0]
            0x6A01005Fu,                     // 1: TST W2, W1
            0x9A8504A5u,                     // 2: CINC X5, X5, NE
            Lse(2, true, true, 1, 1, 0, 2),  // 3: LDCLRAL W1, W2, [X0]
            0x6A01005Fu,                     // 4: TST W2, W1
            0x9A8514A5u,                     // 5: CINC X5, X5, EQ
            0xF1000463u,                     // 6: SUBS X3, X3, #1
            BCond(cond_ne, 7, 0),            // 7: B.NE 0
            svc0,                            // 8
        };
        std::array<u64, 4> violations{};
        RunCores(
            arena, true, code,
            [](A64::Jit& jit, size_t core) {
                jit.SetRegister(0, data_base);
                jit.SetRegister(1, u64{1} << core);
                jit.SetRegister(3, iterations);
                jit.SetRegister(5, 0);
            },
            [&](A64::Jit& jit, size_t core) { violations[core] = jit.GetRegister(5); });
        REQUIRE(violations == std::array<u64, 4>{});
        u32 word{};
        std::memcpy(&word, arena.base + data_base, 4);
        REQUIRE(word == 0);
    }

    SECTION("CASPAL pair counter") {
        const std::vector<u32> code{
            0xA9400C02u,                      // 0: LDP X2, X3, [X0]
            0x91000444u,                      // 1: ADD X4, X2, #1
            0x91000465u,                      // 2: ADD X5, X3, #1
            0xAA0203E6u,                      // 3: MOV X6, X2
            0xAA0303E7u,                      // 4: MOV X7, X3
            Casp(true, true, 2, true, 0, 4),  // 5: CASPAL X2, X3, X4, X5, [X0]
            0xEB06005Fu,                      // 6: CMP X2, X6
            0xFA470060u,                      // 7: CCMP X3, X7, #0, EQ
            BCond(cond_ne, 8, 1),             // 8: B.NE 1
            0xAA0403E2u,                      // 9: MOV X2, X4
            0xAA0503E3u,                      // 10: MOV X3, X5
            0xF1000508u,                      // 11: SUBS X8, X8, #1
            BCond(cond_ne, 12, 1),            // 12: B.NE 1
            svc0,                             // 13
        };
        RunCores(arena, true, code, [](A64::Jit& jit, size_t) {
            jit.SetRegister(0, data_base);
            jit.SetRegister(8, iterations);
        });
        std::array<u64, 2> pair{};
        std::memcpy(pair.data(), arena.base + data_base, 16);
        REQUIRE(pair[0] == 4 * iterations);
        REQUIRE(pair[1] == 4 * iterations);
    }

    SECTION("LDADDAL mixed with LDAXR/STLXR") {
        // Cores 0 and 1 add with LDADDAL, cores 2 and 3 with an exclusive pair, on one counter.
        const std::vector<u32> code{
            Cbnz32(9, 0, 5),                 // 0: CBNZ W9, 5
            Lse(3, true, true, 1, 0, 0, 2),  // 1: LDADDAL X1, X2, [X0]
            0xF1000463u,                     // 2: SUBS X3, X3, #1
            BCond(cond_ne, 3, 1),            // 3: B.NE 1
            svc0,                            // 4
            0xC85FFC02u,                     // 5: LDAXR X2, [X0]
            0x8B010042u,                     // 6: ADD X2, X2, X1
            0xC804FC02u,                     // 7: STLXR W4, X2, [X0]
            Cbnz32(4, 8, 5),                 // 8: CBNZ W4, 5
            0xF1000463u,                     // 9: SUBS X3, X3, #1
            BCond(cond_ne, 10, 5),           // 10: B.NE 5
            svc0,                            // 11
        };
        for (const bool ignore_global_monitor : {false, true}) {
            INFO("Unsafe_IgnoreGlobalMonitor " << ignore_global_monitor);
            g_ignore_global_monitor = ignore_global_monitor;
            std::memset(arena.base + data_base, 0, 8);
            RunCores(arena, true, code, [](A64::Jit& jit, size_t core) {
                jit.SetRegister(0, data_base);
                jit.SetRegister(1, 1);
                jit.SetRegister(3, iterations);
                jit.SetRegister(9, core >= 2 ? 1 : 0);
            });
            g_ignore_global_monitor = false;
            u64 total{};
            std::memcpy(&total, arena.base + data_base, 8);
            REQUIRE(total == 4 * iterations);
        }
    }

    // All of it ran aligned in a normal page: the host atomics, never the slow path.
    REQUIRE(Backend::X64::LseStats::g_fallback.load() == fallbacks_before);
}
