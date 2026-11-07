// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include "dynarmic/mcl/integer_of_size.hpp"
#include "dynarmic/backend/x64/xbyak.h"

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64_memory.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/lse_atomics_stats.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/common/spin_lock_x64.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

void A64EmitX64::GenMemory128Accessors() {
    code.align();
    memory_read_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCallWithReturnPointer(code, [&](Xbyak::Reg64 return_value_ptr, [[maybe_unused]] RegList args) {
        code.mov(code.ABI_PARAM3, code.ABI_PARAM2);
        code.lea(rsp, ptr[rsp - (8 + 16 + ABI_SHADOW_SPACE)]);
        code.lea(return_value_ptr, ptr[rsp + ABI_SHADOW_SPACE]);
    });
    code.movups(xmm1, xword[code.ABI_RETURN]);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    Devirtualize<&A64::UserCallbacks::MemoryRead128>(conf.callbacks).EmitCall(code);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(xmm1, code.ABI_RETURN);
        code.pinsrq(xmm1, code.ABI_RETURN2, 1);
    } else {
        code.movq(xmm1, code.ABI_RETURN);
        code.movq(xmm2, code.ABI_RETURN2);
        code.punpcklqdq(xmm1, xmm2);
    }
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_read_128, code.getCurr(), "a64_memory_read_128");

    code.align();
    memory_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.lea(rsp, ptr[rsp - (8 + 16 + ABI_SHADOW_SPACE)]);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 16 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWrite128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_write_128, code.getCurr(), "a64_memory_write_128");

    code.align();
    memory_exclusive_write_128 = code.getCurr<void (*)()>();
#ifdef _WIN32
    code.lea(rsp, ptr[rsp - (8 + 32 + ABI_SHADOW_SPACE)]);
    code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
    code.lea(code.ABI_PARAM4, ptr[rsp + ABI_SHADOW_SPACE + 16]);
    code.movaps(xword[code.ABI_PARAM3], xmm1);
    code.movaps(xword[code.ABI_PARAM4], xmm2);
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8 + 32 + ABI_SHADOW_SPACE);
#else
    code.lea(rsp, ptr[rsp - 8]);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.movq(code.ABI_PARAM3, xmm1);
        code.pextrq(code.ABI_PARAM4, xmm1, 1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.pextrq(code.ABI_PARAM6, xmm2, 1);
    } else {
        code.movq(code.ABI_PARAM3, xmm1);
        code.punpckhqdq(xmm1, xmm1);
        code.movq(code.ABI_PARAM4, xmm1);
        code.movq(code.ABI_PARAM5, xmm2);
        code.punpckhqdq(xmm2, xmm2);
        code.movq(code.ABI_PARAM6, xmm2);
    }
    Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive128>(conf.callbacks).EmitCall(code);
    code.add(rsp, 8);
#endif
    code.ret();
    PerfMapRegister(memory_exclusive_write_128, code.getCurr(), "a64_memory_exclusive_write_128");
}

void A64EmitX64::GenFastmemFallbacks() {
    const std::initializer_list<int> idxes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::array<std::pair<size_t, ArgCallback>, 4> read_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryRead>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWrite>(conf.callbacks)},
    }};
    const std::array<std::pair<size_t, ArgCallback>, 4> exclusive_write_callbacks{{
        {8, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive8>(conf.callbacks)},
        {16, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive16>(conf.callbacks)},
        {32, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive32>(conf.callbacks)},
        {64, Devirtualize<&A64::UserCallbacks::MemoryWriteExclusive64>(conf.callbacks)},
    }};

    for (auto const ordered : {false, true}) {
        for (auto const vaddr_idx : idxes) {
            if (vaddr_idx == 4 || vaddr_idx == 15) {
                continue;
            }

            for (int value_idx : idxes) {
                code.align();
                read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (ordered) {
                    code.mfence();
                }
                code.call(memory_read_128);
                if (value_idx != 1) {
                    code.movaps(Xbyak::Xmm{value_idx}, xmm1);
                }
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(value_idx));
                code.ret();
                PerfMapRegister(read_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_read_fallback_128");

                code.align();
                write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStack(code);
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                code.call(memory_write_128);
                if (ordered) {
                    code.mfence();
                }
                ABI_PopCallerSaveRegistersAndAdjustStack(code);
                code.ret();
                PerfMapRegister(write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_write_fallback_128");

                code.align();
                exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                if (value_idx != 1) {
                    code.movaps(xmm1, Xbyak::Xmm{value_idx});
                }
                if (code.HasHostFeature(HostFeature::SSE41)) {
                    code.movq(xmm2, rax);
                    code.pinsrq(xmm2, rdx, 1);
                } else {
                    code.movq(xmm2, rax);
                    code.movq(xmm0, rdx);
                    code.punpcklqdq(xmm2, xmm0);
                }
                if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                    code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                }
                code.call(memory_exclusive_write_128);
                ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                code.ret();
                PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, 128, vaddr_idx, value_idx)], code.getCurr(), "a64_exclusive_write_fallback_128");

                if (value_idx == 4 || value_idx == 15) {
                    continue;
                }

                for (const auto& [bitsize, callback] : read_callbacks) {
                    code.align();
                    read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    // params = { this, vaddr, size }
                    if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                    }
                    code.mov(code.ABI_PARAM3, bitsize / CHAR_BIT);
                    if (ordered) code.mfence();
                    callback.EmitCall(code);
                    if (value_idx != code.ABI_RETURN.getIdx()) {
                        code.mov(Xbyak::Reg64{value_idx}, code.ABI_RETURN);
                    }
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(value_idx));
                    code.ZeroExtendFrom(bitsize, Xbyak::Reg64{value_idx});
                    code.ret();
                    PerfMapRegister(read_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_read_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : write_callbacks) {
                    code.align();
                    write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStack(code);
                    // params = { this, vaddr, value, size }
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    code.mov(code.ABI_PARAM4, bitsize / CHAR_BIT);
                    callback.EmitCall(code);
                    if (ordered) code.mfence();
                    ABI_PopCallerSaveRegistersAndAdjustStack(code);
                    code.ret();
                    PerfMapRegister(write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_write_fallback_{}", bitsize));
                }

                for (const auto& [bitsize, callback] : exclusive_write_callbacks) {
                    code.align();
                    exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)] = code.getCurr<void (*)()>();
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    if (vaddr_idx == code.ABI_PARAM3.getIdx() && value_idx == code.ABI_PARAM2.getIdx()) {
                        code.xchg(code.ABI_PARAM2, code.ABI_PARAM3);
                    } else if (vaddr_idx == code.ABI_PARAM3.getIdx()) {
                        code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                    } else {
                        if (value_idx != code.ABI_PARAM3.getIdx()) {
                            code.mov(code.ABI_PARAM3, Xbyak::Reg64{value_idx});
                        }
                        if (vaddr_idx != code.ABI_PARAM2.getIdx()) {
                            code.mov(code.ABI_PARAM2, Xbyak::Reg64{vaddr_idx});
                        }
                    }
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM3);
                    code.mov(code.ABI_PARAM4, rax);
                    code.ZeroExtendFrom(bitsize, code.ABI_PARAM4);
                    callback.EmitCall(code);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
                    code.ret();
                    PerfMapRegister(exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr_idx, value_idx)], code.getCurr(), fmt::format("a64_exclusive_write_fallback_{}", bitsize));
                }
            }
        }
    }
}

#define Axx A64
#include "dynarmic/backend/x64/emit_x64_memory.cpp.inc"
#undef Axx

// ARMv8.1 LSE atomics done with one host atomic (UserConfig::native_lse_atomics).
//
// Fast path: the lock-prefixed x86 instruction on the fastmem pointer - xadd for LDADD, xchg for
// SWP, a cmpxchg loop for the other read-modify-writes, cmpxchg for CAS and the 32-bit CASP,
// cmpxchg16b for the 64-bit CASP. A locked instruction is a full barrier, at least as strong as
// every acquire/release variant, and it returns the old value the ARM instruction returns.
// Slow path - a fault in the fastmem arena (unmapped or write-protected pages), a misaligned
// address (an alignment fault on ARMv8.1; a split lock on the host), a site recompiled without
// fastmem: the memory callbacks, MemoryRead and then MemoryWriteExclusive* (a compare-and-swap
// with the value read) until it succeeds, as the exclusive load + store pair did. Neither path
// touches the exclusive monitor; other cores' exclusive stores compare values, so they still
// fail when a native atomic changed the location.

namespace {

// The o3:opc field of LDADD..SWP.
enum : u8 {
    lse_add = 0b0000,
    lse_clr = 0b0001,
    lse_eor = 0b0010,
    lse_set = 0b0011,
    lse_smax = 0b0100,
    lse_smin = 0b0101,
    lse_umax = 0b0110,
    lse_umin = 0b0111,
    lse_swp = 0b1000,
};

template<typename T>
T LseApply(u8 op, T old, T operand) {
    using S = std::make_signed_t<T>;
    switch (op) {
    case lse_add:
        return static_cast<T>(old + operand);
    case lse_clr:
        return static_cast<T>(old & static_cast<T>(~operand));
    case lse_eor:
        return static_cast<T>(old ^ operand);
    case lse_set:
        return static_cast<T>(old | operand);
    case lse_smax:
        return static_cast<S>(old) >= static_cast<S>(operand) ? old : operand;
    case lse_smin:
        return static_cast<S>(old) <= static_cast<S>(operand) ? old : operand;
    case lse_umax:
        return old >= operand ? old : operand;
    case lse_umin:
        return old <= operand ? old : operand;
    default:  // SWP
        return operand;
    }
}

template<typename T>
bool LseWriteExclusive(A64::UserCallbacks& callbacks, u64 vaddr, T value, T expected) {
    if constexpr (sizeof(T) == 1) {
        return callbacks.MemoryWriteExclusive8(vaddr, value, expected);
    } else if constexpr (sizeof(T) == 2) {
        return callbacks.MemoryWriteExclusive16(vaddr, value, expected);
    } else if constexpr (sizeof(T) == 4) {
        return callbacks.MemoryWriteExclusive32(vaddr, value, expected);
    } else {
        return callbacks.MemoryWriteExclusive64(vaddr, value, expected);
    }
}

// A compare that fails still writes the old value back (as the exclusive pair did), so the value
// returned is one the location held as a whole.
template<typename T>
u64 LseFallbackRMW(const A64::UserConfig& conf, u64 vaddr, u64 operand, u64 op) {
    LseStats::CountFallback();
    for (;;) {
        const T old = static_cast<T>(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
        const T next = LseApply<T>(static_cast<u8>(op), old, static_cast<T>(operand));
        if (LseWriteExclusive<T>(*conf.callbacks, vaddr, next, old)) {
            return old;
        }
    }
}

template<typename T>
u64 LseFallbackCAS(const A64::UserConfig& conf, u64 vaddr, u64 expected, u64 desired) {
    LseStats::CountFallback();
    for (;;) {
        const T old = static_cast<T>(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
        const T next = old == static_cast<T>(expected) ? static_cast<T>(desired) : old;
        if (LseWriteExclusive<T>(*conf.callbacks, vaddr, next, old)) {
            return old;
        }
    }
}

// buffer: expected low, high, desired low, high; the old value is returned in [0], [1].
void LseFallbackCAS128(const A64::UserConfig& conf, u64 vaddr, u64* buffer) {
    LseStats::CountFallback();
    const Vector expected{buffer[0], buffer[1]};
    const Vector desired{buffer[2], buffer[3]};
    for (;;) {
        const Vector old = conf.callbacks->MemoryRead128(vaddr);
        if (conf.callbacks->MemoryWriteExclusive128(vaddr, old == expected ? desired : old, old)) {
            buffer[0] = old[0];
            buffer[1] = old[1];
            return;
        }
    }
}

template<size_t bitsize>
Xbyak::Address LseMem(BlockOfCode& code, const Xbyak::RegExp& addr) {
    if constexpr (bitsize == 8) {
        return code.byte[addr];
    } else if constexpr (bitsize == 16) {
        return code.word[addr];
    } else if constexpr (bitsize == 32) {
        return code.dword[addr];
    } else {
        return code.qword[addr];
    }
}

template<size_t bitsize>
Xbyak::Reg LseReg(Xbyak::Reg64 reg) {
    if constexpr (bitsize == 8) {
        return reg.cvt8();
    } else if constexpr (bitsize == 16) {
        return reg.cvt16();
    } else if constexpr (bitsize == 32) {
        return reg.cvt32();
    } else {
        return reg;
    }
}

// cmov has no 8-bit form; the 32-bit one moves the low bits as well.
template<size_t bitsize>
Xbyak::Reg LseCmovReg(Xbyak::Reg64 reg) {
    if constexpr (bitsize == 64) {
        return reg;
    } else {
        return reg.cvt32();
    }
}

// Misaligned atomics take the slow path.
void EmitLseAlignmentCheck(BlockOfCode& code, size_t bitsize, Xbyak::Reg64 vaddr, Xbyak::Label& abort) {
    if (bitsize > 8) {
        code.test(vaddr.cvt32(), u32(bitsize / 8 - 1));
        code.jnz(abort, code.T_NEAR);
    }
}

void EmitLseCountNative(BlockOfCode& code) {
    code.inc(code.qword[code.ABI_JIT_PTR + offsetof(A64JitState, lse_native_ops)]);
}

}  // namespace

template<std::size_t bitsize>
void A64EmitX64::EmitAtomicRMW(A64EmitContext& ctx, IR::Inst* inst) {
    using T = mcl::unsigned_integer_of_size<bitsize>;
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const u8 op = args[3].GetImmediateU8();
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!fastmem_marker) {
        ctx.reg_alloc.HostCall(code, inst, {}, args[1], args[2]);
        code.mov(code.ABI_PARAM4.cvt32(), u32(op));
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackRMW<T>);
        return;
    }

    const bool single_instruction = op == lse_add || op == lse_swp;
    if (!single_instruction) {
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
    }
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    // LDADD/SWP: the operand's copy receives the old value. The others: the old value is in RAX.
    const Xbyak::Reg64 operand = single_instruction ? ctx.reg_alloc.UseScratchGpr(code, args[2]) : ctx.reg_alloc.UseGpr(code, args[2]);
    const Xbyak::Reg64 result = single_instruction ? operand : rax;
    const Xbyak::Reg64 tmp = single_instruction ? operand : ctx.reg_alloc.ScratchGpr(code);

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
    EmitLseAlignmentCheck(code, bitsize, vaddr, *abort);
    bool require_abort_handling = false;
    const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

    std::array<const void*, 2> locations{};
    if (single_instruction) {
        locations[0] = code.getCurr();
        if (op == lse_add) {
            code.lock();
            code.xadd(LseMem<bitsize>(code, dest_ptr), LseReg<bitsize>(operand));
        } else {
            code.xchg(LseMem<bitsize>(code, dest_ptr), LseReg<bitsize>(operand));
        }
        code.ZeroExtendFrom(bitsize, operand);
    } else {
        // Both the load and the cmpxchg can fault (a write-protected page faults on the second).
        locations[0] = code.getCurr();
        if constexpr (bitsize == 8) {
            code.movzx(eax, code.byte[dest_ptr]);
        } else if constexpr (bitsize == 16) {
            code.movzx(eax, code.word[dest_ptr]);
        } else if constexpr (bitsize == 32) {
            code.mov(eax, code.dword[dest_ptr]);
        } else {
            code.mov(rax, code.qword[dest_ptr]);
        }
        Xbyak::Label loop;
        code.L(loop);
        switch (op) {
        case lse_clr:
            if (code.HasHostFeature(HostFeature::BMI1)) {
                // tmp = ~operand & old
                if constexpr (bitsize == 64) {
                    code.andn(tmp, operand, rax);
                } else {
                    code.andn(tmp.cvt32(), operand.cvt32(), eax);
                }
            } else {
                code.mov(tmp, operand);
                code.not_(tmp);
                code.and_(tmp, rax);
            }
            break;
        case lse_eor:
            code.mov(tmp, rax);
            code.xor_(tmp, operand);
            break;
        case lse_set:
            code.mov(tmp, rax);
            code.or_(tmp, operand);
            break;
        case lse_smax:
            code.mov(tmp, rax);
            code.cmp(LseReg<bitsize>(tmp), LseReg<bitsize>(operand));
            code.cmovl(LseCmovReg<bitsize>(tmp), LseCmovReg<bitsize>(operand));
            break;
        case lse_smin:
            code.mov(tmp, rax);
            code.cmp(LseReg<bitsize>(tmp), LseReg<bitsize>(operand));
            code.cmovg(LseCmovReg<bitsize>(tmp), LseCmovReg<bitsize>(operand));
            break;
        case lse_umax:
            code.mov(tmp, rax);
            code.cmp(LseReg<bitsize>(tmp), LseReg<bitsize>(operand));
            code.cmovb(LseCmovReg<bitsize>(tmp), LseCmovReg<bitsize>(operand));
            break;
        case lse_umin:
            code.mov(tmp, rax);
            code.cmp(LseReg<bitsize>(tmp), LseReg<bitsize>(operand));
            code.cmova(LseCmovReg<bitsize>(tmp), LseCmovReg<bitsize>(operand));
            break;
        default:
            UNREACHABLE();
        }
        locations[1] = code.getCurr();
        code.lock();
        code.cmpxchg(LseMem<bitsize>(code, dest_ptr), LseReg<bitsize>(tmp));
        code.jnz(loop);
        // RAX stays zero-extended: the load extended it and a failing cmpxchg writes only its
        // low bits (or, at 32 bits, all of it).
    }
    EmitLseCountNative(code);
    code.L(*end);

    ctx.deferred_emits.emplace_back([=, this] {
        // Called on a fault at a patched instruction (the exception handler makes the call) and
        // from `abort`: does the operation with the callbacks, leaves the old value in `result`.
        Xbyak::Label slow_path;
        const auto slow_path_fn = code.getCurr<void (*)()>();
        code.L(slow_path);
        ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(result.getIdx()));
        code.push(vaddr);
        code.push(operand);
        code.pop(code.ABI_PARAM3);
        code.pop(code.ABI_PARAM2);
        code.mov(code.ABI_PARAM4.cvt32(), u32(op));
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackRMW<T>);
        if (result.getIdx() != code.ABI_RETURN.getIdx()) {
            code.mov(result, code.ABI_RETURN);
        }
        ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(result.getIdx()));
        code.ret();

        code.L(*abort);
        code.call(slow_path);
        const auto resume = code.getCurr();
        code.jmp(*end, code.T_NEAR);
        for (const void* location : locations) {
            if (location) {
                AddFastmemPatch(location, resume, slow_path_fn, *fastmem_marker, conf.recompile_on_fastmem_failure);
            }
        }
    });

    ctx.reg_alloc.DefineValue(code, inst, result);
}

template<std::size_t bitsize>
void A64EmitX64::EmitAtomicCAS(A64EmitContext& ctx, IR::Inst* inst) {
    using T = mcl::unsigned_integer_of_size<bitsize>;
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!fastmem_marker) {
        ctx.reg_alloc.HostCall(code, inst, {}, args[1], args[2], args[3]);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackCAS<T>);
        return;
    }

    // cmpxchg: RAX holds the expected value and receives the old one.
    ctx.reg_alloc.UseScratch(code, args[2], HostLoc::RAX);
    const Xbyak::Reg64 desired = ctx.reg_alloc.UseGpr(code, args[3]);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
    EmitLseAlignmentCheck(code, bitsize, vaddr, *abort);
    bool require_abort_handling = false;
    const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

    const void* location = code.getCurr();
    code.lock();
    code.cmpxchg(LseMem<bitsize>(code, dest_ptr), LseReg<bitsize>(desired));
    EmitLseCountNative(code);
    code.L(*end);
    code.ZeroExtendFrom(bitsize, rax);

    ctx.deferred_emits.emplace_back([=, this] {
        Xbyak::Label slow_path;
        const auto slow_path_fn = code.getCurr<void (*)()>();
        code.L(slow_path);
        ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
        code.push(vaddr);
        code.push(rax);
        code.push(desired);
        code.pop(code.ABI_PARAM4);
        code.pop(code.ABI_PARAM3);
        code.pop(code.ABI_PARAM2);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackCAS<T>);
        ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLoc::RAX);
        code.ret();

        code.L(*abort);
        code.call(slow_path);
        const auto resume = code.getCurr();
        code.jmp(*end, code.T_NEAR);
        AddFastmemPatch(location, resume, slow_path_fn, *fastmem_marker, conf.recompile_on_fastmem_failure);
    });

    ctx.reg_alloc.DefineValue(code, inst, rax);
}

void A64EmitX64::EmitAtomicCAS128(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto fastmem_marker = ShouldFastmem(ctx, inst);
    constexpr size_t buffer_size = 32;

    if (!fastmem_marker) {
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
        ctx.reg_alloc.Use(code, args[1], ABI_PARAM2);
        ctx.reg_alloc.Use(code, args[2], HostLoc::XMM1);
        ctx.reg_alloc.Use(code, args[3], HostLoc::XMM2);
        ctx.reg_alloc.EndOfAllocScope();
        ctx.reg_alloc.HostCall(code, nullptr);
        ctx.reg_alloc.AllocStackSpace(code, buffer_size + ABI_SHADOW_SPACE);
        code.movups(xword[rsp + ABI_SHADOW_SPACE], xmm1);
        code.movups(xword[rsp + ABI_SHADOW_SPACE + 16], xmm2);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackCAS128);
        code.movups(result, xword[rsp + ABI_SHADOW_SPACE]);
        ctx.reg_alloc.ReleaseStackSpace(code, buffer_size + ABI_SHADOW_SPACE);
        ctx.reg_alloc.DefineValue(code, inst, result);
        return;
    }

    // cmpxchg16b: RDX:RAX expected (receives the old value), RCX:RBX desired.
    ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
    ctx.reg_alloc.ScratchGpr(code, HostLoc::RBX);
    ctx.reg_alloc.ScratchGpr(code, HostLoc::RCX);
    ctx.reg_alloc.ScratchGpr(code, HostLoc::RDX);
    const Xbyak::Xmm expected = ctx.reg_alloc.UseXmm(code, args[2]);
    const Xbyak::Xmm desired = ctx.reg_alloc.UseXmm(code, args[3]);
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);

    code.movq(rax, expected);
    code.movq(rbx, desired);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.pextrq(rdx, expected, 1);
        code.pextrq(rcx, desired, 1);
    } else {
        code.movaps(result, expected);
        code.punpckhqdq(result, result);
        code.movq(rdx, result);
        code.movaps(result, desired);
        code.punpckhqdq(result, result);
        code.movq(rcx, result);
    }

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
    EmitLseAlignmentCheck(code, 128, vaddr, *abort);
    bool require_abort_handling = false;
    const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

    const void* location = code.getCurr();
    code.lock();
    code.cmpxchg16b(xword[dest_ptr]);
    code.movq(result, rax);
    if (code.HasHostFeature(HostFeature::SSE41)) {
        code.pinsrq(result, rdx, 1);
    } else {
        code.movq(xmm0, rdx);
        code.punpcklqdq(result, xmm0);
    }
    EmitLseCountNative(code);
    code.L(*end);

    ctx.deferred_emits.emplace_back([=, this] {
        Xbyak::Label slow_path;
        const auto slow_path_fn = code.getCurr<void (*)()>();
        code.L(slow_path);
        ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
        code.sub(rsp, u32(buffer_size + ABI_SHADOW_SPACE));
        code.movups(xword[rsp + ABI_SHADOW_SPACE], expected);
        code.movups(xword[rsp + ABI_SHADOW_SPACE + 16], desired);
        code.mov(code.ABI_PARAM2, vaddr);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        code.mov(code.ABI_PARAM1, std::bit_cast<u64>(&conf));
        code.CallFunction(&LseFallbackCAS128);
        code.movups(result, xword[rsp + ABI_SHADOW_SPACE]);
        code.add(rsp, u32(buffer_size + ABI_SHADOW_SPACE));
        ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
        code.ret();

        code.L(*abort);
        code.call(slow_path);
        const auto resume = code.getCurr();
        code.jmp(*end, code.T_NEAR);
        AddFastmemPatch(location, resume, slow_path_fn, *fastmem_marker, conf.recompile_on_fastmem_failure);
    });

    ctx.reg_alloc.DefineValue(code, inst, result);
}

void A64EmitX64::EmitA64ReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
}

void A64EmitX64::EmitA64ReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryRead<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<8, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<16, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<32, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<64, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64WriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitMemoryWrite<128, &A64::UserCallbacks::MemoryWrite>(ctx, inst);
}

void A64EmitX64::EmitA64ClearExclusive(A64EmitContext&, IR::Inst*) {
    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExclusiveReadMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<8, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<16, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<32, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<64, &A64::UserCallbacks::MemoryRead>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveReadMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveReadMemoryInline<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    } else {
        EmitExclusiveReadMemory<128, &A64::UserCallbacks::MemoryRead128>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory8(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<8, &A64::UserCallbacks::MemoryWriteExclusive8>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory16(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<16, &A64::UserCallbacks::MemoryWriteExclusive16>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory32(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<32, &A64::UserCallbacks::MemoryWriteExclusive32>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory64(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<64, &A64::UserCallbacks::MemoryWriteExclusive64>(ctx, inst);
    }
}

void A64EmitX64::EmitA64ExclusiveWriteMemory128(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.fastmem_exclusive_access) {
        EmitExclusiveWriteMemoryInline<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    } else {
        EmitExclusiveWriteMemory<128, &A64::UserCallbacks::MemoryWriteExclusive128>(ctx, inst);
    }
}

void A64EmitX64::EmitA64AtomicRMW8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRMW<8>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRMW16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRMW<16>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRMW32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRMW<32>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicRMW64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicRMW<64>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCAS8(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCAS<8>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCAS16(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCAS<16>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCAS32(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCAS<32>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCAS64(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCAS<64>(ctx, inst);
}

void A64EmitX64::EmitA64AtomicCAS128(A64EmitContext& ctx, IR::Inst* inst) {
    EmitAtomicCAS128(ctx, inst);
}

void A64EmitX64::EmitCheckMemoryAbort(A64EmitContext&, IR::Inst* inst, Xbyak::Label* end) {
    if (!conf.check_halt_on_memory_access) {
        return;
    }

    Xbyak::Label skip;

    const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

    code.test(dword[code.ABI_JIT_PTR + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    if (end) {
        code.jz(*end, code.T_NEAR);
    } else {
        code.jz(skip, code.T_NEAR);
    }
    code.mov(rax, current_location.PC());
    code.mov(qword[code.ABI_JIT_PTR + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
    code.L(skip);
}

}  // namespace Dynarmic::Backend::X64
