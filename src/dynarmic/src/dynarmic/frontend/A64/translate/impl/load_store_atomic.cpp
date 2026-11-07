// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// ARMv8.1 LSE atomics (LDADD/LDCLR/LDEOR/LDSET/LD{S,U}{MAX,MIN}/SWP, CAS, CASP) and
// ARMv8.3 RCpc LDAPR.
//
// With TranslationOptions::native_lse_atomics each one is a single A64AtomicRMW/A64AtomicCAS
// instruction that returns the old value, done by the backend with one host atomic.
//
// Otherwise each read-modify-write is built from an exclusive load and an exclusive store. If
// the store fails (another core touched the monitor in between) no register is changed and the
// same instruction is executed again, so the operation stays atomic with multiple cores.

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

static Reg AtomicNextReg(Reg r) {
    return static_cast<Reg>(static_cast<size_t>(r) + 1);
}

static IR::U64 AtomicAddress(TranslatorVisitor& v, Reg Rn) {
    return Rn == Reg::SP ? IR::U64{v.SP(64)} : IR::U64{v.X(64, Rn)};
}

// All-ones when value == 0, otherwise zero.
static IR::U64 AtomicMaskIfZero(TranslatorVisitor& v, IR::U64 value) {
    const IR::U64 nonzero = v.ir.LogicalShiftRight(v.ir.Or(value, v.ir.Sub(v.ir.Imm64(0), value)), v.ir.Imm8(63));
    return v.ir.Sub(nonzero, v.ir.Imm64(1));
}

static IR::U64 AtomicSelect(TranslatorVisitor& v, IR::U64 mask, IR::U64 if_set, IR::U64 if_clear) {
    return v.ir.Or(v.ir.And(if_set, mask), v.ir.AndNot(if_clear, mask));
}

static IR::UAny AtomicTruncate(TranslatorVisitor& v, IR::U64 value, size_t bytes) {
    switch (bytes) {
    case 1:
        return v.ir.LeastSignificantByte(value);
    case 2:
        return v.ir.LeastSignificantHalf(value);
    case 4:
        return v.ir.LeastSignificantWord(value);
    default:
        return value;
    }
}

static IR::U64 AtomicLoadZext(TranslatorVisitor& v, IR::U64 address, size_t bytes, IR::AccType acctype) {
    const IR::UAny value = v.ExclusiveMem(address, bytes, acctype);
    return bytes == 8 ? IR::U64{value} : v.ir.ZeroExtendToLong(value);
}

static IR::U64 AtomicSignExtend(TranslatorVisitor& v, IR::U64 value, size_t bytes) {
    return bytes == 8 ? value : v.ir.SignExtendToLong(AtomicTruncate(v, value, bytes));
}

static IR::U64 AtomicZeroExtendReg(TranslatorVisitor& v, Reg r, size_t bytes) {
    const IR::U64 value = v.X(64, r);
    return bytes == 8 ? value : v.ir.ZeroExtendToLong(AtomicTruncate(v, value, bytes));
}

// Writes `value` to `r` only when the exclusive store succeeded (ok_mask all-ones).
static void AtomicCommitReg(TranslatorVisitor& v, Reg r, IR::U64 ok_mask, IR::U64 value) {
    if (r == Reg::ZR) {
        return;
    }
    v.X(64, r, AtomicSelect(v, ok_mask, value, v.X(64, r)));
}

// Ends the block: continue with the next instruction on success, retry this one on failure.
static bool AtomicFinish(TranslatorVisitor& v, IR::U32 status) {
    v.ir.SetCheckBit(v.ir.IsZero(status));
    v.ir.SetTerm(IR::Term::CheckBit{IR::Term::LinkBlock{v.ir.current_location->AdvancePC(4)},
                                    IR::Term::LinkBlock{*v.ir.current_location}});
    return false;
}

static IR::AccType AtomicAccType(bool acquire, bool release) {
    return acquire || release ? IR::AccType::ORDERED : IR::AccType::ATOMIC;
}

// Writes the zero-extended `bytes`-byte value to Xr (no write for ZR).
static void AtomicSetReg(TranslatorVisitor& v, Reg r, IR::UAny value, size_t bytes) {
    if (bytes == 8) {
        v.X(64, r, IR::U64{value});
    } else {
        v.X(32, r, v.ZeroExtend(value, 32));
    }
}

// Native lowering: the host atomic is a full barrier, which covers every acquire/release variant.
// No block end and no retry: the backend's atomic always completes.
static bool NativeAtomicMemOp(TranslatorVisitor& v, size_t bytes, Reg Rs, size_t opv, Reg Rn, Reg Rt) {
    const IR::U64 address = AtomicAddress(v, Rn);
    const IR::UAny old = v.ir.AtomicRMW(bytes, address, v.X(bytes * 8, Rs), static_cast<u8>(opv));
    AtomicSetReg(v, Rt, old, bytes);
    return true;
}

static bool NativeCompareAndSwap(TranslatorVisitor& v, size_t bytes, Reg Rs, Reg Rn, Reg Rt) {
    const IR::U64 address = AtomicAddress(v, Rn);
    const IR::UAny old = v.ir.AtomicCAS(bytes, address, v.X(bytes * 8, Rs), v.X(bytes * 8, Rt));
    AtomicSetReg(v, Rs, old, bytes);
    return true;
}

static bool NativeCompareAndSwapPair(TranslatorVisitor& v, bool sz, Reg Rs, Reg Rs2, Reg Rn, Reg Rt, Reg Rt2) {
    const IR::U64 address = AtomicAddress(v, Rn);
    if (!sz) {
        // Two 32-bit words, one 64-bit compare-and-swap (the first word is the low half).
        const IR::U64 expected = v.ir.Pack2x32To1x64(v.X(32, Rs), v.X(32, Rs2));
        const IR::U64 desired = v.ir.Pack2x32To1x64(v.X(32, Rt), v.X(32, Rt2));
        const IR::U64 old{v.ir.AtomicCAS(8, address, expected, desired)};
        v.X(32, Rs, v.ir.LeastSignificantWord(old));
        v.X(64, Rs2, v.ir.LogicalShiftRight(old, v.ir.Imm8(32)));
        return true;
    }
    const IR::U128 expected = v.ir.Pack2x64To1x128(v.X(64, Rs), v.X(64, Rs2));
    const IR::U128 desired = v.ir.Pack2x64To1x128(v.X(64, Rt), v.X(64, Rt2));
    const IR::U128 old = v.ir.AtomicCAS128(address, expected, desired);
    v.X(64, Rs, v.ir.VectorGetElement(64, old, 0));
    v.X(64, Rs2, v.ir.VectorGetElement(64, old, 1));
    return true;
}

bool TranslatorVisitor::AtomicMemOp(Imm<2> size, bool A, bool R, Reg Rs, Imm<4> op, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    const size_t opv = op.ZeroExtend<size_t>();
    if (opv > 0b1000) {
        return UnallocatedEncoding();
    }
    if (options.native_lse_atomics) {
        return NativeAtomicMemOp(*this, bytes, Rs, opv, Rn, Rt);
    }

    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(A, R);
    const IR::U64 old = AtomicLoadZext(*this, address, bytes, acctype);
    const IR::U64 s = AtomicZeroExtendReg(*this, Rs, bytes);

    IR::U64 result;
    switch (opv) {
    case 0b0000:  // LDADD
        result = ir.Add(old, s);
        break;
    case 0b0001:  // LDCLR
        result = ir.AndNot(old, s);
        break;
    case 0b0010:  // LDEOR
        result = ir.Eor(old, s);
        break;
    case 0b0011:  // LDSET
        result = ir.Or(old, s);
        break;
    case 0b0100:  // LDSMAX
        result = ir.MaxSigned(AtomicSignExtend(*this, old, bytes), AtomicSignExtend(*this, s, bytes));
        break;
    case 0b0101:  // LDSMIN
        result = ir.MinSigned(AtomicSignExtend(*this, old, bytes), AtomicSignExtend(*this, s, bytes));
        break;
    case 0b0110:  // LDUMAX
        result = ir.MaxUnsigned(old, s);
        break;
    case 0b0111:  // LDUMIN
        result = ir.MinUnsigned(old, s);
        break;
    default:  // SWP
        result = s;
        break;
    }

    const IR::U32 status = ExclusiveMem(address, bytes, acctype, AtomicTruncate(*this, result, bytes));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rt, ok, old);
    return AtomicFinish(*this, status);
}

bool TranslatorVisitor::LDAPR_sz(Imm<2> size, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    const size_t regsize = bytes == 8 ? 64 : 32;
    const IR::UAny data = Mem(AtomicAddress(*this, Rn), bytes, IR::AccType::ORDERED);
    X(regsize, Rt, ZeroExtend(data, regsize));
    return true;
}

bool TranslatorVisitor::CompareAndSwap(Imm<2> size, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    const size_t bytes = size_t{1} << size.ZeroExtend<size_t>();
    if (options.native_lse_atomics) {
        return NativeCompareAndSwap(*this, bytes, Rs, Rn, Rt);
    }
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(L, o0);

    const IR::U64 old = AtomicLoadZext(*this, address, bytes, acctype);
    const IR::U64 expected = AtomicZeroExtendReg(*this, Rs, bytes);
    const IR::U64 desired = AtomicZeroExtendReg(*this, Rt, bytes);
    const IR::U64 equal = AtomicMaskIfZero(*this, ir.Eor(old, expected));
    const IR::U64 result = AtomicSelect(*this, equal, desired, old);

    const IR::U32 status = ExclusiveMem(address, bytes, acctype, AtomicTruncate(*this, result, bytes));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rs, ok, old);
    return AtomicFinish(*this, status);
}

bool TranslatorVisitor::CompareAndSwapPair(bool sz, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    if ((static_cast<size_t>(Rs) & 1) != 0 || (static_cast<size_t>(Rt) & 1) != 0) {
        return UnallocatedEncoding();
    }
    const Reg Rs2 = AtomicNextReg(Rs);
    const Reg Rt2 = AtomicNextReg(Rt);
    if (options.native_lse_atomics) {
        return NativeCompareAndSwapPair(*this, sz, Rs, Rs2, Rn, Rt, Rt2);
    }
    const IR::U64 address = AtomicAddress(*this, Rn);
    const IR::AccType acctype = AtomicAccType(L, o0);

    if (!sz) {
        // Two 32-bit words, accessed as one 64-bit exclusive.
        const IR::U64 old = ExclusiveMem(address, 8, acctype);
        const IR::U64 expected = ir.Pack2x32To1x64(X(32, Rs), X(32, Rs2));
        const IR::U64 desired = ir.Pack2x32To1x64(X(32, Rt), X(32, Rt2));
        const IR::U64 equal = AtomicMaskIfZero(*this, ir.Eor(old, expected));
        const IR::U32 status = ExclusiveMem(address, 8, acctype, AtomicSelect(*this, equal, desired, old));
        const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
        AtomicCommitReg(*this, Rs, ok, ir.ZeroExtendToLong(ir.LeastSignificantWord(old)));
        AtomicCommitReg(*this, Rs2, ok, ir.LogicalShiftRight(old, ir.Imm8(32)));
        return AtomicFinish(*this, status);
    }

    const IR::U128 old = ExclusiveMem(address, 16, acctype);
    const IR::U64 old_lo = ir.VectorGetElement(64, old, 0);
    const IR::U64 old_hi = ir.VectorGetElement(64, old, 1);
    const IR::U64 diff = ir.Or(ir.Eor(old_lo, X(64, Rs)), ir.Eor(old_hi, X(64, Rs2)));
    const IR::U64 equal = AtomicMaskIfZero(*this, diff);
    const IR::U64 new_lo = AtomicSelect(*this, equal, X(64, Rt), old_lo);
    const IR::U64 new_hi = AtomicSelect(*this, equal, X(64, Rt2), old_hi);
    const IR::U32 status = ExclusiveMem(address, 16, acctype, ir.Pack2x64To1x128(new_lo, new_hi));
    const IR::U64 ok = AtomicMaskIfZero(*this, ir.ZeroExtendToLong(status));
    AtomicCommitReg(*this, Rs, ok, old_lo);
    AtomicCommitReg(*this, Rs2, ok, old_hi);
    return AtomicFinish(*this, status);
}

}  // namespace Dynarmic::A64
