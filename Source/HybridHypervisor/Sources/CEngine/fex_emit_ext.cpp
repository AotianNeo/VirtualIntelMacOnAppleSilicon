// SPDX-License-Identifier: MIT
// FEX-Emu CodeEmitter & AVX_128 / Vector / EncryptionOps direct ARM64 lowering bridge.
#include <CodeEmitter/Emitter.h>
#include <cstdint>

using namespace ARMEmitter;

namespace {

inline VRegister V(int r) { return static_cast<VRegister>(r); }
inline XRegister X(int r) { return static_cast<XRegister>(r); }

struct DirectFexEmitter final : public Emitter {
    uint32_t **cc_out;
    explicit DirectFexEmitter(uint32_t **cc_ptr)
        : Emitter(reinterpret_cast<uint8_t *>(*cc_ptr), 4096), cc_out(cc_ptr) {}
    ~DirectFexEmitter() {
        *cc_out = reinterpret_cast<uint32_t *>(Buffer::GetCursorAddress<uint8_t *>());
    }
};

} // namespace

extern "C" {

// FEX AVX_128.cpp:1341-1363 — AVX128_VPERMQ 16-way per-half lowering.
void fex_emit_permq_half(uint32_t **cc_ptr, int d, int s_lo, int s_hi, uint8_t sel4) {
    DirectFexEmitter e(cc_ptr);
    VRegister vd = V(d), lo = V(s_lo), hi = V(s_hi);
    switch (sel4 & 0xf) {
        case 0b00'00: e.dup(SubRegSize::i64Bit, vd.Q(), lo.Q(), 0); break;
        case 0b00'01: e.ext(vd.Q(), lo.Q(), lo.Q(), 8); break;
        case 0b00'10: e.zip1(SubRegSize::i64Bit, vd.Q(), hi.Q(), lo.Q()); break;
        case 0b00'11: e.ext(vd.Q(), hi.Q(), lo.Q(), 8); break;
        case 0b01'00: if (d != s_lo) e.mov(vd.Q(), lo.Q()); break;
        case 0b01'01: e.dup(SubRegSize::i64Bit, vd.Q(), lo.Q(), 1); break;
        case 0b01'10:
            if (d != s_hi) e.mov(vd.Q(), hi.Q());
            e.ins(SubRegSize::i64Bit, vd, 1, lo, 1);
            break;
        case 0b01'11: e.trn2(SubRegSize::i64Bit, vd.Q(), hi.Q(), lo.Q()); break;
        case 0b10'00: e.zip1(SubRegSize::i64Bit, vd.Q(), lo.Q(), hi.Q()); break;
        case 0b10'01: e.ext(vd.Q(), lo.Q(), hi.Q(), 8); break;
        case 0b10'10: e.dup(SubRegSize::i64Bit, vd.Q(), hi.Q(), 0); break;
        case 0b10'11: e.ext(vd.Q(), hi.Q(), hi.Q(), 8); break;
        case 0b11'00:
            if (d != s_lo) e.mov(vd.Q(), lo.Q());
            e.ins(SubRegSize::i64Bit, vd, 1, hi, 1);
            break;
        case 0b11'01: e.trn2(SubRegSize::i64Bit, vd.Q(), lo.Q(), hi.Q()); break;
        case 0b11'10: if (d != s_hi) e.mov(vd.Q(), hi.Q()); break;
        case 0b11'11: e.dup(SubRegSize::i64Bit, vd.Q(), hi.Q(), 1); break;
    }
}

// FEX Vector.cpp:1500-1775 — Single128Bit4ByteVectorShuffle fast cases.
bool fex_emit_pshufd128(uint32_t **cc_ptr, int d, int s, uint8_t imm8, int t0) {
    DirectFexEmitter e(cc_ptr);
    VRegister vd = V(d), vs = V(s), vt = V(t0);
    switch (imm8) {
        case 0b11'10'01'00: if (d != s) e.mov(vd.Q(), vs.Q()); return true;
        case 0b00'00'00'00: case 0b01'01'01'01: case 0b10'10'10'10: case 0b11'11'11'11:
            e.dup(SubRegSize::i32Bit, vd.Q(), vs.Q(), imm8 & 3); return true;
        case 0b01'00'01'00: e.dup(SubRegSize::i64Bit, vd.Q(), vs.Q(), 0); return true;
        case 0b11'10'11'10: e.dup(SubRegSize::i64Bit, vd.Q(), vs.Q(), 1); return true;
        case 0b10'11'00'01: e.rev64(SubRegSize::i32Bit, vd.Q(), vs.Q()); return true;
        case 0b01'00'11'10: e.ext(vd.Q(), vs.Q(), vs.Q(), 8); return true;
        case 0b00'11'10'01: e.ext(vd.Q(), vs.Q(), vs.Q(), 4); return true;
        case 0b10'01'00'11: e.ext(vd.Q(), vs.Q(), vs.Q(), 12); return true;
        case 0b01'01'00'00: e.zip1(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b11'11'10'10: e.zip2(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b10'00'10'00: e.uzp1(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b11'01'11'01: e.uzp2(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b10'10'00'00: e.trn1(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b11'11'01'01: e.trn2(SubRegSize::i32Bit, vd.Q(), vs.Q(), vs.Q()); return true;
        case 0b00'01'10'11:
            e.rev64(SubRegSize::i32Bit, vt.Q(), vs.Q());
            e.ext(vd.Q(), vt.Q(), vt.Q(), 8);
            return true;
        case 0b00'01'00'01:
            e.rev64(SubRegSize::i32Bit, vt.Q(), vs.Q());
            e.zip1(SubRegSize::i64Bit, vd.Q(), vt.Q(), vt.Q());
            return true;
        case 0b10'11'10'11:
            e.rev64(SubRegSize::i32Bit, vt.Q(), vs.Q());
            e.zip2(SubRegSize::i64Bit, vd.Q(), vt.Q(), vt.Q());
            return true;
        case 0b00'11'00'11:
            e.ext(vt.Q(), vs.Q(), vs.Q(), 4);
            e.zip2(SubRegSize::i64Bit, vd.Q(), vt.Q(), vt.Q());
            return true;
        default:
            return false;
    }
}

// FEX Vector.cpp:4182-4310 — VectorBlendImpl for 32-bit elements.
void fex_emit_vblend32_128(uint32_t **cc_ptr, int d, int n, int s, uint8_t mask4, int t0) {
    DirectFexEmitter e(cc_ptr);
    VRegister vd = V(d), vn = V(n), vs = V(s), vt = V(t0);
    mask4 &= 0xf;
    if (mask4 == 0b0000) {
        if (d != n) e.mov(vd.Q(), vn.Q());
        return;
    }
    if (mask4 == 0b1111) {
        if (d != s) e.mov(vd.Q(), vs.Q());
        return;
    }
    if (mask4 == 0b0011 || mask4 == 0b1100) {
        int lane64 = (mask4 == 0b1100) ? 1 : 0;
        VRegister dst = (d == s && d != n) ? vt : vd;
        if (dst != vn) e.mov(dst.Q(), vn.Q());
        e.ins(SubRegSize::i64Bit, dst, lane64, vs, lane64);
        if (dst != vd) e.mov(vd.Q(), dst.Q());
        return;
    }
    if (mask4 == 0b0101) {
        e.rev64(SubRegSize::i32Bit, vt.Q(), vs.Q());
        e.trn2(SubRegSize::i32Bit, vd.Q(), vt.Q(), vn.Q());
        return;
    }
    if (mask4 == 0b1010) {
        e.rev64(SubRegSize::i32Bit, vt.Q(), vn.Q());
        e.trn2(SubRegSize::i32Bit, vd.Q(), vt.Q(), vs.Q());
        return;
    }
    VRegister dst = (d == s && d != n) ? vt : vd;
    if (dst != vn) e.mov(dst.Q(), vn.Q());
    for (int i = 0; i < 4; i++) {
        if ((mask4 >> i) & 1)
            e.ins(SubRegSize::i32Bit, dst, i, vs, i);
    }
    if (dst != vd) e.mov(vd.Q(), dst.Q());
}

// FEX EncryptionOps.cpp:16-98 — In-place VAESEnc / VAESEncLast / VAESDec / VAESDecLast.
void fex_emit_vaes128(uint32_t **cc_ptr, int d, int n, int s, bool dec, bool last, int t0, int t1) {
    DirectFexEmitter e(cc_ptr);
    VRegister vd = V(d), vn = V(n), vs = V(s), vz = V(t0), vtmp = V(t1);
    e.movi(SubRegSize::i64Bit, vz.Q(), 0);
    VRegister work = (d == s) ? vtmp : vd;
    if (work != vn) e.mov(work.Q(), vn.Q());
    if (dec) {
        e.aesd(work.Q(), vz.Q());
        if (!last) e.aesimc(work.Q(), work.Q());
    } else {
        e.aese(work.Q(), vz.Q());
        if (!last) e.aesmc(work.Q(), work.Q());
    }
    e.eor(vd.Q(), work.Q(), vs.Q());
}

// FEX Vector.cpp:5015-5075 — VPERMDIndices: pure NEON index construction without GPR->SIMD transfers or mul.
void fex_emit_vpermd_indices(uint32_t **cc_ptr, int dst, int idx, int t_mask, int t_rep, int t_tmp) {
    DirectFexEmitter e(cc_ptr);
    VRegister vd = V(dst), vidx = V(idx), vmask = V(t_mask), vrep = V(t_rep), vt = V(t_tmp);
    e.and_(vd.Q(), vidx.Q(), vmask.Q());
    e.trn1(SubRegSize::i8Bit, vt.Q(), vd.Q(), vd.Q());
    e.trn1(SubRegSize::i16Bit, vd.Q(), vt.Q(), vt.Q());
    e.shl(SubRegSize::i8Bit, vd.Q(), vd.Q(), 2);
    e.add(SubRegSize::i8Bit, vd.Q(), vd.Q(), vrep.Q());
}

// FEX AVX_128.cpp:880-907 — AVX128_MOVMSKB: pure NEON 128-bit -> 16-bit byte mask with single umov.
void fex_emit_pmovmskb128(uint32_t **cc_ptr, int rd_gpr, int src_vec, int t_mask, int t0, int t1) {
    DirectFexEmitter e(cc_ptr);
    VRegister vs = V(src_vec), vm = V(t_mask), vt0 = V(t0), vt1 = V(t1);
    e.cmlt(SubRegSize::i8Bit, vt0.Q(), vs.Q());
    e.and_(vt0.Q(), vt0.Q(), vm.Q());
    e.addp(SubRegSize::i8Bit, vt0.Q(), vt0.Q(), vt0.Q());
    e.addp(SubRegSize::i8Bit, vt0.Q(), vt0.Q(), vt0.Q());
    e.addp(SubRegSize::i8Bit, vt0.D(), vt0.D(), vt0.D());
    e.umov<SubRegSize::i16Bit>(X(rd_gpr), vt0, 0);
}

} // extern "C"
