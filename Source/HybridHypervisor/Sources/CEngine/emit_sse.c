// SSE and SSE2 to NEON. Where supported, FPCR.AH selects alternate FP behavior for min/max and NaNs.
// MXCSR status lives in FPSR. Unmasked FP exceptions cannot be trapped on this hardware, so the monitor stops on them.
#include <stddef.h>
#include "engine_api.h"
#include "tr_internal.h"

static inline void sse_aes_flush_one(int r) {
    if (tr_aes_mask & (1u << r)) {
        E(0x6E201C00u | (u32)tr_aes_key[r] << 16 | (u32)r << 5 | (u32)r); // eor vr.16b, vr.16b, vk.16b
        tr_aes_mask &= (u16)~(1u << r);
    }
}

void sse_aes_flush_all(void) {
    u16 m = tr_aes_mask;
    tr_aes_mask = 0;
    for (int r = 0; m; r++, m >>= 1)
        if (m & 1)
            E(0x6E201C00u | (u32)tr_aes_key[r] << 16 | (u32)r << 5 | (u32)r);
}

static void logic_imm(int sf, int opc, int rd, int rn, u64 imm) {
    u32 enc = 0;
    logimm(imm, sf, &enc);
    u32 base = opc == 0 ? 0x12000000u : opc == 1 ? 0x32000000u : 0x52000000u;
    if (sf)
        base |= 0x80000000u;
    E(base | enc | (u32)rn << 5 | (u32)rd);
}

static void patch_tb(u32 *at, const u32 *target) {
    *at |= ((u32)(target - at) & 0x3fffu) << 5;
}

static void patch_cb(u32 *at, const u32 *target) {
    *at |= ((u32)(target - at) & 0x7ffffu) << 5;
}

static void patch_b(u32 *at, const u32 *target) {
    *at |= (u32)(target - at) & 0x3ffffffu;
}

// Monitor mxcsr_apply, for the value Geekbench actually writes: exceptions masked,
// no reserved bits, CR0.TS clear. Anything else is still the SVC.
void emit_ldmxcsr_s0(void) {
    ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
    u32 *to_slow_ts = tr_cc;
    tbz(1, S1, 3, 0);
    lsr_imm(0, S1, S0, 16);
    u32 *to_slow_hi = tr_cc;
    cbz(1, 0, S1, 0);
    logic_imm(1, 0, S1, S0, 0x1F80);
    logic_imm(1, 2, S1, S1, 0x1F80);
    u32 *to_slow_mask = tr_cc;
    cbz(1, 1, S1, 0);

    logic_imm(1, 0, S2, S0, 0xFFC0);
    ldst(1, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, mxcsr)});
    logic_reg(0, 2, 0, S3, S2, S3);
    u32 *skip_fpcr = tr_cc;
    cbz(0, 0, S3, 0);

    movimm(S1, 6);
    u32 *skip_daz = tr_cc;
    tbz(0, S0, 6, 0);
    logic_imm(1, 1, S1, S1, 1);
    patch_tb(skip_daz, tr_cc);
    u32 *skip_ftz = tr_cc;
    tbz(0, S0, 15, 0);
    logic_imm(1, 1, S1, S1, 1ULL << 24);
    patch_tb(skip_ftz, tr_cc);
    ubfx(0, S2, S0, 13, 2);
    lsr_imm(0, S3, S2, 1);
    logic_imm(1, 0, S2, S2, 1);
    logic_sh(0, 1, 0, S2, S3, S2, 0, 1);
    logic_sh(1, 1, 0, S1, S1, S2, 0, 22);
    E(0xD51B4400u | S1); // msr fpcr, x
    logic_imm(1, 0, S1, S0, 0xFFC0);
    ldst(0, 4, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, mxcsr)});
    patch_cb(skip_fpcr, tr_cc);

    logic_imm(1, 0, S1, S0, 1);
    ubfx(0, S2, S0, 2, 4);
    logic_sh(0, 1, 0, S1, S1, S2, 0, 1);
    u32 *skip_de = tr_cc;
    tbz(0, S0, 1, 0);
    logic_imm(1, 1, S1, S1, 128);
    patch_tb(skip_de, tr_cc);
    E(0xD53B4420u | S2); // mrs x, fpsr
    logic_reg(1, 2, 0, S2, S2, S1);
    u32 *skip_fpsr = tr_cc;
    cbz(0, 1, S2, 0);
    E(0xD51B4420u | S1); // msr fpsr, x
    patch_cb(skip_fpsr, tr_cc);
    u32 *skip_svc = tr_cc;
    b_rel(0);

    u32 *slow = tr_cc;
    svc(SVC_LDMXCSR);
    patch_tb(to_slow_ts, slow);
    patch_cb(to_slow_hi, slow);
    patch_cb(to_slow_mask, slow);
    patch_b(skip_svc, tr_cc);
}

void emit_stmxcsr_s0(void) {
    ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
    u32 *to_slow = tr_cc;
    tbz(1, S1, 3, 0);
    E(0xD53B4420u | S0); // mrs x, fpsr
    ldst(1, 4, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, mxcsr)});
    logic_imm(1, 0, S2, S0, 1);
    ubfx(0, S3, S0, 7, 1);
    logic_sh(0, 1, 0, S2, S2, S3, 0, 1);
    ubfx(0, S3, S0, 1, 4);
    logic_sh(0, 1, 0, S2, S2, S3, 0, 2);
    logic_reg(0, 1, 0, S0, S1, S2);
    u32 *skip_svc = tr_cc;
    b_rel(0);
    u32 *slow = tr_cc;
    svc(SVC_STMXCSR);
    patch_tb(to_slow, slow);
    patch_b(skip_svc, tr_cc);
}

// kind: 2 s, 3 d, 4 q
static void vldst(int kind, bool load, int rt, Addr a) {
    u32 size = kind == 4 ? 0 : (u32)kind;
    u32 opc = kind == 4 ? (load ? 3 : 2) : (load ? 1 : 0);
    int bytes = 1 << kind;
    if (a.reg_form) {
        E(size << 30 | 0x3C200800u | opc << 22 | a.index << 16 | 3u << 13 | (a.shift ? 1u : 0) << 12 | a.base << 5 | rt);
    } else if (a.imm >= 0 && !(a.imm & (bytes - 1)) && (a.imm >> kind) < 4096) {
        E(size << 30 | 0x3D000000u | opc << 22 | (u32)(a.imm >> kind) << 10 | a.base << 5 | rt);
    } else if (a.imm >= -256 && a.imm < 256) {
        E(size << 30 | 0x3C000000u | opc << 22 | ((u32)a.imm & 0x1ff) << 12 | a.base << 5 | rt);
    } else {
        movimm(S4, (u64)a.imm);
        E(size << 30 | 0x3C200800u | opc << 22 | S4 << 16 | 3u << 13 | a.base << 5 | rt);
    }
}

// #GP(0) unless 16-byte aligned, as every legacy SSE memory operand of that size requires.
static void align16(int r) {
    ubfx(1, S0, r, 0, 4);
    cbz(0, 1, S0, 2);
    svc(SVC_GP0);
}

static Addr mem16(const Insn *in, bool check) {
    if (!check || !tr_precise_flags)
        return addr_prepare(in, 16);
    int r = addr_reg(in);
    // RIP-relative addresses without FS/GS have known low bits. Address-size
    // truncation preserves them, so an aligned constant cannot take #GP here.
    bool aligned_constant = in->riprel && !in->seg && !((in->next + (u64)in->disp) & 15);
    if (!aligned_constant)
        align16(r);
    Addr a = {.base = r};
    return a;
}

// Returns the v-reg holding the rm operand, loading kind-sized memory into V_MEM.
static int src_op(const Insn *in, int kind, bool check) {
    if (in->rm_reg)
        return in->rm;
    if (kind == 4) {
        vldst(4, true, V_MEM, mem16(in, check));
    } else {
        vldst(kind, true, V_MEM, addr_prepare(in, 1 << kind));
    }
    return V_MEM;
}

#define RRR(base, rd, rn, rm) E((u32)(base) | (u32)(rm) << 16 | (u32)(rn) << 5 | (rd))
#define RR(base, rd, rn) E((u32)(base) | (u32)(rn) << 5 | (rd))

static void vmov(int rd, int rn) {
    if (rd != rn)
        E(0x4EA01C00u | rn << 16 | rn << 5 | rd);
}
// ins: lane sizes 1,2,4,8 bytes
static void ins_elem(int rd, int di, int rn, int si, int bytes) {
    int ls = log2sz(bytes);
    u32 imm5 = (u32)(di << (ls + 1)) | (1u << ls);
    u32 imm4 = (u32)si << ls;
    E(0x6E000400u | imm5 << 16 | imm4 << 11 | rn << 5 | rd);
}

// Legacy scalar SSE replaces only the low lane. On hosts without FEAT_AFP
// (including M1), NEP reads as zero and a scalar ARM result clears upper lanes.
static void scalar_fp_result(u32 opcode, int dest, int src, int bytes) {
    RR(opcode, tr_fpcr_merges_scalar ? dest : V_S0, src);
    if (!tr_fpcr_merges_scalar)
        ins_elem(dest, 0, V_S0, 0, bytes);
}

static void umov_d(int rd, int rn, int lane) { E(0x4E083C00u | (u32)lane << 20 | rn << 5 | rd); }

static void bad(const Insn *in, const char *why) { emit_stop(in->rip, why); }

// Floating-point arithmetic, packed and scalar. sel: pfx-derived. Encoded as (packed 4s base, scalar s base).
typedef struct {
    u32 packed, scalar;
} FpOp;
static const FpOp fp_add = {0x4E20D400u, 0x1E202800u}, fp_sub = {0x4EA0D400u, 0x1E203800u}, fp_mul = {0x6E20DC00u, 0x1E200800u},
                  fp_div = {0x6E20FC00u, 0x1E201800u}, fp_min = {0x4EA0F400u, 0x1E205800u}, fp_max = {0x4E20F400u, 0x1E204800u};

static void fp_arith(const Insn *in, FpOp op) {
    int kind = in->pfx == 0xF3 ? 2 : in->pfx == 0xF2 ? 3 : 4;
    int s = src_op(in, kind, true);
    if (kind == 4) {
        u32 enc = in->pfx == 0x66 ? op.packed | 1u << 22 : op.packed;
        RRR(enc, in->reg, in->reg, s);
    } else {
        u32 enc = kind == 3 ? op.scalar | 1u << 22 : op.scalar;
        RRR(enc, tr_fpcr_merges_scalar ? in->reg : V_S0, in->reg, s);
        if (!tr_fpcr_merges_scalar)
            ins_elem(in->reg, 0, V_S0, 0, 1 << kind);
    }
}

// x86 cmpps predicates.
static void emit_cmp_pred(const Insn *in) {
    int kind = in->pfx == 0xF3 ? 2 : in->pfx == 0xF2 ? 3 : 4;
    int s = src_op(in, kind, true), d = in->reg, pred = (int)in->imm & 7;
    if (in->imm & ~7LL) {
        bad(in, "cmpps with an immediate above 7");
        return;
    }
    bool isd = kind == 3 || (kind == 4 && in->pfx == 0x66);
    u32 sz = isd ? 1u << 22 : 0;
    u32 eq = (kind == 4 ? 0x4E20E400u : 0x5E20E400u) | sz, ge = (kind == 4 ? 0x6E20E400u : 0x7E20E400u) | sz,
        gt = (kind == 4 ? 0x6EA0E400u : 0x7EA0E400u) | sz;
    int t = (kind == 4 && pred != 3 && pred != 7) ? d : V_S0;
    switch (pred) {
        case 0:
            RRR(eq, t, d, s);
            break;
        case 1:
            RRR(gt, t, s, d);
            break;
        case 2:
            RRR(ge, t, s, d);
            break;
        case 3:
        case 7:
            RRR(eq, t, d, d);
            RRR(eq, V_S1, s, s);
            E(0x4E201C00u | V_S1 << 16 | t << 5 | t); // and
            if (pred == 3)
                E(0x6E205800u | t << 5 | t); // mvn
            break;
        case 4:
            RRR(eq, t, d, s);
            E(0x6E205800u | t << 5 | t);
            break;
        case 5:
            RRR(gt, t, s, d);
            E(0x6E205800u | t << 5 | t);
            break;
        case 6:
            RRR(ge, t, s, d);
            E(0x6E205800u | t << 5 | t);
            break;
    }
    if (kind == 4) {
        if (t != d)
            vmov(d, t);
    } else
        ins_elem(d, 0, t, 0, kind == 2 ? 4 : 8);
}

static void emit_shift_xmm(const Insn *in, int esize, int kind) { // kind 0 left, 1 logical right, 2 arithmetic right
    int s = src_op(in, 4, true), d = in->reg;
    u32 sz = esize == 16 ? 1 : esize == 32 ? 2 : 3;
    umov_d(S0, s, 0);
    // Counts above the element width behave as the element width.
    movimm(S1, (u64)esize);
    lsr_imm(1, S2, S0, esize == 16 ? 4 : esize == 32 ? 5 : 6);
    cbz(0, 1, S2, 2); // below the width, keep the count
    mov_reg(1, S0, S1);
    if (kind)
        add_reg(1, 1, 0, S0, XZR, S0, 0);
    // dup v24.<T>, x16
    u32 imm5 = esize == 16 ? 2 : esize == 32 ? 4 : 8;
    E(0x4E000C00u | imm5 << 16 | S0 << 5 | V_S0);
    if (kind == 2)
        RRR(0x4E204400u | sz << 22, d, d, V_S0); // sshl
    else
        RRR(0x6E204400u | sz << 22, d, d, V_S0); // ushl
}

static void emit_shift_imm(const Insn *in, int esize, int kind) {
    int d = in->rm, n = (int)(in->imm & 0xff);
    if (!in->rm_reg) {
        emit_stop(in->rip, "shift immediate group with a memory operand");
        return;
    }
    if (kind != 2 && n >= esize) {
        E(0x6E201C00u | d << 16 | d << 5 | d); // eor: zero
        return;
    }
    if (kind == 2 && n >= esize)
        n = esize - 1;
    if (kind == 0) {
        u32 immhb = (u32)(esize + n);
        E(0x4F005400u | immhb << 16 | d << 5 | d);
    } else {
        u32 immhb = (u32)(2 * esize - n);
        if (n == 0)
            return; // an ARM right shift of zero does not exist; x86 leaves the value
        E((kind == 1 ? 0x6F000400u : 0x4F000400u) | immhb << 16 | d << 5 | d);
    }
}

static void pack(const Insn *in, u32 base, u32 base2) {
    int s = src_op(in, 4, true), d = in->reg;
    RR(base, V_S0, d);
    RR(base2, V_S0, s);
    vmov(d, V_S0);
}

static void emit_lanes32(int d, int src1, int src2, int imm) {
    imm &= 0xff;
    if (src1 == src2) {
        switch (imm) {
            case 0xE4: vmov(d, src1); return;
            case 0x00: case 0x55: case 0xAA: case 0xFF: {
                int k = imm & 3;
                E(0x4E040400u | (u32)k << 19 | (u32)src1 << 5 | (u32)d); // dup v.4s, vn.s[k]
                return;
            }
            case 0x44: E(0x4E080400u | (u32)src1 << 5 | (u32)d); return; // dup v.2d, vn.d[0]
            case 0xEE: E(0x4E180400u | (u32)src1 << 5 | (u32)d); return; // dup v.2d, vn.d[1]
            case 0xB1: RR(0x4EA00800u, d, src1); return; // rev64 v.4s, vn.4s
            case 0x4E: E(0x6E004000u | (u32)src1 << 16 | (u32)src1 << 5 | (u32)d); return; // ext #8
            case 0x39: E(0x6E002000u | (u32)src1 << 16 | (u32)src1 << 5 | (u32)d); return; // ext #4
            case 0x93: E(0x6E006000u | (u32)src1 << 16 | (u32)src1 << 5 | (u32)d); return; // ext #12
            case 0x50: RRR(0x4E803800u, d, src1, src1); return; // zip1 .4s
            case 0xFA: RRR(0x4E807800u, d, src1, src1); return; // zip2 .4s
            case 0x88: RRR(0x4E801800u, d, src1, src1); return; // uzp1 .4s
            case 0xDD: RRR(0x4E805800u, d, src1, src1); return; // uzp2 .4s
            case 0xA0: RRR(0x4E802800u, d, src1, src1); return; // trn1 .4s
            case 0xF5: RRR(0x4E806800u, d, src1, src1); return; // trn2 .4s
            case 0x1B:
                RR(0x4EA00800u, V_S0, src1);
                E(0x6E004000u | (u32)V_S0 << 16 | (u32)V_S0 << 5 | (u32)d);
                return;
            case 0x11:
                RR(0x4EA00800u, V_S0, src1);
                RRR(0x4EC03800u, d, V_S0, V_S0);
                return;
            case 0xBB:
                RR(0x4EA00800u, V_S0, src1);
                RRR(0x4EC07800u, d, V_S0, V_S0);
                return;
            case 0x33:
                E(0x6E002000u | (u32)src1 << 16 | (u32)src1 << 5 | V_S0);
                RRR(0x4EC07800u, d, V_S0, V_S0);
                return;
            case 0x22:
                E(0x6E004000u | (u32)src1 << 16 | (u32)src1 << 5 | V_S0); // ext #8 -> [2,3,0,1]
                RRR(0x4E801800u, d, V_S0, V_S0); // uzp1 .4s -> [2,0,2,0]
                return;
            case 0x77:
                E(0x6E004000u | (u32)src1 << 16 | (u32)src1 << 5 | V_S0); // ext #8 -> [2,3,0,1]
                RRR(0x4E805800u, d, V_S0, V_S0); // uzp2 .4s -> [3,1,3,1]
                return;
            case 0x99:
                E(0x6E002000u | (u32)src1 << 16 | (u32)src1 << 5 | V_S0); // ext #4 -> [1,2,3,0]
                RRR(0x4EC03800u, d, V_S0, V_S0); // zip1 .2d -> [1,2,1,2]
                return;
            default:
                break;
        }
    }
    if (imm == 0x44) {
        RRR(0x4EC03800u, d, src1, src2); // zip1 v.2d, vsrc1.2d, vsrc2.2d
        return;
    }
    if (imm == 0xEE) {
        RRR(0x4EC07800u, d, src1, src2); // zip2 v.2d, vsrc1.2d, vsrc2.2d
        return;
    }
    if (imm == 0x4E) {
        E(0x6E004000u | (u32)src2 << 16 | (u32)src1 << 5 | (u32)d); // ext .16b, src1, src2, #8
        return;
    }
    if (imm == 0x88) {
        RRR(0x4E801800u, d, src1, src2); // uzp1 v.4s, vsrc1.4s, vsrc2.4s
        return;
    }
    if (imm == 0xDD) {
        RRR(0x4E805800u, d, src1, src2); // uzp2 v.4s, vsrc1.4s, vsrc2.4s
        return;
    }
    if (imm == 0xE4) {
        int dst = (d == src2 && d != src1) ? V_S0 : d;
        vmov(dst, src1);
        ins_elem(dst, 1, src2, 1, 8);
        if (dst != d)
            vmov(d, dst);
        return;
    }
    int tmp = (d != src1 && d != src2) ? d : V_S0;
    if ((imm & 0xf) == 0x4) {
        ins_elem(tmp, 0, src1, 0, 8);
    } else if ((imm & 0xf) == 0xe) {
        ins_elem(tmp, 0, src1, 1, 8);
    } else {
        ins_elem(tmp, 0, src1, imm & 3, 4);
        ins_elem(tmp, 1, src1, (imm >> 2) & 3, 4);
    }
    if ((imm >> 4) == 0x4) {
        ins_elem(tmp, 1, src2, 0, 8);
    } else if ((imm >> 4) == 0xe) {
        ins_elem(tmp, 1, src2, 1, 8);
    } else {
        ins_elem(tmp, 2, src2, (imm >> 4) & 3, 4);
        ins_elem(tmp, 3, src2, (imm >> 6) & 3, 4);
    }
    if (tmp != d)
        vmov(d, tmp);
}

static void lanes32(const Insn *in, int src1, int src2) {
    emit_lanes32(in->reg, src1, src2, (int)in->imm);
}

// pmovmskb: 6-insn zero-constant NEON USHR+USRA bit-pack, single UMOV, no NZCV.
static void emit_pmovmskb(int rd, int s) {
    RR(0x6F090400u, V_S0, s);      // ushr v24.16b, vs.16b, #7
    RR(0x6F191400u, V_S0, V_S0);   // usra v24.8h, v24.8h, #7
    RR(0x6F321400u, V_S0, V_S0);   // usra v24.4s, v24.4s, #14
    RR(0x6F641400u, V_S0, V_S0);   // usra v24.2d, v24.2d, #28
    ins_elem(V_S0, 1, V_S0, 8, 1); // ins v24.b[1], v24.b[8]
    E(0x0E023C00u | (u32)V_S0 << 5 | (u32)rd); // umov w_rd, v24.h[0]
}

// SSSE3: opcode from the 0F 38 or 0F 3A map, 66 prefix checked by the decoder.
static void emit_ssse3(const Insn *in) {
    int opc = in->opc, d = in->reg;
    int s = src_op(in, 4, true);
    if (in->sub == 2) { // palignr
        int n = (int)(in->imm & 0xff);
        if (n >= 32) {
            E(0x6E201C00u | d << 16 | d << 5 | d);
        } else if (n >= 16) {
            E(0x6E201C00u | V_S0 << 16 | V_S0 << 5 | V_S0);
            if (n > 16)
                E(0x6E000000u | V_S0 << 16 | (u32)(n - 16) << 11 | d << 5 | d);
        } else if (n == 0) {
            vmov(d, s);
        } else {
            E(0x6E000000u | d << 16 | (u32)n << 11 | s << 5 | d);
        }
        return;
    }
    u32 sz = opc <= 0x0B && opc != 0 ? 0 : 0;
    switch (opc) {
        case 0x00: // pshufb: bit 7 clears the byte, otherwise the low nibble picks one. tbl reads 0 for an index out of range.
            E(0x4F04E400u | 15 << 5 | V_S0); // movi v24.16b, #0x8f
            E(0x4E201C00u | V_S0 << 16 | s << 5 | V_S0);
            E(0x4E000000u | V_S0 << 16 | d << 5 | d);
            return;
        case 0x01: // phaddw
        case 0x02: // phaddd
            sz = opc == 1 ? 1 : 2;
            RRR(0x4E20BC00u | sz << 22, d, d, s);
            return;
        case 0x03: // phaddsw
        case 0x05: // phsubw
        case 0x06: // phsubd
        case 0x07: // phsubsw
            sz = opc == 6 ? 2 : 1;
            RRR(0x4E001800u | sz << 22, V_S0, d, s); // uzp1
            RRR(0x4E005800u | sz << 22, V_S1, d, s); // uzp2
            if (opc == 3)
                RRR(0x4E200C00u | sz << 22, d, V_S0, V_S1); // sqadd
            else if (opc == 7)
                RRR(0x4E202C00u | sz << 22, d, V_S0, V_S1); // sqsub
            else
                RRR(0x6E208400u | sz << 22, d, V_S0, V_S1); // sub
            return;
        case 0x04: // pmaddubsw: unsigned bytes of d by signed bytes of s, pairs added with saturation
            RR(0x2F08A400u, V_S0, d); // ushll
            RR(0x6F08A400u, V_S1, d); // ushll2
            RR(0x0F08A400u, V_S2, s); // sshll
            RR(0x4F08A400u, 27, s); // sshll2
            RRR(0x4E609C00u, V_S0, V_S0, V_S2); // mul
            RRR(0x4E609C00u, V_S1, V_S1, 27);
            RRR(0x4E401800u, V_S2, V_S0, V_S1); // uzp1
            RRR(0x4E405800u, 27, V_S0, V_S1); // uzp2
            RRR(0x4E600C00u, d, V_S2, 27); // sqadd
            return;
        case 0x08: // psignb
        case 0x09: // psignw
        case 0x0A: // psignd
            sz = (u32)(opc - 8);
            RR(0x6E20B800u | sz << 22, V_S0, d); // neg
            RR(0x4E20A800u | sz << 22, V_S1, s); // cmlt #0
            RRR(0x6EA01C00u, d, V_S0, V_S1); // bit: take the negated lane where s is negative
            RR(0x4E209800u | sz << 22, V_S1, s); // cmeq #0
            RRR(0x4E601C00u, d, d, V_S1); // bic: clear where s is zero
            return;
        case 0x0B: // pmulhrsw: as hardware, -32768 squared wraps to 0x8000 where sqrdmulh would saturate
            RRR(0x0E60C000u, V_S0, d, s); // smull
            RRR(0x4E60C000u, V_S1, d, s); // smull2
            E(0x4F002400u | 49u << 16 | V_S0 << 5 | V_S0); // srshr .4s, #15
            E(0x4F002400u | 49u << 16 | V_S1 << 5 | V_S1);
            RRR(0x4E401800u, d, V_S0, V_S1); // uzp1 .8h: the low half of each
            return;
        case 0x1C: // pabsb
        case 0x1D: // pabsw
        case 0x1E: // pabsd
            sz = (u32)(opc - 0x1C);
            RR(0x4E20B800u | sz << 22, d, s);
            return;
        default:
            svc(SVC_UD);
    }
}

// addsubps and addsubpd: 3-4 insn pure NEON fast path when !tr_precise_flags; lane-by-lane under precise flags.
static void emit_addsub(const Insn *in) {
    bool isd = in->pfx == 0x66;
    int s = src_op(in, 4, true), d = in->reg;
    if (!tr_precise_flags) {
        if (isd) {
            RRR(0x4EE0D400u, V_S0, d, s); // fsub .2d
            RRR(0x4E60D400u, V_S1, d, s); // fadd .2d
            ins_elem(V_S0, 1, V_S1, 1, 8);
            vmov(d, V_S0);
        } else {
            RRR(0x4EA0D400u, V_S0, d, s); // fsub .4s
            RRR(0x4E20D400u, V_S1, d, s); // fadd .4s
            RR(0x4EA00800u, V_S0, V_S0);  // rev64 .4s -> [sub1, sub0, sub3, sub2]
            RRR(0x4E806800u, d, V_S0, V_S1); // trn2 .4s -> [sub0, add1, sub2, add3]
        }
        return;
    }
    int bytes = isd ? 8 : 4, lanes = 16 / bytes;
    u32 ftype = isd ? 1u << 22 : 0;
    for (int i = 0; i < lanes; i++) {
        ins_elem(V_S0, 0, d, i, bytes);
        ins_elem(V_S1, 0, s, i, bytes);
        RRR((i & 1 ? 0x1E202800u : 0x1E203800u) | ftype, V_S2, V_S0, V_S1);
        ins_elem(27, i, V_S2, 0, bytes);
    }
    vmov(d, 27);
}

static void fmov_to_d(int v, int x) { RR(0x9E670000u, v, x); }
static void ins_x(int v, int lane, int x) { E(0x4E081C00u | (u32)lane << 20 | (u32)x << 5 | v); } // ins v.d[lane], x

static void pcmp_mask_from_cmp(int rd_gpr, int cmp_vec, int v_mask, int vt, bool words) {
    RRR(0x4E201C00u, vt, cmp_vec, v_mask); // and vt.16b, cmp_vec.16b, v_mask.16b
    if (words) {
        RR(0x4E71B800u, vt, vt); // addv h_vt, vt.8h
    } else {
        RRR(0x4E20BC00u, vt, vt, vt); // addp vt.16b, vt.16b, vt.16b
        RRR(0x4E20BC00u, vt, vt, vt); // addp vt.16b, vt.16b, vt.16b
        RRR(0x0E20BC00u, vt, vt, vt); // addp vt.8b, vt.8b, vt.8b
    }
    E(0x0E023C00u | (u32)vt << 5 | (u32)rd_gpr); // umov w_rd, vt.h[0]
}

// Inline EL0 PCMPISTRI / VPCMPISTRI for hot aggregation modes:
// - agg == 2 (Equal Each, 8-bit & 16-bit, all polarities, LSB & MSB) -> strlen, wcslen, strcmp, strncmp
// - agg == 0 (Equal Any, 8-bit, all polarities, LSB & MSB) -> strchr, strpbrk, strcspn, strspn
// - agg == 1 (Ranges, 8-bit unsigned, all polarities, LSB & MSB) -> lexer/tokenizer ranges
// Falls back to SVC_H_PCMPISTRI for rare modes (agg == 3 Equal Ordered, 16-bit Equal Any/Ranges, signed Ranges).
void emit_pcmpistri_el0(int d, int s, u8 imm, int live) {
    bool words = (imm & 1) != 0;
    bool is_signed = (imm & 2) != 0;
    int agg = (imm >> 2) & 3;
    int pol = (imm >> 4) & 3;
    bool msb = (imm & 0x40) != 0;
    bool supported = (agg == 2) || (agg == 0 && !words) || (agg == 1 && !words && !is_signed);
    if (!supported) {
        umov_d(S0, d, 0);
        umov_d(S1, d, 1);
        umov_d(S2, s, 0);
        umov_d(S3, s, 1);
        movimm(S4, (u64)imm);
        svc(SVC_H_PCMPISTRI);
        mov_reg(0, 1, S0);
        msr_nzcv(S2);
        mov_reg(1, R_PF, S3);
        mov_reg(1, R_AF, S4);
        return;
    }

    int n = words ? 8 : 16;
    u32 all = (1u << n) - 1;
    u32 sz = words ? 1u : 0u;
    enum { VT = 27, VM = 28 };
    if (words) {
        movimm(S0, 0x0008000400020001ULL);
        fmov_to_d(VM, S0);
        movimm(S0, 0x0080004000200010ULL);
        ins_x(VM, 1, S0);
    } else {
        movimm(S0, 0x8040201008040201ULL);
        fmov_to_d(VM, S0);
        ins_x(VM, 1, S0);
    }

    // S0 = za (zero mask of a), S1 = zb (zero mask of b)
    RR(0x4E209800u | sz << 22, VT, d); // cmeq #0
    pcmp_mask_from_cmp(S0, VT, VM, VT, words);
    RR(0x4E209800u | sz << 22, VT, s); // cmeq #0
    pcmp_mask_from_cmp(S1, VT, VM, VT, words);

    // S3 = valid_a = (za - 1) & ~za & all, S4 = valid_b = (zb - 1) & ~zb & all
    add_imm(0, 1, 0, S3, S0, 1);
    logic_reg(0, 0, 1, S3, S3, S0);
    logic_imm(0, 0, S3, S3, all);
    add_imm(0, 1, 0, S4, S1, 1);
    logic_reg(0, 0, 1, S4, S4, S1);
    logic_imm(0, 0, S4, S4, all);

    // Compute raw match bitmask in S2.
    if (agg == 2) {
        RRR(0x6E208C00u | sz << 22, VT, d, s); // cmeq d, s
        pcmp_mask_from_cmp(S2, VT, VM, VT, words);
        logic_reg(0, 0, 0, S5, S3, S4); // valid_a & valid_b
        logic_reg(0, 0, 0, S2, S2, S5); // eq & valid_a & valid_b
        movimm(S5, all);
        logic_reg(0, 0, 1, S5, S5, S3); // all & ~valid_a
        logic_reg(0, 0, 1, S5, S5, S4); // all & ~valid_a & ~valid_b
        logic_reg(0, 1, 0, S2, S2, S5); // res
    } else if (agg == 0) {
        vmov(V_S0, d);
        E(0x6E201C00u | V_S1 << 16 | V_S1 << 5 | V_S1); // eor V_S1, V_S1, V_S1
        u32 *skip_loop = tr_cc;
        cbz(0, 0, S3, 0);
        u32 *loop = tr_cc;
        E(0x4E010400u | (u32)V_S0 << 5 | V_S2);          // dup V_S2.16b, V_S0.b[0]
        RRR(0x6E208C00u, V_S2, V_S2, s);                 // cmeq V_S2.16b, V_S2.16b, s.16b
        RRR(0x4EA01C00u, V_S1, V_S1, V_S2);              // orr V_S1.16b, V_S1.16b, V_S2.16b
        E(0x6E000800u | (u32)V_S0 << 16 | (u32)V_S0 << 5 | V_S0); // ext V_S0, V_S0, V_S0, #1
        lsr_imm(0, S3, S3, 1);
        cbz(1, 0, S3, (int)(loop - tr_cc));
        patch_cb(skip_loop, tr_cc);
        pcmp_mask_from_cmp(S2, V_S1, VM, VT, false);
        logic_reg(0, 0, 0, S2, S2, S4); // res &= valid_b
    } else {
        // agg == 1, unsigned 8-bit Ranges
        vmov(V_S0, d);
        E(0x6E201C00u | V_S1 << 16 | V_S1 << 5 | V_S1);
        lsr_imm(0, S5, S3, 1);
        u32 *skip_loop = tr_cc;
        cbz(0, 0, S5, 0);
        u32 *loop = tr_cc;
        E(0x4E010400u | (u32)V_S0 << 5 | V_S2);          // dup V_S2.16b, V_S0.b[0] (lo)
        E(0x4E030400u | (u32)V_S0 << 5 | VT);            // dup VT.16b, V_S0.b[1] (hi)
        RRR(0x6E203C00u, V_S2, s, V_S2);                 // cmhs V_S2.16b, s.16b, lo.16b (s >= lo)
        RRR(0x6E203C00u, VT, VT, s);                     // cmhs VT.16b, hi.16b, s.16b (hi >= s)
        RRR(0x4E201C00u, V_S2, V_S2, VT);                // and V_S2, V_S2, VT
        RRR(0x4EA01C00u, V_S1, V_S1, V_S2);              // orr V_S1, V_S1, V_S2
        E(0x6E001000u | (u32)V_S0 << 16 | (u32)V_S0 << 5 | V_S0); // ext V_S0, V_S0, V_S0, #2
        lsr_imm(0, S3, S3, 2);
        lsr_imm(0, S5, S3, 1);
        cbz(1, 0, S5, (int)(loop - tr_cc));
        patch_cb(skip_loop, tr_cc);
        pcmp_mask_from_cmp(S2, V_S1, VM, VT, false);
        logic_reg(0, 0, 0, S2, S2, S4); // res &= valid_b
    }

    // Apply polarity (1: invert all, 3: invert valid_b only).
    if (pol == 1) {
        movimm(S5, all);
        logic_reg(0, 2, 0, S2, S2, S5);
    } else if (pol == 3) {
        logic_reg(0, 2, 0, S2, S2, S4);
    }

    // Write result index into ECX (zero-extended to RCX).
    if (!msb) {
        logic_imm(0, 1, S5, S2, 1ULL << n);
        rbit(0, S5, S5);
        clz(0, 1, S5);
    } else {
        clz(0, S5, S2);
        movimm(S3, 31);
        add_reg(0, 1, 0, S5, S3, S5, 0);
        add_imm(0, 1, 1, XZR, S2, 0);
        movimm(S3, (u64)n);
        csel(0, 1, S3, S5, 0);
    }

    // Flags: SF=(za!=0), ZF=(zb!=0), CF=(res!=0) [Arm64 C=(res==0)], OF=(res&1), PF=0, AF=0.
    if (live) {
        add_imm(0, 1, 1, XZR, S0, 0);
        cset(S5, 1); // NE: za != 0 -> N
        lsl_imm(1, S5, S5, 31);
        add_imm(0, 1, 1, XZR, S1, 0);
        cset(S3, 1); // NE: zb != 0 -> Z
        logic_sh(1, 1, 0, S5, S5, S3, 0, 30);
        add_imm(0, 1, 1, XZR, S2, 0);
        cset(S3, 0); // EQ: res == 0 -> C (inverted x86 CF)
        logic_sh(1, 1, 0, S5, S5, S3, 0, 29);
        ubfx(1, S3, S2, 0, 1);
        logic_sh(1, 1, 0, S5, S5, S3, 0, 28);
        msr_nzcv(S5);
        if (live & (F_P | F_A))
            movimm(R_PF, 1);
        if (live & F_A)
            mov_reg(1, R_AF, R_PF);
    }
}

// SSE4.1, AES and PCLMUL, the forms libraries and kexts reach for, each with an exact NEON or crypto counterpart.
static void emit_sse41(const Insn *in, int live) {
    int opc = in->opc, d = in->reg, imm = (int)(in->imm & 0xff);
    if (in->sub == 5) {
        int s = src_op(in, 4, false);
        emit_pcmpistri_el0(d, s, (u8)imm, live);
        return;
    }
    if (in->sub == 4) {
        switch (opc) {
            case 0x08:
            case 0x09:
            case 0x0A:
            case 0x0B: {
                // Bit 2 takes the mode from MXCSR, which FPCR follows. Bit 3 suppresses the precision flag. NEON's fixed-mode rounds never set it, so
                // to report it an frintx runs first, its result unused: whether a value is whole does not depend on the mode.
                static const u32 fixed[4] = {0, 2, 1, 3}; // x86 nearest, down, up, zero; ARM rmode N, M is 2, P is 1, Z
                u32 rmode = (imm & 4) ? ((imm & 8) ? 7 : 6) : fixed[imm & 3];
                bool dbl = opc & 1, report = !(imm & 4) && !(imm & 8);
                if (opc >= 0x0A) {
                    int s = src_op(in, dbl ? 3 : 2, false);
                    if (report)
                        E(0x1E244000u | (dbl ? 1u << 22 : 0) | 6u << 15 | (u32)s << 5 | V_S1);
                    E(0x1E244000u | (dbl ? 1u << 22 : 0) | rmode << 15 | (u32)s << 5 | V_S0);
                    ins_elem(d, 0, V_S0, 0, dbl ? 8 : 4);
                } else {
                    int s = src_op(in, 4, true);
                    static const u32 packed[8] = {0x4E218800u, 0x4EA18800u, 0x4E219800u, 0x4EA19800u, 0, 0, 0x6E219800u, 0x6EA19800u};
                    if (report)
                        RR(packed[6] | (dbl ? 1u << 22 : 0), V_S1, s);
                    RR(packed[rmode] | (dbl ? 1u << 22 : 0), d, s);
                }
                return;
            }
            case 0x0C:
            case 0x0D:
            case 0x0E: {
                int s = src_op(in, 4, true), bytes = opc == 0x0C ? 4 : opc == 0x0D ? 8 : 2;
                int lanes = 16 / bytes;
                int mask = imm & ((1 << lanes) - 1);
                if (mask == 0)
                    return;
                if (mask == (1 << lanes) - 1) {
                    vmov(d, s);
                    return;
                }
                if (bytes == 4) {
                    if (mask == 0x3) { ins_elem(d, 0, s, 0, 8); return; }
                    if (mask == 0xC) { ins_elem(d, 1, s, 1, 8); return; }
                    if (mask == 0x5) { RR(0x4EA00800u, V_S0, s); RRR(0x4E806800u, d, V_S0, d); return; }
                    if (mask == 0xA) { RR(0x4EA00800u, V_S0, d); RRR(0x4E806800u, d, V_S0, s); return; }
                } else if (bytes == 2) {
                    if (mask == 0x0F) { ins_elem(d, 0, s, 0, 8); return; }
                    if (mask == 0xF0) { ins_elem(d, 1, s, 1, 8); return; }
                    if (mask == 0x55) { RR(0x6E600800u, V_S0, s); RRR(0x4E406800u, d, V_S0, d); return; }
                    if (mask == 0xAA) { RR(0x6E600800u, V_S0, d); RRR(0x4E406800u, d, V_S0, s); return; }
                    if (mask == 0x33) { RR(0x4EA00800u, V_S0, s); RRR(0x4E806800u, d, V_S0, d); return; }
                    if (mask == 0xCC) { RR(0x4EA00800u, V_S0, d); RRR(0x4E806800u, d, V_S0, s); return; }
                    // Coalesce adjacent matching 16-bit lanes into 32-bit INS when possible.
                    if (((mask & 0x55) << 1) == (mask & 0xAA)) {
                        for (int i = 0; i < 4; i++)
                            if ((mask >> (2 * i)) & 1)
                                ins_elem(d, i, s, i, 4);
                        return;
                    }
                }
                for (int i = 0; i < lanes; i++)
                    if ((mask >> i) & 1)
                        ins_elem(d, i, s, i, bytes);
                return;
            }
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17: {
                int bytes = opc == 0x14 ? 1 : opc == 0x15 ? 2 : (opc == 0x16 && in->sz == 8) ? 8 : 4, lane = imm & (16 / bytes - 1);
                u32 umov = (bytes == 8 ? 0x4E003C00u : 0x0E003C00u) | (u32)(lane << (log2sz(bytes) + 1) | bytes) << 16 | (u32)d << 5;
                if (in->rm_reg) {
                    E(umov | (u32)in->rm);
                } else {
                    Addr a = addr_prepare(in, bytes);
                    E(umov | S0);
                    ldst(0, bytes, S0, a);
                }
                return;
            }
            case 0x20:
            case 0x22: {
                int bytes = opc == 0x20 ? 1 : in->sz == 8 ? 8 : 4, lane = imm & (16 / bytes - 1), s = in->rm;
                if (!in->rm_reg) {
                    ldst(1, bytes, S0, addr_prepare(in, bytes));
                    s = S0;
                }
                E(0x4E001C00u | (u32)(lane << (log2sz(bytes) + 1) | bytes) << 16 | (u32)s << 5 | d);
                return;
            }
            case 0x40: // dpps
            case 0x41: { // dppd. Same product-mask / pairwise-add / broadcast as FEX VDPPSOpImpl.
                bool dbl = opc == 0x41;
                int s = src_op(in, 4, true);
                unsigned srcm = dbl ? ((unsigned)imm >> 4) & 3u : ((unsigned)imm >> 4) & 0xfu;
                unsigned dstm = dbl ? (unsigned)imm & 3u : (unsigned)imm & 0xfu;
                unsigned full = dbl ? 3u : 0xfu;
                if (srcm == 0 || dstm == 0) {
                    E(0x6E201C00u | d << 16 | d << 5 | d);
                    return;
                }
                RRR(dbl ? 0x6E60DC00u : 0x6E20DC00u, V_S0, d, s);
                if (srcm != full) {
                    if (dbl) {
                        if (srcm == 1)
                            RR(0x1E604000u, V_S0, V_S0); // fmov d, d: clear lane 1
                        else
                            E(0x4E081C00u | (u32)XZR << 5 | V_S0); // ins v.d[0], xzr
                    } else if (srcm == 0x3) {
                        RR(0x1E604000u, V_S0, V_S0); // fmov d, d: clear lanes 2,3
                    } else if (srcm == 0xC) {
                        E(0x4E081C00u | (u32)XZR << 5 | V_S0); // ins v.d[0], xzr: clear lanes 0,1
                    } else {
                        for (int i = 0; i < 4; i++)
                            if (!(srcm & (1u << i)))
                                E(0x4E041C00u | (u32)i << 19 | (u32)XZR << 5 | V_S0); // ins v.s[i], wzr
                    }
                }
                E(0x6E201C00u | V_S1 << 16 | V_S1 << 5 | V_S1);
                RRR(dbl ? 0x6E60D400u : 0x6E20D400u, V_S0, V_S0, V_S1);
                if (!dbl)
                    RRR(0x6E20D400u, V_S0, V_S0, V_S1);
                if (dstm == full) {
                    if (dbl)
                        E(0x4E080400u | (u32)V_S0 << 5 | d); // dup.2d
                    else
                        E(0x4E040400u | (u32)V_S0 << 5 | d); // dup.4s
                    return;
                }
                E(0x6E201C00u | V_S1 << 16 | V_S1 << 5 | V_S1);
                int lanes = dbl ? 2 : 4, es = dbl ? 8 : 4;
                for (int i = 0; i < lanes; i++)
                    if (dstm & (1u << i))
                        ins_elem(V_S1, i, V_S0, 0, es);
                vmov(d, V_S1);
                return;
            }
            case 0x21: { // insertps: count_s, count_d, zero mask
                int s = src_op(in, 2, false);
                ins_elem(V_S0, 0, s, in->rm_reg ? (imm >> 6) & 3 : 0, 4);
                ins_elem(d, (imm >> 4) & 3, V_S0, 0, 4);
                for (int i = 0; i < 4; i++)
                    if ((imm >> i) & 1)
                        E(0x4E041C00u | (u32)i << 19 | (u32)XZR << 5 | d);
                return;
            }
            case 0x44: { // pclmulqdq
                int s = src_op(in, 4, true);
                ins_elem(V_S0, 0, d, imm & 1, 8);
                ins_elem(V_S1, 0, s, (imm >> 4) & 1, 8);
                RRR(0x0EE0E000u, d, V_S0, V_S1);
                return;
            }
            case 0xDF: { // aeskeygenassist: aese does ShiftRows and SubBytes, and the table puts each byte back where the x86 result wants it
                int s = src_op(in, 4, true);
                E(0x6E201C00u | V_S0 << 16 | V_S0 << 5 | V_S0);
                vmov(V_S1, s);
                RR(0x4E284800u, V_S1, V_S0);
                movimm(S0, 0x040B0E010B0E0104ULL);
                fmov_to_d(V_S2, S0);
                movimm(S0, 0x0C0306090306090CULL);
                ins_x(V_S2, 1, S0);
                RRR(0x4E000000u, d, V_S1, V_S2); // tbl
                movimm(S0, (u64)imm << 32);
                fmov_to_d(V_S0, S0);
                ins_x(V_S0, 1, S0);
                RRR(0x6E201C00u, d, d, V_S0);
                return;
            }
        }
        svc(SVC_UD);
        return;
    }
    if ((opc >= 0x20 && opc <= 0x25) || (opc >= 0x30 && opc <= 0x35)) { // pmovsx, pmovzx: widen by one step at a time
        static const u8 kind[6] = {3, 2, 1, 3, 2, 3}, from[6] = {0, 0, 0, 1, 1, 2}, to[6] = {1, 2, 3, 2, 3, 3};
        int k = opc & 7, s = src_op(in, kind[k], false);
        u32 base = opc < 0x30 ? 0x0F00A400u : 0x2F00A400u;
        vmov(V_S0, s);
        for (int w = from[k]; w < to[k]; w++)
            RR(base | 1u << (19 + w), V_S0, V_S0);
        vmov(d, V_S0);
        return;
    }
    if (opc == 0x2A) { // movntdqa
        if (in->rm_reg)
            svc(SVC_UD);
        else
            vldst(4, true, d, mem16(in, true));
        return;
    }
    int s = src_op(in, 4, true);
    switch (opc) {
        case 0x10:
        case 0x14:
        case 0x15:
            RR(opc == 0x10 ? 0x4E20A800u : opc == 0x14 ? 0x4EA0A800u : 0x4EE0A800u, V_S0, 0); // cmlt #0 of xmm0: each lane's top bit picks
            RRR(0x6EA01C00u, d, s, V_S0); // bit
            return;
        case 0x17: { // ptest: ZF from d AND s, CF from NOT d AND s, the rest clear
            if (!live)
                return;
            if (live == F_Z) {
                int av = d;
                if (d != s) {
                    RRR(0x4E201C00u, V_S0, d, s);
                    av = V_S0;
                }
                RRR(0x6EA0A400u, V_S0, av, av); // umaxp .4s
                RR(0x9E660000u, S0, V_S0);      // fmov xS0, d24
                add_imm(1, 1, 1, XZR, S0, 0);   // cmp xS0, #0 -> Z = (S0 == 0)
                return;
            }
            if (live == F_C) {
                if (d == s) {
                    rmif(XZR, 0, 2); // ~d & d == 0 -> x86 CF=1 -> Arm64 C=0
                    return;
                }
                RRR(0x4E601C00u, V_S1, s, d);   // bic
                RRR(0x6EA0A400u, V_S1, V_S1, V_S1); // umaxp .4s
                RR(0x9E660000u, S1, V_S1);      // fmov xS1, d25
                add_imm(1, 1, 1, XZR, S1, 1);   // cmp xS1, #1 -> C = (S1 != 0) = !CF
                return;
            }
            if (d == s) {
                RRR(0x6EA0A400u, V_S0, d, d);   // umaxp .4s
                RR(0x9E660000u, S0, V_S0);      // fmov xS0, d24
                add_imm(1, 1, 1, XZR, S0, 0);
                cset(S0, 0);
                lsl_imm(1, S0, S0, 30);         // Z=(d==0), C=0(CF=1), N=0, V=0
                E(0xD51B4200u | S0);            // msr nzcv
            } else {
                RRR(0x4E201C00u, V_S0, d, s);
                RRR(0x4E601C00u, V_S1, s, d);   // bic
                RRR(0x6EA0A400u, V_S0, V_S0, V_S1); // umaxp .4s: d[0]=any(V_S0), d[1]=any(V_S1)
                RR(0x9E660000u, S0, V_S0);      // fmov xS0, d24
                umov_d(S1, V_S0, 1);
                add_imm(1, 1, 1, XZR, S0, 0);
                cset(S0, 0);
                add_imm(1, 1, 1, XZR, S1, 0);
                cset(S1, 1); // the engine keeps x86 CF inverted in C
                lsl_imm(1, S0, S0, 30);
                logic_sh(1, 1, 0, S0, S0, S1, 0, 29);
                E(0xD51B4200u | S0); // msr nzcv
            }
            if (live & (F_P | F_A)) {
                movimm(R_PF, 1); // an odd low byte: PF clear
                mov_reg(1, R_AF, R_PF);
            }
            return;
        }
        case 0x28: // pmuldq: the low half of each 64-bit lane, signed
            RR(0x0EA12800u, V_S0, d);
            RR(0x0EA12800u, V_S1, s);
            RRR(0x0EA0C000u, d, V_S0, V_S1);
            return;
        case 0x29:
            RRR(0x6EE08C00u, d, d, s);
            return;
        case 0x2B: // packusdw
            RR(0x2E612800u, V_S0, d);
            RR(0x6E612800u, V_S0, s);
            vmov(d, V_S0);
            return;
        case 0x37:
            RRR(0x4EE03400u, d, d, s);
            return;
        case 0x38:
            RRR(0x4E206C00u, d, d, s);
            return;
        case 0x39:
            RRR(0x4EA06C00u, d, d, s);
            return;
        case 0x3A:
            RRR(0x6E606C00u, d, d, s);
            return;
        case 0x3B:
            RRR(0x6EA06C00u, d, d, s);
            return;
        case 0x3C:
            RRR(0x4E206400u, d, d, s);
            return;
        case 0x3D:
            RRR(0x4EA06400u, d, d, s);
            return;
        case 0x3E:
            RRR(0x6E606400u, d, d, s);
            return;
        case 0x3F:
            RRR(0x6EA06400u, d, d, s);
            return;
        case 0x40:
            RRR(0x4EA09C00u, d, d, s);
            return;
        case 0xDB:
            RR(0x4E287800u, d, s); // aesimc
            return;
        case 0xDC:
        case 0xDD:
        case 0xDE:
        case 0xDF: { // x86 xors round key last, ARM first: fuse deferred register round key into next AESE/AESD (2 insns/round)
            bool dec = opc >= 0xDE, last = opc & 1;
            if (in->rm_reg && s != d) {
                sse_aes_flush_one(s);
                for (int r = 0; r < 16; r++)
                    if (r != d && (tr_aes_mask & (1u << r)) && tr_aes_key[r] == d)
                        sse_aes_flush_one(r);
                if (tr_aes_mask & (1u << d)) {
                    int kprev = tr_aes_key[d];
                    tr_aes_mask &= (u16)~(1u << d);
                    RR(dec ? 0x4E285800u : 0x4E284800u, d, kprev);
                } else {
                    E(0x4F00E400u | V_S0); // movi v24.16b, #0
                    RR(dec ? 0x4E285800u : 0x4E284800u, d, V_S0);
                }
                if (!last) {
                    RR(dec ? 0x4E287800u : 0x4E286800u, d, d);
                    tr_aes_key[d] = (u8)s;
                    tr_aes_mask |= (u16)(1u << d);
                } else {
                    RRR(0x6E201C00u, d, d, s);
                }
                return;
            }
            for (int r = 0; r < 16; r++)
                if (r != d && (tr_aes_mask & (1u << r)) && tr_aes_key[r] == d)
                    sse_aes_flush_one(r);
            int tmp = (s == d) ? V_S1 : d;
            if (s == d)
                sse_aes_flush_one(d);
            if (tmp != d)
                vmov(tmp, d);
            if (tr_aes_mask & (1u << d)) {
                int kprev = tr_aes_key[d];
                tr_aes_mask &= (u16)~(1u << d);
                RR(dec ? 0x4E285800u : 0x4E284800u, tmp, kprev);
            } else {
                E(0x4F00E400u | V_S0); // movi v24.16b, #0
                RR(dec ? 0x4E285800u : 0x4E284800u, tmp, V_S0);
            }
            if (!last)
                RR(dec ? 0x4E287800u : 0x4E286800u, tmp, tmp);
            RRR(0x6E201C00u, d, tmp, s);
            return;
        }
    }
    svc(SVC_UD);
}

// MMX: the operands in vectors whose upper halves are zero, the xmm form of the operation on those, and the low half written back. Every
// path touches NEON first, so CR0.TS gives #NM as it does for SSE. Like TCG this raises no #MF for a pending x87 exception.
static void mm_ldst(bool load, int v, int mm) {
    Addr a = {.base = CTX, .imm = (i64)__builtin_offsetof(EngineCtx, mm) + mm * 8};
    vldst(3, load, v, a);
    if (load)
        return;
    // An MMX write leaves the register reading as a NaN to x87 code, whatever it held.
    Addr e = {.base = CTX, .imm = (i64)__builtin_offsetof(EngineCtx, fp_exp) + mm * 2};
    movimm(S1, 0xFFFF);
    ldst(0, 2, S1, e);
}
// TOP cleared and every tag valid; emms empties every tag and leaves TOP.
static void mmx_enter(void) {
    ldst(0, 8, XZR, (Addr){.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_d_valid)});
    Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_top)};
    movimm(S1, 0xFF00);
    ldst(0, 2, S1, a);
}
static void mmx_emms(void) {
    ldst(0, 8, XZR, (Addr){.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_d_valid)});
    Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_tags)};
    ldst(0, 1, XZR, a);
}
void emit_sse(const Insn *in, int live);
static void emit_mmx(const Insn *in, int live) {
    enum { A = 28, B = 29 }; // the operations use v24-v27 and V_MEM
    int opc = in->opc, dm = in->reg & 7, sm = in->rm & 7, write = dm;
    Insn t = *in;
    t.pfx = 0x66;
    switch (opc) {
        case 0x77:
            mm_ldst(true, A, 0);
            mmx_emms();
            return;
        case 0xF7:
            bad(in, "maskmovq (MMX)");
            return;
        case 0x6C:
        case 0x6D:
        case 0xD6:
        case 0xF0:
            svc(SVC_UD);
            return;
        case 0x6E:
            t.reg = A;
            emit_sse(&t, live);
            break;
        case 0x7E:
            mm_ldst(true, A, dm);
            t.reg = A;
            emit_sse(&t, live);
            mmx_enter();
            return;
        case 0x6F:
            if (in->rm_reg)
                mm_ldst(true, A, sm);
            else
                vldst(3, true, A, addr_prepare(in, 8));
            break;
        case 0xE7:
        case 0x7F:
            if (opc == 0xE7 && in->rm_reg) {
                svc(SVC_UD);
                return;
            }
            mm_ldst(true, A, dm);
            if (in->rm_reg) {
                mm_ldst(false, A, sm);
            } else {
                vldst(3, false, A, addr_prepare(in, 8));
            }
            mmx_enter();
            return;
        case 0x71:
        case 0x72:
        case 0x73:
            if (opc == 0x73 && ((in->reg & 7) == 3 || (in->reg & 7) == 7)) { // psrldq and pslldq have no MMX form
                svc(SVC_UD);
                return;
            }
            mm_ldst(true, A, sm);
            t.rm = A;
            emit_sse(&t, live);
            write = sm;
            break;
        case 0xC4:
            mm_ldst(true, A, dm);
            t.reg = A;
            t.imm &= 3;
            emit_sse(&t, live);
            break;
        case 0xC5:
        case 0xD7:
            if (!in->rm_reg) {
                svc(SVC_UD);
                return;
            }
            mm_ldst(true, B, sm);
            t.rm = B;
            t.imm &= 3;
            emit_sse(&t, live); // the destination is a general register
            mmx_enter();
            return;
        default: { // two operands, the second register or memory
            mm_ldst(true, A, dm);
            if (in->rm_reg)
                mm_ldst(true, B, sm);
            else
                vldst(3, true, B, addr_prepare(in, 8));
            t.reg = A;
            t.rm = B;
            t.rm_reg = true;
            if (opc == 0x70) {
                t.pfx = 0xF2; // pshuflw treats the low half as pshufw does
            } else if (opc >= 0x68 && opc <= 0x6A) { // the 64-bit "high half" is what punpckl takes after a shift
                E(0x7F600400u | A << 5 | A);
                E(0x7F600400u | B << 5 | B);
                t.opc = opc - 8;
            } else if (opc == 0x63 || opc == 0x67 || opc == 0x6B) { // the xmm form packs each register apart: pack them as one
                ins_elem(A, 1, B, 0, 8);
            }
            emit_sse(&t, live);
        }
    }
    mm_ldst(false, A, write);
    mmx_enter();
}

void emit_sse(const Insn *in, int live) {
    int opc = in->opc, pfx = in->pfx, d = in->reg;
    if (tr_aes_mask) {
        bool is_aes_round = (in->op == OP_SSE && in->sub == 3 && opc >= 0xDC && opc <= 0xDF);
        if (!is_aes_round) {
            bool is_xmm_load = (!tr_precise_flags && in->op == OP_SSE && in->sub == 0 && !in->rm_reg &&
                                (opc == 0x6F || opc == 0x28 || opc == 0x10));
            if (is_xmm_load) {
                sse_aes_flush_one(d);
                for (int r = 0; r < 16; r++)
                    if ((tr_aes_mask & (1u << r)) && tr_aes_key[r] == d)
                        sse_aes_flush_one(r);
            } else {
                sse_aes_flush_all();
            }
        }
    }
    if (in->op == OP_SSE && in->sub >= 3) {
        emit_sse41(in, live);
        return;
    }
    if (in->op == OP_SSE && in->sub) {
        emit_ssse3(in);
        return;
    }
    if (in->op == OP_LDMXCSR) {
        ldst(1, 4, S0, addr_prepare(in, 4));
        emit_ldmxcsr_s0();
        return;
    }
    if (in->op == OP_STMXCSR) {
        Addr a = addr_prepare(in, 4);
        // The value comes back in x16. The address is ready first, so a fault on the store is precise.
        emit_stmxcsr_s0();
        ldst(0, 4, S0, a);
        return;
    }
    // Integer vector opcodes without a 66 prefix are MMX.
    bool intop = (opc >= 0x60 && opc <= 0x76) || (opc >= 0xD1 && opc <= 0xFE && opc != 0xE6);
    if (pfx == 0 && (intop || opc == 0x77 || opc == 0x7E || opc == 0x7F || opc == 0xC4 || opc == 0xC5)) {
        emit_mmx(in, live);
        return;
    }
    if (intop && pfx != 0x66 && !(opc == 0x6F || opc == 0x70 || opc == 0xD6 || opc == 0xF0)) {
        bad(in, "unsupported prefix on an integer SSE opcode");
        return;
    }
    switch (opc) {
        case 0x10:
        case 0x11: {
            bool load = opc == 0x10;
            int kind = pfx == 0xF3 ? 2 : pfx == 0xF2 ? 3 : 4;
            if (in->rm_reg) {
                int rd = load ? d : in->rm, rn = load ? in->rm : d;
                if (kind == 4)
                    vmov(rd, rn);
                else
                    ins_elem(rd, 0, rn, 0, kind == 2 ? 4 : 8);
            } else if (kind == 4)
                vldst(4, load, d, addr_prepare(in, 16));
            else
                vldst(kind, load, d, addr_prepare(in, 1 << kind));
            break;
        }
        case 0x12:
        case 0x13:
            if (opc == 0x12 && pfx == 0xF3) { // movsldup
                int s = src_op(in, 4, true);
                RRR(0x4E802800u, d, s, s); // trn1 .4s
            } else if (opc == 0x12 && pfx == 0xF2) { // movddup
                int s = src_op(in, 3, false);
                E(0x4E080400u | (u32)s << 5 | d);
            } else if (pfx == 0xF3 || pfx == 0xF2) {
                svc(SVC_UD);
            } else if (in->rm_reg) {
                if (opc == 0x12 && pfx == 0)
                    ins_elem(d, 0, in->rm, 1, 8); // movhlps
                else
                    svc(SVC_UD);
            } else if (opc == 0x12) {
                vldst(3, true, V_MEM, addr_prepare(in, 8));
                ins_elem(d, 0, V_MEM, 0, 8);
            } else
                vldst(3, false, d, addr_prepare(in, 8));
            break;
        case 0x14:
        case 0x15: {
            if (pfx != 0 && pfx != 0x66) {
                svc(SVC_UD);
                break;
            }
            int s = src_op(in, 4, true);
            u32 sz = pfx == 0x66 ? 3 : 2;
            RRR((opc == 0x14 ? 0x4E003800u : 0x4E007800u) | sz << 22, d, d, s);
            break;
        }
        case 0x16:
        case 0x17:
            if (opc == 0x16 && pfx == 0xF3) { // movshdup
                int s = src_op(in, 4, true);
                RRR(0x4E806800u, d, s, s); // trn2 .4s
            } else if (pfx == 0xF3 || pfx == 0xF2) {
                svc(SVC_UD);
            } else if (in->rm_reg) {
                if (opc == 0x16 && pfx == 0)
                    ins_elem(d, 1, in->rm, 0, 8); // movlhps
                else
                    svc(SVC_UD);
            } else if (opc == 0x16) {
                vldst(3, true, V_MEM, addr_prepare(in, 8));
                ins_elem(d, 1, V_MEM, 0, 8);
            } else {
                Addr a = addr_prepare(in, 8);
                ins_elem(V_S0, 0, d, 1, 8);
                vldst(3, false, V_S0, a);
            }
            break;
        case 0x28:
        case 0x29:
            if (pfx != 0 && pfx != 0x66) {
                svc(SVC_UD);
                break;
            }
            if (in->rm_reg) {
                if (opc == 0x28)
                    vmov(d, in->rm);
                else
                    vmov(in->rm, d);
            } else
                vldst(4, opc == 0x28, d, mem16(in, true));
            break;
        case 0x2A: {
            if (pfx != 0xF3 && pfx != 0xF2) {
                bad(in, "cvtpi2ps or cvtpi2pd (MMX)");
                break;
            }
            bool x = in->sz == 8;
            int s = in->rm;
            if (!in->rm_reg) {
                ldst(1, x ? 8 : 4, S0, addr_prepare(in, x ? 8 : 4));
                s = S0;
            }
            bool dbl = pfx == 0xF2;
            scalar_fp_result((x ? 0x9E220000u : 0x1E220000u) | (dbl ? 1u << 22 : 0), d, s, dbl ? 8 : 4);
            break;
        }
        case 0x2B:
            if (in->rm_reg || (pfx != 0 && pfx != 0x66))
                svc(SVC_UD);
            else
                vldst(4, false, d, mem16(in, true));
            break;
        case 0x2C:
        case 0x2D: {
            if (pfx != 0xF3 && pfx != 0xF2) {
                bad(in, "cvtps2pi (MMX)");
                break;
            }
            bool x = in->sz == 8, isd = pfx == 0xF2;
            int s = src_op(in, isd ? 3 : 2, false);
            if (!tr_precise_flags) {
                // Avoid ARMv8.5 FRINT32Z/FRINT32X (which incurs a heavy trap/slowpath under HVF):
                // CVTT* truncates directly via FCVTZS; CVT* rounds per FPCR.RMode via FRINTI then FCVTZS.
                if (opc == 0x2C) {
                    RR((x ? 0x9E380000u : 0x1E380000u) | (isd ? 1u << 22 : 0), d, s);
                } else {
                    RR((isd ? 0x1E67C000u : 0x1E27C000u), V_S0, s); // frinti
                    RR((x ? 0x9E380000u : 0x1E380000u) | (isd ? 1u << 22 : 0), d, V_S0);
                }
                break;
            }
            u32 frint = (opc == 0x2C ? (x ? 0x1E294000u : 0x1E284000u) : (x ? 0x1E29C000u : 0x1E28C000u)) | (isd ? 1u << 22 : 0);
            RR(frint, V_S0, s);
            RR((x ? 0x9E380000u : 0x1E380000u) | (isd ? 1u << 22 : 0), d, V_S0);
            break;
        }
        case 0x2E:
        case 0x2F: {
            bool isd = pfx == 0x66;
            if (pfx != 0 && pfx != 0x66) {
                svc(SVC_UD);
                break;
            }
            int s = src_op(in, isd ? 3 : 2, false);
            // fcmp is quiet, fcmpe signals on every NaN: ucomis and comis respectively.
            E((opc == 0x2E ? 0x1E202000u : 0x1E202010u) | (isd ? 1u << 22 : 0) | (u32)s << 16 | (u32)d << 5);
            if (live & F_P)
                cset(R_PF, 7); // vc: ordered gives an odd byte, PF clear; unordered gives 0, PF set
            axflag();
            if (live & F_A)
                mov_reg(1, R_AF, R_PF);
            break;
        }
        case 0x50: {
            if (!in->rm_reg) {
                svc(SVC_UD);
                break;
            }
            if (pfx == 0) {
                // movmskps: 5-insn pure NEON USHR+USRA+XTN + single FMOV.
                RR(0x6F210400u, V_S0, in->rm); // ushr v24.4s, vrm.4s, #31
                RR(0x6F611400u, V_S0, V_S0);   // usra v24.2d, v24.2d, #31
                RR(0x0EA12800u, V_S0, V_S0);   // xtn v24.2s, v24.2d
                RR(0x7F621400u, V_S0, V_S0);   // usra d24, d24, #30
                RR(0x1E260000u, d, V_S0);      // fmov w_d, s24
            } else {
                // movmskpd: 4-insn pure NEON USHR+XTN+USRA + single FMOV.
                RR(0x6F410400u, V_S0, in->rm); // ushr v24.2d, vrm.2d, #63
                RR(0x0EA12800u, V_S0, V_S0);   // xtn v24.2s, v24.2d
                RR(0x7F611400u, V_S0, V_S0);   // usra d24, d24, #31
                RR(0x1E260000u, d, V_S0);      // fmov w_d, s24
            }
            break;
        }
        case 0x51: {
            int kind = pfx == 0xF3 ? 2 : pfx == 0xF2 ? 3 : 4;
            int s = src_op(in, kind, true);
            if (kind == 4)
                RR(pfx == 0x66 ? 0x6EE1F800u : 0x6EA1F800u, d, s);
            else
                scalar_fp_result(kind == 3 ? 0x1E61C000u : 0x1E21C000u, d, s, 1 << kind);
            break;
        }
        case 0x52:
        case 0x53: {
            if (pfx != 0 && pfx != 0xF3) {
                svc(SVC_UD);
                break;
            }
            bool scalar = pfx == 0xF3;
            int s = src_op(in, scalar ? 2 : 4, !scalar);
            if (!tr_precise_flags) {
                // Fast hardware FRSQRTE/FRECPE + 1 Newton-Raphson step (>14-bit accuracy, exceeds Intel's 12-bit spec)
                // without serializing MRS/MSR FPSR stalls.
                if (scalar) {
                    if (opc == 0x52) {
                        RR(0x7EA1D800u, V_S0, s);          // frsqrte s24, s
                        RRR(0x1E200800u, V_S1, V_S0, V_S0); // fmul s25, s24, s24
                        RRR(0x5EA0FC00u, V_S1, s, V_S1);   // frsqrts s25, s, s25
                        RRR(0x1E200800u, tr_fpcr_merges_scalar ? d : V_S2, V_S0, V_S1);
                        if (!tr_fpcr_merges_scalar)
                            ins_elem(d, 0, V_S2, 0, 4);
                    } else {
                        RR(0x5EA1D800u, V_S0, s);          // frecpe s24, s
                        RRR(0x5E20FC00u, V_S1, s, V_S0);   // frecps s25, s, s24
                        RRR(0x1E200800u, tr_fpcr_merges_scalar ? d : V_S2, V_S0, V_S1);
                        if (!tr_fpcr_merges_scalar)
                            ins_elem(d, 0, V_S2, 0, 4);
                    }
                } else if (opc == 0x52) {
                    RR(0x6EA1D800u, V_S0, s);              // frsqrte v24.4s, s.4s
                    RRR(0x6E20DC00u, V_S1, V_S0, V_S0);    // fmul v25.4s, v24.4s, v24.4s
                    RRR(0x4EA0FC00u, V_S1, s, V_S1);       // frsqrts v25.4s, s.4s, v25.4s
                    RRR(0x6E20DC00u, d, V_S0, V_S1);       // fmul d.4s, v24.4s, v25.4s
                } else {
                    RR(0x4EA1D800u, V_S0, s);              // frecpe v24.4s, s.4s
                    RRR(0x4E20FC00u, V_S1, s, V_S0);       // frecps v25.4s, s.4s, v24.4s
                    RRR(0x6E20DC00u, d, V_S0, V_S1);       // fmul d.4s, v24.4s, v25.4s
                }
                break;
            }
            E(0xD53B4420u | S0); // mrs x, fpsr
            E(0x4F03F600u | V_S1); // fmov v.4s, #1.0
            if (scalar) {
                if (opc == 0x52)
                    RR(0x1E21C000u, V_S2, s); // fsqrt s
                RRR(0x1E201800u, V_S2, V_S1, opc == 0x52 ? V_S2 : s); // fdiv s
                ins_elem(d, 0, V_S2, 0, 4);
            } else if (opc == 0x52) {
                RR(0x6EA1F800u, V_S2, s); // fsqrt .4s
                RRR(0x6E20FC00u, d, V_S1, V_S2); // fdiv .4s
            } else {
                RRR(0x6E20FC00u, d, V_S1, s);
            }
            E(0xD51B4420u | S0); // msr fpsr, x
            break;
        }
        case 0x54:
        case 0x55:
        case 0x56:
        case 0x57: {
            if (pfx != 0 && pfx != 0x66) {
                svc(SVC_UD);
                break;
            }
            int s = src_op(in, 4, true);
            if (opc == 0x54)
                RRR(0x4E201C00u, d, d, s);
            else if (opc == 0x55)
                RRR(0x4E601C00u, d, s, d); // bic d, s, d: ~d & s
            else if (opc == 0x56)
                RRR(0x4EA01C00u, d, d, s);
            else
                RRR(0x6E201C00u, d, d, s);
            break;
        }
        case 0x58:
            fp_arith(in, fp_add);
            break;
        case 0x59:
            fp_arith(in, fp_mul);
            break;
        case 0x5A:
            if (pfx == 0xF3) {
                int s = src_op(in, 2, false);
                scalar_fp_result(0x1E22C000u, d, s, 8);
            } else if (pfx == 0xF2) {
                int s = src_op(in, 3, false);
                scalar_fp_result(0x1E624000u, d, s, 4);
            } else if (pfx == 0) {
                int s = src_op(in, 3, false);
                RR(0x0E617800u, d, s); // fcvtl
            } else {
                int s = src_op(in, 4, true);
                RR(0x0E616800u, d, s); // fcvtn, clears the top half
            }
            break;
        case 0x5B: {
            int s = src_op(in, 4, true);
            if (pfx == 0)
                RR(0x4E21D800u, d, s);
            else if (!tr_precise_flags) {
                if (pfx == 0xF3) {
                    RR(0x4EA1B800u, d, s); // fcvtzs .4s
                } else {
                    RR(0x6EA19800u, V_S0, s); // frinti .4s
                    RR(0x4EA1B800u, d, V_S0); // fcvtzs .4s
                }
            } else {
                RR(pfx == 0x66 ? 0x6E21E800u : 0x4E21E800u, V_S0, s); // frint32x, frint32z
                RR(0x4EA1B800u, d, V_S0); // fcvtzs
            }
            break;
        }
        case 0x5C:
            fp_arith(in, fp_sub);
            break;
        case 0x5D:
            fp_arith(in, fp_min);
            break;
        case 0x5E:
            fp_arith(in, fp_div);
            break;
        case 0x5F:
            fp_arith(in, fp_max);
            break;
        case 0x7C: // haddpd, haddps
        case 0x7D: { // hsubpd, hsubps
            if (pfx != 0x66 && pfx != 0xF2) {
                svc(SVC_UD);
                break;
            }
            int s = src_op(in, 4, true);
            u32 fsz = pfx == 0x66 ? 1u << 22 : 0;
            if (opc == 0x7C)
                RRR(0x6E20D400u | fsz, d, d, s); // faddp
            else {
                u32 zsz = pfx == 0x66 ? 3u << 22 : 2u << 22;
                RRR(0x4E001800u | zsz, V_S0, d, s);
                RRR(0x4E005800u | zsz, V_S1, d, s);
                RRR(0x4EA0D400u | fsz, d, V_S0, V_S1); // fsub, even minus odd
            }
            break;
        }
        case 0xD0: // addsubpd, addsubps
            if (pfx != 0x66 && pfx != 0xF2)
                svc(SVC_UD);
            else
                emit_addsub(in);
            break;
        case 0xF0: // lddqu
            if (pfx != 0xF2 || in->rm_reg)
                svc(SVC_UD);
            else
                vldst(4, true, d, addr_prepare(in, 16));
            break;
        // punpck
        case 0x60:
        case 0x61:
        case 0x62:
        case 0x6C:
        case 0x68:
        case 0x69:
        case 0x6A:
        case 0x6D: {
            int s = src_op(in, 4, true);
            u32 sz = opc == 0x60 || opc == 0x68 ? 0 : opc == 0x61 || opc == 0x69 ? 1 : opc == 0x62 || opc == 0x6A ? 2 : 3;
            bool hi = opc >= 0x68 && opc != 0x6C;
            RRR((hi ? 0x4E007800u : 0x4E003800u) | sz << 22, d, d, s);
            break;
        }
        case 0x63:
            pack(in, 0x0E214800u, 0x4E214800u);
            break; // packsswb
        case 0x6B:
            pack(in, 0x0E614800u, 0x4E614800u);
            break; // packssdw
        case 0x67:
            pack(in, 0x2E212800u, 0x6E212800u);
            break; // packuswb
        case 0x64:
        case 0x65:
        case 0x66: {
            int s = src_op(in, 4, true);
            RRR(0x4E203400u | (u32)(opc - 0x64) << 22, d, d, s);
            break;
        }
        case 0x6E: {
            bool x = in->sz == 8;
            if (in->rm_reg)
                RR(x ? 0x9E670000u : 0x1E270000u, d, in->rm);
            else
                vldst(x ? 3 : 2, true, d, addr_prepare(in, x ? 8 : 4));
            break;
        }
        case 0x6F:
        case 0x7F: {
            if (pfx == 0) {
                bad(in, "MMX instruction");
                break;
            }
            if (pfx != 0x66 && pfx != 0xF3) {
                svc(SVC_UD);
                break;
            }
            if (in->rm_reg) {
                if (opc == 0x6F)
                    vmov(d, in->rm);
                else
                    vmov(in->rm, d);
            } else
                vldst(4, opc == 0x6F, d, mem16(in, pfx == 0x66));
            break;
        }
        case 0x70: {
            if (pfx == 0) {
                bad(in, "pshufw (MMX)");
                break;
            }
            int s = src_op(in, 4, true), imm = (int)in->imm;
            if (pfx == 0x66) {
                emit_lanes32(d, s, s, imm);
                break;
            } else {
                bool hi = pfx == 0xF3;
                int im = imm & 0xff;
                if (im == 0xE4) {
                    vmov(d, s);
                    break;
                }
                if (im == 0x00 || im == 0x55 || im == 0xAA || im == 0xFF) {
                    int k = (hi ? 4 : 0) + (im & 3);
                    E(0x4E020400u | (u32)k << 18 | (u32)s << 5 | V_S0); // dup v24.8h, vs.h[k]
                    ins_elem(V_S0, hi ? 0 : 1, s, hi ? 0 : 1, 8);
                    vmov(d, V_S0);
                    break;
                }
                if (im == 0x1B || im == 0xB1 || im == 0x4E) {
                    u32 rev = im == 0x1B ? 0x4E600800u : im == 0xB1 ? 0x6E600800u : 0x4EA00800u;
                    RR(rev, V_S0, s); // rev64.8h / rev32.8h / rev64.4s
                    ins_elem(V_S0, hi ? 0 : 1, s, hi ? 0 : 1, 8);
                    vmov(d, V_S0);
                    break;
                }
                int base = hi ? 4 : 0;
                ins_elem(V_S0, hi ? 0 : 1, s, hi ? 0 : 1, 8);
                for (int i = 0; i < 4; i++)
                    ins_elem(V_S0, base + i, s, base + ((im >> (2 * i)) & 3), 2);
            }
            vmov(d, V_S0);
            break;
        }
        case 0x71:
        case 0x72:
        case 0x73: {
            int sub = d & 7;
            int esize = opc == 0x71 ? 16 : opc == 0x72 ? 32 : 64;
            if (opc == 0x73 && (sub == 3 || sub == 7)) {
                int n = (int)(in->imm & 0xff), rm = in->rm;
                if (n > 15) {
                    E(0x6E201C00u | rm << 16 | rm << 5 | rm);
                    break;
                }
                E(0x6E201C00u | V_S0 << 16 | V_S0 << 5 | V_S0);
                if (n == 0)
                    break;
                if (sub == 3)
                    E(0x6E000000u | V_S0 << 16 | (u32)n << 11 | rm << 5 | rm); // psrldq
                else
                    E(0x6E000000u | (u32)rm << 16 | (u32)(16 - n) << 11 | V_S0 << 5 | rm); // pslldq
                break;
            }
            int kind = sub == 6 ? 0 : sub == 2 ? 1 : sub == 4 && opc != 0x73 ? 2 : -1;
            if (kind < 0) {
                svc(SVC_UD);
                break;
            }
            emit_shift_imm(in, esize, kind);
            break;
        }
        case 0x74:
        case 0x75:
        case 0x76: {
            int s = src_op(in, 4, true);
            RRR(0x6E208C00u | (u32)(opc - 0x74) << 22, d, d, s);
            break;
        }
        case 0x7E:
            if (pfx == 0xF3) { // movq xmm, xmm/m64
                if (in->rm_reg)
                    RR(0x1E604000u, d, in->rm);
                else
                    vldst(3, true, d, addr_prepare(in, 8));
            } else if (pfx == 0x66) {
                bool x = in->sz == 8;
                if (in->rm_reg)
                    RR(x ? 0x9E660000u : 0x1E260000u, in->rm, d);
                else
                    vldst(x ? 3 : 2, false, d, addr_prepare(in, x ? 8 : 4));
            } else
                bad(in, "MMX instruction");
            break;
        case 0xC2:
            emit_cmp_pred(in);
            break;
        case 0xC4: {
            if (pfx != 0x66) {
                bad(in, "pinsrw (MMX)");
                break;
            }
            int s = in->rm;
            if (!in->rm_reg) {
                ldst(1, 2, S0, addr_prepare(in, 2));
                s = S0;
            }
            E(0x4E021C00u | ((u32)(in->imm & 7) << 18) | (u32)s << 5 | d); // ins v.h[i], w
            break;
        }
        case 0xC5: {
            if (pfx != 0x66 || !in->rm_reg) {
                bad(in, "pextrw form");
                break;
            }
            E(0x0E023C00u | ((u32)(in->imm & 7) << 18) | (u32)in->rm << 5 | d); // umov w, v.h[i]
            break;
        }
        case 0xC6: {
            if (pfx != 0 && pfx != 0x66) {
                svc(SVC_UD);
                break;
            }
            int s = src_op(in, 4, true);
            if (pfx == 0)
                lanes32(in, d, s);
            else {
                int imm = (int)in->imm & 3;
                if (imm == 0)
                    RRR(0x4EC03800u, d, d, s); // zip1 .2d
                else if (imm == 3)
                    RRR(0x4EC07800u, d, d, s); // zip2 .2d
                else if (imm == 1)
                    E(0x6E004000u | (u32)s << 16 | (u32)d << 5 | (u32)d); // ext .16b, d, s, #8
                else
                    ins_elem(d, 1, s, 1, 8);
            }
            break;
        }
        case 0xD1:
            emit_shift_xmm(in, 16, 1);
            break;
        case 0xD2:
            emit_shift_xmm(in, 32, 1);
            break;
        case 0xD3:
            emit_shift_xmm(in, 64, 1);
            break;
        case 0xE1:
            emit_shift_xmm(in, 16, 2);
            break;
        case 0xE2:
            emit_shift_xmm(in, 32, 2);
            break;
        case 0xF1:
            emit_shift_xmm(in, 16, 0);
            break;
        case 0xF2:
            emit_shift_xmm(in, 32, 0);
            break;
        case 0xF3:
            emit_shift_xmm(in, 64, 0);
            break;
        case 0xD6:
            if (pfx != 0x66) {
                bad(in, "movq2dq or movdq2q (MMX)");
            } else if (in->rm_reg)
                RR(0x1E604000u, in->rm, d);
            else
                vldst(3, false, d, addr_prepare(in, 8));
            break;
        case 0xD7:
            if (!in->rm_reg)
                svc(SVC_UD);
            else
                emit_pmovmskb(d, in->rm);
            break;
        case 0xE4:
        case 0xE5: {
            int s = src_op(in, 4, true);
            u32 lo = opc == 0xE4 ? 0x2E60C000u : 0x0E60C000u, hi = opc == 0xE4 ? 0x6E60C000u : 0x4E60C000u;
            RRR(lo, V_S0, d, s);
            RRR(hi, V_S1, d, s);
            RRR(0x4E405800u, d, V_S0, V_S1); // uzp2 .8h
            break;
        }
        case 0xE6: {
            if (pfx == 0xF3) {
                int s = src_op(in, 3, false);
                E(0x0F20A400u | (u32)s << 5 | V_S0); // sshll
                RR(0x4E61D800u, d, V_S0);
            } else if (pfx == 0x66 || pfx == 0xF2) {
                int s = src_op(in, 4, true);
                if (!tr_precise_flags) {
                    if (pfx == 0x66) {
                        RR(0x4EE1B800u, V_S0, s); // fcvtzs .2d
                    } else {
                        RR(0x6EE19800u, V_S0, s); // frinti .2d
                        RR(0x4EE1B800u, V_S0, V_S0); // fcvtzs .2d
                    }
                    RR(0x0EA12800u, d, V_S0); // xtn
                } else {
                    RR(pfx == 0x66 ? 0x4E61E800u : 0x6E61E800u, V_S0, s); // frint32z, frint32x
                    RR(0x4EE1B800u, V_S0, V_S0); // fcvtzs .2d
                    RR(0x0EA12800u, d, V_S0); // xtn
                }
            } else
                svc(SVC_UD);
            break;
        }
        case 0xE7:
            if (in->rm_reg)
                svc(SVC_UD);
            else
                vldst(4, false, d, mem16(in, true));
            break;
        case 0xF4: {
            int s = src_op(in, 4, true);
            RR(0x0EA12800u, V_S0, d);
            RR(0x0EA12800u, V_S1, s);
            RRR(0x2EA0C000u, d, V_S0, V_S1);
            break;
        }
        case 0xF5: {
            int s = src_op(in, 4, true);
            RRR(0x0E60C000u, V_S0, d, s);
            RRR(0x4E60C000u, V_S1, d, s);
            RRR(0x4EA0BC00u, d, V_S0, V_S1);
            break;
        }
        case 0xF6: {
            int s = src_op(in, 4, true);
            RRR(0x6E207400u, V_S0, d, s);
            RR(0x6E202800u, V_S0, V_S0);
            RR(0x6E602800u, V_S0, V_S0);
            RR(0x6EA02800u, d, V_S0);
            break;
        }
        case 0xF7:
            bad(in, "maskmovdqu");
            break;
        default: {
            // The plain three-register integer table. 0 means no such opcode here.
            u32 enc = 0;
            switch (opc) {
                case 0xD4:
                    enc = 0x4EE08400u;
                    break;
                case 0xD5:
                    enc = 0x4E609C00u;
                    break;
                case 0xD8:
                    enc = 0x6E202C00u;
                    break;
                case 0xD9:
                    enc = 0x6E602C00u;
                    break;
                case 0xDA:
                    enc = 0x6E206C00u;
                    break;
                case 0xDB:
                    enc = 0x4E201C00u;
                    break;
                case 0xDC:
                    enc = 0x6E200C00u;
                    break;
                case 0xDD:
                    enc = 0x6E600C00u;
                    break;
                case 0xDE:
                    enc = 0x6E206400u;
                    break;
                case 0xE0:
                    enc = 0x6E201400u;
                    break;
                case 0xE3:
                    enc = 0x6E601400u;
                    break;
                case 0xE8:
                    enc = 0x4E202C00u;
                    break;
                case 0xE9:
                    enc = 0x4E602C00u;
                    break;
                case 0xEA:
                    enc = 0x4E606C00u;
                    break;
                case 0xEB:
                    enc = 0x4EA01C00u;
                    break;
                case 0xEC:
                    enc = 0x4E200C00u;
                    break;
                case 0xED:
                    enc = 0x4E600C00u;
                    break;
                case 0xEE:
                    enc = 0x4E606400u;
                    break;
                case 0xEF:
                    enc = 0x6E201C00u;
                    break;
                case 0xF8:
                    enc = 0x6E208400u;
                    break;
                case 0xF9:
                    enc = 0x6E608400u;
                    break;
                case 0xFA:
                    enc = 0x6EA08400u;
                    break;
                case 0xFB:
                    enc = 0x6EE08400u;
                    break;
                case 0xFC:
                    enc = 0x4E208400u;
                    break;
                case 0xFD:
                    enc = 0x4E608400u;
                    break;
                case 0xFE:
                    enc = 0x4EA08400u;
                    break;
                case 0xDF: {
                    int s = src_op(in, 4, true);
                    RRR(0x4E601C00u, d, s, d);
                    return;
                }
                default:
                    bad(in, "untranslated SSE opcode");
                    return;
            }
            int s = src_op(in, 4, true);
            RRR(enc, d, d, s);
        }
    }
}
