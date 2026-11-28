// x87 Complete Extreme Fast Mode:
// Hot FP stack, arithmetic, rounding, compare, integer-convert, and control ops run
// directly in EL0 on shadow IEEE-754 double slots ctx->fp_d[0..7] tracked by the 8-bit
// bitmask ctx->fp_d_valid. Cache misses auto-populate fp_d[p] inline from (mm, fp_exp)
// without trapping to SVC_X87. Bit-exact 64-bit integer FILD/FISTP m64 memcpy preserves
// all 64 bits. Slow paths (SVC_X87, FXSAVE, HC_FALLBACK, HC_BOUNDARY) lazily flush fp_d[].
#include <stddef.h>
#include "tr_internal.h"

enum { NONE, LOAD16, LOAD32, LOAD64, LOAD80, AREA_READ, AREA_WRITE, STORE16, STORE32, STORE64, STORE80 };

static int mem_kind(int esc, int reg) {
    static const u8 table[8][8] = {
        {LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32}, // D8
        {LOAD32, NONE, STORE32, STORE32, AREA_READ, LOAD16, AREA_WRITE, STORE16}, // D9
        {LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32, LOAD32}, // DA
        {LOAD32, STORE32, STORE32, STORE32, NONE, LOAD80, NONE, STORE80}, // DB
        {LOAD64, LOAD64, LOAD64, LOAD64, LOAD64, LOAD64, LOAD64, LOAD64}, // DC
        {LOAD64, STORE64, STORE64, STORE64, AREA_READ, NONE, AREA_WRITE, STORE16}, // DD
        {LOAD16, LOAD16, LOAD16, LOAD16, LOAD16, LOAD16, LOAD16, LOAD16}, // DE
        {LOAD16, STORE16, STORE16, STORE16, LOAD80, LOAD64, STORE80, STORE64}, // DF
    };
    return table[esc][reg];
}

static void call_monitor(const Insn *in) {
    svc(SVC_X87);
    E((u32)(in->sub == 8 ? 0xFFFF : in->sub << 8 | in->opc) | (u32)in->seg << 16 | (in->pfx == 0x66 ? 1u << 18 : 0));
    tr_aot_note_gva64_lit(tr_cc);
    E((u32)in->rip);
    E((u32)(in->rip >> 32));
}

typedef struct {
    u32 *p[32];
    u8 is_tbz[32];
    int n;
} SlowList;

static inline void add_cbz(SlowList *sl, int nz, int sf, int rt) {
    sl->p[sl->n] = tr_cc;
    sl->is_tbz[sl->n] = 0;
    sl->n++;
    cbz(nz, sf, rt, 0);
}

static inline void add_tbz(SlowList *sl, int nz, int rt, int bit) {
    sl->p[sl->n] = tr_cc;
    sl->is_tbz[sl->n] = 1;
    sl->n++;
    tbz(nz, rt, bit, 0);
}

static inline void patch_slow(const SlowList *sl) {
    for (int i = 0; i < sl->n; i++) {
        u32 off = (u32)(tr_cc - sl->p[i]);
        if (sl->is_tbz[i])
            *sl->p[i] |= (off & 0x3fffu) << 5;
        else
            *sl->p[i] |= (off & 0x7ffffu) << 5;
    }
}

static inline void patch_b(u32 *p) { *p |= (u32)(tr_cc - p) & 0x3ffffffu; }
static inline void patch_cbz_here(u32 *p) { *p |= ((u32)(tr_cc - p) & 0x7ffffu) << 5; }

// All encodings verified against clang -arch arm64 + objdump.
static inline void fmov_d_x(int dd, int xn) { E(0x9E670000u | (u32)xn << 5 | (u32)dd); }
static inline void fmov_x_d(int xd, int dn) { E(0x9E660000u | (u32)dn << 5 | (u32)xd); }
static inline void fmov_s_w(int sd, int wn) { E(0x1E270000u | (u32)wn << 5 | (u32)sd); }
static inline void fmov_w_s(int wd, int sn) { E(0x1E260000u | (u32)sn << 5 | (u32)wd); }
static inline void fcvt_d_s(int dd, int sn) { E(0x1E22C000u | (u32)sn << 5 | (u32)dd); }
static inline void fcvt_s_d(int sd, int dn) { E(0x1E624000u | (u32)dn << 5 | (u32)sd); }
static inline void fadd_d(int dd, int dn, int dm) { E(0x1E602800u | (u32)dm << 16 | (u32)dn << 5 | (u32)dd); }
static inline void fsub_d(int dd, int dn, int dm) { E(0x1E603800u | (u32)dm << 16 | (u32)dn << 5 | (u32)dd); }
static inline void fmul_d(int dd, int dn, int dm) { E(0x1E600800u | (u32)dm << 16 | (u32)dn << 5 | (u32)dd); }
static inline void fdiv_d(int dd, int dn, int dm) { E(0x1E601800u | (u32)dm << 16 | (u32)dn << 5 | (u32)dd); }
static inline void fsqrt_d(int dd, int dn) { E(0x1E61C000u | (u32)dn << 5 | (u32)dd); }
static inline void fneg_d(int dd, int dn) { E(0x1E614000u | (u32)dn << 5 | (u32)dd); }
static inline void fmadd_d(int dd, int dn, int dm, int da) { E(0x1F400000u | (u32)dm << 16 | (u32)da << 10 | (u32)dn << 5 | (u32)dd); }
static inline void fmsub_d(int dd, int dn, int dm, int da) { E(0x1F408000u | (u32)dm << 16 | (u32)da << 10 | (u32)dn << 5 | (u32)dd); }
static inline void fcmp_d(int dn, int dm) { E(0x1E602000u | (u32)dm << 16 | (u32)dn << 5); }
static inline void scvtf_d_w(int dd, int wn) { E(0x1E620000u | (u32)wn << 5 | (u32)dd); }
static inline void frintn_d(int dd, int dn) { E(0x1E644000u | (u32)dn << 5 | (u32)dd); }
static inline void frintm_d(int dd, int dn) { E(0x1E654000u | (u32)dn << 5 | (u32)dd); }
static inline void frintp_d(int dd, int dn) { E(0x1E64C000u | (u32)dn << 5 | (u32)dd); }
static inline void frintz_d(int dd, int dn) { E(0x1E65C000u | (u32)dn << 5 | (u32)dd); }
static inline void fcvtns_w_d(int wd, int dn) { E(0x1E600000u | (u32)dn << 5 | (u32)wd); }
static inline void fcvtms_w_d(int wd, int dn) { E(0x1E700000u | (u32)dn << 5 | (u32)wd); }
static inline void fcvtps_w_d(int wd, int dn) { E(0x1E680000u | (u32)dn << 5 | (u32)wd); }
static inline void fcvtzs_w_d(int wd, int dn) { E(0x1E780000u | (u32)dn << 5 | (u32)wd); }
static inline void fcvtns_x_d(int xd, int dn) { E(0x9E600000u | (u32)dn << 5 | (u32)xd); }
static inline void fcvtms_x_d(int xd, int dn) { E(0x9E700000u | (u32)dn << 5 | (u32)xd); }
static inline void fcvtps_x_d(int xd, int dn) { E(0x9E680000u | (u32)dn << 5 | (u32)xd); }
static inline void fcvtzs_x_d(int xd, int dn) { E(0x9E780000u | (u32)dn << 5 | (u32)xd); }

// Guard: CR0.TS|EM == 0 and FSW.ES == 0. Loads S3 = *(u32*)&ctx->fp_top. Clobbers S1, S5.
static void emit_x87_guard(SlowList *sl) {
    ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
    movimm(S5, 0xC);
    logic_reg(1, 0, 0, S1, S1, S5);
    add_cbz(sl, 1, 1, S1);
    ldst(1, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    add_tbz(sl, 1, S3, 23); // FSW.ES
}

// Compute physical reg index p = (top + i) & 7 into `rd`, and verify tag bit is set (non-empty).
static void emit_st_phys_nonempty(int rd, int i, SlowList *sl) {
    if (i == 0) {
        ubfx(0, rd, S3, 0, 3);
    } else {
        add_imm(0, 0, 0, rd, S3, (u32)i);
        ubfx(0, rd, rd, 0, 3);
    }
    movimm(S1, 1);
    var_shift(0, 0, S1, S1, rd);
    lsl_imm(0, S1, S1, 8);
    logic_reg(0, 0, 0, S5, S3, S1);
    add_cbz(sl, 0, 0, S5); // empty -> slow
}

// Load physical slot `phys` into `dreg`.
// Fast path (fp_d_valid[phys] == 1): 1 LDR from fp_d[phys].
// Cold path (fp_d_valid[phys] == 0): inline f80 -> f64 conversion in EL0, caches into fp_d[phys]
// and sets fp_d_valid[phys] = 1 without trapping to SVC_X87!
// Preserves S3, phys (must not be S1/S2/S4/S5). Clobbers S1, S2, S4, S5.
static void emit_load_cached_d(int phys, int dreg, SlowList *sl) {
    ldst(1, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    movimm(S5, 1);
    var_shift(0, 0, S5, S5, phys);
    logic_reg(0, 0, 0, S2, S4, S5);
    u32 *miss = tr_cc;
    cbz(0, 0, S2, 0);

    // Hit: load fp_d[phys]
    add_reg(1, 0, 0, S1, CTX, phys, 3);
    ldst(1, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
    u32 *have_d = tr_cc;
    b_rel(0);

    // Miss: convert (mm[phys], fp_exp[phys]) -> IEEE-754 double inline & cache it!
    patch_cbz_here(miss);
    add_reg(1, 0, 0, S1, CTX, phys, 3);
    ldst(1, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, mm)});
    add_reg(1, 0, 0, S1, CTX, phys, 1);
    ldst(1, 2, S5, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_exp)});
    ubfx(0, S1, S5, 0, 15); // exp80
    u32 *is_zero = tr_cc;
    cbz(0, 0, S1, 0);

    movimm(S2, 15361);
    add_reg(0, 1, 0, S1, S1, S2, 0);
    lsr_imm(0, S2, S1, 11);
    add_cbz(sl, 1, 0, S2);
    add_imm(0, 0, 0, S2, S1, 2);
    add_tbz(sl, 1, S2, 11);

    add_imm(0, 0, 0, S1, S1, 1);
    ubfx(1, S4, S4, 11, 52);
    bfi(1, S4, S1, 52, 11);
    ubfx(1, S2, S5, 15, 1);
    bfi(1, S4, S2, 63, 1);
    u32 *cache_it = tr_cc;
    b_rel(0);

    patch_cbz_here(is_zero);
    add_cbz(sl, 1, 1, S4); // subnormal -> slow
    ubfx(1, S4, S5, 15, 1);
    lsl_imm(1, S4, S4, 63);

    patch_b(cache_it);
    add_reg(1, 0, 0, S1, CTX, phys, 3);
    ldst(0, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
    ldst(1, 8, S2, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    movimm(S5, 1);
    var_shift(0, 0, S5, S5, phys);
    logic_reg(0, 1, 0, S2, S2, S5);
    ldst(0, 8, S2, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});

    patch_b(have_d);
    fmov_d_x(dreg, S4);
}

// Store `dreg` into fp_d[phys] and set bit `phys` in fp_d_valid.
static void emit_store_cached_d(int dreg, int phys) {
    fmov_x_d(S4, dreg);
    add_reg(1, 0, 0, S1, CTX, phys, 3);
    ldst(0, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
    ldst(1, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    movimm(S5, 1);
    var_shift(0, 0, S5, S5, phys);
    logic_reg(0, 1, 0, S4, S4, S5);
    ldst(0, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
}

// Clear bit `phys` in fp_d_valid. Clobbers S4, S5.
static void emit_clear_cached_bit(int phys) {
    ldst(1, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    movimm(S5, 1);
    var_shift(0, 0, S5, S5, phys);
    logic_reg(0, 0, 1, S4, S4, S5); // BIC
    ldst(0, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
}

// Pop ST(0): clear tag bit + fp_d_valid bit for `p0` and advance fp_top = (p0 + 1) & 7 in S3.
static void emit_pop_st0(int p0) {
    emit_clear_cached_bit(p0);
    movimm(S1, 1);
    var_shift(0, 0, S1, S1, p0);
    lsl_imm(0, S1, S1, 8);
    logic_reg(0, 0, 1, S3, S3, S1);
    add_imm(0, 0, 0, S2, p0, 1);
    ubfx(0, S2, S2, 0, 3);
    bfi(0, S3, S2, 0, 3);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
}

// Push raw IEEE-754 double bits in `val_reg` (not S1..S5) into fp_d[new_top].
static void emit_push_cached_x64(int val_reg, SlowList *sl) {
    add_imm(0, 1, 0, S2, S3, 1);
    ubfx(0, S2, S2, 0, 3);
    movimm(S1, 1);
    var_shift(0, 0, S1, S1, S2);
    lsl_imm(0, S1, S1, 8);
    logic_reg(0, 0, 0, S5, S3, S1);
    add_cbz(sl, 1, 0, S5); // overflow -> slow

    add_reg(1, 0, 0, S5, CTX, S2, 3);
    ldst(0, 8, val_reg, (Addr){.base = S5, .imm = offsetof(EngineCtx, fp_d)});
    ldst(1, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    lsr_imm(0, S5, S1, 8);
    logic_reg(0, 1, 0, S4, S4, S5);
    ldst(0, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});

    movimm(S4, (1u << 25) | 7);
    logic_reg(0, 0, 1, S3, S3, S4);
    logic_reg(0, 1, 0, S3, S3, S2);
    logic_reg(0, 1, 0, S3, S3, S1);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
}

// Dispatch 4-way FCW.RC (bits 11:10) for FCVT*S (is64=false -> W_S4, is64=true -> X_S4) or FRINT* (mode=2 -> D_V_S0).
// Clobbers S1, S2, S4.
static void emit_fcw_round_dispatch(int mode) {
    ldst(1, 2, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fcw)});
    ubfx(0, S1, S1, 10, 2); // RC: 0=N, 1=M, 2=P, 3=Z
    u32 *b_n = tr_cc; cbz(0, 0, S1, 0);
    add_imm(0, 1, 0, S2, S1, 1);
    u32 *b_m = tr_cc; cbz(0, 0, S2, 0);
    add_imm(0, 1, 0, S2, S1, 2);
    u32 *b_p = tr_cc; cbz(0, 0, S2, 0);
    // RC == 3 (Z)
    if (mode == 0) fcvtzs_w_d(S4, V_S0);
    else if (mode == 1) fcvtzs_x_d(S4, V_S0);
    else frintz_d(V_S0, V_S0);
    u32 *d3 = tr_cc; b_rel(0);

    patch_cbz_here(b_n);
    if (mode == 0) fcvtns_w_d(S4, V_S0);
    else if (mode == 1) fcvtns_x_d(S4, V_S0);
    else frintn_d(V_S0, V_S0);
    u32 *d0 = tr_cc; b_rel(0);

    patch_cbz_here(b_m);
    if (mode == 0) fcvtms_w_d(S4, V_S0);
    else if (mode == 1) fcvtms_x_d(S4, V_S0);
    else frintm_d(V_S0, V_S0);
    u32 *d1 = tr_cc; b_rel(0);

    patch_cbz_here(b_p);
    if (mode == 0) fcvtps_w_d(S4, V_S0);
    else if (mode == 1) fcvtps_x_d(S4, V_S0);
    else frintp_d(V_S0, V_S0);

    patch_b(d0);
    patch_b(d1);
    patch_b(d3);
}

// Fast EL0 FLD m80 (DB /5)
static void load_m80_fast(const Insn *in) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    ldst(1, 2, S1, (Addr){.base = AD, .imm = 8});
    ubfx(0, S4, S1, 0, 15);
    u32 *is_zero = tr_cc;
    cbz(0, 0, S4, 0);

    movimm(S2, 15361);
    add_reg(0, 1, 0, S4, S4, S2, 0);
    lsr_imm(0, S2, S4, 11);
    add_cbz(&sl, 1, 0, S2);
    add_imm(0, 0, 0, S2, S4, 2);
    add_tbz(&sl, 1, S2, 11);

    add_imm(0, 0, 0, S4, S4, 1);
    ubfx(1, S0, S0, 11, 52);
    bfi(1, S0, S4, 52, 11);
    ubfx(1, S2, S1, 15, 1);
    bfi(1, S0, S2, 63, 1);
    u32 *have_d = tr_cc;
    b_rel(0);

    patch_cbz_here(is_zero);
    add_cbz(&sl, 1, 1, S0);
    ubfx(1, S0, S1, 15, 1);
    lsl_imm(1, S0, S0, 63);

    patch_b(have_d);
    emit_push_cached_x64(S0, &sl);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    ldst(1, 8, S0, (Addr){.base = AD});
    ldst(1, 2, S1, (Addr){.base = AD, .imm = 8});
    call_monitor(in);
    patch_b(done);
}

// Fast EL0 FSTP m80 (DB /7)
static void store_m80_fast(const Insn *in, Addr a, Addr hi) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    emit_st_phys_nonempty(S0, 0, &sl);
    emit_load_cached_d(S0, V_S0, &sl);
    fmov_x_d(S4, V_S0);
    ubfx(1, S1, S4, 52, 11);
    u32 *is_zero = tr_cc;
    cbz(0, 0, S1, 0);

    add_imm(0, 0, 0, S2, S1, 1);
    add_tbz(&sl, 1, S2, 11);

    movimm(S2, 15360);
    add_reg(0, 0, 0, S5, S1, S2, 0);
    ubfx(1, S2, S4, 63, 1);
    bfi(0, S5, S2, 15, 1);
    lsl_imm(1, S4, S4, 11);
    movimm(S2, 1ULL << 63);
    logic_reg(1, 1, 0, S4, S4, S2);
    u32 *do_store = tr_cc;
    b_rel(0);

    patch_cbz_here(is_zero);
    ubfx(1, S2, S4, 0, 52);
    add_cbz(&sl, 1, 1, S2);
    ubfx(1, S5, S4, 63, 1);
    lsl_imm(0, S5, S5, 15);
    movimm(S4, 0);

    patch_b(do_store);
    ldst(0, 8, S4, a);
    ldst(0, 2, S5, hi);
    emit_pop_st0(S0);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    ldst(0, 8, S0, a);
    ldst(0, 2, S1, hi);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    patch_b(done);
}

// Fast EL0 FLDCW (D9 /5) & FNSTCW (D9 /7)
static void fcw_fast(const Insn *in, Addr a, bool is_store) {
    SlowList sl = {.n = 0};
    ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
    movimm(S5, 0xC);
    logic_reg(1, 0, 0, S1, S1, S5);
    add_cbz(&sl, 1, 1, S1);
    if (is_store) {
        ldst(1, 2, S0, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fcw)});
        ldst(0, 2, S0, a);
    } else {
        ldst(1, 2, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fsw)});
        logic_reg(0, 0, 1, S1, S1, S0);
        movimm(S2, 0x3F);
        logic_reg(0, 0, 0, S1, S1, S2);
        add_cbz(&sl, 1, 0, S1);
        ldst(0, 2, S0, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fcw)});
    }
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    if (is_store) {
        ldst(0, 2, S0, a);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    }
    patch_b(done);
}

// Fast FLD m32 (D9 /0) / FLD m64 (DD /0)
static void load_fast_fp(const Insn *in, bool is32) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    if (is32) {
        fmov_s_w(V_S0, S0);
        fcvt_d_s(V_S0, V_S0);
        fmov_x_d(S0, V_S0);
    }
    emit_push_cached_x64(S0, &sl);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    if (is32)
        ldst(1, 4, S0, (Addr){.base = AD});
    call_monitor(in);
    patch_b(done);
}

// Fast FST/FSTP m64 (DD /2, /3) or m32 (D9 /2, /3)
static void store_fast_fp(const Insn *in, Addr a, bool is32, bool pop) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    emit_st_phys_nonempty(S0, 0, &sl);
    emit_load_cached_d(S0, V_S0, &sl);
    if (is32) {
        fcvt_s_d(V_S0, V_S0);
        fmov_w_s(S4, V_S0);
        ldst(0, 4, S4, a);
    } else {
        fmov_x_d(S4, V_S0);
        ldst(0, 8, S4, a);
    }
    if (pop)
        emit_pop_st0(S0);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    ldst(0, is32 ? 4 : 8, S0, a);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    patch_b(done);
}

// Update x87 FSW C3/C2/C0 from ARM64 FCMP V_S0, V_S1 without clobbering x86 NZCV.
static void emit_fcom_body(int p0, bool pop) {
    mrs_nzcv(S4);
    fcmp_d(V_S0, V_S1);
    movimm(S2, 0);
    movimm(S1, 0x40000000u); // C3 (EQ)
    csel(0, S2, S1, S2, 0);
    movimm(S1, 0x01000000u); // C0 (LT: N==1)
    csel(0, S2, S1, S2, 4);
    movimm(S1, 0x45000000u); // C3|C2|C0 (VS: unordered)
    csel(0, S2, S1, S2, 6);
    msr_nzcv(S4);
    movimm(S1, 0x47000000u);
    logic_reg(0, 0, 1, S3, S3, S1);
    logic_reg(0, 1, 0, S3, S3, S2);
    if (pop)
        emit_pop_st0(p0);
    else
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
}

// Fast DC m64, D8 m32, DA int32, DE int16 arithmetic & compare. S0 holds loaded mem value.
// kind_mode: 0=m64, 1=m32, 2=int32, 3=int16
static void arith_mem_fast(const Insn *in, int reg, int kind_mode) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    if (kind_mode == 0) {
        fmov_d_x(V_S1, S0);
    } else if (kind_mode == 1) {
        fmov_s_w(V_S1, S0);
        fcvt_d_s(V_S1, V_S1);
    } else {
        if (kind_mode == 3)
            sbfx(0, S0, S0, 0, 16);
        scvtf_d_w(V_S1, S0);
    }
    emit_st_phys_nonempty(S0, 0, &sl);
    emit_load_cached_d(S0, V_S0, &sl);
    if (reg == 2 || reg == 3) {
        emit_fcom_body(S0, reg == 3);
    } else {
        switch (reg) {
            case 0: fadd_d(V_S0, V_S0, V_S1); break;
            case 1: fmul_d(V_S0, V_S0, V_S1); break;
            case 4: fsub_d(V_S0, V_S0, V_S1); break;
            case 5: fsub_d(V_S0, V_S1, V_S0); break;
            case 6: fdiv_d(V_S0, V_S0, V_S1); break;
            case 7: fdiv_d(V_S0, V_S1, V_S0); break;
        }
        emit_store_cached_d(V_S0, S0);
    }
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    ldst(1, kind_mode == 0 ? 8 : kind_mode == 3 ? 2 : 4, S0, (Addr){.base = AD});
    call_monitor(in);
    patch_b(done);
}

// Lossless EL0 FILD m16 (DF /0) and FILD m32 (DB /0)
static void fild_int32_fast(const Insn *in, bool is16) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    if (is16)
        sbfx(0, S0, S0, 0, 16);
    scvtf_d_w(V_S0, S0);
    fmov_x_d(S0, V_S0);
    emit_push_cached_x64(S0, &sl);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    ldst(1, is16 ? 2 : 4, S0, (Addr){.base = AD});
    call_monitor(in);
    patch_b(done);
}

// Fast EL0 FIST/FISTP m16 (DF /2,/3) and m32 (DB /2,/3) honoring all 4 FCW.RC modes!
static void fist_int_fast(const Insn *in, Addr a, bool is16, bool pop) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    emit_st_phys_nonempty(S0, 0, &sl);
    emit_load_cached_d(S0, V_S0, &sl);
    // Reject NaN/Inf or |x| >= 2^(15 or 31) (exp64 >= 1023 + 14/30 conservative safe bound) -> slow
    fmov_x_d(S4, V_S0);
    ubfx(1, S1, S4, 52, 11);
    movimm(S2, is16 ? 1037 : 1053); // |x| < 2^14 or 2^30: guaranteed no int16/int32 overflow under any RC
    add_reg(0, 1, 0, S1, S1, S2, 0);
    add_tbz(&sl, 0, S1, 31);
    emit_fcw_round_dispatch(0); // W_S4 = rounded int32
    ldst(0, is16 ? 2 : 4, S4, a);
    if (pop)
        emit_pop_st0(S0);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    ldst(0, is16 ? 2 : 4, S0, a);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    patch_b(done);
}

// Bit-exact EL0 FILD m64 (DF /5)
static void fild_m64_fast(const Insn *in) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    add_imm(0, 1, 0, S2, S3, 1);
    ubfx(0, S2, S2, 0, 3);
    movimm(S1, 1);
    var_shift(0, 0, S1, S1, S2);
    lsl_imm(0, S1, S1, 8);
    logic_reg(0, 0, 0, S5, S3, S1);
    add_cbz(&sl, 1, 0, S5);

    u32 *is_zero = tr_cc;
    cbz(0, 1, S0, 0);

    asr_imm(1, S4, S0, 63);
    logic_reg(1, 2, 0, S5, S0, S4);
    add_reg(1, 1, 0, S5, S5, S4, 0);
    clz(1, S4, S5);
    var_shift(1, 0, S5, S5, S4);
    movimm(AD, 16446);
    add_reg(0, 1, 0, S4, AD, S4, 0);
    ubfx(1, AD, S0, 63, 1);
    bfi(0, S4, AD, 15, 1);
    u32 *do_store = tr_cc;
    b_rel(0);

    patch_cbz_here(is_zero);
    movimm(S5, 0);
    movimm(S4, 0);

    patch_b(do_store);
    add_reg(1, 0, 0, AD, CTX, S2, 3);
    ldst(0, 8, S5, (Addr){.base = AD, .imm = offsetof(EngineCtx, mm)});
    add_reg(1, 0, 0, AD, CTX, S2, 1);
    ldst(0, 2, S4, (Addr){.base = AD, .imm = offsetof(EngineCtx, fp_exp)});
    emit_clear_cached_bit(S2);
    movimm(S0, (1u << 25) | 7);
    logic_reg(0, 0, 1, S3, S3, S0);
    logic_reg(0, 1, 0, S3, S3, S2);
    logic_reg(0, 1, 0, S3, S3, S1);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    patch_b(done);
}

// Dual-path EL0 FISTP m64 (DF /7): cached double (all 4 FCW.RC) + exact 64-bit integer memcpy.
static void fistp_m64_fast(const Insn *in, Addr a) {
    SlowList sl = {.n = 0};
    emit_x87_guard(&sl);
    emit_st_phys_nonempty(S0, 0, &sl);
    ldst(1, 8, S4, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_d_valid)});
    movimm(S5, 1);
    var_shift(0, 0, S5, S5, S0);
    logic_reg(0, 0, 0, S5, S4, S5);
    u32 *is_exact80 = tr_cc;
    cbz(0, 0, S5, 0);

    // Cached double path: require |x| < 2^62 (exp64 < 1085) -> guaranteed no int64 overflow
    add_reg(1, 0, 0, S1, CTX, S0, 3);
    ldst(1, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
    ubfx(1, S1, S4, 52, 11);
    movimm(S2, 1085);
    add_reg(0, 1, 0, S1, S1, S2, 0);
    add_tbz(&sl, 0, S1, 31);
    fmov_d_x(V_S0, S4);
    emit_fcw_round_dispatch(1); // X_S4 = rounded int64
    ldst(0, 8, S4, a);
    emit_pop_st0(S0);
    u32 *done_d = tr_cc;
    b_rel(0);

    patch_cbz_here(is_exact80);
    add_reg(1, 0, 0, S1, CTX, S0, 3);
    ldst(1, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, mm)});
    add_reg(1, 0, 0, S1, CTX, S0, 1);
    ldst(1, 2, S5, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_exp)});

    ubfx(0, S1, S5, 0, 15);
    u32 *is_zero = tr_cc;
    cbz(0, 0, S1, 0);

    movimm(S2, 16383);
    add_reg(0, 1, 0, S1, S1, S2, 0);
    lsr_imm(0, S2, S1, 6);
    add_cbz(&sl, 1, 0, S2);
    add_imm(0, 0, 0, S2, S1, 1);
    add_tbz(&sl, 1, S2, 6);

    movimm(S2, 63);
    add_reg(0, 1, 0, S2, S2, S1, 0);
    add_imm(0, 0, 0, S1, S1, 1);
    var_shift(1, 0, S1, S4, S1);
    add_cbz(&sl, 1, 1, S1);

    var_shift(1, 1, S4, S4, S2);
    u32 *pos = tr_cc;
    tbz(0, S5, 15, 0);
    add_reg(1, 1, 0, S4, XZR, S4, 0);
    u32 *do_store = tr_cc;
    b_rel(0);

    patch_cbz_here(is_zero);
    add_cbz(&sl, 1, 1, S4);
    movimm(S4, 0);

    patch_b(do_store);
    *pos |= ((u32)(tr_cc - pos) & 0x3fffu) << 5;
    ldst(0, 8, S4, a);
    emit_pop_st0(S0);
    u32 *done = tr_cc;
    b_rel(0);

    patch_slow(&sl);
    call_monitor(in);
    ldst(0, 8, S0, a);
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
    patch_b(done);
    patch_b(done_d);
}

// Fast register-register x87 ops.
static bool try_fast_reg(const Insn *in) {
    int esc = in->sub, m = in->opc, reg = (m >> 3) & 7, rm = m & 7;

    // FWAIT (9B)
    if (esc == 8) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // DF E0: FNSTSW AX
    if (esc == 7 && m == 0xE0) {
        SlowList sl = {.n = 0};
        ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
        movimm(S5, 0xC);
        logic_reg(1, 0, 0, S1, S1, S5);
        add_cbz(&sl, 1, 1, S1);
        ldst(1, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        lsr_imm(0, S0, S3, 16);
        ubfx(0, S1, S3, 0, 3);
        bfi(0, S0, S1, 11, 3);
        gpr_write(0, 2, false, S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        gpr_write(0, 2, false, S0);
        patch_b(done);
        return true;
    }

    // D9 D0: FNOP
    if (esc == 1 && m == 0xD0) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 E8..EE: FLD1, FLDL2T, FLDL2E, FLDPI, FLDLG2, FLDLN2, FLDZ
    if (esc == 1 && m >= 0xE8 && m <= 0xEE) {
        static const u64 kconsts[7] = {
            0x3FF0000000000000ULL, // 1.0
            0x400A934F0979A371ULL, // log2(10)
            0x3FF71547652B82FEULL, // log2(e)
            0x400921FB54442D18ULL, // pi
            0x3FD34413509F79FFULL, // log10(2)
            0x3FE62E42FEFA39EFULL, // ln(2)
            0x0000000000000000ULL, // 0.0
        };
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        movimm(S0, kconsts[m - 0xE8]);
        emit_push_cached_x64(S0, &sl);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 E4: FTST (compare ST(0) with +0.0)
    if (esc == 1 && m == 0xE4) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_d_x(V_S1, XZR);
        emit_fcom_body(S0, false);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 E5: FXAM (fast path for normal finite numbers & zero; rare subnormal/Inf/NaN/empty -> monitor)
    if (esc == 1 && m == 0xE5) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S4, V_S0);
        ubfx(1, S1, S4, 52, 11);
        movimm(S5, 0x7FF);
        add_reg(1, 1, 1, XZR, S1, S5, 0);
        cset(S2, 0); // EQ: Inf or NaN -> slow
        add_cbz(&sl, 1, 0, S2);
        // Check if exp == 0 (zero or subnormal)
        u32 *b_exp_nz = tr_cc;
        cbz(1, 1, S1, 0);
        // exp == 0: if mantissa != 0 (subnormal), slow; else zero (FC3 = 0x4000)
        ubfx(1, S2, S4, 0, 52);
        add_cbz(&sl, 1, 1, S2);
        movimm(S1, 0x4000u << 16); // FC3
        u32 *b_have_cc = tr_cc;
        b_rel(0);
        patch_cbz_here(b_exp_nz);
        movimm(S1, 0x0400u << 16); // FC2 (normal)
        patch_b(b_have_cc);
        // Add FC1 (0x0200 << 16 = bit 25) if sign bit 63 is set
        lsr_imm(1, S2, S4, 63);
        lsl_imm(0, S2, S2, 25);
        logic_reg(0, 1, 0, S1, S1, S2);
        movimm(S5, 0x4700u << 16);
        logic_reg(0, 0, 1, S3, S3, S5);
        logic_reg(0, 1, 0, S3, S3, S1);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 F6 (FDECSTP) / D9 F7 (FINCSTP)
    if (esc == 1 && (m == 0xF6 || m == 0xF7)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        add_imm(0, 0, 0, S1, S3, m == 0xF7 ? 1u : 7u);
        ubfx(0, S1, S1, 0, 3);
        bfi(0, S3, S1, 0, 3);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 F5 (FPREM1) / D9 F8 (FPREM)
    if (esc == 1 && (m == 0xF5 || m == 0xF8)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, 1, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        fdiv_d(V_S2, V_S0, V_S1);
        fmov_x_d(S1, V_S2);
        ubfx(1, S2, S1, 52, 11);
        movimm(S5, 1073); // |q| < 2^50 (also rejects NaN/Inf from b==0 or a==Inf)
        add_reg(1, 1, 1, XZR, S2, S5, 0);
        cset(S1, 2); // CS: >= 1073 -> slow
        add_cbz(&sl, 1, 0, S1);
        if (m == 0xF5)
            frintn_d(V_S2, V_S2);
        else
            frintz_d(V_S2, V_S2);
        fcvtzs_x_d(S4, V_S2);
        fmsub_d(V_S0, V_S2, V_S1, V_S0); // V_S0 = a - q * b
        emit_store_cached_d(V_S0, S0);
        // x87 FPREM sets C0,C3,C1 = bits 2,1,0 of |q|, and C2 = 0
        add_reg(1, 1, 1, XZR, S4, XZR, 0);
        add_reg(1, 1, 0, S1, XZR, S4, 0);
        csel(1, S4, S1, S4, 11); // LT -> -S4
        movimm(S5, 0x4700u << 16);
        logic_reg(0, 0, 1, S3, S3, S5);
        // bit 0 -> C1 (bit 25), bit 1 -> C3 (bit 30), bit 2 -> C0 (bit 24)
        ubfx(0, S1, S4, 0, 1);
        bfi(0, S3, S1, 25, 1);
        ubfx(0, S1, S4, 1, 1);
        bfi(0, S3, S1, 30, 1);
        ubfx(0, S1, S4, 2, 1);
        bfi(0, S3, S1, 24, 1);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 FD: FSCALE (ST(0) *= 2^trunc(ST(1)))
    if (esc == 1 && m == 0xFD) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, 1, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        fmov_x_d(S4, V_S0);
        ubfx(1, S1, S4, 52, 11); // exp_a
        add_cbz(&sl, 0, 0, S1);  // exp_a == 0 (zero/subnormal) -> slow
        movimm(S5, 0x7FF);
        add_reg(1, 1, 1, XZR, S1, S5, 0);
        cset(S2, 0);
        add_cbz(&sl, 1, 0, S2);  // Inf/NaN -> slow
        fmov_x_d(S2, V_S1);
        ubfx(1, S2, S2, 52, 11); // exp_b
        movimm(S5, 1032);        // |b| < 512
        add_reg(1, 1, 1, XZR, S2, S5, 0);
        cset(S2, 2);
        add_cbz(&sl, 1, 0, S2);
        fcvtzs_w_d(S2, V_S1);
        sbfx(1, S2, S2, 0, 32);
        add_reg(1, 0, 0, S1, S1, S2, 0); // new_exp = exp_a + trunc(b)
        add_imm(1, 1, 1, S2, S1, 1);     // (new_exp - 1) <u 2046
        movimm(S5, 2046);
        add_reg(1, 1, 1, XZR, S2, S5, 0);
        cset(S2, 2); // CS: out of normal double exponent range -> slow
        add_cbz(&sl, 1, 0, S2);
        bfi(1, S4, S1, 52, 11);
        add_reg(1, 0, 0, S1, CTX, S0, 3);
        ldst(0, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 F2: FPTAN (fast EL0 double tan(x) for |x| < 1024, replaces ST(0) with tan(x), pushes 1.0, clears C2)
    if (esc == 1 && m == 0xF2) {
        static const u64 ktan_c[12] = {
            0x3FD5555555555555ULL, // 1/3
            0x3FC1111111111111ULL, // 2/15
            0x3FABA1BA1BA1BA1CULL, // 17/315
            0x3F9664F4882C10FAULL, // 62/2835
            0x3F8226E355E6C23DULL, // 1382/155925
            0x3F6D6D3D0E157DE0ULL, // 21844/6081075
            0x3F57DA36452B75E3ULL, // 929569/638512875
            0x3F43558248036744ULL, // 6404582/10854718875
            0x3F2F57D7734D1664ULL, // 443861162/1856156927625
            0x3F1967E18AFCAFADULL, // 18888466084/194896477400625
            0x3F0497D8EEA25259ULL, // 113927491862/2900518163668125
            0x3EEF1E3B2B1B7C50ULL, // 58870668456604/3698160658676859375
        };
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        // Check stack not full (ST(7) must be empty before modifying ST(0) and pushing 1.0)
        add_imm(0, 0, 0, S1, S3, 7);
        ubfx(0, S1, S1, 0, 3);
        lsr_imm(0, S2, S3, 8);
        var_shift(0, 1, S2, S2, S1);
        add_tbz(&sl, 1, S2, 0); // stack overflow -> slow
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S1, V_S0);
        ubfx(1, S2, S1, 0, 63); // |x| bits
        movimm(S5, 0x4090000000000000ULL); // 1024.0
        add_reg(1, 1, 1, XZR, S2, S5, 0);
        cset(S1, 2); // CS: >= 1024.0 or NaN/Inf -> slow
        add_cbz(&sl, 1, 0, S1);
        // Cody-Waite range reduction: k = round(x * (2/pi)), r = (x - k*PIO2_HI) - k*PIO2_LO
        movimm(S1, 0x3FE45F306DC9C883ULL); // 2/pi
        fmov_d_x(28, S1);
        fmul_d(V_S1, V_S0, 28);
        frintn_d(V_S1, V_S1);
        fcvtzs_w_d(S4, V_S1); // int32 k in S4
        movimm(S1, 0x3FF921FB54400000ULL); // PIO2_HI
        fmov_d_x(28, S1);
        fmsub_d(V_S0, V_S1, 28, V_S0);
        movimm(S1, 0x3DD0B4611A626331ULL); // PIO2_LO
        fmov_d_x(28, S1);
        fmsub_d(V_S0, V_S1, 28, V_S0);
        // u = r * 0.5, z = u * u, uz = u * z
        movimm(S1, 0x3FE0000000000000ULL); // 0.5
        fmov_d_x(28, S1);
        fmul_d(V_S0, V_S0, 28); // V_S0 = u
        fmul_d(V_S1, V_S0, V_S0); // V_S1 = z = u^2
        fmul_d(27, V_S0, V_S1);   // v27 = u * z
        movimm(S1, ktan_c[11]);
        fmov_d_x(V_S2, S1);
        for (int ci = 10; ci >= 0; ci--) {
            movimm(S1, ktan_c[ci]);
            fmov_d_x(28, S1);
            fmadd_d(V_S2, V_S2, V_S1, 28);
        }
        fmadd_d(V_S0, 27, V_S2, V_S0); // V_S0 = tu = u + (u*z)*p
        // res = (tu + tu) / (1.0 - tu * tu)
        fadd_d(27, V_S0, V_S0);
        movimm(S1, 0x3FF0000000000000ULL); // 1.0
        fmov_d_x(28, S1);
        fmsub_d(V_S1, V_S0, V_S0, 28);
        fdiv_d(V_S0, 27, V_S1);
        // If k & 1: res = -1.0 / res
        u32 *b_even = tr_cc;
        tbz(0, S4, 0, 0);
        movimm(S1, 0xBFF0000000000000ULL); // -1.0
        fmov_d_x(28, S1);
        fdiv_d(V_S0, 28, V_S0);
        *b_even |= ((u32)(tr_cc - b_even) & 0x3fffu) << 5;
        // Store tan(x) in ST(0), clear C2 (bit 26), push 1.0
        emit_store_cached_d(V_S0, S0);
        movimm(S1, 1u << 26);
        logic_reg(0, 0, 1, S3, S3, S1);
        movimm(S0, 0x3FF0000000000000ULL);
        emit_push_cached_x64(S0, &sl);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 F0: F2XM1 (2^ST(0) - 1, |ST(0)| <= 1.0) in extreme fast double mode
    if (esc == 1 && m == 0xF0) {
        static const u64 kexp_c[14] = {
            0x3FE0000000000000ULL, 0x3FC5555555555555ULL, 0x3FA5555555555555ULL, 0x3F81111111111111ULL,
            0x3F56C16C16C16C17ULL, 0x3F2A01A01A01A01AULL, 0x3EFA01A01A01A01AULL, 0x3EC71DE3A556C734ULL,
            0x3E927E4FB7789F5CULL, 0x3E5AE64567F544E4ULL, 0x3E21EED8EFF8D898ULL, 0x3DE6124613A86D09ULL,
            0x3DA93974A8C07C9DULL, 0x3D6AE7F3E733B81FULL,
        };
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S1, V_S0);
        ubfx(1, S2, S1, 0, 63); // |x| bits
        movimm(S5, 0x3FF0000000000000ULL); // 1.0
        add_reg(1, 1, 1, XZR, S5, S2, 0);
        cset(S1, 3); // CC: |x| > 1.0 or NaN/Inf -> slow
        add_cbz(&sl, 1, 0, S1);
        movimm(S1, 0x3FE62E42FEFA39EFULL); // ln(2)
        fmov_d_x(28, S1);
        fmul_d(V_S0, V_S0, 28); // u = x * ln(2)
        fmul_d(V_S1, V_S0, V_S0); // u^2
        movimm(S1, kexp_c[13]);
        fmov_d_x(V_S2, S1);
        for (int ci = 12; ci >= 0; ci--) {
            movimm(S1, kexp_c[ci]);
            fmov_d_x(28, S1);
            fmadd_d(V_S2, V_S2, V_S0, 28);
        }
        fmadd_d(V_S0, V_S1, V_S2, V_S0); // u + u^2 * p(u)
        emit_store_cached_d(V_S0, S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 FB (FSINCOS) / D9 FE (FSIN) / D9 FF (FCOS) in extreme fast double mode
    if (esc == 1 && (m == 0xFB || m == 0xFE || m == 0xFF)) {
        static const u64 ksin_c[7] = {
            0xBFC5555555555555ULL, 0x3F81111111111111ULL, 0xBF2A01A01A01A01AULL,
            0x3EC71DE3A556C734ULL, 0xBE5AE64567F544E4ULL, 0x3DE6124613A86D09ULL,
            0xBD6AE7F3E733B81FULL,
        };
        static const u64 kcos_c[7] = {
            0xBFE0000000000000ULL, 0x3FA5555555555555ULL, 0xBF56C16C16C16C17ULL,
            0x3EFA01A01A01A01AULL, 0xBE927E4FB7789F5CULL, 0x3E21EED8EFF8D898ULL,
            0xBDA93974A8C07C9DULL,
        };
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        if (m == 0xFB) {
            // Check stack not full (ST(7) must be empty before pushing cos)
            add_imm(0, 0, 0, S1, S3, 7);
            ubfx(0, S1, S1, 0, 3);
            lsr_imm(0, S2, S3, 8);
            var_shift(0, 1, S2, S2, S1);
            add_tbz(&sl, 1, S2, 0);
        }
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S1, V_S0);
        ubfx(1, S2, S1, 0, 63);
        movimm(S5, 0x4090000000000000ULL); // 1024.0
        add_reg(1, 1, 1, XZR, S2, S5, 0);
        cset(S1, 2); // CS: >= 1024.0 or NaN/Inf -> slow
        add_cbz(&sl, 1, 0, S1);
        // Cody-Waite range reduction: k = round(x * (2/pi)), r = (x - k*PIO2_HI) - k*PIO2_LO
        movimm(S1, 0x3FE45F306DC9C883ULL); // 2/pi
        fmov_d_x(28, S1);
        fmul_d(V_S1, V_S0, 28);
        frintn_d(V_S1, V_S1);
        fcvtzs_w_d(S4, V_S1); // int32 k in S4
        movimm(S1, 0x3FF921FB54400000ULL); // PIO2_HI
        fmov_d_x(28, S1);
        fmsub_d(V_S0, V_S1, 28, V_S0);
        movimm(S1, 0x3DD0B4611A626331ULL); // PIO2_LO
        fmov_d_x(28, S1);
        fmsub_d(V_S0, V_S1, 28, V_S0); // V_S0 = r
        fmul_d(V_S1, V_S0, V_S0);      // V_S1 = r^2
        fmul_d(27, V_S0, V_S1);        // v27 = r^3
        // sin(r) -> V_S2
        movimm(S1, ksin_c[6]);
        fmov_d_x(V_S2, S1);
        for (int ci = 5; ci >= 0; ci--) {
            movimm(S1, ksin_c[ci]);
            fmov_d_x(28, S1);
            fmadd_d(V_S2, V_S2, V_S1, 28);
        }
        fmadd_d(V_S2, 27, V_S2, V_S0); // V_S2 = s = sin(r)
        // cos(r) -> 29
        movimm(S1, kcos_c[6]);
        fmov_d_x(29, S1);
        for (int ci = 5; ci >= 0; ci--) {
            movimm(S1, kcos_c[ci]);
            fmov_d_x(28, S1);
            fmadd_d(29, 29, V_S1, 28);
        }
        movimm(S1, 0x3FF0000000000000ULL); // 1.0
        fmov_d_x(28, S1);
        fmadd_d(29, V_S1, 29, 28); // v29 = c = cos(r)
        // Quadrant adjustment using k in S4:
        // if (k & 1): (s, c) = (c, -s)
        u32 *b_k0 = tr_cc;
        tbz(0, S4, 0, 0);
        fneg_d(28, V_S2);          // v28 = -s
        E(0x1E604000u | (29u << 5) | V_S2); // fmov d(V_S2), d29  (s = c)
        E(0x1E604000u | (28u << 5) | 29u);  // fmov d29, d28      (c = -s)
        *b_k0 |= ((u32)(tr_cc - b_k0) & 0x3fffu) << 5;
        // if (k & 2): (s, c) = (-s, -c)
        u32 *b_k1 = tr_cc;
        tbz(0, S4, 1, 0);
        fneg_d(V_S2, V_S2);
        fneg_d(29, 29);
        *b_k1 |= ((u32)(tr_cc - b_k1) & 0x3fffu) << 5;
        // Clear C2 (bit 26) in S3
        movimm(S1, 1u << 26);
        logic_reg(0, 0, 1, S3, S3, S1);
        if (m == 0xFE) {
            emit_store_cached_d(V_S2, S0);
            ldst(0, 4, S3, (Addr){.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_top)});
        } else if (m == 0xFF) {
            emit_store_cached_d(29, S0);
            ldst(0, 4, S3, (Addr){.base = CTX, .imm = __builtin_offsetof(EngineCtx, fp_top)});
        } else {
            // FSINCOS: store sin in ST(0), push cos onto new ST(0) (updates fp_top/sw)
            emit_store_cached_d(V_S2, S0);
            fmov_x_d(S0, 29);
            emit_push_cached_x64(S0, &sl);
        }
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 FA (FSQRT) / D9 FC (FRNDINT)
    if (esc == 1 && (m == 0xFA || m == 0xFC)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        if (m == 0xFA)
            fsqrt_d(V_S0, V_S0);
        else
            emit_fcw_round_dispatch(2);
        emit_store_cached_d(V_S0, S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 C0..C7: FLD ST(i)
    if (esc == 1 && reg == 0) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, rm, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S0, V_S0);
        emit_push_cached_x64(S0, &sl);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 C8..CF: FXCH ST(i)
    if (esc == 1 && reg == 1) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, rm, &sl);
        if (rm != 0) {
            emit_load_cached_d(S0, V_S0, &sl);
            emit_load_cached_d(AD, V_S1, &sl);
            fmov_x_d(S4, V_S1);
            fmov_x_d(S5, V_S0);
            add_reg(1, 0, 0, S1, CTX, S0, 3);
            ldst(0, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
            add_reg(1, 0, 0, S1, CTX, AD, 3);
            ldst(0, 8, S5, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
        }
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D9 E0 (FCHS) / D9 E1 (FABS)
    if (esc == 1 && (m == 0xE0 || m == 0xE1)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        fmov_x_d(S4, V_S0);
        movimm(S5, 1ULL << 63);
        if (m == 0xE0)
            logic_reg(1, 2, 0, S4, S4, S5);
        else
            logic_reg(1, 0, 1, S4, S4, S5);
        add_reg(1, 0, 0, S1, CTX, S0, 3);
        ldst(0, 8, S4, (Addr){.base = S1, .imm = offsetof(EngineCtx, fp_d)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // DD C0..C7: FFREE ST(i)
    if (esc == 5 && reg == 0) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        add_imm(0, 0, 0, S0, S3, (u32)rm);
        ubfx(0, S0, S0, 0, 3);
        emit_clear_cached_bit(S0);
        movimm(S1, 1);
        var_shift(0, 0, S1, S1, S0);
        lsl_imm(0, S1, S1, 8);
        logic_reg(0, 0, 1, S3, S3, S1);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // DD D0..DF: FST/FSTP ST(i)
    if (esc == 5 && (reg == 2 || reg == 3)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        if (rm != 0) {
            emit_load_cached_d(S0, V_S0, &sl);
            add_imm(0, 0, 0, AD, S3, (u32)rm);
            ubfx(0, AD, AD, 0, 3);
            emit_store_cached_d(V_S0, AD);
            movimm(S1, 1);
            var_shift(0, 0, S1, S1, AD);
            lsl_imm(0, S1, S1, 8);
            logic_reg(0, 1, 0, S3, S3, S1);
            if (reg == 2)
                ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        }
        if (reg == 3)
            emit_pop_st0(S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // DE D9 (FCOMPP) / DA E9 (FUCOMPP)
    if ((esc == 6 && m == 0xD9) || (esc == 2 && m == 0xE9)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, 1, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        emit_fcom_body(S0, true); // pops ST(0), S3 updated
        emit_pop_st0(AD);         // pops ST(1)
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // DB E8..F7 (FUCOMI/FCOMI) & DF E8..F7 (FUCOMIP/FCOMIP)
    if ((esc == 3 || esc == 7) && (reg == 5 || reg == 6)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, rm, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        fcmp_d(V_S0, V_S1);
        movimm(S4, 0x20000000u); // > : C=1, Z=0
        movimm(S1, 0x00000000u);
        csel(0, S4, S1, S4, 4);  // MI (<) : C=0, Z=0
        movimm(S1, 0x60000000u);
        csel(0, S4, S1, S4, 0);  // EQ (==) : C=1, Z=1
        movimm(S1, 0x40000000u);
        csel(0, S4, S1, S4, 6);  // VS (unordered) : C=0, Z=1
        cset(PFR, 7);            // VC -> 1 (PF=0), VS -> 0 (PF=1)
        mov_reg(1, AFR, PFR);
        msr_nzcv(S4);
        if (esc == 7)
            emit_pop_st0(S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D8 D0..DF (FCOM/FCOMP ST(i)) and DD E0..EF (FUCOM/FUCOMP ST(i))
    if ((esc == 0 && (reg == 2 || reg == 3)) || (esc == 5 && (reg == 4 || reg == 5))) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, rm, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        emit_fcom_body(S0, (reg & 1) != 0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }

    // D8 / DC / DE register-register binary arithmetic
    if ((esc == 0 || esc == 4 || esc == 6) && (reg == 0 || reg == 1 || reg >= 4)) {
        SlowList sl = {.n = 0};
        emit_x87_guard(&sl);
        emit_st_phys_nonempty(S0, 0, &sl);
        emit_st_phys_nonempty(AD, rm, &sl);
        emit_load_cached_d(S0, V_S0, &sl);
        emit_load_cached_d(AD, V_S1, &sl);
        int dst_phys = (esc == 0) ? S0 : AD;
        if (esc == 0) {
            switch (reg) {
                case 0: fadd_d(V_S0, V_S0, V_S1); break;
                case 1: fmul_d(V_S0, V_S0, V_S1); break;
                case 4: fsub_d(V_S0, V_S0, V_S1); break;
                case 5: fsub_d(V_S0, V_S1, V_S0); break;
                case 6: fdiv_d(V_S0, V_S0, V_S1); break;
                case 7: fdiv_d(V_S0, V_S1, V_S0); break;
            }
        } else {
            switch (reg) {
                case 0: fadd_d(V_S0, V_S1, V_S0); break;
                case 1: fmul_d(V_S0, V_S1, V_S0); break;
                case 4: fsub_d(V_S0, V_S0, V_S1); break;
                case 5: fsub_d(V_S0, V_S1, V_S0); break;
                case 6: fdiv_d(V_S0, V_S0, V_S1); break;
                case 7: fdiv_d(V_S0, V_S1, V_S0); break;
            }
        }
        emit_store_cached_d(V_S0, dst_phys);
        if (esc == 6)
            emit_pop_st0(S0);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        patch_b(done);
        return true;
    }
    return false;
}

void emit_x87(const Insn *in) {
    if (in->rm_reg) {
        if (try_fast_reg(in))
            return;
        call_monitor(in);
        if (in->sub == 7 && in->opc == 0xE0)
            gpr_write(0, 2, false, S0);
        return;
    }
    int reg = (in->opc >> 3) & 7, kind = mem_kind(in->sub, reg);
    if (kind == NONE) {
        emit_stop(in->rip, "x87: reserved memory form");
        return;
    }
    E(0x4F00E400u | V_MEM); // movi v30.16b, #0: TS gives #NM ahead of any fault on the operand
    int r = addr_reg(in);
    if (r != AD)
        mov_reg(1, AD, r);
    Addr a = {.base = AD}, hi = {.base = AD, .imm = 8};
    switch (kind) {
        case LOAD16:
            ldst(1, 2, S0, a);
            break;
        case LOAD32:
            ldst(1, 4, S0, a);
            break;
        case LOAD64:
            ldst(1, 8, S0, a);
            break;
        case LOAD80:
            ldst(1, 8, S0, a);
            ldst(1, 2, S1, hi);
            break;
        case AREA_READ:
        case AREA_WRITE: {
            int bytes = in->sub == 1 ? (in->pfx == 0x66 ? 14 : 28) : (in->pfx == 0x66 ? 94 : 108);
            Addr last = {.base = AD, .imm = bytes - 1};
            ldst(1, 1, S0, a);
            ldst(1, 1, S1, last);
            if (kind == AREA_WRITE) {
                ldst(0, 1, S0, a);
                ldst(0, 1, S1, last);
            }
            break;
        }
        default:
            break;
    }
    if ((in->sub == 1 || in->sub == 5) && reg == 0) {
        load_fast_fp(in, in->sub == 1);
        return;
    }
    if (in->sub == 3 && reg == 5) { // fld m80
        load_m80_fast(in);
        return;
    }
    if (in->sub == 3 && reg == 7) { // fstp m80
        store_m80_fast(in, a, hi);
        return;
    }
    if (in->sub == 1 && (reg == 5 || reg == 7)) { // fldcw / fnstcw
        fcw_fast(in, a, reg == 7);
        return;
    }
    if (in->sub == 5 && reg == 7) { // fnstsw m16
        SlowList sl = {.n = 0};
        ldst(1, 8, S1, (Addr){.base = CTX, .imm = offsetof(EngineCtx, cr0)});
        movimm(S5, 0xC);
        logic_reg(1, 0, 0, S1, S1, S5);
        add_cbz(&sl, 1, 1, S1);
        ldst(1, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        lsr_imm(0, S0, S3, 16);
        ubfx(0, S1, S3, 0, 3);
        bfi(0, S0, S1, 11, 3);
        ldst(0, 2, S0, a);
        u32 *done = tr_cc;
        b_rel(0);
        patch_slow(&sl);
        call_monitor(in);
        ldst(0, 2, S0, a);
        ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
        patch_b(done);
        return;
    }
    if ((in->sub == 3 || in->sub == 7) && reg == 0) { // fild m32 / m16
        fild_int32_fast(in, in->sub == 7);
        return;
    }
    if ((in->sub == 3 || in->sub == 7) && (reg == 2 || reg == 3)) { // fist / fistp m32 / m16
        fist_int_fast(in, a, in->sub == 7, reg == 3);
        return;
    }
    if (in->sub == 7 && reg == 5) { // fild m64
        fild_m64_fast(in);
        return;
    }
    if (in->sub == 7 && reg == 7) { // fistp m64
        fistp_m64_fast(in, a);
        return;
    }
    if ((in->sub == 5 || in->sub == 1) && (reg == 2 || reg == 3)) { // fst / fstp m64 / m32
        store_fast_fp(in, a, in->sub == 1, reg == 3);
        return;
    }
    if (in->sub == 4 || in->sub == 0 || in->sub == 2 || in->sub == 6) { // DC m64, D8 m32, DA int32, DE int16
        int km = in->sub == 4 ? 0 : in->sub == 0 ? 1 : in->sub == 2 ? 2 : 3;
        arith_mem_fast(in, reg, km);
        return;
    }
    call_monitor(in);
    if (kind < STORE16)
        return;
    switch (kind) {
        case STORE16:
            ldst(0, 2, S0, a);
            break;
        case STORE32:
            ldst(0, 4, S0, a);
            break;
        case STORE64:
            ldst(0, 8, S0, a);
            break;
        case STORE80:
            ldst(0, 8, S0, a);
            ldst(0, 2, S1, hi);
            break;
        default:
            break;
    }
    ldst(0, 4, S3, (Addr){.base = CTX, .imm = offsetof(EngineCtx, fp_top)});
}
