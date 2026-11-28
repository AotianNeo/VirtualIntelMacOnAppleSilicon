// Extended x86-64 ISA translator backed by Zydis v4.1.1 decoder and Box64's MIT arm64_emitter.h.
// Covers Alder Lake scalar/crypto/vector extensions (XGETBV, BMI1, BMI2, ADX, SHA-NI, AVX/AVX2/FMA).
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tr_internal.h"
#include <Zydis.h>
#include "arm64_emitter.h"

_Static_assert(ZYDIS_MNEMONIC_MAX_VALUE <= 65535, "ZydisMnemonic must fit in u16");

#define RRR(base, rd, rn, rm) E((u32)(base) | (u32)(rm) << 16 | (u32)(rn) << 5 | (u32)(rd))
#define RR(base, rd, rn) E((u32)(base) | (u32)(rn) << 5 | (u32)(rd))

int convert_bitmask(uint64_t bitmask) {
    u32 enc;
    return logimm(bitmask, 1, &enc) ? (int)enc : 0;
}

static _Thread_local ZydisDecoder dec64, dec32;
static _Thread_local bool dec_inited;

static void ensure_decoders(void) {
    if (__builtin_expect(dec_inited, 1))
        return;
    ZydisDecoderInit(&dec64, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    ZydisDecoderInit(&dec32, ZYDIS_MACHINE_MODE_LONG_COMPAT_32, ZYDIS_STACK_WIDTH_32);
    dec_inited = true;
}

static const ZydisDecoder *active_decoder(void) {
    ensure_decoders();
    return tr_compat ? &dec32 : &dec64;
}

static int zydis_gpr_idx(ZydisRegister r, u8 *sz, bool *high) {
    if (high)
        *high = false;
    if (r >= ZYDIS_REGISTER_RAX && r <= ZYDIS_REGISTER_R15) {
        if (sz) *sz = 8;
        return (int)(r - ZYDIS_REGISTER_RAX);
    }
    if (r >= ZYDIS_REGISTER_EAX && r <= ZYDIS_REGISTER_R15D) {
        if (sz) *sz = 4;
        return (int)(r - ZYDIS_REGISTER_EAX);
    }
    if (r >= ZYDIS_REGISTER_AX && r <= ZYDIS_REGISTER_R15W) {
        if (sz) *sz = 2;
        return (int)(r - ZYDIS_REGISTER_AX);
    }
    if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_R15B) {
        if (r >= ZYDIS_REGISTER_AH && r <= ZYDIS_REGISTER_BH) {
            if (sz) *sz = 1;
            if (high) *high = true;
            return (int)(r - ZYDIS_REGISTER_AH) + 4;
        }
        if (sz) *sz = 1;
        return (int)(r - ZYDIS_REGISTER_AL);
    }
    return -1;
}

static int zydis_vec_idx(ZydisRegister r, u8 *bytes) {
    if (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15) {
        if (bytes) *bytes = 16;
        return (int)(r - ZYDIS_REGISTER_XMM0);
    }
    if (r >= ZYDIS_REGISTER_YMM0 && r <= ZYDIS_REGISTER_YMM15) {
        if (bytes) *bytes = 32;
        return (int)(r - ZYDIS_REGISTER_YMM0);
    }
    return -1;
}

// YMM upper 128-bit state. ymm0-7 highs stay in v16-v23 across translated blocks.
// Monitor/vectors.s spills those eight regs to EngineCtx.ymm_hi on every EL0
// exception and reloads them in mon_enter_guest. ymm8-15 live in EngineCtx.ymm_hi
// across blocks, and are dynamically cached within blocks in {v24, v25, v29, v31}.
static inline u32 ymm_hi_off(int reg) {
    return (u32)(offsetof(EngineCtx, ymm_hi) + (size_t)reg * 16);
}

static inline int ymm_resident(int reg) {
    return reg < 8 ? 16 + reg : -1;
}

#define YMM_CACHE_SLOTS 4
static const u8 ymm_pool_vreg[YMM_CACHE_SLOTS] = {24, 25, 29, 31};
typedef struct {
    int8_t xreg; // 8..15, or -1 if empty
    bool dirty;
    u32 age;
} YmmCacheSlot;
static _Thread_local YmmCacheSlot ymm_cache[YMM_CACHE_SLOTS];
static _Thread_local u32 ymm_cache_clk;
static _Thread_local u16 ymm_zero_mask; // bit r (0..15): architectural ymm_hi[r] is known all-zero
static _Thread_local int ymm_dirty_run_cnt;
static _Thread_local u8 ymm_pinned_mask; // bit s (0..3): Preheader-pinned slot (never evicted/dropped inside loop)
static _Thread_local u8 ymm_t2_resident_mask; // bit s (0..3): Tier-2 full RW resident slot (trap-synced via T2TrapMeta)
static _Thread_local int ymm_dirty_run_limit = 8;

void ext_ymm_cache_reset(void) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        ymm_cache[s].xreg = -1;
        ymm_cache[s].dirty = false;
        ymm_cache[s].age = 0;
    }
    ymm_zero_mask = 0;
    ymm_dirty_run_cnt = 0;
    ymm_pinned_mask = 0;
    ymm_t2_resident_mask = 0;
    ymm_dirty_run_limit = 8;
}

void ext_ymm_cache_reset_unpinned(void) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_pinned_mask & (1u << s))
            continue;
        ymm_cache[s].xreg = -1;
        ymm_cache[s].dirty = false;
        ymm_cache[s].age = 0;
    }
    ymm_zero_mask = 0;
    ymm_dirty_run_cnt = 0;
}

void ext_ymm_set_dirty_limit(int limit) {
    ymm_dirty_run_limit = limit > 0 ? limit : 8;
}

void ext_ymm_cache_pin_ro(int xreg) {
    if (!tr_opt_ymm_cache || xreg < 8 || xreg > 15)
        return;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_cache[s].xreg == (int8_t)xreg) {
            ymm_pinned_mask |= (u8)(1u << s);
            return;
        }
    }
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (!(ymm_pinned_mask & (1u << s)) && ymm_cache[s].xreg < 0) {
            int vreg = ymm_pool_vreg[s];
            VLDR128_U12(vreg, CTX, ymm_hi_off(xreg));
            ymm_cache[s].xreg = (int8_t)xreg;
            ymm_cache[s].dirty = false;
            ymm_cache[s].age = ++ymm_cache_clk;
            ymm_pinned_mask |= (u8)(1u << s);
            return;
        }
    }
}

void ext_ymm_cache_pin_rw(int xreg) {
    if (!tr_opt_ymm_cache || xreg < 8 || xreg > 15)
        return;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_cache[s].xreg == (int8_t)xreg) {
            ymm_pinned_mask |= (u8)(1u << s);
            ymm_t2_resident_mask |= (u8)(1u << s);
            ymm_cache[s].dirty = false;
            return;
        }
    }
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (!(ymm_pinned_mask & (1u << s)) && ymm_cache[s].xreg < 0) {
            int vreg = ymm_pool_vreg[s];
            // Always preload entry value so a #PF/trap before the first write in Iter 0 reconstructs exact state.
            VLDR128_U12(vreg, CTX, ymm_hi_off(xreg));
            ymm_cache[s].xreg = (int8_t)xreg;
            ymm_cache[s].dirty = false;
            ymm_cache[s].age = ++ymm_cache_clk;
            ymm_pinned_mask |= (u8)(1u << s);
            ymm_t2_resident_mask |= (u8)(1u << s);
            return;
        }
    }
}

void ext_ymm_cache_get_resident_map(u8 out_regs[4]) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        out_regs[s] = ((ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg >= 8)
                          ? (u8)ymm_cache[s].xreg
                          : 0;
    }
}

void ext_ymm_emit_resident_flush(u8 wr_mask) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if ((ymm_t2_resident_mask & (1u << s)) && (wr_mask & (1u << s)) && ymm_cache[s].xreg >= 8)
            VSTR128_U12(ymm_pool_vreg[s], CTX, ymm_hi_off(ymm_cache[s].xreg));
    }
}

void ext_ymm_cache_preload_clean(int xreg) {
    if (!tr_opt_ymm_cache || xreg < 8 || xreg > 15)
        return;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++)
        if (ymm_cache[s].xreg == (int8_t)xreg)
            return;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (!(ymm_pinned_mask & (1u << s)) && ymm_cache[s].xreg < 0) {
            int vreg = ymm_pool_vreg[s];
            VLDR128_U12(vreg, CTX, ymm_hi_off(xreg));
            ymm_cache[s].xreg = (int8_t)xreg;
            ymm_cache[s].dirty = false;
            ymm_cache[s].age = ++ymm_cache_clk;
            return;
        }
    }
}

bool ext_ymm_cache_has_dirty(void) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++)
        if (!(ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg >= 8 && ymm_cache[s].dirty)
            return true;
    return false;
}

int ext_ymm_cache_dirty_run(void) {
    return ymm_dirty_run_cnt;
}

int ext_ymm_cache_snapshot_dirty(u8 vregs[4], u8 xregs[4]) {
    int n = 0;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (!(ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg >= 8 && ymm_cache[s].dirty) {
            vregs[n] = ymm_pool_vreg[s];
            xregs[n] = (u8)ymm_cache[s].xreg;
            n++;
        }
    }
    return n;
}

void ext_ymm_emit_flush_list(int cnt, const u8 vregs[4], const u8 xregs[4]) {
    for (int i = 0; i < cnt; i++)
        VSTR128_U12(vregs[i], CTX, ymm_hi_off(xregs[i]));
}

void ext_ymm_cache_flush_dirty(void) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (!(ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg >= 8 && ymm_cache[s].dirty) {
            VSTR128_U12(ymm_pool_vreg[s], CTX, ymm_hi_off(ymm_cache[s].xreg));
            ymm_cache[s].dirty = false;
        }
    }
    ymm_dirty_run_cnt = 0;
}

void ext_ymm_cache_flush_and_drop(void) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_cache[s].xreg >= 8 && (ymm_cache[s].dirty || (ymm_t2_resident_mask & (1u << s))))
            VSTR128_U12(ymm_pool_vreg[s], CTX, ymm_hi_off(ymm_cache[s].xreg));
        ymm_cache[s].xreg = -1;
        ymm_cache[s].dirty = false;
    }
    ymm_dirty_run_cnt = 0;
    ymm_pinned_mask = 0;
    ymm_t2_resident_mask = 0;
}

static void ymm_cache_evict_vreg(int vreg) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_pool_vreg[s] == (u8)vreg) {
            if (ymm_cache[s].xreg >= 8 && (ymm_cache[s].dirty || (ymm_t2_resident_mask & (1u << s))))
                VSTR128_U12(vreg, CTX, ymm_hi_off(ymm_cache[s].xreg));
            ymm_cache[s].xreg = -1;
            ymm_cache[s].dirty = false;
            ymm_pinned_mask &= (u8)~(1u << s);
            ymm_t2_resident_mask &= (u8)~(1u << s);
            return;
        }
    }
}

static void ymm_cache_drop_xreg(int reg) {
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_cache[s].xreg == (int8_t)reg) {
            if (ymm_t2_resident_mask & (1u << s))
                return;
            ymm_cache[s].xreg = -1;
            ymm_cache[s].dirty = false;
            ymm_pinned_mask &= (u8)~(1u << s);
            return;
        }
    }
}

// Acquire a physical NEON register holding ymm_hi[reg] (0..15).
// For reg < 8 returns v16+reg. For reg >= 8 uses the 4-slot dynamic pool when tr_opt_ymm_cache is enabled.
static int ymm_cache_get(int reg, bool load_if_miss, bool will_write, u8 *pin_mask) {
    int home = ymm_resident(reg);
    if (home >= 0) {
        if (will_write)
            ymm_zero_mask &= (u16)~(1u << reg);
        return home;
    }
    if (!tr_opt_ymm_cache)
        return -1;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_cache[s].xreg == (int8_t)reg) {
            ymm_cache[s].age = ++ymm_cache_clk;
            if (will_write) {
                ymm_zero_mask &= (u16)~(1u << reg);
                if (!(ymm_t2_resident_mask & (1u << s))) {
                    ymm_cache[s].dirty = true;
                    ymm_pinned_mask &= (u8)~(1u << s);
                }
            }
            if (pin_mask)
                *pin_mask |= (u8)(1u << s);
            return ymm_pool_vreg[s];
        }
    }
    int pick = -1;
    for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
        if (ymm_pinned_mask & (1u << s))
            continue;
        if (pin_mask && (*pin_mask & (1u << s)))
            continue;
        if (ymm_cache[s].xreg < 0) {
            pick = s;
            break;
        }
        if (pick < 0 ||
            (!ymm_cache[s].dirty && ymm_cache[pick].dirty) ||
            (ymm_cache[s].dirty == ymm_cache[pick].dirty && ymm_cache[s].age < ymm_cache[pick].age))
            pick = s;
    }
    if (pick < 0)
        return -1;
    int vreg = ymm_pool_vreg[pick];
    if (ymm_cache[pick].xreg >= 8 && ymm_cache[pick].dirty)
        VSTR128_U12(vreg, CTX, ymm_hi_off(ymm_cache[pick].xreg));
    ymm_cache[pick].xreg = (int8_t)reg;
    ymm_cache[pick].dirty = will_write;
    ymm_cache[pick].age = ++ymm_cache_clk;
    if (pin_mask)
        *pin_mask |= (u8)(1u << pick);
    if (load_if_miss) {
        if (ymm_zero_mask & (1u << reg))
            VEORQ(vreg, vreg, vreg);
        else
            VLDR128_U12(vreg, CTX, ymm_hi_off(reg));
    }
    if (will_write)
        ymm_zero_mask &= (u16)~(1u << reg);
    return vreg;
}

static inline void ymm_zero_hi(int reg) {
    if (reg >= 8 && ymm_t2_resident_mask) {
        for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
            if ((ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg == (int8_t)reg) {
                if (!(tr_opt_ymm_cache && (ymm_zero_mask & (1u << reg)))) {
                    int vreg = ymm_pool_vreg[s];
                    VEORQ(vreg, vreg, vreg);
                }
                if (tr_opt_ymm_cache)
                    ymm_zero_mask |= (u16)(1u << reg);
                return;
            }
        }
    }
    if (tr_opt_ymm_cache && (ymm_zero_mask & (1u << reg))) {
        if (reg >= 8)
            ymm_cache_drop_xreg(reg);
        return;
    }
    int home = ymm_resident(reg);
    if (home >= 0) {
        VEORQ(home, home, home);
        if (tr_opt_ymm_cache)
            ymm_zero_mask |= (u16)(1u << reg);
        return;
    }
    ymm_cache_drop_xreg(reg);
    u32 off = ymm_hi_off(reg);
    STRx_U12(XZR, CTX, off);
    STRx_U12(XZR, CTX, off + 8);
    if (tr_opt_ymm_cache)
        ymm_zero_mask |= (u16)(1u << reg);
}

static inline void ymm_load_hi(int vtmp, int reg) {
    int home = ymm_resident(reg);
    if (home >= 0) {
        if (vtmp != home)
            VMOVQ(vtmp, home);
        return;
    }
    if (reg >= 8 && ymm_t2_resident_mask) {
        for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
            if ((ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg == (int8_t)reg) {
                int vreg = ymm_pool_vreg[s];
                if (vtmp != vreg)
                    VMOVQ(vtmp, vreg);
                return;
            }
        }
    }
    if (tr_opt_ymm_cache && (ymm_zero_mask & (1u << reg))) {
        VEORQ(vtmp, vtmp, vtmp);
        return;
    }
    VLDR128_U12(vtmp, CTX, ymm_hi_off(reg));
}

static inline void ymm_store_hi(int vtmp, int reg) {
    ymm_zero_mask &= (u16)~(1u << reg);
    int home = ymm_resident(reg);
    if (home >= 0) {
        if (vtmp != home)
            VMOVQ(home, vtmp);
        return;
    }
    if (reg >= 8 && ymm_t2_resident_mask) {
        for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
            if ((ymm_t2_resident_mask & (1u << s)) && ymm_cache[s].xreg == (int8_t)reg) {
                int vreg = ymm_pool_vreg[s];
                if (vtmp != vreg)
                    VMOVQ(vreg, vtmp);
                return;
            }
        }
    }
    ymm_cache_drop_xreg(reg);
    VSTR128_U12(vtmp, CTX, ymm_hi_off(reg));
}

static inline void vmov128(int rd, int rn) {
    if (rd != rn)
        VMOVQ(rd, rn);
}

static inline void ins_elem(int rd, int di, int rn, int si, int bytes) {
    int ls = log2sz(bytes);
    u32 imm5 = (u32)(di << (ls + 1)) | (1u << ls);
    u32 imm4 = (u32)si << ls;
    E(0x6E000400u | imm5 << 16 | imm4 << 11 | (u32)rn << 5 | (u32)rd);
}

static inline void umov_d(int rd, int rn, int lane) {
    E(0x4E083C00u | (u32)lane << 20 | (u32)rn << 5 | (u32)rd);
}

static inline void ins_x(int v, int lane, int x) {
    E(0x4E081C00u | (u32)lane << 20 | (u32)x << 5 | (u32)v);
}

static inline void fmov_to_d(int v, int x) {
    RR(0x9E670000u, v, x);
}

// Vector memory load/store (kind: 0=8b, 1=16b, 2=32b, 3=64b, 4=128b).
static void ext_vldst(int kind, bool load, int rt, Addr a) {
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

static bool ext_simple_base_disp(const Insn *in) {
    return !in->riprel && !in->no_base && in->no_index && in->seg == 0 && !in->addr32 && !in->addr16 && !tr_compat;
}

static void ext_load256_pair(const Insn *in, int vlo, int vhi) {
    if (ext_simple_base_disp(in)) {
        if (vlo != vhi && in->disp >= -1024 && in->disp <= 1008 && (in->disp & 15) == 0) {
            VLDP128_I7(vlo, vhi, in->base, in->disp);
            return;
        }
        if ((in->disp >= 0 && !(in->disp & 15) && ((in->disp + 16) >> 4) < 4096) ||
            (in->disp >= -256 && in->disp + 16 < 256)) {
            ext_vldst(4, true, vlo, (Addr){.base = in->base, .imm = in->disp});
            ext_vldst(4, true, vhi, (Addr){.base = in->base, .imm = in->disp + 16});
            return;
        }
    }
    int r = addr_reg(in);
    if (vlo != vhi) {
        VLDP128_I7(vlo, vhi, r, 0);
        return;
    }
    ext_vldst(4, true, vlo, (Addr){.base = r, .imm = 0});
    ext_vldst(4, true, vhi, (Addr){.base = r, .imm = 16});
}

static void ext_store256_pair(const Insn *in, int vlo, int vhi) {
    if (ext_simple_base_disp(in)) {
        if (in->disp >= -1024 && in->disp <= 1008 && (in->disp & 15) == 0) {
            VSTP128_I7(vlo, vhi, in->base, in->disp);
            return;
        }
        if ((in->disp >= 0 && !(in->disp & 15) && ((in->disp + 16) >> 4) < 4096) ||
            (in->disp >= -256 && in->disp + 16 < 256)) {
            ext_vldst(4, false, vhi, (Addr){.base = in->base, .imm = in->disp + 16});
            ext_vldst(4, false, vlo, (Addr){.base = in->base, .imm = in->disp});
            return;
        }
    }
    int r = addr_reg(in);
    VSTP128_I7(vlo, vhi, r, 0);
}

static int ext_vec_src128(const Insn *in, int opi, int scratch_v) {
    if (in->ext.op_type[opi] == EXT_OP_VEC)
        return in->ext.op_reg[opi];
    Addr a = addr_prepare(in, 16);
    ext_vldst(4, true, scratch_v, a);
    return scratch_v;
}

static int ext_gpr_src(const Insn *in, int opi, int sz, int scratch_r) {
    if (in->ext.op_type[opi] == EXT_OP_GPR) {
        int r = in->ext.op_reg[opi];
        if (sz == 4) {
            mov_reg(0, scratch_r, r);
            return scratch_r;
        }
        return r;
    }
    Addr a = addr_prepare(in, sz);
    ldst(1, sz, scratch_r, a);
    return scratch_r;
}

static bool ext_populate_mem(const ZydisDecodedInstruction *zi, const ZydisDecodedOperand *op, Insn *in) {
    if (op->type != ZYDIS_OPERAND_TYPE_MEMORY)
        return true;
    const ZydisDecodedOperandMem *m = &op->mem;
    if (m->type != ZYDIS_MEMOP_TYPE_MEM && m->type != ZYDIS_MEMOP_TYPE_VSIB)
        return false;
    in->memop = true;
    in->vsib = m->type == ZYDIS_MEMOP_TYPE_VSIB;
    if (m->segment == ZYDIS_REGISTER_FS)
        in->seg = 1;
    else if (m->segment == ZYDIS_REGISTER_GS)
        in->seg = 2;
    else if (m->segment != ZYDIS_REGISTER_DS && m->segment != ZYDIS_REGISTER_SS &&
             m->segment != ZYDIS_REGISTER_ES && m->segment != ZYDIS_REGISTER_CS &&
             m->segment != ZYDIS_REGISTER_NONE)
        return false;

    if (zi->address_width == 16)
        in->addr16 = tr_compat;
    else if (zi->address_width == 32)
        in->addr32 = true;

    if (m->base == ZYDIS_REGISTER_RIP || m->base == ZYDIS_REGISTER_EIP) {
        in->riprel = true;
    } else if (m->base == ZYDIS_REGISTER_NONE) {
        in->no_base = true;
    } else {
        int b = zydis_gpr_idx(m->base, NULL, NULL);
        if (b < 0)
            return false;
        in->base = (u8)b;
    }

    if (m->index == ZYDIS_REGISTER_NONE) {
        in->no_index = true;
    } else if (in->vsib) {
        int ix = zydis_vec_idx(m->index, NULL);
        if (ix < 0)
            return false;
        in->no_index = false;
        in->index = (u8)ix;
        in->scale = m->scale == 8 ? 3 : m->scale == 4 ? 2 : m->scale == 2 ? 1 : 0;
    } else {
        int ix = zydis_gpr_idx(m->index, NULL, NULL);
        if (ix < 0)
            return false;
        in->no_index = false;
        in->index = (u8)ix;
        in->scale = m->scale == 8 ? 3 : m->scale == 4 ? 2 : m->scale == 2 ? 1 : 0;
    }

    in->disp = m->disp.has_displacement ? m->disp.value : 0;
    return true;
}
static bool is_fma_mnemonic(ZydisMnemonic m, int *order, int *kind, bool *is_scalar, bool *is_double) {
    int ord = 0, k = 0;
    bool sc = false, dbl = false;
    switch (m) {
        #define FMA_CASE(ORD, KIND, SUF, SC, DBL) \
            case ZYDIS_MNEMONIC_V##KIND##ORD##SUF: ord = ORD; k = KIND_ID_##KIND; sc = SC; dbl = DBL; break
        enum { KIND_ID_FMADD = 0, KIND_ID_FMSUB = 1, KIND_ID_FNMADD = 2, KIND_ID_FNMSUB = 3,
               KIND_ID_FMADDSUB = 4, KIND_ID_FMSUBADD = 5 };
        FMA_CASE(132, FMADD, PS, false, false); FMA_CASE(213, FMADD, PS, false, false); FMA_CASE(231, FMADD, PS, false, false);
        FMA_CASE(132, FMADD, PD, false, true);  FMA_CASE(213, FMADD, PD, false, true);  FMA_CASE(231, FMADD, PD, false, true);
        FMA_CASE(132, FMADD, SS, true,  false); FMA_CASE(213, FMADD, SS, true,  false); FMA_CASE(231, FMADD, SS, true,  false);
        FMA_CASE(132, FMADD, SD, true,  true);  FMA_CASE(213, FMADD, SD, true,  true);  FMA_CASE(231, FMADD, SD, true,  true);

        FMA_CASE(132, FMSUB, PS, false, false); FMA_CASE(213, FMSUB, PS, false, false); FMA_CASE(231, FMSUB, PS, false, false);
        FMA_CASE(132, FMSUB, PD, false, true);  FMA_CASE(213, FMSUB, PD, false, true);  FMA_CASE(231, FMSUB, PD, false, true);
        FMA_CASE(132, FMSUB, SS, true,  false); FMA_CASE(213, FMSUB, SS, true,  false); FMA_CASE(231, FMSUB, SS, true,  false);
        FMA_CASE(132, FMSUB, SD, true,  true);  FMA_CASE(213, FMSUB, SD, true,  true);  FMA_CASE(231, FMSUB, SD, true,  true);

        FMA_CASE(132, FNMADD, PS, false, false); FMA_CASE(213, FNMADD, PS, false, false); FMA_CASE(231, FNMADD, PS, false, false);
        FMA_CASE(132, FNMADD, PD, false, true);  FMA_CASE(213, FNMADD, PD, false, true);  FMA_CASE(231, FNMADD, PD, false, true);
        FMA_CASE(132, FNMADD, SS, true,  false); FMA_CASE(213, FNMADD, SS, true,  false); FMA_CASE(231, FNMADD, SS, true,  false);
        FMA_CASE(132, FNMADD, SD, true,  true);  FMA_CASE(213, FNMADD, SD, true,  true);  FMA_CASE(231, FNMADD, SD, true,  true);

        FMA_CASE(132, FNMSUB, PS, false, false); FMA_CASE(213, FNMSUB, PS, false, false); FMA_CASE(231, FNMSUB, PS, false, false);
        FMA_CASE(132, FNMSUB, PD, false, true);  FMA_CASE(213, FNMSUB, PD, false, true);  FMA_CASE(231, FNMSUB, PD, false, true);
        FMA_CASE(132, FNMSUB, SS, true,  false); FMA_CASE(213, FNMSUB, SS, true,  false); FMA_CASE(231, FNMSUB, SS, true,  false);
        FMA_CASE(132, FNMSUB, SD, true,  true);  FMA_CASE(213, FNMSUB, SD, true,  true);  FMA_CASE(231, FNMSUB, SD, true,  true);

        // Even lane subtracts the addend, odd lane adds it. No scalar form.
        FMA_CASE(132, FMADDSUB, PS, false, false); FMA_CASE(213, FMADDSUB, PS, false, false); FMA_CASE(231, FMADDSUB, PS, false, false);
        FMA_CASE(132, FMADDSUB, PD, false, true);  FMA_CASE(213, FMADDSUB, PD, false, true);  FMA_CASE(231, FMADDSUB, PD, false, true);
        FMA_CASE(132, FMSUBADD, PS, false, false); FMA_CASE(213, FMSUBADD, PS, false, false); FMA_CASE(231, FMSUBADD, PS, false, false);
        FMA_CASE(132, FMSUBADD, PD, false, true);  FMA_CASE(213, FMSUBADD, PD, false, true);  FMA_CASE(231, FMSUBADD, PD, false, true);
        #undef FMA_CASE
        default:
            return false;
    }
    if (order) *order = ord;
    if (kind) *kind = k;
    if (is_scalar) *is_scalar = sc;
    if (is_double) *is_double = dbl;
    return true;
}

static bool is_lanewise_vex(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_VADDPS: case ZYDIS_MNEMONIC_VADDPD:
        case ZYDIS_MNEMONIC_VSUBPS: case ZYDIS_MNEMONIC_VSUBPD:
        case ZYDIS_MNEMONIC_VMULPS: case ZYDIS_MNEMONIC_VMULPD:
        case ZYDIS_MNEMONIC_VDIVPS: case ZYDIS_MNEMONIC_VDIVPD:
        case ZYDIS_MNEMONIC_VMINPS: case ZYDIS_MNEMONIC_VMINPD:
        case ZYDIS_MNEMONIC_VMAXPS: case ZYDIS_MNEMONIC_VMAXPD:
        case ZYDIS_MNEMONIC_VSQRTPS: case ZYDIS_MNEMONIC_VSQRTPD:
        case ZYDIS_MNEMONIC_VRSQRTPS: case ZYDIS_MNEMONIC_VRCPPS:
        case ZYDIS_MNEMONIC_VANDPS: case ZYDIS_MNEMONIC_VANDPD: case ZYDIS_MNEMONIC_VPAND:
        case ZYDIS_MNEMONIC_VORPS:  case ZYDIS_MNEMONIC_VORPD:  case ZYDIS_MNEMONIC_VPOR:
        case ZYDIS_MNEMONIC_VXORPS: case ZYDIS_MNEMONIC_VXORPD: case ZYDIS_MNEMONIC_VPXOR:
        case ZYDIS_MNEMONIC_VANDNPS: case ZYDIS_MNEMONIC_VANDNPD: case ZYDIS_MNEMONIC_VPANDN:
        case ZYDIS_MNEMONIC_VPADDB: case ZYDIS_MNEMONIC_VPADDW: case ZYDIS_MNEMONIC_VPADDD: case ZYDIS_MNEMONIC_VPADDQ:
        case ZYDIS_MNEMONIC_VPSUBB: case ZYDIS_MNEMONIC_VPSUBW: case ZYDIS_MNEMONIC_VPSUBD: case ZYDIS_MNEMONIC_VPSUBQ:
        case ZYDIS_MNEMONIC_VPADDSB: case ZYDIS_MNEMONIC_VPADDSW:
        case ZYDIS_MNEMONIC_VPADDUSB: case ZYDIS_MNEMONIC_VPADDUSW:
        case ZYDIS_MNEMONIC_VPSUBSB: case ZYDIS_MNEMONIC_VPSUBSW:
        case ZYDIS_MNEMONIC_VPSUBUSB: case ZYDIS_MNEMONIC_VPSUBUSW:
        case ZYDIS_MNEMONIC_VPMULLW: case ZYDIS_MNEMONIC_VPMULLD:
        case ZYDIS_MNEMONIC_VPMULHW: case ZYDIS_MNEMONIC_VPMULHUW: case ZYDIS_MNEMONIC_VPMULHRSW:
        case ZYDIS_MNEMONIC_VPMULUDQ: case ZYDIS_MNEMONIC_VPMULDQ:
        case ZYDIS_MNEMONIC_VPMADDWD: case ZYDIS_MNEMONIC_VPMADDUBSW:
        case ZYDIS_MNEMONIC_VPAVGB: case ZYDIS_MNEMONIC_VPAVGW:
        case ZYDIS_MNEMONIC_VPMINSB: case ZYDIS_MNEMONIC_VPMINSW: case ZYDIS_MNEMONIC_VPMINSD:
        case ZYDIS_MNEMONIC_VPMAXSB: case ZYDIS_MNEMONIC_VPMAXSW: case ZYDIS_MNEMONIC_VPMAXSD:
        case ZYDIS_MNEMONIC_VPMINUB: case ZYDIS_MNEMONIC_VPMINUW: case ZYDIS_MNEMONIC_VPMINUD:
        case ZYDIS_MNEMONIC_VPMAXUB: case ZYDIS_MNEMONIC_VPMAXUW: case ZYDIS_MNEMONIC_VPMAXUD:
        case ZYDIS_MNEMONIC_VPABSB: case ZYDIS_MNEMONIC_VPABSW: case ZYDIS_MNEMONIC_VPABSD:
        case ZYDIS_MNEMONIC_VPSIGNB: case ZYDIS_MNEMONIC_VPSIGNW: case ZYDIS_MNEMONIC_VPSIGND:
        case ZYDIS_MNEMONIC_VPCMPEQB: case ZYDIS_MNEMONIC_VPCMPEQW: case ZYDIS_MNEMONIC_VPCMPEQD: case ZYDIS_MNEMONIC_VPCMPEQQ:
        case ZYDIS_MNEMONIC_VPCMPGTB: case ZYDIS_MNEMONIC_VPCMPGTW: case ZYDIS_MNEMONIC_VPCMPGTD: case ZYDIS_MNEMONIC_VPCMPGTQ:
        case ZYDIS_MNEMONIC_VPSHUFB: case ZYDIS_MNEMONIC_VPSHUFD: case ZYDIS_MNEMONIC_VPSHUFHW: case ZYDIS_MNEMONIC_VPSHUFLW:
        case ZYDIS_MNEMONIC_VSHUFPS: case ZYDIS_MNEMONIC_VSHUFPD:
        case ZYDIS_MNEMONIC_VPUNPCKLBW: case ZYDIS_MNEMONIC_VPUNPCKLWD: case ZYDIS_MNEMONIC_VPUNPCKLDQ: case ZYDIS_MNEMONIC_VPUNPCKLQDQ:
        case ZYDIS_MNEMONIC_VPUNPCKHBW: case ZYDIS_MNEMONIC_VPUNPCKHWD: case ZYDIS_MNEMONIC_VPUNPCKHDQ: case ZYDIS_MNEMONIC_VPUNPCKHQDQ:
        case ZYDIS_MNEMONIC_VUNPCKLPS: case ZYDIS_MNEMONIC_VUNPCKLPD:
        case ZYDIS_MNEMONIC_VUNPCKHPS: case ZYDIS_MNEMONIC_VUNPCKHPD:
        case ZYDIS_MNEMONIC_VPACKSSWB: case ZYDIS_MNEMONIC_VPACKSSDW:
        case ZYDIS_MNEMONIC_VPACKUSWB: case ZYDIS_MNEMONIC_VPACKUSDW:
        case ZYDIS_MNEMONIC_VPSADBW:
        case ZYDIS_MNEMONIC_VPERMILPS: case ZYDIS_MNEMONIC_VPERMILPD:
        case ZYDIS_MNEMONIC_VPALIGNR:
        case ZYDIS_MNEMONIC_VPSLLDQ: case ZYDIS_MNEMONIC_VPSRLDQ:
        case ZYDIS_MNEMONIC_VPSLLW: case ZYDIS_MNEMONIC_VPSLLD: case ZYDIS_MNEMONIC_VPSLLQ:
        case ZYDIS_MNEMONIC_VPSRLW: case ZYDIS_MNEMONIC_VPSRLD: case ZYDIS_MNEMONIC_VPSRLQ:
        case ZYDIS_MNEMONIC_VPSRAW: case ZYDIS_MNEMONIC_VPSRAD:
        case ZYDIS_MNEMONIC_VPSLLVD: case ZYDIS_MNEMONIC_VPSLLVQ:
        case ZYDIS_MNEMONIC_VPSRLVD: case ZYDIS_MNEMONIC_VPSRLVQ:
        case ZYDIS_MNEMONIC_VPSRAVD:
        case ZYDIS_MNEMONIC_VPBLENDW: case ZYDIS_MNEMONIC_VBLENDPS: case ZYDIS_MNEMONIC_VBLENDPD:
        case ZYDIS_MNEMONIC_VPBLENDVB: case ZYDIS_MNEMONIC_VBLENDVPS: case ZYDIS_MNEMONIC_VBLENDVPD:
        case ZYDIS_MNEMONIC_VHADDPS: case ZYDIS_MNEMONIC_VHADDPD:
        case ZYDIS_MNEMONIC_VHSUBPS: case ZYDIS_MNEMONIC_VHSUBPD:
        case ZYDIS_MNEMONIC_VPHADDW: case ZYDIS_MNEMONIC_VPHADDD: case ZYDIS_MNEMONIC_VPHADDSW:
        case ZYDIS_MNEMONIC_VPHSUBW: case ZYDIS_MNEMONIC_VPHSUBD: case ZYDIS_MNEMONIC_VPHSUBSW:
        case ZYDIS_MNEMONIC_VADDSUBPS: case ZYDIS_MNEMONIC_VADDSUBPD:
        case ZYDIS_MNEMONIC_VDPPS: case ZYDIS_MNEMONIC_VDPPD:
        case ZYDIS_MNEMONIC_VROUNDPS: case ZYDIS_MNEMONIC_VROUNDPD:
        case ZYDIS_MNEMONIC_VCMPPS: case ZYDIS_MNEMONIC_VCMPPD:
        case ZYDIS_MNEMONIC_VMOVSHDUP: case ZYDIS_MNEMONIC_VMOVSLDUP: case ZYDIS_MNEMONIC_VMOVDDUP:
        case ZYDIS_MNEMONIC_VAESENC: case ZYDIS_MNEMONIC_VAESENCLAST:
        case ZYDIS_MNEMONIC_VAESDEC: case ZYDIS_MNEMONIC_VAESDECLAST:
        case ZYDIS_MNEMONIC_VAESIMC: case ZYDIS_MNEMONIC_VAESKEYGENASSIST:
        case ZYDIS_MNEMONIC_VPCLMULQDQ:
            return true;
        default:
            return false;
    }
}

static bool ext_is_supported(const ZydisDecodedInstruction *zi, const ZydisDecodedOperand *ops, Insn *in) {
    if (zi->attributes & ZYDIS_ATTRIB_HAS_LOCK)
        return false;
    for (u8 i = 0; i < zi->operand_count_visible; i++) {
        if (ops[i].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[i].reg.value >= ZYDIS_REGISTER_ZMM0 && ops[i].reg.value <= ZYDIS_REGISTER_ZMM31)
            return false;
    }
    if (is_fma_mnemonic(zi->mnemonic, NULL, NULL, NULL, NULL))
        return true;
    if (is_lanewise_vex(zi->mnemonic))
        return true;
    switch (zi->mnemonic) {
        // Stage 3: XGETBV (XSETBV runs on TCG fallback once at boot so QEMU env->xcr0 stays in sync)
        case ZYDIS_MNEMONIC_XGETBV:
            return true;

        // Stage 4a: BMI1
        case ZYDIS_MNEMONIC_ANDN:
        case ZYDIS_MNEMONIC_BEXTR:
        case ZYDIS_MNEMONIC_BLSI:
        case ZYDIS_MNEMONIC_BLSMSK:
        case ZYDIS_MNEMONIC_BLSR:
            in->writes = F_ARITH;
            return true;

        // Stage 4b: BMI2
        case ZYDIS_MNEMONIC_BZHI:
            in->writes = F_ARITH;
            return true;
        case ZYDIS_MNEMONIC_MULX:
        case ZYDIS_MNEMONIC_RORX:
        case ZYDIS_MNEMONIC_SARX:
        case ZYDIS_MNEMONIC_SHLX:
        case ZYDIS_MNEMONIC_SHRX:
        case ZYDIS_MNEMONIC_PDEP:
        case ZYDIS_MNEMONIC_PEXT:
            return true;

        // Stage 4c: ADX
        case ZYDIS_MNEMONIC_ADCX:
            in->reads = F_C;
            in->writes = F_C;
            return true;
        case ZYDIS_MNEMONIC_ADOX:
            in->reads = F_V;
            in->writes = F_V;
            return true;

        // Stage 4d: SHA-NI
        case ZYDIS_MNEMONIC_SHA1NEXTE:
        case ZYDIS_MNEMONIC_SHA1MSG1:
        case ZYDIS_MNEMONIC_SHA1MSG2:
        case ZYDIS_MNEMONIC_SHA1RNDS4:
        case ZYDIS_MNEMONIC_SHA256MSG1:
        case ZYDIS_MNEMONIC_SHA256MSG2:
        case ZYDIS_MNEMONIC_SHA256RNDS2:
            return true;

        // Stage 5: Special / cross-lane / scalar VEX AVX & AVX2
        case ZYDIS_MNEMONIC_VZEROUPPER:
        case ZYDIS_MNEMONIC_VZEROALL:
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVUPS: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQ: case ZYDIS_MNEMONIC_VMOVNTDQA:
        case ZYDIS_MNEMONIC_VMOVNTPS: case ZYDIS_MNEMONIC_VMOVNTPD:
        case ZYDIS_MNEMONIC_VLDDQU:
        case ZYDIS_MNEMONIC_VMOVSS:  case ZYDIS_MNEMONIC_VMOVSD:
        case ZYDIS_MNEMONIC_VMOVD:   case ZYDIS_MNEMONIC_VMOVQ:
        case ZYDIS_MNEMONIC_VMOVLPS: case ZYDIS_MNEMONIC_VMOVLPD:
        case ZYDIS_MNEMONIC_VMOVHPS: case ZYDIS_MNEMONIC_VMOVHPD:
        case ZYDIS_MNEMONIC_VMOVLHPS: case ZYDIS_MNEMONIC_VMOVHLPS:
        case ZYDIS_MNEMONIC_VADDSS:  case ZYDIS_MNEMONIC_VADDSD:
        case ZYDIS_MNEMONIC_VSUBSS:  case ZYDIS_MNEMONIC_VSUBSD:
        case ZYDIS_MNEMONIC_VMULSS:  case ZYDIS_MNEMONIC_VMULSD:
        case ZYDIS_MNEMONIC_VDIVSS:  case ZYDIS_MNEMONIC_VDIVSD:
        case ZYDIS_MNEMONIC_VMINSS:  case ZYDIS_MNEMONIC_VMINSD:
        case ZYDIS_MNEMONIC_VMAXSS:  case ZYDIS_MNEMONIC_VMAXSD:
        case ZYDIS_MNEMONIC_VSQRTSS: case ZYDIS_MNEMONIC_VSQRTSD:
        case ZYDIS_MNEMONIC_VRSQRTSS: case ZYDIS_MNEMONIC_VRCPSS:
        case ZYDIS_MNEMONIC_VBROADCASTSS: case ZYDIS_MNEMONIC_VBROADCASTSD:
        case ZYDIS_MNEMONIC_VBROADCASTF128: case ZYDIS_MNEMONIC_VBROADCASTI128:
        case ZYDIS_MNEMONIC_VPBROADCASTB: case ZYDIS_MNEMONIC_VPBROADCASTW:
        case ZYDIS_MNEMONIC_VPBROADCASTD: case ZYDIS_MNEMONIC_VPBROADCASTQ:
        case ZYDIS_MNEMONIC_VINSERTF128:  case ZYDIS_MNEMONIC_VINSERTI128:
        case ZYDIS_MNEMONIC_VEXTRACTF128: case ZYDIS_MNEMONIC_VEXTRACTI128:
        case ZYDIS_MNEMONIC_VPERM2F128:   case ZYDIS_MNEMONIC_VPERM2I128:
        case ZYDIS_MNEMONIC_VPERMQ:       case ZYDIS_MNEMONIC_VPERMPD:
        case ZYDIS_MNEMONIC_VPERMD:       case ZYDIS_MNEMONIC_VPERMPS:
        case ZYDIS_MNEMONIC_VPBLENDD:
        case ZYDIS_MNEMONIC_VPEXTRB: case ZYDIS_MNEMONIC_VPEXTRW: case ZYDIS_MNEMONIC_VPEXTRD: case ZYDIS_MNEMONIC_VPEXTRQ:
        case ZYDIS_MNEMONIC_VPINSRB: case ZYDIS_MNEMONIC_VPINSRW: case ZYDIS_MNEMONIC_VPINSRD: case ZYDIS_MNEMONIC_VPINSRQ:
        case ZYDIS_MNEMONIC_VINSERTPS:
        case ZYDIS_MNEMONIC_VEXTRACTPS:
        case ZYDIS_MNEMONIC_VROUNDSS: case ZYDIS_MNEMONIC_VROUNDSD:
        case ZYDIS_MNEMONIC_VCMPSS: case ZYDIS_MNEMONIC_VCMPSD:
        case ZYDIS_MNEMONIC_VLDMXCSR: case ZYDIS_MNEMONIC_VSTMXCSR:
        case ZYDIS_MNEMONIC_VPMASKMOVD: case ZYDIS_MNEMONIC_VPMASKMOVQ:
        case ZYDIS_MNEMONIC_VMASKMOVPS: case ZYDIS_MNEMONIC_VMASKMOVPD:
        case ZYDIS_MNEMONIC_VGATHERDPS: case ZYDIS_MNEMONIC_VGATHERDPD:
        case ZYDIS_MNEMONIC_VGATHERQPS: case ZYDIS_MNEMONIC_VGATHERQPD:
        case ZYDIS_MNEMONIC_VPGATHERDD: case ZYDIS_MNEMONIC_VPGATHERDQ:
        case ZYDIS_MNEMONIC_VPGATHERQD: case ZYDIS_MNEMONIC_VPGATHERQQ:
        case ZYDIS_MNEMONIC_VPHMINPOSUW:
        case ZYDIS_MNEMONIC_VMPSADBW:
        case ZYDIS_MNEMONIC_MPSADBW:
        case ZYDIS_MNEMONIC_VMASKMOVDQU:
        case ZYDIS_MNEMONIC_MASKMOVDQU:
        case ZYDIS_MNEMONIC_VPMOVMSKB: case ZYDIS_MNEMONIC_VMOVMSKPS: case ZYDIS_MNEMONIC_VMOVMSKPD:
        case ZYDIS_MNEMONIC_VPMOVSXBW: case ZYDIS_MNEMONIC_VPMOVSXBD: case ZYDIS_MNEMONIC_VPMOVSXBQ:
        case ZYDIS_MNEMONIC_VPMOVSXWD: case ZYDIS_MNEMONIC_VPMOVSXWQ: case ZYDIS_MNEMONIC_VPMOVSXDQ:
        case ZYDIS_MNEMONIC_VPMOVZXBW: case ZYDIS_MNEMONIC_VPMOVZXBD: case ZYDIS_MNEMONIC_VPMOVZXBQ:
        case ZYDIS_MNEMONIC_VPMOVZXWD: case ZYDIS_MNEMONIC_VPMOVZXWQ: case ZYDIS_MNEMONIC_VPMOVZXDQ:
        case ZYDIS_MNEMONIC_VCVTSI2SS: case ZYDIS_MNEMONIC_VCVTSI2SD:
        case ZYDIS_MNEMONIC_VCVTSS2SI: case ZYDIS_MNEMONIC_VCVTTSS2SI:
        case ZYDIS_MNEMONIC_VCVTSD2SI: case ZYDIS_MNEMONIC_VCVTTSD2SI:
        case ZYDIS_MNEMONIC_VCVTSS2SD: case ZYDIS_MNEMONIC_VCVTSD2SS:
        case ZYDIS_MNEMONIC_VCVTPH2PS: case ZYDIS_MNEMONIC_VCVTPS2PH:
        case ZYDIS_MNEMONIC_VCVTPS2PD: case ZYDIS_MNEMONIC_VCVTPD2PS:
        case ZYDIS_MNEMONIC_VCVTDQ2PD: case ZYDIS_MNEMONIC_VCVTPD2DQ: case ZYDIS_MNEMONIC_VCVTTPD2DQ:
        case ZYDIS_MNEMONIC_VCVTDQ2PS: case ZYDIS_MNEMONIC_VCVTPS2DQ: case ZYDIS_MNEMONIC_VCVTTPS2DQ:
            return true;

        case ZYDIS_MNEMONIC_VPTEST:
        case ZYDIS_MNEMONIC_VTESTPS:
        case ZYDIS_MNEMONIC_VTESTPD:
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
        case ZYDIS_MNEMONIC_VCOMISS:  case ZYDIS_MNEMONIC_VCOMISD:
        case ZYDIS_MNEMONIC_VPCMPISTRI:
            in->writes = F_ARITH;
            return true;

        default:
            return false;
    }
}

// -1: the instruction starts in the last 14 bytes of a page and this fetch did
// not accept it. The next page may be absent, or mapped but not yet the final
// bytes (a 15-byte read then does not end on the boundary). A cached one-byte
// #UD watches only this page, so the store that fills the rest never kills it.
static int ext_reject(u64 rip) {
    if ((rip & 0xfff) > 4096 - 15)
        return -1;
    return 0;
}

static bool ext_fill_from_zydis(const ZydisDecodedInstruction *zi, const ZydisDecodedOperand *ops, Insn *in) {
    in->ext.mnem = (u16)zi->mnemonic;
    u8 cnt = zi->operand_count_visible <= 4 ? zi->operand_count_visible : 4;
    in->ext.op_cnt = cnt;
    for (u8 i = 0; i < 4; i++) {
        in->ext.op_type[i] = EXT_OP_NONE;
        in->ext.op_reg[i] = 0;
        in->ext.op_bytes[i] = 0;
    }
    for (u8 i = 0; i < cnt; i++) {
        if (ops[i].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            u8 vb = 0;
            int vr = zydis_vec_idx(ops[i].reg.value, &vb);
            if (vr >= 0) {
                in->ext.op_type[i] = EXT_OP_VEC;
                in->ext.op_reg[i] = (u8)vr;
                in->ext.op_bytes[i] = vb;
            } else {
                u8 gs = 0;
                int gr = zydis_gpr_idx(ops[i].reg.value, &gs, NULL);
                if (gr < 0)
                    return false;
                in->ext.op_type[i] = EXT_OP_GPR;
                in->ext.op_reg[i] = (u8)gr;
                in->ext.op_bytes[i] = gs;
            }
        } else if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            in->ext.op_type[i] = EXT_OP_MEM;
            in->ext.op_bytes[i] = (u8)(ops[i].size >> 3);
        } else if (ops[i].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            in->ext.op_type[i] = EXT_OP_IMM;
            in->ext.op_bytes[i] = (u8)(ops[i].size >> 3);
            in->imm = (i64)ops[i].imm.value.u;
        } else {
            return false;
        }
    }
    return true;
}

static bool fast_vex_modrm(const u8 **pp, const u8 *end, int R, int X, int B, Insn *in, u8 *out_mod, u8 *out_reg, u8 *out_rm) {
    if (*pp >= end)
        return false;
    u8 m = *(*pp)++;
    u8 mod = m >> 6, rm = m & 7;
    *out_mod = mod;
    *out_reg = (u8)(((m >> 3) & 7) | (R << 3));
    if (mod == 3) {
        *out_rm = (u8)(rm | (B << 3));
        return true;
    }
    in->memop = true;
    in->no_index = true;
    if (rm == 4) {
        if (*pp >= end)
            return false;
        u8 s = *(*pp)++;
        u8 idx = (u8)(((s >> 3) & 7) | (X << 3));
        u8 bs = s & 7;
        if (idx != 4) {
            in->no_index = false;
            in->index = idx;
            in->scale = s >> 6;
        }
        if (bs == 5 && mod == 0) {
            if (*pp + 4 > end)
                return false;
            in->no_base = true;
            i32 d;
            memcpy(&d, *pp, 4);
            in->disp = d;
            *pp += 4;
        } else {
            in->base = (u8)(bs | (B << 3));
        }
    } else if (rm == 5 && mod == 0) {
        if (*pp + 4 > end)
            return false;
        in->riprel = true;
        i32 d;
        memcpy(&d, *pp, 4);
        in->disp = d;
        *pp += 4;
    } else {
        in->base = (u8)(rm | (B << 3));
    }
    if (mod == 1) {
        if (*pp >= end)
            return false;
        in->disp = *(const int8_t *)(*pp)++;
    } else if (mod == 2) {
        if (*pp + 4 > end)
            return false;
        i32 d;
        memcpy(&d, *pp, 4);
        in->disp = d;
        *pp += 4;
    }
    return true;
}

static inline void ext_set_op(ExtDecoded *e, int idx, u8 type, u8 reg, u8 bytes) {
    e->op_type[idx] = type;
    e->op_reg[idx] = reg;
    e->op_bytes[idx] = bytes;
}

// Box64-style direct byte decoder for hot 64-bit VEX (0xC5/0xC4) AVX/AVX2/FMA/BMI1/BMI2 instructions.
// Bypasses ZydisDecoderDecodeFull (~500 ns -> ~15 ns). Falls back to Zydis on any uncommon/invalid/edge encoding.
static bool fast_decode_vex(u64 rip, Insn *in, const u8 *b, int avail) {
    if (tr_compat || avail < 3 || (b[0] != 0xC5 && b[0] != 0xC4))
        return false;
    const u8 *end = b + (avail > 15 ? 15 : avail);
    const u8 *p;
    int R, X, B, W, L, pp, mm, vvvv;
    u8 opc;
    if (b[0] == 0xC5) {
        u8 b1 = b[1];
        R = !(b1 >> 7);
        X = 0; B = 0; W = 0;
        mm = 1;
        vvvv = (~(b1 >> 3)) & 15;
        L = (b1 >> 2) & 1;
        pp = b1 & 3;
        opc = b[2];
        p = b + 3;
    } else {
        if (avail < 4)
            return false;
        u8 b1 = b[1], b2 = b[2];
        R = !(b1 >> 7);
        X = !((b1 >> 6) & 1);
        B = !((b1 >> 5) & 1);
        mm = b1 & 0x1f;
        if (mm < 1 || mm > 3)
            return false;
        W = (b2 >> 7) & 1;
        vvvv = (~(b2 >> 3)) & 15;
        L = (b2 >> 2) & 1;
        pp = b2 & 3;
        opc = b[3];
        p = b + 4;
    }

    u8 vb = L ? 32 : 16;
    Insn tmp = *in;
    tmp.op = OP_EXT;
    tmp.reads = 0;
    tmp.writes = 0;
    tmp.memop = false;
    tmp.riprel = false;
    tmp.no_base = false;
    tmp.no_index = true;
    tmp.vsib = false;
    tmp.seg = 0;
    tmp.addr32 = false;
    tmp.addr16 = false;
    tmp.disp = 0;
    tmp.imm = 0;
    memset(&tmp.ext, 0, sizeof tmp.ext);

    // Special no-ModRM VEX opcodes: VZEROUPPER / VZEROALL
    if (mm == 1 && opc == 0x77) {
        if (pp != 0 || vvvv != 0)
            return false;
        tmp.ext.mnem = (u16)(L ? ZYDIS_MNEMONIC_VZEROALL : ZYDIS_MNEMONIC_VZEROUPPER);
        tmp.ext.op_cnt = 0;
        tmp.next = rip + (u64)(p - b);
        *in = tmp;
        return true;
    }

    u8 mod = 0, reg = 0, rm = 0;
    if (!fast_vex_modrm(&p, end, R, X, B, &tmp, &mod, &reg, &rm))
        return false;
    u8 rm_type = mod == 3 ? EXT_OP_VEC : EXT_OP_MEM;
    u8 rm_reg = mod == 3 ? rm : 0;

    u16 mn = 0;
    bool need_imm = false;
    int form = 0;
    // form: 1=3vec(reg,vvvv,rm), 2=2vec(reg,rm,vvvv==0), 3=2vec_store(rm,reg,vvvv==0),
    //       4=3vec_imm(reg,rm,imm,vvvv==0), 5=4vec_imm(reg,vvvv,rm,imm),
    //       6=custom (already filled tmp.ext)

    if (mm == 1) {
        switch (opc) {
            case 0x10: case 0x11:
                if (pp <= 1) {
                    if (vvvv != 0) return false;
                    mn = pp ? ZYDIS_MNEMONIC_VMOVUPD : ZYDIS_MNEMONIC_VMOVUPS;
                    form = opc == 0x10 ? 2 : 3;
                } else {
                    if (L != 0) return false;
                    mn = pp == 2 ? ZYDIS_MNEMONIC_VMOVSS : ZYDIS_MNEMONIC_VMOVSD;
                    u8 mb = pp == 2 ? 4 : 8;
                    if (mod == 3) {
                        tmp.ext.op_cnt = 3;
                        if (opc == 0x10) {
                            ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                            ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                            ext_set_op(&tmp.ext, 2, EXT_OP_VEC, rm, 16);
                        } else {
                            ext_set_op(&tmp.ext, 0, EXT_OP_VEC, rm, 16);
                            ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                            ext_set_op(&tmp.ext, 2, EXT_OP_VEC, reg, 16);
                        }
                    } else {
                        if (vvvv != 0) return false;
                        tmp.ext.op_cnt = 2;
                        if (opc == 0x10) {
                            ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                            ext_set_op(&tmp.ext, 1, EXT_OP_MEM, 0, mb);
                        } else {
                            ext_set_op(&tmp.ext, 0, EXT_OP_MEM, 0, mb);
                            ext_set_op(&tmp.ext, 1, EXT_OP_VEC, reg, 16);
                        }
                    }
                    form = 6;
                }
                break;
            case 0x28: case 0x29:
                if (pp > 1 || vvvv != 0) return false;
                mn = pp ? ZYDIS_MNEMONIC_VMOVAPD : ZYDIS_MNEMONIC_VMOVAPS;
                form = opc == 0x28 ? 2 : 3;
                break;
            case 0x6F: case 0x7F:
                if ((pp != 1 && pp != 2) || vvvv != 0) return false;
                mn = pp == 1 ? ZYDIS_MNEMONIC_VMOVDQA : ZYDIS_MNEMONIC_VMOVDQU;
                form = opc == 0x6F ? 2 : 3;
                break;
            case 0x6E:
                if (pp != 1 || L != 0 || vvvv != 0) return false;
                mn = W ? ZYDIS_MNEMONIC_VMOVQ : ZYDIS_MNEMONIC_VMOVD;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                ext_set_op(&tmp.ext, 1, mod == 3 ? EXT_OP_GPR : EXT_OP_MEM, rm_reg, W ? 8 : 4);
                form = 6;
                break;
            case 0x7E:
                if (L != 0 || vvvv != 0) return false;
                if (pp == 1) {
                    mn = W ? ZYDIS_MNEMONIC_VMOVQ : ZYDIS_MNEMONIC_VMOVD;
                    tmp.ext.op_cnt = 2;
                    ext_set_op(&tmp.ext, 0, mod == 3 ? EXT_OP_GPR : EXT_OP_MEM, rm_reg, W ? 8 : 4);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, reg, 16);
                    form = 6;
                } else if (pp == 2) {
                    mn = ZYDIS_MNEMONIC_VMOVQ;
                    tmp.ext.op_cnt = 2;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                    ext_set_op(&tmp.ext, 1, rm_type, rm_reg, mod == 3 ? 16 : 8);
                    form = 6;
                } else return false;
                break;
            case 0xD6:
                if (pp != 1 || L != 0 || vvvv != 0) return false;
                mn = ZYDIS_MNEMONIC_VMOVQ;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, rm_type, rm_reg, mod == 3 ? 16 : 8);
                ext_set_op(&tmp.ext, 1, EXT_OP_VEC, reg, 16);
                form = 6;
                break;
            case 0x58: case 0x59: case 0x5C: case 0x5D: case 0x5E: case 0x5F: {
                static const u16 fp_tbl[6][4] = {
                    {ZYDIS_MNEMONIC_VADDPS, ZYDIS_MNEMONIC_VADDPD, ZYDIS_MNEMONIC_VADDSS, ZYDIS_MNEMONIC_VADDSD},
                    {ZYDIS_MNEMONIC_VMULPS, ZYDIS_MNEMONIC_VMULPD, ZYDIS_MNEMONIC_VMULSS, ZYDIS_MNEMONIC_VMULSD},
                    {ZYDIS_MNEMONIC_VSUBPS, ZYDIS_MNEMONIC_VSUBPD, ZYDIS_MNEMONIC_VSUBSS, ZYDIS_MNEMONIC_VSUBSD},
                    {ZYDIS_MNEMONIC_VMINPS, ZYDIS_MNEMONIC_VMINPD, ZYDIS_MNEMONIC_VMINSS, ZYDIS_MNEMONIC_VMINSD},
                    {ZYDIS_MNEMONIC_VDIVPS, ZYDIS_MNEMONIC_VDIVPD, ZYDIS_MNEMONIC_VDIVSS, ZYDIS_MNEMONIC_VDIVSD},
                    {ZYDIS_MNEMONIC_VMAXPS, ZYDIS_MNEMONIC_VMAXPD, ZYDIS_MNEMONIC_VMAXSS, ZYDIS_MNEMONIC_VMAXSD},
                };
                int row = opc == 0x58 ? 0 : opc == 0x59 ? 1 : opc - 0x5A;
                mn = fp_tbl[row][pp];
                if (pp >= 2) {
                    if (L != 0) return false;
                    tmp.ext.op_cnt = 3;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, mod == 3 ? 16 : (pp == 2 ? 4 : 8));
                    form = 6;
                } else {
                    form = 1;
                }
                break;
            }
            case 0x51:
                if (pp <= 1) {
                    if (vvvv != 0) return false;
                    mn = pp ? ZYDIS_MNEMONIC_VSQRTPD : ZYDIS_MNEMONIC_VSQRTPS;
                    form = 2;
                } else {
                    if (L != 0) return false;
                    mn = pp == 2 ? ZYDIS_MNEMONIC_VSQRTSS : ZYDIS_MNEMONIC_VSQRTSD;
                    tmp.ext.op_cnt = 3;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, mod == 3 ? 16 : (pp == 2 ? 4 : 8));
                    form = 6;
                }
                break;
            case 0x52: case 0x53:
                if (pp == 0) {
                    if (vvvv != 0) return false;
                    mn = opc == 0x52 ? ZYDIS_MNEMONIC_VRSQRTPS : ZYDIS_MNEMONIC_VRCPPS;
                    form = 2;
                } else if (pp == 2 && L == 0) {
                    mn = opc == 0x52 ? ZYDIS_MNEMONIC_VRSQRTSS : ZYDIS_MNEMONIC_VRCPSS;
                    tmp.ext.op_cnt = 3;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, mod == 3 ? 16 : 4);
                    form = 6;
                } else return false;
                break;
            case 0x54: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VANDPD : ZYDIS_MNEMONIC_VANDPS; form = 1; break;
            case 0x55: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VANDNPD : ZYDIS_MNEMONIC_VANDNPS; form = 1; break;
            case 0x56: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VORPD : ZYDIS_MNEMONIC_VORPS; form = 1; break;
            case 0x57: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VXORPD : ZYDIS_MNEMONIC_VXORPS; form = 1; break;
            case 0x14: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VUNPCKLPD : ZYDIS_MNEMONIC_VUNPCKLPS; form = 1; break;
            case 0x15: if (pp > 1) return false; mn = pp ? ZYDIS_MNEMONIC_VUNPCKHPD : ZYDIS_MNEMONIC_VUNPCKHPS; form = 1; break;
            case 0x7C: if (pp != 1 && pp != 3) return false; mn = pp == 1 ? ZYDIS_MNEMONIC_VHADDPD : ZYDIS_MNEMONIC_VHADDPS; form = 1; break;
            case 0x7D: if (pp != 1 && pp != 3) return false; mn = pp == 1 ? ZYDIS_MNEMONIC_VHSUBPD : ZYDIS_MNEMONIC_VHSUBPS; form = 1; break;
            case 0xD0: if (pp != 1 && pp != 3) return false; mn = pp == 1 ? ZYDIS_MNEMONIC_VADDSUBPD : ZYDIS_MNEMONIC_VADDSUBPS; form = 1; break;
            case 0x12:
                if (vvvv != 0) return false;
                if (pp == 2) { mn = ZYDIS_MNEMONIC_VMOVSLDUP; form = 2; }
                else if (pp == 3) {
                    mn = ZYDIS_MNEMONIC_VMOVDDUP;
                    tmp.ext.op_cnt = 2;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
                    ext_set_op(&tmp.ext, 1, rm_type, rm_reg, (mod == 3 || L) ? vb : 8);
                    form = 6;
                } else return false;
                break;
            case 0x16:
                if (pp != 2 || vvvv != 0) return false;
                mn = ZYDIS_MNEMONIC_VMOVSHDUP; form = 2;
                break;
            case 0x5B:
                if (pp > 2 || vvvv != 0) return false;
                mn = pp == 0 ? ZYDIS_MNEMONIC_VCVTDQ2PS : pp == 1 ? ZYDIS_MNEMONIC_VCVTPS2DQ : ZYDIS_MNEMONIC_VCVTTPS2DQ;
                form = 2;
                break;
            case 0x2E: case 0x2F:
                if (pp > 1 || L != 0 || vvvv != 0) return false;
                mn = opc == 0x2E ? (pp ? ZYDIS_MNEMONIC_VUCOMISD : ZYDIS_MNEMONIC_VUCOMISS)
                                 : (pp ? ZYDIS_MNEMONIC_VCOMISD : ZYDIS_MNEMONIC_VCOMISS);
                tmp.writes = F_ARITH;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                ext_set_op(&tmp.ext, 1, rm_type, rm_reg, mod == 3 ? 16 : (pp ? 8 : 4));
                form = 6;
                break;
            case 0x50:
                if (pp > 1 || vvvv != 0 || mod != 3) return false;
                mn = pp ? ZYDIS_MNEMONIC_VMOVMSKPD : ZYDIS_MNEMONIC_VMOVMSKPS;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, 4);
                ext_set_op(&tmp.ext, 1, EXT_OP_VEC, rm, vb);
                form = 6;
                break;
            case 0xD7:
                if (pp != 1 || vvvv != 0 || mod != 3) return false;
                mn = ZYDIS_MNEMONIC_VPMOVMSKB;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, 4);
                ext_set_op(&tmp.ext, 1, EXT_OP_VEC, rm, vb);
                form = 6;
                break;
            case 0x70:
                if (pp == 0 || vvvv != 0) return false;
                mn = pp == 1 ? ZYDIS_MNEMONIC_VPSHUFD : pp == 2 ? ZYDIS_MNEMONIC_VPSHUFHW : ZYDIS_MNEMONIC_VPSHUFLW;
                form = 4; need_imm = true;
                break;
            case 0xC6:
                if (pp > 1) return false;
                mn = pp ? ZYDIS_MNEMONIC_VSHUFPD : ZYDIS_MNEMONIC_VSHUFPS;
                form = 5; need_imm = true;
                break;
            case 0xC2:
                if (pp <= 1) {
                    mn = pp ? ZYDIS_MNEMONIC_VCMPPD : ZYDIS_MNEMONIC_VCMPPS;
                    form = 5; need_imm = true;
                } else {
                    if (L != 0 || p >= end) return false;
                    mn = pp == 2 ? ZYDIS_MNEMONIC_VCMPSS : ZYDIS_MNEMONIC_VCMPSD;
                    tmp.imm = *p++;
                    tmp.ext.op_cnt = 4;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, mod == 3 ? 16 : (pp == 2 ? 4 : 8));
                    ext_set_op(&tmp.ext, 3, EXT_OP_IMM, 0, 1);
                    form = 6;
                }
                break;
            case 0x71: case 0x72: case 0x73: {
                if (pp != 1 || mod != 3 || R != 0 || p >= end) return false;
                int sub = reg & 7;
                if (opc == 0x71)
                    mn = sub == 2 ? ZYDIS_MNEMONIC_VPSRLW : sub == 4 ? ZYDIS_MNEMONIC_VPSRAW : sub == 6 ? ZYDIS_MNEMONIC_VPSLLW : 0;
                else if (opc == 0x72)
                    mn = sub == 2 ? ZYDIS_MNEMONIC_VPSRLD : sub == 4 ? ZYDIS_MNEMONIC_VPSRAD : sub == 6 ? ZYDIS_MNEMONIC_VPSLLD : 0;
                else
                    mn = sub == 2 ? ZYDIS_MNEMONIC_VPSRLQ : sub == 3 ? ZYDIS_MNEMONIC_VPSRLDQ :
                         sub == 6 ? ZYDIS_MNEMONIC_VPSLLQ : sub == 7 ? ZYDIS_MNEMONIC_VPSLLDQ : 0;
                if (!mn) return false;
                tmp.imm = *p++;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_VEC, (u8)vvvv, vb);
                ext_set_op(&tmp.ext, 1, EXT_OP_VEC, rm, vb);
                ext_set_op(&tmp.ext, 2, EXT_OP_IMM, 0, 1);
                form = 6;
                break;
            }
            default:
                if (pp != 1) return false;
                switch (opc) {
                    case 0x60: mn = ZYDIS_MNEMONIC_VPUNPCKLBW; break;
                    case 0x61: mn = ZYDIS_MNEMONIC_VPUNPCKLWD; break;
                    case 0x62: mn = ZYDIS_MNEMONIC_VPUNPCKLDQ; break;
                    case 0x63: mn = ZYDIS_MNEMONIC_VPACKSSWB; break;
                    case 0x64: mn = ZYDIS_MNEMONIC_VPCMPGTB; break;
                    case 0x65: mn = ZYDIS_MNEMONIC_VPCMPGTW; break;
                    case 0x66: mn = ZYDIS_MNEMONIC_VPCMPGTD; break;
                    case 0x67: mn = ZYDIS_MNEMONIC_VPACKUSWB; break;
                    case 0x68: mn = ZYDIS_MNEMONIC_VPUNPCKHBW; break;
                    case 0x69: mn = ZYDIS_MNEMONIC_VPUNPCKHWD; break;
                    case 0x6A: mn = ZYDIS_MNEMONIC_VPUNPCKHDQ; break;
                    case 0x6B: mn = ZYDIS_MNEMONIC_VPACKSSDW; break;
                    case 0x6C: mn = ZYDIS_MNEMONIC_VPUNPCKLQDQ; break;
                    case 0x6D: mn = ZYDIS_MNEMONIC_VPUNPCKHQDQ; break;
                    case 0x74: mn = ZYDIS_MNEMONIC_VPCMPEQB; break;
                    case 0x75: mn = ZYDIS_MNEMONIC_VPCMPEQW; break;
                    case 0x76: mn = ZYDIS_MNEMONIC_VPCMPEQD; break;
                    case 0xD4: mn = ZYDIS_MNEMONIC_VPADDQ; break;
                    case 0xD5: mn = ZYDIS_MNEMONIC_VPMULLW; break;
                    case 0xD8: mn = ZYDIS_MNEMONIC_VPSUBUSB; break;
                    case 0xD9: mn = ZYDIS_MNEMONIC_VPSUBUSW; break;
                    case 0xDA: mn = ZYDIS_MNEMONIC_VPMINUB; break;
                    case 0xDB: mn = ZYDIS_MNEMONIC_VPAND; break;
                    case 0xDC: mn = ZYDIS_MNEMONIC_VPADDUSB; break;
                    case 0xDD: mn = ZYDIS_MNEMONIC_VPADDUSW; break;
                    case 0xDE: mn = ZYDIS_MNEMONIC_VPMAXUB; break;
                    case 0xDF: mn = ZYDIS_MNEMONIC_VPANDN; break;
                    case 0xE0: mn = ZYDIS_MNEMONIC_VPAVGB; break;
                    case 0xE3: mn = ZYDIS_MNEMONIC_VPAVGW; break;
                    case 0xE4: mn = ZYDIS_MNEMONIC_VPMULHUW; break;
                    case 0xE5: mn = ZYDIS_MNEMONIC_VPMULHW; break;
                    case 0xE8: mn = ZYDIS_MNEMONIC_VPSUBSB; break;
                    case 0xE9: mn = ZYDIS_MNEMONIC_VPSUBSW; break;
                    case 0xEA: mn = ZYDIS_MNEMONIC_VPMINSW; break;
                    case 0xEB: mn = ZYDIS_MNEMONIC_VPOR; break;
                    case 0xEC: mn = ZYDIS_MNEMONIC_VPADDSB; break;
                    case 0xED: mn = ZYDIS_MNEMONIC_VPADDSW; break;
                    case 0xEE: mn = ZYDIS_MNEMONIC_VPMAXSW; break;
                    case 0xEF: mn = ZYDIS_MNEMONIC_VPXOR; break;
                    case 0xF4: mn = ZYDIS_MNEMONIC_VPMULUDQ; break;
                    case 0xF5: mn = ZYDIS_MNEMONIC_VPMADDWD; break;
                    case 0xF6: mn = ZYDIS_MNEMONIC_VPSADBW; break;
                    case 0xF8: mn = ZYDIS_MNEMONIC_VPSUBB; break;
                    case 0xF9: mn = ZYDIS_MNEMONIC_VPSUBW; break;
                    case 0xFA: mn = ZYDIS_MNEMONIC_VPSUBD; break;
                    case 0xFB: mn = ZYDIS_MNEMONIC_VPSUBQ; break;
                    case 0xFC: mn = ZYDIS_MNEMONIC_VPADDB; break;
                    case 0xFD: mn = ZYDIS_MNEMONIC_VPADDW; break;
                    case 0xFE: mn = ZYDIS_MNEMONIC_VPADDD; break;
                    default: return false;
                }
                form = 1;
                break;
        }
    } else if (mm == 2) {
        if (pp == 1 && opc >= 0x96 && opc <= 0xBF) {
            int grp = (opc - 0x90) >> 4; // 0=132, 1=213, 2=231
            int lo = opc & 0xf;
            if (lo < 6) return false;
            static const u16 fma_ps[3][10] = {
                {ZYDIS_MNEMONIC_VFMADDSUB132PS, ZYDIS_MNEMONIC_VFMSUBADD132PS, ZYDIS_MNEMONIC_VFMADD132PS, ZYDIS_MNEMONIC_VFMADD132SS,
                 ZYDIS_MNEMONIC_VFMSUB132PS, ZYDIS_MNEMONIC_VFMSUB132SS, ZYDIS_MNEMONIC_VFNMADD132PS, ZYDIS_MNEMONIC_VFNMADD132SS,
                 ZYDIS_MNEMONIC_VFNMSUB132PS, ZYDIS_MNEMONIC_VFNMSUB132SS},
                {ZYDIS_MNEMONIC_VFMADDSUB213PS, ZYDIS_MNEMONIC_VFMSUBADD213PS, ZYDIS_MNEMONIC_VFMADD213PS, ZYDIS_MNEMONIC_VFMADD213SS,
                 ZYDIS_MNEMONIC_VFMSUB213PS, ZYDIS_MNEMONIC_VFMSUB213SS, ZYDIS_MNEMONIC_VFNMADD213PS, ZYDIS_MNEMONIC_VFNMADD213SS,
                 ZYDIS_MNEMONIC_VFNMSUB213PS, ZYDIS_MNEMONIC_VFNMSUB213SS},
                {ZYDIS_MNEMONIC_VFMADDSUB231PS, ZYDIS_MNEMONIC_VFMSUBADD231PS, ZYDIS_MNEMONIC_VFMADD231PS, ZYDIS_MNEMONIC_VFMADD231SS,
                 ZYDIS_MNEMONIC_VFMSUB231PS, ZYDIS_MNEMONIC_VFMSUB231SS, ZYDIS_MNEMONIC_VFNMADD231PS, ZYDIS_MNEMONIC_VFNMADD231SS,
                 ZYDIS_MNEMONIC_VFNMSUB231PS, ZYDIS_MNEMONIC_VFNMSUB231SS},
            };
            static const u16 fma_pd[3][10] = {
                {ZYDIS_MNEMONIC_VFMADDSUB132PD, ZYDIS_MNEMONIC_VFMSUBADD132PD, ZYDIS_MNEMONIC_VFMADD132PD, ZYDIS_MNEMONIC_VFMADD132SD,
                 ZYDIS_MNEMONIC_VFMSUB132PD, ZYDIS_MNEMONIC_VFMSUB132SD, ZYDIS_MNEMONIC_VFNMADD132PD, ZYDIS_MNEMONIC_VFNMADD132SD,
                 ZYDIS_MNEMONIC_VFNMSUB132PD, ZYDIS_MNEMONIC_VFNMSUB132SD},
                {ZYDIS_MNEMONIC_VFMADDSUB213PD, ZYDIS_MNEMONIC_VFMSUBADD213PD, ZYDIS_MNEMONIC_VFMADD213PD, ZYDIS_MNEMONIC_VFMADD213SD,
                 ZYDIS_MNEMONIC_VFMSUB213PD, ZYDIS_MNEMONIC_VFMSUB213SD, ZYDIS_MNEMONIC_VFNMADD213PD, ZYDIS_MNEMONIC_VFNMADD213SD,
                 ZYDIS_MNEMONIC_VFNMSUB213PD, ZYDIS_MNEMONIC_VFNMSUB213SD},
                {ZYDIS_MNEMONIC_VFMADDSUB231PD, ZYDIS_MNEMONIC_VFMSUBADD231PD, ZYDIS_MNEMONIC_VFMADD231PD, ZYDIS_MNEMONIC_VFMADD231SD,
                 ZYDIS_MNEMONIC_VFMSUB231PD, ZYDIS_MNEMONIC_VFMSUB231SD, ZYDIS_MNEMONIC_VFNMADD231PD, ZYDIS_MNEMONIC_VFNMADD231SD,
                 ZYDIS_MNEMONIC_VFNMSUB231PD, ZYDIS_MNEMONIC_VFNMSUB231SD},
            };
            bool is_sc = lo >= 9 && (lo & 1);
            mn = W ? fma_pd[grp][lo - 6] : fma_ps[grp][lo - 6];
            if (is_sc) {
                if (L != 0) return false;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 16);
                ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 16);
                ext_set_op(&tmp.ext, 2, rm_type, rm_reg, mod == 3 ? 16 : (W ? 8 : 4));
                form = 6;
            } else {
                form = 1;
            }
        } else if (opc >= 0xF2 && opc <= 0xF7) {
            if (L != 0) return false;
            u8 gsz = W ? 8 : 4;
            u8 grm_t = mod == 3 ? EXT_OP_GPR : EXT_OP_MEM;
            if (opc == 0xF2 && pp == 0) {
                mn = ZYDIS_MNEMONIC_ANDN;
                tmp.writes = F_ARITH;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, gsz);
                ext_set_op(&tmp.ext, 1, EXT_OP_GPR, (u8)vvvv, gsz);
                ext_set_op(&tmp.ext, 2, grm_t, rm_reg, gsz);
                form = 6;
            } else if (opc == 0xF3 && pp == 0 && R == 0) {
                int sub = reg & 7;
                mn = sub == 1 ? ZYDIS_MNEMONIC_BLSR : sub == 2 ? ZYDIS_MNEMONIC_BLSMSK : sub == 3 ? ZYDIS_MNEMONIC_BLSI : 0;
                if (!mn) return false;
                tmp.writes = F_ARITH;
                tmp.ext.op_cnt = 2;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, (u8)vvvv, gsz);
                ext_set_op(&tmp.ext, 1, grm_t, rm_reg, gsz);
                form = 6;
            } else if (opc == 0xF5 && (pp == 0 || pp == 2 || pp == 3)) {
                mn = pp == 0 ? ZYDIS_MNEMONIC_BZHI : pp == 2 ? ZYDIS_MNEMONIC_PEXT : ZYDIS_MNEMONIC_PDEP;
                if (pp == 0) tmp.writes = F_ARITH;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, gsz);
                if (pp == 0) {
                    ext_set_op(&tmp.ext, 1, grm_t, rm_reg, gsz);
                    ext_set_op(&tmp.ext, 2, EXT_OP_GPR, (u8)vvvv, gsz);
                } else {
                    ext_set_op(&tmp.ext, 1, EXT_OP_GPR, (u8)vvvv, gsz);
                    ext_set_op(&tmp.ext, 2, grm_t, rm_reg, gsz);
                }
                form = 6;
            } else if (opc == 0xF6 && pp == 3) {
                mn = ZYDIS_MNEMONIC_MULX;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, gsz);
                ext_set_op(&tmp.ext, 1, EXT_OP_GPR, (u8)vvvv, gsz);
                ext_set_op(&tmp.ext, 2, grm_t, rm_reg, gsz);
                form = 6;
            } else if (opc == 0xF7) {
                mn = pp == 0 ? ZYDIS_MNEMONIC_BEXTR : pp == 1 ? ZYDIS_MNEMONIC_SHLX : pp == 2 ? ZYDIS_MNEMONIC_SARX : ZYDIS_MNEMONIC_SHRX;
                if (pp == 0) tmp.writes = F_ARITH;
                tmp.ext.op_cnt = 3;
                ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, gsz);
                ext_set_op(&tmp.ext, 1, grm_t, rm_reg, gsz);
                ext_set_op(&tmp.ext, 2, EXT_OP_GPR, (u8)vvvv, gsz);
                form = 6;
            } else return false;
        } else if (pp == 1) {
            switch (opc) {
                case 0x00: mn = ZYDIS_MNEMONIC_VPSHUFB; form = 1; break;
                case 0x01: mn = ZYDIS_MNEMONIC_VPHADDW; form = 1; break;
                case 0x02: mn = ZYDIS_MNEMONIC_VPHADDD; form = 1; break;
                case 0x03: mn = ZYDIS_MNEMONIC_VPHADDSW; form = 1; break;
                case 0x04: mn = ZYDIS_MNEMONIC_VPMADDUBSW; form = 1; break;
                case 0x05: mn = ZYDIS_MNEMONIC_VPHSUBW; form = 1; break;
                case 0x06: mn = ZYDIS_MNEMONIC_VPHSUBD; form = 1; break;
                case 0x07: mn = ZYDIS_MNEMONIC_VPHSUBSW; form = 1; break;
                case 0x08: mn = ZYDIS_MNEMONIC_VPSIGNB; form = 1; break;
                case 0x09: mn = ZYDIS_MNEMONIC_VPSIGNW; form = 1; break;
                case 0x0A: mn = ZYDIS_MNEMONIC_VPSIGND; form = 1; break;
                case 0x0B: mn = ZYDIS_MNEMONIC_VPMULHRSW; form = 1; break;
                case 0x0E: if (vvvv || W) return false; mn = ZYDIS_MNEMONIC_VTESTPS; tmp.writes = F_ARITH; form = 2; break;
                case 0x0F: if (vvvv || W) return false; mn = ZYDIS_MNEMONIC_VTESTPD; tmp.writes = F_ARITH; form = 2; break;
                case 0x16: if (!L || W) return false; mn = ZYDIS_MNEMONIC_VPERMPS; form = 1; break;
                case 0x17: if (vvvv || W) return false; mn = ZYDIS_MNEMONIC_VPTEST; tmp.writes = F_ARITH; form = 2; break;
                case 0x1C: if (vvvv) return false; mn = ZYDIS_MNEMONIC_VPABSB; form = 2; break;
                case 0x1D: if (vvvv) return false; mn = ZYDIS_MNEMONIC_VPABSW; form = 2; break;
                case 0x1E: if (vvvv) return false; mn = ZYDIS_MNEMONIC_VPABSD; form = 2; break;
                case 0x28: mn = ZYDIS_MNEMONIC_VPMULDQ; form = 1; break;
                case 0x29: mn = ZYDIS_MNEMONIC_VPCMPEQQ; form = 1; break;
                case 0x2B: mn = ZYDIS_MNEMONIC_VPACKUSDW; form = 1; break;
                case 0x36: if (!L || W) return false; mn = ZYDIS_MNEMONIC_VPERMD; form = 1; break;
                case 0x37: mn = ZYDIS_MNEMONIC_VPCMPGTQ; form = 1; break;
                case 0x38: mn = ZYDIS_MNEMONIC_VPMINSB; form = 1; break;
                case 0x39: mn = ZYDIS_MNEMONIC_VPMINSD; form = 1; break;
                case 0x3A: mn = ZYDIS_MNEMONIC_VPMINUW; form = 1; break;
                case 0x3B: mn = ZYDIS_MNEMONIC_VPMINUD; form = 1; break;
                case 0x3C: mn = ZYDIS_MNEMONIC_VPMAXSB; form = 1; break;
                case 0x3D: mn = ZYDIS_MNEMONIC_VPMAXSD; form = 1; break;
                case 0x3E: mn = ZYDIS_MNEMONIC_VPMAXUW; form = 1; break;
                case 0x3F: mn = ZYDIS_MNEMONIC_VPMAXUD; form = 1; break;
                case 0x40: mn = ZYDIS_MNEMONIC_VPMULLD; form = 1; break;
                case 0x45: mn = W ? ZYDIS_MNEMONIC_VPSRLVQ : ZYDIS_MNEMONIC_VPSRLVD; form = 1; break;
                case 0x46: if (W) return false; mn = ZYDIS_MNEMONIC_VPSRAVD; form = 1; break;
                case 0x47: mn = W ? ZYDIS_MNEMONIC_VPSLLVQ : ZYDIS_MNEMONIC_VPSLLVD; form = 1; break;
                case 0x18: case 0x19: case 0x1A: case 0x58: case 0x59: case 0x5A: case 0x78: case 0x79: {
                    if (vvvv || W) return false;
                    u8 eb = 4;
                    if (opc == 0x18) { mn = ZYDIS_MNEMONIC_VBROADCASTSS; eb = 4; }
                    else if (opc == 0x19) { if (!L) return false; mn = ZYDIS_MNEMONIC_VBROADCASTSD; eb = 8; }
                    else if (opc == 0x1A) { if (!L || mod == 3) return false; mn = ZYDIS_MNEMONIC_VBROADCASTF128; eb = 16; }
                    else if (opc == 0x5A) { if (!L || mod == 3) return false; mn = ZYDIS_MNEMONIC_VBROADCASTI128; eb = 16; }
                    else if (opc == 0x78) { mn = ZYDIS_MNEMONIC_VPBROADCASTB; eb = 1; }
                    else if (opc == 0x79) { mn = ZYDIS_MNEMONIC_VPBROADCASTW; eb = 2; }
                    else if (opc == 0x58) { mn = ZYDIS_MNEMONIC_VPBROADCASTD; eb = 4; }
                    else { mn = ZYDIS_MNEMONIC_VPBROADCASTQ; eb = 8; }
                    tmp.ext.op_cnt = 2;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
                    ext_set_op(&tmp.ext, 1, rm_type, rm_reg, mod == 3 ? 16 : eb);
                    form = 6;
                    break;
                }
                case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25:
                case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: {
                    if (vvvv) return false;
                    static const u16 sx_tbl[6] = {
                        ZYDIS_MNEMONIC_VPMOVSXBW, ZYDIS_MNEMONIC_VPMOVSXBD, ZYDIS_MNEMONIC_VPMOVSXBQ,
                        ZYDIS_MNEMONIC_VPMOVSXWD, ZYDIS_MNEMONIC_VPMOVSXWQ, ZYDIS_MNEMONIC_VPMOVSXDQ,
                    };
                    static const u16 zx_tbl[6] = {
                        ZYDIS_MNEMONIC_VPMOVZXBW, ZYDIS_MNEMONIC_VPMOVZXBD, ZYDIS_MNEMONIC_VPMOVZXBQ,
                        ZYDIS_MNEMONIC_VPMOVZXWD, ZYDIS_MNEMONIC_VPMOVZXWQ, ZYDIS_MNEMONIC_VPMOVZXDQ,
                    };
                    static const u8 mb128[6] = {8, 4, 2, 8, 4, 8};
                    int k = opc & 7;
                    mn = opc < 0x30 ? sx_tbl[k] : zx_tbl[k];
                    tmp.ext.op_cnt = 2;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
                    ext_set_op(&tmp.ext, 1, rm_type, rm_reg, mod == 3 ? 16 : (u8)(mb128[k] << L));
                    form = 6;
                    break;
                }
                default: return false;
            }
        } else return false;
    } else { // mm == 3
        if (opc == 0xF0) {
            if (pp != 3 || L != 0 || vvvv != 0 || p >= end) return false;
            u8 gsz = W ? 8 : 4;
            mn = ZYDIS_MNEMONIC_RORX;
            tmp.imm = *p++;
            tmp.ext.op_cnt = 3;
            ext_set_op(&tmp.ext, 0, EXT_OP_GPR, reg, gsz);
            ext_set_op(&tmp.ext, 1, mod == 3 ? EXT_OP_GPR : EXT_OP_MEM, rm_reg, gsz);
            ext_set_op(&tmp.ext, 2, EXT_OP_IMM, 0, 1);
            form = 6;
        } else if (pp == 1) {
            switch (opc) {
                case 0x00: if (!L || !W || vvvv) return false; mn = ZYDIS_MNEMONIC_VPERMQ; form = 4; need_imm = true; break;
                case 0x01: if (!L || !W || vvvv) return false; mn = ZYDIS_MNEMONIC_VPERMPD; form = 4; need_imm = true; break;
                case 0x02: if (W) return false; mn = ZYDIS_MNEMONIC_VPBLENDD; form = 5; need_imm = true; break;
                case 0x04: if (W || vvvv) return false; mn = ZYDIS_MNEMONIC_VPERMILPS; form = 4; need_imm = true; break;
                case 0x05: if (W || vvvv) return false; mn = ZYDIS_MNEMONIC_VPERMILPD; form = 4; need_imm = true; break;
                case 0x06: if (!L || W) return false; mn = ZYDIS_MNEMONIC_VPERM2F128; form = 5; need_imm = true; break;
                case 0x08: if (vvvv) return false; mn = ZYDIS_MNEMONIC_VROUNDPS; form = 4; need_imm = true; break;
                case 0x09: if (vvvv) return false; mn = ZYDIS_MNEMONIC_VROUNDPD; form = 4; need_imm = true; break;
                case 0x0C: mn = ZYDIS_MNEMONIC_VBLENDPS; form = 5; need_imm = true; break;
                case 0x0D: mn = ZYDIS_MNEMONIC_VBLENDPD; form = 5; need_imm = true; break;
                case 0x0E: mn = ZYDIS_MNEMONIC_VPBLENDW; form = 5; need_imm = true; break;
                case 0x0F: mn = ZYDIS_MNEMONIC_VPALIGNR; form = 5; need_imm = true; break;
                case 0x18: case 0x38:
                    if (!L || W || p >= end) return false;
                    mn = opc == 0x18 ? ZYDIS_MNEMONIC_VINSERTF128 : ZYDIS_MNEMONIC_VINSERTI128;
                    tmp.imm = *p++;
                    tmp.ext.op_cnt = 4;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, 32);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, 32);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, 16);
                    ext_set_op(&tmp.ext, 3, EXT_OP_IMM, 0, 1);
                    form = 6;
                    break;
                case 0x19: case 0x39:
                    if (!L || W || vvvv || p >= end) return false;
                    mn = opc == 0x19 ? ZYDIS_MNEMONIC_VEXTRACTF128 : ZYDIS_MNEMONIC_VEXTRACTI128;
                    tmp.imm = *p++;
                    tmp.ext.op_cnt = 3;
                    ext_set_op(&tmp.ext, 0, rm_type, rm_reg, 16);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, reg, 32);
                    ext_set_op(&tmp.ext, 2, EXT_OP_IMM, 0, 1);
                    form = 6;
                    break;
                case 0x40: mn = ZYDIS_MNEMONIC_VDPPS; form = 5; need_imm = true; break;
                case 0x41: if (L) return false; mn = ZYDIS_MNEMONIC_VDPPD; form = 5; need_imm = true; break;
                case 0x46: if (!L || W) return false; mn = ZYDIS_MNEMONIC_VPERM2I128; form = 5; need_imm = true; break;
                case 0x4A: case 0x4B: case 0x4C: {
                    if (W || p >= end) return false;
                    u8 im = *p++;
                    mn = opc == 0x4A ? ZYDIS_MNEMONIC_VBLENDVPS : opc == 0x4B ? ZYDIS_MNEMONIC_VBLENDVPD : ZYDIS_MNEMONIC_VPBLENDVB;
                    tmp.ext.op_cnt = 4;
                    ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
                    ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, vb);
                    ext_set_op(&tmp.ext, 2, rm_type, rm_reg, vb);
                    ext_set_op(&tmp.ext, 3, EXT_OP_VEC, (u8)(im >> 4), vb);
                    form = 6;
                    break;
                }
                default: return false;
            }
        } else return false;
    }

    if (need_imm) {
        if (p >= end)
            return false;
        tmp.imm = *p++;
    }
    int len = (int)(p - b);
    if (len > 15 || len > avail)
        return false;

    tmp.ext.mnem = mn;
    if (form == 1) {
        tmp.ext.op_cnt = 3;
        ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
        ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, vb);
        ext_set_op(&tmp.ext, 2, rm_type, rm_reg, vb);
    } else if (form == 2) {
        tmp.ext.op_cnt = 2;
        ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
        ext_set_op(&tmp.ext, 1, rm_type, rm_reg, vb);
    } else if (form == 3) {
        tmp.ext.op_cnt = 2;
        ext_set_op(&tmp.ext, 0, rm_type, rm_reg, vb);
        ext_set_op(&tmp.ext, 1, EXT_OP_VEC, reg, vb);
    } else if (form == 4) {
        tmp.ext.op_cnt = 3;
        ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
        ext_set_op(&tmp.ext, 1, rm_type, rm_reg, vb);
        ext_set_op(&tmp.ext, 2, EXT_OP_IMM, 0, 1);
    } else if (form == 5) {
        tmp.ext.op_cnt = 4;
        ext_set_op(&tmp.ext, 0, EXT_OP_VEC, reg, vb);
        ext_set_op(&tmp.ext, 1, EXT_OP_VEC, (u8)vvvv, vb);
        ext_set_op(&tmp.ext, 2, rm_type, rm_reg, vb);
        ext_set_op(&tmp.ext, 3, EXT_OP_IMM, 0, 1);
    }
    tmp.next = rip + (u64)len;
    *in = tmp;
    return true;
}

static int verify_fast_vex_mode = -1;

int x86_decode_ext(u64 rip, Insn *in, const u8 *b, int avail) {
    if (__builtin_expect(verify_fast_vex_mode < 0, 0)) {
        const char *e = getenv("LC_HVDBT_VERIFY_FAST_VEX");
        verify_fast_vex_mode = (e && e[0] == '2') ? 2 : (e && e[0] == '1') ? 1 : 0;
    }
    Insn fast_in = *in;
    if (fast_decode_vex(rip, &fast_in, b, avail)) {
        if (__builtin_expect(!verify_fast_vex_mode, 1)) {
            *in = fast_in;
            return 1;
        }
        // Self-verification mode: compare every decoded field against Zydis!
        ZydisDecodedInstruction zi;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        ZyanStatus st = ZydisDecoderDecodeFull(active_decoder(), b, avail > 0 ? (ZyanUSize)avail : 0, &zi, ops);
        Insn z_in = *in;
        z_in.op = OP_EXT;
        z_in.next = rip + zi.length;
        z_in.reads = 0; z_in.writes = 0; z_in.memop = false; z_in.riprel = false;
        z_in.no_base = false; z_in.no_index = true; z_in.vsib = false;
        z_in.seg = 0; z_in.addr32 = tr_compat; z_in.addr16 = false; z_in.disp = 0; z_in.imm = 0;
        memset(&z_in.ext, 0, sizeof z_in.ext);
        bool z_ok = ZYAN_SUCCESS(st) && zi.length > 0 && zi.length <= 15 && zi.length <= (u8)avail;
        if (z_ok) {
            for (u8 i = 0; i < zi.operand_count_visible; i++)
                if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY && !ext_populate_mem(&zi, &ops[i], &z_in))
                    z_ok = false;
            if (z_ok && (!ext_is_supported(&zi, ops, &z_in) || !ext_fill_from_zydis(&zi, ops, &z_in)))
                z_ok = false;
        }
        if (!z_ok || fast_in.next != z_in.next || fast_in.reads != z_in.reads || fast_in.writes != z_in.writes ||
            fast_in.memop != z_in.memop || fast_in.riprel != z_in.riprel || fast_in.no_base != z_in.no_base ||
            fast_in.no_index != z_in.no_index || fast_in.base != z_in.base || fast_in.index != z_in.index ||
            fast_in.scale != z_in.scale || fast_in.disp != z_in.disp || fast_in.imm != z_in.imm ||
            memcmp(&fast_in.ext, &z_in.ext, sizeof(ExtDecoded)) != 0) {
            fprintf(stderr, "FAST_VEX_MISMATCH rip=%llx bytes=%02x %02x %02x %02x %02x %02x z_ok=%d "
                    "fast(mn=%u cnt=%u len=%llu r=%u w=%u m=%d rip=%d nb=%d ni=%d b=%u i=%u s=%u d=%lld im=%lld "
                    "op=[%u:%u:%u,%u:%u:%u,%u:%u:%u,%u:%u:%u]) "
                    "zydis(mn=%u cnt=%u len=%llu r=%u w=%u m=%d rip=%d nb=%d ni=%d b=%u i=%u s=%u d=%lld im=%lld "
                    "op=[%u:%u:%u,%u:%u:%u,%u:%u:%u,%u:%u:%u])\n",
                    (unsigned long long)rip, b[0], b[1], b[2], b[3], b[4], b[5], z_ok,
                    fast_in.ext.mnem, fast_in.ext.op_cnt, (unsigned long long)(fast_in.next - rip),
                    fast_in.reads, fast_in.writes, fast_in.memop, fast_in.riprel, fast_in.no_base, fast_in.no_index,
                    fast_in.base, fast_in.index, fast_in.scale, (long long)fast_in.disp, (long long)fast_in.imm,
                    fast_in.ext.op_type[0], fast_in.ext.op_reg[0], fast_in.ext.op_bytes[0],
                    fast_in.ext.op_type[1], fast_in.ext.op_reg[1], fast_in.ext.op_bytes[1],
                    fast_in.ext.op_type[2], fast_in.ext.op_reg[2], fast_in.ext.op_bytes[2],
                    fast_in.ext.op_type[3], fast_in.ext.op_reg[3], fast_in.ext.op_bytes[3],
                    z_in.ext.mnem, z_in.ext.op_cnt, (unsigned long long)(z_in.next - rip),
                    z_in.reads, z_in.writes, z_in.memop, z_in.riprel, z_in.no_base, z_in.no_index,
                    z_in.base, z_in.index, z_in.scale, (long long)z_in.disp, (long long)z_in.imm,
                    z_in.ext.op_type[0], z_in.ext.op_reg[0], z_in.ext.op_bytes[0],
                    z_in.ext.op_type[1], z_in.ext.op_reg[1], z_in.ext.op_bytes[1],
                    z_in.ext.op_type[2], z_in.ext.op_reg[2], z_in.ext.op_bytes[2],
                    z_in.ext.op_type[3], z_in.ext.op_reg[3], z_in.ext.op_bytes[3]);
            if (verify_fast_vex_mode == 1)
                abort();
        }
        *in = fast_in;
        return 1;
    }

    ZydisDecodedInstruction zi;
    ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
    // Pass only the bytes tr_fetch really read. The buffer past `avail` is
    // zero padding, and Zydis will happily finish a VEX opcode from those
    // zeros.
    ZyanStatus st = ZydisDecoderDecodeFull(active_decoder(), b, avail > 0 ? (ZyanUSize)avail : 0, &zi, ops);
    if (!ZYAN_SUCCESS(st))
        return ext_reject(rip);
    if (zi.length == 0 || zi.length > 15)
        return ext_reject(rip);
    if (zi.length > (u8)avail)
        return ext_reject(rip);

    Insn tmp = *in;
    tmp.op = OP_EXT;
    tmp.next = rip + zi.length;
    tmp.reads = 0;
    tmp.writes = 0;
    tmp.memop = false;
    tmp.riprel = false;
    tmp.no_base = false;
    tmp.no_index = true;
    tmp.vsib = false;
    tmp.seg = 0;
    tmp.addr32 = tr_compat;
    tmp.addr16 = false;
    tmp.disp = 0;
    tmp.imm = 0;
    memset(&tmp.ext, 0, sizeof tmp.ext);

    for (u8 i = 0; i < zi.operand_count_visible; i++) {
        if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY && !ext_populate_mem(&zi, &ops[i], &tmp))
            return ext_reject(rip);
    }
    if (!ext_is_supported(&zi, ops, &tmp))
        return ext_reject(rip);
    if (!ext_fill_from_zydis(&zi, ops, &tmp))
        return ext_reject(rip);

    *in = tmp;
    return 1;
}
// Sets SF, ZF, PF from res (width sz=4 or 8), clears OF and AF, and sets x86 CF:
// cf_mode == 0: CF = 0 (Arm64 C = 1)
// cf_mode == 1: CF = cf_reg[0] (Arm64 C = !cf_reg[0])
// cf_mode == 2: CF = !cf_reg[0] (Arm64 C = cf_reg[0])
static void emit_bmi_flags(int sf, int res, int live, int cf_mode, int cf_reg) {
    if (!live)
        return;
    logic_reg(sf, 3, 0, XZR, res, res); // ANDS XZR, res, res -> N=SF, Z=ZF, C=0, V=0
    if (cf_mode == 0) {
        cfinv(); // Arm64 C = 1 (x86 CF = 0)
    } else if (cf_mode == 1) {
        rmif(cf_reg, 63, 2); // Arm64 C = cf_reg[0]
        cfinv();             // Arm64 C = !cf_reg[0]
    } else {
        rmif(cf_reg, 63, 2); // Arm64 C = cf_reg[0] (x86 CF = !cf_reg[0])
    }
    if (live & F_P)
        mov_reg(0, R_PF, res);
    if (live & F_A)
        mov_reg(1, R_AF, R_PF);
}

static inline void vrev128_32(int vd, int vn) {
    VREV64Q_32(vd, vn);
    VEXTQ_8(vd, vd, vd, 8);
}

static bool ext_emit_pshufd128(int d, int s, u8 imm8, int t0);

static void ext_lanes32(int d, int src1, int src2, int imm) {
    enum { T0 = 27 };
    imm &= 0xff;
    if (src1 == src2 && ext_emit_pshufd128(d, src1, (u8)imm, T0))
        return;
    if (imm == 0x44) {
        VZIP1Q_64(d, src1, src2);
        return;
    }
    if (imm == 0xEE) {
        VZIP2Q_64(d, src1, src2);
        return;
    }
    if (imm == 0x4E) {
        VEXTQ_8(d, src1, src2, 8);
        return;
    }
    if (imm == 0x88) {
        RRR(0x4E801800u, d, src1, src2); // uzp1 .4s
        return;
    }
    if (imm == 0xDD) {
        RRR(0x4E805800u, d, src1, src2); // uzp2 .4s
        return;
    }
    if (imm == 0xE4) {
        int dst = (d == src2 && d != src1) ? T0 : d;
        vmov128(dst, src1);
        ins_elem(dst, 1, src2, 1, 8);
        vmov128(d, dst);
        return;
    }
    int tmp = (d != src1 && d != src2) ? d : T0;
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
        vmov128(d, tmp);
}

static void ext_shift_imm128(int d, int s, int esize, int kind, int n) {
    if (kind != 2 && n >= esize) {
        VEORQ(d, d, d);
        return;
    }
    if (kind == 2 && n >= esize)
        n = esize - 1;
    if (n == 0) {
        vmov128(d, s);
        return;
    }
    if (kind == 0) {
        u32 immhb = (u32)(esize + n);
        E(0x4F005400u | immhb << 16 | (u32)s << 5 | (u32)d);
    } else {
        u32 immhb = (u32)(2 * esize - n);
        E((kind == 1 ? 0x6F000400u : 0x4F000400u) | immhb << 16 | (u32)s << 5 | (u32)d);
    }
}

static void ext_shift_cnt128(int d, int s, int vcnt, int esize, int kind) {
    enum { T0 = 27 };
    u32 sz = esize == 16 ? 1 : esize == 32 ? 2 : 3;
    umov_d(S0, vcnt, 0);
    movimm(S1, (u64)esize);
    lsr_imm(1, S2, S0, esize == 16 ? 4 : esize == 32 ? 5 : 6);
    cbz(0, 1, S2, 2);
    mov_reg(1, S0, S1);
    if (kind)
        add_reg(1, 1, 0, S0, XZR, S0, 0);
    u32 imm5 = esize == 16 ? 2 : esize == 32 ? 4 : 8;
    E(0x4E000C00u | imm5 << 16 | S0 << 5 | T0);
    if (kind == 2)
        RRR(0x4E204400u | sz << 22, d, s, T0); // sshl
    else
        RRR(0x6E204400u | sz << 22, d, s, T0); // ushl
}

// Per-lane variable shifts: VPSLLVD/Q (kind=0), VPSRLVD/Q (kind=1), VPSRAVD (kind=2).
static void ext_shift_var128(int d, int s, int vcnt, int esize, int kind) {
    enum { T0 = 27, T1 = 28 };
    u32 sz = esize == 32 ? 2 : 3;
    // Clamp unsigned lane counts in vcnt to esize (32 or 64) so counts >= esize produce 0 (or sign bit for SAR).
    if (esize == 32) {
        E(0x4F010400u | T0); // movi T0.4s, #32
        RRR(0x6EA06C00u, T1, vcnt, T0); // umin v.4s, vcnt.4s, T0.4s
    } else {
        movimm(S0, 64);
        VDUPQD(T0, S0);
        // T1 = (64 > vcnt); BSL T1, vcnt, T0 selects vcnt where (64 > vcnt), else keeps T0 (64).
        RRR(0x6EE03400u, T1, T0, vcnt); // cmhi v.2d, T0.2d, vcnt.2d
        RRR(0x6E601C00u, T1, vcnt, T0); // bsl T1.16b, vcnt.16b, T0.16b
    }
    if (kind != 0)
        RR(0x6E20B800u | sz << 22, T1, T1); // neg T1.<T>, T1.<T>
    if (kind == 2)
        RRR(0x4E204400u | sz << 22, d, s, T1); // sshl
    else
        RRR(0x6E204400u | sz << 22, d, s, T1); // ushl
}

static void umov_w(int rd, int rn, int lane, int bytes) {
    int ls = log2sz(bytes);
    u32 imm5 = (u32)(lane << (ls + 1)) | (1u << ls);
    E(0x0E003C00u | imm5 << 16 | (u32)rn << 5 | (u32)rd);
}

static void ins_gpr(int vd, int lane, int rt, int bytes) {
    int ls = log2sz(bytes);
    u32 imm5 = (u32)(lane << (ls + 1)) | (1u << ls);
    E(0x4E001C00u | imm5 << 16 | (u32)rt << 5 | (u32)vd);
}

// Boolean result of VCMP*. pred bit 4 only changes exception signaling, which this engine does not raise.
static void emit_vcmp_mask(int d, int n, int s, int pred, bool isd) {
    enum { T0 = 27, T1 = 28 };
    int p = pred & 15;
    u32 sz = isd ? 1u << 22 : 0;
    u32 eq = 0x4E20E400u | sz, ge = 0x6E20E400u | sz, gt = 0x6EA0E400u | sz;
    switch (p) {
        case 0: RRR(eq, d, n, s); return;
        case 1: RRR(gt, d, s, n); return;
        case 2: RRR(ge, d, s, n); return;
        case 3:
            RRR(eq, T0, n, n);
            RRR(eq, T1, s, s);
            VANDQ(d, T0, T1);
            RR(0x6E205800u, d, d);
            return;
        case 4: RRR(eq, d, n, s); RR(0x6E205800u, d, d); return;
        case 5: RRR(gt, d, s, n); RR(0x6E205800u, d, d); return;
        case 6: RRR(ge, d, s, n); RR(0x6E205800u, d, d); return;
        case 7:
            RRR(eq, T0, n, n);
            RRR(eq, T1, s, s);
            VANDQ(d, T0, T1);
            return;
        case 8:
            RRR(eq, T0, n, s);
            RRR(eq, d, n, n);
            RRR(eq, T1, s, s);
            VANDQ(d, d, T1);
            RR(0x6E205800u, d, d);
            VORRQ(d, d, T0);
            return;
        case 9: RRR(ge, d, n, s); RR(0x6E205800u, d, d); return;
        case 10: RRR(gt, d, n, s); RR(0x6E205800u, d, d); return;
        case 11: VEORQ(d, d, d); return;
        case 12:
            RRR(eq, T0, n, s);
            RR(0x6E205800u, T0, T0);
            RRR(eq, d, n, n);
            RRR(eq, T1, s, s);
            VANDQ(d, d, T1);
            VANDQ(d, d, T0);
            return;
        case 13: RRR(ge, d, n, s); return;
        case 14: RRR(gt, d, n, s); return;
        default:
            VEORQ(d, d, d);
            RR(0x6E205800u, d, d);
            return;
    }
}

// VPERMILPS variable: each dword's low 2 bits select a dword inside this 128-bit lane.
static void ext_permilps128(int d, int data, int ctrl) {
    enum { T0 = 27, T1 = 28, T2 = 29 };
    vmov128(T2, data);
    movimm(S0, 0x0000000300000003ULL);
    fmov_to_d(T0, S0);
    ins_x(T0, 1, S0);
    VANDQ(T1, ctrl, T0);
    ext_shift_imm128(T1, T1, 32, 0, 2); // low byte of each dword is 0, 4, 8 or 12
    movimm(S0, 0x0101010101010101ULL);
    fmov_to_d(T0, S0);
    ins_x(T0, 1, S0);
    RRR(0x4EA09C00u, T1, T1, T0); // mul.4s: splat that byte across the dword
    movimm(S0, 0x0302010003020100ULL);
    fmov_to_d(T0, S0);
    ins_x(T0, 1, S0);
    VADDQ_32(T1, T1, T0);
    RRR(0x4E000000u, d, T2, T1);
}

// VPERMILPD variable: bit 1 of each qword selects the low or high double of this lane.
static void ext_permilpd128(int d, int data, int ctrl) {
    umov_d(S0, ctrl, 0);
    umov_d(S1, ctrl, 1);
    ubfx(1, S0, S0, 1, 1);
    ubfx(1, S1, S1, 1, 1);
    umov_d(S2, data, 0);
    umov_d(S3, data, 1);
    mrs_nzcv(S4);
    add_reg(1, 1, 1, XZR, S0, XZR, 0);
    csel(1, S0, S3, S2, 1);
    add_reg(1, 1, 1, XZR, S1, XZR, 0);
    csel(1, S1, S3, S2, 1);
    msr_nzcv(S4);
    VEORQ(d, d, d);
    ins_x(d, 0, S0);
    ins_x(d, 1, S1);
}

// Pure-C ARM64 NEON lowering helpers (replacing fex_emit_ext.cpp for standalone CEngine).
static void ext_emit_permq_half(int d, int s_lo, int s_hi, u8 sel4) {
    switch (sel4 & 0xf) {
        case 0x0: VDUPQ_64(d, s_lo, 0); break;
        case 0x1: VEXTQ_8(d, s_lo, s_lo, 8); break;
        case 0x2: VZIP1Q_64(d, s_hi, s_lo); break;
        case 0x3: VEXTQ_8(d, s_hi, s_lo, 8); break;
        case 0x4: vmov128(d, s_lo); break;
        case 0x5: VDUPQ_64(d, s_lo, 1); break;
        case 0x6:
            vmov128(d, s_hi);
            ins_elem(d, 1, s_lo, 1, 8);
            break;
        case 0x7: RRR(0x4EC06800u, d, s_hi, s_lo); break; // trn2 .2d
        case 0x8: VZIP1Q_64(d, s_lo, s_hi); break;
        case 0x9: VEXTQ_8(d, s_lo, s_hi, 8); break;
        case 0xA: VDUPQ_64(d, s_hi, 0); break;
        case 0xB: VEXTQ_8(d, s_hi, s_hi, 8); break;
        case 0xC:
            vmov128(d, s_lo);
            ins_elem(d, 1, s_hi, 1, 8);
            break;
        case 0xD: RRR(0x4EC06800u, d, s_lo, s_hi); break; // trn2 .2d
        case 0xE: vmov128(d, s_hi); break;
        case 0xF: VDUPQ_64(d, s_hi, 1); break;
    }
}

static bool ext_emit_pshufd128(int d, int s, u8 imm8, int t0) {
    switch (imm8) {
        case 0xE4: vmov128(d, s); return true;
        case 0x00: case 0x55: case 0xAA: case 0xFF:
            VDUPQ_32(d, s, imm8 & 3); return true;
        case 0x44: VDUPQ_64(d, s, 0); return true;
        case 0xEE: VDUPQ_64(d, s, 1); return true;
        case 0xB1: VREV64Q_32(d, s); return true;
        case 0x4E: VEXTQ_8(d, s, s, 8); return true;
        case 0x39: VEXTQ_8(d, s, s, 4); return true;
        case 0x93: VEXTQ_8(d, s, s, 12); return true;
        case 0x50: VZIP1Q_32(d, s, s); return true;
        case 0xFA: VZIP2Q_32(d, s, s); return true;
        case 0x88: RRR(0x4E801800u, d, s, s); return true; // uzp1 .4s
        case 0xDD: RRR(0x4E805800u, d, s, s); return true; // uzp2 .4s
        case 0xA0: RRR(0x4E802800u, d, s, s); return true; // trn1 .4s
        case 0xF5: RRR(0x4E806800u, d, s, s); return true; // trn2 .4s
        case 0x1B:
            VREV64Q_32(t0, s);
            VEXTQ_8(d, t0, t0, 8);
            return true;
        case 0x11:
            VREV64Q_32(t0, s);
            VZIP1Q_64(d, t0, t0);
            return true;
        case 0xBB:
            VREV64Q_32(t0, s);
            VZIP2Q_64(d, t0, t0);
            return true;
        case 0x33:
            VEXTQ_8(t0, s, s, 4);
            VZIP2Q_64(d, t0, t0);
            return true;
        case 0x22:
            VEXTQ_8(t0, s, s, 8);
            RRR(0x4E801800u, d, t0, t0); // uzp1 .4s -> [2,0,2,0]
            return true;
        case 0x77:
            VEXTQ_8(t0, s, s, 8);
            RRR(0x4E805800u, d, t0, t0); // uzp2 .4s -> [3,1,3,1]
            return true;
        case 0x99:
            VEXTQ_8(t0, s, s, 4);
            VZIP1Q_64(d, t0, t0); // [1,2,1,2]
            return true;
        default:
            return false;
    }
}

static void ext_emit_vblend32_128(int d, int n, int s, u8 mask4, int t0) {
    mask4 &= 0xf;
    if (mask4 == 0x0) {
        vmov128(d, n);
        return;
    }
    if (mask4 == 0xF) {
        vmov128(d, s);
        return;
    }
    if (mask4 == 0x3 || mask4 == 0xC) {
        int lane64 = mask4 == 0xC ? 1 : 0;
        int dst = (d == s && d != n) ? t0 : d;
        vmov128(dst, n);
        ins_elem(dst, lane64, s, lane64, 8);
        vmov128(d, dst);
        return;
    }
    if (mask4 == 0x5) {
        VREV64Q_32(t0, s);
        RRR(0x4E806800u, d, t0, n); // trn2 .4s
        return;
    }
    if (mask4 == 0xA) {
        VREV64Q_32(t0, n);
        RRR(0x4E806800u, d, t0, s); // trn2 .4s
        return;
    }
    int dst = (d == s && d != n) ? t0 : d;
    vmov128(dst, n);
    for (int i = 0; i < 4; i++)
        if ((mask4 >> i) & 1)
            ins_elem(dst, i, s, i, 4);
    vmov128(d, dst);
}

static void ext_emit_vaes128(int d, int n, int s, bool dec, bool last, int t0, int t1) {
    E(0x4F00E400u | (u32)t0); // movi t0.16b, #0
    int work = (d == s) ? t1 : d;
    vmov128(work, n);
    if (dec) {
        RR(0x4E285800u, work, t0); // aesd
        if (!last)
            RR(0x4E287800u, work, work); // aesimc
    } else {
        RR(0x4E284800u, work, t0); // aese
        if (!last)
            RR(0x4E286800u, work, work); // aesmc
    }
    VEORQ(d, work, s);
}

static void ext_emit_vpermd_indices(int dst, int idx, int t_mask, int t_rep, int t_tmp) {
    VANDQ(dst, idx, t_mask);
    RRR(0x4E002800u, t_tmp, dst, dst); // trn1 .16b
    RRR(0x4E402800u, dst, t_tmp, t_tmp); // trn1 .8h
    E(0x4F0A5400u | (u32)dst << 5 | (u32)dst); // shl dst.16b, dst.16b, #2
    VADDQ_8(dst, dst, t_rep);
}

static void ext_emit_pmovmskb128(int rd_gpr, int src_vec, int t0) {
    RR(0x6F090400u, t0, src_vec);  // ushr t0.16b, src.16b, #7
    RR(0x6F191400u, t0, t0);       // usra t0.8h, t0.8h, #7
    RR(0x6F321400u, t0, t0);       // usra t0.4s, t0.4s, #14
    RR(0x6F641400u, t0, t0);       // usra t0.2d, t0.2d, #28
    ins_elem(t0, 1, t0, 8, 1);     // ins t0.b[1], t0.b[8]
    umov_w(rd_gpr, t0, 0, 2);      // umov w_rd, t0.h[0]
}

static void ext_emit_vpmovmskb256(int rd_gpr, int src_lo, int src_hi, int t0, int t1) {
    RR(0x6F090400u, t0, src_lo);   // ushr t0.16b, src_lo.16b, #7
    RR(0x6F090400u, t1, src_hi);   // ushr t1.16b, src_hi.16b, #7
    RR(0x6F191400u, t0, t0);       // usra t0.8h, t0.8h, #7
    RR(0x6F191400u, t1, t1);       // usra t1.8h, t1.8h, #7
    RR(0x6F321400u, t0, t0);       // usra t0.4s, t0.4s, #14
    RR(0x6F321400u, t1, t1);       // usra t1.4s, t1.4s, #14
    RR(0x6F641400u, t0, t0);       // usra t0.2d, t0.2d, #28
    RR(0x6F641400u, t1, t1);       // usra t1.2d, t1.2d, #28
    ins_elem(t0, 1, t0, 8, 1);
    ins_elem(t0, 2, t1, 0, 1);
    ins_elem(t0, 3, t1, 8, 1);
    umov_w(rd_gpr, t0, 0, 4);      // umov w_rd, t0.s[0] (full 32-bit mask!)
}

// Executes one 128-bit lane operation: vd = op(vn, vm, vc, imm).
// Uses only T0=27, T1=28, T2=29 as internal vector scratch so caller's V_S0(24), V_S1(25), V_S2(26), V_MEM(30) are untouched.
static void emit_lane128_op(ZydisMnemonic m, int d, int n, int s, int c, int imm, bool has_imm) {
    enum { T0 = 27, T1 = 28, T2 = 29 };
    switch (m) {
        // FP packed arithmetic
        case ZYDIS_MNEMONIC_VADDPS:  RRR(0x4E20D400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VADDPD:  RRR(0x4E60D400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VSUBPS:  RRR(0x4EA0D400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VSUBPD:  RRR(0x4EE0D400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMULPS:  RRR(0x6E20DC00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMULPD:  RRR(0x6E60DC00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VDIVPS:  RRR(0x6E20FC00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VDIVPD:  RRR(0x6E60FC00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMINPS:  RRR(0x4EA0F400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMINPD:  RRR(0x4EE0F400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMAXPS:  RRR(0x4E20F400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VMAXPD:  RRR(0x4E60F400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VSQRTPS: RR(0x6EA1F800u, d, n); return;
        case ZYDIS_MNEMONIC_VSQRTPD: RR(0x6EE1F800u, d, n); return;
        case ZYDIS_MNEMONIC_VRSQRTPS:
            RR(0x6EA1D800u, T0, n);
            RRR(0x6E20DC00u, T1, T0, T0);
            RRR(0x4EA0FC00u, T1, n, T1);
            RRR(0x6E20DC00u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VRCPPS:
            RR(0x4EA1D800u, T0, n);
            RRR(0x4E20FC00u, T1, n, T0);
            RRR(0x6E20DC00u, d, T0, T1);
            return;

        // Bitwise logical
        case ZYDIS_MNEMONIC_VANDPS:
        case ZYDIS_MNEMONIC_VANDPD:
        case ZYDIS_MNEMONIC_VPAND:
            VANDQ(d, n, s);
            return;
        case ZYDIS_MNEMONIC_VORPS:
        case ZYDIS_MNEMONIC_VORPD:
        case ZYDIS_MNEMONIC_VPOR:
            VORRQ(d, n, s);
            return;
        case ZYDIS_MNEMONIC_VXORPS:
        case ZYDIS_MNEMONIC_VXORPD:
        case ZYDIS_MNEMONIC_VPXOR:
            VEORQ(d, n, s);
            return;
        case ZYDIS_MNEMONIC_VANDNPS:
        case ZYDIS_MNEMONIC_VANDNPD:
        case ZYDIS_MNEMONIC_VPANDN:
            VBICQ(d, s, n); // ~n & s
            return;

        // Integer add / sub / sat / avg / min / max / abs / sign / cmp
        case ZYDIS_MNEMONIC_VPADDB:   VADDQ_8(d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDW:   VADDQ_16(d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDD:   VADDQ_32(d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDQ:   VADDQ_64(d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBB:   VSUBQ_8(d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBW:   VSUBQ_16(d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBD:   VSUBQ_32(d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBQ:   VSUBQ_64(d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDSB:  RRR(0x4E200C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDSW:  RRR(0x4E600C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDUSB: RRR(0x6E200C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPADDUSW: RRR(0x6E600C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBSB:  RRR(0x4E202C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBSW:  RRR(0x4E602C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBUSB: RRR(0x6E202C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPSUBUSW: RRR(0x6E602C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPAVGB:   RRR(0x6E201400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPAVGW:   RRR(0x6E601400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINSB:  RRR(0x4E206C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINSW:  RRR(0x4E606C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINSD:  RRR(0x4EA06C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXSB:  RRR(0x4E206400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXSW:  RRR(0x4E606400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXSD:  RRR(0x4EA06400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINUB:  RRR(0x6E206C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINUW:  RRR(0x6E606C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMINUD:  RRR(0x6EA06C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXUB:  RRR(0x6E206400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXUW:  RRR(0x6E606400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMAXUD:  RRR(0x6EA06400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPABSB:   RR(0x4E20B800u, d, n); return;
        case ZYDIS_MNEMONIC_VPABSW:   RR(0x4E60B800u, d, n); return;
        case ZYDIS_MNEMONIC_VPABSD:   RR(0x4EA0B800u, d, n); return;
        case ZYDIS_MNEMONIC_VPSIGNB:
        case ZYDIS_MNEMONIC_VPSIGNW:
        case ZYDIS_MNEMONIC_VPSIGND: {
            u32 sz = m == ZYDIS_MNEMONIC_VPSIGNB ? 0 : m == ZYDIS_MNEMONIC_VPSIGNW ? 1 : 2;
            RR(0x6E20B800u | sz << 22, T0, n); // neg
            RR(0x4E20A800u | sz << 22, T1, s); // cmlt #0
            vmov128(T2, n);
            RRR(0x6EA01C00u, T2, T0, T1);      // bit
            RR(0x4E209800u | sz << 22, T1, s); // cmeq #0
            RRR(0x4E601C00u, d, T2, T1);       // bic
            return;
        }
        case ZYDIS_MNEMONIC_VPCMPEQB: RRR(0x6E208C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPEQW: RRR(0x6E608C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPEQD: RRR(0x6EA08C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPEQQ: RRR(0x6EE08C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPGTB: RRR(0x4E203400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPGTW: RRR(0x4E603400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPGTD: RRR(0x4EA03400u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPCMPGTQ: RRR(0x4EE03400u, d, n, s); return;

        // Multiplies
        case ZYDIS_MNEMONIC_VPMULLW:  RRR(0x4E609C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMULLD:  RRR(0x4EA09C00u, d, n, s); return;
        case ZYDIS_MNEMONIC_VPMULHUW:
        case ZYDIS_MNEMONIC_VPMULHW: {
            bool uns = m == ZYDIS_MNEMONIC_VPMULHUW;
            RRR(uns ? 0x2E60C000u : 0x0E60C000u, T0, n, s);
            RRR(uns ? 0x6E60C000u : 0x4E60C000u, T1, n, s);
            RRR(0x4E405800u, d, T0, T1); // uzp2 .8h
            return;
        }
        case ZYDIS_MNEMONIC_VPMULHRSW:
            RRR(0x0E60C000u, T0, n, s);
            RRR(0x4E60C000u, T1, n, s);
            E(0x4F002400u | 49u << 16 | T0 << 5 | T0);
            E(0x4F002400u | 49u << 16 | T1 << 5 | T1);
            RRR(0x4E401800u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VPMULUDQ:
            RR(0x0EA12800u, T0, n);
            RR(0x0EA12800u, T1, s);
            RRR(0x2EA0C000u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VPMULDQ:
            RR(0x0EA12800u, T0, n);
            RR(0x0EA12800u, T1, s);
            RRR(0x0EA0C000u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VPMADDWD:
            RRR(0x0E60C000u, T0, n, s);
            RRR(0x4E60C000u, T1, n, s);
            RRR(0x4EA0BC00u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VPMADDUBSW:
            RR(0x2F08A400u, T0, n);
            RR(0x6F08A400u, T1, n);
            RR(0x0F08A400u, T2, s);
            RRR(0x4E609C00u, T0, T0, T2);
            RR(0x4F08A400u, T2, s);
            RRR(0x4E609C00u, T1, T1, T2);
            RRR(0x4E401800u, T2, T0, T1);
            RRR(0x4E405800u, T0, T0, T1);
            RRR(0x4E600C00u, d, T2, T0);
            return;

        // Shuffles / unpacks / packs / alignr
        case ZYDIS_MNEMONIC_VPSHUFB:
            E(0x4F04E400u | 15 << 5 | T0); // movi v27.16b, #0x8f
            VANDQ(T0, s, T0);
            E(0x4E000000u | T0 << 16 | n << 5 | d); // tbl d.16b, {n.16b}, T0.16b
            return;
        case ZYDIS_MNEMONIC_VPSHUFD:
            if (!ext_emit_pshufd128(d, n, (u8)imm, T0))
                ext_lanes32(d, n, n, imm);
            return;
        case ZYDIS_MNEMONIC_VPSHUFHW:
        case ZYDIS_MNEMONIC_VPSHUFLW: {
            bool hi = m == ZYDIS_MNEMONIC_VPSHUFHW;
            int im = imm & 0xff;
            if (im == 0xE4) {
                vmov128(d, n);
                return;
            }
            if (im == 0x00 || im == 0x55 || im == 0xAA || im == 0xFF) {
                int k = (hi ? 4 : 0) + (im & 3);
                E(0x4E020400u | (u32)k << 18 | (u32)n << 5 | T0); // dup T0.8h, n.h[k]
                ins_elem(T0, hi ? 0 : 1, n, hi ? 0 : 1, 8);
                vmov128(d, T0);
                return;
            }
            if (im == 0x1B || im == 0xB1 || im == 0x4E) {
                u32 rev = im == 0x1B ? 0x4E600800u : im == 0xB1 ? 0x6E600800u : 0x4EA00800u;
                RR(rev, T0, n);
                ins_elem(T0, hi ? 0 : 1, n, hi ? 0 : 1, 8);
                vmov128(d, T0);
                return;
            }
            int base = hi ? 4 : 0;
            ins_elem(T0, hi ? 0 : 1, n, hi ? 0 : 1, 8);
            for (int i = 0; i < 4; i++)
                ins_elem(T0, base + i, n, base + ((im >> (2 * i)) & 3), 2);
            vmov128(d, T0);
            return;
        }
        case ZYDIS_MNEMONIC_VSHUFPS:
            ext_lanes32(d, n, s, imm);
            return;
        case ZYDIS_MNEMONIC_VSHUFPD: {
            int im = imm & 3;
            if (im == 0) {
                VZIP1Q_64(d, n, s);
            } else if (im == 3) {
                VZIP2Q_64(d, n, s);
            } else if (im == 1) {
                VEXTQ_8(d, n, s, 8);
            } else {
                int dst = (d == s && d != n) ? T0 : d;
                vmov128(dst, n);
                ins_elem(dst, 1, s, 1, 8);
                vmov128(d, dst);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VPUNPCKLBW:  VZIP1Q_8(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKLWD:  VZIP1Q_16(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKLDQ:
        case ZYDIS_MNEMONIC_VUNPCKLPS:   VZIP1Q_32(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKLQDQ:
        case ZYDIS_MNEMONIC_VUNPCKLPD:   VZIP1Q_64(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKHBW:  VZIP2Q_8(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKHWD:  VZIP2Q_16(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKHDQ:
        case ZYDIS_MNEMONIC_VUNPCKHPS:   VZIP2Q_32(d, n, s); return;
        case ZYDIS_MNEMONIC_VPUNPCKHQDQ:
        case ZYDIS_MNEMONIC_VUNPCKHPD:   VZIP2Q_64(d, n, s); return;
        case ZYDIS_MNEMONIC_VPACKSSWB:
            RR(0x0E214800u, T0, n);
            RR(0x4E214800u, T0, s);
            vmov128(d, T0);
            return;
        case ZYDIS_MNEMONIC_VPACKSSDW:
            RR(0x0E614800u, T0, n);
            RR(0x4E614800u, T0, s);
            vmov128(d, T0);
            return;
        case ZYDIS_MNEMONIC_VPACKUSWB:
            RR(0x2E212800u, T0, n);
            RR(0x6E212800u, T0, s);
            vmov128(d, T0);
            return;
        case ZYDIS_MNEMONIC_VPACKUSDW:
            RR(0x2E612800u, T0, n);
            RR(0x6E612800u, T0, s);
            vmov128(d, T0);
            return;
        case ZYDIS_MNEMONIC_VPALIGNR: {
            int k = imm & 0xff;
            if (k >= 32) {
                VEORQ(d, d, d);
            } else if (k >= 16) {
                VEORQ(T0, T0, T0);
                if (k == 16)
                    vmov128(d, n);
                else
                    VEXTQ_8(d, n, T0, k - 16);
            } else if (k == 0) {
                vmov128(d, s);
            } else {
                VEXTQ_8(d, s, n, k);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VPSLLDQ:
        case ZYDIS_MNEMONIC_VPSRLDQ: {
            int k = imm & 0xff;
            if (k > 15) {
                VEORQ(d, d, d);
                return;
            }
            if (k == 0) {
                vmov128(d, n);
                return;
            }
            VEORQ(T0, T0, T0);
            if (m == ZYDIS_MNEMONIC_VPSRLDQ)
                VEXTQ_8(d, n, T0, k);
            else
                VEXTQ_8(d, T0, n, 16 - k);
            return;
        }

        // Shifts (immediate, register count, and variable per-lane)
        case ZYDIS_MNEMONIC_VPSLLW:
        case ZYDIS_MNEMONIC_VPSLLD:
        case ZYDIS_MNEMONIC_VPSLLQ:
        case ZYDIS_MNEMONIC_VPSRLW:
        case ZYDIS_MNEMONIC_VPSRLD:
        case ZYDIS_MNEMONIC_VPSRLQ:
        case ZYDIS_MNEMONIC_VPSRAW:
        case ZYDIS_MNEMONIC_VPSRAD: {
            int esize = (m == ZYDIS_MNEMONIC_VPSLLW || m == ZYDIS_MNEMONIC_VPSRLW || m == ZYDIS_MNEMONIC_VPSRAW) ? 16 :
                        (m == ZYDIS_MNEMONIC_VPSLLD || m == ZYDIS_MNEMONIC_VPSRLD || m == ZYDIS_MNEMONIC_VPSRAD) ? 32 : 64;
            int kind = (m == ZYDIS_MNEMONIC_VPSLLW || m == ZYDIS_MNEMONIC_VPSLLD || m == ZYDIS_MNEMONIC_VPSLLQ) ? 0 :
                       (m == ZYDIS_MNEMONIC_VPSRLW || m == ZYDIS_MNEMONIC_VPSRLD || m == ZYDIS_MNEMONIC_VPSRLQ) ? 1 : 2;
            if (has_imm)
                ext_shift_imm128(d, n, esize, kind, imm & 0xff);
            else
                ext_shift_cnt128(d, n, s, esize, kind);
            return;
        }
        case ZYDIS_MNEMONIC_VPSLLVD: ext_shift_var128(d, n, s, 32, 0); return;
        case ZYDIS_MNEMONIC_VPSLLVQ: ext_shift_var128(d, n, s, 64, 0); return;
        case ZYDIS_MNEMONIC_VPSRLVD: ext_shift_var128(d, n, s, 32, 1); return;
        case ZYDIS_MNEMONIC_VPSRLVQ: ext_shift_var128(d, n, s, 64, 1); return;
        case ZYDIS_MNEMONIC_VPSRAVD: ext_shift_var128(d, n, s, 32, 2); return;

        // Blends
        case ZYDIS_MNEMONIC_VBLENDPS:
            ext_emit_vblend32_128(d, n, s, (u8)imm, T0);
            return;
        case ZYDIS_MNEMONIC_VPBLENDW:
        case ZYDIS_MNEMONIC_VBLENDPD: {
            int bytes = m == ZYDIS_MNEMONIC_VPBLENDW ? 2 : 8;
            int lanes = 16 / bytes;
            int mask = imm & ((1 << lanes) - 1);
            if (mask == 0) {
                vmov128(d, n);
                return;
            }
            if (mask == (1 << lanes) - 1) {
                vmov128(d, s);
                return;
            }
            if (bytes == 2) {
                if (mask == 0x0F || mask == 0xF0) {
                    int lane64 = mask == 0xF0 ? 1 : 0;
                    int dst = (d == s && d != n) ? T0 : d;
                    vmov128(dst, n);
                    ins_elem(dst, lane64, s, lane64, 8);
                    vmov128(d, dst);
                    return;
                }
                if (mask == 0x55) {
                    RR(0x6E600800u, T0, s);         // rev32 .8h
                    RRR(0x4E406800u, d, T0, n);     // trn2 .8h
                    return;
                }
                if (mask == 0xAA) {
                    RR(0x6E600800u, T0, n);         // rev32 .8h
                    RRR(0x4E406800u, d, T0, s);     // trn2 .8h
                    return;
                }
                if (mask == 0x33 || mask == 0xCC) {
                    ext_emit_vblend32_128(d, n, s, mask == 0x33 ? 0x5 : 0xA, T0);
                    return;
                }
                if (((mask & 0x55) << 1) == (mask & 0xAA)) {
                    u8 m4 = (u8)((mask & 1) | ((mask >> 1) & 2) | ((mask >> 2) & 4) | ((mask >> 3) & 8));
                    ext_emit_vblend32_128(d, n, s, m4, T0);
                    return;
                }
            }
            int s_cnt = __builtin_popcount((unsigned)mask);
            bool use_n = (lanes - s_cnt) >= s_cnt;
            int base_v = use_n ? n : s, patch_v = use_n ? s : n;
            int dst = (d == patch_v && d != base_v) ? T0 : d;
            vmov128(dst, base_v);
            for (int i = 0; i < lanes; i++) {
                bool from_s = (mask >> i) & 1;
                if (from_s == use_n)
                    ins_elem(dst, i, patch_v, i, bytes);
            }
            vmov128(d, dst);
            return;
        }
        case ZYDIS_MNEMONIC_VPBLENDVB:
        case ZYDIS_MNEMONIC_VBLENDVPS:
        case ZYDIS_MNEMONIC_VBLENDVPD: {
            // Sign bit selects s, else n. BSL/BIT/BIF read the mask as a full bit mask, so CMLT #0 first.
            u32 cmlt = m == ZYDIS_MNEMONIC_VPBLENDVB ? 0x4E20A800u : m == ZYDIS_MNEMONIC_VBLENDVPS ? 0x4EA0A800u : 0x4EE0A800u;
            if (d != n && d != s) {
                RR(cmlt, d, c);                  // d = mask
                RRR(0x6E601C00u, d, s, n);       // bsl d, s, n
            } else if (d == n) {
                RR(cmlt, T0, c);
                RRR(0x6EA01C00u, d, s, T0);      // bit d, s, mask: keep d where the mask bit is 0
            } else {
                RR(cmlt, T0, c);
                RRR(0x6EE01C00u, d, n, T0);      // bif d, n, mask: keep d where the mask bit is 1
            }
            return;
        }

        // Horizontal add/sub & addsub & round & cmpps/pd & movdup
        case ZYDIS_MNEMONIC_VHADDPS:
        case ZYDIS_MNEMONIC_VHADDPD: {
            u32 fsz = m == ZYDIS_MNEMONIC_VHADDPD ? 1u << 22 : 0;
            RRR(0x6E20D400u | fsz, d, n, s);
            return;
        }
        case ZYDIS_MNEMONIC_VHSUBPS:
        case ZYDIS_MNEMONIC_VHSUBPD: {
            u32 fsz = m == ZYDIS_MNEMONIC_VHSUBPD ? 1u << 22 : 0;
            u32 zsz = m == ZYDIS_MNEMONIC_VHSUBPD ? 3u << 22 : 2u << 22;
            RRR(0x4E001800u | zsz, T0, n, s);
            RRR(0x4E005800u | zsz, T1, n, s);
            RRR(0x4EA0D400u | fsz, d, T0, T1);
            return;
        }
        case ZYDIS_MNEMONIC_VPHADDW:
        case ZYDIS_MNEMONIC_VPHADDD: {
            u32 sz = m == ZYDIS_MNEMONIC_VPHADDW ? 1 : 2;
            RRR(0x4E20BC00u | sz << 22, d, n, s);
            return;
        }
        case ZYDIS_MNEMONIC_VPHADDSW:
        case ZYDIS_MNEMONIC_VPHSUBW:
        case ZYDIS_MNEMONIC_VPHSUBD:
        case ZYDIS_MNEMONIC_VPHSUBSW: {
            u32 sz = m == ZYDIS_MNEMONIC_VPHSUBD ? 2 : 1;
            RRR(0x4E001800u | sz << 22, T0, n, s);
            RRR(0x4E005800u | sz << 22, T1, n, s);
            if (m == ZYDIS_MNEMONIC_VPHADDSW)
                RRR(0x4E200C00u | sz << 22, d, T0, T1);
            else if (m == ZYDIS_MNEMONIC_VPHSUBSW)
                RRR(0x4E202C00u | sz << 22, d, T0, T1);
            else
                RRR(0x6E208400u | sz << 22, d, T0, T1);
            return;
        }
        case ZYDIS_MNEMONIC_VADDSUBPS:
        case ZYDIS_MNEMONIC_VADDSUBPD: {
            bool isd = m == ZYDIS_MNEMONIC_VADDSUBPD;
            if (!tr_precise_flags) {
                if (isd) {
                    RRR(0x4EE0D400u, T0, n, s); // fsub .2d
                    RRR(0x4E60D400u, T1, n, s); // fadd .2d
                    ins_elem(T0, 1, T1, 1, 8);
                    vmov128(d, T0);
                } else {
                    RRR(0x4EA0D400u, T0, n, s); // fsub .4s
                    RRR(0x4E20D400u, T1, n, s); // fadd .4s
                    RR(0x4EA00800u, T0, T0);    // rev64 .4s
                    RRR(0x4E806800u, d, T0, T1); // trn2 .4s
                }
                return;
            }
            int bytes = isd ? 8 : 4, lanes = 16 / bytes;
            u32 ftype = isd ? 1u << 22 : 0;
            for (int i = 0; i < lanes; i++) {
                ins_elem(T0, 0, n, i, bytes);
                ins_elem(T1, 0, s, i, bytes);
                RRR((i & 1 ? 0x1E202800u : 0x1E203800u) | ftype, T0, T0, T1);
                ins_elem(T2, i, T0, 0, bytes);
            }
            vmov128(d, T2);
            return;
        }
        case ZYDIS_MNEMONIC_VROUNDPS:
        case ZYDIS_MNEMONIC_VROUNDPD: {
            static const u32 fixed[4] = {0, 2, 1, 3};
            static const u32 packed[8] = {0x4E218800u, 0x4EA18800u, 0x4E219800u, 0x4EA19800u, 0, 0, 0x6E219800u, 0x6EA19800u};
            u32 rmode = (imm & 4) ? ((imm & 8) ? 7 : 6) : fixed[imm & 3];
            bool dbl = m == ZYDIS_MNEMONIC_VROUNDPD;
            RR(packed[rmode] | (dbl ? 1u << 22 : 0), d, n);
            return;
        }
        case ZYDIS_MNEMONIC_VCMPPS:
        case ZYDIS_MNEMONIC_VCMPPD:
            emit_vcmp_mask(d, n, s, imm, m == ZYDIS_MNEMONIC_VCMPPD);
            return;
        case ZYDIS_MNEMONIC_VPERMILPS:
            if (has_imm) {
                if (!ext_emit_pshufd128(d, n, (u8)imm, T0))
                    ext_lanes32(d, n, n, imm);
            } else {
                ext_permilps128(d, n, s);
            }
            return;
        case ZYDIS_MNEMONIC_VPERMILPD:
            if (has_imm) {
                int im = imm & 3;
                if (im == 0) VDUPQ_64(d, n, 0);
                else if (im == 3) VDUPQ_64(d, n, 1);
                else if (im == 1) VEXTQ_8(d, n, n, 8);
                else vmov128(d, n);
            } else {
                ext_permilpd128(d, n, s);
            }
            return;
        case ZYDIS_MNEMONIC_VPSADBW:
            RRR(0x6E207400u, T0, n, s);
            RR(0x6E202800u, T0, T0);
            RR(0x6E602800u, T0, T0);
            RR(0x6EA02800u, d, T0);
            return;
        case ZYDIS_MNEMONIC_VMOVSLDUP:
            RRR(0x4E802800u, d, n, n); // trn1 .4s
            return;
        case ZYDIS_MNEMONIC_VMOVSHDUP:
            RRR(0x4E806800u, d, n, n); // trn2 .4s
            return;
        case ZYDIS_MNEMONIC_VMOVDDUP:
            VDUPQ_64(d, n, 0);
            return;

        // VEX AES-NI & VPCLMULQDQ
        case ZYDIS_MNEMONIC_VAESIMC:
            AESIMC(d, n);
            return;
        case ZYDIS_MNEMONIC_VAESENC:
        case ZYDIS_MNEMONIC_VAESENCLAST:
        case ZYDIS_MNEMONIC_VAESDEC:
        case ZYDIS_MNEMONIC_VAESDECLAST: {
            bool dec = m == ZYDIS_MNEMONIC_VAESDEC || m == ZYDIS_MNEMONIC_VAESDECLAST;
            bool last = m == ZYDIS_MNEMONIC_VAESENCLAST || m == ZYDIS_MNEMONIC_VAESDECLAST;
            ext_emit_vaes128(d, n, s, dec, last, T0, T1);
            return;
        }
        case ZYDIS_MNEMONIC_VAESKEYGENASSIST:
            VEORQ(T0, T0, T0);
            vmov128(T1, n);
            AESE(T1, T0);
            movimm(S0, 0x040B0E010B0E0104ULL);
            fmov_to_d(T2, S0);
            movimm(S0, 0x0C0306090306090CULL);
            ins_x(T2, 1, S0);
            RRR(0x4E000000u, d, T1, T2); // tbl
            movimm(S0, (u64)(imm & 0xff) << 32);
            fmov_to_d(T0, S0);
            ins_x(T0, 1, S0);
            VEORQ(d, d, T0);
            return;
        case ZYDIS_MNEMONIC_VPCLMULQDQ:
            ins_elem(T0, 0, n, imm & 1, 8);
            ins_elem(T1, 0, s, (imm >> 4) & 1, 8);
            RRR(0x0EE0E000u, d, T0, T1);
            return;
        case ZYDIS_MNEMONIC_VDPPS:
        case ZYDIS_MNEMONIC_VDPPD: {
            // FEX VDPPSOpImpl: mask products, pairwise-add against zero, broadcast.
            // YMM VDPPS is two independent 128-bit operations, which is how the caller splits it.
            bool dbl = m == ZYDIS_MNEMONIC_VDPPD;
            unsigned srcm = dbl ? ((unsigned)imm >> 4) & 3u : ((unsigned)imm >> 4) & 0xfu;
            unsigned dstm = dbl ? (unsigned)imm & 3u : (unsigned)imm & 0xfu;
            unsigned full = dbl ? 3u : 0xfu;
            if (srcm == 0 || dstm == 0) {
                VEORQ(d, d, d);
                return;
            }
            RRR(dbl ? 0x6E60DC00u : 0x6E20DC00u, T0, n, s);
            if (srcm != full) {
                if (dbl) {
                    if (srcm == 1)
                        RR(0x1E604000u, T0, T0); // fmov d, d: clear lane 1
                    else
                        ins_gpr(T0, 0, XZR, 8);
                } else if (srcm == 0x3) {
                    RR(0x1E604000u, T0, T0); // clear lanes 2,3
                } else if (srcm == 0xC) {
                    ins_gpr(T0, 0, XZR, 8);  // clear lanes 0,1
                } else {
                    for (int i = 0; i < 4; i++)
                        if (!(srcm & (1u << i)))
                            ins_gpr(T0, i, XZR, 4);
                }
            }
            VEORQ(T1, T1, T1);
            RRR(dbl ? 0x6E60D400u : 0x6E20D400u, T0, T0, T1);
            if (!dbl)
                RRR(0x6E20D400u, T0, T0, T1);
            if (dstm == full) {
                if (dbl) VDUPQ_64(d, T0, 0);
                else VDUPQ_32(d, T0, 0);
                return;
            }
            VEORQ(T1, T1, T1);
            int lanes = dbl ? 2 : 4, es = dbl ? 8 : 4;
            for (int i = 0; i < lanes; i++)
                if (dstm & (1u << i))
                    ins_elem(T1, i, T0, 0, es);
            vmov128(d, T1);
            return;
        }

        default:
            break;
    }
}

static void emit_fma128(int d, int op1, int op2, int op3, int order, int kind, bool is_scalar, bool is_double) {
    enum { T0 = 27, T1 = 28 };
    int mulA = order == 132 ? op1 : op2;
    int mulB = order == 213 ? op1 : op3;
    int acc  = order == 132 ? op2 : (order == 213 ? op3 : op1);
    // vfmadd231 / vfnmadd231: acc ± a*b in-place when d == acc. NEON FMLA/FMLS reads sources
    // before writing d, so a destination aliasing a multiplier is exact.
    if (!is_scalar && (kind == 0 || kind == 2) && d == acc) {
        if (kind == 0) {
            if (is_double) VFMLAQD(d, mulA, mulB); else VFMLAQS(d, mulA, mulB);
        } else {
            if (is_double) VFMLSQD(d, mulA, mulB); else VFMLSQS(d, mulA, mulB);
        }
        return;
    }
    // FEX VFMAddSubImpl: XOR the sign bit of even (FMADDSUB) or odd (FMSUBADD)
    // addend lanes, then one FMLA. That is a single rounding of product ± addend.
    if (kind >= 4) {
        bool sub_even = kind == 4;
        if (!is_double) {
            E(0x4F046400u | (u32)T1); // movi T1.4s, #0x80, lsl #24
            if (sub_even)
                RR(0x6F600400u, T1, T1); // ushr T1.2d, T1.2d, #32 -> [0x80000000, 0, 0x80000000, 0]
            else
                RR(0x4F605400u, T1, T1); // shl T1.2d, T1.2d, #32 -> [0, 0x80000000, 0, 0x80000000]
        } else {
            E(0x0F046400u | (u32)T1); // movi T1.2s, #0x80, lsl #24 (d0=0x80000000_80000000, d1=0)
            RR(0x5F605400u, T1, T1);  // shl dT1, dT1, #32 -> [0x80000000_00000000, 0]
            if (!sub_even)
                VEXTQ_8(T1, T1, T1, 8); // [0, 0x80000000_00000000]
        }
        VEORQ(T0, acc, T1);
        if (is_double) VFMLAQD(T0, mulA, mulB); else VFMLAQS(T0, mulA, mulB);
        vmov128(d, T0);
        return;
    }
    if (is_scalar) {
        // ARM64 scalar FMADD(d, a, b, c) = c + a*b; FMSUB = c - a*b; FNMADD = -c - a*b; FNMSUB = -c + a*b
        // x86: FMADD = a*b + c (FMADD), FMSUB = a*b - c (FNMSUB), FNMADD = -a*b + c (FMSUB), FNMSUB = -a*b - c (FNMADD)
        u32 base = is_double ? 0x1F400000u : 0x1F000000u;
        u32 arm_op = kind == 0 ? 0u : (kind == 1 ? 0x00208000u : (kind == 2 ? 0x00008000u : 0x00200000u));
        E(base | arm_op | (u32)mulB << 16 | (u32)acc << 10 | (u32)mulA << 5 | T0);
        if (d != op1)
            vmov128(d, op1);
        ins_elem(d, 0, T0, 0, is_double ? 8 : 4);
        return;
    }
    // Packed 128-bit FMA:
    // kind 0 (FMADD):  acc + mulA * mulB -> T0 = acc; FMLA(T0, mulA, mulB)
    // kind 1 (FMSUB): -acc + mulA * mulB -> T0 = -acc; FMLA(T0, mulA, mulB)
    // kind 2 (FNMADD): acc - mulA * mulB -> T0 = acc; FMLS(T0, mulA, mulB)
    // kind 3 (FNMSUB):-acc - mulA * mulB -> T0 = -acc; FMLS(T0, mulA, mulB)
    int dst = (d != mulA && d != mulB) ? d : T0;
    if (kind == 1 || kind == 3) {
        if (is_double) VFNEGQD(dst, acc); else VFNEGQS(dst, acc);
    } else {
        vmov128(dst, acc);
    }
    if (kind == 0 || kind == 1) {
        if (is_double) VFMLAQD(dst, mulA, mulB); else VFMLAQS(dst, mulA, mulB);
    } else {
        if (is_double) VFMLSQD(dst, mulA, mulB); else VFMLSQS(dst, mulA, mulB);
    }
    vmov128(d, dst);
}

// S0 holds the sign-extended element index. The address is left in S1.
static void gather_addr(const Insn *in) {
    if (in->riprel) {
        tr_emit_movimm_gva(S1, (u64)(in->next + in->disp), false);
        add_reg(1, 0, 0, S1, S1, S0, in->scale);
    } else {
        if (in->no_base)
            add_reg(1, 0, 0, S1, XZR, S0, in->scale);
        else
            add_reg(1, 0, 0, S1, in->base, S0, in->scale);
        if (in->disp > 0 && in->disp < 4096)
            add_imm(1, 0, 0, S1, S1, (u32)in->disp);
        else if (in->disp < 0 && in->disp > -4096)
            add_imm(1, 1, 0, S1, S1, (u32)-in->disp);
        else if (in->disp) {
            movimm(S4, (u64)in->disp);
            add_reg(1, 0, 0, S1, S1, S4, 0);
        }
    }
    if (in->seg == 1)
        add_reg(1, 0, 0, S1, S1, R_FS, 0);
    else if (in->seg == 2)
        add_reg(1, 0, 0, S1, S1, R_GS, 0);
    if (in->addr32)
        mov_reg(0, S1, S1);
    else if (in->addr16)
        ubfx(0, S1, S1, 0, 16);
}

static void emit_mpsadbw_lane(int vd, int vs1, int vs2, u8 lane_imm) {
    int s1_off = (lane_imm & 4) ? 4 : 0;
    int s2_off = (lane_imm & 3) * 4;
    int acc = 27;
    for (int k = 0; k < 4; k++) {
        u32 imm5_b = (u32)(((s2_off + k) << 1) | 1);
        E(0x4E000400u | (imm5_b << 16) | ((u32)vs2 << 5) | V_S2); // DUP V_S2.16B, vs2.B[s2_off+k]
        VEXTQ_8(V_S1, vs1, vs1, s1_off + k);
        if (k == 0)
            E(0x2E207000u | ((u32)V_S2 << 16) | ((u32)V_S1 << 5) | acc); // UABDL acc.8H, V_S1.8B, V_S2.8B
        else
            E(0x2E205000u | ((u32)V_S2 << 16) | ((u32)V_S1 << 5) | acc); // UABAL acc.8H, V_S1.8B, V_S2.8B
    }
    vmov128(vd, acc);
}

bool ext_insn_keeps_ymm_cache(const Insn *in) {
    if (in->op != OP_EXT)
        return false;
    const ExtDecoded *ex = &in->ext;
    ZydisMnemonic mnem = (ZydisMnemonic)ex->mnem;
    int fma_ord = 0, fma_kind = 0;
    bool fma_sc = false, fma_dbl = false;
    if (is_fma_mnemonic(mnem, &fma_ord, &fma_kind, &fma_sc, &fma_dbl))
        return true;
    if (is_lanewise_vex(mnem)) {
        bool has_imm = false;
        for (u8 i = 0; i < ex->op_cnt; i++)
            if (ex->op_type[i] == EXT_OP_IMM)
                has_imm = true;
        bool uses_t2 = mnem == ZYDIS_MNEMONIC_VPSIGNB || mnem == ZYDIS_MNEMONIC_VPSIGNW ||
                       mnem == ZYDIS_MNEMONIC_VPSIGND || mnem == ZYDIS_MNEMONIC_VPMADDUBSW ||
                       (tr_precise_flags && (mnem == ZYDIS_MNEMONIC_VADDSUBPS || mnem == ZYDIS_MNEMONIC_VADDSUBPD)) ||
                       (mnem == ZYDIS_MNEMONIC_VPERMILPS && !has_imm) ||
                       mnem == ZYDIS_MNEMONIC_VAESKEYGENASSIST;
        return !uses_t2;
    }
    switch (mnem) {
        case ZYDIS_MNEMONIC_ANDN: case ZYDIS_MNEMONIC_BEXTR: case ZYDIS_MNEMONIC_BLSI:
        case ZYDIS_MNEMONIC_BLSMSK: case ZYDIS_MNEMONIC_BLSR: case ZYDIS_MNEMONIC_BZHI:
        case ZYDIS_MNEMONIC_MULX: case ZYDIS_MNEMONIC_RORX: case ZYDIS_MNEMONIC_SARX:
        case ZYDIS_MNEMONIC_SHLX: case ZYDIS_MNEMONIC_SHRX: case ZYDIS_MNEMONIC_PDEP:
        case ZYDIS_MNEMONIC_PEXT: case ZYDIS_MNEMONIC_ADCX: case ZYDIS_MNEMONIC_ADOX:
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVUPS: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQ: case ZYDIS_MNEMONIC_VMOVNTDQA:
        case ZYDIS_MNEMONIC_VMOVNTPS: case ZYDIS_MNEMONIC_VMOVNTPD:
        case ZYDIS_MNEMONIC_VLDDQU:
        case ZYDIS_MNEMONIC_VMOVSS: case ZYDIS_MNEMONIC_VMOVSD:
        case ZYDIS_MNEMONIC_VMOVD: case ZYDIS_MNEMONIC_VMOVQ:
        case ZYDIS_MNEMONIC_VMOVLPS: case ZYDIS_MNEMONIC_VMOVLPD:
        case ZYDIS_MNEMONIC_VMOVHPS: case ZYDIS_MNEMONIC_VMOVHPD:
        case ZYDIS_MNEMONIC_VMOVLHPS: case ZYDIS_MNEMONIC_VMOVHLPS:
        case ZYDIS_MNEMONIC_VADDSS: case ZYDIS_MNEMONIC_VADDSD:
        case ZYDIS_MNEMONIC_VSUBSS: case ZYDIS_MNEMONIC_VSUBSD:
        case ZYDIS_MNEMONIC_VMULSS: case ZYDIS_MNEMONIC_VMULSD:
        case ZYDIS_MNEMONIC_VDIVSS: case ZYDIS_MNEMONIC_VDIVSD:
        case ZYDIS_MNEMONIC_VMINSS: case ZYDIS_MNEMONIC_VMINSD:
        case ZYDIS_MNEMONIC_VMAXSS: case ZYDIS_MNEMONIC_VMAXSD:
        case ZYDIS_MNEMONIC_VSQRTSS: case ZYDIS_MNEMONIC_VSQRTSD:
        case ZYDIS_MNEMONIC_VRSQRTSS: case ZYDIS_MNEMONIC_VRCPSS:
        case ZYDIS_MNEMONIC_VCVTSI2SS: case ZYDIS_MNEMONIC_VCVTSI2SD:
        case ZYDIS_MNEMONIC_VCVTSS2SI: case ZYDIS_MNEMONIC_VCVTTSS2SI:
        case ZYDIS_MNEMONIC_VCVTSD2SI: case ZYDIS_MNEMONIC_VCVTTSD2SI:
        case ZYDIS_MNEMONIC_VCVTSS2SD: case ZYDIS_MNEMONIC_VCVTSD2SS:
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
        case ZYDIS_MNEMONIC_VCOMISS: case ZYDIS_MNEMONIC_VCOMISD:
        case ZYDIS_MNEMONIC_VBROADCASTF128: case ZYDIS_MNEMONIC_VBROADCASTI128:
        case ZYDIS_MNEMONIC_VBROADCASTSS: case ZYDIS_MNEMONIC_VBROADCASTSD:
        case ZYDIS_MNEMONIC_VPBROADCASTB: case ZYDIS_MNEMONIC_VPBROADCASTW:
        case ZYDIS_MNEMONIC_VPBROADCASTD: case ZYDIS_MNEMONIC_VPBROADCASTQ:
        case ZYDIS_MNEMONIC_VPBLENDD:
        case ZYDIS_MNEMONIC_VINSERTF128: case ZYDIS_MNEMONIC_VINSERTI128:
        case ZYDIS_MNEMONIC_VEXTRACTF128: case ZYDIS_MNEMONIC_VEXTRACTI128:
        case ZYDIS_MNEMONIC_VPERM2F128: case ZYDIS_MNEMONIC_VPERM2I128:
        case ZYDIS_MNEMONIC_VPERMQ: case ZYDIS_MNEMONIC_VPERMPD:
        case ZYDIS_MNEMONIC_VPEXTRB: case ZYDIS_MNEMONIC_VPEXTRW:
        case ZYDIS_MNEMONIC_VPEXTRD: case ZYDIS_MNEMONIC_VPEXTRQ:
        case ZYDIS_MNEMONIC_VPINSRB: case ZYDIS_MNEMONIC_VPINSRW:
        case ZYDIS_MNEMONIC_VPINSRD: case ZYDIS_MNEMONIC_VPINSRQ:
        case ZYDIS_MNEMONIC_VINSERTPS: case ZYDIS_MNEMONIC_VEXTRACTPS:
        case ZYDIS_MNEMONIC_VROUNDSS: case ZYDIS_MNEMONIC_VROUNDSD:
        case ZYDIS_MNEMONIC_VCMPSS: case ZYDIS_MNEMONIC_VCMPSD:
        case ZYDIS_MNEMONIC_VPMOVSXBW: case ZYDIS_MNEMONIC_VPMOVSXBD: case ZYDIS_MNEMONIC_VPMOVSXBQ:
        case ZYDIS_MNEMONIC_VPMOVSXWD: case ZYDIS_MNEMONIC_VPMOVSXWQ: case ZYDIS_MNEMONIC_VPMOVSXDQ:
        case ZYDIS_MNEMONIC_VPMOVZXBW: case ZYDIS_MNEMONIC_VPMOVZXBD: case ZYDIS_MNEMONIC_VPMOVZXBQ:
        case ZYDIS_MNEMONIC_VPMOVZXWD: case ZYDIS_MNEMONIC_VPMOVZXWQ: case ZYDIS_MNEMONIC_VPMOVZXDQ:
        case ZYDIS_MNEMONIC_VPMOVMSKB: case ZYDIS_MNEMONIC_VMOVMSKPS: case ZYDIS_MNEMONIC_VMOVMSKPD:
        case ZYDIS_MNEMONIC_VPTEST:
        case ZYDIS_MNEMONIC_VCVTDQ2PS: case ZYDIS_MNEMONIC_VCVTPS2DQ: case ZYDIS_MNEMONIC_VCVTTPS2DQ:
            return true;
        default:
            return false;
    }
}

void ext_insn_ymm_rw(const Insn *in, u16 *out_read_256, u16 *out_write_any) {
    u16 rd = 0, wr = 0;
    if (in->op != OP_EXT) {
        *out_read_256 = 0;
        *out_write_any = 0;
        return;
    }
    const ExtDecoded *ex = &in->ext;
    ZydisMnemonic mnem = (ZydisMnemonic)ex->mnem;
    if (mnem == ZYDIS_MNEMONIC_VZEROUPPER || mnem == ZYDIS_MNEMONIC_VZEROALL) {
        *out_read_256 = 0;
        *out_write_any = 0xFFFFu;
        return;
    }
    bool op0_read_only = (mnem == ZYDIS_MNEMONIC_VPTEST || mnem == ZYDIS_MNEMONIC_VTESTPS ||
                          mnem == ZYDIS_MNEMONIC_VTESTPD || mnem == ZYDIS_MNEMONIC_VUCOMISS ||
                          mnem == ZYDIS_MNEMONIC_VUCOMISD || mnem == ZYDIS_MNEMONIC_VCOMISS ||
                          mnem == ZYDIS_MNEMONIC_VCOMISD || mnem == ZYDIS_MNEMONIC_VPCMPISTRI);
    int fma_ord = 0, fma_kind = 0;
    bool fma_sc = false, fma_dbl = false;
    bool op0_is_rw = is_fma_mnemonic(mnem, &fma_ord, &fma_kind, &fma_sc, &fma_dbl);
    for (u8 i = 0; i < ex->op_cnt; i++) {
        if (ex->op_type[i] != EXT_OP_VEC)
            continue;
        u16 bit = (u16)(1u << ex->op_reg[i]);
        if (i == 0) {
            if (op0_read_only) {
                if (ex->op_bytes[0] == 32)
                    rd |= bit;
            } else {
                wr |= bit;
                if (op0_is_rw && ex->op_bytes[0] == 32)
                    rd |= bit;
            }
        } else {
            if (ex->op_bytes[i] == 32)
                rd |= bit;
        }
    }
    *out_read_256 = rd;
    *out_write_any = wr;
}

bool ext_insn_s5_free(const Insn *in) {
    if (in->op != OP_EXT)
        return true;
    ZydisMnemonic mnem = (ZydisMnemonic)in->ext.mnem;
    switch (mnem) {
        case ZYDIS_MNEMONIC_PDEP:
        case ZYDIS_MNEMONIC_PEXT:
        case ZYDIS_MNEMONIC_VPCMPISTRI:
            return false;
        default:
            return true;
    }
}

bool ext_insn_s4_free(const Insn *in) {
    if (in->op != OP_EXT)
        return true;
    ZydisMnemonic mnem = (ZydisMnemonic)in->ext.mnem;
    switch (mnem) {
        case ZYDIS_MNEMONIC_ADCX:
        case ZYDIS_MNEMONIC_ADOX:
        case ZYDIS_MNEMONIC_BEXTR:
        case ZYDIS_MNEMONIC_BZHI:
        case ZYDIS_MNEMONIC_PDEP:
        case ZYDIS_MNEMONIC_PEXT:
        case ZYDIS_MNEMONIC_VCVTPS2PH:
        case ZYDIS_MNEMONIC_VPHMINPOSUW:
        case ZYDIS_MNEMONIC_VPCMPISTRI:
        case ZYDIS_MNEMONIC_VGATHERDPS:
        case ZYDIS_MNEMONIC_VGATHERDPD:
        case ZYDIS_MNEMONIC_VGATHERQPS:
        case ZYDIS_MNEMONIC_VGATHERQPD:
        case ZYDIS_MNEMONIC_VPGATHERDD:
        case ZYDIS_MNEMONIC_VPGATHERDQ:
        case ZYDIS_MNEMONIC_VPGATHERQD:
        case ZYDIS_MNEMONIC_VPGATHERQQ:
            return false;
        default:
            break;
    }
    if (in->memop) {
        if (in->riprel || in->no_base || !in->no_index || in->seg || in->addr32 || in->addr16 || tr_compat)
            return false;
        if (in->disp < -256 || in->disp > 1008 || (in->disp & 15) != 0)
            return false;
    }
    return true;
}

u16 ext_insn_written_gprs(const Insn *in) {
    if (in->op != OP_EXT)
        return 0;
    const ExtDecoded *ex = &in->ext;
    ZydisMnemonic mnem = (ZydisMnemonic)ex->mnem;
    if (mnem == ZYDIS_MNEMONIC_VPCMPISTRI)
        return (u16)(1u << 1); // writes RCX/ECX
    if (mnem == ZYDIS_MNEMONIC_XGETBV)
        return (u16)((1u << 0) | (1u << 2));
    if (mnem == ZYDIS_MNEMONIC_MULX) {
        u16 m = 0;
        if (ex->op_cnt >= 1 && ex->op_type[0] == EXT_OP_GPR) m |= (u16)(1u << ex->op_reg[0]);
        if (ex->op_cnt >= 2 && ex->op_type[1] == EXT_OP_GPR) m |= (u16)(1u << ex->op_reg[1]);
        return m;
    }
    if (ex->op_cnt >= 1 && ex->op_type[0] == EXT_OP_GPR)
        return (u16)(1u << ex->op_reg[0]);
    return 0;
}

void emit_ext(const Insn *in, int live) {
    const ExtDecoded *ex = &in->ext;
    ZydisMnemonic mnem = (ZydisMnemonic)ex->mnem;
    u8 op_cnt = ex->op_cnt;
    if (mnem == ZYDIS_MNEMONIC_INVALID) {
        emit_stop(in->rip, "ext: invalid cached opcode");
        return;
    }

    int fma_ord = 0, fma_kind = 0;
    bool fma_sc = false, fma_dbl = false;
    if (is_fma_mnemonic(mnem, &fma_ord, &fma_kind, &fma_sc, &fma_dbl)) {
        u8 vbytes = ex->op_bytes[0];
        int d = ex->op_reg[0];
        int n = ex->op_reg[1];
        if (fma_sc) {
            int s = ex->op_type[2] == EXT_OP_VEC
                        ? ex->op_reg[2]
                        : (ext_vldst(fma_dbl ? 3 : 2, true, V_MEM, addr_prepare(in, fma_dbl ? 8 : 4)), V_MEM);
            emit_fma128(d, d, n, s, fma_ord, fma_kind, true, fma_dbl);
            ymm_zero_hi(d);
            return;
        }
        if (vbytes == 16) {
            int s = ext_vec_src128(in, 2, V_MEM);
            emit_fma128(d, d, n, s, fma_ord, fma_kind, false, fma_dbl);
            ymm_zero_hi(d);
            return;
        }
        if (tr_opt_ymm_cache) {
            u8 pin = 0;
            int s_lo = V_MEM, s_hi_r = V_S2;
            if (ex->op_type[2] == EXT_OP_VEC) {
                int sr = ex->op_reg[2];
                s_lo = sr;
                s_hi_r = ymm_cache_get(sr, true, false, &pin);
            } else {
                ext_load256_pair(in, V_MEM, V_S2);
            }
            int n_hi_r = ymm_cache_get(n, true, false, &pin);
            int d_hi_r = ymm_cache_get(d, true, true, &pin);
            emit_fma128(d, d, n, s_lo, fma_ord, fma_kind, false, fma_dbl);
            emit_fma128(d_hi_r, d_hi_r, n_hi_r, s_hi_r, fma_ord, fma_kind, false, fma_dbl);
            if (d >= 8)
                ymm_dirty_run_cnt++;
            return;
        }
        int s_lo = V_MEM, s_hi_r = V_S2;
        if (ex->op_type[2] == EXT_OP_VEC) {
            int sr = ex->op_reg[2];
            s_lo = sr;
            int home = ymm_resident(sr);
            if (home >= 0)
                s_hi_r = home;
            else
                ymm_load_hi(V_S2, sr);
        } else {
            ext_load256_pair(in, V_MEM, V_S2);
        }
        int d_home = ymm_resident(d);
        int n_home = ymm_resident(n);
        int d_hi_r = d_home;
        int n_hi_r = n_home;
        if (d_home < 0) {
            ymm_load_hi(V_S0, d);
            d_hi_r = V_S0;
        }
        if (n_home < 0) {
            ymm_load_hi(V_S1, n);
            n_hi_r = V_S1;
        }
        emit_fma128(d, d, n, s_lo, fma_ord, fma_kind, false, fma_dbl);
        emit_fma128(d_hi_r, d_hi_r, n_hi_r, s_hi_r, fma_ord, fma_kind, false, fma_dbl);
        if (d_home < 0)
            ymm_store_hi(d_hi_r, d);
        else
            ymm_zero_mask &= (u16)~(1u << d);
        return;
    }

    if (is_lanewise_vex(mnem)) {
        u8 vbytes = ex->op_bytes[0];
        int d = ex->op_reg[0];
        int n = ex->op_reg[1];
        int s_idx = 2;
        bool two_vec = (op_cnt == 2) || (op_cnt == 3 && ex->op_type[2] == EXT_OP_IMM);
        if (two_vec)
            s_idx = 1;
        int imm = (int)in->imm;
        bool has_imm = false;
        int c_reg = 0;
        for (u8 i = 0; i < op_cnt; i++) {
            if (ex->op_type[i] == EXT_OP_IMM) {
                has_imm = true;
            }
        }
        if (mnem == ZYDIS_MNEMONIC_VPBLENDVB ||
            mnem == ZYDIS_MNEMONIC_VBLENDVPS ||
            mnem == ZYDIS_MNEMONIC_VBLENDVPD) {
            c_reg = ex->op_reg[3];
        }
        bool cnt_is_128 = (mnem == ZYDIS_MNEMONIC_VPSLLW || mnem == ZYDIS_MNEMONIC_VPSLLD ||
                           mnem == ZYDIS_MNEMONIC_VPSLLQ || mnem == ZYDIS_MNEMONIC_VPSRLW ||
                           mnem == ZYDIS_MNEMONIC_VPSRLD || mnem == ZYDIS_MNEMONIC_VPSRLQ ||
                           mnem == ZYDIS_MNEMONIC_VPSRAW || mnem == ZYDIS_MNEMONIC_VPSRAD) && !has_imm;
        bool uses_t2 = mnem == ZYDIS_MNEMONIC_VPSIGNB || mnem == ZYDIS_MNEMONIC_VPSIGNW ||
                       mnem == ZYDIS_MNEMONIC_VPSIGND || mnem == ZYDIS_MNEMONIC_VPMADDUBSW ||
                       (tr_precise_flags && (mnem == ZYDIS_MNEMONIC_VADDSUBPS || mnem == ZYDIS_MNEMONIC_VADDSUBPD)) ||
                       (mnem == ZYDIS_MNEMONIC_VPERMILPS && !has_imm) ||
                       mnem == ZYDIS_MNEMONIC_VAESKEYGENASSIST;
        if (uses_t2)
            ymm_cache_evict_vreg(29);

        if (vbytes == 16) {
            int s = 0;
            if (two_vec) {
                n = ext_vec_src128(in, 1, V_MEM);
            } else {
                s = ext_vec_src128(in, s_idx, V_MEM);
            }
            emit_lane128_op(mnem, d, n, s, c_reg, imm, has_imm);
            ymm_zero_hi(d);
            return;
        }

        // 256-bit YMM lane-wise operation
        if ((mnem == ZYDIS_MNEMONIC_VXORPS || mnem == ZYDIS_MNEMONIC_VXORPD ||
             mnem == ZYDIS_MNEMONIC_VPXOR) &&
            ex->op_type[s_idx] == EXT_OP_VEC && ex->op_reg[s_idx] == (u8)n) {
            VEORQ(d, d, d);
            ymm_zero_hi(d);
            return;
        }
        int hi_imm = (mnem == ZYDIS_MNEMONIC_VBLENDPS) ? (imm >> 4) :
                     (mnem == ZYDIS_MNEMONIC_VBLENDPD || mnem == ZYDIS_MNEMONIC_VSHUFPD ||
                      mnem == ZYDIS_MNEMONIC_VPERMILPD) ? (imm >> 2) : imm;
        if (tr_opt_ymm_cache) {
            u8 pin = uses_t2 ? (u8)(1u << 2) : 0; // slot 2 is v29 (T2)
            int n_lo = n, n_hi = V_S2, s_lo = 0, s_hi = V_S2, c_lo = c_reg, c_hi = c_reg;
            if (two_vec) {
                if (ex->op_type[1] == EXT_OP_VEC) {
                    int nr = ex->op_reg[1];
                    n_lo = nr;
                    n_hi = ymm_cache_get(nr, true, false, &pin);
                } else {
                    ext_load256_pair(in, V_MEM, V_S2);
                    n_lo = V_MEM;
                    n_hi = V_S2;
                }
            } else if (cnt_is_128) {
                n_hi = ymm_cache_get(n, true, false, &pin);
                s_lo = ext_vec_src128(in, s_idx, V_MEM);
                s_hi = s_lo;
            } else {
                n_hi = ymm_cache_get(n, true, false, &pin);
                if (ex->op_type[s_idx] == EXT_OP_VEC) {
                    int sr = ex->op_reg[s_idx];
                    s_lo = sr;
                    s_hi = ymm_cache_get(sr, true, false, &pin);
                } else {
                    ext_load256_pair(in, V_MEM, V_S2);
                    s_lo = V_MEM;
                    s_hi = V_S2;
                }
            }
            if (mnem == ZYDIS_MNEMONIC_VPBLENDVB ||
                mnem == ZYDIS_MNEMONIC_VBLENDVPS ||
                mnem == ZYDIS_MNEMONIC_VBLENDVPD) {
                c_hi = ymm_cache_get(c_reg, true, false, &pin);
            }
            int d_hi = ymm_cache_get(d, false, true, &pin);
            if (d_hi >= 0) {
                emit_lane128_op(mnem, d_hi, n_hi, s_hi, c_hi, hi_imm, has_imm);
                emit_lane128_op(mnem, d, n_lo, s_lo, c_lo, imm, has_imm);
                if (d >= 8)
                    ymm_dirty_run_cnt++;
                return;
            }
            ext_ymm_cache_flush_and_drop();
        }
        int n_lo = n, n_hi = V_S0, s_lo = 0, s_hi = V_S2, c_lo = c_reg, c_hi = c_reg;
        if (two_vec) {
            if (ex->op_type[1] == EXT_OP_VEC) {
                int nr = ex->op_reg[1];
                n_lo = nr;
                int nh = ymm_resident(nr);
                if (nh >= 0) n_hi = nh;
                else ymm_load_hi(V_S0, nr);
            } else {
                ext_load256_pair(in, V_MEM, V_S0);
                n_lo = V_MEM;
            }
        } else if (cnt_is_128) {
            // Shift count operand is always 128-bit XMM/m128 for both halves!
            int nh = ymm_resident(n);
            if (nh >= 0) n_hi = nh;
            else ymm_load_hi(V_S0, n);
            s_lo = ext_vec_src128(in, s_idx, V_MEM);
            s_hi = s_lo;
        } else {
            int nh = ymm_resident(n);
            if (nh >= 0) n_hi = nh;
            else ymm_load_hi(V_S0, n);
            if (ex->op_type[s_idx] == EXT_OP_VEC) {
                int sr = ex->op_reg[s_idx];
                s_lo = sr;
                int sh = ymm_resident(sr);
                if (sh >= 0) s_hi = sh;
                else ymm_load_hi(V_S2, sr);
            } else {
                ext_load256_pair(in, V_MEM, V_S2);
                s_lo = V_MEM;
            }
        }
        // Compute high 128-bit half first so if d == s_lo (e.g. cnt_is_128), s_lo is not clobbered before high half uses it.
        if (mnem == ZYDIS_MNEMONIC_VPBLENDVB ||
            mnem == ZYDIS_MNEMONIC_VBLENDVPS ||
            mnem == ZYDIS_MNEMONIC_VBLENDVPD) {
            int ch = ymm_resident(c_reg);
            if (ch >= 0) c_hi = ch;
            else { ymm_load_hi(V_S1, c_reg); c_hi = V_S1; }
        }
        int d_home = ymm_resident(d);
        bool d_hi_direct = d_home >= 0 && !(cnt_is_128 && d == s_lo) &&
                           !(mnem == ZYDIS_MNEMONIC_VPBLENDVB && d == 0);
        int d_hi = d_hi_direct ? d_home : V_S0;
        emit_lane128_op(mnem, d_hi, n_hi, s_hi, c_hi, hi_imm, has_imm);
        emit_lane128_op(mnem, d, n_lo, s_lo, c_lo, imm, has_imm);
        if (!d_hi_direct)
            ymm_store_hi(d_hi, d);
        else
            ymm_zero_mask &= (u16)~(1u << d);
        return;
    }

    // Rare vector/SHA-NI switch cases use V_S0/V_S1/T2 directly; preserve cache across integer/BMI/ADX and all cache-safe VEX cases.
    switch (mnem) {
        case ZYDIS_MNEMONIC_XGETBV: case ZYDIS_MNEMONIC_XSETBV:
        case ZYDIS_MNEMONIC_ANDN: case ZYDIS_MNEMONIC_BEXTR: case ZYDIS_MNEMONIC_BLSI:
        case ZYDIS_MNEMONIC_BLSMSK: case ZYDIS_MNEMONIC_BLSR: case ZYDIS_MNEMONIC_BZHI:
        case ZYDIS_MNEMONIC_MULX: case ZYDIS_MNEMONIC_RORX: case ZYDIS_MNEMONIC_SARX:
        case ZYDIS_MNEMONIC_SHLX: case ZYDIS_MNEMONIC_SHRX: case ZYDIS_MNEMONIC_PDEP:
        case ZYDIS_MNEMONIC_PEXT: case ZYDIS_MNEMONIC_ADCX: case ZYDIS_MNEMONIC_ADOX:
        case ZYDIS_MNEMONIC_VZEROUPPER: case ZYDIS_MNEMONIC_VZEROALL:
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVUPS: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQ: case ZYDIS_MNEMONIC_VMOVNTDQA:
        case ZYDIS_MNEMONIC_VMOVNTPS: case ZYDIS_MNEMONIC_VMOVNTPD:
        case ZYDIS_MNEMONIC_VLDDQU:
        case ZYDIS_MNEMONIC_VMOVSS: case ZYDIS_MNEMONIC_VMOVSD:
        case ZYDIS_MNEMONIC_VMOVD: case ZYDIS_MNEMONIC_VMOVQ:
        case ZYDIS_MNEMONIC_VMOVLPS: case ZYDIS_MNEMONIC_VMOVLPD:
        case ZYDIS_MNEMONIC_VMOVHPS: case ZYDIS_MNEMONIC_VMOVHPD:
        case ZYDIS_MNEMONIC_VMOVLHPS: case ZYDIS_MNEMONIC_VMOVHLPS:
        case ZYDIS_MNEMONIC_VADDSS: case ZYDIS_MNEMONIC_VADDSD:
        case ZYDIS_MNEMONIC_VSUBSS: case ZYDIS_MNEMONIC_VSUBSD:
        case ZYDIS_MNEMONIC_VMULSS: case ZYDIS_MNEMONIC_VMULSD:
        case ZYDIS_MNEMONIC_VDIVSS: case ZYDIS_MNEMONIC_VDIVSD:
        case ZYDIS_MNEMONIC_VMINSS: case ZYDIS_MNEMONIC_VMINSD:
        case ZYDIS_MNEMONIC_VMAXSS: case ZYDIS_MNEMONIC_VMAXSD:
        case ZYDIS_MNEMONIC_VSQRTSS: case ZYDIS_MNEMONIC_VSQRTSD:
        case ZYDIS_MNEMONIC_VRSQRTSS: case ZYDIS_MNEMONIC_VRCPSS:
        case ZYDIS_MNEMONIC_VCVTSI2SS: case ZYDIS_MNEMONIC_VCVTSI2SD:
        case ZYDIS_MNEMONIC_VCVTSS2SI: case ZYDIS_MNEMONIC_VCVTTSS2SI:
        case ZYDIS_MNEMONIC_VCVTSD2SI: case ZYDIS_MNEMONIC_VCVTTSD2SI:
        case ZYDIS_MNEMONIC_VCVTSS2SD: case ZYDIS_MNEMONIC_VCVTSD2SS:
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
        case ZYDIS_MNEMONIC_VCOMISS: case ZYDIS_MNEMONIC_VCOMISD:
        case ZYDIS_MNEMONIC_VBROADCASTF128: case ZYDIS_MNEMONIC_VBROADCASTI128:
        case ZYDIS_MNEMONIC_VBROADCASTSS: case ZYDIS_MNEMONIC_VBROADCASTSD:
        case ZYDIS_MNEMONIC_VPBROADCASTB: case ZYDIS_MNEMONIC_VPBROADCASTW:
        case ZYDIS_MNEMONIC_VPBROADCASTD: case ZYDIS_MNEMONIC_VPBROADCASTQ:
        case ZYDIS_MNEMONIC_VPBLENDD:
        case ZYDIS_MNEMONIC_VINSERTF128: case ZYDIS_MNEMONIC_VINSERTI128:
        case ZYDIS_MNEMONIC_VEXTRACTF128: case ZYDIS_MNEMONIC_VEXTRACTI128:
        case ZYDIS_MNEMONIC_VPERM2F128: case ZYDIS_MNEMONIC_VPERM2I128:
        case ZYDIS_MNEMONIC_VPERMQ: case ZYDIS_MNEMONIC_VPERMPD:
        case ZYDIS_MNEMONIC_VPEXTRB: case ZYDIS_MNEMONIC_VPEXTRW:
        case ZYDIS_MNEMONIC_VPEXTRD: case ZYDIS_MNEMONIC_VPEXTRQ:
        case ZYDIS_MNEMONIC_VPINSRB: case ZYDIS_MNEMONIC_VPINSRW:
        case ZYDIS_MNEMONIC_VPINSRD: case ZYDIS_MNEMONIC_VPINSRQ:
        case ZYDIS_MNEMONIC_VINSERTPS: case ZYDIS_MNEMONIC_VEXTRACTPS:
        case ZYDIS_MNEMONIC_VROUNDSS: case ZYDIS_MNEMONIC_VROUNDSD:
        case ZYDIS_MNEMONIC_VCMPSS: case ZYDIS_MNEMONIC_VCMPSD:
        case ZYDIS_MNEMONIC_VPMOVSXBW: case ZYDIS_MNEMONIC_VPMOVSXBD: case ZYDIS_MNEMONIC_VPMOVSXBQ:
        case ZYDIS_MNEMONIC_VPMOVSXWD: case ZYDIS_MNEMONIC_VPMOVSXWQ: case ZYDIS_MNEMONIC_VPMOVSXDQ:
        case ZYDIS_MNEMONIC_VPMOVZXBW: case ZYDIS_MNEMONIC_VPMOVZXBD: case ZYDIS_MNEMONIC_VPMOVZXBQ:
        case ZYDIS_MNEMONIC_VPMOVZXWD: case ZYDIS_MNEMONIC_VPMOVZXWQ: case ZYDIS_MNEMONIC_VPMOVZXDQ:
        case ZYDIS_MNEMONIC_VPMOVMSKB: case ZYDIS_MNEMONIC_VMOVMSKPS: case ZYDIS_MNEMONIC_VMOVMSKPD:
        case ZYDIS_MNEMONIC_VPTEST:
        case ZYDIS_MNEMONIC_VCVTDQ2PS: case ZYDIS_MNEMONIC_VCVTPS2DQ: case ZYDIS_MNEMONIC_VCVTTPS2DQ:
            break;
        default:
            ext_ymm_cache_flush_and_drop();
            break;
    }

    switch (mnem) {
        case ZYDIS_MNEMONIC_XGETBV: {
            u32 *bad_ecx = tr_cc;
            cbz(1, 0, 1, 0);
            LDRx_U12(S0, CTX, (u32)offsetof(EngineCtx, cr4));
            u32 *bad_cr4 = tr_cc;
            tbz(0, S0, 18, 0);
            LDRx_U12(S0, CTX, (u32)offsetof(EngineCtx, xcr0));
            mov_reg(0, 0, S0);
            lsr_imm(1, 2, S0, 32);
            u32 *done = tr_cc;
            b_rel(0);
            *bad_ecx |= ((u32)(tr_cc - bad_ecx) & 0x7ffff) << 5;
            *bad_cr4 |= ((u32)(tr_cc - bad_cr4) & 0x3fff) << 5;
            svc(SVC_UD);
            *done |= (u32)(tr_cc - done) & 0x3ffffff;
            return;
        }
        case ZYDIS_MNEMONIC_XSETBV: {
            mov_reg(0, S0, 0);
            bfi(1, S0, 2, 32, 32);
            STRx_U12(S0, CTX, (u32)offsetof(EngineCtx, xcr0));
            return;
        }

        // ---- BMI1 ----
        case ZYDIS_MNEMONIC_ANDN: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src1 = ex->op_reg[1];
            int src2 = ext_gpr_src(in, 2, sz, S0);
            logic_reg(sf, 0, 1, dst, src2, src1);
            emit_bmi_flags(sf, dst, live, 0, 0);
            return;
        }
        case ZYDIS_MNEMONIC_BEXTR: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8, width = sf ? 64 : 32;
            int src1 = ext_gpr_src(in, 1, sz, S0);
            int src2 = ex->op_reg[2];
            ubfx(1, S1, src2, 0, 8);
            ubfx(1, S2, src2, 8, 8);
            var_shift(sf, 1, S3, src1, S1);
            add_imm(1, 1, 1, XZR, S1, (u32)width);
            csel(sf, S3, S3, XZR, 3);
            movimm(S1, ~0ULL);
            var_shift(1, 0, S4, S1, S2);
            add_imm(1, 1, 1, XZR, S2, (u32)width);
            csel(1, S4, S4, XZR, 3);
            logic_reg(sf, 0, 1, dst, S3, S4);
            emit_bmi_flags(sf, dst, live, 0, 0);
            return;
        }
        case ZYDIS_MNEMONIC_BLSI: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src = ext_gpr_src(in, 1, sz, S0);
            if (live & F_C) {
                add_imm(sf, 1, 1, XZR, src, 0);
                cset(S3, 1);
            }
            add_reg(sf, 1, 0, S1, XZR, src, 0);
            logic_reg(sf, 0, 0, dst, S1, src);
            emit_bmi_flags(sf, dst, live, 1, S3);
            return;
        }
        case ZYDIS_MNEMONIC_BLSMSK: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src = ext_gpr_src(in, 1, sz, S0);
            if (live & F_C) {
                add_imm(sf, 1, 1, XZR, src, 0);
                cset(S3, 0);
            }
            add_imm(sf, 1, 0, S1, src, 1);
            logic_reg(sf, 2, 0, dst, S1, src);
            emit_bmi_flags(sf, dst, live, 1, S3);
            return;
        }
        case ZYDIS_MNEMONIC_BLSR: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src = ext_gpr_src(in, 1, sz, S0);
            if (live & F_C) {
                add_imm(sf, 1, 1, XZR, src, 0);
                cset(S3, 0);
            }
            add_imm(sf, 1, 0, S1, src, 1);
            logic_reg(sf, 0, 0, dst, S1, src);
            emit_bmi_flags(sf, dst, live, 1, S3);
            return;
        }

        // ---- BMI2 ----
        case ZYDIS_MNEMONIC_BZHI: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8, width = sf ? 64 : 32;
            int src1 = ext_gpr_src(in, 1, sz, S0);
            int src2 = ex->op_reg[2];
            ubfx(1, S1, src2, 0, 8);
            movimm(S2, ~0ULL);
            var_shift(1, 0, S2, S2, S1);
            add_imm(1, 1, 1, XZR, S1, (u32)width);
            csel(1, S2, S2, XZR, 3);
            cset(S3, 2);
            logic_reg(sf, 0, 1, dst, src1, S2);
            emit_bmi_flags(sf, dst, live, 1, S3);
            return;
        }
        case ZYDIS_MNEMONIC_MULX: {
            u8 sz = ex->op_bytes[0];
            int hi = ex->op_reg[0];
            int lo = ex->op_reg[1];
            int sf = sz == 8;
            int src2 = ext_gpr_src(in, 2, sz, S0);
            if (sf) {
                mul(1, S1, 2, src2);
                umulh(S2, 2, src2);
                mov_reg(1, lo, S1);
                mov_reg(1, hi, S2);
            } else {
                mov_reg(0, S1, 2);
                mov_reg(0, S2, src2);
                mul(1, S1, S1, S2);
                mov_reg(0, lo, S1);
                lsr_imm(1, hi, S1, 32);
            }
            return;
        }
        case ZYDIS_MNEMONIC_RORX: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src = ext_gpr_src(in, 1, sz, S0);
            int count = (int)((u64)in->imm & (sf ? 63 : 31));
            extr(sf, dst, src, src, count);
            return;
        }
        case ZYDIS_MNEMONIC_SARX:
        case ZYDIS_MNEMONIC_SHLX:
        case ZYDIS_MNEMONIC_SHRX: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src1 = ext_gpr_src(in, 1, sz, S0);
            int src2 = ex->op_reg[2];
            int op2 = mnem == ZYDIS_MNEMONIC_SHLX ? 0 : (mnem == ZYDIS_MNEMONIC_SHRX ? 1 : 2);
            var_shift(sf, op2, dst, src1, src2);
            return;
        }
        case ZYDIS_MNEMONIC_PDEP: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src1 = ex->op_reg[1];
            int mask = ext_gpr_src(in, 2, sz, S0);
            mov_reg(sf, S1, src1);
            mov_reg(sf, S2, mask);
            mov_reg(1, S3, XZR);
            u32 *skip = tr_cc;
            cbz(0, sf, S2, 0);
            u32 *loop = tr_cc;
            add_reg(sf, 1, 0, S4, XZR, S2, 0);
            logic_reg(sf, 0, 0, S4, S4, S2);
            sbfx(sf, S5, S1, 0, 1);
            logic_reg(sf, 0, 0, S5, S5, S4);
            logic_reg(sf, 1, 0, S3, S3, S5);
            lsr_imm(sf, S1, S1, 1);
            logic_reg(sf, 2, 0, S2, S2, S4);
            cbz(1, sf, S2, (int)(loop - tr_cc));
            *skip |= ((u32)(tr_cc - skip) & 0x7ffff) << 5;
            mov_reg(sf, dst, S3);
            return;
        }
        case ZYDIS_MNEMONIC_PEXT: {
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src1 = ex->op_reg[1];
            int mask = ext_gpr_src(in, 2, sz, S0);
            mov_reg(sf, S1, src1);
            mov_reg(sf, S2, mask);
            mov_reg(1, S3, XZR);
            movimm(S4, 1);
            u32 *skip = tr_cc;
            cbz(0, sf, S2, 0);
            u32 *loop = tr_cc;
            add_reg(sf, 1, 0, S5, XZR, S2, 0);
            logic_reg(sf, 0, 0, S5, S5, S2);
            logic_reg(sf, 2, 0, S2, S2, S5);
            logic_reg(sf, 0, 0, S5, S1, S5);
            add_reg(1, 1, 0, S0, XZR, S5, 0);
            logic_reg(1, 1, 0, S5, S5, S0);
            asr_imm(1, S5, S5, 63);
            logic_reg(sf, 0, 0, S5, S5, S4);
            logic_reg(sf, 1, 0, S3, S3, S5);
            lsl_imm(sf, S4, S4, 1);
            cbz(1, sf, S2, (int)(loop - tr_cc));
            *skip |= ((u32)(tr_cc - skip) & 0x7ffff) << 5;
            mov_reg(sf, dst, S3);
            return;
        }

        // ---- ADX ----
        case ZYDIS_MNEMONIC_ADCX:
        case ZYDIS_MNEMONIC_ADOX: {
            bool is_adcx = mnem == ZYDIS_MNEMONIC_ADCX;
            u8 sz = ex->op_bytes[0];
            int dst = ex->op_reg[0];
            int sf = sz == 8;
            int src = ext_gpr_src(in, 1, sz, S0);
            mrs_nzcv(S3);
            if (is_adcx) {
                logic_reg(1, 1, 1, S2, XZR, S3);
                ubfx(1, S2, S2, 29, 1);
            } else {
                ubfx(1, S2, S3, 28, 1);
            }
            if (!sf) {
                mov_reg(0, S1, dst);
                mov_reg(0, S0, src);
                add_reg(1, 0, 0, S1, S1, S0, 0);
                add_reg(1, 0, 0, S1, S1, S2, 0);
                lsr_imm(1, S2, S1, 32);
                mov_reg(0, dst, S1);
            } else {
                add_reg(1, 0, 1, S1, dst, src, 0);
                cset(S4, 2);
                add_reg(1, 0, 1, dst, S1, S2, 0);
                cset(S2, 2);
                logic_reg(1, 1, 0, S2, S4, S2);
            }
            if (live & (is_adcx ? F_C : F_V)) {
                if (is_adcx) {
                    logic_reg(1, 1, 1, S2, XZR, S2);
                    bfi(1, S3, S2, 29, 1);
                } else {
                    bfi(1, S3, S2, 28, 1);
                }
            }
            msr_nzcv(S3);
            return;
        }

        // ---- SHA-NI ----
        case ZYDIS_MNEMONIC_SHA1NEXTE: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            VDUPQ_32(V_S0, vd, 3);
            SHA1H(V_S0, V_S0);
            VEXTQ_8(V_S0, V_S0, V_S0, 4);
            VADDQ_32(vd, vs, V_S0);
            return;
        }
        case ZYDIS_MNEMONIC_SHA1MSG1: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            VEXTQ_8(V_S0, vs, vd, 8);
            VEORQ(vd, vd, V_S0);
            return;
        }
        case ZYDIS_MNEMONIC_SHA1MSG2: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            vrev128_32(V_S0, vd);
            vrev128_32(V_S1, vs);
            SHA1SU1(V_S0, V_S1);
            vrev128_32(vd, V_S0);
            return;
        }
        case ZYDIS_MNEMONIC_SHA1RNDS4: {
            static const u32 sha1_k[4] = {0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u};
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            int func = (int)((u64)in->imm & 3);
            vrev128_32(V_S0, vd);
            vrev128_32(V_S1, vs);
            movimm(S0, sha1_k[func]);
            VDUPQS(V_S2, S0);
            VADDQ_32(V_S1, V_S1, V_S2);
            VEORQ(V_S2, V_S2, V_S2);
            if (func == 0)
                SHA1C(V_S0, V_S2, V_S1);
            else if (func == 2)
                SHA1M(V_S0, V_S2, V_S1);
            else
                SHA1P(V_S0, V_S2, V_S1);
            vrev128_32(vd, V_S0);
            return;
        }
        case ZYDIS_MNEMONIC_SHA256MSG1: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            SHA256SU0(vd, vs);
            return;
        }
        case ZYDIS_MNEMONIC_SHA256MSG2: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            VEORQ(V_S0, V_S0, V_S0);
            VZIP2Q_64(V_S1, V_S0, vs);
            SHA256SU1(vd, V_S0, V_S1);
            return;
        }
        case ZYDIS_MNEMONIC_SHA256RNDS2: {
            int vd = ex->op_reg[0];
            int vs = ext_vec_src128(in, 1, V_MEM);
            VREV64Q_32(V_S0, vs);
            VREV64Q_32(V_S1, vd);
            VZIP2Q_64(V_S2, V_S0, V_S1);
            VZIP1Q_64(V_S0, V_S0, V_S1);
            VMOVQ(V_S1, V_S2);
            SHA256H(V_S2, V_S0, 0);
            SHA256H2(V_S0, V_S1, 0);
            VZIP2Q_64(vd, V_S0, V_S2);
            VREV64Q_32(vd, vd);
            return;
        }

        // ---- Stage 5: Special / Cross-lane / Scalar VEX AVX & AVX2 ----
        case ZYDIS_MNEMONIC_VZEROUPPER:
        case ZYDIS_MNEMONIC_VZEROALL: {
            bool all = mnem == ZYDIS_MNEMONIC_VZEROALL;
            if (all) {
                for (int r = 0; r < 16; r++)
                    VEORQ(r, r, r);
            }
            for (int r = 0; r < 8; r++)
                ymm_zero_hi(r);
            // Drop cached ymm8..15 without flushing dirty (all are being zeroed).
            for (int s = 0; s < YMM_CACHE_SLOTS; s++) {
                ymm_cache[s].xreg = -1;
                ymm_cache[s].dirty = false;
            }
            ymm_dirty_run_cnt = 0;
            u32 hi_need = tr_opt_ymm_cache ? (~(u32)(ymm_zero_mask >> 8) & 0xffu) : 0xffu;
            if (__builtin_popcount(hi_need) >= 3) {
                tr_ad_page = ~0ULL;
                VEORQ(27, 27, 27);
                movimm(AD, ymm_hi_off(8));
                add_reg(1, 0, 0, AD, CTX, AD, 0);
                VSTP128_I7(27, 27, AD, 0);
                VSTP128_I7(27, 27, AD, 32);
                VSTP128_I7(27, 27, AD, 64);
                VSTP128_I7(27, 27, AD, 96);
                if (tr_opt_ymm_cache)
                    ymm_zero_mask = 0xffffu;
            } else {
                for (int r = 8; r < 16; r++)
                    ymm_zero_hi(r);
            }
            return;
        }

        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVUPS: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQ: case ZYDIS_MNEMONIC_VMOVNTDQA:
        case ZYDIS_MNEMONIC_VMOVNTPS: case ZYDIS_MNEMONIC_VMOVNTPD:
        case ZYDIS_MNEMONIC_VLDDQU: {
            if (ex->op_type[0] == EXT_OP_VEC) {
                u8 vbytes = ex->op_bytes[0];
                int d = ex->op_reg[0];
                if (ex->op_type[1] == EXT_OP_VEC) {
                    int s = ex->op_reg[1];
                    vmov128(d, s);
                    if (vbytes == 32) {
                        if (tr_opt_ymm_cache && (ymm_zero_mask & (1u << s))) {
                            ymm_zero_hi(d);
                        } else if (tr_opt_ymm_cache) {
                            u8 pin = 0;
                            int sh = ymm_cache_get(s, true, false, &pin);
                            int dh = ymm_cache_get(d, false, true, &pin);
                            vmov128(dh, sh);
                            if (d >= 8)
                                ymm_dirty_run_cnt++;
                        } else {
                            int sh = ymm_resident(s);
                            if (sh >= 0) {
                                ymm_store_hi(sh, d);
                            } else {
                                ymm_load_hi(V_S0, s);
                                ymm_store_hi(V_S0, d);
                            }
                        }
                    } else {
                        ymm_zero_hi(d);
                    }
                } else if (vbytes == 32) {
                    if (tr_opt_ymm_cache) {
                        u8 pin = 0;
                        int dh = ymm_cache_get(d, false, true, &pin);
                        if (tr_t2_mode) {
                            // Cross-page #PF exactness: load into scratch pair before committing architectural d and resident dh.
                            ext_load256_pair(in, V_MEM, V_S2);
                            vmov128(d, V_MEM);
                            vmov128(dh, V_S2);
                        } else {
                            ext_load256_pair(in, d, dh);
                        }
                        if (d >= 8 && !(ymm_t2_resident_mask & pin))
                            ymm_dirty_run_cnt++;
                    } else {
                        int dh = ymm_resident(d);
                        if (dh >= 0) {
                            ext_load256_pair(in, d, dh);
                        } else {
                            ext_load256_pair(in, d, V_S1);
                            ymm_store_hi(V_S1, d);
                        }
                    }
                } else {
                    ext_vldst(4, true, d, addr_prepare(in, 16));
                    ymm_zero_hi(d);
                }
            } else {
                u8 vbytes = ex->op_bytes[1];
                int s = ex->op_reg[1];
                if (vbytes == 32) {
                    if (tr_opt_ymm_cache) {
                        u8 pin = 0;
                        int sh = ymm_cache_get(s, true, false, &pin);
                        ext_store256_pair(in, s, sh);
                    } else {
                        int sh = ymm_resident(s);
                        if (sh >= 0) {
                            ext_store256_pair(in, s, sh);
                        } else {
                            ymm_load_hi(V_S0, s);
                            ext_store256_pair(in, s, V_S0);
                        }
                    }
                } else {
                    ext_vldst(4, false, s, addr_prepare(in, 16));
                }
            }
            return;
        }

        case ZYDIS_MNEMONIC_VMOVSS:
        case ZYDIS_MNEMONIC_VMOVSD: {
            enum { T0 = 27 };
            bool isd = mnem == ZYDIS_MNEMONIC_VMOVSD;
            int kind = isd ? 3 : 2, bytes = isd ? 8 : 4;
            if (op_cnt == 3) {
                int d = ex->op_reg[0], n = ex->op_reg[1], s = ex->op_reg[2];
                int dst = (d == s && d != n) ? T0 : d;
                vmov128(dst, n);
                ins_elem(dst, 0, s, 0, bytes);
                vmov128(d, dst);
                ymm_zero_hi(d);
            } else if (ex->op_type[0] == EXT_OP_VEC) {
                int d = ex->op_reg[0];
                ext_vldst(kind, true, d, addr_prepare(in, bytes));
                ymm_zero_hi(d);
            } else {
                int s = ex->op_reg[1];
                ext_vldst(kind, false, s, addr_prepare(in, bytes));
            }
            return;
        }

        case ZYDIS_MNEMONIC_VMOVD:
        case ZYDIS_MNEMONIC_VMOVQ: {
            if (ex->op_type[0] == EXT_OP_VEC) {
                int d = ex->op_reg[0];
                if (ex->op_type[1] == EXT_OP_VEC) {
                    RR(0x1E604000u, d, ex->op_reg[1]); // fmov d, ds (clears upper 64 bits)
                } else if (ex->op_type[1] == EXT_OP_GPR) {
                    RR(ex->op_bytes[1] == 8 ? 0x9E670000u : 0x1E270000u, d, ex->op_reg[1]);
                } else {
                    bool is64 = mnem == ZYDIS_MNEMONIC_VMOVQ;
                    ext_vldst(is64 ? 3 : 2, true, d, addr_prepare(in, is64 ? 8 : 4));
                }
                ymm_zero_hi(d);
            } else {
                int s = ex->op_reg[1];
                if (ex->op_type[0] == EXT_OP_GPR) {
                    RR(ex->op_bytes[0] == 8 ? 0x9E660000u : 0x1E260000u, ex->op_reg[0], s);
                } else {
                    bool is64 = mnem == ZYDIS_MNEMONIC_VMOVQ;
                    ext_vldst(is64 ? 3 : 2, false, s, addr_prepare(in, is64 ? 8 : 4));
                }
            }
            return;
        }

        case ZYDIS_MNEMONIC_VMOVLPS:
        case ZYDIS_MNEMONIC_VMOVLPD:
        case ZYDIS_MNEMONIC_VMOVHPS:
        case ZYDIS_MNEMONIC_VMOVHPD: {
            enum { T0 = 27 };
            bool hi = mnem == ZYDIS_MNEMONIC_VMOVHPS || mnem == ZYDIS_MNEMONIC_VMOVHPD;
            if (op_cnt == 3) {
                int d = ex->op_reg[0], n = ex->op_reg[1];
                ext_vldst(3, true, V_MEM, addr_prepare(in, 8));
                vmov128(d, n);
                ins_elem(d, hi ? 1 : 0, V_MEM, 0, 8);
                ymm_zero_hi(d);
            } else {
                int s = ex->op_reg[1];
                Addr a = addr_prepare(in, 8);
                if (hi) {
                    ins_elem(T0, 0, s, 1, 8);
                    ext_vldst(3, false, T0, a);
                } else {
                    ext_vldst(3, false, s, a);
                }
            }
            return;
        }
        case ZYDIS_MNEMONIC_VMOVLHPS:
        case ZYDIS_MNEMONIC_VMOVHLPS: {
            enum { T0 = 27 };
            bool lh = mnem == ZYDIS_MNEMONIC_VMOVLHPS;
            int d = ex->op_reg[0], n = ex->op_reg[1], s = ex->op_reg[2];
            int dst = (d == s && d != n) ? T0 : d;
            vmov128(dst, n);
            ins_elem(dst, lh ? 1 : 0, s, lh ? 0 : 1, 8);
            vmov128(d, dst);
            ymm_zero_hi(d);
            return;
        }

        case ZYDIS_MNEMONIC_VCVTPH2PS: {
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            int s = ex->op_type[1] == EXT_OP_VEC
                        ? ex->op_reg[1]
                        : (ext_vldst(vbytes == 32 ? 4 : 3, true, V_MEM, addr_prepare(in, vbytes == 32 ? 16 : 8)), V_MEM);
            int src = s;
            if (d == s) {
                vmov128(V_S1, s);
                src = V_S1;
            }
            RR(0x0E217800u, d, src); // fcvtl.4s, .4h
            if (vbytes == 32) {
                RR(0x4E217800u, V_S0, src); // fcvtl2
                ymm_store_hi(V_S0, d);
            } else {
                ymm_zero_hi(d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VCVTPS2PH: {
            u8 vbytes = ex->op_bytes[1];
            int s = ex->op_reg[1];
            int imm = (int)in->imm;
            bool mxcsr = (imm & 4) != 0;
            if (!mxcsr) {
                int arm = (int[]){0, 2, 1, 3}[imm & 3];
                E(0xD53B4400u | S4); // mrs FPCR
                movimm(S0, ~(3ULL << 22));
                logic_reg(1, 0, 0, S3, S4, S0);
                if (arm) {
                    movimm(S0, (u64)arm << 22);
                    logic_reg(1, 1, 0, S3, S3, S0);
                }
                E(0xD51B4400u | S3);
            }
            int packed = V_S1;
            if (vbytes == 32) {
                ymm_load_hi(V_S0, s);
                RR(0x0E216800u, packed, s);      // fcvtn low 4
                RR(0x4E216800u, packed, V_S0);   // fcvtn2 high 4
            } else {
                RR(0x0E216800u, packed, s);
            }
            if (!mxcsr)
                E(0xD51B4400u | S4);
            if (ex->op_type[0] == EXT_OP_VEC) {
                int d = ex->op_reg[0];
                vmov128(d, packed);
                ymm_zero_hi(d);
            } else {
                ext_vldst(vbytes == 32 ? 4 : 3, false, packed, addr_prepare(in, vbytes == 32 ? 16 : 8));
            }
            return;
        }

        // Approximate reciprocal / reciprocal-sqrt. Not bit-exact; one Newton step.
        case ZYDIS_MNEMONIC_VRSQRTSS:
        case ZYDIS_MNEMONIC_VRCPSS: {
            enum { T0 = 27, T1 = 28 };
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int s = ex->op_type[2] == EXT_OP_VEC
                        ? ex->op_reg[2]
                        : (ext_vldst(2, true, V_MEM, addr_prepare(in, 4)), V_MEM);
            if (mnem == ZYDIS_MNEMONIC_VRSQRTSS) {
                RR(0x7EA1D800u, T0, s);          // frsqrte
                RRR(0x1E200800u, T1, T0, T0);    // y*y
                RRR(0x5EA0FC00u, T1, s, T1);     // frsqrts
                RRR(0x1E200800u, T0, T0, T1);
            } else {
                RR(0x5EA1D800u, T0, s);          // frecpe
                RRR(0x5E20FC00u, T1, s, T0);     // frecps
                RRR(0x1E200800u, T0, T0, T1);
            }
            vmov128(d, n);
            ins_elem(d, 0, T0, 0, 4);
            ymm_zero_hi(d);
            return;
        }
        case ZYDIS_MNEMONIC_VADDSS:  case ZYDIS_MNEMONIC_VADDSD:
        case ZYDIS_MNEMONIC_VSUBSS:  case ZYDIS_MNEMONIC_VSUBSD:
        case ZYDIS_MNEMONIC_VMULSS:  case ZYDIS_MNEMONIC_VMULSD:
        case ZYDIS_MNEMONIC_VDIVSS:  case ZYDIS_MNEMONIC_VDIVSD:
        case ZYDIS_MNEMONIC_VMINSS:  case ZYDIS_MNEMONIC_VMINSD:
        case ZYDIS_MNEMONIC_VMAXSS:  case ZYDIS_MNEMONIC_VMAXSD:
        case ZYDIS_MNEMONIC_VSQRTSS: case ZYDIS_MNEMONIC_VSQRTSD: {
            enum { T0 = 27 };
            bool isd = mnem == ZYDIS_MNEMONIC_VADDSD || mnem == ZYDIS_MNEMONIC_VSUBSD ||
                       mnem == ZYDIS_MNEMONIC_VMULSD || mnem == ZYDIS_MNEMONIC_VDIVSD ||
                       mnem == ZYDIS_MNEMONIC_VMINSD || mnem == ZYDIS_MNEMONIC_VMAXSD ||
                       mnem == ZYDIS_MNEMONIC_VSQRTSD;
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int s = ex->op_type[2] == EXT_OP_VEC
                        ? ex->op_reg[2]
                        : (ext_vldst(isd ? 3 : 2, true, V_MEM, addr_prepare(in, isd ? 8 : 4)), V_MEM);
            u32 fsz = isd ? 1u << 22 : 0;
            bool direct_d = (d == n) && tr_fpcr_merges_scalar;
            int res_v = direct_d ? d : T0;
            if (mnem == ZYDIS_MNEMONIC_VSQRTSS || mnem == ZYDIS_MNEMONIC_VSQRTSD) {
                RR(0x1E21C000u | fsz, res_v, s);
            } else {
                u32 base = (mnem == ZYDIS_MNEMONIC_VADDSS || mnem == ZYDIS_MNEMONIC_VADDSD) ? 0x1E202800u :
                           (mnem == ZYDIS_MNEMONIC_VSUBSS || mnem == ZYDIS_MNEMONIC_VSUBSD) ? 0x1E203800u :
                           (mnem == ZYDIS_MNEMONIC_VMULSS || mnem == ZYDIS_MNEMONIC_VMULSD) ? 0x1E200800u :
                           (mnem == ZYDIS_MNEMONIC_VDIVSS || mnem == ZYDIS_MNEMONIC_VDIVSD) ? 0x1E201800u :
                           (mnem == ZYDIS_MNEMONIC_VMINSS || mnem == ZYDIS_MNEMONIC_VMINSD) ? 0x1E205800u : 0x1E204800u;
                RRR(base | fsz, res_v, n, s);
            }
            if (!direct_d) {
                vmov128(d, n);
                ins_elem(d, 0, T0, 0, isd ? 8 : 4);
            }
            ymm_zero_hi(d);
            return;
        }

        // Scalar FP conversions & comparisons
        case ZYDIS_MNEMONIC_VCVTSI2SS:
        case ZYDIS_MNEMONIC_VCVTSI2SD: {
            enum { T0 = 27 };
            bool isd = mnem == ZYDIS_MNEMONIC_VCVTSI2SD;
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int sz = ex->op_bytes[2] == 8 ? 8 : 4;
            int sr = ext_gpr_src(in, 2, sz, S0);
            bool direct_d = (d == n) && tr_fpcr_merges_scalar;
            RR((sz == 8 ? 0x9E220000u : 0x1E220000u) | (isd ? 1u << 22 : 0), direct_d ? d : T0, sr);
            if (!direct_d) {
                vmov128(d, n);
                ins_elem(d, 0, T0, 0, isd ? 8 : 4);
            }
            ymm_zero_hi(d);
            return;
        }
        case ZYDIS_MNEMONIC_VCVTSS2SI:
        case ZYDIS_MNEMONIC_VCVTTSS2SI:
        case ZYDIS_MNEMONIC_VCVTSD2SI:
        case ZYDIS_MNEMONIC_VCVTTSD2SI: {
            enum { T0 = 27 };
            bool isd = mnem == ZYDIS_MNEMONIC_VCVTSD2SI || mnem == ZYDIS_MNEMONIC_VCVTTSD2SI;
            bool trunc = mnem == ZYDIS_MNEMONIC_VCVTTSS2SI || mnem == ZYDIS_MNEMONIC_VCVTTSD2SI;
            u8 gsz = ex->op_bytes[0];
            int gd = ex->op_reg[0];
            int s = ex->op_type[1] == EXT_OP_VEC
                        ? ex->op_reg[1]
                        : (ext_vldst(isd ? 3 : 2, true, V_MEM, addr_prepare(in, isd ? 8 : 4)), V_MEM);
            if (trunc) {
                RR((gsz == 8 ? 0x9E380000u : 0x1E380000u) | (isd ? 1u << 22 : 0), gd, s);
            } else {
                RR(isd ? 0x1E67C000u : 0x1E27C000u, T0, s);
                RR((gsz == 8 ? 0x9E380000u : 0x1E380000u) | (isd ? 1u << 22 : 0), gd, T0);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VCVTSS2SD:
        case ZYDIS_MNEMONIC_VCVTSD2SS: {
            enum { T0 = 27 };
            bool to_dbl = mnem == ZYDIS_MNEMONIC_VCVTSS2SD;
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int s = ex->op_type[2] == EXT_OP_VEC
                        ? ex->op_reg[2]
                        : (ext_vldst(to_dbl ? 2 : 3, true, V_MEM, addr_prepare(in, to_dbl ? 4 : 8)), V_MEM);
            bool direct_d = (d == n) && tr_fpcr_merges_scalar;
            RR(to_dbl ? 0x1E22C000u : 0x1E624000u, direct_d ? d : T0, s);
            if (!direct_d) {
                vmov128(d, n);
                ins_elem(d, 0, T0, 0, to_dbl ? 8 : 4);
            }
            ymm_zero_hi(d);
            return;
        }
        case ZYDIS_MNEMONIC_VCVTPS2PD:
        case ZYDIS_MNEMONIC_VCVTDQ2PD: {
            bool is_int = mnem == ZYDIS_MNEMONIC_VCVTDQ2PD;
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            if (vbytes == 16) {
                int s = ex->op_type[1] == EXT_OP_VEC
                            ? ex->op_reg[1]
                            : (ext_vldst(3, true, V_MEM, addr_prepare(in, 8)), V_MEM);
                if (is_int) {
                    E(0x0F20A400u | (u32)s << 5 | V_S0); // sshll vS0.2d, vS.2s, #0
                    RR(0x4E61D800u, d, V_S0);            // scvtf vD.2d, vS0.2d
                } else {
                    RR(0x0E617800u, d, s);               // fcvtl vD.2d, vS.2s
                }
                ymm_zero_hi(d);
            } else {
                int s = ex->op_type[1] == EXT_OP_VEC
                            ? ex->op_reg[1]
                            : (ext_vldst(4, true, V_MEM, addr_prepare(in, 16)), V_MEM);
                vmov128(V_S0, s);
                if (is_int) {
                    E(0x4F20A400u | (u32)V_S0 << 5 | V_S1); // sshll2 vS1.2d, vS0.4s, #0
                    RR(0x4E61D800u, V_S1, V_S1);            // scvtf vS1.2d, vS1.2d
                    E(0x0F20A400u | (u32)V_S0 << 5 | V_S0); // sshll vS0.2d, vS0.2s, #0
                    RR(0x4E61D800u, d, V_S0);               // scvtf vD.2d, vS0.2d
                } else {
                    RR(0x4E617800u, V_S1, V_S0);            // fcvtl2 vS1.2d, vS0.4s
                    RR(0x0E617800u, d, V_S0);               // fcvtl vD.2d, vS0.2s
                }
                ymm_store_hi(V_S1, d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VCVTPD2PS:
        case ZYDIS_MNEMONIC_VCVTPD2DQ:
        case ZYDIS_MNEMONIC_VCVTTPD2DQ: {
            bool is_ps = mnem == ZYDIS_MNEMONIC_VCVTPD2PS;
            bool trunc = mnem == ZYDIS_MNEMONIC_VCVTTPD2DQ;
            int d = ex->op_reg[0];
            u8 sbytes = ex->op_bytes[1] == 32 ? 32 : 16;
            if (sbytes == 16) {
                int s = ex->op_type[1] == EXT_OP_VEC
                            ? ex->op_reg[1]
                            : (ext_vldst(4, true, V_MEM, addr_prepare(in, 16)), V_MEM);
                if (is_ps) {
                    RR(0x0E616800u, d, s); // fcvtn vD.2s, vS.2d
                } else {
                    if (!tr_precise_flags) {
                        if (trunc) {
                            RR(0x4EE1B800u, V_S0, s);
                        } else {
                            RR(0x6EE19800u, V_S0, s);
                            RR(0x4EE1B800u, V_S0, V_S0);
                        }
                    } else {
                        RR(trunc ? 0x4E61E800u : 0x6E61E800u, V_S0, s);
                        RR(0x4EE1B800u, V_S0, V_S0);
                    }
                    RR(0x0EA12800u, d, V_S0); // xtn vD.2s, vS0.2d
                }
            } else {
                int s_lo = V_MEM, s_hi = V_S2;
                if (ex->op_type[1] == EXT_OP_VEC) {
                    int sr = ex->op_reg[1];
                    vmov128(V_MEM, sr);
                    ymm_load_hi(V_S2, sr);
                } else {
                    ext_load256_pair(in, V_MEM, V_S2);
                }
                if (is_ps) {
                    RR(0x0E616800u, d, s_lo);    // fcvtn vD.2s, s_lo.2d
                    RR(0x4E616800u, d, s_hi);    // fcvtn2 vD.4s, s_hi.2d
                } else {
                    if (!tr_precise_flags) {
                        if (trunc) {
                            RR(0x4EE1B800u, V_S0, s_lo);
                            RR(0x4EE1B800u, V_S1, s_hi);
                        } else {
                            RR(0x6EE19800u, V_S0, s_lo);
                            RR(0x4EE1B800u, V_S0, V_S0);
                            RR(0x6EE19800u, V_S1, s_hi);
                            RR(0x4EE1B800u, V_S1, V_S1);
                        }
                    } else {
                        RR(trunc ? 0x4E61E800u : 0x6E61E800u, V_S0, s_lo);
                        RR(0x4EE1B800u, V_S0, V_S0);
                        RR(trunc ? 0x4E61E800u : 0x6E61E800u, V_S1, s_hi);
                        RR(0x4EE1B800u, V_S1, V_S1);
                    }
                    RR(0x0EA12800u, d, V_S0);    // xtn vD.2s, vS0.2d
                    RR(0x4EA12800u, d, V_S1);    // xtn2 vD.4s, vS1.2d
                }
            }
            ymm_zero_hi(d);
            return;
        }
        case ZYDIS_MNEMONIC_VCVTDQ2PS:
        case ZYDIS_MNEMONIC_VCVTPS2DQ:
        case ZYDIS_MNEMONIC_VCVTTPS2DQ: {
            enum { T0 = 27, T1 = 28 };
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            u8 pin = 0;
            int s_lo = V_MEM, s_hi = V_S2;
            if (ex->op_type[1] == EXT_OP_VEC) {
                int sr = ex->op_reg[1];
                s_lo = sr;
                if (vbytes == 32) {
                    int sh = tr_opt_ymm_cache ? ymm_cache_get(sr, true, false, &pin) : ymm_resident(sr);
                    if (sh >= 0) s_hi = sh;
                    else ymm_load_hi(V_S2, sr);
                }
            } else if (vbytes == 32) {
                ext_load256_pair(in, V_MEM, V_S2);
            } else {
                ext_vldst(4, true, V_MEM, addr_prepare(in, 16));
            }
            int d_hi = T1;
            bool d_hi_cached = false;
            if (vbytes == 32) {
                int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
                if (dh >= 0) {
                    d_hi = dh;
                    d_hi_cached = true;
                }
            }
            for (int lane = 0; lane < (vbytes == 32 ? 2 : 1); lane++) {
                int src = lane == 0 ? s_lo : s_hi;
                int dst = lane == 0 ? d : d_hi;
                if (mnem == ZYDIS_MNEMONIC_VCVTDQ2PS) {
                    RR(0x4E21D800u, dst, src);
                } else if (!tr_precise_flags) {
                    if (mnem == ZYDIS_MNEMONIC_VCVTTPS2DQ) {
                        RR(0x4EA1B800u, dst, src);
                    } else {
                        RR(0x6EA19800u, T0, src);
                        RR(0x4EA1B800u, dst, T0);
                    }
                } else {
                    RR(mnem == ZYDIS_MNEMONIC_VCVTPS2DQ ? 0x6E21E800u : 0x4E21E800u, T0, src);
                    RR(0x4EA1B800u, dst, T0);
                }
            }
            if (vbytes == 32) {
                if (!d_hi_cached)
                    ymm_store_hi(d_hi, d);
                else if (d >= 8)
                    ymm_dirty_run_cnt++;
                else
                    ymm_zero_mask &= (u16)~(1u << d);
            } else {
                ymm_zero_hi(d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
        case ZYDIS_MNEMONIC_VCOMISS:  case ZYDIS_MNEMONIC_VCOMISD: {
            bool isd = mnem == ZYDIS_MNEMONIC_VUCOMISD || mnem == ZYDIS_MNEMONIC_VCOMISD;
            bool sig = mnem == ZYDIS_MNEMONIC_VCOMISS || mnem == ZYDIS_MNEMONIC_VCOMISD;
            int d = ex->op_reg[0];
            int s = ex->op_type[1] == EXT_OP_VEC
                        ? ex->op_reg[1]
                        : (ext_vldst(isd ? 3 : 2, true, V_MEM, addr_prepare(in, isd ? 8 : 4)), V_MEM);
            E((sig ? 0x1E202010u : 0x1E202000u) | (isd ? 1u << 22 : 0) | (u32)s << 16 | (u32)d << 5);
            if (live & F_P)
                cset(R_PF, 7);
            axflag();
            if (live & F_A)
                mov_reg(1, R_AF, R_PF);
            return;
        }

        // Broadcasts
        case ZYDIS_MNEMONIC_VBROADCASTF128:
        case ZYDIS_MNEMONIC_VBROADCASTI128: {
            int d = ex->op_reg[0];
            if (tr_t2_mode) {
                ext_vldst(4, true, V_MEM, addr_prepare(in, 16));
                vmov128(d, V_MEM);
            } else {
                ext_vldst(4, true, d, addr_prepare(in, 16));
            }
            if (tr_opt_ymm_cache) {
                u8 pin = 0;
                int dh = ymm_cache_get(d, false, true, &pin);
                vmov128(dh, d);
                if (d >= 8 && !(ymm_t2_resident_mask & pin))
                    ymm_dirty_run_cnt++;
            } else {
                ymm_store_hi(d, d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VBROADCASTSS:
        case ZYDIS_MNEMONIC_VBROADCASTSD:
        case ZYDIS_MNEMONIC_VPBROADCASTB:
        case ZYDIS_MNEMONIC_VPBROADCASTW:
        case ZYDIS_MNEMONIC_VPBROADCASTD:
        case ZYDIS_MNEMONIC_VPBROADCASTQ: {
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            int ebytes = mnem == ZYDIS_MNEMONIC_VPBROADCASTB ? 1 :
                         mnem == ZYDIS_MNEMONIC_VPBROADCASTW ? 2 :
                         (mnem == ZYDIS_MNEMONIC_VBROADCASTSD || mnem == ZYDIS_MNEMONIC_VPBROADCASTQ) ? 8 : 4;
            int s = V_MEM;
            if (ex->op_type[1] == EXT_OP_VEC) {
                s = ex->op_reg[1];
            } else {
                ext_vldst(log2sz(ebytes), true, V_MEM, addr_prepare(in, ebytes));
            }
            if (ebytes == 1) VDUPQ_8(d, s, 0);
            else if (ebytes == 2) VDUPQ_16(d, s, 0);
            else if (ebytes == 4) VDUPQ_32(d, s, 0);
            else VDUPQ_64(d, s, 0);
            if (vbytes == 32) {
                if (tr_opt_ymm_cache) {
                    u8 pin = 0;
                    int dh = ymm_cache_get(d, false, true, &pin);
                    vmov128(dh, d);
                    if (d >= 8)
                        ymm_dirty_run_cnt++;
                } else {
                    ymm_store_hi(d, d);
                }
            } else {
                ymm_zero_hi(d);
            }
            return;
        }

        // Insert / Extract 128-bit half (YMM cache-aware: uses only T0=27, T1=28, V_S2=26, V_MEM=30)
        case ZYDIS_MNEMONIC_VINSERTF128:
        case ZYDIS_MNEMONIC_VINSERTI128: {
            enum { T0 = 27, T1 = 28 };
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int s = ext_vec_src128(in, 2, V_MEM);
            int lane = (int)((u64)in->imm & 1);
            u8 pin = 0;
            if (lane == 0) {
                int nh = tr_opt_ymm_cache ? ymm_cache_get(n, true, false, &pin) : ymm_resident(n);
                int n_hi = nh >= 0 ? nh : (ymm_load_hi(T0, n), T0);
                int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
                if (dh >= 0) {
                    vmov128(dh, n_hi);
                    if (d >= 8) ymm_dirty_run_cnt++;
                    else ymm_zero_mask &= (u16)~(1u << d);
                } else {
                    ymm_store_hi(n_hi, d);
                }
                vmov128(d, s);
            } else {
                int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
                if (dh >= 0) {
                    vmov128(dh, s);
                    if (d >= 8) ymm_dirty_run_cnt++;
                    else ymm_zero_mask &= (u16)~(1u << d);
                } else {
                    vmov128(T1, s);
                    ymm_store_hi(T1, d);
                }
                vmov128(d, n);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VEXTRACTF128:
        case ZYDIS_MNEMONIC_VEXTRACTI128: {
            enum { T0 = 27 };
            int s = ex->op_reg[1];
            int lane = (int)((u64)in->imm & 1);
            int src = s;
            if (lane == 1) {
                u8 pin = 0;
                int sh = tr_opt_ymm_cache ? ymm_cache_get(s, true, false, &pin) : ymm_resident(s);
                if (sh >= 0) {
                    src = sh;
                } else {
                    ymm_load_hi(T0, s);
                    src = T0;
                }
            }
            if (ex->op_type[0] == EXT_OP_VEC) {
                int d = ex->op_reg[0];
                vmov128(d, src);
                ymm_zero_hi(d);
            } else {
                ext_vldst(4, false, src, addr_prepare(in, 16));
            }
            return;
        }

        // 256-bit cross-lane permutations (YMM cache-aware)
        case ZYDIS_MNEMONIC_VPERM2F128:
        case ZYDIS_MNEMONIC_VPERM2I128: {
            enum { T0 = 27, T1 = 28 };
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int imm = (int)((u64)in->imm & 0xff);
            u8 pin = 0;
            int s_lo = V_MEM, s_hi = V_S2;
            if (ex->op_type[2] == EXT_OP_VEC) {
                int sr = ex->op_reg[2];
                s_lo = sr;
                int sh = tr_opt_ymm_cache ? ymm_cache_get(sr, true, false, &pin) : ymm_resident(sr);
                if (sh >= 0) s_hi = sh;
                else ymm_load_hi(V_S2, sr);
            } else {
                ext_load256_pair(in, V_MEM, V_S2);
            }
            int nh = tr_opt_ymm_cache ? ymm_cache_get(n, true, false, &pin) : ymm_resident(n);
            int n_hi = nh;
            if (n_hi < 0) {
                if (ex->op_type[2] == EXT_OP_VEC) {
                    ymm_load_hi(V_MEM, n);
                    n_hi = V_MEM;
                } else {
                    ext_ymm_cache_flush_and_drop();
                    ymm_load_hi(V_S1, n);
                    n_hi = V_S1;
                }
            }
            int halves[4] = {n, n_hi, s_lo, s_hi};
            if (imm & 0x08) VEORQ(T0, T0, T0); else vmov128(T0, halves[imm & 3]);
            if (imm & 0x80) VEORQ(T1, T1, T1); else vmov128(T1, halves[(imm >> 4) & 3]);
            vmov128(d, T0);
            u8 dpin = 0;
            int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &dpin) : ymm_resident(d);
            if (dh >= 0) {
                vmov128(dh, T1);
                if (d >= 8) ymm_dirty_run_cnt++;
                else ymm_zero_mask &= (u16)~(1u << d);
            } else {
                ymm_store_hi(T1, d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VPERMQ:
        case ZYDIS_MNEMONIC_VPERMPD: {
            enum { T0 = 27, T1 = 28 };
            int d = ex->op_reg[0];
            int imm = (int)((u64)in->imm & 0xff);
            u8 pin = 0;
            int s_lo = V_MEM, s_hi = V_S2;
            if (ex->op_type[1] == EXT_OP_VEC) {
                int sr = ex->op_reg[1];
                s_lo = sr;
                int sh = tr_opt_ymm_cache ? ymm_cache_get(sr, true, false, &pin) : ymm_resident(sr);
                if (sh >= 0) s_hi = sh;
                else ymm_load_hi(V_S2, sr);
            } else {
                ext_load256_pair(in, V_MEM, V_S2);
            }
            int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
            bool direct_hi = dh >= 0 && dh != s_lo && dh != s_hi;
            int out_lo = (d != s_lo && d != s_hi) ? d : T0;
            int out_hi = direct_hi ? dh : T1;
            ext_emit_permq_half(out_lo, s_lo, s_hi, (u8)(imm & 0xf));
            if ((imm & 0xf) == ((imm >> 4) & 0xf))
                vmov128(out_hi, out_lo);
            else
                ext_emit_permq_half(out_hi, s_lo, s_hi, (u8)((imm >> 4) & 0xf));
            vmov128(d, out_lo);
            if (dh >= 0) {
                if (!direct_hi)
                    vmov128(dh, out_hi);
                if (d >= 8) ymm_dirty_run_cnt++;
                else ymm_zero_mask &= (u16)~(1u << d);
            } else {
                ymm_store_hi(out_hi, d);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VPERMD:
        case ZYDIS_MNEMONIC_VPERMPS: {
            enum { T0 = 27, T1 = 28, T2 = 29 };
            int d = ex->op_reg[0];
            int idx_r = ex->op_reg[1];
            if (ex->op_type[2] == EXT_OP_VEC) {
                int sr = ex->op_reg[2];
                vmov128(V_S0, sr);
                ymm_load_hi(V_S1, sr);
            } else {
                ext_load256_pair(in, V_S0, V_S1);
            }
            E(0x4F000400u | (7u << 5) | T2); // movi T2.4s, #7
            movimm(S0, 0x03020100u);
            VDUPQS(V_MEM, S0);
            ext_emit_vpermd_indices(T0, idx_r, T2, V_MEM, V_S2);
            E(0x4E002000u | (u32)T0 << 16 | (u32)V_S0 << 5 | (u32)T0); // tbl T0.16b, {v24.16b, v25.16b}, T0.16b
            int idx_hi = ymm_resident(idx_r);
            if (idx_hi < 0) {
                ymm_load_hi(T1, idx_r);
                idx_hi = T1;
            }
            ext_emit_vpermd_indices(T1, idx_hi, T2, V_MEM, V_S2);
            E(0x4E002000u | (u32)T1 << 16 | (u32)V_S0 << 5 | (u32)T1); // tbl T1.16b, {v24.16b, v25.16b}, T1.16b
            vmov128(d, T0);
            ymm_store_hi(T1, d);
            return;
        }
        case ZYDIS_MNEMONIC_VPBLENDD: {
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int imm = (int)((u64)in->imm & 0xff);
            if (vbytes == 16) {
                int s = ext_vec_src128(in, 2, V_MEM);
                ext_emit_vblend32_128(d, n, s, (u8)(imm & 0xf), 27);
                ymm_zero_hi(d);
            } else if (tr_opt_ymm_cache) {
                u8 pin = 0;
                int s_lo = V_MEM, s_hi = V_S2;
                if (ex->op_type[2] == EXT_OP_VEC) {
                    int sr = ex->op_reg[2];
                    s_lo = sr;
                    s_hi = ymm_cache_get(sr, true, false, &pin);
                } else {
                    ext_load256_pair(in, V_MEM, V_S2);
                }
                int n_hi = ymm_cache_get(n, true, false, &pin);
                int d_hi = ymm_cache_get(d, false, true, &pin);
                ext_emit_vblend32_128(d_hi, n_hi, s_hi, (u8)((imm >> 4) & 0xf), 27);
                ext_emit_vblend32_128(d, n, s_lo, (u8)(imm & 0xf), 27);
                if (d >= 8)
                    ymm_dirty_run_cnt++;
            } else {
                int s_lo = V_MEM, s_hi = V_S2;
                if (ex->op_type[2] == EXT_OP_VEC) {
                    int sr = ex->op_reg[2];
                    s_lo = sr;
                    int sh = ymm_resident(sr);
                    if (sh >= 0) s_hi = sh;
                    else ymm_load_hi(V_S2, sr);
                } else {
                    ext_load256_pair(in, V_MEM, V_S2);
                }
                int nh = ymm_resident(n);
                int n_hi = nh >= 0 ? nh : (ymm_load_hi(V_S0, n), V_S0);
                int dh = ymm_resident(d);
                int d_hi = dh >= 0 ? dh : V_S0;
                ext_emit_vblend32_128(d_hi, n_hi, s_hi, (u8)((imm >> 4) & 0xf), 27);
                ext_emit_vblend32_128(d, n, s_lo, (u8)(imm & 0xf), 27);
                if (dh < 0)
                    ymm_store_hi(d_hi, d);
                else
                    ymm_zero_mask &= (u16)~(1u << d);
            }
            return;
        }

        // VPEXTR* / VPINSR*
        case ZYDIS_MNEMONIC_VPEXTRB:
        case ZYDIS_MNEMONIC_VPEXTRW:
        case ZYDIS_MNEMONIC_VPEXTRD:
        case ZYDIS_MNEMONIC_VPEXTRQ: {
            int bytes = mnem == ZYDIS_MNEMONIC_VPEXTRB ? 1 :
                        mnem == ZYDIS_MNEMONIC_VPEXTRW ? 2 :
                        mnem == ZYDIS_MNEMONIC_VPEXTRD ? 4 : 8;
            int s = ex->op_reg[1];
            int lane = (int)((u64)in->imm & (u64)(16 / bytes - 1));
            u32 umov = (bytes == 8 ? 0x4E003C00u : 0x0E003C00u) | (u32)(lane << (log2sz(bytes) + 1) | bytes) << 16 | (u32)s << 5;
            if (ex->op_type[0] == EXT_OP_GPR) {
                E(umov | (u32)ex->op_reg[0]);
            } else {
                Addr a = addr_prepare(in, bytes);
                E(umov | S0);
                ldst(0, bytes, S0, a);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VPINSRB:
        case ZYDIS_MNEMONIC_VPINSRW:
        case ZYDIS_MNEMONIC_VPINSRD:
        case ZYDIS_MNEMONIC_VPINSRQ: {
            int bytes = mnem == ZYDIS_MNEMONIC_VPINSRB ? 1 :
                        mnem == ZYDIS_MNEMONIC_VPINSRW ? 2 :
                        mnem == ZYDIS_MNEMONIC_VPINSRD ? 4 : 8;
            int d = ex->op_reg[0], n = ex->op_reg[1];
            int sr = ext_gpr_src(in, 2, bytes, S0);
            int lane = (int)((u64)in->imm & (u64)(16 / bytes - 1));
            vmov128(d, n);
            E(0x4E001C00u | (u32)(lane << (log2sz(bytes) + 1) | bytes) << 16 | (u32)sr << 5 | (u32)d);
            ymm_zero_hi(d);
            return;
        }

        // VPMOVSX* / VPMOVZX* (128-bit and 256-bit, YMM cache-aware + 2-insn UXTL2/SXTL2 fast path)
        case ZYDIS_MNEMONIC_VPMOVSXBW: case ZYDIS_MNEMONIC_VPMOVSXBD: case ZYDIS_MNEMONIC_VPMOVSXBQ:
        case ZYDIS_MNEMONIC_VPMOVSXWD: case ZYDIS_MNEMONIC_VPMOVSXWQ: case ZYDIS_MNEMONIC_VPMOVSXDQ:
        case ZYDIS_MNEMONIC_VPMOVZXBW: case ZYDIS_MNEMONIC_VPMOVZXBD: case ZYDIS_MNEMONIC_VPMOVZXBQ:
        case ZYDIS_MNEMONIC_VPMOVZXWD: case ZYDIS_MNEMONIC_VPMOVZXWQ: case ZYDIS_MNEMONIC_VPMOVZXDQ: {
            enum { T0 = 27, T1 = 28 };
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            bool sx = mnem >= ZYDIS_MNEMONIC_VPMOVSXBD && mnem <= ZYDIS_MNEMONIC_VPMOVSXWQ;
            int from = (mnem == ZYDIS_MNEMONIC_VPMOVSXBW || mnem == ZYDIS_MNEMONIC_VPMOVSXBD ||
                        mnem == ZYDIS_MNEMONIC_VPMOVSXBQ || mnem == ZYDIS_MNEMONIC_VPMOVZXBW ||
                        mnem == ZYDIS_MNEMONIC_VPMOVZXBD || mnem == ZYDIS_MNEMONIC_VPMOVZXBQ) ? 0 :
                       (mnem == ZYDIS_MNEMONIC_VPMOVSXDQ || mnem == ZYDIS_MNEMONIC_VPMOVZXDQ) ? 2 : 1;
            int to = (mnem == ZYDIS_MNEMONIC_VPMOVSXBW || mnem == ZYDIS_MNEMONIC_VPMOVZXBW) ? 1 :
                     (mnem == ZYDIS_MNEMONIC_VPMOVSXBD || mnem == ZYDIS_MNEMONIC_VPMOVSXWD ||
                      mnem == ZYDIS_MNEMONIC_VPMOVZXBD || mnem == ZYDIS_MNEMONIC_VPMOVZXWD) ? 2 : 3;
            int half_bytes = 16 >> (to - from);
            int s = V_MEM;
            if (ex->op_type[1] == EXT_OP_VEC) {
                s = ex->op_reg[1];
            } else {
                int load_bytes = vbytes == 32 ? half_bytes * 2 : half_bytes;
                ext_vldst(log2sz(load_bytes), true, V_MEM, addr_prepare(in, load_bytes));
            }
            u32 base = sx ? 0x0F00A400u : 0x2F00A400u;
            if (vbytes == 32 && to - from == 1) {
                u8 pin = 0;
                int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
                int hi_dst = dh >= 0 ? dh : T1;
                RR((base | 0x40000000u) | (1u << (19 + from)), hi_dst, s);
                RR(base | (1u << (19 + from)), d, s);
                if (dh >= 0) {
                    if (d >= 8) ymm_dirty_run_cnt++;
                    else ymm_zero_mask &= (u16)~(1u << d);
                } else {
                    ymm_store_hi(hi_dst, d);
                }
                return;
            }
            vmov128(T0, s);
            for (int w = from; w < to; w++)
                RR(base | 1u << (19 + w), T0, T0);
            if (vbytes == 32) {
                u8 pin = 0;
                int dh = tr_opt_ymm_cache ? ymm_cache_get(d, false, true, &pin) : ymm_resident(d);
                int hi_dst = dh >= 0 ? dh : T1;
                VEXTQ_8(hi_dst, s, s, half_bytes);
                for (int w = from; w < to; w++)
                    RR(base | 1u << (19 + w), hi_dst, hi_dst);
                if (dh >= 0) {
                    if (d >= 8) ymm_dirty_run_cnt++;
                    else ymm_zero_mask &= (u16)~(1u << d);
                } else {
                    ymm_store_hi(hi_dst, d);
                }
            } else {
                ymm_zero_hi(d);
            }
            vmov128(d, T0);
            return;
        }

        // VPMOVMSKB / VMOVMSKPS / VMOVMSKPD (zero-constant USHR+USRA+XTN, single FMOV/UMOV, YMM cache-aware)
        case ZYDIS_MNEMONIC_VPMOVMSKB: {
            enum { T0 = 27, T1 = 28 };
            int gd = ex->op_reg[0];
            u8 vbytes = ex->op_bytes[1];
            int s = ex->op_reg[1];
            if (vbytes == 32) {
                u8 pin = 0;
                int sh = tr_opt_ymm_cache ? ymm_cache_get(s, true, false, &pin) : ymm_resident(s);
                if (sh < 0) {
                    ymm_load_hi(T1, s);
                    sh = T1;
                }
                ext_emit_vpmovmskb256(gd, s, sh, T0, T1);
            } else {
                ext_emit_pmovmskb128(gd, s, T0);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VMOVMSKPS:
        case ZYDIS_MNEMONIC_VMOVMSKPD: {
            enum { T0 = 27, T1 = 28 };
            bool isd = mnem == ZYDIS_MNEMONIC_VMOVMSKPD;
            int gd = ex->op_reg[0];
            u8 vbytes = ex->op_bytes[1];
            int s = ex->op_reg[1];
            if (vbytes == 16) {
                if (!isd) {
                    RR(0x6F210400u, T0, s);    // ushr T0.4s, s.4s, #31
                    RR(0x6F611400u, T0, T0);   // usra T0.2d, T0.2d, #31
                    RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                    RR(0x7F621400u, T0, T0);   // usra dT0, dT0, #30
                    RR(0x1E260000u, gd, T0);   // fmov w_gd, sT0
                } else {
                    RR(0x6F410400u, T0, s);    // ushr T0.2d, s.2d, #63
                    RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                    RR(0x7F611400u, T0, T0);   // usra dT0, dT0, #31
                    RR(0x1E260000u, gd, T0);   // fmov w_gd, sT0
                }
                return;
            }
            u8 pin = 0;
            int sh = tr_opt_ymm_cache ? ymm_cache_get(s, true, false, &pin) : ymm_resident(s);
            if (sh < 0) {
                ymm_load_hi(T1, s);
                sh = T1;
            }
            if (!isd) {
                RR(0x6F210400u, T0, s);    // ushr T0.4s, s.4s, #31
                RR(0x6F210400u, T1, sh);   // ushr T1.4s, sh.4s, #31
                RR(0x6F611400u, T0, T0);   // usra T0.2d, #31
                RR(0x6F611400u, T1, T1);   // usra T1.2d, #31
                RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                RR(0x4EA12800u, T0, T1);   // xtn2 T0.4s, T1.2d
                RR(0x6F621400u, T0, T0);   // usra T0.2d, #30
                RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                RR(0x7F641400u, T0, T0);   // usra dT0, dT0, #28
                RR(0x1E260000u, gd, T0);   // fmov w_gd, sT0
            } else {
                RR(0x6F410400u, T0, s);    // ushr T0.2d, s.2d, #63
                RR(0x6F410400u, T1, sh);   // ushr T1.2d, sh.2d, #63
                RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                RR(0x4EA12800u, T0, T1);   // xtn2 T0.4s, T1.2d
                RR(0x6F611400u, T0, T0);   // usra T0.2d, #31
                RR(0x0EA12800u, T0, T0);   // xtn T0.2s, T0.2d
                RR(0x7F621400u, T0, T0);   // usra dT0, dT0, #30
                RR(0x1E260000u, gd, T0);   // fmov w_gd, sT0
            }
            return;
        }

        // VPTEST (YMM cache-aware + UMAXP + d==s & live==F_Z/F_C fast paths) / VTESTPS / VTESTPD
        case ZYDIS_MNEMONIC_VPTEST: {
            enum { T0 = 27, T1 = 28 };
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            u8 pin = 0;
            bool same_reg = (ex->op_type[1] == EXT_OP_VEC && ex->op_reg[1] == (u8)d);
            int s_lo = V_MEM, s_hi = V_S2;
            if (ex->op_type[1] == EXT_OP_VEC) {
                int sr = ex->op_reg[1];
                s_lo = sr;
                if (vbytes == 32) {
                    int sh = tr_opt_ymm_cache ? ymm_cache_get(sr, true, false, &pin) : ymm_resident(sr);
                    if (sh >= 0) s_hi = sh;
                    else ymm_load_hi(V_S2, sr);
                }
            } else if (vbytes == 32) {
                ext_load256_pair(in, V_MEM, V_S2);
            } else {
                s_lo = ext_vec_src128(in, 1, V_MEM);
            }
            if (!live)
                return;
            int d_hi = -1;
            if (vbytes == 32) {
                if (same_reg) {
                    d_hi = s_hi;
                } else {
                    int dh = tr_opt_ymm_cache ? ymm_cache_get(d, true, false, &pin) : ymm_resident(d);
                    if (dh >= 0) d_hi = dh;
                    else { ymm_load_hi(T1, d); d_hi = T1; }
                }
            }
            if (live == F_Z) {
                int av;
                if (same_reg) {
                    if (vbytes == 32) { VORRQ(T0, d, d_hi); av = T0; }
                    else av = d;
                } else {
                    VANDQ(T0, d, s_lo);
                    if (vbytes == 32) {
                        VANDQ(V_MEM, d_hi, s_hi);
                        VORRQ(T0, T0, V_MEM);
                    }
                    av = T0;
                }
                RRR(0x6EA0A400u, T0, av, av); // umaxp .4s
                RR(0x9E660000u, S0, T0);      // fmov xS0, d27
                add_imm(1, 1, 1, XZR, S0, 0); // cmp xS0, #0 -> Z=(S0==0)
                return;
            }
            if (live == F_C) {
                if (same_reg) {
                    rmif(XZR, 0, 2); // ~d & d == 0 -> CF=1 -> Arm64 C=0
                    return;
                }
                VBICQ(T0, s_lo, d);
                if (vbytes == 32) {
                    VBICQ(V_MEM, s_hi, d_hi);
                    VORRQ(T0, T0, V_MEM);
                }
                RRR(0x6EA0A400u, T0, T0, T0);
                RR(0x9E660000u, S1, T0);
                add_imm(1, 1, 1, XZR, S1, 1); // cmp xS1, #1 -> C=(S1!=0)=!CF
                return;
            }
            if (same_reg) {
                int av = d;
                if (vbytes == 32) { VORRQ(T0, d, d_hi); av = T0; }
                RRR(0x6EA0A400u, T0, av, av);
                RR(0x9E660000u, S0, T0);
                add_imm(1, 1, 1, XZR, S0, 0);
                cset(S0, 0);
                lsl_imm(1, S0, S0, 30);
                msr_nzcv(S0);
            } else {
                VANDQ(T0, d, s_lo);
                if (vbytes == 32) {
                    VANDQ(V_MEM, d_hi, s_hi);
                    VORRQ(T0, T0, V_MEM);
                }
                VBICQ(V_MEM, s_lo, d);
                if (vbytes == 32) {
                    VBICQ(T1, s_hi, d_hi);
                    VORRQ(V_MEM, V_MEM, T1);
                }
                RRR(0x6EA0A400u, T0, T0, V_MEM); // d[0]=any(a), d[1]=any(b)
                RR(0x9E660000u, S0, T0);
                umov_d(S1, T0, 1);
                add_imm(1, 1, 1, XZR, S0, 0);
                cset(S0, 0);
                add_imm(1, 1, 1, XZR, S1, 0);
                cset(S1, 1);
                lsl_imm(1, S0, S0, 30);
                logic_sh(1, 1, 0, S0, S0, S1, 0, 29);
                msr_nzcv(S0);
            }
            if (live & (F_P | F_A)) {
                movimm(R_PF, 1);
                mov_reg(1, R_AF, R_PF);
            }
            return;
        }
        case ZYDIS_MNEMONIC_VTESTPS:
        case ZYDIS_MNEMONIC_VTESTPD: {
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            int s_lo = V_MEM, s_hi = V_S2;
            if (ex->op_type[1] == EXT_OP_VEC) {
                int sr = ex->op_reg[1];
                s_lo = sr;
                if (vbytes == 32)
                    ymm_load_hi(V_S2, sr);
            } else if (vbytes == 32) {
                ext_load256_pair(in, V_MEM, V_S2);
            } else {
                s_lo = ext_vec_src128(in, 1, V_MEM);
            }
            u32 cmlt = mnem == ZYDIS_MNEMONIC_VTESTPD ? 0x4EE0A800u : 0x4EA0A800u;
            enum { T0 = 27, T1 = 28, T2 = 29 };
            RR(cmlt, T0, d);
            RR(cmlt, T1, s_lo);
            VANDQ(V_S0, T0, T1);
            VBICQ(V_S1, T1, T0);
            if (vbytes == 32) {
                ymm_load_hi(T0, d);
                RR(cmlt, T0, T0);
                RR(cmlt, T1, s_hi);
                VANDQ(T2, T0, T1);
                VORRQ(V_S0, V_S0, T2);
                VBICQ(T2, T1, T0);
                VORRQ(V_S1, V_S1, T2);
            }
            RRR(0x6EA0A400u, V_S0, V_S0, V_S1);
            RR(0x9E660000u, S0, V_S0);
            umov_d(S1, V_S0, 1);
            add_imm(1, 1, 1, XZR, S0, 0);
            cset(S0, 0);
            add_imm(1, 1, 1, XZR, S1, 0);
            cset(S1, 1);
            lsl_imm(1, S0, S0, 30);
            logic_sh(1, 1, 0, S0, S0, S1, 0, 29);
            msr_nzcv(S0);
            if (live & (F_P | F_A)) {
                movimm(R_PF, 1);
                mov_reg(1, R_AF, R_PF);
            }
            return;
        }

        case ZYDIS_MNEMONIC_VROUNDSS:
        case ZYDIS_MNEMONIC_VROUNDSD: {
            enum { T0 = 27 };
            bool dbl = mnem == ZYDIS_MNEMONIC_VROUNDSD;
            int bytes = dbl ? 8 : 4;
            int d = ex->op_reg[0], src2 = ex->op_reg[1];
            int imm = (int)in->imm;
            int scalar = V_MEM;
            if (ex->op_type[2] == EXT_OP_VEC)
                scalar = ex->op_reg[2];
            else
                ext_vldst(dbl ? 3 : 2, true, V_MEM, addr_prepare(in, bytes));
            static const u32 fixed[4] = {0, 2, 1, 3};
            u32 rmode = (imm & 4) ? ((imm & 8) ? 7 : 6) : fixed[imm & 3];
            bool report = !(imm & 4) && !(imm & 8);
            if (report)
                E(0x1E244000u | (dbl ? 1u << 22 : 0) | 6u << 15 | (u32)scalar << 5 | T0);
            E(0x1E244000u | (dbl ? 1u << 22 : 0) | rmode << 15 | (u32)scalar << 5 | T0);
            vmov128(d, src2);
            ins_elem(d, 0, T0, 0, bytes);
            ymm_zero_hi(d);
            return;
        }

        case ZYDIS_MNEMONIC_VCMPSS:
        case ZYDIS_MNEMONIC_VCMPSD: {
            enum { T0 = 27, T1 = 28 };
            bool dbl = mnem == ZYDIS_MNEMONIC_VCMPSD;
            int bytes = dbl ? 8 : 4;
            int d = ex->op_reg[0], src1 = ex->op_reg[1];
            int imm = (int)in->imm;
            int scalar = V_MEM;
            if (ex->op_type[2] == EXT_OP_VEC)
                scalar = ex->op_reg[2];
            else
                ext_vldst(dbl ? 3 : 2, true, V_MEM, addr_prepare(in, bytes));
            emit_vcmp_mask(V_S2, src1, scalar, imm, dbl);
            vmov128(d, src1);
            ins_elem(d, 0, V_S2, 0, bytes);
            ymm_zero_hi(d);
            return;
        }

        case ZYDIS_MNEMONIC_VINSERTPS: {
            enum { T0 = 27 };
            int d = ex->op_reg[0], src2 = ex->op_reg[1];
            int imm = (int)in->imm;
            int lane = 0;
            int src3 = V_MEM;
            if (ex->op_type[2] == EXT_OP_VEC) {
                lane = (imm >> 6) & 3;
                src3 = ex->op_reg[2];
            } else {
                ext_vldst(2, true, V_MEM, addr_prepare(in, 4));
            }
            int dst = (d == src3 && d != src2) ? T0 : d;
            vmov128(dst, src2);
            ins_elem(dst, (imm >> 4) & 3, src3, lane, 4);
            for (int i = 0; i < 4; i++)
                if ((imm >> i) & 1)
                    ins_gpr(dst, i, XZR, 4);
            vmov128(d, dst);
            ymm_zero_hi(d);
            return;
        }

        case ZYDIS_MNEMONIC_VEXTRACTPS: {
            int src = ex->op_reg[1];
            int lane = (int)((u64)in->imm & 3);
            if (ex->op_type[0] == EXT_OP_GPR) {
                umov_w(ex->op_reg[0], src, lane, 4);
            } else {
                Addr a = addr_prepare(in, 4);
                umov_w(S0, src, lane, 4);
                ldst(0, 4, S0, a);
            }
            return;
        }

        case ZYDIS_MNEMONIC_VLDMXCSR:
            ldst(1, 4, S0, addr_prepare(in, 4));
            emit_ldmxcsr_s0();
            return;
        case ZYDIS_MNEMONIC_VSTMXCSR: {
            Addr a = addr_prepare(in, 4);
            emit_stmxcsr_s0();
            ldst(0, 4, S0, a);
            return;
        }

        case ZYDIS_MNEMONIC_VPHMINPOSUW: {
            int d = ex->op_reg[0];
            int src = ext_vec_src128(in, 1, V_MEM);
            mrs_nzcv(S4);
            RR(0x6E71A800u, V_S0, src); // uminv h0, v.8h: lowest unsigned word
            umov_w(S0, V_S0, 0, 2);
            movimm(S1, 0);
            u32 *leave[8];
            for (int i = 0; i < 8; i++) {
                umov_w(S2, src, i, 2);
                add_reg(0, 1, 1, XZR, S2, S0, 0); // cmp the zero-extended words
                u32 *ne = tr_cc;
                b_cond(1, 0);
                movimm(S1, (u64)i);
                leave[i] = tr_cc;
                b_rel(0);
                *ne |= ((u32)(tr_cc - ne) & 0x7ffff) << 5;
            }
            for (int i = 0; i < 8; i++)
                *leave[i] |= (u32)(tr_cc - leave[i]) & 0x3ffffff;
            lsl_imm(0, S1, S1, 16);
            logic_reg(0, 1, 0, S0, S0, S1);
            VEORQ(d, d, d);
            ins_gpr(d, 0, S0, 4);
            ymm_zero_hi(d);
            msr_nzcv(S4);
            return;
        }

        case ZYDIS_MNEMONIC_VGATHERDPS:
        case ZYDIS_MNEMONIC_VGATHERDPD:
        case ZYDIS_MNEMONIC_VGATHERQPS:
        case ZYDIS_MNEMONIC_VGATHERQPD:
        case ZYDIS_MNEMONIC_VPGATHERDD:
        case ZYDIS_MNEMONIC_VPGATHERDQ:
        case ZYDIS_MNEMONIC_VPGATHERQD:
        case ZYDIS_MNEMONIC_VPGATHERQQ: {
            ZydisMnemonic gm = mnem;
            bool qindex = gm == ZYDIS_MNEMONIC_VGATHERQPS || gm == ZYDIS_MNEMONIC_VGATHERQPD ||
                          gm == ZYDIS_MNEMONIC_VPGATHERQD || gm == ZYDIS_MNEMONIC_VPGATHERQQ;
            bool qdata = gm == ZYDIS_MNEMONIC_VGATHERDPD || gm == ZYDIS_MNEMONIC_VGATHERQPD ||
                         gm == ZYDIS_MNEMONIC_VPGATHERDQ || gm == ZYDIS_MNEMONIC_VPGATHERQQ;
            int idx_sz = qindex ? 8 : 4;
            int data_sz = qdata ? 8 : 4;
            u8 db = ex->op_bytes[0];
            int dest = ex->op_reg[0];
            int mask = ex->op_reg[2];
            int idx = in->index;
            int nlanes = (int)db / data_sz;
            int index_bytes = nlanes * idx_sz;
            int sign = data_sz * 8 - 1;
            if (!in->vsib) {
                emit_stop(in->rip, "ext: gather operand");
                return;
            }
            if (index_bytes > 16)
                ymm_load_hi(V_S0, idx);
            if (db == 32) {
                ymm_load_hi(V_S2, dest);
                ymm_load_hi(V_S1, mask);
            }
            int idx_lo = idx;
            if (idx == dest || idx == mask) {
                vmov128(27, idx);
                idx_lo = 27;
            }
            int mask_lo = mask;
            if (mask == dest) {
                vmov128(28, mask);
                mask_lo = 28;
            }
            for (int i = 0; i < nlanes; i++) {
                int dhalf = (i * data_sz) >= 16;
                int dlane = ((i * data_sz) & 15) / data_sz;
                int ihalf = (i * idx_sz) >= 16;
                int ilane = ((i * idx_sz) & 15) / idx_sz;
                int mvec = dhalf ? V_S1 : mask_lo;
                int ivec = ihalf ? V_S0 : idx_lo;
                int dvec = dhalf ? V_S2 : dest;
                if (data_sz == 8)
                    umov_d(S0, mvec, dlane);
                else
                    umov_w(S0, mvec, dlane, 4);
                u32 *skip = tr_cc;
                tbz(0, S0, sign, 0);
                if (idx_sz == 8)
                    umov_d(S0, ivec, ilane);
                else {
                    umov_w(S0, ivec, ilane, 4);
                    sbfx(1, S0, S0, 0, 32);
                }
                gather_addr(in);
                Addr a = {.base = S1, .imm = 0};
                ldst(1, data_sz, S2, a);
                ins_gpr(dvec, dlane, S2, data_sz);
                *skip |= ((u32)(tr_cc - skip) & 0x3fff) << 5;
            }
            if (db == 32)
                ymm_store_hi(V_S2, dest);
            else
                ymm_zero_hi(dest);
            VEORQ(mask, mask, mask);
            ymm_zero_hi(mask);
            return;
        }

        case ZYDIS_MNEMONIC_VPMASKMOVD:
        case ZYDIS_MNEMONIC_VPMASKMOVQ:
        case ZYDIS_MNEMONIC_VMASKMOVPS:
        case ZYDIS_MNEMONIC_VMASKMOVPD: {
            bool qword = mnem == ZYDIS_MNEMONIC_VPMASKMOVQ || mnem == ZYDIS_MNEMONIC_VMASKMOVPD;
            int esize = qword ? 8 : 4;
            int per = 16 / esize;
            int sign = esize * 8 - 1;
            bool load = ex->op_type[0] == EXT_OP_VEC;
            u8 vbytes = load ? ex->op_bytes[0] : ex->op_bytes[2];
            int mask = ex->op_reg[1];
            int data = load ? ex->op_reg[0] : ex->op_reg[2];
            int nlanes = (int)vbytes / esize;
            int base = addr_reg(in);
            if (base != S3)
                mov_reg(1, S3, base);

            int m_hi = -1;
            if (vbytes == 32) {
                int mh = ymm_resident(mask);
                if (mh >= 0) {
                    m_hi = mh;
                } else {
                    ymm_load_hi(V_S0, mask);
                    m_hi = V_S0;
                }
            }

            u32 cmlt = qword ? 0x4EE0A800u : 0x4EA0A800u;
            RR(cmlt, 28, mask);
            if (vbytes == 32)
                RR(cmlt, 29, m_hi);

            u16 saved_zmask = ymm_zero_mask;
            int any_vec = 28;
            if (vbytes == 32) {
                VORRQ(V_MEM, 28, 29);
                any_vec = V_MEM;
            }
            umov_d(S0, any_vec, 0);
            umov_d(S1, any_vec, 1);
            logic_reg(1, 1, 0, S0, S0, S1);
            u32 *not_all_zero = tr_cc;
            cbz(1, 1, S0, 0);
            if (load) {
                VEORQ(data, data, data);
                ymm_zero_hi(data);
            }
            u32 *done_zero = tr_cc;
            b_rel(0);

            *not_all_zero |= ((u32)(tr_cc - not_all_zero) & 0x7ffffu) << 5;
            ymm_zero_mask = saved_zmask;

            int all_vec = 28;
            if (vbytes == 32) {
                VANDQ(V_MEM, 28, 29);
                all_vec = V_MEM;
            }
            umov_d(S0, all_vec, 0);
            umov_d(S1, all_vec, 1);
            logic_reg(1, 0, 0, S0, S0, S1);
            E(0xAA2003E0u | (u32)S0 << 16 | (u32)S0);
            u32 *to_partial = tr_cc;
            cbz(1, 1, S0, 0);
            if (load) {
                if (vbytes == 32) {
                    int dh = ymm_resident(data);
                    if (dh >= 0) {
                        VLDP128_I7(data, dh, S3, 0);
                    } else {
                        VLDP128_I7(data, V_S1, S3, 0);
                        ymm_store_hi(V_S1, data);
                    }
                } else {
                    ext_vldst(4, true, data, (Addr){.base = S3, .imm = 0});
                    ymm_zero_hi(data);
                }
            } else {
                if (vbytes == 32) {
                    int dh = ymm_resident(data);
                    if (dh >= 0) {
                        VSTP128_I7(data, dh, S3, 0);
                    } else {
                        ymm_load_hi(V_S1, data);
                        VSTP128_I7(data, V_S1, S3, 0);
                    }
                } else {
                    ext_vldst(4, false, data, (Addr){.base = S3, .imm = 0});
                }
            }
            u32 *done_ones = tr_cc;
            b_rel(0);

            *to_partial |= ((u32)(tr_cc - to_partial) & 0x7ffffu) << 5;
            ymm_zero_mask = saved_zmask;
            int d_hi = V_S1;
            if (!load && vbytes == 32) {
                int dh = ymm_resident(data);
                if (dh >= 0)
                    d_hi = dh;
                else
                    ymm_load_hi(V_S1, data);
            }
            int lo_res = V_S2, hi_res = 27;
            if (load) {
                VEORQ(lo_res, lo_res, lo_res);
                if (vbytes == 32)
                    VEORQ(hi_res, hi_res, hi_res);
            }
            for (int i = 0; i < nlanes; i++) {
                int half = i >= per;
                int lane = i - half * per;
                int mvec = half ? m_hi : mask;
                if (esize == 8)
                    umov_d(S0, mvec, lane);
                else
                    umov_w(S0, mvec, lane, 4);
                u32 *skip = tr_cc;
                tbz(0, S0, sign, 0);
                Addr a = {.base = S3, .imm = (i64)i * esize};
                if (load) {
                    int res = half ? hi_res : lo_res;
                    ldst(1, esize, S1, a);
                    ins_gpr(res, lane, S1, esize);
                } else {
                    int dvec = half ? d_hi : data;
                    if (esize == 8)
                        umov_d(S1, dvec, lane);
                    else
                        umov_w(S1, dvec, lane, 4);
                    ldst(0, esize, S1, a);
                }
                *skip |= ((u32)(tr_cc - skip) & 0x3fff) << 5;
            }
            if (load) {
                vmov128(data, lo_res);
                if (vbytes == 32)
                    ymm_store_hi(hi_res, data);
                else
                    ymm_zero_hi(data);
                if (vbytes == 32)
                    ymm_zero_mask = (u16)(saved_zmask & ~(1u << data));
                else if (tr_opt_ymm_cache)
                    ymm_zero_mask = (u16)(saved_zmask | (1u << data));
            }
            *done_zero |= (u32)(tr_cc - done_zero) & 0x3ffffffu;
            *done_ones |= (u32)(tr_cc - done_ones) & 0x3ffffffu;
            return;
        }

        case ZYDIS_MNEMONIC_VPCMPISTRI: {
            int d = ex->op_reg[0];
            int s = ext_vec_src128(in, 1, V_MEM);
            u8 imm = (u8)in->imm;
            emit_pcmpistri_el0(d, s, imm, live);
            return;
        }

        case ZYDIS_MNEMONIC_MPSADBW:
        case ZYDIS_MNEMONIC_VMPSADBW: {
            bool is_vex = mnem == ZYDIS_MNEMONIC_VMPSADBW;
            u8 vbytes = ex->op_bytes[0];
            int d = ex->op_reg[0];
            int n = is_vex ? ex->op_reg[1] : d;
            int s_idx = is_vex ? 2 : 1;
            u8 imm = (u8)in->imm;

            if (vbytes == 16) {
                int s = ext_vec_src128(in, s_idx, V_MEM);
                emit_mpsadbw_lane(d, n, s, imm & 7);
                if (is_vex)
                    ymm_zero_hi(d);
                return;
            }
            int s_lo = V_MEM, s_hi = V_S0;
            if (ex->op_type[s_idx] == EXT_OP_VEC) {
                int sr = ex->op_reg[s_idx];
                s_lo = sr;
                int home = ymm_resident(sr);
                if (home >= 0)
                    s_hi = home;
                else
                    ymm_load_hi(V_S0, sr);
            } else {
                ext_load256_pair(in, V_MEM, V_S0);
            }
            int n_home = ymm_resident(n);
            int n_hi = n_home >= 0 ? n_home : (ymm_load_hi(V_S1, n), V_S1);
            emit_mpsadbw_lane(d, n, s_lo, imm & 7);
            int d_home = ymm_resident(d);
            int d_hi = d_home >= 0 ? d_home : 28;
            emit_mpsadbw_lane(d_hi, n_hi, s_hi, (imm >> 3) & 7);
            if (d_home < 0)
                ymm_store_hi(d_hi, d);
            else
                ymm_zero_mask &= (u16)~(1u << d);
            return;
        }

        case ZYDIS_MNEMONIC_MASKMOVDQU:
        case ZYDIS_MNEMONIC_VMASKMOVDQU: {
            int src = ex->op_reg[0];
            int mask = ex->op_reg[1];
            int base = 7; // RDI
            if (tr_compat) {
                mov_reg(0, S2, 7);
                base = S2;
            }
            for (int i = 0; i < 16; i++) {
                u32 imm5_b = (u32)((i << 1) | 1);
                E(0x0E003C00u | (imm5_b << 16) | ((u32)mask << 5) | S0); // UMOV wS0, mask.B[i]
                tbz(0, S0, 7, 3); // if bit 7 == 0, skip next 2 insns
                E(0x0E003C00u | (imm5_b << 16) | ((u32)src << 5) | S1);  // UMOV wS1, src.B[i]
                E(0x39000000u | ((u32)i << 10) | ((u32)base << 5) | S1); // STRB wS1, [base, #i]
            }
            return;
        }

        default:
            break;
    }

    emit_stop(in->rip, "ext: unimplemented opcode");
}
