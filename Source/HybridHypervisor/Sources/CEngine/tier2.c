// Tier-2 Hot-Loop Background Asynchronous Recompiler (Round 2).
// Discovers Top-1% hot inner loops via lockless EL1 IRQ PC sampling (EngineCtx.t2_sample_ring),
// asynchronously optimizes them on a low-priority QOS_CLASS_UTILITY pthread without blocking vCPUs,
// and atomically hot-swaps the loop header's single WORD_NOP to a 32-bit B into CC_T2 (mid-loop OSR).
//
// Round 2 architecture (100% exact #PF / IRQ / SMC / trap architectural semantics):
//   1. Memop-safe full dirty YMM8..15 cross-iteration residency in {v24,v25,v29,v31} backed by
//      per-T2-block T2TrapMeta (OFF_META_T2) and zero-hot-path-cost EL1 trap sync (t2_sync_on_trap).
//   2. Guarded stack-slot register promotion ([rsp+disp]/[rbp+disp] -> {x29/S5, x24/S4}) with
//      sound zero-flag alias guard (falling back directly to T1_body on overlap) and exact exit/trap writeback.
//   3. 4x unrolling for ultra-tight loops (<=10 insns), 2x unrolling up to 64 insns, loop admission up to 96 insns,
//      expanded AVX2/scalar Preheader LICM (tr_ad_page), and exact PcMap at every x86 instruction.
//   4. Publication verifies epoch, !hb->dead, and raw guest byte snapshot under t2_pub_lock (< 5 us).
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tr_internal.h"

#define T2_MAX_LOOPS_PER_BUCKET 8192U
#define T2_BUCKETS 5U // 0..3: p0 ring slots 0..3, 4: p1 private
#define T2_MAX_BODY_INSNS 96
#define T2_MAX_UNROLL_INSNS 192
#define T2_MAX_CODE_WORDS 8192
#define T2_MAX_PCMAP 1536
#define T2_MAX_PEND 256
#define T2_MAX_FWD 256
#define T2_MAX_BAIL 16
#define T2_WINDOW_MS 400ULL
#define T2_POLL_US 15000U

enum { T2_IDLE = 0, T2_COMPILING = 1, T2_PROMOTED = 2, T2_REJECTED = 3, T2_DEAD = 4 };

typedef struct {
    u32 t1_idx;
    u32 part;
    u32 p0_slot;
    u32 entry_word;
    u32 arm_words;
    u32 patch_word;
    u32 key;
    u64 entry_rip;
    u64 head_rip;
    u64 frame0;
    u64 epoch;
    u16 min_off;
    u16 span_bytes;
    u8 cpl;
    bool compat;
    u8 state;
    u8 attempts;
    u32 hits;
    u64 win_start_ms;
} T2LoopDesc;

typedef struct {
    u32 *site;
    u64 target;
    u32 kind;
    int form;
    u8 dirty_cnt;
    u8 dirty_vreg[4];
    u8 dirty_xreg[4];
} T2Pend;

static void t2_add_pend(T2Pend *pend, int *pend_cnt, u32 *site, u64 target, int form, bool has_t2_dirty) {
    if (*pend_cnt >= T2_MAX_PEND)
        return;
    T2Pend *p = &pend[(*pend_cnt)++];
    p->site = site;
    p->target = target;
    p->form = form;
    p->dirty_cnt = (u8)ext_ymm_cache_snapshot_dirty(p->dirty_vreg, p->dirty_xreg);
    p->kind = (p->dirty_cnt || has_t2_dirty || form == 1) ? SVC_DIRECT : SVC_JCC;
}

typedef struct {
    u32 *site;
    int target_idx;
    int form;
} T2Fwd;

static pthread_mutex_t t2_meta_lock = PTHREAD_MUTEX_INITIALIZER;
static T2LoopDesc *t2_buckets[T2_BUCKETS];
static u32 t2_bucket_cnt[T2_BUCKETS];

// Hash table mapping t1_idx -> (bucket << 24) | pos + 1 for O(1) kill & eager promotion.
#define T2_T1_HASH_BITS 16
#define T2_T1_HASH_SIZE (1U << T2_T1_HASH_BITS)
typedef struct {
    u32 t1_idx;
    u32 encoded; // 0 = empty, ~0u = deleted, else ((bucket << 24) | (pos + 1))
} T2T1Map;
static T2T1Map *t2_t1_map;

static TrConfig t2_cfg;
static EngineCtx *t2_ctx;
static u32 *t2_cc_base;
static bool t2_is_enabled;
static bool t2_is_eager;
static bool t2_log_enabled;
static bool t2_no_mem_ymm;
static bool t2_no_stk_promo;
static bool t2_no_unroll4;
static u32 t2_hit_thresh = 6;
static pthread_t t2_thread;
static bool t2_thread_started;
static volatile bool t2_thread_stop;
static mach_timebase_info_data_t t2_tb;
static u32 t2_sample_tail[MAX_VCPUS];

// Diagnostics (relaxed counters).
static u64 t2_stat_registered;
static u64 t2_stat_samples;
static u64 t2_stat_compiled;
static u64 t2_stat_promoted;
static u64 t2_stat_unrolled;
static u64 t2_stat_unrolled4;
static u64 t2_stat_ro_ymm_loops;
static u64 t2_stat_res_ymm_loops;
static u64 t2_stat_stk_promo_loops;
static u64 t2_stat_licm_ad_loops;

static inline u64 t2_now_ms(void) {
    u64 t = mach_absolute_time();
    return (u64)((__uint128_t)t * t2_tb.numer / ((u64)t2_tb.denom * 1000000ULL));
}

static inline u8 *t2_ram_at(u64 gpa) {
    return t2_cfg.ram + (t2_cfg.ram_below_4g && gpa >= (1ULL << 32) ? gpa - (1ULL << 32) + t2_cfg.ram_below_4g : gpa);
}

static inline u32 t2_bucket_of(u32 part, u32 p0_slot) {
    return part ? 4U : (p0_slot & 3U);
}

static inline u32 t2_t1_hash(u32 t1_idx) {
    return (u32)((t1_idx * 2654435761U) >> (32 - T2_T1_HASH_BITS));
}

static void t2_t1_map_insert(u32 t1_idx, u32 bucket, u32 pos) {
    u32 enc = (bucket << 24) | (pos + 1U);
    for (u32 h = t2_t1_hash(t1_idx), n = 0; n < T2_T1_HASH_SIZE; n++, h = (h + 1U) & (T2_T1_HASH_SIZE - 1U)) {
        if (t2_t1_map[h].encoded == 0 || t2_t1_map[h].encoded == ~0U || t2_t1_map[h].t1_idx == t1_idx) {
            t2_t1_map[h].t1_idx = t1_idx;
            t2_t1_map[h].encoded = enc;
            return;
        }
    }
}

static T2LoopDesc *t2_t1_map_find(u32 t1_idx) {
    if (!t2_t1_map)
        return NULL;
    for (u32 h = t2_t1_hash(t1_idx), n = 0; n < T2_T1_HASH_SIZE; n++, h = (h + 1U) & (T2_T1_HASH_SIZE - 1U)) {
        u32 enc = t2_t1_map[h].encoded;
        if (enc == 0)
            return NULL;
        if (enc != ~0U && t2_t1_map[h].t1_idx == t1_idx) {
            u32 b = enc >> 24;
            u32 pos = (enc & 0xFFFFFFU) - 1U;
            if (b < T2_BUCKETS && pos < t2_bucket_cnt[b])
                return &t2_buckets[b][pos];
            return NULL;
        }
    }
    return NULL;
}

static void t2_t1_map_remove(u32 t1_idx) {
    if (!t2_t1_map)
        return;
    for (u32 h = t2_t1_hash(t1_idx), n = 0; n < T2_T1_HASH_SIZE; n++, h = (h + 1U) & (T2_T1_HASH_SIZE - 1U)) {
        u32 enc = t2_t1_map[h].encoded;
        if (enc == 0)
            return;
        if (enc != ~0U && t2_t1_map[h].t1_idx == t1_idx) {
            t2_t1_map[h].encoded = ~0U;
            return;
        }
    }
}

static void t2_clear_bucket(u32 b) {
    if (b >= T2_BUCKETS || !t2_buckets[b])
        return;
    for (u32 i = 0; i < t2_bucket_cnt[b]; i++)
        t2_t1_map_remove(t2_buckets[b][i].t1_idx);
    t2_bucket_cnt[b] = 0;
}

bool t2_enabled(void) { return t2_is_enabled; }
bool t2_eager_mode(void) { return t2_is_enabled && t2_is_eager; }
u32 t2_ablation_bits(void) {
    u32 b = 0;
    if (t2_no_mem_ymm) b |= AOT_MODE_T2_NO_MEM_YMM;
    if (t2_no_stk_promo) b |= AOT_MODE_T2_NO_STK_PROMO;
    if (t2_no_unroll4) b |= AOT_MODE_T2_NO_UNROLL4;
    return b;
}

// ---- Tier-2 instruction classification & safety checks ----
static bool t2_op_forbidden(const Insn *in) {
    if (in->lock || tr_is_monitor_or_stop_op(in->op))
        return true;
    switch (in->op) {
        case OP_BAD: case OP_UD:
        case OP_POPF: case OP_PUSHF: case OP_LAHF: case OP_SAHF:
        case OP_X87: case OP_STRING:
        case OP_CALL: case OP_CALLI: case OP_JMPI: case OP_RET:
        case OP_ENTER: case OP_LEAVE: case OP_FAR:
        case OP_PUSHA: case OP_XLAT:
        case OP_CMPXCHG: case OP_XADD:
            return true;
        case OP_XCHG:
            return in->memop; // xchg [mem], reg has an implicit LOCK prefix on x86
        default:
            return false;
    }
}

static bool t2_insn_preserves_ad(const Insn *in) {
    if (!tr_ad_page_safe_op(in->op))
        return false;
    if (in->op == OP_ALU && (in->sub == ALU_ADC || in->sub == ALU_SBB) && in->sz < 4)
        return false; // narrow ADC/SBB alu_flags clobbers AD
    if (in->op == OP_LEA && !in->riprel)
        return false;
    if (in->op == OP_LEA && in->riprel) {
        if (in->seg || in->addr32 || in->addr16 || tr_compat)
            return false;
        return true;
    }
    if (!in->memop)
        return true;
    if (in->seg || in->addr32 || in->addr16 || in->vsib)
        return false;
    bool simple_ldst = (in->op == OP_MOV) || (in->op == OP_MOVZX) || (in->op == OP_MOVSX) ||
                       (in->op == OP_ALU && in->to_reg) || (in->op == OP_SSE) || (in->op == OP_EXT);
    if (!simple_ldst)
        return false;
    if (in->op == OP_EXT) {
        for (u8 i = 0; i < in->ext.op_cnt; i++) {
            if (in->ext.op_type[i] == EXT_OP_MEM && in->ext.op_bytes[i] == 32) {
                if (in->riprel || in->no_base || !in->no_index || tr_compat)
                    return false;
                // Matches ext_load256_pair / ext_store256_pair fast paths that never call addr_reg().
                if (in->disp >= -1024 && in->disp <= 1008 && (in->disp & 15) == 0)
                    return true;
                if (in->disp >= 0 && !(in->disp & 15) && ((in->disp + 16) >> 4) < 4096)
                    return true;
                return in->disp >= -256 && in->disp + 16 < 256;
            }
        }
    }
    if (in->riprel) {
        if (tr_compat)
            return false;
        u64 target = in->next + (u64)in->disp;
        u32 off = (u32)(target & 0xfffULL);
        int access = in->op == OP_SSE ? 16 : (in->sz ? in->sz : 8);
        if (in->op == OP_EXT) {
            access = 16;
            for (u8 i = 0; i < in->ext.op_cnt; i++) {
                if (in->ext.op_type[i] == EXT_OP_MEM && in->ext.op_bytes[i]) {
                    access = in->ext.op_bytes[i];
                    break;
                }
            }
            if (access == 32)
                return false; // 32B RIP-rel load uses ext_load256_pair -> addr_reg() -> clobbers AD
        }
        return (off & (u32)(access - 1)) == 0 || off < 256;
    }
    // addr_prepare() never clobbers AD for [base + disp] (no index, no seg, !addr32, !riprel).
    return !in->no_base && in->no_index && in->disp >= -256 && in->disp <= 4095;
}

// ---- Round 2 Guarded Stack-Slot Register Promotion helpers ----
static inline int t2_gpr_phys(int r, bool high) { return high ? r - 4 : r; }

static u16 t2_insn_written_gprs(const Insn *in) {
    switch (in->op) {
        case OP_NOP: case OP_FENCE: case OP_PREFETCH: case OP_CLFLUSH:
        case OP_JCC: case OP_JMP: case OP_CLC: case OP_STC: case OP_CMC:
        case OP_CLD: case OP_STD: case OP_LDMXCSR: case OP_STMXCSR:
            return 0;
        case OP_MOV:
            if (in->to_reg)
                return (u16)(1u << t2_gpr_phys(in->reg, in->high_reg));
            if (in->rm_reg)
                return (u16)(1u << t2_gpr_phys(in->rm, in->high_rm));
            return 0;
        case OP_ALU:
            if (in->sub == ALU_CMP || in->sub == ALU_TEST)
                return 0;
            if (in->to_reg)
                return (u16)(1u << t2_gpr_phys(in->reg, in->high_reg));
            if (in->rm_reg)
                return (u16)(1u << t2_gpr_phys(in->rm, in->high_rm));
            return 0;
        case OP_INCDEC: case OP_NEG: case OP_NOT: case OP_SHIFT: case OP_SHD:
        case OP_BSWAP: case OP_SETCC:
            return in->rm_reg ? (u16)(1u << t2_gpr_phys(in->rm, in->high_rm)) : 0;
        case OP_BT:
            return (in->sub != 0 && in->rm_reg) ? (u16)(1u << in->rm) : 0;
        case OP_LEA: case OP_MOVZX: case OP_MOVSX: case OP_CMOV: case OP_IMUL:
        case OP_BSF: case OP_BSR: case OP_TZCNT: case OP_LZCNT: case OP_POPCNT: case OP_CRC32:
            return (u16)(1u << in->reg);
        case OP_MOVBE:
            return in->to_reg ? (u16)(1u << in->reg) : 0;
        case OP_XCHG: {
            u16 m = (u16)(1u << t2_gpr_phys(in->reg, in->high_reg));
            if (in->rm_reg) m |= (u16)(1u << t2_gpr_phys(in->rm, in->high_rm));
            return m;
        }
        case OP_MUL: case OP_DIV:
            return (u16)((1u << 0) | (1u << 2)); // RAX, RDX
        case OP_CBW:
            return (u16)(1u << 0);
        case OP_CWD:
            return (u16)(1u << 2);
        case OP_JRCXZ:
            return in->sub >= 2 ? (u16)(1u << 1) : 0;
        case OP_EXT:
            return ext_insn_written_gprs(in);
        case OP_SSE: {
            u16 m = (u16)(1u << in->reg);
            if (in->rm_reg) m |= (u16)(1u << in->rm);
            m |= (u16)(1u << 1); // conservative for PCMPISTRI (RCX)
            return m;
        }
        default:
            return 0xFFFFu;
    }
}

static bool t2_insn_stores_mem(const Insn *in) {
    if (!in->memop)
        return false;
    switch (in->op) {
        case OP_PREFETCH: case OP_CLFLUSH: case OP_LEA:
            return false;
        case OP_MOV: case OP_MOVBE:
            return !in->to_reg;
        case OP_MOVZX: case OP_MOVSX: case OP_CMOV: case OP_IMUL: case OP_MUL: case OP_DIV:
        case OP_BSF: case OP_BSR: case OP_TZCNT: case OP_LZCNT: case OP_POPCNT: case OP_CRC32:
        case OP_LDMXCSR:
            return false;
        case OP_ALU:
            return !in->to_reg && in->sub != ALU_CMP && in->sub != ALU_TEST;
        case OP_INCDEC: case OP_NEG: case OP_NOT: case OP_SHIFT: case OP_SHD:
        case OP_SETCC: case OP_STMXCSR:
            return !in->rm_reg;
        case OP_BT:
            return in->sub != 0 && !in->rm_reg;
        case OP_EXT:
            return in->ext.op_cnt >= 1 && in->ext.op_type[0] == EXT_OP_MEM;
        case OP_SSE:
            return !in->to_reg;
        default:
            return true;
    }
}

static bool t2_insn_uses_s5(const Insn *in, int live) {
    if (in->op == OP_SHIFT || in->op == OP_SHD || in->op == OP_CMPXCHG || in->op == OP_CMPXCHG8B)
        return true;
    if ((in->op == OP_JCC || in->op == OP_CMOV || in->op == OP_SETCC) && (in->cc == 10 || in->cc == 11))
        return true; // emit_parity() clobbers S5
    if ((in->op == OP_ALU || in->op == OP_INCDEC || in->op == OP_NEG) && live != 0 && in->sz < 4)
        return true; // narrow alu_flags() clobbers S5
    if (in->op == OP_SSE)
        return true; // conservative (covers PCMPISTRI)
    if (in->op == OP_EXT && !ext_insn_s5_free(in))
        return true;
    return false;
}

static bool t2_insn_uses_s4(const Insn *in, int live, bool is_promoted) {
    if (in->op == OP_SHIFT || in->op == OP_SHD || in->op == OP_MUL || in->op == OP_IMUL ||
        in->op == OP_DIV || in->op == OP_BT || in->op == OP_STRING || in->op == OP_CMPXCHG ||
        in->op == OP_CMPXCHG8B || in->op == OP_SSE || in->op == OP_X87)
        return true;
    if (in->op == OP_ALU && (in->sub == ALU_ADC || in->sub == ALU_SBB))
        return true;
    if (in->op == OP_INCDEC && (live & F_C) && (!in->rm_reg || in->sz < 4) && !is_promoted)
        return true;
    if (in->op == OP_LEA && (in->disp < -4095 || in->disp > 4095))
        return true;
    if ((in->op == OP_JCC || in->op == OP_CMOV || in->op == OP_SETCC) && (in->cc == 10 || in->cc == 11))
        return true;
    if (in->op == OP_EXT && !ext_insn_s4_free(in))
        return true;
    if (in->memop && !is_promoted && in->op != OP_EXT) {
        int sz = in->sz ? in->sz : 8;
        bool imm_ok = !in->riprel && !in->no_base && in->no_index && !in->seg && !in->addr32 && !in->addr16 &&
                      ((in->disp >= 0 && !(in->disp & (sz - 1)) && (in->disp / sz) < 4096) ||
                       (in->disp >= -256 && in->disp < 256));
        if (!imm_ok)
            return true; // addr_reg() or ldst() may spill large disp via S4
    }
    return false;
}

static bool t2_stk_op_supported(const Insn *in, bool *out_reads, bool *out_writes) {
    *out_reads = false;
    *out_writes = false;
    if (!in->memop || in->rm_reg || in->riprel || in->no_base || !in->no_index ||
        in->seg || in->addr32 || in->addr16 || in->vsib || in->lock ||
        in->high_reg || in->high_rm)
        return false;
    if (in->disp < -1024 || in->disp > 1024)
        return false;
    switch (in->op) {
        case OP_MOV:
            if (in->sz != 4 && in->sz != 8)
                return false;
            *out_reads = in->to_reg;
            *out_writes = !in->to_reg;
            return true;
        case OP_ALU:
            if ((in->sz != 4 && in->sz != 8) || in->sub == ALU_ADC || in->sub == ALU_SBB)
                return false;
            *out_reads = true;
            *out_writes = !in->to_reg && in->sub != ALU_CMP && in->sub != ALU_TEST;
            return true;
        case OP_INCDEC:
        case OP_NEG:
        case OP_NOT:
            if (in->sz != 4 && in->sz != 8)
                return false;
            *out_reads = true;
            *out_writes = true;
            return true;
        case OP_CMOV:
            if (in->sz != 4 && in->sz != 8)
                return false;
            *out_reads = true;
            return true;
        case OP_IMUL:
            if (in->sz != 4 && in->sz != 8)
                return false;
            *out_reads = true;
            return true;
        case OP_MOVSX:
            if (in->srcsz != 4 || in->sz != 8)
                return false;
            *out_reads = true;
            return true;
        default:
            return false;
    }
}

static int t2_mem_access_bytes(const Insn *in) {
    if (in->op == OP_EXT) {
        for (u8 i = 0; i < in->ext.op_cnt; i++)
            if (in->ext.op_type[i] == EXT_OP_MEM && in->ext.op_bytes[i])
                return in->ext.op_bytes[i];
        return 32;
    }
    if (in->op == OP_SSE)
        return 16;
    if (in->op == OP_MOVZX || in->op == OP_MOVSX)
        return in->srcsz ? in->srcsz : 8;
    return in->sz ? in->sz : 8;
}

static bool t2_gpr_bounded_stride(const Insn *body, int bcnt, u8 r, i64 *out_max_abs) {
    i64 max_abs = 0;
    for (int i = 0; i < bcnt; i++) {
        const Insn *in = &body[i];
        if (!(t2_insn_written_gprs(in) & (1u << r)))
            continue;
        if (in->op == OP_ALU && in->rm_reg && in->rm == r && !in->to_reg && in->has_imm &&
            in->sz == 8 && (in->sub == ALU_ADD || in->sub == ALU_SUB)) {
            i64 d = in->imm;
            if (d < -2048 || d > 2048)
                return false;
            max_abs += d < 0 ? -d : d;
        } else if (in->op == OP_INCDEC && in->rm_reg && in->rm == r && in->sz == 8) {
            max_abs += 1;
        } else if (in->op == OP_LEA && in->reg == r && in->sz == 8 && !in->riprel &&
                   !in->no_base && in->no_index && in->base == r && !in->seg && !in->addr32 && !in->addr16) {
            i64 d = in->disp;
            if (d < -2048 || d > 2048)
                return false;
            max_abs += d < 0 ? -d : d;
        } else if (in->op == OP_JRCXZ && in->sub >= 2 && r == 1 && !in->addr32 && !in->addr16) {
            max_abs += 1;
        } else {
            return false;
        }
        if (max_abs > 2048)
            return false;
    }
    *out_max_abs = max_abs;
    return true;
}

// Emit 32/64-bit load/store for promoted stack slot using only S0 (never clobbers S4, S5, AD, or NZCV).
static void t2_emit_stk_ldst(bool load, int sz, int rt, int base, i64 disp) {
    u32 s = sz == 8 ? 3u : 2u;
    u32 opc = load ? 1u : 0u;
    if (disp >= 0 && !(disp & (sz - 1)) && (disp >> s) < 4096) {
        E(s << 30 | 0x39000000u | opc << 22 | (u32)(disp >> s) << 10 | (u32)base << 5 | (u32)rt);
    } else if (disp >= -256 && disp < 256) {
        E(s << 30 | 0x38000000u | opc << 22 | ((u32)disp & 0x1ffu) << 12 | (u32)base << 5 | (u32)rt);
    } else {
        bool neg = disp < 0;
        u32 mag = (u32)(neg ? -disp : disp);
        add_imm(1, neg ? 1 : 0, 0, S0, base, mag);
        E(s << 30 | 0x39000000u | opc << 22 | (u32)S0 << 5 | (u32)rt);
    }
}

typedef struct {
    i64 disp;
    u8 sz;
    u8 phys;
    bool writes;
    int accesses;
} T2StkSlot;

typedef struct {
    u8 base;
    u8 idx;   // 0xFF = no index
    u8 scale; // 0..3
} T2GuardExpr;

// Zero-flag ±8 KiB alias guard: branches via CBZ S0 to bail stub if |(ea0) - stk_base| < 8192.
// Covers disp_store in [-1024,1024] + disp_slot in [-1024,1024] + stride*copies <= 2048 + 32B width = 4128 < 8192.
static u32 *t2_emit_alias_guard(T2GuardExpr g, u8 stk_base) {
    if (g.idx == 0xFF) {
        add_reg(1, 1, 0, S0, g.base, stk_base, 0);        // sub x16, rb, stk_base
    } else {
        add_reg(1, 0, 0, S0, g.base, g.idx, g.scale);     // add x16, rb, idx, lsl #scale
        add_reg(1, 1, 0, S0, S0, stk_base, 0);            // sub x16, x16, stk_base
    }
    add_imm_lsl12(1, 0, 0, S0, S0, 2);                    // add x16, x16, #0x2000 (+8192)
    lsr_imm(1, S0, S0, 14);                               // lsr x16, x16, #14 (0 iff diff in [-8192, +8192))
    u32 *site = tr_cc;
    cbz(0, 1, S0, 0);                                     // cbz x16, bail_stub
    return site;
}

static int t2_match_stk_slot(const Insn *in, u8 stk_base, const T2StkSlot *slots, int cnt) {
    if (cnt <= 0 || !in->memop || in->rm_reg || in->riprel || in->no_base || !in->no_index ||
        in->seg || in->addr32 || in->addr16 || in->base != stk_base)
        return -1;
    bool rd = false, wr = false;
    if (!t2_stk_op_supported(in, &rd, &wr))
        return -1;
    int msz = (in->op == OP_MOVSX) ? in->srcsz : in->sz;
    for (int k = 0; k < cnt; k++) {
        if (slots[k].disp == in->disp && slots[k].sz == (u8)msz)
            return k;
    }
    return -1;
}

static int t2_find_rip(const Insn *list, int count, u64 rip) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (list[mid].rip < rip)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < count && list[lo].rip == rip) ? lo : -1;
}

static inline void t2_stub_lit(u64 v) {
    tr_aot_note_gva64_lit(tr_cc);
    E((u32)v);
    E((u32)(v >> 32));
}

static inline void t2_fix(u32 *site, int form, u32 *to) {
    i64 d = to - site;
    if (form == 0)
        *site = (*site & ~(0x7ffffu << 5)) | (((u32)d & 0x7ffffu) << 5);
    else if (form == 2)
        *site = (*site & ~(0x3fffu << 5)) | (((u32)d & 0x3fffu) << 5);
    else
        *site = (*site & 0xFC000000u) | ((u32)d & 0x3ffffffu);
}

static inline bool t2_is_uncond_term(u8 op) {
    return op == OP_JMP || op == OP_JMPI || op == OP_RET || op == OP_BAD || op == OP_UD || op == OP_IRET;
}

static inline bool t2_overlaps(const Insn *list, int cnt, u64 s, u64 e) {
    for (int i = 0; i < cnt; i++)
        if (!(e <= list[i].rip || s >= list[i].next))
            return true;
    return false;
}

// Decode a clean single-entry loop body starting at head_rip inside the immutable byte snapshot.
static int t2_decode_loop_body(const T2LoopDesc *d, Insn *body) {
    u64 page = d->head_rip & ~0xfffULL;
    u64 win_lo = page + d->min_off;
    u64 win_hi = win_lo + d->span_bytes;
    int count = 0;

    u64 q[16];
    int qh = 0, qt = 0;
    q[qt++] = d->head_rip;

    while (qh < qt && count < T2_MAX_BODY_INSNS) {
        u64 spc = q[qh++];
        if (spc < d->head_rip || spc < win_lo || spc >= win_hi)
            continue;
        bool seen = false;
        for (int i = 0; i < count; i++)
            if (body[i].rip == spc) { seen = true; break; }
        if (seen)
            continue;

        Insn tmp[T2_MAX_BODY_INSNS];
        int tc = 0;
        bool ok = true;
        while (count + tc < T2_MAX_BODY_INSNS) {
            bool hit = false;
            for (int i = 0; i < count; i++)
                if (body[i].rip == spc) { hit = true; break; }
            if (hit)
                break;
            if (spc < d->head_rip || spc < win_lo || spc >= win_hi || (spc & ~0xfffULL) != page) {
                ok = false;
                break;
            }
            memset(&tmp[tc], 0, sizeof(Insn));
            x86_decode(spc, &tmp[tc]);
            Insn *in = &tmp[tc];
            if (in->next <= in->rip || in->next > win_hi || t2_op_forbidden(in)) {
                ok = false;
                break;
            }
            if (t2_overlaps(body, count, in->rip, in->next)) {
                ok = false;
                break;
            }
            spc = in->next;
            tc++;
            if (insn_is_terminator(in) ||
                ((in->op == OP_JCC || in->op == OP_JRCXZ) && in->target == d->head_rip))
                break;
        }
        if (!ok || tc == 0) {
            if (count == 0)
                return 0;
            continue;
        }
        memcpy(&body[count], tmp, (size_t)tc * sizeof(Insn));
        int base = count;
        count += tc;
        for (int i = base; i < count; i++) {
            u8 op = body[i].op;
            if (op != OP_JCC && op != OP_JRCXZ && op != OP_JMP)
                continue;
            u64 t = body[i].target;
            if (t <= body[i].rip || t < d->head_rip || t >= win_hi)
                continue;
            bool dup = false;
            for (int k = 0; k < qt; k++)
                if (q[k] == t) { dup = true; break; }
            if (!dup && qt < 16)
                q[qt++] = t;
        }
    }
    if (count < 2)
        return 0;

    // Sort by rip ascending.
    for (int i = 1; i < count; i++) {
        Insn cur = body[i];
        int j = i - 1;
        while (j >= 0 && body[j].rip > cur.rip) {
            body[j + 1] = body[j];
            j--;
        }
        body[j + 1] = cur;
    }
    if (body[0].rip != d->head_rip)
        return 0;

    // Trim any instructions strictly after the last backedge to head_rip (post-loop tail stays in Tier-1).
    int last_back = -1;
    for (int i = 0; i < count; i++) {
        u8 op = body[i].op;
        if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) && body[i].target == d->head_rip)
            last_back = i;
    }
    if (last_back < 1)
        return 0;
    count = last_back + 1;

    // Reject if any internal jump lands inside (0..count-1) via a backedge other than to 0
    // (keep Round 1 MVP focused on clean single-header natural loops).
    for (int i = 0; i < count; i++) {
        u8 op = body[i].op;
        if (op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) {
            int ti = t2_find_rip(body, count, body[i].target);
            if (ti > 0 && ti <= i)
                return 0; // nested inner backedge
        }
    }
    return count;
}

// Same-page exit flag-liveness peek inside the SMC-verified snapshot window:
// eliminates dead CF/AF/PF/OF preservation (e.g. `dec rcx` 5 insns -> 1 `subs`!) whenever post-loop code overwrites flags.
static u8 t2_peek_exit_flags(u64 rip, u64 win_lo, u64 win_hi) {
    u8 need = 0, written = 0;
    for (int i = 0; i < 16; i++) {
        if (rip < win_lo || rip + 15 > win_hi)
            return (u8)(need | (F_ARITH & ~written));
        Insn in;
        memset(&in, 0, sizeof in);
        x86_decode(rip, &in);
        if (in.op == OP_BAD || in.op == OP_UD || in.next <= rip || in.next > win_hi)
            return (u8)(need | (F_ARITH & ~written));
        need |= (u8)(tr_insn_reads(&in) & ~written);
        written |= in.writes;
        if ((written & F_ARITH) == F_ARITH)
            return need;
        if (in.op == OP_JMP) {
            rip = in.target;
            continue;
        }
        if (in.op == OP_JCC || in.op == OP_JRCXZ || in.op == OP_CALL || in.op == OP_CALLI ||
            in.op == OP_RET || in.op == OP_JMPI || tr_is_monitor_or_stop_op(in.op))
            return (u8)(need | (F_ARITH & ~written));
        rip = in.next;
    }
    return (u8)(need | (F_ARITH & ~written));
}

static bool t2_compile_and_publish(const T2LoopDesc *d) {
    if (d->span_bytes == 0 || d->span_bytes > 4096 || (u32)d->min_off + d->span_bytes > 4096)
        return false;
    if (d->frame0 >= t2_cfg.ram_size)
        return false;

    u8 snap[4096];
    memcpy(snap, t2_ram_at(d->frame0 + d->min_off), d->span_bytes);

    u64 page = d->head_rip & ~0xfffULL;
    u64 win_lo = page + d->min_off;
    u64 win_hi = win_lo + d->span_bytes;
    tr_t2_mode = true;
    tr_cpl = (int)d->cpl;
    tr_compat = d->compat;
    cur_part = (int)d->part;
    tr_t2_set_snapshot(win_lo, snap, d->span_bytes);

    Insn body[T2_MAX_BODY_INSNS];
    int bcnt = t2_decode_loop_body(d, body);
    if (bcnt < 2) {
        tr_t2_clear_snapshot();
        tr_t2_mode = false;
        return false;
    }

    // ---- Pass 1: Loop Analysis (memop, memop-safe YMM8..15 residency, LICM tr_ad_page, 4x/2x unroll) ----
    bool has_memop = false;
    bool all_ymm_safe = true;
    bool has_4vec_blend = false;
    bool all_preserve_ad = true;
    u64 licm_ad_page = ~0ULL;
    int licm_ad_pages = 0;
    u16 ymm_rd256 = 0, ymm_wr_any = 0;
    u16 insn_hi_any[T2_MAX_BODY_INSNS];
    u16 ymm_freq[16];
    memset(ymm_freq, 0, sizeof ymm_freq);
    memset(insn_hi_any, 0, sizeof insn_hi_any);
    int backedge_cnt = 0;
    int est_words = 0;

    for (int i = 0; i < bcnt; i++) {
        const Insn *in = &body[i];
        if (in->memop || in->op == OP_PUSH || in->op == OP_POP)
            has_memop = true;
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR) {
            all_ymm_safe = false;
        } else if (in->op == OP_EXT) {
            if (!ext_insn_keeps_ymm_cache(in))
                all_ymm_safe = false;
            u16 r256 = 0, wany = 0;
            ext_insn_ymm_rw(in, &r256, &wany);
            ymm_rd256 |= r256;
            ymm_wr_any |= wany;
            u16 hi_any = (u16)((r256 | wany) & 0xFF00u);
            insn_hi_any[i] = hi_any;
            for (int r = 8; r < 16; r++) {
                if (r256 & (1u << r))
                    ymm_freq[r] += 2;
                if (wany & (1u << r))
                    ymm_freq[r] += 2;
            }
            if (in->ext.op_cnt >= 4 && in->ext.op_type[3] == EXT_OP_VEC)
                has_4vec_blend = true;
        }
        if (all_preserve_ad) {
            if (!t2_insn_preserves_ad(in)) {
                all_preserve_ad = false;
            } else if (in->riprel && (in->memop || in->op == OP_LEA)) {
                u64 pg = (in->next + (u64)in->disp) & ~0xfffULL;
                if (licm_ad_pages == 0) {
                    licm_ad_page = pg;
                    licm_ad_pages = 1;
                } else if (pg != licm_ad_page) {
                    licm_ad_pages = 2;
                }
            }
        }
        u8 op = in->op;
        if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) && in->target == d->head_rip)
            backedge_cnt++;
        est_words += in->op == OP_EXT ? 16 : in->op == OP_SSE ? 12 : 4;
    }
    if (!all_preserve_ad || licm_ad_pages != 1)
        licm_ad_page = ~0ULL;

    // Select memop-safe full RW resident YMM8..15 slots ({v24,v25,v29,v31}) backed by T2TrapMeta trap sync.
    // Invariant: res_ymm_cnt + max_unres_per_insn <= 4 so ymm_cache_get() can NEVER overflow.
    u8 res_ymm_regs[4], ro_pin_regs[4], rw_preload_regs[4];
    int res_ymm_cnt = 0, ro_pin_cnt = 0, rw_preload_cnt = 0;
    u8 res_ymm_wr_mask = 0;
    bool all_hi_fit_pool = false;
    if (tr_opt_ymm_cache && all_ymm_safe && !tr_precise_flags) {
        u16 all_hi = (u16)((ymm_rd256 | ymm_wr_any) & 0xFF00u);
        int total_hi = __builtin_popcount((unsigned)all_hi);
        all_hi_fit_pool = (total_hi > 0 && total_hi <= 4);

        if (!t2_no_mem_ymm && total_hi > 0) {
            // Rank candidate ymm8..15 descending by access weight (tie-break: written regs first).
            u8 cands[8];
            int ncands = 0;
            u16 rem = all_hi;
            while (rem) {
                int best = -1;
                for (int r = 8; r < 16; r++) {
                    if (!(rem & (1u << r)))
                        continue;
                    bool b_wr = (ymm_wr_any & (1u << r)) != 0;
                    bool cur_wr = best >= 0 && ((ymm_wr_any & (1u << best)) != 0);
                    if (best < 0 || ymm_freq[r] > ymm_freq[best] ||
                        (ymm_freq[r] == ymm_freq[best] && b_wr && !cur_wr))
                        best = r;
                }
                if (best < 0)
                    break;
                cands[ncands++] = (u8)best;
                rem &= (u16)~(1u << best);
            }
            int try_max = ncands < 4 ? ncands : 4;
            for (int k = try_max; k >= 1; k--) {
                u16 rmask = 0;
                for (int j = 0; j < k; j++)
                    rmask |= (u16)(1u << cands[j]);
                int max_unres = 0;
                for (int i = 0; i < bcnt; i++) {
                    int un = __builtin_popcount((unsigned)(insn_hi_any[i] & ~rmask));
                    if (un > max_unres)
                        max_unres = un;
                }
                if (k + max_unres <= 4) {
                    res_ymm_cnt = k;
                    for (int j = 0; j < k; j++) {
                        res_ymm_regs[j] = cands[j];
                        if (ymm_wr_any & (1u << cands[j]))
                            res_ymm_wr_mask |= (u8)(1u << j);
                    }
                    break;
                }
            }
        } else {
            // Legacy Round-1 fallback (when LC_HVDBT_T2_NO_MEM_YMM=1): RO pins + mem-free RW preload.
            u16 ro_mask = (u16)(ymm_rd256 & ~ymm_wr_any & 0xFF00u);
            u16 rw_mask = (u16)(ymm_rd256 & ymm_wr_any & 0xFF00u);
            int max_pins = 0;
            if (ro_mask) {
                if (total_hi <= 4) max_pins = 4;
                else if (!has_4vec_blend) max_pins = 1;
            }
            while (ro_pin_cnt < max_pins && ro_mask) {
                int best = -1;
                for (int r = 8; r < 16; r++) {
                    if (!(ro_mask & (1u << r)))
                        continue;
                    if (best < 0 || ymm_freq[r] > ymm_freq[best])
                        best = r;
                }
                if (best < 0)
                    break;
                ro_pin_regs[ro_pin_cnt++] = (u8)best;
                ro_mask &= (u16)~(1u << best);
            }
            if (!has_memop && all_hi_fit_pool && backedge_cnt == 1) {
                for (int r = 8; r < 16 && ro_pin_cnt + rw_preload_cnt < 4; r++) {
                    if (rw_mask & (1u << r))
                        rw_preload_regs[rw_preload_cnt++] = (u8)r;
                }
            }
        }
    }

    // Decide unroll factor: 4x for ultra-tight loops (<=10 insns), 2x up to 64 insns, 1x up to 96 insns.
    const Insn *latch = &body[bcnt - 1];
    bool clean_latch = (backedge_cnt == 1) && (latch->target == d->head_rip) &&
                       (latch->op == OP_JMP || latch->op == OP_JCC ||
                        (latch->op == OP_JRCXZ && latch->sub == 2 && !latch->addr16));
    int copies = 1;
    if (clean_latch) {
        if (!t2_no_unroll4 && bcnt <= 10 && est_words * 4 <= 2048 && bcnt * 4 <= T2_MAX_UNROLL_INSNS)
            copies = 4;
        else if (bcnt <= 64 && est_words * 2 <= 2400 && bcnt * 2 <= T2_MAX_UNROLL_INSNS)
            copies = 2;
    }
    bool dirty_ymm_boost = !has_memop && all_ymm_safe && ((ymm_wr_any & 0xFF00u) != 0);

    // ---- Pass 2: Build Unrolled IR & Explicit CFG ----
    Insn ulist[T2_MAX_UNROLL_INSNS];
    int u_tgt[T2_MAX_UNROLL_INSNS];      // >=0 internal idx, -2 backedge to loop_hdr (idx 0), -1 external
    int u_fall[T2_MAX_UNROLL_INSNS];     // >=0 internal idx, -2 backedge to 0, -1 external, -3 none
    u64 u_ext_tgt[T2_MAX_UNROLL_INSNS];
    u64 u_ext_fall[T2_MAX_UNROLL_INSNS];
    bool u_is_target[T2_MAX_UNROLL_INSNS];
    bool u_iter0_jrcxz_inv[T2_MAX_UNROLL_INSNS];
    bool u_skip_nop[T2_MAX_UNROLL_INSNS];

    int ucnt = bcnt * copies;
    memset(u_is_target, 0, (size_t)ucnt * sizeof(bool));
    memset(u_iter0_jrcxz_inv, 0, (size_t)ucnt * sizeof(bool));
    memset(u_skip_nop, 0, (size_t)ucnt * sizeof(bool));

    for (int c = 0; c < copies; c++) {
        int base = c * bcnt;
        for (int j = 0; j < bcnt; j++) {
            int idx = base + j;
            ulist[idx] = body[j];
            Insn *in = &ulist[idx];
            u_tgt[idx] = -3;
            u_fall[idx] = -3;
            u_ext_tgt[idx] = 0;
            u_ext_fall[idx] = 0;

            bool is_mid_latch = (copies > 1) && (c < copies - 1) && (j == bcnt - 1);
            if (is_mid_latch) {
                int next_iter = (c + 1) * bcnt;
                if (in->op == OP_JMP) {
                    // Eliminate intermediate backedge JMP completely: fall straight into next copy!
                    u_skip_nop[idx] = true;
                    in->op = OP_NOP;
                    in->reads = 0;
                    in->writes = 0;
                    u_fall[idx] = next_iter;
                } else if (in->op == OP_JCC) {
                    // Invert condition to exit on !cc, falling straight through into next copy!
                    in->cc ^= 1;
                    in->reads = cc_reads[in->cc];
                    u_tgt[idx] = -1;
                    u_ext_tgt[idx] = body[j].next; // loop exit RIP
                    u_fall[idx] = next_iter;
                } else {
                    // OP_JRCXZ sub==2 (loop): decrement rcx/ecx and cbz -> exit, falling into next copy!
                    u_iter0_jrcxz_inv[idx] = true;
                    u_tgt[idx] = -1;
                    u_ext_tgt[idx] = body[j].next;
                    u_fall[idx] = next_iter;
                }
                continue;
            }

            if (in->op == OP_JCC || in->op == OP_JRCXZ || in->op == OP_JMP) {
                if (in->target == d->head_rip) {
                    u_tgt[idx] = -2; // backedge to loop_hdr_arm (idx 0)
                } else {
                    int ti = t2_find_rip(body, bcnt, in->target);
                    if (ti >= 0) {
                        u_tgt[idx] = base + ti;
                        u_is_target[base + ti] = true;
                    } else {
                        u_tgt[idx] = -1;
                        u_ext_tgt[idx] = in->target;
                    }
                }
            }
            if (!t2_is_uncond_term(in->op)) {
                if (j + 1 < bcnt && body[j + 1].rip == in->next) {
                    u_fall[idx] = idx + 1;
                } else {
                    int fi = t2_find_rip(body, bcnt, in->next);
                    if (fi == 0) {
                        u_fall[idx] = -2;
                    } else if (fi > 0) {
                        u_fall[idx] = base + fi;
                        u_is_target[base + fi] = true;
                    } else {
                        u_fall[idx] = -1;
                        u_ext_fall[idx] = in->next;
                    }
                }
            }
            if (j > 0 && (t2_is_uncond_term(body[j - 1].op) || body[j - 1].next != in->rip))
                u_is_target[idx] = true;
        }
    }

    // ---- Pass 3: Cross-Backedge Fixed-Point Flag Liveness (with same-page exit peek) ----
    u8 ext_tgt_live[T2_MAX_UNROLL_INSNS];
    u8 ext_fall_live[T2_MAX_UNROLL_INSNS];
    for (int i = 0; i < ucnt; i++) {
        ext_tgt_live[i] = u_tgt[i] == -1 ? t2_peek_exit_flags(u_ext_tgt[i], win_lo, win_hi) : F_ARITH;
        ext_fall_live[i] = u_fall[i] == -1 ? t2_peek_exit_flags(u_ext_fall[i], win_lo, win_hi) : F_ARITH;
    }
    tr_t2_clear_snapshot();

    u8 live_after[T2_MAX_UNROLL_INSNS];
    u8 live_before[T2_MAX_UNROLL_INSNS];
    memset(live_after, 0, (size_t)ucnt);
    memset(live_before, 0, (size_t)ucnt);
    bool changed = true;
    for (int pass = 0; pass < 12 && changed; pass++) {
        changed = false;
        for (int i = ucnt - 1; i >= 0; i--) {
            const Insn *in = &ulist[i];
            u8 out = 0;
            if (u_skip_nop[i]) {
                out = live_before[u_fall[i]];
            } else if (in->op == OP_JMP) {
                int t = u_tgt[i];
                out = t == -2 ? live_before[0] : (t >= 0 ? live_before[t] : ext_tgt_live[i]);
            } else if (in->op == OP_JCC || in->op == OP_JRCXZ) {
                int t = u_tgt[i], f = u_fall[i];
                u8 lt = t == -2 ? live_before[0] : (t >= 0 ? live_before[t] : ext_tgt_live[i]);
                u8 lf = f == -2 ? live_before[0] : (f >= 0 ? live_before[f] : ext_fall_live[i]);
                out = (u8)(lt | lf);
            } else {
                int f = u_fall[i];
                out = f == -2 ? live_before[0] : (f >= 0 ? live_before[f] : ext_fall_live[i]);
            }
            u8 lb = (u8)((out & ~in->writes) | tr_insn_reads(in));
            if (out != live_after[i] || lb != live_before[i]) {
                live_after[i] = out;
                live_before[i] = lb;
                changed = true;
            }
        }
    }
    if (changed) {
        memset(live_after, F_ARITH, (size_t)ucnt);
        memset(live_before, F_ARITH, (size_t)ucnt);
    }

    // ---- Pass 3B: Guarded Stack-Slot Register Promotion ([rsp+disp]/[rbp+disp] -> {S5/x29, S4/x24}) ----
    T2StkSlot stk_slots[T2_MAX_STK_SLOTS];
    int stk_cnt = 0;
    u8 stk_base = 0;
    u8 stk_dirty_mask = 0;
    T2GuardExpr pre_guards[3];
    int pre_guard_cnt = 0;
    T2GuardExpr iter_guards[2];
    int iter_guard_cnt = 0;

    if (!t2_no_stk_promo && !tr_compat && !tr_precise_flags) {
        bool stk_loop_ok = true;
        u16 gpr_written = 0;
        for (int i = 0; i < bcnt; i++) {
            const Insn *in = &body[i];
            if (in->lock || in->op == OP_FENCE || in->op == OP_PUSH || in->op == OP_POP) {
                stk_loop_ok = false;
                break;
            }
            gpr_written |= t2_insn_written_gprs(in);
        }
        // Check S5 (x29) availability across the entire unrolled body using exact live_after[].
        bool s5_free = stk_loop_ok;
        for (int i = 0; s5_free && i < ucnt; i++) {
            if (!u_skip_nop[i] && t2_insn_uses_s5(&ulist[i], live_after[i]))
                s5_free = false;
        }
        if (stk_loop_ok && s5_free) {
            // Post-dominance: body[i] executes unconditionally on every entry iff no prior branch exits or skips past i.
            bool body_uncond[T2_MAX_BODY_INSNS];
            bool saw_exit = false;
            int max_fwd = -1;
            for (int i = 0; i < bcnt; i++) {
                body_uncond[i] = (!saw_exit && i >= max_fwd);
                const Insn *in = &body[i];
                if (in->op == OP_JCC || in->op == OP_JRCXZ || in->op == OP_JMP) {
                    if (in->target == d->head_rip) {
                        if (in->op == OP_JMP)
                            saw_exit = true;
                    } else {
                        int ti = t2_find_rip(body, bcnt, in->target);
                        if (ti < 0)
                            saw_exit = true;
                        else if (ti > i && ti > max_fwd)
                            max_fwd = ti;
                    }
                }
            }

            // Try RSP (4) first, then RBP (5).
            const u8 bases[2] = {4, 5};
            for (int bi = 0; bi < 2 && stk_cnt == 0; bi++) {
                u8 cand_base = bases[bi];
                if (gpr_written & (1u << cand_base))
                    continue;
                // Reject base if any instruction performs indexed addressing through cand_base.
                bool indexed_base = false;
                for (int i = 0; i < bcnt; i++) {
                    const Insn *in = &body[i];
                    if (in->memop && !in->riprel && !in->no_index &&
                        ((!in->no_base && in->base == cand_base) || in->index == cand_base)) {
                        indexed_base = true;
                        break;
                    }
                }
                if (indexed_base)
                    continue;

                // Gather candidate [cand_base + disp] slots.
                struct {
                    i64 disp;
                    u8 sz;
                    int reads, writes;
                    int uncond_reads, uncond_writes;
                    bool bad;
                } cands[16];
                int ncand = 0;
                for (int i = 0; i < bcnt; i++) {
                    const Insn *in = &body[i];
                    if (!in->memop || in->rm_reg || in->riprel || in->no_base || !in->no_index ||
                        in->seg || in->addr32 || in->addr16 || in->base != cand_base)
                        continue;
                    bool rd = false, wr = false;
                    if (!t2_stk_op_supported(in, &rd, &wr))
                        continue;
                    u8 msz = (u8)(in->op == OP_MOVSX ? in->srcsz : in->sz);
                    int found = -1;
                    for (int k = 0; k < ncand; k++) {
                        if (cands[k].disp == in->disp && cands[k].sz == msz) { found = k; break; }
                    }
                    if (found < 0 && ncand < 16) {
                        found = ncand++;
                        cands[found].disp = in->disp;
                        cands[found].sz = msz;
                        cands[found].reads = cands[found].writes = 0;
                        cands[found].uncond_reads = cands[found].uncond_writes = 0;
                        cands[found].bad = false;
                    }
                    if (found >= 0) {
                        if (rd) { cands[found].reads++; if (body_uncond[i]) cands[found].uncond_reads++; }
                        if (wr) { cands[found].writes++; if (body_uncond[i]) cands[found].uncond_writes++; }
                    }
                }
                if (ncand == 0)
                    continue;

                // Disqualify any candidate overlapping a different-size/different-disp/unsupported access on cand_base.
                for (int i = 0; i < bcnt; i++) {
                    const Insn *in = &body[i];
                    if (!in->memop || in->rm_reg || in->riprel || in->no_base || !in->no_index ||
                        in->seg || in->addr32 || in->addr16 || in->base != cand_base)
                        continue;
                    bool rd = false, wr = false;
                    bool supp = t2_stk_op_supported(in, &rd, &wr);
                    int msz = supp ? (in->op == OP_MOVSX ? in->srcsz : in->sz) : t2_mem_access_bytes(in);
                    i64 lo = in->disp, hi = in->disp + msz;
                    for (int k = 0; k < ncand; k++) {
                        i64 clo = cands[k].disp, chi = cands[k].disp + cands[k].sz;
                        if (lo < chi && hi > clo) {
                            if (!supp || lo != clo || msz != cands[k].sz)
                                cands[k].bad = true;
                        }
                    }
                }

                // Zero speculative #PF filter + benefit filter; pick up to 2 hottest non-overlapping slots.
                T2StkSlot picked[T2_MAX_STK_SLOTS];
                int npick = 0;
                for (int pick_i = 0; pick_i < (int)T2_MAX_STK_SLOTS; pick_i++) {
                    int best = -1, best_score = 0;
                    for (int k = 0; k < ncand; k++) {
                        if (cands[k].bad)
                            continue;
                        // Unconditional access guarantees Preheader LDR (and write-probe STR for RW) never speculatively #PFs.
                        if (cands[k].writes > 0) {
                            if (cands[k].uncond_writes == 0)
                                continue;
                        } else {
                            if (cands[k].uncond_reads == 0)
                                continue;
                        }
                        int total = cands[k].reads + cands[k].writes;
                        if (total < 2 && cands[k].writes == 0 && copies <= 1)
                            continue;
                        // Must not overlap already-picked slots.
                        bool ov = false;
                        for (int p = 0; p < npick; p++) {
                            if (cands[k].disp < picked[p].disp + picked[p].sz &&
                                cands[k].disp + cands[k].sz > picked[p].disp)
                                ov = true;
                        }
                        if (ov)
                            continue;
                        int score = cands[k].writes * 3 + cands[k].reads * 2;
                        if (score > best_score) {
                            best_score = score;
                            best = k;
                        }
                    }
                    if (best < 0)
                        break;
                    picked[npick].disp = cands[best].disp;
                    picked[npick].sz = cands[best].sz;
                    picked[npick].writes = cands[best].writes > 0;
                    picked[npick].accesses = cands[best].reads + cands[best].writes;
                    picked[npick].phys = (npick == 0) ? S5 : S4;
                    npick++;
                }
                if (npick == 0)
                    continue;

                // If 2 slots were picked, verify S4 (x24) is actually free across all non-promoted instructions.
                if (npick > 1) {
                    bool s4_free = true;
                    for (int i = 0; s4_free && i < ucnt; i++) {
                        if (u_skip_nop[i])
                            continue;
                        bool prom = t2_match_stk_slot(&ulist[i], cand_base, picked, npick) >= 0;
                        if (t2_insn_uses_s4(&ulist[i], live_after[i], prom))
                            s4_free = false;
                    }
                    if (!s4_free)
                        npick = 1;
                }

                // Sound alias analysis for ALL non-promoted stores in body[0..bcnt-1].
                T2GuardExpr pg[3], ig[2];
                int pg_cnt = 0, ig_cnt = 0;
                bool alias_ok = true;
                for (int i = 0; i < bcnt && alias_ok; i++) {
                    const Insn *in = &body[i];
                    if (!t2_insn_stores_mem(in))
                        continue;
                    if (t2_match_stk_slot(in, cand_base, picked, npick) >= 0)
                        continue;
                    if (!in->riprel && !in->no_base && in->no_index && !in->seg && !in->addr32 && !in->addr16) {
                        if (in->base == cand_base)
                            continue; // already verified non-overlapping with picked[] above
                        if (in->disp < -1024 || in->disp > 1024) {
                            alias_ok = false;
                            break;
                        }
                        u8 rb = in->base;
                        if (!(gpr_written & (1u << rb))) {
                            bool dup = false;
                            for (int g = 0; g < pg_cnt; g++)
                                if (pg[g].base == rb && pg[g].idx == 0xFF) dup = true;
                            if (!dup) {
                                if (pg_cnt >= 3) { alias_ok = false; break; }
                                pg[pg_cnt++] = (T2GuardExpr){rb, 0xFF, 0};
                            }
                        } else {
                            i64 stride = 0;
                            if (!t2_gpr_bounded_stride(body, bcnt, rb, &stride) || stride * copies > 2048) {
                                alias_ok = false;
                                break;
                            }
                            bool dup = false;
                            for (int g = 0; g < ig_cnt; g++)
                                if (ig[g].base == rb && ig[g].idx == 0xFF) dup = true;
                            if (!dup) {
                                if (ig_cnt >= 2) { alias_ok = false; break; }
                                ig[ig_cnt++] = (T2GuardExpr){rb, 0xFF, 0};
                            }
                        }
                    } else if (!in->riprel && !in->no_base && !in->no_index && !in->seg &&
                               !in->addr32 && !in->addr16 && in->disp >= -1024 && in->disp <= 1024 &&
                               in->base != cand_base && in->index != cand_base &&
                               !(gpr_written & (1u << in->base))) {
                        u8 rb = in->base, idx = in->index, sc = in->scale;
                        if (!(gpr_written & (1u << idx))) {
                            bool dup = false;
                            for (int g = 0; g < pg_cnt; g++)
                                if (pg[g].base == rb && pg[g].idx == idx && pg[g].scale == sc) dup = true;
                            if (!dup) {
                                if (pg_cnt >= 3) { alias_ok = false; break; }
                                pg[pg_cnt++] = (T2GuardExpr){rb, idx, sc};
                            }
                        } else {
                            i64 stride = 0;
                            if (!t2_gpr_bounded_stride(body, bcnt, idx, &stride) ||
                                (stride << sc) * copies > 2048) {
                                alias_ok = false;
                                break;
                            }
                            bool dup = false;
                            for (int g = 0; g < ig_cnt; g++)
                                if (ig[g].base == rb && ig[g].idx == idx && ig[g].scale == sc) dup = true;
                            if (!dup) {
                                if (ig_cnt >= 2) { alias_ok = false; break; }
                                ig[ig_cnt++] = (T2GuardExpr){rb, idx, sc};
                            }
                        }
                    } else {
                        alias_ok = false;
                    }
                }
                if (!alias_ok)
                    continue;

                stk_base = cand_base;
                stk_cnt = npick;
                for (int k = 0; k < npick; k++) {
                    stk_slots[k] = picked[k];
                    if (picked[k].writes)
                        stk_dirty_mask |= (u8)(1u << k);
                }
                pre_guard_cnt = pg_cnt;
                for (int g = 0; g < pg_cnt; g++) pre_guards[g] = pg[g];
                iter_guard_cnt = ig_cnt;
                for (int g = 0; g < ig_cnt; g++) iter_guards[g] = ig[g];
            }
        }
    }

    // Only promote if at least one Tier-2 optimization genuinely improves this loop.
    if (copies <= 1 && res_ymm_cnt == 0 && ro_pin_cnt == 0 && stk_cnt == 0 &&
        licm_ad_page == ~0ULL && !dirty_ymm_boost) {
        tr_t2_mode = false;
        return false;
    }

    bool has_t2_dirty = (res_ymm_wr_mask != 0) || (stk_dirty_mask != 0);

    // ---- Pass 4: Emit Preheader + Optimized Loop Body into Thread-Local Buffer ----
    u32 local_buf[T2_MAX_CODE_WORDS];
    PcMap local_pcm[T2_MAX_PCMAP];
    u32 pcm_cnt = 0;
    u32 *insn_arm[T2_MAX_UNROLL_INSNS];
    T2Pend pend[T2_MAX_PEND];
    int pend_cnt = 0;
    T2Fwd fwd[T2_MAX_FWD];
    int fwd_cnt = 0;
    u32 *pre_bail_sites[4];
    int pre_bail_cnt = 0;
    u32 *body_bail_sites[4];
    int body_bail_cnt = 0;
    u32 bail_sites[T2_MAX_BAIL];
    u32 bail_cnt = 0;
    bool overflow = false;

    tr_cc = local_buf;
    if (tr_aot_active) {
        tr_aot_block_start = local_buf;
        aot_capture_begin(&tr_aot_cap);
    }
    tr_ad_page = ~0ULL;
    tr_zext8 = tr_zext16 = 0;
    tr_aes_mask = 0;
    ext_ymm_cache_reset();

    // Entry PcMap at offset 0 (head_rip).
    local_pcm[pcm_cnt++] = (PcMap){0, 0};

    // Preheader (non-flag-setting instructions only: preserves NZCV/R_PF/R_AF from Tier-1 OSR entry!).
    // 1. Preheader-only alias guards (before modifying any resident YMM/stack state -> zero-flush direct B to T1_body!).
    for (int g = 0; g < pre_guard_cnt; g++)
        pre_bail_sites[pre_bail_cnt++] = t2_emit_alias_guard(pre_guards[g], stk_base);
    // 2. Pin memop-safe resident YMM8..15 (or legacy RO pins + mem-free preloads).
    for (int k = 0; k < res_ymm_cnt; k++)
        ext_ymm_cache_pin_rw(res_ymm_regs[k]);
    for (int k = 0; k < ro_pin_cnt; k++)
        ext_ymm_cache_pin_ro(ro_pin_regs[k]);
    for (int k = 0; k < rw_preload_cnt; k++)
        ext_ymm_cache_preload_clean(rw_preload_regs[k]);
    // 3. Load promoted stack slots into {S5/x29, S4/x24} + write-probe dirty RW slots before body_start_word.
    for (int k = 0; k < stk_cnt; k++) {
        t2_emit_stk_ldst(true, stk_slots[k].sz, stk_slots[k].phys, stk_base, stk_slots[k].disp);
        if (stk_slots[k].writes)
            t2_emit_stk_ldst(false, stk_slots[k].sz, stk_slots[k].phys, stk_base, stk_slots[k].disp);
    }
    // 4. Preheader LICM: hoist RIP-relative page base into AD (x20).
    if (licm_ad_page != ~0ULL) {
        tr_emit_movimm_gva(AD, licm_ad_page, false);
        tr_ad_page = licm_ad_page;
    }
    if (dirty_ymm_boost)
        ext_ymm_set_dirty_limit(24);

    u32 *loop_hdr_arm = tr_cc;
    u32 body_start_word = (u32)(loop_hdr_arm - local_buf);
    if (loop_hdr_arm > local_buf && pcm_cnt < T2_MAX_PCMAP)
        local_pcm[pcm_cnt++] = (PcMap){(u16)body_start_word, 0};

    // Per-unrolled-iteration alias guards (for bounded-stride stores): placed atomically at loop_hdr_arm (head_rip).
    for (int g = 0; g < iter_guard_cnt; g++)
        body_bail_sites[body_bail_cnt++] = t2_emit_alias_guard(iter_guards[g], stk_base);

    int dirty_limit = dirty_ymm_boost ? 24 : 8;

    for (int i = 0; i < ucnt; i++) {
        if ((u32)(tr_cc - local_buf) >= T2_MAX_CODE_WORDS - 512) {
            overflow = true;
            break;
        }
        const Insn *in = &ulist[i];
        if (u_skip_nop[i]) {
            insn_arm[i] = tr_cc;
            continue;
        }
        bool prev_fell = i > 0 && !u_skip_nop[i - 1] && !t2_is_uncond_term(ulist[i - 1].op) &&
                         u_fall[i - 1] == i;
        if (u_is_target[i]) {
            if (prev_fell) {
                if (tr_aes_mask)
                    sse_aes_flush_all();
                if (ext_ymm_cache_has_dirty())
                    ext_ymm_cache_flush_dirty();
            }
            if (licm_ad_page == ~0ULL)
                tr_ad_page = ~0ULL;
            tr_zext8 = tr_zext16 = 0;
            ext_ymm_cache_reset_unpinned();
        }
        // Non-resident dynamic YMM slots still flush cleanly before memop/target/dirty_limit;
        // resident slots (ymm_t2_resident_mask) are skipped by ext_ymm_cache_has_dirty() and stay live!
        bool op_needs_flush = dirty_ymm_boost
                                  ? (in->memop || u_is_target[i] || ext_ymm_cache_dirty_run() >= dirty_limit ||
                                     in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR)
                                  : (in->op != OP_EXT || in->memop || u_is_target[i] || ext_ymm_cache_dirty_run() >= dirty_limit);
        if (ext_ymm_cache_has_dirty() && op_needs_flush) {
            ext_ymm_cache_flush_dirty();
        }
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR) {
            ext_ymm_cache_flush_and_drop();
        }
        if (licm_ad_page == ~0ULL && !tr_ad_page_safe_op(in->op))
            tr_ad_page = ~0ULL;
        if (in->op != OP_SSE && tr_aes_mask)
            sse_aes_flush_all();

        insn_arm[i] = tr_cc;
        if (!ext_ymm_cache_has_dirty() && pcm_cnt < T2_MAX_PCMAP) {
            // Note: for i == 0 with iter_guard_cnt > 0, keep PcMap at loop_hdr_arm (before the guard)
            // so the per-iteration guard executes atomically with insn 0.
            if (!(i == 0 && iter_guard_cnt > 0)) {
                u16 aw = (u16)(tr_cc - local_buf);
                u16 ro = (u16)(in->rip - d->head_rip);
                if (pcm_cnt == 0 || local_pcm[pcm_cnt - 1].arm_word != aw)
                    local_pcm[pcm_cnt++] = (PcMap){aw, ro};
            }
        }

        // Peephole fusions across adjacent non-target instructions (same core fusions as Tier-1).
        bool fused = false;
        bool next_same_seq = (i + 1 < ucnt) && !u_is_target[i + 1] && !u_skip_nop[i + 1] &&
                             (u_fall[i] == i + 1) && (in->next == ulist[i + 1].rip);
        if (tr_opt_peep && next_same_seq) {
            const Insn *nx = &ulist[i + 1];
            int madd_consumed = 0;
            if (in->op == OP_EXT && nx->op == OP_EXT &&
                ext_try_emit_madd_dot(ulist, i, ucnt, u_is_target, &madd_consumed)) {
                for (int k = 1; k < madd_consumed; k++)
                    insn_arm[i + k] = insn_arm[i];
                tr_zext8 = tr_zext16 = 0;
                i += madd_consumed - 1;
                fused = true;
            } else if (in->op == OP_SSE && nx->op == OP_SSE &&
                       sse_try_emit_madd_dot(ulist, i, ucnt, u_is_target, &madd_consumed)) {
                for (int k = 1; k < madd_consumed; k++)
                    insn_arm[i + k] = insn_arm[i];
                tr_zext8 = tr_zext16 = 0;
                i += madd_consumed - 1;
                fused = true;
            } else if (in->op == OP_ALU && nx->op == OP_JCC && (nx->cc == 4 || nx->cc == 5) && live_after[i + 1] == 0) {
                int cbz_reg = -1, cbz_sf = 0;
                if (in->sub == ALU_TEST && !in->has_imm && in->rm_reg && !in->high_reg && !in->high_rm && in->reg == in->rm) {
                    if (in->sz == 4 || in->sz == 8) { cbz_reg = in->rm; cbz_sf = in->sz == 8; }
                    else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) { cbz_reg = in->rm; cbz_sf = 0; }
                    else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) { cbz_reg = in->rm; cbz_sf = 0; }
                } else if (in->sub == ALU_CMP && in->has_imm && in->imm == 0 && in->rm_reg && !in->high_rm) {
                    if (in->sz == 4 || in->sz == 8) { cbz_reg = in->rm; cbz_sf = in->sz == 8; }
                    else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) { cbz_reg = in->rm; cbz_sf = 0; }
                    else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) { cbz_reg = in->rm; cbz_sf = 0; }
                }
                if (cbz_reg >= 0) {
                    int t = u_tgt[i + 1];
                    bool dirty_now = ext_ymm_cache_has_dirty();
                    if (t == -2) {
                        if (tr_aes_mask) sse_aes_flush_all();
                        if (dirty_now) ext_ymm_cache_flush_dirty();
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, (int)(loop_hdr_arm - tr_cc));
                    } else if (t >= 0) {
                        if (dirty_now) ext_ymm_cache_flush_dirty();
                        if (fwd_cnt < T2_MAX_FWD)
                            fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 0};
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    } else {
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i + 1], 0, has_t2_dirty);
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    i++;
                    fused = true;
                }
            }
            // 2) TEST r, pow2 + JZ/JNZ -> TBZ/TBNZ
            if (!fused && in->op == OP_ALU && in->sub == ALU_TEST && in->has_imm && in->rm_reg && !in->high_rm &&
                nx->op == OP_JCC && (nx->cc == 4 || nx->cc == 5) && live_after[i + 1] == 0) {
                u64 mask = in->sz == 8 ? (u64)in->imm : in->sz == 4 ? (u32)in->imm : in->sz == 2 ? (u16)in->imm : (u8)in->imm;
                if (mask && (mask & (mask - 1)) == 0) {
                    int bit = __builtin_ctzll(mask);
                    int t = u_tgt[i + 1];
                    bool dirty_now = ext_ymm_cache_has_dirty();
                    if (t == -2) {
                        if (tr_aes_mask) sse_aes_flush_all();
                        if (dirty_now) ext_ymm_cache_flush_dirty();
                        i64 bd = loop_hdr_arm - tr_cc;
                        if (bd >= -(1 << 13) && bd < (1 << 13)) {
                            tbz(nx->cc == 5, in->rm, bit, (int)bd);
                            insn_arm[i + 1] = insn_arm[i];
                            i++;
                            fused = true;
                        }
                    } else if (t >= 0) {
                        if (dirty_now) ext_ymm_cache_flush_dirty();
                        if (fwd_cnt < T2_MAX_FWD)
                            fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 2};
                        tbz(nx->cc == 5, in->rm, bit, 0);
                        insn_arm[i + 1] = insn_arm[i];
                        i++;
                        fused = true;
                    }
                }
            }
            // 3) BT r32/64, imm + JC/JNC -> TBNZ/TBZ
            if (!fused && in->op == OP_BT && in->sub == 0 && in->has_imm && in->rm_reg && in->sz >= 4 &&
                nx->op == OP_JCC && (nx->cc == 2 || nx->cc == 3) && live_after[i + 1] == 0) {
                int bit = (int)in->imm & (in->sz * 8 - 1);
                int t = u_tgt[i + 1];
                bool dirty_now = ext_ymm_cache_has_dirty();
                if (t == -2) {
                    if (tr_aes_mask) sse_aes_flush_all();
                    if (dirty_now) ext_ymm_cache_flush_dirty();
                    i64 bd = loop_hdr_arm - tr_cc;
                    if (bd >= -(1 << 13) && bd < (1 << 13)) {
                        tbz(nx->cc == 2, in->rm, bit, (int)bd);
                        insn_arm[i + 1] = insn_arm[i];
                        i++;
                        fused = true;
                    }
                } else if (t >= 0) {
                    if (dirty_now) ext_ymm_cache_flush_dirty();
                    if (fwd_cnt < T2_MAX_FWD)
                        fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 2};
                    tbz(nx->cc == 2, in->rm, bit, 0);
                    insn_arm[i + 1] = insn_arm[i];
                    i++;
                    fused = true;
                }
            }
            // 4) SETcc r8 + MOVZX r32/64, r8 -> single CSET
            if (!fused && in->op == OP_SETCC && in->rm_reg && !in->high_rm &&
                nx->op == OP_MOVZX && nx->rm_reg && !nx->high_rm && nx->srcsz == 1 &&
                nx->rm == in->rm && nx->reg == in->rm && x86_to_arm_cc[in->cc] >= 0) {
                cset(nx->reg, x86_to_arm_cc[in->cc]);
                tr_zext8 |= (u16)(1u << nx->reg);
                tr_zext16 |= (u16)(1u << nx->reg);
                insn_arm[i + 1] = insn_arm[i];
                i++;
                fused = true;
            }
            // 5) Adjacent MOV [base+d], r1; MOV [base+d±sz], r2 -> LDP/STP (skip if either is a promoted stack slot!)
            if (!fused && in->op == OP_MOV && nx->op == OP_MOV && !tr_compat && !tr_precise_flags &&
                !in->has_imm && !nx->has_imm && !in->rm_reg && !nx->rm_reg &&
                in->to_reg == nx->to_reg && in->sz == nx->sz && (in->sz == 4 || in->sz == 8) &&
                !in->seg && !nx->seg && !in->riprel && !nx->riprel &&
                !in->addr32 && !nx->addr32 && !in->addr16 && !nx->addr16 &&
                !in->no_base && !nx->no_base && in->no_index && nx->no_index &&
                in->base == nx->base && (tr_cpl == 3 || in->base == 4 || in->base == 5) &&
                t2_match_stk_slot(in, stk_base, stk_slots, stk_cnt) < 0 &&
                t2_match_stk_slot(nx, stk_base, stk_slots, stk_cnt) < 0) {
                int sz = in->sz;
                i64 d1 = in->disp, d2 = nx->disp;
                bool fwd_pair = (d2 == d1 + sz);
                bool rev_pair = (d1 == d2 + sz);
                i64 lo_disp = fwd_pair ? d1 : d2;
                int r_lo = fwd_pair ? in->reg : nx->reg;
                int r_hi = fwd_pair ? nx->reg : in->reg;
                bool safe_load = !in->to_reg || (in->reg != in->base && in->reg != nx->reg);
                if ((fwd_pair || rev_pair) && safe_load && (lo_disp & (sz - 1)) == 0 &&
                    lo_disp >= -64 * sz && lo_disp <= 63 * sz) {
                    u32 imm7 = (u32)(lo_disp / sz) & 0x7fu;
                    u32 opc = sz == 8 ? (in->to_reg ? 0xA9400000u : 0xA9000000u)
                                      : (in->to_reg ? 0x29400000u : 0x29000000u);
                    E(opc | (imm7 << 15) | ((u32)r_hi << 10) | ((u32)in->base << 5) | (u32)r_lo);
                    if (in->to_reg) {
                        tr_update_zext(in);
                        tr_update_zext(nx);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    i++;
                    fused = true;
                }
            }
        }

        if (!fused) {
            if (u_iter0_jrcxz_inv[i]) {
                // Intermediate inverted LOOP (sub==2, !addr16): dec rcx/ecx; cbz -> exit, else fall into next copy.
                int sf = in->addr32 ? 0 : 1;
                add_imm(sf, 1, 0, 1, 1, 1);
                t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0, has_t2_dirty);
                cbz(0, sf, 1, 0);
            } else if (in->op == OP_JCC) {
                int t = u_tgt[i];
                bool dirty_now = ext_ymm_cache_has_dirty();
                if (t == -2) {
                    if (tr_aes_mask) sse_aes_flush_all();
                    if (dirty_now) ext_ymm_cache_flush_dirty();
                } else if (t >= 0 && dirty_now) {
                    ext_ymm_cache_flush_dirty();
                }
                int acc = x86_to_arm_cc[in->cc];
                if (acc < 0) {
                    emit_parity(S4);
                    if (t == -2) {
                        cbz(in->cc == 10, 0, S4, (int)(loop_hdr_arm - tr_cc));
                    } else if (t >= 0) {
                        if (fwd_cnt < T2_MAX_FWD)
                            fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 0};
                        cbz(in->cc == 10, 0, S4, 0);
                    } else {
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0, has_t2_dirty);
                        cbz(in->cc == 10, 0, S4, 0);
                    }
                } else {
                    if (t == -2) {
                        b_cond(acc, (int)(loop_hdr_arm - tr_cc));
                    } else if (t >= 0) {
                        if (fwd_cnt < T2_MAX_FWD)
                            fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 0};
                        b_cond(acc, 0);
                    } else {
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0, has_t2_dirty);
                        b_cond(acc, 0);
                    }
                }
            } else if (in->op == OP_JRCXZ) {
                int t = u_tgt[i];
                bool dirty_now = ext_ymm_cache_has_dirty();
                if (t == -2) {
                    if (tr_aes_mask) sse_aes_flush_all();
                    if (dirty_now) ext_ymm_cache_flush_dirty();
                } else if (t >= 0 && dirty_now) {
                    ext_ymm_cache_flush_dirty();
                }
                int sub = in->sub, sf = in->addr32 ? 0 : 1;
                if (in->addr16) {
                    if (sub >= 2) {
                        add_imm(0, 1, 0, S0, 1, 1);
                        bfi(1, 1, S0, 0, 16);
                    }
                    ubfx(0, S0, 1, 0, 16);
                    u32 *over = NULL;
                    if (sub >= 3) { over = tr_cc; b_cond(sub == 3 ? 1 : 0, 0); }
                    if (t == -2) {
                        cbz(sub >= 2, 0, S0, (int)(loop_hdr_arm - tr_cc));
                    } else if (t >= 0) {
                        if (fwd_cnt < T2_MAX_FWD) fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 0};
                        cbz(sub >= 2, 0, S0, 0);
                    } else {
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0, has_t2_dirty);
                        cbz(sub >= 2, 0, S0, 0);
                    }
                    if (over) *over |= ((u32)(tr_cc - over) & 0x7ffffu) << 5;
                } else {
                    if (sub >= 2) add_imm(sf, 1, 0, 1, 1, 1);
                    u32 *over = NULL;
                    if (sub >= 3) { over = tr_cc; b_cond(sub == 3 ? 1 : 0, 0); }
                    int csf = (sub == 0) ? 1 : (sub == 1 ? 0 : sf);
                    if (t == -2) {
                        cbz(sub >= 2, csf, 1, (int)(loop_hdr_arm - tr_cc));
                    } else if (t >= 0) {
                        if (fwd_cnt < T2_MAX_FWD) fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 0};
                        cbz(sub >= 2, csf, 1, 0);
                    } else {
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0, has_t2_dirty);
                        cbz(sub >= 2, csf, 1, 0);
                    }
                    if (over) *over |= ((u32)(tr_cc - over) & 0x7ffffu) << 5;
                }
                tr_update_zext(in);
            } else if (in->op == OP_JMP) {
                int t = u_tgt[i];
                if (tr_aes_mask) sse_aes_flush_all();
                if (ext_ymm_cache_has_dirty()) ext_ymm_cache_flush_dirty();
                if (t == -2) {
                    b_rel((int)(loop_hdr_arm - tr_cc));
                } else if (t >= 0) {
                    if (fwd_cnt < T2_MAX_FWD)
                        fwd[fwd_cnt++] = (T2Fwd){tr_cc, t, 1};
                    b_rel(0);
                } else {
                    // Route external JMP through cold post-body_end_word stub so all exit flushes stay strictly outside body.
                    t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 1, has_t2_dirty);
                    b_rel(0);
                }
            } else if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR) {
                emit_sse(in, live_after[i]);
                tr_update_zext(in);
            } else if (in->op == OP_EXT) {
                emit_ext(in, live_after[i]);
                tr_update_zext(in);
            } else {
                int sk = t2_match_stk_slot(in, stk_base, stk_slots, stk_cnt);
                if (sk >= 0) {
                    Insn tmp = *in;
                    tmp.rm_reg = true;
                    tmp.memop = false;
                    tmp.rm = stk_slots[sk].phys;
                    emit_int(&tmp, live_after[i]);
                } else {
                    emit_int(in, live_after[i]);
                }
                tr_update_zext(in);
            }
        }

        // Disconnected fallthrough (non-adjacent sub-block inside loop).
        int f = u_fall[i];
        if (!t2_is_uncond_term(ulist[i].op) && f != i + 1 && i + 1 < ucnt) {
            if (tr_aes_mask) sse_aes_flush_all();
            if (ext_ymm_cache_has_dirty()) ext_ymm_cache_flush_dirty();
            if (f == -2) {
                b_rel((int)(loop_hdr_arm - tr_cc));
            } else if (f >= 0) {
                if (fwd_cnt < T2_MAX_FWD)
                    fwd[fwd_cnt++] = (T2Fwd){tr_cc, f, 1};
                b_rel(0);
            } else if (f == -1) {
                t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_fall[i], 1, has_t2_dirty);
                b_rel(0);
            }
        }
    }

    if (overflow) {
        ext_ymm_cache_reset();
        tr_t2_mode = false;
        return false;
    }

    if (tr_aes_mask)
        sse_aes_flush_all();
    if (ext_ymm_cache_has_dirty())
        ext_ymm_cache_flush_dirty();

    // Mark exact end of live resident-state loop body before epilogue & cold exit stubs.
    u32 body_end_word = (u32)(tr_cc - local_buf);

    // Resolve internal forward jumps.
    for (int k = 0; k < fwd_cnt; k++)
        t2_fix(fwd[k].site, fwd[k].form, insn_arm[fwd[k].target_idx]);

    // Fallthrough after last instruction of loop (typically loop exit after final latch JCC).
    int last_f = u_fall[ucnt - 1];
    if (!t2_is_uncond_term(ulist[ucnt - 1].op)) {
        if (last_f == -2) {
            b_rel((int)(loop_hdr_arm - tr_cc));
        } else if (last_f == -1) {
            if (res_ymm_wr_mask)
                ext_ymm_emit_resident_flush(res_ymm_wr_mask);
            for (int sk = 0; sk < stk_cnt; sk++)
                if (stk_slots[sk].writes)
                    t2_emit_stk_ldst(false, stk_slots[sk].sz, stk_slots[sk].phys, stk_base, stk_slots[sk].disp);
            svc(SVC_DIRECT);
            t2_stub_lit(u_ext_fall[ucnt - 1]);
        }
    }

    // Emit side-exit stubs (with exact dirty YMM8..15 + dirty promoted stack-slot flush trampolines).
    for (int k = 0; k < pend_cnt; k++) {
        if ((u32)(tr_cc - local_buf) >= T2_MAX_CODE_WORDS - 32) {
            overflow = true;
            break;
        }
        u32 *stub = tr_cc;
        if (pend[k].dirty_cnt > 0 || has_t2_dirty) {
            if (pend[k].dirty_cnt > 0)
                ext_ymm_emit_flush_list(pend[k].dirty_cnt, pend[k].dirty_vreg, pend[k].dirty_xreg);
            if (res_ymm_wr_mask)
                ext_ymm_emit_resident_flush(res_ymm_wr_mask);
            for (int sk = 0; sk < stk_cnt; sk++)
                if (stk_slots[sk].writes)
                    t2_emit_stk_ldst(false, stk_slots[sk].sz, stk_slots[sk].phys, stk_base, stk_slots[sk].disp);
            svc(SVC_DIRECT);
            t2_stub_lit(pend[k].target);
            t2_fix(pend[k].site, pend[k].form, stub);
        } else {
            svc(pend[k].kind);
            t2_stub_lit(pend[k].target);
            if (pend[k].kind == SVC_JCC)
                E((u32)(stub - pend[k].site));
            t2_fix(pend[k].site, pend[k].form, stub);
        }
    }

    // Emit alias-guard Tier-1 fallback trampolines (patched by tr_t2_publish to direct B (t1_patch + 1)).
    if (pre_bail_cnt > 0 && (u32)(tr_cc - local_buf) < T2_MAX_CODE_WORDS - 8 && bail_cnt < T2_MAX_BAIL) {
        u32 *stub = tr_cc;
        for (int g = 0; g < pre_bail_cnt; g++)
            t2_fix(pre_bail_sites[g], 0, stub);
        bail_sites[bail_cnt++] = (u32)(tr_cc - local_buf);
        b_rel(0);
    }
    if (body_bail_cnt > 0 && (u32)(tr_cc - local_buf) < T2_MAX_CODE_WORDS - 24 && bail_cnt < T2_MAX_BAIL) {
        u32 *stub = tr_cc;
        for (int g = 0; g < body_bail_cnt; g++)
            t2_fix(body_bail_sites[g], 0, stub);
        if (res_ymm_wr_mask)
            ext_ymm_emit_resident_flush(res_ymm_wr_mask);
        for (int sk = 0; sk < stk_cnt; sk++)
            if (stk_slots[sk].writes)
                t2_emit_stk_ldst(false, stk_slots[sk].sz, stk_slots[sk].phys, stk_base, stk_slots[sk].disp);
        bail_sites[bail_cnt++] = (u32)(tr_cc - local_buf);
        b_rel(0);
    }

    T2TrapMeta tmeta;
    memset(&tmeta, 0, sizeof tmeta);
    if (has_t2_dirty) {
        tmeta.body_start_word = body_start_word;
        tmeta.body_end_word = body_end_word;
        ext_ymm_cache_get_resident_map(tmeta.ymm_reg);
        tmeta.ymm_wr_mask = res_ymm_wr_mask;
        tmeta.stk_cnt = (u8)stk_cnt;
        tmeta.stk_base = stk_base;
        tmeta.stk_dirty_mask = stk_dirty_mask;
        for (int sk = 0; sk < stk_cnt; sk++) {
            tmeta.stk_sz[sk] = stk_slots[sk].sz;
            tmeta.stk_phys[sk] = stk_slots[sk].phys;
            tmeta.stk_disp[sk] = stk_slots[sk].disp;
        }
        tmeta.flags = T2_META_FLAG_ACTIVE;
    }

    u32 code_words = (u32)(tr_cc - local_buf);
    ext_ymm_cache_reset();
    tr_t2_mode = false;

    if (overflow || code_words == 0 || code_words >= T2_MAX_CODE_WORDS) {
        tr_aot_cap.active = false;
        return false;
    }

    __atomic_add_fetch(&t2_stat_compiled, 1, __ATOMIC_RELAXED);
    bool ok = tr_t2_publish(d->t1_idx, d->part, d->p0_slot, d->epoch, d->head_rip,
                            d->key, d->cpl, d->compat, d->patch_word, d->frame0,
                            d->min_off, d->span_bytes, snap,
                            local_buf, code_words, local_pcm, pcm_cnt,
                            &tmeta, bail_sites, bail_cnt);
    if (ok) {
        if (tr_aot_active && tr_aot_cap.active && !tr_aot_cap.ineligible) {
            tr_aot_record_t2(d->frame0, d->head_rip, d->cpl, d->compat,
                             local_buf, code_words, local_pcm, pcm_cnt,
                             &tmeta, bail_sites, bail_cnt, &tr_aot_cap);
        }
        tr_aot_cap.active = false;
        __atomic_add_fetch(&t2_stat_promoted, 1, __ATOMIC_RELAXED);
        if (copies == 4) __atomic_add_fetch(&t2_stat_unrolled4, 1, __ATOMIC_RELAXED);
        else if (copies == 2) __atomic_add_fetch(&t2_stat_unrolled, 1, __ATOMIC_RELAXED);
        if (res_ymm_cnt > 0) __atomic_add_fetch(&t2_stat_res_ymm_loops, 1, __ATOMIC_RELAXED);
        if (ro_pin_cnt > 0) __atomic_add_fetch(&t2_stat_ro_ymm_loops, 1, __ATOMIC_RELAXED);
        if (stk_cnt > 0) __atomic_add_fetch(&t2_stat_stk_promo_loops, 1, __ATOMIC_RELAXED);
        if (licm_ad_page != ~0ULL) __atomic_add_fetch(&t2_stat_licm_ad_loops, 1, __ATOMIC_RELAXED);
        if (t2_log_enabled) {
            fprintf(stderr, "[hvdbt-t2] promoted rip=%llx part=%u slot=%u body=%d unroll=%d res_ymm=%d(wr=%x) stk=%d(wr=%x,pg=%d,ig=%d) licm_ad=%d words=%u\n",
                    (unsigned long long)d->head_rip, d->part, d->p0_slot, bcnt, copies,
                    res_ymm_cnt, res_ymm_wr_mask, stk_cnt, stk_dirty_mask, pre_guard_cnt, iter_guard_cnt,
                    licm_ad_page != ~0ULL ? 1 : 0, code_words);
        }
    } else {
        tr_aot_cap.active = false;
    }
    return ok;
}

static T2LoopDesc *t2_find_by_word_locked(u32 word) {
    u32 b;
    if (word >= CC_PART2_WORD) {
        b = 4U;
    } else if (t2_ctx && t2_ctx->p0_ring_active) {
        int s = engine_p0_slot_of_word(word);
        if (s < 0)
            return NULL;
        b = (u32)s;
    } else {
        b = 0U;
    }
    T2LoopDesc *arr = t2_buckets[b];
    u32 cnt = t2_bucket_cnt[b];
    if (!arr || cnt == 0)
        return NULL;
    u32 lo = 0, hi = cnt;
    while (lo < hi) {
        u32 mid = lo + (hi - lo) / 2;
        if (arr[mid].entry_word <= word)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return NULL;
    T2LoopDesc *ld = &arr[lo - 1];
    return (word >= ld->entry_word && word < ld->entry_word + ld->arm_words) ? ld : NULL;
}

static void *t2_worker_main(void *arg) {
    (void)arg;
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    struct timespec ts = {0, (long)T2_POLL_US * 1000L};

    while (!__atomic_load_n(&t2_thread_stop, __ATOMIC_ACQUIRE)) {
        nanosleep(&ts, NULL);
        if (!t2_ctx || !t2_is_enabled)
            continue;
        u32 vc = __atomic_load_n(&t2_ctx->vcpu_count, __ATOMIC_RELAXED);
        if (vc == 0) vc = 1;
        if (vc > MAX_VCPUS) vc = MAX_VCPUS;

        u64 now = t2_now_ms();
        T2LoopDesc batch[4];
        int batch_n = 0;

        pthread_mutex_lock(&t2_meta_lock);
        for (u32 vi = 0; vi < vc; vi++) {
            const EngineCtx *vc_ctx = (const EngineCtx *)((const u8 *)t2_ctx + (u64)vi * CTX_SLOT);
            u32 head = __atomic_load_n(&vc_ctx->t2_sample_head, __ATOMIC_ACQUIRE);
            u32 tail = t2_sample_tail[vi];
            if (head == tail)
                continue;
            if (head - tail > T2_SAMPLE_RING)
                tail = head - T2_SAMPLE_RING;
            while (tail != head) {
                u64 pc = vc_ctx->t2_sample_ring[tail & (T2_SAMPLE_RING - 1U)];
                tail++;
                t2_stat_samples++;
                if (pc < PRIV_VA + OFF_CC || pc >= PRIV_VA + OFF_CC2 + CC_SIZE)
                    continue;
                u32 word = (u32)((pc - (PRIV_VA + OFF_CC)) >> 2);
                // Ignore samples already inside CC_T2.
                if ((word >= CC_T2_WORD0 && word < CC_T2_WORD0 + CC_T2_WORDS) ||
                    (word >= CC_T2_WORD1 && word < CC_T2_WORD1 + CC_T2_WORDS))
                    continue;
                T2LoopDesc *ld = t2_find_by_word_locked(word);
                if (!ld || ld->state != T2_IDLE)
                    continue;
                if (now - ld->win_start_ms > T2_WINDOW_MS) {
                    ld->win_start_ms = now;
                    ld->hits = 1;
                } else {
                    ld->hits++;
                }
                if (ld->hits >= t2_hit_thresh && ld->attempts < 2 && batch_n < 4) {
                    ld->state = T2_COMPILING;
                    ld->attempts++;
                    batch[batch_n++] = *ld;
                }
            }
            t2_sample_tail[vi] = tail;
        }
        pthread_mutex_unlock(&t2_meta_lock);

        for (int i = 0; i < batch_n; i++) {
            bool ok = t2_compile_and_publish(&batch[i]);
            pthread_mutex_lock(&t2_meta_lock);
            T2LoopDesc *ld = t2_t1_map_find(batch[i].t1_idx);
            if (ld && ld->epoch == batch[i].epoch && ld->state == T2_COMPILING)
                ld->state = ok ? T2_PROMOTED : T2_REJECTED;
            pthread_mutex_unlock(&t2_meta_lock);
        }
    }
    return NULL;
}

static bool t2_summary_registered;
static void t2_print_summary(void) {
    u64 reg = __atomic_load_n(&t2_stat_registered, __ATOMIC_RELAXED);
    u64 smp = __atomic_load_n(&t2_stat_samples, __ATOMIC_RELAXED);
    u64 cmp = __atomic_load_n(&t2_stat_compiled, __ATOMIC_RELAXED);
    u64 pro = __atomic_load_n(&t2_stat_promoted, __ATOMIC_RELAXED);
    u64 unr = __atomic_load_n(&t2_stat_unrolled, __ATOMIC_RELAXED);
    u64 un4 = __atomic_load_n(&t2_stat_unrolled4, __ATOMIC_RELAXED);
    u64 res = __atomic_load_n(&t2_stat_res_ymm_loops, __ATOMIC_RELAXED);
    u64 ro  = __atomic_load_n(&t2_stat_ro_ymm_loops, __ATOMIC_RELAXED);
    u64 stk = __atomic_load_n(&t2_stat_stk_promo_loops, __ATOMIC_RELAXED);
    u64 ad  = __atomic_load_n(&t2_stat_licm_ad_loops, __ATOMIC_RELAXED);
    fprintf(stderr, "[hvdbt-t2-summary] reg=%llu samples=%llu compiled=%llu promoted=%llu unroll2=%llu unroll4=%llu res_ymm=%llu ro_ymm=%llu stk_promo=%llu licm_ad=%llu\n",
            (unsigned long long)reg, (unsigned long long)smp, (unsigned long long)cmp,
            (unsigned long long)pro, (unsigned long long)unr, (unsigned long long)un4,
            (unsigned long long)res, (unsigned long long)ro, (unsigned long long)stk,
            (unsigned long long)ad);
}

void t2_init(const TrConfig *cfg, EngineCtx *ctx, u32 *cc_base_p) {
    mach_timebase_info(&t2_tb);
    const char *no_t2 = getenv("LC_HVDBT_NO_T2");
    const char *eager = getenv("LC_HVDBT_T2_EAGER");
    const char *thresh = getenv("LC_HVDBT_T2_THRESH");
    const char *log_e = getenv("LC_HVDBT_T2_LOG");
    const char *stats_e = getenv("LC_HVDBT_T2_STATS");
    const char *no_mem_ymm = getenv("LC_HVDBT_T2_NO_MEM_YMM");
    const char *no_stk = getenv("LC_HVDBT_T2_NO_STK_PROMO");
    const char *no_u4 = getenv("LC_HVDBT_T2_NO_UNROLL4");

    t2_is_enabled = !(no_t2 && !strcmp(no_t2, "1")) && !cfg->precise_flags && !cfg->all_flags_live && !cfg->profile_compile_inputs;
    t2_is_eager = (eager && !strcmp(eager, "1"));
    t2_log_enabled = (log_e && !strcmp(log_e, "1"));
    t2_no_mem_ymm = (no_mem_ymm && !strcmp(no_mem_ymm, "1"));
    t2_no_stk_promo = (no_stk && !strcmp(no_stk, "1"));
    t2_no_unroll4 = (no_u4 && !strcmp(no_u4, "1"));
    if ((t2_log_enabled || (stats_e && !strcmp(stats_e, "1"))) && !t2_summary_registered) {
        atexit(t2_print_summary);
        t2_summary_registered = true;
    }
    t2_hit_thresh = 4;
    if (thresh && thresh[0]) {
        int v = atoi(thresh);
        if (v >= 1 && v <= 1000)
            t2_hit_thresh = (u32)v;
    }

    pthread_mutex_lock(&t2_meta_lock);
    t2_cfg = *cfg;
    t2_ctx = ctx;
    t2_cc_base = cc_base_p;
    memset(t2_sample_tail, 0, sizeof t2_sample_tail);
    if (!t2_t1_map)
        t2_t1_map = calloc(T2_T1_HASH_SIZE, sizeof(T2T1Map));
    else
        memset(t2_t1_map, 0, T2_T1_HASH_SIZE * sizeof(T2T1Map));
    for (u32 b = 0; b < T2_BUCKETS; b++) {
        if (!t2_buckets[b])
            t2_buckets[b] = calloc(T2_MAX_LOOPS_PER_BUCKET, sizeof(T2LoopDesc));
        t2_bucket_cnt[b] = 0;
    }
    t2_stat_registered = t2_stat_samples = t2_stat_compiled = t2_stat_promoted = 0;
    t2_stat_unrolled = t2_stat_unrolled4 = t2_stat_ro_ymm_loops = t2_stat_res_ymm_loops = 0;
    t2_stat_stk_promo_loops = t2_stat_licm_ad_loops = 0;
    pthread_mutex_unlock(&t2_meta_lock);

    if (t2_is_enabled && !t2_is_eager && !t2_thread_started) {
        t2_thread_stop = false;
        if (pthread_create(&t2_thread, NULL, t2_worker_main, NULL) == 0) {
            pthread_detach(t2_thread);
            t2_thread_started = true;
        }
    }
}

void t2_register_loop(u32 t1_idx, u32 part, u32 p0_slot, u32 entry_word, u32 arm_words,
                      u32 patch_word, u64 entry_rip, u64 head_rip, u64 frame0,
                      u16 min_off, u16 span_bytes, u32 key, u8 cpl, bool compat, u64 epoch) {
    if (!t2_is_enabled || !t2_t1_map)
        return;
    u32 b = t2_bucket_of(part, p0_slot);
    pthread_mutex_lock(&t2_meta_lock);
    if (t2_buckets[b] && t2_bucket_cnt[b] < T2_MAX_LOOPS_PER_BUCKET) {
        u32 pos = t2_bucket_cnt[b];
        // Maintain strict ascending entry_word order within each bucket (invariant of bump cursor).
        if (pos == 0 || t2_buckets[b][pos - 1].entry_word < entry_word) {
            T2LoopDesc *ld = &t2_buckets[b][pos];
            memset(ld, 0, sizeof *ld);
            ld->t1_idx = t1_idx;
            ld->part = part;
            ld->p0_slot = p0_slot;
            ld->entry_word = entry_word;
            ld->arm_words = arm_words;
            ld->patch_word = patch_word;
            ld->key = key;
            ld->entry_rip = entry_rip;
            ld->head_rip = head_rip;
            ld->frame0 = frame0;
            ld->epoch = epoch;
            ld->min_off = min_off;
            ld->span_bytes = span_bytes;
            ld->cpl = cpl;
            ld->compat = compat;
            ld->state = T2_IDLE;
            t2_bucket_cnt[b] = pos + 1U;
            t2_t1_map_insert(t1_idx, b, pos);
            t2_stat_registered++;
        }
    }
    pthread_mutex_unlock(&t2_meta_lock);
}

void t2_on_kill(u32 t1_idx) {
    if (!t2_is_enabled || !t2_t1_map)
        return;
    pthread_mutex_lock(&t2_meta_lock);
    T2LoopDesc *ld = t2_t1_map_find(t1_idx);
    if (ld) {
        ld->state = T2_DEAD;
        t2_t1_map_remove(t1_idx);
    }
    pthread_mutex_unlock(&t2_meta_lock);
}

void t2_on_flush(bool everything, int evicted_p0_slot, u64 new_epoch0, u64 new_epoch1) {
    (void)new_epoch0;
    (void)new_epoch1;
    if (!t2_t1_map)
        return;
    pthread_mutex_lock(&t2_meta_lock);
    if (everything) {
        for (u32 b = 0; b < T2_BUCKETS; b++)
            t2_bucket_cnt[b] = 0;
        memset(t2_t1_map, 0, T2_T1_HASH_SIZE * sizeof(T2T1Map));
    } else {
        if (evicted_p0_slot >= 0 && evicted_p0_slot < (int)P0_SLOTS)
            t2_clear_bucket((u32)evicted_p0_slot);
        else if (evicted_p0_slot < 0)
            t2_clear_bucket(4U); // private p1 flush
    }
    pthread_mutex_unlock(&t2_meta_lock);
}

void t2_eager_promote_now(u32 t1_idx) {
    if (!t2_is_enabled || !t2_is_eager)
        return;
    T2LoopDesc desc;
    bool found = false;
    pthread_mutex_lock(&t2_meta_lock);
    T2LoopDesc *ld = t2_t1_map_find(t1_idx);
    if (ld && ld->state == T2_IDLE) {
        ld->state = T2_COMPILING;
        ld->attempts++;
        desc = *ld;
        found = true;
    }
    pthread_mutex_unlock(&t2_meta_lock);
    if (!found)
        return;
    bool ok = t2_compile_and_publish(&desc);
    pthread_mutex_lock(&t2_meta_lock);
    ld = t2_t1_map_find(t1_idx);
    if (ld && ld->epoch == desc.epoch && ld->state == T2_COMPILING)
        ld->state = ok ? T2_PROMOTED : T2_REJECTED;
    pthread_mutex_unlock(&t2_meta_lock);
}
