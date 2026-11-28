// Tier-2 Hot-Loop Background Asynchronous Recompiler (Round 1 MVP).
// Discovers Top-1% hot inner loops via lockless EL1 IRQ PC sampling (EngineCtx.t2_sample_ring),
// asynchronously optimizes them on a low-priority QOS_CLASS_UTILITY pthread without blocking vCPUs,
// and atomically hot-swaps the loop header's single WORD_NOP to a 32-bit B into CC_T2 (mid-loop OSR).
//
// Strict invariants (zero Monitor/vectors.s changes, 100% Ring-0/Ring-3 precise exceptions & SMC safe):
//   1. Never holds dirty ymm8..15 across guest memop or loop latch backedge; read-only loop-invariant
//      ymm8..15 are Preheader-pinned in {v24,v25,v29,v31} (EngineCtx.ymm_hi always up-to-date).
//   2. Preheader LICM uses non-flag-setting instructions only, preserving incoming RFLAGS at loop entry.
//   3. 2x loop unrolling eliminates 50% of backedge branches, carries tr_ad_page / tr_zext8/16 /
//      clean+mem-free-dirty ymm8..15 across Iter 0 -> Iter 1, and emits exact PcMaps & dirty exit flushes.
//   4. Publication verifies epoch, !hb->dead, and raw guest byte snapshot under t2_lock (< 5 us).
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tr_internal.h"

#define T2_MAX_LOOPS_PER_BUCKET 8192U
#define T2_BUCKETS 5U // 0..3: p0 ring slots 0..3, 4: p1 private
#define T2_MAX_BODY_INSNS 64
#define T2_MAX_UNROLL_INSNS (T2_MAX_BODY_INSNS * 2)
#define T2_MAX_CODE_WORDS 8192
#define T2_MAX_PCMAP 1024
#define T2_MAX_PEND 256
#define T2_MAX_FWD 256
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

static void t2_add_pend(T2Pend *pend, int *pend_cnt, u32 *site, u64 target, int form) {
    if (*pend_cnt >= T2_MAX_PEND)
        return;
    T2Pend *p = &pend[(*pend_cnt)++];
    p->site = site;
    p->target = target;
    p->form = form;
    p->dirty_cnt = (u8)ext_ymm_cache_snapshot_dirty(p->dirty_vreg, p->dirty_xreg);
    p->kind = p->dirty_cnt ? SVC_DIRECT : SVC_JCC;
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
static u64 t2_stat_ro_ymm_loops;
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
            if (in->ext.op_type[i] == EXT_OP_MEM && in->ext.op_bytes[i] == 32)
                return !in->riprel && !in->no_base && in->no_index && !tr_compat && in->disp >= 0 && in->disp < 256;
        }
    }
    if (in->riprel) {
        if (tr_compat)
            return false;
        u64 target = in->next + (u64)in->disp;
        u32 off = (u32)(target & 0xfffULL);
        int access = in->op == OP_EXT ? 16 : in->op == OP_SSE ? 16 : (in->sz ? in->sz : 8);
        return (off & (u32)(access - 1)) == 0 || off < 256;
    }
    // Only [base + small positive disp < 256] is guaranteed to use immediate addressing without clobbering AD (x20).
    return !in->no_base && in->no_index && in->disp >= 0 && in->disp < 256;
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

    // ---- Pass 1: Loop Analysis (memop, YMM8..15 RW, LICM tr_ad_page, unroll eligibility) ----
    bool has_memop = false;
    bool all_ymm_safe = true;
    bool has_4vec_blend = false;
    bool all_preserve_ad = true;
    u64 licm_ad_page = ~0ULL;
    int licm_ad_pages = 0;
    u16 ymm_rd256 = 0, ymm_wr_any = 0;
    u16 ymm_freq[16];
    memset(ymm_freq, 0, sizeof ymm_freq);
    int backedge_cnt = 0;
    int est_words = 0;

    for (int i = 0; i < bcnt; i++) {
        const Insn *in = &body[i];
        if (in->memop || in->op == OP_PUSH || in->op == OP_POP)
            has_memop = true;
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR ||
            (in->op == OP_POPCNT && !tr_hybrid)) {
            all_ymm_safe = false;
        } else if (in->op == OP_EXT) {
            if (!ext_insn_keeps_ymm_cache(in))
                all_ymm_safe = false;
            u16 r256 = 0, wany = 0;
            ext_insn_ymm_rw(in, &r256, &wany);
            ymm_rd256 |= r256;
            ymm_wr_any |= wany;
            for (int r = 8; r < 16; r++)
                if (r256 & (1u << r))
                    ymm_freq[r]++;
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

    // Pick Preheader read-only ymm8..15 pins + optional clean RW preloads (when all loop ymm8..15 fit in 4 slots!).
    u8 ro_pin_regs[4], rw_preload_regs[4];
    int ro_pin_cnt = 0, rw_preload_cnt = 0;
    bool all_hi_fit_pool = false;
    if (tr_opt_ymm_cache && all_ymm_safe && !tr_precise_flags) {
        u16 ro_mask = (u16)(ymm_rd256 & ~ymm_wr_any & 0xFF00u);
        u16 rw_mask = (u16)(ymm_rd256 & ymm_wr_any & 0xFF00u);
        int total_hi = __builtin_popcount((unsigned)((ymm_rd256 | ymm_wr_any) & 0xFF00u));
        all_hi_fit_pool = (total_hi > 0 && total_hi <= 4);
        int max_pins = 0;
        if (ro_mask) {
            if (total_hi <= 4)
                max_pins = 4;
            else if (!has_4vec_blend)
                max_pins = 1; // leaves 3 unpinned slots free for any 3-operand AVX2/FMA instruction
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
        // In single-backedge mem-free loops where ALL ymm8..15 fit in the 4-slot pool,
        // preload RW ymm8..15 once in the Preheader so even accumulators have 0 reloads inside the loop!
        if (!has_memop && all_hi_fit_pool && backedge_cnt == 1) {
            for (int r = 8; r < 16 && ro_pin_cnt + rw_preload_cnt < 4; r++) {
                if (rw_mask & (1u << r))
                    rw_preload_regs[rw_preload_cnt++] = (u8)r;
            }
        }
    }

    // Decide 2x unrolling: single latch at bcnt-1 targeting 0, clean JCC/JMP/LOOP, bounded size.
    const Insn *latch = &body[bcnt - 1];
    bool can_unroll2 = (bcnt <= 48) && (backedge_cnt == 1) && (est_words * 2 <= 2048) &&
                       (latch->target == d->head_rip) &&
                       (latch->op == OP_JMP || latch->op == OP_JCC ||
                        (latch->op == OP_JRCXZ && latch->sub == 2 && !latch->addr16));

    // Only promote if at least one Tier-2 optimization genuinely improves this loop.
    bool dirty_ymm_boost = !has_memop && all_ymm_safe && ((ymm_wr_any & 0xFF00u) != 0);
    if (!can_unroll2 && ro_pin_cnt == 0 && licm_ad_page == ~0ULL && !dirty_ymm_boost) {
        tr_t2_clear_snapshot();
        tr_t2_mode = false;
        return false;
    }

    // ---- Pass 2: Build Unrolled IR & Explicit CFG ----
    Insn ulist[T2_MAX_UNROLL_INSNS];
    int u_tgt[T2_MAX_UNROLL_INSNS];      // >=0 internal idx, -2 backedge to loop_hdr (idx 0), -1 external
    int u_fall[T2_MAX_UNROLL_INSNS];     // >=0 internal idx, -2 backedge to 0, -1 external, -3 none
    u64 u_ext_tgt[T2_MAX_UNROLL_INSNS];
    u64 u_ext_fall[T2_MAX_UNROLL_INSNS];
    bool u_is_target[T2_MAX_UNROLL_INSNS];
    bool u_iter0_jrcxz_inv[T2_MAX_UNROLL_INSNS];
    bool u_skip_nop[T2_MAX_UNROLL_INSNS];

    int ucnt = can_unroll2 ? bcnt * 2 : bcnt;
    memset(u_is_target, 0, (size_t)ucnt * sizeof(bool));
    memset(u_iter0_jrcxz_inv, 0, (size_t)ucnt * sizeof(bool));
    memset(u_skip_nop, 0, (size_t)ucnt * sizeof(bool));

    int copies = can_unroll2 ? 2 : 1;
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

            bool is_iter0_latch = can_unroll2 && c == 0 && j == bcnt - 1;
            if (is_iter0_latch) {
                if (in->op == OP_JMP) {
                    // Eliminate Iter 0 backedge JMP completely: fall straight into Iter 1!
                    u_skip_nop[idx] = true;
                    in->op = OP_NOP;
                    in->reads = 0;
                    in->writes = 0;
                    u_fall[idx] = bcnt;
                } else if (in->op == OP_JCC) {
                    // Invert condition to exit on !cc, falling straight through into Iter 1!
                    in->cc ^= 1;
                    in->reads = cc_reads[in->cc];
                    u_tgt[idx] = -1;
                    u_ext_tgt[idx] = body[j].next; // loop exit RIP
                    u_fall[idx] = bcnt;
                } else {
                    // OP_JRCXZ sub==2 (loop): decrement rcx/ecx and cbz -> exit, falling into Iter 1!
                    u_iter0_jrcxz_inv[idx] = true;
                    u_tgt[idx] = -1;
                    u_ext_tgt[idx] = body[j].next;
                    u_fall[idx] = bcnt;
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

    // ---- Pass 4: Emit Preheader + Optimized Loop Body into Thread-Local Buffer ----
    u32 local_buf[T2_MAX_CODE_WORDS];
    PcMap local_pcm[T2_MAX_PCMAP];
    u32 pcm_cnt = 0;
    u32 *insn_arm[T2_MAX_UNROLL_INSNS];
    T2Pend pend[T2_MAX_PEND];
    int pend_cnt = 0;
    T2Fwd fwd[T2_MAX_FWD];
    int fwd_cnt = 0;
    bool overflow = false;

    tr_cc = local_buf;
    tr_ad_page = ~0ULL;
    tr_zext8 = tr_zext16 = 0;
    tr_aes_mask = 0;
    ext_ymm_cache_reset();

    // Entry PcMap at offset 0 (head_rip).
    local_pcm[pcm_cnt++] = (PcMap){0, 0};

    // Preheader (non-flag-setting instructions only: preserves NZCV/R_PF/R_AF from Tier-1 OSR entry!).
    for (int k = 0; k < ro_pin_cnt; k++)
        ext_ymm_cache_pin_ro(ro_pin_regs[k]);
    for (int k = 0; k < rw_preload_cnt; k++)
        ext_ymm_cache_preload_clean(rw_preload_regs[k]);
    if (licm_ad_page != ~0ULL) {
        movimm(AD, licm_ad_page);
        tr_ad_page = licm_ad_page;
    }
    if (dirty_ymm_boost)
        ext_ymm_set_dirty_limit(24);

    u32 *loop_hdr_arm = tr_cc;
    if (loop_hdr_arm > local_buf && pcm_cnt < T2_MAX_PCMAP)
        local_pcm[pcm_cnt++] = (PcMap){(u16)(loop_hdr_arm - local_buf), 0};

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
        // In mem-free Tier-2 loops, pure non-memop integer ops (dec/add/cmp/jcc) never touch v24..v31 and never fault:
        // keep dirty ymm8..15 live across Iter 0 -> Iter 1 and flush cleanly once at backedge / cold exit stubs!
        bool op_needs_flush = dirty_ymm_boost
                                  ? (in->memop || u_is_target[i] || ext_ymm_cache_dirty_run() >= dirty_limit ||
                                     in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR ||
                                     (in->op == OP_POPCNT && !tr_hybrid))
                                  : (in->op != OP_EXT || in->memop || u_is_target[i] || ext_ymm_cache_dirty_run() >= dirty_limit);
        if (ext_ymm_cache_has_dirty() && op_needs_flush) {
            ext_ymm_cache_flush_dirty();
        }
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR ||
            (in->op == OP_POPCNT && !tr_hybrid)) {
            ext_ymm_cache_flush_and_drop();
        }
        if (licm_ad_page == ~0ULL && !tr_ad_page_safe_op(in->op))
            tr_ad_page = ~0ULL;
        if (in->op != OP_SSE && tr_aes_mask)
            sse_aes_flush_all();

        insn_arm[i] = tr_cc;
        if (!ext_ymm_cache_has_dirty() && pcm_cnt < T2_MAX_PCMAP) {
            u16 aw = (u16)(tr_cc - local_buf);
            u16 ro = (u16)(in->rip - d->head_rip);
            if (pcm_cnt == 0 || local_pcm[pcm_cnt - 1].arm_word != aw)
                local_pcm[pcm_cnt++] = (PcMap){aw, ro};
        }

        // Peephole fusions across adjacent non-target instructions (same core fusions as Tier-1).
        bool fused = false;
        bool next_same_seq = (i + 1 < ucnt) && !u_is_target[i + 1] && !u_skip_nop[i + 1] &&
                             (u_fall[i] == i + 1) && (in->next == ulist[i + 1].rip);
        if (tr_opt_peep && next_same_seq) {
            const Insn *nx = &ulist[i + 1];
            // 1) TEST r,r / CMP r,0 + JZ/JNZ -> CBZ/CBNZ
            if (in->op == OP_ALU && nx->op == OP_JCC && (nx->cc == 4 || nx->cc == 5) && live_after[i + 1] == 0) {
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
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i + 1], 0);
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
            // 5) Adjacent MOV [base+d], r1; MOV [base+d±sz], r2 -> LDP/STP
            if (!fused && in->op == OP_MOV && nx->op == OP_MOV && !tr_compat && !tr_precise_flags &&
                !in->has_imm && !nx->has_imm && !in->rm_reg && !nx->rm_reg &&
                in->to_reg == nx->to_reg && in->sz == nx->sz && (in->sz == 4 || in->sz == 8) &&
                !in->seg && !nx->seg && !in->riprel && !nx->riprel &&
                !in->addr32 && !nx->addr32 && !in->addr16 && !nx->addr16 &&
                !in->no_base && !nx->no_base && in->no_index && nx->no_index &&
                in->base == nx->base && (tr_cpl == 3 || in->base == 4 || in->base == 5)) {
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
                // Iter 0 inverted LOOP (sub==2, !addr16): dec rcx/ecx; cbz -> exit, else fall into Iter 1.
                int sf = in->addr32 ? 0 : 1;
                add_imm(sf, 1, 0, 1, 1, 1);
                t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0);
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
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0);
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
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0);
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
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0);
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
                        t2_add_pend(pend, &pend_cnt, tr_cc, u_ext_tgt[i], 0);
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
                    svc(SVC_DIRECT);
                    t2_stub_lit(u_ext_tgt[i]);
                }
            } else if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR) {
                emit_sse(in, live_after[i]);
                tr_update_zext(in);
            } else if (in->op == OP_EXT) {
                emit_ext(in, live_after[i]);
                tr_update_zext(in);
            } else {
                emit_int(in, live_after[i]);
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
                svc(SVC_DIRECT);
                t2_stub_lit(u_ext_fall[i]);
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

    // Resolve internal forward jumps.
    for (int k = 0; k < fwd_cnt; k++)
        t2_fix(fwd[k].site, fwd[k].form, insn_arm[fwd[k].target_idx]);

    // Fallthrough after last instruction of loop (typically loop exit after Iter 1 latch JCC).
    int last_f = u_fall[ucnt - 1];
    if (!t2_is_uncond_term(ulist[ucnt - 1].op)) {
        if (last_f == -2) {
            b_rel((int)(loop_hdr_arm - tr_cc));
        } else if (last_f == -1) {
            svc(SVC_DIRECT);
            t2_stub_lit(u_ext_fall[ucnt - 1]);
        }
    }

    // Emit side-exit stubs (with exact dirty YMM8..15 flush trampoline when needed).
    for (int k = 0; k < pend_cnt; k++) {
        if ((u32)(tr_cc - local_buf) >= T2_MAX_CODE_WORDS - 32) {
            overflow = true;
            break;
        }
        u32 *stub = tr_cc;
        if (pend[k].dirty_cnt > 0) {
            ext_ymm_emit_flush_list(pend[k].dirty_cnt, pend[k].dirty_vreg, pend[k].dirty_xreg);
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

    u32 code_words = (u32)(tr_cc - local_buf);
    ext_ymm_cache_reset();
    tr_t2_mode = false;

    if (overflow || code_words == 0 || code_words >= T2_MAX_CODE_WORDS)
        return false;

    __atomic_add_fetch(&t2_stat_compiled, 1, __ATOMIC_RELAXED);
    bool ok = tr_t2_publish(d->t1_idx, d->part, d->p0_slot, d->epoch, d->head_rip,
                            d->key, d->cpl, d->compat, d->patch_word, d->frame0,
                            d->min_off, d->span_bytes, snap,
                            local_buf, code_words, local_pcm, pcm_cnt);
    if (ok) {
        __atomic_add_fetch(&t2_stat_promoted, 1, __ATOMIC_RELAXED);
        if (can_unroll2) __atomic_add_fetch(&t2_stat_unrolled, 1, __ATOMIC_RELAXED);
        if (ro_pin_cnt > 0) __atomic_add_fetch(&t2_stat_ro_ymm_loops, 1, __ATOMIC_RELAXED);
        if (licm_ad_page != ~0ULL) __atomic_add_fetch(&t2_stat_licm_ad_loops, 1, __ATOMIC_RELAXED);
        if (t2_log_enabled) {
            fprintf(stderr, "[hvdbt-t2] promoted rip=%llx part=%u slot=%u body=%d unroll=%d ro_ymm=%d licm_ad=%d words=%u\n",
                    (unsigned long long)d->head_rip, d->part, d->p0_slot, bcnt, can_unroll2 ? 2 : 1,
                    ro_pin_cnt, licm_ad_page != ~0ULL ? 1 : 0, code_words);
        }
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
    u64 ro  = __atomic_load_n(&t2_stat_ro_ymm_loops, __ATOMIC_RELAXED);
    u64 ad  = __atomic_load_n(&t2_stat_licm_ad_loops, __ATOMIC_RELAXED);
    fprintf(stderr, "[hvdbt-t2-summary] reg=%llu samples=%llu compiled=%llu promoted=%llu unrolled=%llu ro_ymm=%llu licm_ad=%llu\n",
            (unsigned long long)reg, (unsigned long long)smp, (unsigned long long)cmp,
            (unsigned long long)pro, (unsigned long long)unr, (unsigned long long)ro, (unsigned long long)ad);
}

void t2_init(const TrConfig *cfg, EngineCtx *ctx, u32 *cc_base_p) {
    mach_timebase_info(&t2_tb);
    const char *no_t2 = getenv("LC_HVDBT_NO_T2");
    const char *eager = getenv("LC_HVDBT_T2_EAGER");
    const char *thresh = getenv("LC_HVDBT_T2_THRESH");
    const char *log_e = getenv("LC_HVDBT_T2_LOG");
    const char *stats_e = getenv("LC_HVDBT_T2_STATS");

    t2_is_enabled = !(no_t2 && !strcmp(no_t2, "1")) && !cfg->precise_flags && !cfg->all_flags_live && !cfg->profile_compile_inputs;
    t2_is_eager = (eager && !strcmp(eager, "1"));
    t2_log_enabled = (log_e && !strcmp(log_e, "1"));
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
    t2_stat_unrolled = t2_stat_ro_ymm_loops = t2_stat_licm_ad_loops = 0;
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
