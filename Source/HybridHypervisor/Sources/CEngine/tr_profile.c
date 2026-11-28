// HVDBT Full-Stack Profiler implementation:
// - Per-vCPU lock-free sample ring (zero contention at 1-4 kHz host hv_vcpus_exit() sampling)
// - Host-side BlockEntry + PcMap reverse lookup (mirrors Monitor/monitor.c:600-707 for p0 ring, p1, T2)
// - Guest CR3 4-level page-table walk + Zydis x86-64 disassembly side-by-side with emitted ARM64 words
// - Exact Tier-2 rejection diagnosis matching tier2.c admission rules
// - Per-workload EL0/EL1/HVF delta time slicing + FlameGraph folded stack + JSON summary output
#include "tr_profile.h"

#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Zydis.h"

#define PROF_RING_CAP        65536U // power of two per vCPU (~2.5 MiB/vCPU; 65s @ 1 kHz/vCPU)
#define PROF_RING_MASK       (PROF_RING_CAP - 1U)
#define PROF_MAX_MON_SYMS    2048U
#define PROF_HASH_BITS       16U
#define PROF_HASH_SIZE       (1U << PROF_HASH_BITS)
#define PROF_TOP_BLOCKS      50U
#define PROF_MAX_BLOCK_INSNS 256U
#define PROF_T2_MAX_INSNS    96U
#define PROF_TICK_HZ         24000000.0

typedef struct {
    uint64_t va;
    char name[64];
} MonSym;

typedef struct {
    TrProfSample ring[PROF_RING_CAP];
    uint64_t head;
    uint64_t tail;
    uint64_t dropped;
} VcpuProfBuf;

typedef struct {
    uint64_t key;        // block arm_off (word) | (is_t2 << 63)
    uint64_t count;
    uint64_t rip;
    uint64_t cr3;
    uint32_t arm_off;
    uint32_t arm_words;
    uint32_t map_off;
    uint32_t map_count;
    uint32_t tbl_idx;
    uint16_t cpl;
    uint8_t is_t2;
    uint8_t priv_part;
    uint8_t used;
    uint32_t insn_hits[PROF_MAX_BLOCK_INSNS];
} BlockAgg;

extern size_t tr_profile_read_block_x86(uint32_t tbl_idx, bool priv_part, bool is_t2,
                                        uint64_t rip, uint8_t *dst, size_t max_len) __attribute__((weak));

typedef struct {
    uint64_t va;
    uint64_t count;
    char label[80];
} El1Agg;

static uint8_t *g_priv;
static uint8_t *g_ram;
static uint64_t g_ram_size;
static uint64_t g_ram_below_4g;
static EngineCtx *g_ctx0;
static bool g_enabled;
static uint32_t g_hz = 1000U;
static char g_prefix[512];
static VcpuProfBuf *g_vbufs; // MAX_VCPUS entries allocated only when enabled
static TrProfVcpuSnap g_base_snaps[MAX_VCPUS];
static uint32_t g_base_nvcpus;
static MonSym g_mon_syms[PROF_MAX_MON_SYMS];
static uint32_t g_mon_sym_cnt;
static pthread_mutex_t g_prof_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t g_sigusr1_pending;

static void prof_sigusr1_handler(int sig) {
    (void)sig;
    g_sigusr1_pending = 1;
}

void tr_profile_install_sigusr1(void) {
    signal(SIGUSR1, prof_sigusr1_handler);
}

bool tr_profile_take_sigusr1(void) {
    if (!g_sigusr1_pending)
        return false;
    g_sigusr1_pending = 0;
    return true;
}

static const char *g_hc_names[32] = {
    [HC_NONE] = "HC_NONE",
    [HC_TRANSLATE] = "HC_TRANSLATE",
    [HC_SMC] = "HC_SMC",
    [HC_VIEW_INVALIDATE_VA] = "HC_VIEW_INVALIDATE_VA",
    [HC_STOP] = "HC_STOP",
    [HC_SYSCALL] = "HC_SYSCALL",
    [HC_MMIO] = "HC_MMIO",
    [HC_PORT] = "HC_PORT",
    [HC_RDMSR] = "HC_RDMSR",
    [HC_WRMSR] = "HC_WRMSR",
    [HC_CPUID] = "HC_CPUID",
    [HC_HLT] = "HC_HLT",
    [HC_IRQ] = "HC_IRQ",
    [HC_APIC_EOI] = "HC_APIC_EOI",
    [HC_APIC_ICR] = "HC_APIC_ICR",
    [HC_BOUNDARY] = "HC_BOUNDARY",
    [HC_VIEW_DROP] = "HC_VIEW_DROP",
    [HC_DEBUG] = "HC_DEBUG",
    [HC_NOTE] = "HC_NOTE",
    [HC_FALLBACK] = "HC_FALLBACK",
    [HC_YIELD] = "HC_YIELD",
    [HC_PARK] = "HC_PARK",
    [HC_VIEW_FORGET] = "HC_VIEW_FORGET",
    [HC_CALL_TAIL] = "HC_CALL_TAIL",
    [HC_CODE_FETCH_RETRY] = "HC_CODE_FETCH_RETRY",
    [HC_APIC_TPR] = "HC_APIC_TPR",
    [HC_VMPORT] = "HC_VMPORT",
};

const char *tr_profile_t2_diag_str(TrProfT2Diag d) {
    switch (d) {
        case TR_PROF_T2_PROMOTED:              return "PROMOTED_T2";
        case TR_PROF_T2_ELIGIBLE_NOT_PROMOTED: return "ELIGIBLE_NOT_PROMOTED";
        case TR_PROF_T2_NO_BACKEDGE:           return "NO_BACKEDGE";
        case TR_PROF_T2_HAS_CALL_RET:          return "HAS_CALL_RET";
        case TR_PROF_T2_FORBIDDEN_OP:          return "FORBIDDEN_OP";
        case TR_PROF_T2_TOO_LARGE:             return "TOO_LARGE(>96)";
        case TR_PROF_T2_CROSS_PAGE:            return "CROSS_PAGE";
        case TR_PROF_T2_NESTED_LOOP:           return "NESTED_LOOP";
        case TR_PROF_T2_UNREADABLE_GUEST:      return "UNREADABLE_GUEST";
    }
    return "UNKNOWN";
}

void tr_profile_init_mem(uint8_t *priv, uint8_t *ram, uint64_t ram_size, uint64_t ram_below_4g,
                         EngineCtx *ctx0, const char *explicit_prefix) {
    pthread_mutex_lock(&g_prof_lock);
    g_enabled = false;
    g_prefix[0] = '\0';
    g_base_nvcpus = 0;
    memset(g_base_snaps, 0, sizeof(g_base_snaps));
    g_priv = priv;
    g_ram = ram;
    g_ram_size = ram_size;
    g_ram_below_4g = ram_below_4g;
    g_ctx0 = ctx0;

    const char *env_path = explicit_prefix ? explicit_prefix : getenv("LC_HVDBT_PROFILE");
    if (!env_path || env_path[0] == '\0') {
        pthread_mutex_unlock(&g_prof_lock);
        return;
    }
    snprintf(g_prefix, sizeof(g_prefix), "%s", env_path);

    const char *hz_env = getenv("LC_HVDBT_PROFILE_HZ");
    uint32_t hz = 1000U;
    if (hz_env && hz_env[0] != '\0') {
        long v = strtol(hz_env, NULL, 10);
        if (v < 100) v = 100;
        if (v > 4000) v = 4000;
        hz = (uint32_t)v;
    }
    g_hz = hz;

    if (!g_vbufs) {
        g_vbufs = (VcpuProfBuf *)calloc(MAX_VCPUS, sizeof(VcpuProfBuf));
    } else {
        memset(g_vbufs, 0, MAX_VCPUS * sizeof(VcpuProfBuf));
    }
    g_enabled = (g_vbufs != NULL);
    pthread_mutex_unlock(&g_prof_lock);
}

void tr_profile_shutdown(void) {
    pthread_mutex_lock(&g_prof_lock);
    g_enabled = false;
    free(g_vbufs);
    g_vbufs = NULL;
    g_mon_sym_cnt = 0;
    pthread_mutex_unlock(&g_prof_lock);
}

bool tr_profile_enabled(void) { return g_enabled; }
uint32_t tr_profile_hz(void) { return g_hz; }
const char *tr_profile_prefix(void) { return g_prefix; }

static int cmp_mon_sym(const void *a, const void *b) {
    const MonSym *sa = (const MonSym *)a;
    const MonSym *sb = (const MonSym *)b;
    return (sa->va > sb->va) - (sa->va < sb->va);
}

void tr_profile_register_mon_sym(uint64_t va, const char *name) {
    if (!name || name[0] == '\0')
        return;
    pthread_mutex_lock(&g_prof_lock);
    if (g_mon_sym_cnt < PROF_MAX_MON_SYMS) {
        MonSym *m = &g_mon_syms[g_mon_sym_cnt++];
        m->va = va;
        snprintf(m->name, sizeof(m->name), "%s", name);
        qsort(g_mon_syms, g_mon_sym_cnt, sizeof(MonSym), cmp_mon_sym);
    }
    pthread_mutex_unlock(&g_prof_lock);
}

void tr_profile_record_sample(const TrProfSample *s) {
    if (!g_enabled || !g_vbufs || !s || s->cpu >= MAX_VCPUS)
        return;
    VcpuProfBuf *vb = &g_vbufs[s->cpu];
    uint64_t h = __atomic_load_n(&vb->head, __ATOMIC_RELAXED);
    uint64_t t = __atomic_load_n(&vb->tail, __ATOMIC_ACQUIRE);
    if (h - t >= PROF_RING_CAP) {
        __atomic_fetch_add(&vb->dropped, 1U, __ATOMIC_RELAXED);
        return;
    }
    vb->ring[h & PROF_RING_MASK] = *s;
    __atomic_store_n(&vb->head, h + 1U, __ATOMIC_RELEASE);
}

void tr_profile_reset(const TrProfVcpuSnap *baseline_snaps, uint32_t nvcpus) {
    pthread_mutex_lock(&g_prof_lock);
    if (g_vbufs) {
        for (uint32_t i = 0; i < MAX_VCPUS; i++) {
            uint64_t h = __atomic_load_n(&g_vbufs[i].head, __ATOMIC_ACQUIRE);
            __atomic_store_n(&g_vbufs[i].tail, h, __ATOMIC_RELEASE);
            __atomic_store_n(&g_vbufs[i].dropped, 0U, __ATOMIC_RELAXED);
        }
    }
    memset(g_base_snaps, 0, sizeof(g_base_snaps));
    g_base_nvcpus = 0;
    if (baseline_snaps && nvcpus > 0) {
        uint32_t n = nvcpus < MAX_VCPUS ? nvcpus : MAX_VCPUS;
        memcpy(g_base_snaps, baseline_snaps, (size_t)n * sizeof(TrProfVcpuSnap));
        g_base_nvcpus = n;
    }
    pthread_mutex_unlock(&g_prof_lock);
}

// ---- Host-side RAM translation & guest x86 fetch (read-only) ----
static inline const uint8_t *prof_ram_ptr(uint64_t gpa) {
    if (!g_ram)
        return NULL;
    if (g_ram_below_4g != 0) {
        if (gpa >= g_ram_below_4g && gpa < (1ULL << 32))
            return NULL; // PCI hole
        if (gpa >= (1ULL << 32)) {
            uint64_t off = (gpa - (1ULL << 32)) + g_ram_below_4g;
            uint64_t host_bytes = g_ram_size - ((1ULL << 32) - g_ram_below_4g);
            if (off >= host_bytes)
                return NULL;
            return g_ram + off;
        }
    }
    if (gpa >= g_ram_size)
        return NULL;
    return g_ram + gpa;
}

static bool prof_walk_x86_va(uint64_t cr3, uint64_t va, uint64_t *out_gpa) {
    uint64_t tbl = cr3 & 0x000FFFFFFFFFF000ULL;
    if (tbl == 0)
        return false;
    for (int lvl = 3; lvl >= 0; lvl--) {
        uint64_t idx = (va >> (12 + lvl * 9)) & 0x1FFULL;
        const uint8_t *p = prof_ram_ptr(tbl + idx * 8ULL);
        if (!p)
            return false;
        uint64_t pte = 0;
        memcpy(&pte, p, sizeof(pte));
        if ((pte & 1ULL) == 0)
            return false;
        if (lvl > 0 && (pte & 0x80ULL) != 0) { // 1 GiB or 2 MiB large page
            uint64_t mask = (1ULL << (12 + lvl * 9)) - 1ULL;
            *out_gpa = (pte & 0x000FFFFFFFFFF000ULL & ~mask) | (va & mask);
            return true;
        }
        tbl = pte & 0x000FFFFFFFFFF000ULL;
    }
    *out_gpa = tbl | (va & 0xFFFULL);
    return true;
}

static size_t prof_read_guest(uint64_t cr3, uint64_t va, uint8_t *dst, size_t len) {
    size_t done = 0;
    while (done < len) {
        uint64_t cur = va + done;
        uint64_t gpa = 0;
        if (!prof_walk_x86_va(cr3, cur, &gpa))
            break;
        size_t chunk = (size_t)(4096ULL - (cur & 0xFFFULL));
        if (chunk > len - done)
            chunk = len - done;
        const uint8_t *src = prof_ram_ptr(gpa);
        if (!src)
            break;
        memcpy(dst + done, src, chunk);
        done += chunk;
    }
    return done;
}

// ---- Host-side BlockEntry + PcMap reverse lookup (exact mirror of Monitor/monitor.c:600-707) ----
static const BlockEntry *prof_find_block(uint64_t pc, bool *out_is_t2, uint32_t *out_tbl_idx, bool *out_priv_part) {
    *out_is_t2 = false;
    *out_tbl_idx = 0;
    *out_priv_part = false;
    if (!g_priv || !g_ctx0)
        return NULL;
    const uint64_t cc = PRIV_VA + OFF_CC;
    if ((pc < cc || pc >= cc + CC_SIZE) &&
        (pc < PRIV_VA + OFF_CC2 || pc >= PRIV_VA + OFF_CC2 + CC_SIZE))
        return NULL;
    uint32_t word = (uint32_t)((pc - cc) >> 2);

    // 1. Tier-2 mid-cache islands
    if ((word >= CC_T2_WORD0 && word < CC_T2_WORD0 + CC_T2_WORDS) ||
        (word >= CC_T2_WORD1 && word < CC_T2_WORD1 + CC_T2_WORDS)) {
        uint32_t part = word >= CC_PART2_WORD ? 1U : 0U;
        const BlockEntry *t2 = (const BlockEntry *)(g_priv + OFF_BLOCKS_T2);
        uint64_t first = (uint64_t)part * T2_BLOCKS_MAX;
        uint64_t cnt = __atomic_load_n(&g_ctx0->t2_block_count[part], __ATOMIC_ACQUIRE);
        if (cnt > T2_BLOCKS_MAX) cnt = T2_BLOCKS_MAX;
        uint64_t lo = first, hi = first + cnt;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2U;
            if (t2[mid].arm_off <= word) lo = mid + 1U;
            else hi = mid;
        }
        if (lo > first) {
            const BlockEntry *b = &t2[lo - 1U];
            if (word >= b->arm_off && word < b->arm_off + b->arm_words) {
                *out_is_t2 = true;
                *out_tbl_idx = (uint32_t)((lo - 1U) - first);
                *out_priv_part = (part != 0U);
                return b;
            }
        }
    }

    // 2. Tier-1 part 0 (4-slot ring or linear) and part 1 (private)
    const BlockEntry *t = (const BlockEntry *)(g_priv + OFF_BLOCKS);
    bool priv_part = word >= CC_PART2_WORD;
    uint64_t first = 0, count = 0;
    if (priv_part) {
        first = (OFF_BLOCKS2 - OFF_BLOCKS) / sizeof(BlockEntry);
        count = __atomic_load_n(&g_ctx0->block_count_private, __ATOMIC_ACQUIRE);
        if (count > BLOCKS_MAX_PRIV) count = BLOCKS_MAX_PRIV;
    } else if (__atomic_load_n(&g_ctx0->p0_ring_active, __ATOMIC_ACQUIRE)) {
        int slot = engine_p0_slot_of_word(word);
        if (slot < 0) return NULL;
        first = (uint64_t)slot * P0_BLOCKS_PER_SLOT;
        count = __atomic_load_n(&g_ctx0->p0_slot_block_count[slot], __ATOMIC_ACQUIRE);
        if (count > P0_BLOCKS_PER_SLOT) count = P0_BLOCKS_PER_SLOT;
    } else {
        first = 0;
        count = __atomic_load_n(&g_ctx0->block_count, __ATOMIC_ACQUIRE);
        if (count > BLOCKS_MAX) count = BLOCKS_MAX;
    }
    uint64_t lo = first, hi = first + count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2U;
        if (t[mid].arm_off <= word) lo = mid + 1U;
        else hi = mid;
    }
    if (lo == first)
        return NULL;
    const BlockEntry *b = &t[lo - 1U];
    if (word >= b->arm_off && word < b->arm_off + b->arm_words) {
        *out_priv_part = priv_part;
        *out_tbl_idx = priv_part ? (uint32_t)((lo - 1U) - ((OFF_BLOCKS2 - OFF_BLOCKS) / sizeof(BlockEntry))) : (uint32_t)(lo - 1U);
        return b;
    }
    return NULL;
}

static const PcMap *prof_pcmap_for(const BlockEntry *b, bool is_t2) {
    if (!g_priv || !b || b->map_count == 0)
        return NULL;
    // Mirror Monitor/monitor.c:696: both part 0 and part 1 map_off are indexed from OFF_PCMAP!
    uint64_t base_off = is_t2 ? OFF_PCMAP_T2 : OFF_PCMAP;
    uint64_t end_off = base_off + ((uint64_t)b->map_off + (uint64_t)b->map_count) * sizeof(PcMap);
    if (end_off > PRIV_SIZE)
        return NULL;
    return (const PcMap *)(g_priv + base_off) + b->map_off;
}

static const uint32_t *prof_cc_words(uint32_t arm_off) {
    if (!g_priv)
        return NULL;
    return (const uint32_t *)(g_priv + OFF_CC) + arm_off;
}

static const char *prof_stub_name(uint64_t pc) {
    uint64_t bases[4] = {
        PRIV_VA + OFF_STUBS,
        PRIV_VA + OFF_STUBS_HI,
        PRIV_VA + OFF_STUBS2,
        PRIV_VA + OFF_STUBS2_HI,
    };
    for (int i = 0; i < 4; i++) {
        if (pc >= bases[i] && pc < bases[i] + 0x2000ULL) {
            uint64_t off = pc - bases[i];
            if (off < 0x80) return "_stub_lookup";
            if (off < 0x100) return "_stub_ret_miss";
            if (off < 0x180) return "_stub_pushf";
            if (off < 0x200) return "_stub_sahf";
            if (off < 0x280) return "_stub_rep_movs";
            if (off < 0x300) return "_stub_rep_stos";
            if (off < 0x380) return "_stub_rep_page_movs";
            if (off < 0x400) return "_stub_rep_page_stos";
            if (off < 0x5F8) return "_stub_fex_enter";
            if (off < 0x690) return "_stub_fex_exit";
            if (off < 0x800) return "_stub_fex_exit_native";
            if (off < 0x830) return "_stub_fex_hash_probe";
            if (off < 0x860) return "_stub_fex_direct_exit";
            if (off < 0x900) return "_stub_lookup_l1";
            if (off < 0x930) return "_stub_lookup_l1_fast";
            if (off < 0x960) return "_stub_ret_miss_l1";
            return "_stub_other";
        }
    }
    return NULL;
}

static void prof_symbolize_el1(uint64_t pc, char *out, size_t cap) {
    if (g_mon_sym_cnt > 0 && pc >= PRIV_VA && pc < PRIV_VA + OFF_MON_DATA) {
        uint32_t lo = 0, hi = g_mon_sym_cnt;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2U;
            if (g_mon_syms[mid].va <= pc) lo = mid + 1U;
            else hi = mid;
        }
        if (lo > 0) {
            const MonSym *m = &g_mon_syms[lo - 1U];
            snprintf(out, cap, "%s+0x%" PRIx64, m->name, pc - m->va);
            return;
        }
    }
    snprintf(out, cap, "mon_0x%" PRIx64, pc >= PRIV_VA ? pc - PRIV_VA : pc);
}

// ---- Concise ARM64 instruction decoder for hot-block side-by-side dump ----
void tr_profile_fmt_arm64(uint32_t w, char *buf, size_t cap) {
    if (w == 0xD503201Fu) { snprintf(buf, cap, "nop"); return; }
    if (w == 0xD5033FDFu) { snprintf(buf, cap, "isb"); return; }
    if ((w & 0xFFFFFC1Fu) == 0xD65F0000u) { snprintf(buf, cap, "ret x%u", (w >> 5) & 31u); return; }
    if ((w & 0xFFFFFC1Fu) == 0xD61F0000u) { snprintf(buf, cap, "br x%u", (w >> 5) & 31u); return; }
    if ((w & 0xFFFFFC1Fu) == 0xD63F0000u) { snprintf(buf, cap, "blr x%u", (w >> 5) & 31u); return; }
    if ((w & 0xFC000000u) == 0x14000000u) {
        int32_t imm = (int32_t)(w << 6) >> 6;
        snprintf(buf, cap, "b %+d", imm * 4);
        return;
    }
    if ((w & 0xFC000000u) == 0x94000000u) {
        int32_t imm = (int32_t)(w << 6) >> 6;
        snprintf(buf, cap, "bl %+d", imm * 4);
        return;
    }
    if ((w & 0xFF000010u) == 0x54000000u) {
        static const char *cc[16] = {"eq","ne","cs","cc","mi","pl","vs","vc","hi","ls","ge","lt","gt","le","al","nv"};
        int32_t imm = ((int32_t)(w << 8) >> 13) * 4;
        snprintf(buf, cap, "b.%s %+d", cc[w & 15u], imm);
        return;
    }
    if ((w & 0x7E000000u) == 0x34000000u) {
        snprintf(buf, cap, "%s %c%u", (w & (1u << 24)) ? "cbnz" : "cbz", (w >> 31) ? 'x' : 'w', w & 31u);
        return;
    }
    if ((w & 0xFFE0001Fu) == 0xD4000001u) {
        snprintf(buf, cap, "svc #0x%x", (w >> 5) & 0xFFFFu);
        return;
    }
    if ((w & 0x7F800000u) == 0x52800000u) {
        const char *op = ((w >> 29) & 3u) == 0 ? "movn" : ((w >> 29) & 3u) == 2 ? "movz" : "movk";
        snprintf(buf, cap, "%s %c%u, #0x%x", op, (w >> 31) ? 'x' : 'w', w & 31u, (w >> 5) & 0xFFFFu);
        return;
    }
    if ((w & 0x1F000000u) == 0x11000000u) {
        snprintf(buf, cap, "%s%s %c%u, %c%u, #0x%x", (w & (1u << 30)) ? "sub" : "add",
                 (w & (1u << 29)) ? "s" : "", (w >> 31) ? 'x' : 'w', w & 31u,
                 (w >> 31) ? 'x' : 'w', (w >> 5) & 31u, (w >> 10) & 0xFFFu);
        return;
    }
    if ((w & 0x1F200000u) == 0x0B000000u) {
        snprintf(buf, cap, "%s%s %c%u, %c%u, %c%u", (w & (1u << 30)) ? "sub" : "add",
                 (w & (1u << 29)) ? "s" : "", (w >> 31) ? 'x' : 'w', w & 31u,
                 (w >> 31) ? 'x' : 'w', (w >> 5) & 31u, (w >> 31) ? 'x' : 'w', (w >> 16) & 31u);
        return;
    }
    if ((w & 0x1F000000u) == 0x0A000000u) {
        static const char *lop[4] = {"and","orr","eor","ands"};
        snprintf(buf, cap, "%s %c%u, %c%u, %c%u", lop[(w >> 29) & 3u],
                 (w >> 31) ? 'x' : 'w', w & 31u, (w >> 31) ? 'x' : 'w', (w >> 5) & 31u,
                 (w >> 31) ? 'x' : 'w', (w >> 16) & 31u);
        return;
    }
    if ((w & 0x3B000000u) == 0x39000000u) {
        bool simd = (w & (1u << 26)) != 0;
        bool load = (w & (1u << 22)) != 0;
        snprintf(buf, cap, "%s %c%u, [x%u]", load ? "ldr" : "str", simd ? 'q' : ((w >> 30) == 3 ? 'x' : 'w'),
                 w & 31u, (w >> 5) & 31u);
        return;
    }
    if ((w & 0x3A000000u) == 0x28000000u) {
        bool simd = (w & (1u << 26)) != 0;
        bool load = (w & (1u << 22)) != 0;
        snprintf(buf, cap, "%s %c%u, %c%u, [x%u]", load ? "ldp" : "stp", simd ? 'q' : 'x',
                 w & 31u, simd ? 'q' : 'x', (w >> 10) & 31u, (w >> 5) & 31u);
        return;
    }
    if ((w & 0x9F000000u) == 0x10000000u) {
        snprintf(buf, cap, "%s x%u", (w >> 31) ? "adrp" : "adr", w & 31u);
        return;
    }
    if ((w & 0x1E000000u) == 0x0E000000u || (w & 0x1E000000u) == 0x1E000000u) {
        snprintf(buf, cap, "simd/fp (0x%08x)", w);
        return;
    }
    snprintf(buf, cap, ".word 0x%08x", w);
}

// ---- Exact Tier-2 admission classifier matching tier2.c:205-222 & 605-715 ----
static TrProfT2Diag prof_diagnose_t2(const BlockAgg *ba, const PcMap *pcm, const uint8_t *x86_buf, size_t x86_len) {
    if (ba->is_t2)
        return TR_PROF_T2_PROMOTED;
    if (x86_len == 0)
        return TR_PROF_T2_UNREADABLE_GUEST;

    ZydisDecoder dec;
    ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    uint64_t blk_start = ba->rip;
    uint64_t page = blk_start & ~0xFFFULL;

    bool has_backedge = false;
    bool has_call_ret = false;
    bool has_forbidden = false;
    bool cross_page = false;
    bool nested_loop = false;
    uint64_t loop_head = ~0ULL;

    uint32_t n = ba->map_count < PROF_MAX_BLOCK_INSNS ? ba->map_count : PROF_MAX_BLOCK_INSNS;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t roff = pcm[i].rip_off;
        if (roff >= x86_len)
            break;
        uint64_t cur_rip = blk_start + roff;
        if ((cur_rip & ~0xFFFULL) != page)
            cross_page = true;

        ZydisDecodedInstruction zi;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec, x86_buf + roff, x86_len - roff, &zi, ops)))
            continue;
        if (zi.attributes & ZYDIS_ATTRIB_HAS_LOCK)
            has_forbidden = true;

        switch (zi.mnemonic) {
            case ZYDIS_MNEMONIC_CALL:
            case ZYDIS_MNEMONIC_RET:
                has_call_ret = true;
                break;
            case ZYDIS_MNEMONIC_JMP:
                if (zi.operand_count_visible > 0 && ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                    has_call_ret = true;
                } else if (zi.operand_count_visible > 0) {
                    uint64_t tgt = cur_rip + zi.length + (uint64_t)ops[0].imm.value.s;
                    if (tgt >= blk_start && tgt <= cur_rip) {
                        has_backedge = true;
                        if (loop_head != ~0ULL && tgt != loop_head) nested_loop = true;
                        if (tgt < loop_head) loop_head = tgt;
                    }
                }
                break;
            case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JCXZ:
            case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JKNZD: case ZYDIS_MNEMONIC_JKZD:
            case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNB:
            case ZYDIS_MNEMONIC_JNBE: case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_JNLE:
            case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_JNP: case ZYDIS_MNEMONIC_JNS:
            case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_JO: case ZYDIS_MNEMONIC_JP:
            case ZYDIS_MNEMONIC_JRCXZ: case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JZ:
            case ZYDIS_MNEMONIC_LOOP: case ZYDIS_MNEMONIC_LOOPE: case ZYDIS_MNEMONIC_LOOPNE:
                if (zi.operand_count_visible > 0 && ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                    uint64_t tgt = cur_rip + zi.length + (uint64_t)ops[0].imm.value.s;
                    if (tgt >= blk_start && tgt <= cur_rip) {
                        has_backedge = true;
                        if (loop_head != ~0ULL && tgt != loop_head) nested_loop = true;
                        if (tgt < loop_head) loop_head = tgt;
                    }
                }
                break;
            case ZYDIS_MNEMONIC_PUSHF: case ZYDIS_MNEMONIC_PUSHFD: case ZYDIS_MNEMONIC_PUSHFQ:
            case ZYDIS_MNEMONIC_POPF: case ZYDIS_MNEMONIC_POPFD: case ZYDIS_MNEMONIC_POPFQ:
            case ZYDIS_MNEMONIC_LAHF: case ZYDIS_MNEMONIC_SAHF:
            case ZYDIS_MNEMONIC_CMPXCHG: case ZYDIS_MNEMONIC_CMPXCHG8B: case ZYDIS_MNEMONIC_CMPXCHG16B:
            case ZYDIS_MNEMONIC_XADD: case ZYDIS_MNEMONIC_ENTER: case ZYDIS_MNEMONIC_LEAVE:
            case ZYDIS_MNEMONIC_MOVSB: case ZYDIS_MNEMONIC_MOVSW: case ZYDIS_MNEMONIC_MOVSD: case ZYDIS_MNEMONIC_MOVSQ:
            case ZYDIS_MNEMONIC_STOSB: case ZYDIS_MNEMONIC_STOSW: case ZYDIS_MNEMONIC_STOSD: case ZYDIS_MNEMONIC_STOSQ:
            case ZYDIS_MNEMONIC_SYSCALL: case ZYDIS_MNEMONIC_SYSRET: case ZYDIS_MNEMONIC_IRET:
            case ZYDIS_MNEMONIC_IRETD: case ZYDIS_MNEMONIC_IRETQ: case ZYDIS_MNEMONIC_CPUID:
            case ZYDIS_MNEMONIC_RDTSC: case ZYDIS_MNEMONIC_RDTSCP: case ZYDIS_MNEMONIC_XCHG:
                if (zi.mnemonic != ZYDIS_MNEMONIC_XCHG ||
                    (zi.operand_count_visible >= 2 &&
                     (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY || ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY)))
                    has_forbidden = true;
                break;
            default:
                if (zi.meta.isa_set == ZYDIS_ISA_SET_X87)
                    has_forbidden = true;
                break;
        }
    }

    if (!has_backedge) return TR_PROF_T2_NO_BACKEDGE;
    if (cross_page) return TR_PROF_T2_CROSS_PAGE;
    if (has_call_ret) return TR_PROF_T2_HAS_CALL_RET;
    if (has_forbidden) return TR_PROF_T2_FORBIDDEN_OP;
    if (nested_loop) return TR_PROF_T2_NESTED_LOOP;
    if (n > PROF_T2_MAX_INSNS) return TR_PROF_T2_TOO_LARGE;
    return TR_PROF_T2_ELIGIBLE_NOT_PROMOTED;
}

static int cmp_block_agg_desc(const void *a, const void *b) {
    const BlockAgg *ba = *(const BlockAgg *const *)a;
    const BlockAgg *bb = *(const BlockAgg *const *)b;
    return (bb->count > ba->count) - (bb->count < ba->count);
}

static int cmp_el1_agg_desc(const void *a, const void *b) {
    const El1Agg *ea = (const El1Agg *)a;
    const El1Agg *eb = (const El1Agg *)b;
    return (eb->count > ea->count) - (eb->count < ea->count);
}

static inline uint64_t diff_u64(uint64_t cur, uint64_t base) {
    return cur >= base ? (cur - base) : cur;
}

int64_t tr_profile_dump(const char *tag, const TrProfVcpuSnap *cur_snaps, uint32_t nvcpus) {
    pthread_mutex_lock(&g_prof_lock);
    if (!g_enabled || !g_vbufs) {
        pthread_mutex_unlock(&g_prof_lock);
        return -1;
    }

    const char *t = (tag && tag[0]) ? tag : "run";
    char path_txt[640], path_fold[640], path_json[640];
    snprintf(path_txt, sizeof(path_txt), "%s.%s.hotblocks.txt", g_prefix, t);
    snprintf(path_fold, sizeof(path_fold), "%s.%s.flamegraph.folded", g_prefix, t);
    snprintf(path_json, sizeof(path_json), "%s.%s.summary.json", g_prefix, t);

    BlockAgg *ht = (BlockAgg *)calloc(PROF_HASH_SIZE, sizeof(BlockAgg));
    El1Agg *el1_tab = (El1Agg *)calloc(512, sizeof(El1Agg));
    if (!ht || !el1_tab) {
        free(ht);
        free(el1_tab);
        pthread_mutex_unlock(&g_prof_lock);
        return -1;
    }
    uint32_t el1_cnt = 0;

    uint64_t total_samples = 0, total_dropped = 0;
    uint64_t cnt_el0_t1 = 0, cnt_el0_t2 = 0, cnt_el0_stub = 0, cnt_el1_mon = 0, cnt_unknown = 0;
    struct { const char *name; uint64_t count; } stub_bins[16] = {
        {"_stub_lookup", 0}, {"_stub_ret_miss", 0}, {"_stub_pushf", 0}, {"_stub_sahf", 0},
        {"_stub_rep_movs", 0}, {"_stub_rep_stos", 0}, {"_stub_rep_page_movs", 0}, {"_stub_rep_page_stos", 0},
        {"_stub_fex_enter", 0}, {"_stub_fex_exit", 0}, {"_stub_fex_exit_native", 0},
        {"_stub_fex_hash_probe", 0}, {"_stub_fex_direct_exit", 0},
        {"_stub_lookup_l1", 0}, {"_stub_lookup_l1_fast", 0}, {"_stub_ret_miss_l1", 0}
    };

    for (uint32_t cpu = 0; cpu < MAX_VCPUS; cpu++) {
        VcpuProfBuf *vb = &g_vbufs[cpu];
        total_dropped += __atomic_load_n(&vb->dropped, __ATOMIC_RELAXED);
        uint64_t h = __atomic_load_n(&vb->head, __ATOMIC_ACQUIRE);
        uint64_t tail = __atomic_load_n(&vb->tail, __ATOMIC_RELAXED);
        for (uint64_t idx = tail; idx < h; idx++) {
            TrProfSample s = vb->ring[idx & PROF_RING_MASK];
            total_samples++;

            if (s.el != 0) {
                cnt_el1_mon++;
                uint64_t bucket = s.pc & ~15ULL;
                uint32_t found = el1_cnt;
                for (uint32_t k = 0; k < el1_cnt; k++) {
                    if (el1_tab[k].va == bucket) { found = k; break; }
                }
                if (found < el1_cnt) {
                    el1_tab[found].count++;
                } else if (el1_cnt < 512U) {
                    el1_tab[el1_cnt].va = bucket;
                    el1_tab[el1_cnt].count = 1U;
                    prof_symbolize_el1(bucket, el1_tab[el1_cnt].label, sizeof(el1_tab[el1_cnt].label));
                    el1_cnt++;
                }
                continue;
            }

            const char *sname = prof_stub_name(s.pc);
            if (sname) {
                cnt_el0_stub++;
                for (int k = 0; k < 16; k++) {
                    if (strcmp(stub_bins[k].name, sname) == 0) {
                        stub_bins[k].count++;
                        break;
                    }
                }
                continue;
            }

            bool is_t2 = false, priv_part = false;
            uint32_t tbl_idx = 0;
            const BlockEntry *b = prof_find_block(s.pc, &is_t2, &tbl_idx, &priv_part);
            if (!b) {
                cnt_unknown++;
                continue;
            }
            if (is_t2) cnt_el0_t2++;
            else cnt_el0_t1++;

            uint64_t key = (uint64_t)b->arm_off | ((uint64_t)is_t2 << 63);
            uint32_t slot = (uint32_t)((key * 11400714819323198485ULL) >> (64U - PROF_HASH_BITS));
            BlockAgg *ba = NULL;
            for (uint32_t probe = 0; probe < PROF_HASH_SIZE; probe++) {
                BlockAgg *cand = &ht[(slot + probe) & (PROF_HASH_SIZE - 1U)];
                if (!cand->used) {
                    cand->used = 1;
                    cand->key = key;
                    cand->rip = b->rip;
                    cand->cr3 = s.cr3;
                    cand->arm_off = b->arm_off;
                    cand->arm_words = b->arm_words;
                    cand->map_off = b->map_off;
                    cand->map_count = b->map_count;
                    cand->tbl_idx = tbl_idx;
                    cand->cpl = s.cpl;
                    cand->is_t2 = is_t2 ? 1U : 0U;
                    cand->priv_part = priv_part ? 1U : 0U;
                    ba = cand;
                    break;
                }
                if (cand->key == key) {
                    ba = cand;
                    break;
                }
            }
            if (ba) {
                ba->count++;
                const PcMap *pcm = prof_pcmap_for(b, is_t2);
                if (pcm && b->map_count > 0) {
                    uint32_t rel_w = (uint32_t)((s.pc - (PRIV_VA + OFF_CC)) >> 2) - b->arm_off;
                    uint32_t best = 0;
                    uint32_t mc = b->map_count < PROF_MAX_BLOCK_INSNS ? b->map_count : PROF_MAX_BLOCK_INSNS;
                    for (uint32_t i = 0; i < mc; i++) {
                        if (pcm[i].arm_word <= rel_w) best = i;
                        else break;
                    }
                    ba->insn_hits[best]++;
                }
            }
        }
        __atomic_store_n(&vb->tail, h, __ATOMIC_RELEASE);
    }

    BlockAgg **ranked = (BlockAgg **)malloc(PROF_HASH_SIZE * sizeof(BlockAgg *));
    uint32_t ranked_cnt = 0;
    if (ranked) {
        for (uint32_t i = 0; i < PROF_HASH_SIZE; i++) {
            if (ht[i].used && ht[i].count > 0)
                ranked[ranked_cnt++] = &ht[i];
        }
        qsort(ranked, ranked_cnt, sizeof(BlockAgg *), cmp_block_agg_desc);
    }
    if (el1_cnt > 1)
        qsort(el1_tab, el1_cnt, sizeof(El1Agg), cmp_el1_agg_desc);

    double tot_vcpu_s = 0.0, tot_el1_svc_s = 0.0, tot_el1_dabt_s = 0.0, tot_el1_irq_s = 0.0, tot_el1_hvc_s = 0.0;
    double tot_hc_s[32] = {0}, tot_hc_sum_s = 0.0, tot_tr_wait_s = 0.0, tot_mon_wait_s = 0.0;
    uint64_t tot_hc_cnt[32] = {0}, tot_svc_cnt[128] = {0};
    uint64_t tot_exits = 0, tot_fills = 0, tot_cr3 = 0, tot_syscalls = 0, tot_irets = 0, tot_smc = 0;

    if (cur_snaps && nvcpus > 0) {
        uint32_t n = nvcpus < MAX_VCPUS ? nvcpus : MAX_VCPUS;
        for (uint32_t i = 0; i < n; i++) {
            const TrProfVcpuSnap *c = &cur_snaps[i];
            const TrProfVcpuSnap *b = (i < g_base_nvcpus) ? &g_base_snaps[i] : NULL;
            tot_vcpu_s     += (double)diff_u64(c->ticks_in_vcpu,       b ? b->ticks_in_vcpu : 0) / PROF_TICK_HZ;
            tot_el1_svc_s  += (double)diff_u64(c->el1_sync_svc_ticks,  b ? b->el1_sync_svc_ticks : 0) / PROF_TICK_HZ;
            tot_el1_dabt_s += (double)diff_u64(c->el1_sync_dabt_ticks, b ? b->el1_sync_dabt_ticks : 0) / PROF_TICK_HZ;
            tot_el1_irq_s  += (double)diff_u64(c->el1_irq_ticks,       b ? b->el1_irq_ticks : 0) / PROF_TICK_HZ;
            tot_el1_hvc_s  += (double)diff_u64(c->el1_hvc_ticks,       b ? b->el1_hvc_ticks : 0) / PROF_TICK_HZ;
            tot_tr_wait_s  += (double)diff_u64(c->translator_wait_ticks, b ? b->translator_wait_ticks : 0) / PROF_TICK_HZ;
            tot_mon_wait_s += (double)diff_u64(c->mon_wait_ticks,      b ? b->mon_wait_ticks : 0) / PROF_TICK_HZ;
            tot_exits      += diff_u64(c->vm_exits,     b ? b->vm_exits : 0);
            tot_fills      += diff_u64(c->shadow_fills, b ? b->shadow_fills : 0);
            tot_cr3        += diff_u64(c->cr3_writes,   b ? b->cr3_writes : 0);
            tot_syscalls   += diff_u64(c->syscalls,     b ? b->syscalls : 0);
            tot_irets      += diff_u64(c->irets,        b ? b->irets : 0);
            tot_smc        += diff_u64(c->smc_events,   b ? b->smc_events : 0);
            for (int k = 0; k < 32; k++) {
                uint64_t dc = diff_u64(c->hc_count[k], b ? b->hc_count[k] : 0);
                double ds = (double)diff_u64(c->hc_ticks[k], b ? b->hc_ticks[k] : 0) / PROF_TICK_HZ;
                tot_hc_cnt[k] += dc;
                tot_hc_s[k] += ds;
                if (k != HC_HLT) tot_hc_sum_s += ds;
            }
            for (int k = 0; k < 128; k++)
                tot_svc_cnt[k] += diff_u64(c->svc_count[k], b ? b->svc_count[k] : 0);
        }
    }
    double tot_el1_pure_s = tot_el1_svc_s + tot_el1_dabt_s + tot_el1_irq_s;
    double tot_el0_pure_s = tot_vcpu_s > tot_el1_pure_s ? (tot_vcpu_s - tot_el1_pure_s) : 0.0;
    double tot_active_s = tot_vcpu_s + tot_hc_sum_s;

    FILE *ftxt = fopen(path_txt, "w");
    FILE *ffold = fopen(path_fold, "w");
    FILE *fjson = fopen(path_json, "w");

    double denom = total_samples > 0 ? (double)total_samples : 1.0;

    if (ftxt) {
        fprintf(ftxt, "=== HVDBT Full-Stack Profile [%s] ===\n", t);
        fprintf(ftxt, "Samples: %" PRIu64 " (dropped: %" PRIu64 ", target %u Hz)\n", total_samples, total_dropped, g_hz);
        fprintf(ftxt, "Domain Distribution (by PC sample):\n");
        fprintf(ftxt, "  EL0-Tier1   : %8" PRIu64 " (%6.2f%%)\n", cnt_el0_t1,   100.0 * (double)cnt_el0_t1 / denom);
        fprintf(ftxt, "  EL0-Tier2   : %8" PRIu64 " (%6.2f%%)\n", cnt_el0_t2,   100.0 * (double)cnt_el0_t2 / denom);
        fprintf(ftxt, "  EL0-Stubs   : %8" PRIu64 " (%6.2f%%)\n", cnt_el0_stub, 100.0 * (double)cnt_el0_stub / denom);
        fprintf(ftxt, "  EL1-Monitor : %8" PRIu64 " (%6.2f%%)\n", cnt_el1_mon,  100.0 * (double)cnt_el1_mon / denom);
        fprintf(ftxt, "  Unknown     : %8" PRIu64 " (%6.2f%%)\n", cnt_unknown,  100.0 * (double)cnt_unknown / denom);

        fprintf(ftxt, "\nEL0 Stub Breakdown:\n");
        for (int k = 0; k < 16; k++) {
            if (stub_bins[k].count > 0)
                fprintf(ftxt, "  %-22s %8" PRIu64 " (%6.2f%%)\n", stub_bins[k].name, stub_bins[k].count,
                        100.0 * (double)stub_bins[k].count / denom);
        }

        if (cur_snaps && nvcpus > 0) {
            double act_d = tot_active_s > 1e-9 ? tot_active_s : 1.0;
            fprintf(ftxt, "\nExact Time Slicing (summed across %u vCPUs, excluding HC_HLT idle):\n", nvcpus);
            fprintf(ftxt, "  EL0_Pure       : %8.3f s (%5.1f%%)\n", tot_el0_pure_s, 100.0 * tot_el0_pure_s / act_d);
            fprintf(ftxt, "  EL1_Sync_SVC   : %8.3f s (%5.1f%%)\n", tot_el1_svc_s,  100.0 * tot_el1_svc_s / act_d);
            fprintf(ftxt, "  EL1_Sync_DABT  : %8.3f s (%5.1f%%)\n", tot_el1_dabt_s, 100.0 * tot_el1_dabt_s / act_d);
            fprintf(ftxt, "  EL1_IRQ        : %8.3f s (%5.1f%%)\n", tot_el1_irq_s,  100.0 * tot_el1_irq_s / act_d);
            fprintf(ftxt, "  HVF_Host(nonHLT): %7.3f s (%5.1f%%)  [translatorWait=%.3fs monWait=%.3fs]\n",
                    tot_hc_sum_s, 100.0 * tot_hc_sum_s / act_d, tot_tr_wait_s, tot_mon_wait_s);
            fprintf(ftxt, "  (EL1_HVC_sub=%.3fs, vm_exits=%" PRIu64 ", fills=%" PRIu64 ", cr3=%" PRIu64
                          ", syscalls=%" PRIu64 ", irets=%" PRIu64 ", smc=%" PRIu64 ")\n",
                    tot_el1_hvc_s, tot_exits, tot_fills, tot_cr3, tot_syscalls, tot_irets, tot_smc);
            fprintf(ftxt, "\nTop HVF Host Calls:\n");
            for (int k = 0; k < 32; k++) {
                if (tot_hc_cnt[k] > 0 && k != HC_HLT) {
                    const char *nm = g_hc_names[k] ? g_hc_names[k] : "HC_?";
                    fprintf(ftxt, "  %-22s count=%10" PRIu64 "  total=%8.3f s  avg=%6.2f us\n",
                            nm, tot_hc_cnt[k], tot_hc_s[k], tot_hc_s[k] * 1e6 / (double)tot_hc_cnt[k]);
                }
            }
            fprintf(ftxt, "\nTop EL1 SVC Counts:\n");
            for (int k = 0; k < 128; k++) {
                if (tot_svc_cnt[k] > 0)
                    fprintf(ftxt, "  SVC_%-3d                count=%10" PRIu64 "\n", k, tot_svc_cnt[k]);
            }
        }

        fprintf(ftxt, "\nTop EL1 Monitor Hotspots:\n");
        for (uint32_t i = 0; i < el1_cnt && i < 20U; i++) {
            fprintf(ftxt, "  %6.2f%%  %8" PRIu64 "  %s\n",
                    100.0 * (double)el1_tab[i].count / denom, el1_tab[i].count, el1_tab[i].label);
        }
    }

    if (ffold) {
        for (int k = 0; k < 16; k++) {
            if (stub_bins[k].count > 0)
                fprintf(ffold, "EL0;Stub;%s %" PRIu64 "\n", stub_bins[k].name, stub_bins[k].count);
        }
        for (uint32_t i = 0; i < el1_cnt; i++) {
            fprintf(ffold, "EL1;Monitor;%s %" PRIu64 "\n", el1_tab[i].label, el1_tab[i].count);
        }
        if (cnt_unknown > 0)
            fprintf(ffold, "EL0;Unknown %" PRIu64 "\n", cnt_unknown);
        for (int k = 0; k < 32; k++) {
            if (k != HC_HLT && tot_hc_s[k] > 0.0) {
                uint64_t synth = (uint64_t)(tot_hc_s[k] * (double)g_hz);
                if (synth > 0)
                    fprintf(ffold, "Host;%s %" PRIu64 "\n", g_hc_names[k] ? g_hc_names[k] : "HC", synth);
            }
        }
    }

    ZydisDecoder zdec;
    ZydisFormatter zfmt;
    ZydisDecoderInit(&zdec, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    ZydisFormatterInit(&zfmt, ZYDIS_FORMATTER_STYLE_INTEL);

    if (fjson) {
        fprintf(fjson, "{\n  \"tag\": \"%s\",\n  \"hz\": %u,\n  \"total_samples\": %" PRIu64 ",\n  \"dropped\": %" PRIu64 ",\n",
                t, g_hz, total_samples, total_dropped);
        fprintf(fjson, "  \"domains_pct\": {\"el0_t1\": %.2f, \"el0_t2\": %.2f, \"el0_stub\": %.2f, \"el1_mon\": %.2f, \"unknown\": %.2f},\n",
                100.0 * (double)cnt_el0_t1 / denom, 100.0 * (double)cnt_el0_t2 / denom,
                100.0 * (double)cnt_el0_stub / denom, 100.0 * (double)cnt_el1_mon / denom,
                100.0 * (double)cnt_unknown / denom);
        fprintf(fjson, "  \"time_s\": {\"el0_pure\": %.4f, \"el1_svc\": %.4f, \"el1_dabt\": %.4f, \"el1_irq\": %.4f, \"hvf_host\": %.4f, \"tr_wait\": %.4f, \"mon_wait\": %.4f},\n",
                tot_el0_pure_s, tot_el1_svc_s, tot_el1_dabt_s, tot_el1_irq_s, tot_hc_sum_s, tot_tr_wait_s, tot_mon_wait_s);
        fprintf(fjson, "  \"counters\": {\"vm_exits\": %" PRIu64 ", \"shadow_fills\": %" PRIu64 ", \"cr3_writes\": %" PRIu64 ", \"syscalls\": %" PRIu64 ", \"irets\": %" PRIu64 ", \"smc\": %" PRIu64 "},\n",
                tot_exits, tot_fills, tot_cr3, tot_syscalls, tot_irets, tot_smc);
        fprintf(fjson, "  \"top_blocks\": [\n");
    }

    if (ftxt)
        fprintf(ftxt, "\n=== Top-%u Hot Blocks (x86 <-> ARM64 Annotated Disassembly) ===\n", PROF_TOP_BLOCKS);

    uint32_t top_n = ranked_cnt < PROF_TOP_BLOCKS ? ranked_cnt : PROF_TOP_BLOCKS;
    for (uint32_t bi = 0; bi < top_n; bi++) {
        const BlockAgg *ba = ranked[bi];
        BlockEntry tmp_be = {
            .rip = ba->rip, .arm_off = ba->arm_off, .arm_words = ba->arm_words,
            .map_off = ba->map_off, .map_count = ba->map_count
        };
        const PcMap *pcm = prof_pcmap_for(&tmp_be, ba->is_t2 != 0);
        const uint32_t *arm = prof_cc_words(ba->arm_off);

        uint8_t x86_buf[2048];
        size_t x86_want = 0;
        if (pcm && ba->map_count > 0) {
            uint32_t last_off = pcm[ba->map_count - 1U].rip_off;
            x86_want = (size_t)last_off + 16U;
            if (x86_want > sizeof(x86_buf)) x86_want = sizeof(x86_buf);
        }
        size_t x86_len = (x86_want > 0) ? prof_read_guest(ba->cr3, ba->rip, x86_buf, x86_want) : 0;
        if (x86_len == 0 && x86_want > 0 && tr_profile_read_block_x86) {
            x86_len = tr_profile_read_block_x86(ba->tbl_idx, ba->priv_part != 0, ba->is_t2 != 0,
                                                ba->rip, x86_buf, x86_want);
        }
        TrProfT2Diag t2d = prof_diagnose_t2(ba, pcm, x86_buf, x86_len);
        double ratio = ba->map_count > 0 ? (double)ba->arm_words / (double)ba->map_count : 0.0;
        double pct = 100.0 * (double)ba->count / denom;

        if (ftxt) {
            fprintf(ftxt, "\n#%02u  %6.2f%% (%7" PRIu64 " samples)  [%s | T2:%s]  rip=0x%" PRIx64
                          "  cpl=%u  x86_insns=%u  arm_words=%u  expansion=%.2f w/i\n",
                    bi + 1U, pct, ba->count, ba->is_t2 ? "Tier-2" : "Tier-1",
                    tr_profile_t2_diag_str(t2d), ba->rip, ba->cpl, ba->map_count, ba->arm_words, ratio);
        }
        if (fjson) {
            fprintf(fjson, "    {\"rank\": %u, \"pct\": %.2f, \"samples\": %" PRIu64
                           ", \"tier\": \"%s\", \"t2_diag\": \"%s\", \"rip\": \"0x%" PRIx64
                           "\", \"x86_insns\": %u, \"arm_words\": %u, \"words_per_insn\": %.2f}%s\n",
                    bi + 1U, pct, ba->count, ba->is_t2 ? "T2" : "T1", tr_profile_t2_diag_str(t2d),
                    ba->rip, ba->map_count, ba->arm_words, ratio, (bi + 1U < top_n) ? "," : "");
        }

        uint32_t mc = ba->map_count < PROF_MAX_BLOCK_INSNS ? ba->map_count : PROF_MAX_BLOCK_INSNS;
        for (uint32_t i = 0; i < mc && pcm; i++) {
            uint32_t roff = pcm[i].rip_off;
            uint32_t aw0 = pcm[i].arm_word;
            uint32_t aw1 = (i + 1U < mc) ? pcm[i + 1U].arm_word : ba->arm_words;
            if (aw1 < aw0) aw1 = aw0;
            uint32_t aw_cnt = aw1 - aw0;

            char x86_str[128] = "<unmapped>";
            if (roff < x86_len) {
                ZydisDecodedInstruction zi;
                ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
                if (ZYAN_SUCCESS(ZydisDecoderDecodeFull(&zdec, x86_buf + roff, x86_len - roff, &zi, ops))) {
                    ZydisFormatterFormatInstruction(&zfmt, &zi, ops, zi.operand_count_visible,
                                                    x86_str, sizeof(x86_str), ba->rip + roff, NULL);
                } else {
                    snprintf(x86_str, sizeof(x86_str), ".byte 0x%02x", x86_buf[roff]);
                }
            }

            if (ffold && ba->insn_hits[i] > 0) {
                char clean[64];
                size_t ci = 0;
                for (size_t k = 0; x86_str[k] && ci + 1 < sizeof(clean); k++) {
                    char ch = x86_str[k];
                    clean[ci++] = (ch == ';' || ch == '\n' || ch == '\r') ? '_' : ch;
                }
                clean[ci] = '\0';
                fprintf(ffold, "EL0;%s;%s;blk_0x%" PRIx64 ";0x%" PRIx64 "_%s %" PRIu32 "\n",
                        ba->is_t2 ? "Tier2" : "Tier1", tr_profile_t2_diag_str(t2d),
                        ba->rip, ba->rip + roff, clean, ba->insn_hits[i]);
            }

            if (ftxt && (bi < 15U || ba->insn_hits[i] > 0)) {
                double ipct = 100.0 * (double)ba->insn_hits[i] / denom;
                fprintf(ftxt, "    %5.2f%% (%5u)  0x%" PRIx64 ": %-36s | %2u ARM64w:",
                        ipct, ba->insn_hits[i], ba->rip + roff, x86_str, aw_cnt);
                if (arm) {
                    uint32_t show = aw_cnt < 6U ? aw_cnt : 6U;
                    for (uint32_t w = 0; w < show; w++) {
                        char abuf[48];
                        tr_profile_fmt_arm64(arm[aw0 + w], abuf, sizeof(abuf));
                        fprintf(ftxt, " %s;", abuf);
                    }
                    if (aw_cnt > show)
                        fprintf(ftxt, " ...(+%u)", aw_cnt - show);
                }
                fprintf(ftxt, "\n");
            }
        }
    }

    if (fjson)
        fprintf(fjson, "  ]\n}\n");

    if (ftxt) fclose(ftxt);
    if (ffold) fclose(ffold);
    if (fjson) fclose(fjson);
    free(ranked);
    free(ht);
    free(el1_tab);
    pthread_mutex_unlock(&g_prof_lock);
    return (int64_t)total_samples;
}
