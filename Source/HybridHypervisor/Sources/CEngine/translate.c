// Block builder: decodes x86 into a superblock, runs flag liveness, emits AArch64 and side exits, and owns block bookkeeping.
#include <dlfcn.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "tr_internal.h"

_Thread_local u64 tr_ad_page = ~0ULL;
_Thread_local u16 tr_zext8 = 0, tr_zext16 = 0;
bool tr_opt_peep = true, tr_opt_ymm_cache = true;
static bool native_dag = true, native_l1 = true;
static bool p0_ring_enabled = false;
static bool par_translate_enabled = true;
static int no_uframe_share = 0;
_Thread_local u16 tr_aes_mask = 0;
_Thread_local u8 tr_aes_key[16];

typedef struct {
    void (*x86_decode)(u64 rip, Insn *in);
    bool (*insn_is_terminator)(const Insn *in);
    void (*emit_int)(const Insn *in, int live);
    void (*emit_sse)(const Insn *in, int live);
    void (*sse_aes_flush_all)(void);
    void (*emit_x87)(const Insn *in);
    void (*emit_ext)(const Insn *in, int live);
    void (*movimm)(int rd, u64 v);
    void (*emit_parity)(int rd);
} TrBackend;

static TrBackend tr_be = {
    .x86_decode = x86_decode,
    .insn_is_terminator = insn_is_terminator,
    .emit_int = emit_int,
    .emit_sse = emit_sse,
    .sse_aes_flush_all = sse_aes_flush_all,
    .emit_x87 = emit_x87,
    .emit_ext = emit_ext,
    .movimm = movimm,
    .emit_parity = emit_parity,
};

bool tr_reload_backend(const char *dylib_path) {
    if (!dylib_path || !dylib_path[0])
        return false;
    void *h = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "hvdbt: tr_reload_backend dlopen(%s) failed: %s\n", dylib_path, dlerror());
        return false;
    }
    TrBackend next = {
        .x86_decode = dlsym(h, "x86_decode"),
        .insn_is_terminator = dlsym(h, "insn_is_terminator"),
        .emit_int = dlsym(h, "emit_int"),
        .emit_sse = dlsym(h, "emit_sse"),
        .sse_aes_flush_all = dlsym(h, "sse_aes_flush_all"),
        .emit_x87 = dlsym(h, "emit_x87"),
        .emit_ext = dlsym(h, "emit_ext"),
        .movimm = dlsym(h, "movimm"),
        .emit_parity = dlsym(h, "emit_parity"),
    };
    if (!next.x86_decode || !next.insn_is_terminator || !next.emit_int || !next.emit_sse ||
        !next.sse_aes_flush_all || !next.emit_x87 || !next.emit_ext || !next.movimm || !next.emit_parity) {
        fprintf(stderr, "hvdbt: tr_reload_backend(%s) missing required backend symbols\n", dylib_path);
        dlclose(h);
        return false;
    }
    tr_be = next;
    fprintf(stderr, "hvdbt: hot-reloaded JIT backend from %s\n", dylib_path);
    return true;
}

#define x86_decode tr_be.x86_decode
#define insn_is_terminator tr_be.insn_is_terminator
#define emit_int tr_be.emit_int
#define emit_sse tr_be.emit_sse
#define sse_aes_flush_all tr_be.sse_aes_flush_all
#define emit_x87 tr_be.emit_x87
#define emit_ext tr_be.emit_ext
#define movimm tr_be.movimm
#define emit_parity tr_be.emit_parity

#define MAX_INSNS 256
#define MAX_PENDING (MAX_INSNS * 2)
#define MAX_FWD_FIX (MAX_INSNS * 2)
#define MAX_L1_MISS MAX_INSNS
#define MAX_DAG_BLOCKS 12
#define MAX_DAG_SPAN 1024
#define STUB_OFF_LOOKUP_L1 0x860ULL
#define STUB_OFF_LOOKUP_L1_FAST 0x900ULL
#define STUB_OFF_RET_MISS_L1 0x930ULL
#define MAP_BITS 22
#define MAP_SIZE (1u << MAP_BITS)

_Thread_local u32 *tr_cc;
bool tr_precise_flags, tr_has_tzcnt, tr_hybrid, tr_no_bulk_rep;
bool tr_fpcr_merges_scalar;
bool tr_profile_shift_helpers;
bool tr_inline_shr8;
bool tr_page_rep;
_Thread_local int tr_cpl;
_Thread_local bool tr_compat;
_Thread_local bool tr_t2_mode;
_Thread_local bool tr_par_active;
_Thread_local u32 *tr_par_block_start;
bool tr_aot_active;
_Thread_local AotCapture tr_aot_cap;
_Thread_local u32 *tr_aot_block_start;
static AotCache *aot_cache;
static AotStubTable aot_stubs;
static u64 aot_abi_hash;
static bool aot_eager_link = true;
static bool aot_atexit_registered;
u64 tr_stub_lookup, tr_stub_ret_miss, tr_stub_pushf, tr_stub_sahf;

static TrConfig cfg;
static u32 *cc_base;
static u64 cc_limit;
static BlockEntry *block_table;
static PcMap *pc_map;
static BlockEntry *t2_block_table;
static PcMap *t2_pc_map;
static T2TrapMeta *t2_trap_meta;
// The code cache has two parts, each with its share of the block table and pc maps. Shared domains go in the first: the kernel and system libraries, which every
// process needs and which live long. Blocks of one view go in the second, and most die with their process, unusable until a reset. Links join blocks of one key, so
// none crosses between the parts, and the second can reset alone without taking the first with it.
#define PCMAP_ENTRIES (PCMAP_BYTES / sizeof(PcMap))
static u32 pc_map_used[2] = {0, (OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap)};
static u32 *cc_cursor[2];
_Thread_local int cur_part;
static EngineCtx *ctx_ptr;
static void code_published(void);

// Part 0 4-slot ring-generation state (active when p0_ring_enabled == true).
static u32 *p0_cc_cursor[P0_SLOTS];
static u32 p0_block_count[P0_SLOTS];
static u32 p0_pcmap_used[P0_SLOTS];
static u64 p0_slot_gen[P0_SLOTS];
static u64 p0_gen_seq;
static u32 p0_cur_slot;
static bool p0_ring_need_evict;
static u64 p0_ring_evictions;

// Tier-2 code cache state (CC_T2 mid-cache islands at +120..+124 MiB of part 0 and part 1).
typedef struct {
    u64 rip;
    u32 view;
    u32 entry_word;
    u32 arm_words;
    u32 t1_idx;
    u8 cpl;
    u8 p0_slot;
    bool compat;
    bool dead;
} T2BlockMeta;
static pthread_mutex_t t2_pub_lock = PTHREAD_MUTEX_INITIALIZER;
static u32 *cc_t2_cursor[2];
static u32 t2_block_cnt[2];
static u32 t2_pcmap_used[2];
static u64 t2_epoch[2];
static T2BlockMeta *t2_meta[2];

typedef struct {
    u64 rip;
    u32 view;
    u32 entry_word; // arm_off
    u64 frame[2]; // ~0 when unused
    u32 next_on_frame[2]; // block index + 1
    u32 next_on_page[2]; // block index + 1: the x86 pages it was read from, for a change to their mappings
    u32 next_in_view; // block index + 1: blocks of one view, not of a shared kernel domain
    bool dead, registered;
    // For reuse. Only a block on one frame is ever reused, so the chains here are its first ones.
    bool on_frame, on_page, on_view, compat;
    u8 cpl;
    u8 p0_slot;
    bool t2_has_loop;
    u16 read_off, read_bytes; // what translating it read of its frame, a copy of which is at read_copy. 0 bytes: never reused
    u32 read_copy;
    u32 first_word; // its entry word, which a kill overwrites
    u32 t2_patch_word; // word holding WORD_NOP OSR anchor (may equal entry_word)
    u32 t2_entry_word; // nonzero if promoted into CC_T2
    u32 t2_meta_idx;   // index into t2_meta[part] + 1
} HostBlock;
static HostBlock *hblocks;
static u32 hblock_count[2];
// Only domain blocks (part 0) need this sidecar: 32 bytes * BLOCKS_MAX = 16 MiB, allocated only for the prototype.
// Bits are a superset: replacement, forgetting and view reuse may leave stale bits, never omit a published target.
static u64 (*shared_hash_views)[MAX_VIEWS / 64];
#define BLOCK_BASE(part) ((part) ? BLOCKS_MAX : 0)
#define PART_BLOCKS_MAX(part) ((part) ? BLOCKS_MAX_PRIV : BLOCKS_MAX)
#define BLOCK_TABLE_PART2 ((OFF_BLOCKS2 - OFF_BLOCKS) / sizeof(BlockEntry))
#define KEY_PART(key) (((key) & KEY_DOMAIN) ? 0 : 1)
static u32 *frame_head; // per GPA frame: block index + 1
#define PAGE_HEADS (1u << 18)
static u32 *page_head; // by x86 page number, modulo PAGE_HEADS: block index + 1
static u32 view_head[MAX_VIEWS]; // block index + 1

static inline u32 page_slot(u64 va) { return (u32)(va >> 12) & (PAGE_HEADS - 1); }

// Lookup: (view, rip) to block index + 1. Open addressing, deleted entries keep probing.
typedef struct {
    u64 rip;
    u32 view;
    u32 block; // 0 empty, ~0 deleted, otherwise index + 1
} MapEntry;
static MapEntry *block_map;

static TrStats stats;
static mach_timebase_info_data_t invalidation_timebase;

static u64 invalidation_ns(u64 ticks) {
    return (u64)((__uint128_t)ticks * invalidation_timebase.numer / invalidation_timebase.denom);
}

static pthread_mutex_t stop_reason_lock = PTHREAD_MUTEX_INITIALIZER;
static const char *stop_reasons[4096];
static u32 stop_reason_count;

// Monotonic invalidation/flush sequence counters for lock-free parallel translate commit verification.
#define PAR_GEN_BUCKETS 4096U
static u64 tr_frame_inval_gen[PAR_GEN_BUCKETS];
static u64 tr_pt_inval_gen;
static u64 tr_flush_gen;

u64 tr_cc_va(const u32 *p) {
    if (__builtin_expect(tr_par_active, 0))
        return PRIV_VA + (cur_part ? OFF_CC2 : OFF_CC) + (u64)((const u8 *)p - (const u8 *)tr_par_block_start);
    return PRIV_VA + OFF_CC + (u64)((const u8 *)p - (const u8 *)cc_base);
}
static u32 *cc_host(u64 va) { return (u32 *)((u8 *)cc_base + (va - PRIV_VA - OFF_CC)); }

static u32 map_hash(int view, u64 rip) { return (u32)((rip ^ ((u64)view << 40)) * 0x9E3779B97F4A7C15ULL >> (64 - MAP_BITS)); }

static u32 map_find(int view, u64 rip) {
    for (u32 h = map_hash(view, rip);; h = (h + 1) & (MAP_SIZE - 1)) {
        MapEntry *e = &block_map[h];
        if (!e->block)
            return 0;
        if (e->block != ~0u && e->rip == rip && e->view == (u32)view)
            return e->block;
    }
}
// Advisory lock-free probe used ONLY to avoid taking translatorLock before parallel prepare.
// Authoritative lookup always runs under translatorLock in tr_translate_fast / tr_translate_commit.
static bool map_find_hint(int view, u64 rip) {
    u32 h = map_hash(view, rip);
    for (u32 probes = 0; probes < 32; probes++, h = (h + 1) & (MAP_SIZE - 1)) {
        MapEntry *e = &block_map[h];
        u32 blk = __atomic_load_n(&e->block, __ATOMIC_RELAXED);
        if (!blk)
            return false;
        if (blk != ~0u &&
            __atomic_load_n(&e->rip, __ATOMIC_RELAXED) == rip &&
            __atomic_load_n(&e->view, __ATOMIC_RELAXED) == (u32)view)
            return true;
    }
    return true; // long chain -> let locked fast path decide
}
static void map_insert(int view, u64 rip, u32 block) {
    for (u32 h = map_hash(view, rip);; h = (h + 1) & (MAP_SIZE - 1)) {
        MapEntry *e = &block_map[h];
        if (!e->block || e->block == ~0u) {
            __atomic_store_n(&e->rip, rip, __ATOMIC_RELAXED);
            __atomic_store_n(&e->view, (u32)view, __ATOMIC_RELAXED);
            __atomic_store_n(&e->block, block, __ATOMIC_RELEASE);
            return;
        }
    }
}
static void map_remove(int view, u64 rip) {
    for (u32 h = map_hash(view, rip);; h = (h + 1) & (MAP_SIZE - 1)) {
        MapEntry *e = &block_map[h];
        if (!e->block)
            return;
        if (e->block != ~0u && e->rip == rip && e->view == (u32)view) {
            __atomic_store_n(&e->block, ~0u, __ATOMIC_RELAXED);
            return;
        }
    }
}

static u8 *ram_at(u64 gpa) { return cfg.ram + (cfg.ram_below_4g && gpa >= 1ULL << 32 ? gpa - (1ULL << 32) + cfg.ram_below_4g : gpa); }
static bool is_ram(u64 gpa) { return gpa < cfg.ram_size && (cfg.priv[OFF_PAGEINFO + (gpa >> 12)] & PI_RAM); }

// Ring-0 code in the high half is shared by every address space that reaches it through the same PML4 entry target: an x86 kernel keeps
// those entries identical, and a change to them invalidates through the page-table write path. Everything else belongs to one view.
#define KEY_DOMAIN 0x40000000
#define KEY_USER_FRAME (KEY_DOMAIN | 0x20000000)
static int view_high;
// A rip under a page directory that several address spaces share maps the same bytes in all of them, so one translation serves them all. The last page stays out,
// as a block starting there could run into the next directory.
static int shared_user_key(u64 cr3, int cpl, u64 rip) {
    bool compat = tr_compat && engine_shared_compat_address(rip);
    if ((cpl != 3 && !(compat && cpl == 0)) || !ctx_ptr->shared_dir_count ||
        (!compat && (tr_compat || rip < SHARED_USER_BASE || rip >= SHARED_USER_END - 4096)))
        return 0;
    u64 table = cr3 & 0x000FFFFFFFFFF000ULL;
    if (!is_ram(table))
        return 0;
    u64 e = *(u64 *)(ram_at(table) + ((rip >> 39) & 0x1ff) * 8);
    table = e & 0x000FFFFFFFFFF000ULL;
    if (!(e & 1) || !is_ram(table))
        return 0;
    e = *(u64 *)(ram_at(table) + ((rip >> 30) & 0x1ff) * 8);
    if (!(e & 1) || (e & 0x80))
        return 0;
    u64 directory = e & 0x000FFFFFFFFFF000ULL;
    if (compat) {
        if (!is_ram(directory))
            return 0;
        e = *(u64 *)(ram_at(directory) + ((rip >> 21) & 0x1ff) * 8);
        if (!(e & 1) || (e & 0x80))
            return 0;
        u64 page_table = e & 0x000FFFFFFFFFF000ULL;
        if (!is_ram(page_table))
            return 0;
        directory = engine_shared_compat_domain(rip, page_table);
        if (cpl == 0)
            directory |= SHARED_COMPAT_KERNEL;
    }
    u32 shared_cnt = __atomic_load_n(&ctx_ptr->shared_dir_count, __ATOMIC_ACQUIRE);
    for (u32 i = 0; i < shared_cnt; i++)
        if (ctx_ptr->shared_dirs[i] == directory)
            return compat ? (int)(0xC0000000u | i) : KEY_DOMAIN | (int)((directory >> 12) & 0x3FFFFFFF);
    return 0;
}

static int shared_user_frame_key(u64 cr3, int cpl, u64 rip) {
    if (no_uframe_share || cpl != 3 || tr_compat || (rip >> 47) || (rip & 0xfffULL) >= 4096 - 16)
        return 0;
    u64 table = cr3 & 0x000FFFFFFFFFF000ULL;
    if (!is_ram(table))
        return 0;
    u64 e = *(u64 *)(ram_at(table) + ((rip >> 39) & 0x1ff) * 8);
    table = e & 0x000FFFFFFFFFF000ULL;
    if (!(e & 1) || !(e & 4) || (e & (1ULL << 63)) || !is_ram(table))
        return 0;
    e = *(u64 *)(ram_at(table) + ((rip >> 30) & 0x1ff) * 8);
    if (!(e & 1) || !(e & 4) || (e & (1ULL << 63)))
        return 0;
    u64 gpa;
    if (e & 0x80) {
        if (e & 2)
            return 0;
        gpa = (e & 0x000FFFFFC0000000ULL) | (rip & 0x3ffff000ULL);
    } else {
        table = e & 0x000FFFFFFFFFF000ULL;
        if (!is_ram(table))
            return 0;
        e = *(u64 *)(ram_at(table) + ((rip >> 21) & 0x1ff) * 8);
        if (!(e & 1) || !(e & 4) || (e & (1ULL << 63)))
            return 0;
        if (e & 0x80) {
            if (e & 2)
                return 0;
            gpa = (e & 0x000FFFFFFFE00000ULL) | (rip & 0x1ff000ULL);
        } else {
            table = e & 0x000FFFFFFFFFF000ULL;
            if (!is_ram(table))
                return 0;
            e = *(u64 *)(ram_at(table) + ((rip >> 12) & 0x1ff) * 8);
            if (!(e & 1) || !(e & 4) || (e & (1ULL << 63)) || (e & 2))
                return 0;
            gpa = e & 0x000FFFFFFFFFF000ULL;
        }
    }
    if (!is_ram(gpa))
        return 0;
    u64 pfn = gpa >> 12;
    if (pfn >= (1ULL << 29))
        return 0;
    return KEY_USER_FRAME | (int)pfn;
}

static int block_key(int view, u64 cr3, int cpl, u64 rip) {
    if (cpl == 0 && (rip >> 47)) {
        u64 table = cr3 & 0x000FFFFFFFFFF000ULL;
        if (is_ram(table)) {
            u64 e = *(u64 *)(ram_at(table) + ((rip >> 39) & 0x1ff) * 8);
            return KEY_DOMAIN | (int)((e >> 12) & 0x3FFFFFFF);
        }
    }
    int shared = shared_user_key(cr3, cpl, rip);
    if (shared)
        return shared;
    int uframe = shared_user_frame_key(cr3, cpl, rip);
    return uframe ? uframe : view;
}

// ---- Guest memory, host's read-only view (thread-local for 16-core parallel translate) ----
static _Thread_local u64 walk_cr3;
static _Thread_local int walk_cpl;
#define X_P 1ULL
#define X_US 4ULL
#define X_PS 0x80ULL
#define X_NX (1ULL << 63)
#define X_ADDR 0x000FFFFFFFFFF000ULL

// Instruction-fetch walk. Returns false for anything that would be a fetch fault. Permission intersection as the monitor does it.
static bool fetch_walk(u64 va, u64 *gpa) {
    u64 table = walk_cr3 & X_ADDR;
    bool user = true, nx = false;
    for (int level = 3; level >= 0; level--) {
        if (!is_ram(table))
            return false;
        u64 e = *(u64 *)(ram_at(table) + ((va >> (12 + 9 * level)) & 0x1ff) * 8);
        if (!(e & X_P))
            return false;
        user = user && (e & X_US);
        nx = nx || (e & X_NX);
        if (level && level < 3 && (e & X_PS)) {
            u64 size = 1ULL << (12 + 9 * level);
            u64 base = e & X_ADDR & ~(size - 1);
            *gpa = base | (va & (size - 1));
            goto ok;
        }
        if (!level) {
            *gpa = (e & X_ADDR) | (va & 0xfff);
            goto ok;
        }
        table = e & X_ADDR;
    }
    return false;
ok:
    if (nx || (walk_cpl == 3 && !user))
        return false;
    return is_ram(*gpa);
}

static _Thread_local u64 fetch_lo, fetch_hi; // all a translation has read
static _Thread_local bool fetch_cache_valid;
static _Thread_local u64 fetch_cache_va_page, fetch_cache_gpa_page, fetch_cache_cr3;
static _Thread_local const u8 *fetch_cache_host_page;
static _Thread_local int fetch_cache_cpl;

// ---- Inert analyzed-input recurrence census ----
// One entry per direct-mapped identity, full population: this is not a sampled cache with an inflated retention budget.
// Canonical zero holes plus an exact occupancy mask describe only bytes actually returned by tr_fetch, including liveness.
// No executable pointers, cached instructions, extra guest reads, or changes to compilation decisions live here.
#define COMPILE_INPUT_SLOTS 8192u
#define COMPILE_INPUT_PAGE 4096u
typedef struct {
    u64 rip, config, frame;
    u32 key;
    u8 cpl;
    bool compat, valid;
    u64 mask[COMPILE_INPUT_PAGE / 64];
    u8 bytes[COMPILE_INPUT_PAGE];
} CompileInputEntry;
typedef struct {
    CompileInputEntry input;
    u32 reject;
    bool active;
} CompileInputCapture;
_Static_assert(sizeof(CompileInputEntry) * COMPILE_INPUT_SLOTS + sizeof(CompileInputCapture) <= 40u * 1024 * 1024,
               "compile-input census must remain bounded by 40 MiB");
static CompileInputEntry *compile_inputs_table;
static CompileInputCapture *compile_inputs_capture;

static void compile_inputs_init(void) {
    free(compile_inputs_table);
    free(compile_inputs_capture);
    compile_inputs_table = NULL;
    compile_inputs_capture = NULL;
    stats.compile_inputs_enabled = cfg.profile_compile_inputs;
    stats.compile_inputs_available = stats.compile_inputs_bytes = stats.compile_inputs_allocation_failures = 0;
    stats.compile_inputs_sampling_shift = stats.compile_inputs_config_base = 0;
    stats.compile_inputs_attempts = stats.compile_inputs_admitted = 0;
    memset(stats.compile_inputs_reject, 0, sizeof stats.compile_inputs_reject);
    stats.compile_inputs_empty = stats.compile_inputs_evictions = stats.compile_inputs_prior = stats.compile_inputs_exact = 0;
    stats.compile_inputs_coverage_mismatch = stats.compile_inputs_bytes_mismatch = 0;
    stats.compile_inputs_exact_same_key = stats.compile_inputs_exact_different_key = 0;
    stats.compile_inputs_exact_same_pfn = stats.compile_inputs_exact_different_pfn = 0;
    memset(stats.compile_inputs_exact_joint, 0, sizeof stats.compile_inputs_exact_joint);
    if (!cfg.profile_compile_inputs)
        return;
    // Schema 1. The table is cleared only here, so this tr_init is its fixed configuration epoch.
    stats.compile_inputs_config_base = 1ULL << 32 | (u64)cfg.precise_flags | (u64)cfg.has_tzcnt_lzcnt << 1 |
        (u64)cfg.hybrid << 2 | (u64)cfg.keep_kernel_tables << 3 | (u64)cfg.no_bulk_rep << 4 |
        (u64)cfg.no_return_prediction << 5 | (u64)cfg.all_flags_live << 6 |
        (u64)cfg.profile_shift_helpers << 7 | (u64)cfg.inline_shr8 << 8 | (u64)cfg.page_rep << 9;
    compile_inputs_table = calloc(COMPILE_INPUT_SLOTS, sizeof *compile_inputs_table);
    compile_inputs_capture = calloc(1, sizeof *compile_inputs_capture);
    if (!compile_inputs_table || !compile_inputs_capture) {
        free(compile_inputs_table);
        free(compile_inputs_capture);
        compile_inputs_table = NULL;
        compile_inputs_capture = NULL;
        stats.compile_inputs_allocation_failures = 1;
        return;
    }
    stats.compile_inputs_available = 1;
    stats.compile_inputs_bytes = sizeof(CompileInputEntry) * COMPILE_INPUT_SLOTS + sizeof(CompileInputCapture);
}

static void compile_inputs_begin(u64 rip, int cpl, bool compat, u32 key, u64 frame, bool single) {
    if (!cfg.profile_compile_inputs)
        return;
    stats.compile_inputs_attempts++;
    if (compile_inputs_capture)
        compile_inputs_capture->active = false;
    if (single || !compile_inputs_capture) {
        stats.compile_inputs_reject[single ? TR_CI_SINGLE : TR_CI_UNAVAILABLE]++;
        return;
    }
    memset(compile_inputs_capture, 0, sizeof *compile_inputs_capture);
    CompileInputCapture *c = compile_inputs_capture;
    c->active = true;
    c->reject = TR_CI_REJECT_KINDS;
    c->input.rip = rip;
    // vcpu_count is installed after tr_init and can grow on SMP attachment: capture the emitter-relevant bit now.
    c->input.config = stats.compile_inputs_config_base | (u64)(ctx_ptr->vcpu_count > 1) << 16;
    c->input.frame = frame;
    c->input.key = key;
    c->input.cpl = (u8)cpl;
    c->input.compat = compat;
}

static void compile_inputs_reject(u32 reason) {
    CompileInputCapture *c = compile_inputs_capture;
    if (cfg.profile_compile_inputs && c && c->active && c->reject == TR_CI_REJECT_KINDS)
        c->reject = reason;
}

static void compile_inputs_request(u64 va, int n) {
    CompileInputCapture *c = compile_inputs_capture;
    if (!c || !c->active)
        return;
    u64 page = c->input.rip & ~0xfffULL;
    if (n <= 0)
        compile_inputs_reject(TR_CI_EMPTY);
    else if (va < page || va - page >= COMPILE_INPUT_PAGE || (u64)n > COMPILE_INPUT_PAGE - (va - page))
        compile_inputs_reject(TR_CI_CROSS_PAGE);
}

static void compile_inputs_read(u64 va, const u8 *bytes, int n, u64 gpa) {
    CompileInputCapture *c = compile_inputs_capture;
    if (!c || !c->active || c->reject != TR_CI_REJECT_KINDS)
        return;
    compile_inputs_request(va, n);
    if (c->reject != TR_CI_REJECT_KINDS)
        return;
    if ((gpa & ~0xfffULL) != c->input.frame) {
        compile_inputs_reject(TR_CI_MIXED_FRAME);
        return;
    }
    u32 off = (u32)(va & 0xfff);
    for (int i = 0; i < n; i++) {
        u32 at = off + (u32)i;
        u64 bit = 1ULL << (at & 63);
        if ((c->input.mask[at >> 6] & bit) && c->input.bytes[at] != bytes[i]) {
            compile_inputs_reject(TR_CI_INCONSISTENT);
            return;
        }
        c->input.mask[at >> 6] |= bit;
        c->input.bytes[at] = bytes[i];
    }
}

static void compile_inputs_finish(void) {
    CompileInputCapture *c = compile_inputs_capture;
    if (!cfg.profile_compile_inputs || !c || !c->active)
        return;
    c->active = false;
    bool any = false;
    for (u32 i = 0; i < COMPILE_INPUT_PAGE / 64; i++)
        any |= c->input.mask[i] != 0;
    if (!any && c->reject == TR_CI_REJECT_KINDS)
        c->reject = TR_CI_EMPTY;
    if (c->reject != TR_CI_REJECT_KINDS) {
        stats.compile_inputs_reject[c->reject]++;
        return;
    }
    stats.compile_inputs_admitted++;
    CompileInputEntry *in = &c->input;
    u64 hash = (in->rip ^ in->config ^ (u64)in->cpl << 61 ^ (u64)in->compat << 60) * 0x9E3779B97F4A7C15ULL;
    CompileInputEntry *old = &compile_inputs_table[hash >> (64 - 13)];
    if (!old->valid) {
        stats.compile_inputs_empty++;
    } else if (old->rip != in->rip || old->config != in->config || old->cpl != in->cpl || old->compat != in->compat) {
        stats.compile_inputs_evictions++;
    } else {
        stats.compile_inputs_prior++;
        if (memcmp(old->mask, in->mask, sizeof in->mask)) {
            stats.compile_inputs_coverage_mismatch++;
        } else if (memcmp(old->bytes, in->bytes, sizeof in->bytes)) {
            stats.compile_inputs_bytes_mismatch++;
        } else {
            stats.compile_inputs_exact++;
            bool dk = old->key != in->key, df = old->frame != in->frame;
            if (dk) stats.compile_inputs_exact_different_key++; else stats.compile_inputs_exact_same_key++;
            if (df) stats.compile_inputs_exact_different_pfn++; else stats.compile_inputs_exact_same_pfn++;
            stats.compile_inputs_exact_joint[dk * 2 + df]++;
        }
    }
    // Keep only the most recent observation of this identity, including mismatches; no multi-version history.
    *old = *in;
    old->valid = true;
}
// ---- End inert analyzed-input recurrence census ----

static _Thread_local u64 t2_snap_base;
static _Thread_local const u8 *t2_snap_ptr;
static _Thread_local u32 t2_snap_len;

void tr_t2_set_snapshot(u64 base_va, const u8 *bytes, u32 len) {
    t2_snap_base = base_va;
    t2_snap_ptr = bytes;
    t2_snap_len = len;
}

void tr_t2_clear_snapshot(void) {
    t2_snap_ptr = NULL;
    t2_snap_len = 0;
}

int tr_fetch(u64 va, u8 *out, int n) {
    if (__builtin_expect(tr_t2_mode, 0)) {
        if (!t2_snap_ptr || va < t2_snap_base || va >= t2_snap_base + t2_snap_len)
            return 0;
        u64 avail = (t2_snap_base + t2_snap_len) - va;
        int got = (u64)n < avail ? n : (int)avail;
        memcpy(out, t2_snap_ptr + (va - t2_snap_base), (size_t)got);
        return got;
    }
    int got = 0;
    if (cfg.profile_compile_inputs)
        compile_inputs_request(va, n);
    if (va < fetch_lo)
        fetch_lo = va;
    if (va + (u64)n > fetch_hi)
        fetch_hi = va + (u64)n;
    while (got < n) {
        u64 cur = va + (u64)got;
        u64 cur_pg = cur & ~0xfffULL;
        u64 gpa;
        if (fetch_cache_valid && cur_pg == fetch_cache_va_page &&
            walk_cr3 == fetch_cache_cr3 && walk_cpl == fetch_cache_cpl) {
            gpa = fetch_cache_gpa_page | (cur & 0xfffULL);
        } else {
            if (!fetch_walk(cur, &gpa)) {
                compile_inputs_reject(TR_CI_PARTIAL);
                return got;
            }
            fetch_cache_va_page = cur_pg;
            fetch_cache_gpa_page = gpa & ~0xfffULL;
            fetch_cache_host_page = ram_at(fetch_cache_gpa_page);
            fetch_cache_cr3 = walk_cr3;
            fetch_cache_cpl = walk_cpl;
            fetch_cache_valid = true;
        }
        int chunk = 4096 - (int)(cur & 0xfff);
        if (chunk > n - got)
            chunk = n - got;
        memcpy(out + got, ram_at(gpa), chunk);
        if (cfg.profile_compile_inputs)
            compile_inputs_read(cur, out + got, chunk, gpa);
        got += chunk;
    }
    return got;
}

const u8 *tr_fetch_fast(u64 va, int *avail) {
    if (__builtin_expect(tr_t2_mode || cfg.profile_compile_inputs, 0))
        return NULL;
    u64 off = va & 0xfffULL;
    if (__builtin_expect(off > 4096ULL - 32ULL, 0))
        return NULL;
    u64 cur_pg = va & ~0xfffULL;
    if (__builtin_expect(!(fetch_cache_valid && cur_pg == fetch_cache_va_page &&
                           walk_cr3 == fetch_cache_cr3 && walk_cpl == fetch_cache_cpl), 0)) {
        u64 gpa;
        if (!fetch_walk(va, &gpa))
            return NULL;
        fetch_cache_va_page = cur_pg;
        fetch_cache_gpa_page = gpa & ~0xfffULL;
        fetch_cache_host_page = ram_at(fetch_cache_gpa_page);
        fetch_cache_cr3 = walk_cr3;
        fetch_cache_cpl = walk_cpl;
        fetch_cache_valid = true;
    }
    if (va < fetch_lo)
        fetch_lo = va;
    if (va + 15ULL > fetch_hi)
        fetch_hi = va + 15ULL;
    *avail = 15;
    return fetch_cache_host_page + off;
}

// With several vCPUs a block is found the moment it exists, by vCPUs that may already have seen a later store. So its bytes must be read
// protected. A writer clears the mark before it takes the translator lock to kill, so a mark seen here means any later store kills what this reads.
static bool frame_unprotected(u64 va, u64 *frame) {
    u64 gpa;
    if (ctx_ptr->vcpu_count < 2 || !fetch_walk(va, &gpa))
        return false;
    *frame = gpa & ~0xfffULL;
    return !(__atomic_load_n(&cfg.priv[OFF_PAGEINFO + (gpa >> 12)], __ATOMIC_ACQUIRE) & PI_CODE);
}

// ---- Liveness ----
static int insn_reads(const Insn *in) {
    int r = in->reads;
    if (tr_precise_flags && in->memop)
        r = F_ARITH; // a fault here shows every flag in the x86 exception frame
    return r;
}

// Which flags can be read from `rip` on before being overwritten. Only looks inside the page `page`: anything else can change
// without invalidating the block asking, so it counts as all live.
// Use the instructions this block will actually emit when a local branch or
// loop returns into it. Apart from avoiding repeated decoding/page walks, this
// keeps its flag analysis consistent with that decoded input. Nothing survives
// this translation: other targets still go through the normal tracked fetch.
static void flags_decode(u64 rip, Insn *out, const Insn *block, int count) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (block[mid].rip < rip)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < count && block[lo].rip == rip)
        *out = block[lo];
    else
        x86_decode(rip, out);
}

static int flags_needed_at(u64 rip, u64 page, int depth, const Insn *block, int count) {
    int need = 0, written = 0;
    for (int i = 0; i < 32; i++) {
        if ((rip & ~0xfffULL) != page || ((rip + 15) & ~0xfffULL) != page)
            return need | (F_ARITH & ~written);
        Insn in;
        flags_decode(rip, &in, block, count);
        need |= insn_reads(&in) & ~written;
        written |= in.writes;
        if ((written & F_ARITH) == F_ARITH)
            return need;
        if (in.op == OP_JCC || in.op == OP_JRCXZ) {
            if (!depth)
                return need | (F_ARITH & ~written);
            return need | ((flags_needed_at(in.target, page, depth - 1, block, count) | flags_needed_at(in.next, page, depth - 1, block, count)) & ~written);
        }
        if (in.op == OP_JMP) {
            rip = in.target;
            continue;
        }
        if (in.op == OP_CALL && !tr_precise_flags && depth > 0 &&
            (in.target & ~0xfffULL) == page && ((in.target + 15) & ~0xfffULL) == page) {
            return need | (flags_needed_at(in.target, page, depth - 1, block, count) & ~written);
        }
        if (insn_is_terminator(&in) || in.op == OP_CALL || in.op == OP_CALLI)
            return need | (F_ARITH & ~written);
        rip = in.next;
    }
    return need | (F_ARITH & ~written);
}

// ---- Stubs ----
typedef struct {
    u32 *site; // the branch to fix up, or NULL for a plain fall-through stub
    u64 target;
    int kind; // SVC_DIRECT, SVC_JCC, SVC_CALL
    int form; // 0 b.cond/cbz (19-bit), 1 b/bl (26-bit), 2 tbz (unused)
} Pending;
static _Thread_local Pending pend[MAX_PENDING];
static _Thread_local int pend_count;

void emit_stop(u64 rip, const char *why) {
    compile_inputs_reject(TR_CI_UNSUPPORTED);
    aot_capture_mark_ineligible(&tr_aot_cap);
    (void)rip;
    pthread_mutex_lock(&stop_reason_lock);
    u32 idx = stop_reason_count;
    for (u32 i = 0; i < stop_reason_count; i++)
        if (stop_reasons[i] == why || !strcmp(stop_reasons[i], why)) {
            idx = i;
            break;
        }
    if (idx == stop_reason_count && stop_reason_count < 4096) {
        stop_reasons[stop_reason_count] = why;
        __atomic_store_n(&stop_reason_count, stop_reason_count + 1U, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&stop_reason_lock);
    svc(SVC_STOP);
    E(idx);
}

const char *tr_stop_reason(uint32_t index) {
    return index < __atomic_load_n(&stop_reason_count, __ATOMIC_ACQUIRE) ? stop_reasons[index] : "?";
}

static void add_pending(u32 *site, u64 target, int kind, int form) {
    pend[pend_count].site = site;
    pend[pend_count].target = target;
    pend[pend_count].kind = kind;
    pend[pend_count].form = form;
    pend_count++;
}

// Reserves a b.cond/cbz/b word to be resolved to a stub at the block end.
static void stub_lit(u64 target) {
    tr_aot_note_gva64_lit(tr_cc);
    E((u32)target);
    E((u32)(target >> 32));
}

static inline bool use_native_l1(int cpl, bool compat) {
    return native_l1 && cpl == 3 && !compat;
}

typedef struct {
    u32 *site1; // cbz x16, miss
    u32 *site2; // cbnz x23, miss
    u32 *blr_site; // for OP_CALLI: jump back to blr x22
    bool is_call;
} L1Miss;
static _Thread_local L1Miss l1_miss[MAX_L1_MISS];
static _Thread_local int l1_miss_count;

static _Thread_local int cur_key;
static _Thread_local u64 cur_entry_rip, cur_page;
static _Thread_local bool cur_single;

static u32 *try_direct_link_target(int key, u64 entry_rip, u64 page, u64 target, bool single) {
    if (single || target == entry_rip || tr_par_active || (tr_aot_active && tr_aot_cap.active))
        return NULL;
    if ((key & KEY_USER_FRAME) == KEY_USER_FRAME && (target & ~0xfffULL) != page)
        return NULL;
    u32 hit = map_find(key, target);
    if (!hit)
        return NULL;
    HostBlock *hb = &hblocks[hit - 1];
    if (hb->dead || hb->view != (u32)key || hb->cpl != (u8)tr_cpl || hb->compat != tr_compat)
        return NULL;
    if (p0_ring_enabled && cur_part == 0 && p0_slot_gen[p0_cur_slot] > p0_slot_gen[hb->p0_slot])
        return NULL; // never directly link newer ring slot -> older slot (prevents dangling links on ring recycle)
    return cc_base + hb->entry_word;
}

static void emit_indirect_tail(void) {
    // x16 holds the x86 target. Lookup returns the translation in x22, and the branch stays here so that each site has
    // its own predictor entry.
    if (use_native_l1(tr_cpl, tr_compat) && l1_miss_count < MAX_L1_MISS) {
        tr_ad_page = ~0ULL;
        u32 *s1 = tr_cc;
        cbz(0, 1, S0, 0);
        add_imm_lsl12(1, 0, 0, AD, CTX, 0x400);
        logic_sh(1, 2, 0, S1, S0, S0, 1, 4); // eor x17, x16, x16, lsr #4
        ubfiz(1, S1, S1, 4, 12);
        add_reg(1, 0, 0, AD, AD, S1, 0);
        E(0xA9400000u | (u32)S3 << 10 | (u32)AD << 5 | (u32)S2); // ldp x22, x23, [x20]
        logic_reg(1, 2, 0, S3, S3, S0);
        u32 *s2 = tr_cc;
        cbz(1, 1, S3, 0);
        br(S2);
        l1_miss[l1_miss_count++] = (L1Miss){s1, s2, NULL, false};
        return;
    }
    tr_aot_note_stub(tr_cc, AOT_STUB_LOOKUP, true);
    bl_rel((int)((i64)(STUB(tr_stub_lookup) - tr_cc_va(tr_cc)) / 4));
    br(S2);
}

static _Thread_local const Insn *block_insns;
static _Thread_local u32 *insn_arm[MAX_INSNS];
static _Thread_local int insn_total;

typedef struct {
    u32 *site;
    int target_idx;
    int form;
} FwdFix;
static _Thread_local FwdFix fwd_fix[MAX_FWD_FIX];
static _Thread_local int fwd_count;

// If target is an instruction already emitted in this block, returns where.
static u32 *internal_target(u64 target, int upto) {
    for (int i = 0; i <= upto && i < insn_total; i++)
        if (block_insns[i].rip == target)
            return insn_arm[i];
    return NULL;
}

// If target is a later instruction in this same superblock, returns its index.
static int forward_target_idx(u64 target, int from) {
    for (int i = from + 1; i < insn_total; i++)
        if (block_insns[i].rip == target)
            return i;
    return -1;
}

static void add_fwd(u32 *site, int target_idx, int form) {
    if (fwd_count < MAX_FWD_FIX)
        fwd_fix[fwd_count++] = (FwdFix){site, target_idx, form};
}

static void emit_monitor_op(const Insn *in);

static void emit_control(const Insn *in, int idx, int live_after) {
    switch (in->op) {
        case OP_JCC: {
            u32 *back = internal_target(in->target, idx);
            int fwd = back ? -1 : forward_target_idx(in->target, idx);
            u32 *direct = (back || fwd >= 0) ? NULL : try_direct_link_target(cur_key, cur_entry_rip, cur_page, in->target, cur_single);
            int acc = x86_to_arm_cc[in->cc];
            if (acc < 0) {
                emit_parity(S4);
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (back) {
                    cbz(in->cc == 10, 0, S4, (int)(back - tr_cc));
                } else if (fwd >= 0) {
                    add_fwd(tr_cc, fwd, 0);
                    cbz(in->cc == 10, 0, S4, 0);
                } else if (direct) {
                    cbz(in->cc == 10, 0, S4, (int)dd);
                } else {
                    add_pending(tr_cc, in->target, SVC_JCC, 0);
                    cbz(in->cc == 10, 0, S4, 0);
                }
            } else {
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (back) {
                    b_cond(acc, (int)(back - tr_cc));
                } else if (fwd >= 0) {
                    add_fwd(tr_cc, fwd, 0);
                    b_cond(acc, 0);
                } else if (direct) {
                    b_cond(acc, (int)dd);
                } else {
                    add_pending(tr_cc, in->target, SVC_JCC, 0);
                    b_cond(acc, 0);
                }
            }
            break;
        }
        case OP_JRCXZ: {
            u32 *back = internal_target(in->target, idx);
            int fwd = back ? -1 : forward_target_idx(in->target, idx);
            u32 *direct = (back || fwd >= 0) ? NULL : try_direct_link_target(cur_key, cur_entry_rip, cur_page, in->target, cur_single);
            int sub = in->sub;
            int sf = in->addr32 ? 0 : 1; // with 67h the count is ecx
            if (in->addr16) { // 32-bit code with 67h: the count is cx, and ecx's top half stays
                if (sub >= 2) {
                    add_imm(0, 1, 0, S0, 1, 1);
                    bfi(1, 1, S0, 0, 16);
                }
                ubfx(0, S0, 1, 0, 16);
                u32 *over = 0;
                if (sub >= 3) {
                    over = tr_cc;
                    b_cond(sub == 3 ? 1 : 0, 0);
                }
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (!back && !direct) {
                    if (fwd >= 0)
                        add_fwd(tr_cc, fwd, 0);
                    else
                        add_pending(tr_cc, in->target, SVC_JCC, 0);
                }
                cbz(sub >= 2, 0, S0, back ? (int)(back - tr_cc) : direct ? (int)dd : 0);
                if (over)
                    *over |= ((u32)(tr_cc - over) & 0x7ffff) << 5;
                break;
            }
            if (sub >= 2)
                add_imm(sf, 1, 0, 1, 1, 1);
            if (sub == 0 || sub == 1) {
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (!back && !direct) {
                    if (fwd >= 0)
                        add_fwd(tr_cc, fwd, 0);
                    else
                        add_pending(tr_cc, in->target, SVC_JCC, 0);
                }
                cbz(0, sub == 1 ? 0 : 1, 1, back ? (int)(back - tr_cc) : direct ? (int)dd : 0);
            } else if (sub == 2) {
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (!back && !direct) {
                    if (fwd >= 0)
                        add_fwd(tr_cc, fwd, 0);
                    else
                        add_pending(tr_cc, in->target, SVC_JCC, 0);
                }
                cbz(1, sf, 1, back ? (int)(back - tr_cc) : direct ? (int)dd : 0);
            } else {
                u32 *over = tr_cc;
                b_cond(sub == 3 ? 1 : 0, 0); // loope leaves when ZF is clear, loopne when it is set
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                    direct = NULL;
                if (!back && !direct) {
                    if (fwd >= 0)
                        add_fwd(tr_cc, fwd, 0);
                    else
                        add_pending(tr_cc, in->target, SVC_JCC, 0);
                }
                cbz(1, sf, 1, back ? (int)(back - tr_cc) : direct ? (int)dd : 0);
                *over |= ((u32)(tr_cc - over) & 0x7ffff) << 5;
            }
            break;
        }
        case OP_JMP: {
            u32 *back = internal_target(in->target, idx);
            int fwd = back ? -1 : forward_target_idx(in->target, idx);
            if (back) {
                b_rel((int)(back - tr_cc));
            } else if (fwd >= 0) {
                add_fwd(tr_cc, fwd, 1);
                b_rel(0);
            } else {
                u32 *direct = try_direct_link_target(cur_key, cur_entry_rip, cur_page, in->target, cur_single);
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
                    b_rel((int)dd);
                } else {
                    // Inline direct-link stub avoids b stub -> b target double-jump once linked.
                    u32 *svc_site = tr_cc;
                    svc(SVC_DIRECT);
                    tr_aot_note_cc_branch(svc_site, AOT_CC_DIRECT, tr_cc);
                    stub_lit(in->target);
                }
            }
            break;
        }
        case OP_CALL: {
            // Guest push first (it can fault), shadow pair next, then a real bl so that the matching ret is predicted.
            tr_emit_movimm_gva(S0, in->next, tr_compat);
            if (tr_compat)
                E(0xB8000C00u | ((u32)-4 & 0x1ff) << 12 | 4 << 5 | S0);
            else
                E(0xF8000C00u | ((u32)-8 & 0x1ff) << 12 | 4 << 5 | S0);
            adr_words(S1, 3);
            E(0xA9800000u | ((u32)-2 & 0x7f) << 15 | S1 << 10 | RS << 5 | S0); // stp x16, x17, [x26, #-16]!
            u32 *direct = try_direct_link_target(cur_key, cur_entry_rip, cur_page, in->target, cur_single);
            i64 dd = direct ? (direct - tr_cc) : 0;
            if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
                bl_rel((int)dd);
            } else {
                add_pending(tr_cc, in->target, SVC_CALL, 1);
                bl_rel(0);
            }
            break;
        }
        case OP_CALLI: {
            if (in->rm_reg)
                mov_reg(1, S0, in->rm);
            else
                ldst(1, in->sz, S0, addr_prepare(in, in->sz));
            tr_emit_movimm_gva(S1, in->next, tr_compat);
            if (tr_compat)
                E(0xB8000C00u | ((u32)-4 & 0x1ff) << 12 | 4 << 5 | S1);
            else
                E(0xF8000C00u | ((u32)-8 & 0x1ff) << 12 | 4 << 5 | S1);
            if (use_native_l1(tr_cpl, tr_compat)) {
                tr_ad_page = ~0ULL;
                adr_words(AD, 4);
                E(0xA9800000u | ((u32)-2 & 0x7f) << 15 | AD << 10 | RS << 5 | S1);
                tr_aot_note_stub(tr_cc, AOT_STUB_LOOKUP_L1_FAST, true);
                bl_rel((int)((i64)(STUB(PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1_FAST) - tr_cc_va(tr_cc)) / 4));
                blr(S2);
                break;
            }
            adr_words(AD, 4);
            E(0xA9800000u | ((u32)-2 & 0x7f) << 15 | AD << 10 | RS << 5 | S1);
            tr_aot_note_stub(tr_cc, AOT_STUB_LOOKUP, true);
            bl_rel((int)((i64)(STUB(tr_stub_lookup) - tr_cc_va(tr_cc)) / 4));
            E(0xD63F0000u | S2 << 5); // blr x22
            // landing pad: the next instruction (adr above counts from the adr: push, stp, bl, blr, then here)
            break;
        }
        case OP_JMPI:
            if (in->rm_reg)
                mov_reg(1, S0, in->rm);
            else
                ldst(1, in->sz, S0, addr_prepare(in, in->sz));
            emit_indirect_tail();
            break;
        case OP_RET: {
            // shortcut: post-index leaves esp past 4 GB for a stack at the very top of memory. No XNU process has one, as the commpage sits there
            if (tr_compat)
                E(0xB8400400u | 4 << 12 | 4 << 5 | S0); // ldr w16, [x4], #4
            else
                E(0xF8400400u | 8 << 12 | 4 << 5 | S0); // ldr x16, [x4], #8
            if (in->imm) {
                if (in->imm < 4096)
                    add_imm(!tr_compat, 0, 0, 4, 4, (u32)in->imm);
                else {
                    movimm(S1, (u64)in->imm);
                    add_reg(!tr_compat, 0, 0, 4, 4, S1, 0);
                }
            }
            bool use_l1_ret = use_native_l1(tr_cpl, tr_compat);
            u64 ret_miss_va = use_l1_ret
                                  ? STUB(PRIV_VA + OFF_STUBS + STUB_OFF_RET_MISS_L1)
                                  : STUB(tr_stub_ret_miss);
            AotStubId ret_sid = use_l1_ret ? AOT_STUB_RET_MISS_L1 : AOT_STUB_RET_MISS;
            if (cfg.no_return_prediction) { // the guest's value decides, by lookup
                tr_aot_note_stub(tr_cc, ret_sid, false);
                b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
                break;
            }
            E(0xA8C00000u | 2 << 15 | LR << 10 | RS << 5 | S1); // ldp x17, x30, [x26], #16
            logic_reg(1, 2, 0, S1, S1, S0);
            cbz(1, 1, S1, 2);
            ret_reg(LR);
            tr_aot_note_stub(tr_cc, ret_sid, false);
            b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
            break;
        }
        default:
            break;
    }
    (void)live_after;
}

// Patch helper: resolves a forward reference.
static void fix(u32 *site, int form, u32 *to) {
    i64 d = to - site;
    if (form == 0)
        *site = (*site & ~(0x7ffffu << 5)) | ((u32)d & 0x7ffff) << 5;
    else if (form == 2)
        *site = (*site & ~(0x3fffu << 5)) | ((u32)d & 0x3fff) << 5;
    else
        *site = (*site & 0xFC000000u) | ((u32)d & 0x3ffffff);
}

static bool p0_slot_full(u32 s) {
    u32 *base = cc_base + engine_p0_slot_word_base(s);
    // Reserve 2 MiB (~43k lookup tails) per slot so cross-slot tr_lookup_tail never starves.
    return (u8 *)p0_cc_cursor[s] - (u8 *)base > (long)(P0_SLOT_BYTES - (2ULL << 20)) ||
           p0_block_count[s] >= P0_BLOCKS_PER_SLOT - 32 ||
           p0_pcmap_used[s] + MAX_INSNS + 1024 >= (s + 1U) * P0_PCMAP_PER_SLOT;
}

// Chronological 4-slot ring (0 -> 1 -> 2 -> 3 -> 0): older slots can permanently direct-link into newer slots;
// newer slots linking into older slots use fast inline L1 lookup tails (tr_lookup_tail).
static inline u32 p0_next_ring_slot(u32 cur) {
    return (cur + 1U) & (P0_SLOTS - 1U);
}

static bool p0_try_advance_empty_slot(void) {
    if (!p0_ring_enabled || !p0_slot_full(p0_cur_slot))
        return true;
    u32 next = p0_next_ring_slot(p0_cur_slot);
    if (p0_block_count[next] == 0) {
        p0_cur_slot = next;
        p0_slot_gen[next] = ++p0_gen_seq;
        __atomic_store_n(&ctx_ptr->p0_slot_gen[next], p0_slot_gen[next], __ATOMIC_RELEASE);
        p0_ring_need_evict = false;
        return true;
    }
    p0_ring_need_evict = true;
    return false;
}

static bool part_full(int part) {
    if (part == 0 && p0_ring_enabled) {
        if (!p0_slot_full(p0_cur_slot))
            return false;
        u32 next = p0_next_ring_slot(p0_cur_slot);
        if (p0_block_count[next] == 0)
            return false; // can advance to empty ring slot inline without parking
        p0_ring_need_evict = true;
        return true;
    }
    u32 *start = cc_base + part * CC_PART2_WORD;
    u32 first_map = part ? (OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap) : 0;
    u64 lim = part == 1 && cc_limit >= CC_SIZE ? CC_SIZE - (4ULL << 20) : cc_limit;
    return (u8 *)cc_cursor[part] - (u8 *)start > (long)(lim - 0x80000) ||
           hblock_count[part] >= PART_BLOCKS_MAX(part) - 32 ||
           pc_map_used[part] + MAX_INSNS + 1024 >= first_map + PCMAP_ENTRIES;
}

bool tr_needs_flush(void) { return part_full(0) || part_full(1); }

static void flush_everything(void) {
    tr_flush_all();
}

// ---- Reusing dead blocks ----
// Blocks often die with nothing wrong with them: a view reused, a mapping rewritten. A block is a function of its x86 address, the bytes translating it read,
// its CPL and code size, and settings fixed for the run. So if those match it can come back, under the protection a new translation would read them under.
#define DEAD_SLOTS (1u << 20)
#define READ_COPY_BYTES (512u << 20) // virtual: touched as used. Past it, blocks just aren't reusable until the next reset
#define WORD_NOP 0xD503201Fu
#define WORD_DEAD (0xD4000001u | SVC_DEAD << 5)
static u32 *dead_slots; // by key and rip: a block index + 1, the last to die there. A lost entry only means translating again
static u8 *read_copies;
static u32 read_copied[2] = {1, READ_COPY_BYTES / 2};

static inline u32 dead_slot(u32 key, u64 rip) { return (u32)((rip * 0x9E3779B97F4A7C15ULL + key) >> 40) & (DEAD_SLOTS - 1); }

static void flush_dead_slots(bool everything) {
    if (!cfg.retain_shared_dead) {
        memset(dead_slots, 0, DEAD_SLOTS * sizeof(u32));
        return;
    }
    // Values are encoded block index + 1. A private reset preserves P0's code,
    // HostBlocks and saved bytes; P1 indices will be recycled. Keep only existing
    // allocated P0 hints, without repopulating candidates lost to a collision.
    // Reuse still applies every original key/frame/byte/permission/entry guard.
    u32 shared_count = hblock_count[0];
    for (u32 slot = 0; slot < DEAD_SLOTS; slot++) {
        u32 hint = dead_slots[slot];
        if (!hint)
            continue;
        if (!everything && hint <= shared_count && hint <= BLOCKS_MAX) {
            stats.shared_dead_retained_hints++;
        } else {
            dead_slots[slot] = 0;
            stats.shared_dead_cleared_hints++;
        }
    }
}

static void remember_reads(HostBlock *hb, u64 rip) {
    u64 page_end = (rip & ~0xfffULL) + 4096, lo = fetch_lo, hi = fetch_hi < page_end ? fetch_hi : page_end;
    u32 end = cur_part ? READ_COPY_BYTES : READ_COPY_BYTES / 2;
    if (cfg.no_reuse || hb->frame[1] != ~0ULL || hb->frame[0] >= cfg.ram_size || read_copied[cur_part] + (hi - lo) > end)
        return;
    memcpy(read_copies + read_copied[cur_part], ram_at(hb->frame[0] + (lo & 0xfff)), hi - lo);
    hb->read_copy = read_copied[cur_part];
    hb->read_off = (u16)(lo & 0xfff);
    hb->read_bytes = (u16)(hi - lo);
    read_copied[cur_part] += (u32)(hi - lo);
}

static inline bool reuse_rejected(u32 reason) {
    if (cfg.profile_invalidations)
        stats.reuse_reject[reason]++;
    return false;
}

static bool reuse_dead_block(u32 key, int cpl, u64 rip, TrResult *res) {
    if (cfg.profile_invalidations)
        stats.reuse_attempts++;
    u32 it = cfg.no_reuse ? 0 : dead_slots[dead_slot(key, rip)];
    if (!it)
        return reuse_rejected(cfg.no_reuse ? TR_REUSE_DISABLED : TR_REUSE_NO_SLOT);
    HostBlock *hb = &hblocks[it - 1];
    u64 gpa;
    // Keep the original short-circuit order, especially the permission walk, byte comparison and entry CAS.
    if (!hb->dead) return reuse_rejected(TR_REUSE_NOT_DEAD);
    if (hb->rip != rip) return reuse_rejected(TR_REUSE_RIP);
    if (hb->view != key) return reuse_rejected(TR_REUSE_KEY);
    if (!hb->read_bytes) return reuse_rejected(TR_REUSE_NO_READ_COPY);
    if (hb->cpl != cpl) return reuse_rejected(TR_REUSE_CPL);
    if (hb->compat != tr_compat) return reuse_rejected(TR_REUSE_COMPAT);
    if (!fetch_walk(rip, &gpa)) return reuse_rejected(TR_REUSE_FETCH);
    if ((gpa & ~0xfffULL) != hb->frame[0]) return reuse_rejected(TR_REUSE_FRAME);
    if (hb->t2_has_loop && hb->t2_patch_word != hb->entry_word)
        return reuse_rejected(TR_REUSE_SMP_ENTRY);
    if (ctx_ptr->vcpu_count > 1 && hb->first_word != WORD_NOP) // only a nop can come back under a peer running it
        return reuse_rejected(TR_REUSE_SMP_ENTRY);
    if (memcmp(ram_at(hb->frame[0] + hb->read_off), read_copies + hb->read_copy, hb->read_bytes))
        return reuse_rejected(TR_REUSE_BYTES);
    u32 *entry = cc_base + hb->entry_word, dead = WORD_DEAD;
    if (!__atomic_compare_exchange_n(entry, &dead, hb->first_word, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return reuse_rejected(TR_REUSE_ENTRY_CAS); // something that ran into it has since been sent on to a newer block, for good
    hb->t2_entry_word = 0;
    hb->t2_meta_idx = 0;
    sys_icache_invalidate(entry, 4);
    if (entry[1] == 0x10000050u) {
        entry[3] = WORD_NOP;
        sys_icache_invalidate(entry + 3, 4);
    }
    code_published();
    hb->dead = false;
    u32 idx = it - 1;
    if (!hb->on_frame) {
        u32 *head = &frame_head[hb->frame[0] >> 12];
        hb->next_on_frame[0] = *head;
        *head = idx + 1;
        hb->on_frame = true;
    }
    if (!hb->on_page) {
        u32 *head = &page_head[page_slot(rip)];
        hb->next_on_page[0] = *head;
        *head = idx + 1;
        hb->on_page = true;
    }
    if (!(key & KEY_DOMAIN) && !hb->on_view) {
        hb->next_in_view = view_head[key];
        view_head[key] = idx + 1;
        hb->on_view = true;
    }
    map_insert((int)key, rip, idx + 1);
    res->arm = tr_cc_va(entry);
    res->frames[0] = hb->frame[0];
    stats.reused++;
    return true;
}

// A block joins the chains that find it to kill it.
static void link_block(u32 idx) {
    HostBlock *hb = &hblocks[idx];
    for (int f = 0; f < 2; f++) {
        if (hb->frame[f] != ~0ULL && hb->frame[f] < cfg.ram_size) {
            u32 *head = &frame_head[hb->frame[f] >> 12];
            hb->next_on_frame[f] = *head;
            *head = idx + 1;
        }
    }
    u32 *head = &page_head[page_slot(hb->rip)];
    hb->next_on_page[0] = *head;
    *head = idx + 1;
    if (hb->frame[1] != ~0ULL) { // it runs on into the next page
        head = &page_head[page_slot(hb->rip + 4096)];
        hb->next_on_page[1] = *head;
        *head = idx + 1;
    }
    if (!(hb->view & KEY_DOMAIN)) {
        hb->next_in_view = view_head[hb->view];
        view_head[hb->view] = idx + 1;
    }
    hb->on_frame = hb->on_page = hb->on_view = true;
}

void tr_set_lock_hooks(void (*unlock)(void *), void (*relock)(void *), void *ctx) {
    (void)unlock;
    (void)relock;
    (void)ctx;
}

static bool is_monitor_or_stop_op(u8 op) {
    switch (op) {
        case OP_BAD: case OP_UD:
        case OP_SYSCALL: case OP_SYSRET: case OP_IRET: case OP_INT: case OP_INT3:
        case OP_HLT: case OP_CPUID: case OP_RDTSC: case OP_RDTSCP:
        case OP_RDMSR: case OP_WRMSR: case OP_CLI: case OP_STI:
        case OP_MOV_CR: case OP_MOV_DR: case OP_INVLPG: case OP_INVPCID:
        case OP_WBINVD: case OP_CLTS: case OP_LGDT: case OP_LTR:
        case OP_MOV_SEG: case OP_IN: case OP_OUT: case OP_SWAPGS:
        case OP_SYSENTER: case OP_SYSEXIT: case OP_FAR:
        case OP_FXSAVE: case OP_FXRSTOR:
            return true;
        default:
            return false;
    }
}

static inline bool is_uncond_term_op(u8 op) {
    return op == OP_JMP || op == OP_JMPI || op == OP_RET || op == OP_BAD || op == OP_UD || op == OP_IRET;
}

static int idx_of_rip(const Insn *list, int count, u64 target) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (list[mid].rip < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < count && list[lo].rip == target) ? lo : -1;
}

static int linear_find_rip(const Insn *list, int count, u64 target) {
    for (int i = 0; i < count; i++)
        if (list[i].rip == target)
            return i;
    return -1;
}

static bool overlaps_any(const Insn *list, int count, u64 start_rip, u64 end_rip) {
    for (int i = 0; i < count; i++)
        if (!(end_rip <= list[i].rip || start_rip >= list[i].next))
            return true;
    return false;
}

static int est_insn_words(const Insn *in) {
    if (in->op == OP_EXT) return 24;
    if (in->op == OP_SSE || in->op == OP_X87 || in->op == OP_STRING) return 16;
    return 4;
}

static void expand_forward_dag(Insn *list, int *count_p, int key, u64 entry_rip, u64 page) {
    int count = *count_p;
    bool has_backedge = false;
    for (int i = 0; i < count; i++) {
        u8 op = list[i].op;
        if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) &&
            list[i].target >= entry_rip && list[i].target <= list[i].rip) {
            has_backedge = true;
            break;
        }
    }
    int max_dag_blocks = tr_cpl == 0 ? 4 : 6;
    int max_dag_insns = tr_cpl == 0 ? 96 : 144;
    u64 max_dag_span = tr_cpl == 0 ? 384 : 512;
    int max_est_words = tr_cpl == 0 ? 1200 : 1600;
    if (count <= 0 || count >= max_dag_insns)
        return;
    int est_words = 0;
    u64 max_end = entry_rip;
    for (int i = 0; i < count; i++) {
        if (is_monitor_or_stop_op(list[i].op) || list[i].op == OP_POPF)
            return;
        est_words += est_insn_words(&list[i]);
        if (list[i].next > max_end)
            max_end = list[i].next;
    }
    if (est_words >= max_est_words)
        return;

    u64 loop_end = has_backedge ? list[count - 1].next : ~0ULL;
    u64 q[MAX_DAG_BLOCKS * 2];
    int qh = 0, qt = 0;
    for (int i = 0; i < count; i++) {
        u8 op = list[i].op;
        if (op != OP_JCC && op != OP_JRCXZ && op != OP_JMP)
            continue;
        u64 t = list[i].target;
        if (t <= list[i].rip || t < entry_rip || t >= entry_rip + max_dag_span || t >= loop_end)
            continue;
        if ((t & ~0xfffULL) != page || (t & 0xfffULL) > 4096 - 16)
            continue;
        if (t < max_end && linear_find_rip(list, count, t) >= 0)
            continue;
        bool dup = false;
        for (int k = 0; k < qt; k++)
            if (q[k] == t) { dup = true; break; }
        if (!dup && qt < MAX_DAG_BLOCKS * 2)
            q[qt++] = t;
    }

    int blocks_added = 0;
    Insn tmp[MAX_INSNS];
    while (qh < qt && blocks_added < max_dag_blocks && count < max_dag_insns && est_words < max_est_words) {
        u64 t = q[qh++];
        if (t < max_end && linear_find_rip(list, count, t) >= 0)
            continue;
        if (!cfg.profile_compile_inputs && !tr_par_active) {
            u32 hit = map_find(key, t);
            if (hit && !hblocks[hit - 1].dead)
                continue;
        }
        int tc = 0, test_w = 0;
        u64 spc = t;
        bool ok = true;
        while (count + tc < max_dag_insns && est_words + test_w < max_est_words) {
            if (spc < max_end && linear_find_rip(list, count, spc) >= 0)
                break;
            if ((spc & ~0xfffULL) != page || (spc & 0xfffULL) > 4096 - 15) {
                ok = false;
                break;
            }
            x86_decode(spc, &tmp[tc]);
            Insn *in = &tmp[tc];
            if (in->next > page + 4096 || in->next <= in->rip ||
                is_monitor_or_stop_op(in->op) || in->op == OP_POPF) {
                ok = false;
                break;
            }
            if (in->rip < max_end && overlaps_any(list, count, in->rip, in->next)) {
                ok = false;
                break;
            }
            test_w += est_insn_words(in);
            spc = in->next;
            tc++;
            if (insn_is_terminator(in) ||
                (t2_enabled() && (in->op == OP_JCC || in->op == OP_JRCXZ) &&
                 in->target >= entry_rip && in->target < in->rip))
                break;
        }
        if (!ok || tc == 0)
            continue;
        memcpy(&list[count], tmp, (size_t)tc * sizeof(Insn));
        int base = count;
        count += tc;
        est_words += test_w;
        blocks_added++;
        if (tmp[tc - 1].next > max_end)
            max_end = tmp[tc - 1].next;
        for (int i = base; i < count; i++) {
            u8 op = list[i].op;
            if (op != OP_JCC && op != OP_JRCXZ && op != OP_JMP)
                continue;
            u64 nt = list[i].target;
            if (nt <= list[i].rip || nt < entry_rip || nt >= entry_rip + max_dag_span)
                continue;
            if ((nt & ~0xfffULL) != page || (nt & 0xfffULL) > 4096 - 16)
                continue;
            if (nt < max_end && linear_find_rip(list, count, nt) >= 0)
                continue;
            bool dup = false;
            for (int k = 0; k < qt; k++)
                if (q[k] == nt) { dup = true; break; }
            if (!dup && qt < MAX_DAG_BLOCKS * 2)
                q[qt++] = nt;
        }
    }

    if (blocks_added > 0) {
        for (int i = 1; i < count; i++) {
            Insn cur = list[i];
            int j = i - 1;
            while (j >= 0 && list[j].rip > cur.rip) {
                list[j + 1] = list[j];
                j--;
            }
            list[j + 1] = cur;
        }
        *count_p = count;
    }
}

static inline bool ad_page_safe_op(u8 op) {
    switch (op) {
        case OP_SSE: case OP_EXT:
        case OP_MOV: case OP_ALU: case OP_LEA: case OP_MOVZX: case OP_MOVSX:
        case OP_INCDEC: case OP_CMOV: case OP_SETCC:
        case OP_SHIFT: case OP_SHD: case OP_NEG: case OP_NOT:
        case OP_PUSH: case OP_POP: case OP_LEAVE:
        case OP_BSWAP: case OP_CBW: case OP_CWD:
        case OP_JCC: case OP_JRCXZ: case OP_JMP:
        case OP_NOP: case OP_CLC: case OP_STC: case OP_CMC: case OP_CLD: case OP_STD:
        case OP_PREFETCH: case OP_FENCE:
        case OP_BSF: case OP_BSR: case OP_TZCNT: case OP_LZCNT: case OP_POPCNT: case OP_CRC32:
        case OP_MOVBE: case OP_XCHG: case OP_XADD: case OP_CMPXCHG:
            return true;
        default:
            return false;
    }
}

static inline int gpr_phys(int r, bool high) {
    return high ? (r - 4) : r;
}

static void update_zext(const Insn *in) {
    switch (in->op) {
        case OP_NOP: case OP_FENCE: case OP_PREFETCH: case OP_CLFLUSH:
        case OP_JCC: case OP_JMP: case OP_PUSHF: case OP_POPF:
        case OP_SAHF: case OP_CMC: case OP_CLC: case OP_STC: case OP_CLD: case OP_STD:
        case OP_LDMXCSR: case OP_STMXCSR:
            return;
        case OP_MOV: {
            if (!in->to_reg && !in->rm_reg)
                return;
            bool high = in->to_reg ? in->high_reg : in->high_rm;
            int d = gpr_phys(in->to_reg ? in->reg : in->rm, high);
            u16 bit = (u16)(1u << d);
            tr_zext8 &= (u16)~bit;
            tr_zext16 &= (u16)~bit;
            if (in->has_imm && !high && in->sz >= 4) {
                u64 v = in->sz == 4 ? (u32)in->imm : (u64)in->imm;
                if (v <= 0xffULL) { tr_zext8 |= bit; tr_zext16 |= bit; }
                else if (v <= 0xffffULL) tr_zext16 |= bit;
            } else if (!in->has_imm && in->rm_reg && !high && in->sz >= 4) {
                int s = in->to_reg ? in->rm : in->reg;
                bool sh = in->to_reg ? in->high_rm : in->high_reg;
                if (!sh) {
                    if (tr_zext8 & (1u << s)) tr_zext8 |= bit;
                    if (tr_zext16 & (1u << s)) tr_zext16 |= bit;
                }
            }
            return;
        }
        case OP_MOVZX: {
            u16 bit = (u16)(1u << in->reg);
            tr_zext8 &= (u16)~bit;
            tr_zext16 &= (u16)~bit;
            if (in->sz >= 4) {
                if (in->srcsz == 1) { tr_zext8 |= bit; tr_zext16 |= bit; }
                else if (in->srcsz == 2) tr_zext16 |= bit;
            }
            return;
        }
        case OP_ALU: {
            if (in->sub == ALU_CMP || in->sub == ALU_TEST)
                return;
            if (!in->to_reg && !in->rm_reg)
                return;
            bool high = in->to_reg ? in->high_reg : in->high_rm;
            int d = gpr_phys(in->to_reg ? in->reg : in->rm, high);
            u16 bit = (u16)(1u << d);
            tr_zext8 &= (u16)~bit;
            tr_zext16 &= (u16)~bit;
            if (!high && in->sz >= 4) {
                if ((in->sub == ALU_XOR || in->sub == ALU_SUB) && !in->has_imm && in->rm_reg &&
                    in->reg == in->rm && !in->high_reg && !in->high_rm) {
                    tr_zext8 |= bit;
                    tr_zext16 |= bit;
                } else if (in->sub == ALU_AND && in->has_imm) {
                    u64 m = in->sz == 4 ? (u32)in->imm : (u64)in->imm;
                    if ((m & ~0xffULL) == 0) { tr_zext8 |= bit; tr_zext16 |= bit; }
                    else if ((m & ~0xffffULL) == 0) tr_zext16 |= bit;
                }
            }
            return;
        }
        case OP_SHIFT: {
            if (!in->rm_reg)
                return;
            int d = gpr_phys(in->rm, in->high_rm);
            u16 bit = (u16)(1u << d);
            tr_zext8 &= (u16)~bit;
            tr_zext16 &= (u16)~bit;
            if (in->sub == SH_SHR && !in->by_cl && !in->high_rm) {
                int amt = (int)(in->imm & (in->sz == 8 ? 63 : 31));
                if ((in->sz == 4 && amt >= 24) || (in->sz == 8 && amt >= 56)) {
                    tr_zext8 |= bit;
                    tr_zext16 |= bit;
                } else if ((in->sz == 4 && amt >= 16) || (in->sz == 8 && amt >= 48)) {
                    tr_zext16 |= bit;
                }
            }
            return;
        }
        case OP_INCDEC: case OP_NEG: case OP_NOT: case OP_BSWAP: case OP_SETCC: case OP_SHD:
            if (in->rm_reg) {
                int d = gpr_phys(in->rm, in->high_rm);
                tr_zext8 &= (u16)~(1u << d);
                tr_zext16 &= (u16)~(1u << d);
            }
            return;
        case OP_BT:
            if (in->sub != 0 && in->rm_reg) {
                tr_zext8 &= (u16)~(1u << in->rm);
                tr_zext16 &= (u16)~(1u << in->rm);
            }
            return;
        case OP_LEA: case OP_MOVSX: case OP_CMOV: case OP_IMUL:
        case OP_BSF: case OP_BSR: case OP_TZCNT: case OP_LZCNT: case OP_POPCNT: case OP_CRC32:
            tr_zext8 &= (u16)~(1u << in->reg);
            tr_zext16 &= (u16)~(1u << in->reg);
            return;
        case OP_MOVBE:
            if (in->to_reg) {
                tr_zext8 &= (u16)~(1u << in->reg);
                tr_zext16 &= (u16)~(1u << in->reg);
            }
            return;
        case OP_PUSH:
            tr_zext8 &= (u16)~(1u << 4);
            tr_zext16 &= (u16)~(1u << 4);
            return;
        case OP_POP: {
            u16 clr = (u16)(1u << 4) | (in->rm_reg ? (u16)(1u << in->rm) : 0);
            tr_zext8 &= (u16)~clr;
            tr_zext16 &= (u16)~clr;
            return;
        }
        case OP_LEAVE: case OP_ENTER:
            tr_zext8 &= (u16)~((1u << 4) | (1u << 5));
            tr_zext16 &= (u16)~((1u << 4) | (1u << 5));
            return;
        case OP_XCHG: case OP_XADD: {
            int dr = gpr_phys(in->reg, in->high_reg);
            u16 clr = (u16)(1u << dr) | (in->rm_reg ? (u16)(1u << gpr_phys(in->rm, in->high_rm)) : 0);
            tr_zext8 &= (u16)~clr;
            tr_zext16 &= (u16)~clr;
            return;
        }
        case OP_CBW: case OP_XLAT: case OP_LAHF:
            tr_zext8 &= (u16)~1u;
            tr_zext16 &= (u16)~1u;
            return;
        case OP_CWD:
            tr_zext8 &= (u16)~(1u << 2);
            tr_zext16 &= (u16)~(1u << 2);
            return;
        case OP_JRCXZ:
            if (in->sub >= 2) {
                tr_zext8 &= (u16)~(1u << 1);
                tr_zext16 &= (u16)~(1u << 1);
            }
            return;
        default:
            tr_zext8 = tr_zext16 = 0;
            return;
    }
}

u8 tr_insn_reads(const Insn *in) { return (u8)insn_reads(in); }
void tr_update_zext(const Insn *in) { update_zext(in); }
bool tr_ad_page_safe_op(u8 op) { return ad_page_safe_op(op); }
bool tr_is_monitor_or_stop_op(u8 op) { return is_monitor_or_stop_op(op); }

static bool t2_loop_op_forbidden(const Insn *in) {
    if (in->lock || is_monitor_or_stop_op(in->op))
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
            return in->memop;
        default:
            return false;
    }
}

// Identify the innermost clean single-entry loop [head_i .. tail_i] eligible for Tier-2 OSR.
static int t2_find_clean_loop(const Insn *list, int count, const int *tgt_idx) {
    if (!t2_enabled() || count < 2)
        return -1;
    int best_head = -1, best_len = 9999;
    for (int i = 0; i < count; i++) {
        u8 op = list[i].op;
        if (op != OP_JCC && op != OP_JRCXZ && op != OP_JMP)
            continue;
        int h = tgt_idx[i];
        if (h < 0 || h > i)
            continue;
        int len = i - h + 1;
        if (len < 2 || len > 96 || len >= best_len)
            continue;
        bool ok = true;
        for (int j = h; j <= i; j++) {
            if (t2_loop_op_forbidden(&list[j])) { ok = false; break; }
            int tj = tgt_idx[j];
            if (tj > h && tj <= j) { ok = false; break; } // nested inner backedge; tj < h is a valid cold exit
        }
        // Single-entry check: no instruction before h may jump strictly inside (h..i].
        for (int j = 0; ok && j < h; j++) {
            int tj = tgt_idx[j];
            if (tj > h && tj <= i)
                ok = false;
        }
        if (ok) {
            best_head = h;
            best_len = len;
        }
    }
    return best_head;
}

// ---- Persistent AOT Cache Helpers ----
static u32 aot_build_mode_bits(int cpl, bool compat, bool is_t2) {
    u32 m = (u32)(cpl & 3);
    if (compat) m |= AOT_MODE_COMPAT;
    if (is_t2) m |= AOT_MODE_IS_T2;
    if (ctx_ptr->vcpu_count > 1) m |= AOT_MODE_SMP;
    if (tr_precise_flags) m |= AOT_MODE_PRECISE_FLAGS;
    if (tr_has_tzcnt) m |= AOT_MODE_TZCNT_LZCNT;
    if (tr_hybrid) m |= AOT_MODE_HYBRID;
    if (tr_fpcr_merges_scalar) m |= AOT_MODE_FPCR_MERGE;
    if (native_dag) m |= AOT_MODE_NATIVE_DAG;
    if (native_l1) m |= AOT_MODE_NATIVE_L1;
    if (tr_opt_peep) m |= AOT_MODE_OPT_PEEP;
    if (tr_opt_ymm_cache) m |= AOT_MODE_OPT_YMM_CACHE;
    if (t2_enabled()) m |= AOT_MODE_T2_ENABLED;
    if (cfg.keep_kernel_tables) m |= AOT_MODE_KEEP_KPTI;
    if (cfg.no_bulk_rep) m |= AOT_MODE_NO_BULK_REP;
    if (cfg.no_return_prediction) m |= AOT_MODE_NO_RET_PRED;
    if (cfg.all_flags_live) m |= AOT_MODE_ALL_FLAGS_LIVE;
    if (cfg.profile_shift_helpers) m |= AOT_MODE_PROF_SHIFT;
    if (cfg.inline_shr8) m |= AOT_MODE_INLINE_SHR8;
    if (cfg.page_rep) m |= AOT_MODE_PAGE_REP;
    m |= t2_ablation_bits();
    return m;
}

typedef struct {
    int key;
    u64 entry_rip;
    u64 page;
} AotLinkCtx;

static u64 aot_resolve_live_target(void *vctx, u64 target_rip) {
    if (!aot_eager_link)
        return 0;
    const AotLinkCtx *lc = (const AotLinkCtx *)vctx;
    if (target_rip == lc->entry_rip)
        return 0;
    if ((lc->key & KEY_USER_FRAME) == KEY_USER_FRAME && (target_rip & ~0xfffULL) != lc->page)
        return 0;
    u32 hit = map_find(lc->key, target_rip);
    if (!hit)
        return 0;
    HostBlock *hb = &hblocks[hit - 1];
    if (hb->dead || hb->view != (u32)lc->key || hb->cpl != (u8)tr_cpl || hb->compat != tr_compat)
        return 0;
    if (p0_ring_enabled && cur_part == 0 && p0_slot_gen[p0_cur_slot] > p0_slot_gen[hb->p0_slot])
        return 0;
    return tr_cc_va(cc_base + hb->entry_word);
}

static void aot_apply_live_eager_links(u32 *start, u32 code_words, const AotCapture *cap,
                                       const AotLinkCtx *lc) {
    if (!aot_eager_link || !cap || !cap->count)
        return;
    u64 base_va = tr_cc_va(start);
    for (u32 i = 0; i < cap->count; i++) {
        const AotReloc *r = &cap->relocs[i];
        if (r->type != AOT_RELOC_CC_BRANCH)
            continue;
        u32 w = r->word_off, lit_w = r->aux2;
        if (w >= code_words || lit_w + 1U >= code_words)
            continue;
        u64 target_rip = (u64)start[lit_w] | ((u64)start[lit_w + 1U] << 32);
        u64 target_arm = aot_resolve_live_target((void *)lc, target_rip);
        if (!target_arm)
            continue;
        i64 d = (i64)(target_arm - (base_va + (u64)w * 4ULL)) / 4;
        if (r->aux == AOT_CC_DIRECT && d >= -(1LL << 25) && d < (1LL << 25))
            start[w] = 0x14000000u | ((u32)d & 0x3FFFFFFu);
        else if (r->aux == AOT_CC_CALL && d >= -(1LL << 25) && d < (1LL << 25))
            start[w] = 0x94000000u | ((u32)d & 0x3FFFFFFu);
        else if (r->aux == AOT_CC_JCC && d >= -(1LL << 18) && d < (1LL << 18))
            start[w] = (start[w] & ~(0x7FFFFu << 5)) | (((u32)d & 0x7FFFFu) << 5);
    }
}

void tr_aot_record_t2(u64 frame0, u64 head_rip, u8 cpl, bool compat,
                      const u32 *code, u32 code_words, const PcMap *pcm, u32 pcm_count,
                      const T2TrapMeta *trap_meta, const u32 *bail_sites, u32 bail_cnt,
                      const AotCapture *cap) {
    if (!tr_aot_active || !aot_cache || !cap || cap->ineligible || frame0 >= cfg.ram_size)
        return;
    AotCapture full_cap = *cap;
    for (u32 k = 0; k < bail_cnt; k++)
        aot_capture_add(&full_cap, bail_sites[k], AOT_RELOC_T2_BAILOUT, 0, 0);
    if (full_cap.ineligible)
        return;
    u64 page_hash = aot_hash_frame_cached(aot_cache, frame0, ram_at(frame0));
    AotBlockView v;
    memset(&v, 0, sizeof v);
    v.key.page_hash = page_hash;
    v.key.abi_hash = aot_abi_hash;
    v.key.mode_bits = aot_build_mode_bits((int)cpl, compat, true);
    v.key.page_offset = (u16)(head_rip & 0xFFFULL);
    v.orig_rip = head_rip;
    v.code = code;
    v.code_words = code_words;
    v.pcm = pcm;
    v.pcm_count = (u16)pcm_count;
    v.relocs = full_cap.relocs;
    v.reloc_count = (u16)full_cap.count;
    v.t2_patch_word_off = AOT_NO_PATCH_WORD;
    v.priority = AOT_PRIO_TIER2;
    if (trap_meta) {
        v.has_t2_meta = true;
        v.t2_meta = *trap_meta;
    }
    (void)aot_record(aot_cache, &v);
}

void tr_aot_flush(void) {
    if (aot_cache)
        (void)aot_save(aot_cache);
}

static void tr_aot_atexit_flush(void) {
    if (aot_cache) {
        AotCache *c = aot_cache;
        aot_cache = NULL;
        tr_aot_active = false;
        aot_close(c, true);
    }
}

// ---- 16-Core Lock-Free Parallel Translation Pipeline ----
#define PAR_CODE_CAP 32768U
#define PAR_PCM_CAP (MAX_INSNS + 256U)
#define PAR_MAX_STUBS 512U
#define PAR_MAX_LINKS (MAX_PENDING + 64U)

typedef struct {
    u32 word_off;
    u8 stub_id;
    u8 is_bl;
} ParStubSite;

typedef struct {
    u32 word_off;
    u32 lit_off;
    u8 kind;
} ParLinkSite;

typedef enum {
    PAR_RES_NONE = 0,
    PAR_RES_AOT_HIT,
    PAR_RES_COLD,
    PAR_RES_FETCH_FAULT,
    PAR_RES_UNPROTECTED,
    PAR_RES_FALLBACK_LOCKED,
} ParResKind;

typedef struct {
    u64 op_mask_lo, op_mask_hi;
    u32 op_insns_d[128];
    u32 op_words_d[128];
    u32 insn_count;
    u32 exit_words;
} BlockStatsDelta;

static inline void block_stats_reset(BlockStatsDelta *d) {
    d->op_mask_lo = 0;
    d->op_mask_hi = 0;
    d->insn_count = 0;
    d->exit_words = 0;
}

static inline void block_stats_touch_op(BlockStatsDelta *d, u8 op) {
    u32 o = op & 127u;
    if (o < 64u) {
        u64 bit = 1ULL << o;
        if (!(d->op_mask_lo & bit)) {
            d->op_mask_lo |= bit;
            d->op_insns_d[o] = 0;
            d->op_words_d[o] = 0;
        }
    } else {
        u64 bit = 1ULL << (o - 64u);
        if (!(d->op_mask_hi & bit)) {
            d->op_mask_hi |= bit;
            d->op_insns_d[o] = 0;
            d->op_words_d[o] = 0;
        }
    }
}

static inline void block_stats_add_insn(BlockStatsDelta *d, u8 op) {
    block_stats_touch_op(d, op);
    d->op_insns_d[op & 127u]++;
    d->insn_count++;
}

static inline void block_stats_add_words(BlockStatsDelta *d, u8 op, u32 words) {
    block_stats_touch_op(d, op);
    d->op_words_d[op & 127u] += words;
}

static void block_stats_merge(const BlockStatsDelta *d) {
    stats.insns += d->insn_count;
    stats.exit_words += d->exit_words;
    u64 m = d->op_mask_lo;
    while (m) {
        u32 o = (u32)__builtin_ctzll(m);
        m &= m - 1ULL;
        stats.op_insns[o] += d->op_insns_d[o];
        stats.op_words[o] += d->op_words_d[o];
    }
    m = d->op_mask_hi;
    while (m) {
        u32 o = 64u + (u32)__builtin_ctzll(m);
        m &= m - 1ULL;
        stats.op_insns[o] += d->op_insns_d[o];
        stats.op_words[o] += d->op_words_d[o];
    }
}

typedef struct {
    // Captured before parallel prepare
    int view, key, cpl;
    u64 cr3, rip, page;
    bool compat;
    u64 prep_ticks;
    u64 fgen_flush;
    u64 pt_gen;
    u64 fgen0, fgen1;
    bool have_gpa0;
    u64 frame0, frame1;

    // Produced unlocked in tr_translate_prepare()
    ParResKind res_kind;
    u64 fault_or_unprot;
    u32 code_words;
    u32 pcm_count;
    u32 stub_cnt;
    u32 link_cnt;
    bool par_overflow;

    // Guest read-bytes snapshot for remember_reads()
    u16 snap_off, snap_len;
    u8 snap_bytes[4096];

    // AOT / Tier-2 metadata
    bool aot_have_page_hash;
    u64 aot_page_hash;
    bool aot_record_eligible;
    bool t2_has_loop;
    u32 t2_patch_off; // word offset relative to start, or ~0U
    u16 aot_t2_patch_off, aot_t2_head_off, aot_t2_min_off, aot_t2_span_bytes;
    u64 t2_head_rip;

    BlockStatsDelta sdelta;

    // Post-commit state executed unlocked in tr_translate_post()
    bool need_post;
    bool post_is_aot_hit;
    u32 pub_idx, pub_part, pub_slot;
    u32 pub_entry_word, pub_arm_words, pub_t2_patch_word;
    u16 pub_read_off, pub_read_bytes;
    u64 pub_t2_epoch;

    // Per-vCPU scratch buffers (heap-allocated once per thread)
    u32 code[PAR_CODE_CAP];
    PcMap pcm[PAR_PCM_CAP];
    Insn list[MAX_INSNS];
    ParStubSite stubs[PAR_MAX_STUBS];
    ParLinkSite links[PAR_MAX_LINKS];
    AotCapture aot_cap;
} ParScratch;

static _Thread_local ParScratch *tls_par;

static ParScratch *par_scratch_get(void) {
    if (__builtin_expect(!tls_par, 0))
        tls_par = (ParScratch *)calloc(1, sizeof(ParScratch));
    return tls_par;
}

void tr_par_record_stub(u32 word_off, AotStubId id, bool is_bl) {
    ParScratch *ps = tls_par;
    if (!ps || ps->stub_cnt >= PAR_MAX_STUBS) {
        if (ps)
            ps->par_overflow = true;
        return;
    }
    ps->stubs[ps->stub_cnt++] = (ParStubSite){ word_off, (u8)id, is_bl ? 1u : 0u };
}

void tr_par_record_cc_branch(u32 word_off, AotCcBranchKind kind, u32 lit_off) {
    ParScratch *ps = tls_par;
    if (!ps || ps->link_cnt >= PAR_MAX_LINKS) {
        if (ps)
            ps->par_overflow = true;
        return;
    }
    ps->links[ps->link_cnt++] = (ParLinkSite){ word_off, lit_off, (u8)kind };
}

bool tr_par_enabled(void) {
    return par_translate_enabled && !cfg.profile_compile_inputs;
}

typedef struct {
    int count;
    int t2_head_i;
    u32 t2_patch_off; // relative to block start, or ~0U
    u32 pcm_written;
    u16 aot_t2_patch_off, aot_t2_head_off, aot_t2_min_off, aot_t2_span_bytes;
    u64 t2_head_rip;
    bool overflow;
} ColdEmitOut;

// Shared cold emission engine: writes ARM64 into `start` (either live cc_base or thread-local ParScratch.code)
// and PcMap into `pcm_dst[0..pcm_cap)`. Zero shared mutable global writes when `tr_par_active == true`.
static void emit_cold_block(int key, u64 rip, u64 page, int cpl, bool single, u64 frame0, u64 frame1,
                            Insn *list, int count, u32 *start, u32 code_cap,
                            PcMap *pcm_dst, u32 pcm_cap, BlockStatsDelta *sdelta, ColdEmitOut *out) {
    memset(out, 0, sizeof *out);
    out->t2_head_i = -1;
    out->t2_patch_off = ~0U;
    out->aot_t2_patch_off = AOT_NO_PATCH_WORD;
    block_stats_reset(sdelta);

    tr_cc = start;
    Insn *last = &list[count - 1];
    bool fallthrough_end = !(last->op == OP_JMP || last->op == OP_JMPI || last->op == OP_RET || last->op == OP_BAD ||
                             last->op == OP_UD || last->op == OP_IRET);
    block_insns = list;
    insn_total = count;
    pend_count = 0;
    fwd_count = 0;
    tr_ad_page = ~0ULL;
    tr_aes_mask = 0;
    if (tr_aot_active && !single && frame1 == ~0ULL && !cfg.profile_compile_inputs) {
        tr_aot_block_start = start;
        aot_capture_begin(&tr_aot_cap);
    } else {
        tr_aot_cap.active = false;
    }
    if (native_dag && !single && count > 0 && count < MAX_INSNS && frame1 == ~0ULL) {
        expand_forward_dag(list, &count, key, rip, page);
        insn_total = count;
        last = &list[count - 1];
        fallthrough_end = !is_uncond_term_op(last->op);
    }
    out->count = count;
    cur_key = key;
    cur_entry_rip = rip;
    cur_page = page;
    cur_single = single;
    l1_miss_count = 0;
    tr_zext8 = tr_zext16 = 0;
    ext_ymm_cache_reset();

    int tgt_idx[MAX_INSNS], fall_idx[MAX_INSNS];
    u8 ext_tgt_live[MAX_INSNS], ext_fall_live[MAX_INSNS];
    bool is_target[MAX_INSNS];
    memset(is_target, 0, (size_t)count * sizeof(bool));
    for (int i = 0; i < count; i++) {
        const Insn *in = &list[i];
        tgt_idx[i] = -1;
        fall_idx[i] = -1;
        ext_tgt_live[i] = 0;
        ext_fall_live[i] = 0;
        if (in->op == OP_JCC || in->op == OP_JRCXZ || in->op == OP_JMP) {
            int ti = idx_of_rip(list, count, in->target);
            tgt_idx[i] = ti;
            if (ti >= 0)
                is_target[ti] = true;
            else
                ext_tgt_live[i] = (u8)flags_needed_at(in->target, page, in->op == OP_JMP ? 3 : 2, list, count);
        } else if (in->op == OP_CALL) {
            if (!tr_precise_flags && (in->target & ~0xfffULL) == page && ((in->target + 15) & ~0xfffULL) == page)
                ext_tgt_live[i] = (u8)flags_needed_at(in->target, page, 2, list, count);
            else
                ext_tgt_live[i] = F_ARITH;
        }
        if (!is_uncond_term_op(in->op)) {
            if (i + 1 < count && list[i + 1].rip == in->next) {
                fall_idx[i] = i + 1;
            } else {
                int fi = idx_of_rip(list, count, in->next);
                fall_idx[i] = fi;
                if (fi >= 0)
                    is_target[fi] = true;
                else
                    ext_fall_live[i] = (u8)flags_needed_at(in->next, page, 3, list, count);
            }
        }
        if (i > 0 && (is_uncond_term_op(list[i - 1].op) || list[i - 1].next != in->rip))
            is_target[i] = true;
    }

    u8 live_after[MAX_INSNS];
    u8 live_before[MAX_INSNS];
    if (cfg.all_flags_live) {
        memset(live_after, F_ARITH, (size_t)count);
        memset(live_before, F_ARITH, (size_t)count);
    } else {
        memset(live_after, 0, (size_t)count);
        memset(live_before, 0, (size_t)count);
        bool changed = true;
        for (int pass = 0; pass < 8 && changed; pass++) {
            changed = false;
            for (int i = count - 1; i >= 0; i--) {
                const Insn *in = &list[i];
                u8 out_f;
                if (in->op == OP_JMP) {
                    out_f = tgt_idx[i] >= 0 ? live_before[tgt_idx[i]] : ext_tgt_live[i];
                } else if (is_uncond_term_op(in->op) || in->op == OP_CALLI) {
                    out_f = F_ARITH;
                } else if (in->op == OP_CALL) {
                    out_f = ext_tgt_live[i];
                } else if (in->op == OP_JCC || in->op == OP_JRCXZ) {
                    u8 f = fall_idx[i] >= 0 ? live_before[fall_idx[i]] : ext_fall_live[i];
                    u8 t = tgt_idx[i] >= 0 ? live_before[tgt_idx[i]] : ext_tgt_live[i];
                    out_f = f | t;
                } else {
                    out_f = fall_idx[i] >= 0 ? live_before[fall_idx[i]] : ext_fall_live[i];
                }
                u8 lb = (u8)((out_f & ~in->writes) | insn_reads(in));
                if (out_f != live_after[i] || lb != live_before[i]) {
                    live_after[i] = out_f;
                    live_before[i] = lb;
                    changed = true;
                }
            }
        }
        if (changed) {
            memset(live_after, F_ARITH, (size_t)count);
            memset(live_before, F_ARITH, (size_t)count);
        }
    }

    int t2_head_i = (!single && (cpl == 3 || t2_eager_mode()) &&
                     frame1 == ~0ULL && frame0 != ~0ULL && frame0 < cfg.ram_size &&
                     !tr_precise_flags && !cfg.all_flags_live && !cfg.profile_compile_inputs)
                        ? t2_find_clean_loop(list, count, tgt_idx)
                        : -1;
    out->t2_head_i = t2_head_i;
    u32 t2_patch_off = ~0U;
    if (ctx_ptr->vcpu_count > 1 || t2_head_i == 0) // a kill or Tier-2 OSR patches the entry NOP atomically
        E(WORD_NOP);
    if (t2_head_i == 0)
        t2_patch_off = 0U;

    u32 pcm_n = 0;
    for (int i = 0; i < count; i++) {
        if (code_cap && (u32)(tr_cc - start) + 1024U >= code_cap) {
            out->overflow = true;
            return;
        }
        const Insn *in = &list[i];
        bool prev_fell_through = i > 0 && !is_uncond_term_op(list[i - 1].op) && list[i - 1].next == in->rip;
        if (is_target[i]) {
            if (prev_fell_through) {
                if (tr_aes_mask)
                    sse_aes_flush_all();
                if (ext_ymm_cache_has_dirty())
                    ext_ymm_cache_flush_dirty();
            }
            tr_ad_page = ~0ULL;
            tr_zext8 = tr_zext16 = 0;
            ext_ymm_cache_reset();
        }
        if (ext_ymm_cache_has_dirty() &&
            (in->op != OP_EXT || in->memop || is_target[i] || ext_ymm_cache_dirty_run() >= 8)) {
            ext_ymm_cache_flush_dirty();
        }
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR ||
            in->op == OP_X87 || (in->op == OP_POPCNT && !tr_hybrid)) {
            ext_ymm_cache_flush_and_drop();
        }
        if ((!tr_opt_peep && in->op != OP_SSE && in->op != OP_EXT) ||
            (tr_opt_peep && !ad_page_safe_op(in->op))) {
            tr_ad_page = ~0ULL;
        }
        if (in->op != OP_SSE && tr_aes_mask)
            sse_aes_flush_all();
        if (i == 0 && t2_head_i == 0) {
            insn_arm[0] = start;
        } else {
            insn_arm[i] = tr_cc;
        }
        if (i)
            block_stats_add_words(sdelta, list[i - 1].op, (u32)(tr_cc - insn_arm[i - 1]));
        block_stats_add_insn(sdelta, in->op);
        if (!ext_ymm_cache_has_dirty()) {
            if (pcm_cap && pcm_n >= pcm_cap) {
                out->overflow = true;
                return;
            }
            pcm_dst[pcm_n].arm_word = (u16)(tr_cc - start);
            pcm_dst[pcm_n].rip_off = (u16)(in->rip - rip);
            pcm_n++;
        }
        if (i > 0 && i == t2_head_i) {
            t2_patch_off = (u32)(tr_cc - start);
            E(WORD_NOP);
        }

        // P1-A multi-instruction peephole fusions (adjacent non-target pairs).
        bool fused = false;
        if (tr_opt_peep && !single && i + 1 < count && !is_target[i + 1] && in->next == list[i + 1].rip) {
            const Insn *nx = &list[i + 1];
            // 1) TEST r, r / CMP r, 0 + JE/JNE -> CBZ/CBNZ (flags dead after Jcc).
            if (in->op == OP_ALU && nx->op == OP_JCC && (nx->cc == 4 || nx->cc == 5) && live_after[i + 1] == 0) {
                int cbz_reg = -1, cbz_sf = 0;
                if (in->sub == ALU_TEST && !in->has_imm && in->rm_reg && !in->high_reg && !in->high_rm && in->reg == in->rm) {
                    if (in->sz == 4 || in->sz == 8) {
                        cbz_reg = in->rm;
                        cbz_sf = in->sz == 8;
                    } else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    } else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    }
                } else if (in->sub == ALU_CMP && in->has_imm && in->imm == 0 && in->rm_reg && !in->high_rm) {
                    if (in->sz == 4 || in->sz == 8) {
                        cbz_reg = in->rm;
                        cbz_sf = in->sz == 8;
                    } else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    } else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    }
                }
                if (cbz_reg >= 0) {
                    u32 *back = internal_target(nx->target, i);
                    int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                    u32 *direct = (back || fwd >= 0) ? NULL : try_direct_link_target(cur_key, cur_entry_rip, cur_page, nx->target, cur_single);
                    i64 dd = direct ? (direct - tr_cc) : 0;
                    if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                        direct = NULL;
                    if (back) {
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, (int)(back - tr_cc));
                    } else if (fwd >= 0) {
                        add_fwd(tr_cc, fwd, 0);
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    } else if (direct) {
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, (int)dd);
                    } else {
                        add_pending(tr_cc, nx->target, SVC_JCC, 0);
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    block_stats_add_insn(sdelta, nx->op);
                    i++;
                    fused = true;
                } else if (in->sub == ALU_TEST && in->has_imm && (in->rm_reg || !tr_precise_flags)) {
                    // 2) TEST r/m, #(1<<k) + JE/JNE -> TBZ/TBNZ (STRICTLY internal targets only; 14-bit range!).
                    u64 m = in->sz == 1 ? (u8)in->imm : in->sz == 2 ? (u16)in->imm : in->sz == 4 ? (u32)in->imm : (u64)in->imm;
                    if (m != 0 && (m & (m - 1)) == 0) {
                        u32 *back = internal_target(nx->target, i);
                        int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                        bool back_ok = back && (back - tr_cc) >= -(1 << 13) + 16 && (back - tr_cc) < (1 << 13) - 16;
                        bool fwd_ok = false;
                        if (fwd >= 0 && (fwd - (i + 1)) <= 48 && fwd_count < MAX_FWD_FIX) {
                            fwd_ok = true;
                            for (int k = i + 2; k < fwd; k++) {
                                u8 kop = list[k].op;
                                if (kop == OP_EXT || kop == OP_SSE || kop == OP_X87 || kop == OP_STRING) {
                                    fwd_ok = false;
                                    break;
                                }
                            }
                        }
                        if (back_ok || fwd_ok) {
                            int bit = __builtin_ctzll(m);
                            int treg;
                            if (in->rm_reg) {
                                treg = gpr_phys(in->rm, in->high_rm);
                                if (in->high_rm)
                                    bit += 8;
                            } else {
                                ldst(1, in->sz, S0, addr_prepare(in, in->sz));
                                treg = S0;
                            }
                            if (back_ok) {
                                tbz(nx->cc == 5, treg, bit, (int)(back - tr_cc));
                            } else {
                                add_fwd(tr_cc, fwd, 2);
                                tbz(nx->cc == 5, treg, bit, 0);
                            }
                            insn_arm[i + 1] = insn_arm[i];
                            block_stats_add_insn(sdelta, nx->op);
                            i++;
                            fused = true;
                        }
                    }
                }
            } else if (in->op == OP_SETCC && in->rm_reg && !in->high_rm &&
                       nx->op == OP_MOVZX && nx->rm_reg && !nx->high_rm &&
                       nx->srcsz == 1 && nx->sz >= 4 && nx->rm == in->rm && nx->reg == in->rm) {
                // 2b) SETcc r8; MOVZX r32/64, r8 (same reg) -> single CSET (1 insn vs 3).
                int acc = x86_to_arm_cc[in->cc];
                if (acc < 0) {
                    emit_parity(nx->reg);
                    if (in->cc == 11)
                        E(0x52000000u | (u32)nx->reg << 5 | (u32)nx->reg); // eor wd, wd, #1
                } else {
                    cset(nx->reg, acc);
                }
                u16 bit = (u16)(1u << nx->reg);
                tr_zext8 |= bit;
                tr_zext16 |= bit;
                insn_arm[i + 1] = insn_arm[i];
                block_stats_add_insn(sdelta, nx->op);
                i++;
                fused = true;
            } else if (in->op == OP_BT && in->sub == 0 && in->has_imm && in->rm_reg && in->sz >= 4 &&
                       nx->op == OP_JCC && (nx->cc == 2 || nx->cc == 3) && live_after[i + 1] == 0) {
                // 2c) BT r32/64, imm; JC/JNC -> TBNZ/TBZ (internal target within 14-bit range).
                u32 *back = internal_target(nx->target, i);
                int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                bool back_ok = back && (back - tr_cc) >= -(1 << 13) + 16 && (back - tr_cc) < (1 << 13) - 16;
                bool fwd_ok = false;
                if (fwd >= 0 && (fwd - (i + 1)) <= 48 && fwd_count < MAX_FWD_FIX) {
                    fwd_ok = true;
                    for (int k = i + 2; k < fwd; k++) {
                        u8 kop = list[k].op;
                        if (kop == OP_EXT || kop == OP_SSE || kop == OP_X87 || kop == OP_STRING) {
                            fwd_ok = false;
                            break;
                        }
                    }
                }
                if (back_ok || fwd_ok) {
                    int bit = (int)((u64)in->imm & (u64)(in->sz * 8 - 1));
                    int nz = nx->cc == 2 ? 1 : 0; // cc==2 is JB/JC (CF==1 -> bit==1 -> TBNZ)
                    if (back_ok) {
                        tbz(nz, in->rm, bit, (int)(back - tr_cc));
                    } else {
                        add_fwd(tr_cc, fwd, 2);
                        tbz(nz, in->rm, bit, 0);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    block_stats_add_insn(sdelta, nx->op);
                    i++;
                    fused = true;
                }
            } else if (!tr_compat && !tr_precise_flags) {
                // 3) PUSH r1; PUSH r2 -> stp r2, r1, [x4, #-16]!
                if (in->op == OP_PUSH && in->rm_reg && !in->has_imm && !in->sub && in->sz == 8 && in->rm != 4 &&
                    nx->op == OP_PUSH && nx->rm_reg && !nx->has_imm && !nx->sub && nx->sz == 8 && nx->rm != 4) {
                    E(0xA9800000u | ((u32)-2 & 0x7fu) << 15 | (u32)in->rm << 10 | 4u << 5 | (u32)nx->rm);
                    tr_zext8 &= (u16)~(1u << 4);
                    tr_zext16 &= (u16)~(1u << 4);
                    insn_arm[i + 1] = insn_arm[i];
                    block_stats_add_insn(sdelta, nx->op);
                    i++;
                    fused = true;
                } else if (in->op == OP_POP && in->rm_reg && !in->sub && in->sz == 8 && in->rm != 4 &&
                           nx->op == OP_POP && nx->rm_reg && !nx->sub && nx->sz == 8 && nx->rm != 4 &&
                           in->rm != nx->rm) {
                    // 4) POP r1; POP r2 -> ldp r1, r2, [x4], #16 (r1 != r2 required by ARM64)
                    E(0xA8C00000u | (2u << 15) | (u32)nx->rm << 10 | 4u << 5 | (u32)in->rm);
                    u16 clr = (u16)((1u << 4) | (1u << in->rm) | (1u << nx->rm));
                    tr_zext8 &= (u16)~clr;
                    tr_zext16 &= (u16)~clr;
                    insn_arm[i + 1] = insn_arm[i];
                    block_stats_add_insn(sdelta, nx->op);
                    i++;
                    fused = true;
                } else if (in->op == OP_POP && in->rm_reg && !in->sub && in->sz == 8 && in->rm != 4 &&
                           nx->op == OP_RET && nx->imm == 0) {
                    // 5) POP r1; RET -> ldp r1, x16, [x4], #16 + 4-insn RAS return
                    bool use_l1_ret = use_native_l1(tr_cpl, tr_compat);
                    u64 ret_miss_va = use_l1_ret
                                          ? STUB(PRIV_VA + OFF_STUBS + STUB_OFF_RET_MISS_L1)
                                          : STUB(tr_stub_ret_miss);
                    AotStubId ret_sid = use_l1_ret ? AOT_STUB_RET_MISS_L1 : AOT_STUB_RET_MISS;
                    E(0xA8C00000u | (2u << 15) | (u32)S0 << 10 | 4u << 5 | (u32)in->rm);
                    if (cfg.no_return_prediction) {
                        tr_aot_note_stub(tr_cc, ret_sid, false);
                        b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
                    } else {
                        E(0xA8C00000u | 2 << 15 | LR << 10 | RS << 5 | S1); // ldp x17, x30, [x26], #16
                        logic_reg(1, 2, 0, S1, S1, S0);
                        cbz(1, 1, S1, 2);
                        ret_reg(LR);
                        tr_aot_note_stub(tr_cc, ret_sid, false);
                        b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
                    }
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    ext_ymm_cache_reset();
                    insn_arm[i + 1] = insn_arm[i];
                    block_stats_add_insn(sdelta, nx->op);
                    i++;
                    fused = true;
                } else if (in->op == OP_MOV && nx->op == OP_MOV &&
                           !in->has_imm && !nx->has_imm && !in->rm_reg && !nx->rm_reg &&
                           in->sz == nx->sz && (in->sz == 4 || in->sz == 8) &&
                           in->to_reg == nx->to_reg &&
                           !in->riprel && !nx->riprel && !in->no_base && !nx->no_base &&
                           in->no_index && nx->no_index && in->seg == 0 && nx->seg == 0 &&
                           !in->addr32 && !nx->addr32 && !in->addr16 && !nx->addr16 &&
                           in->base == nx->base &&
                           (tr_cpl == 3 || in->base == 4 || in->base == 5)) {
                    // 6) Adjacent MOV [base+d], r1; MOV [base+d+sz], r2 -> STP/LDP (Win64 prologues/epilogues & struct spills).
                    int sz = in->sz;
                    bool fwd_pair = (nx->disp == in->disp + sz);
                    bool rev_pair = (in->disp == nx->disp + sz);
                    if (fwd_pair || rev_pair) {
                        i64 lo = fwd_pair ? in->disp : nx->disp;
                        int rt1 = fwd_pair ? in->reg : nx->reg;
                        int rt2 = fwd_pair ? nx->reg : in->reg;
                        bool load = in->to_reg;
                        bool ldp_ok = !load || (rt1 != rt2 && in->reg != in->base && nx->reg != in->base);
                        if (ldp_ok && (lo & (sz - 1)) == 0 && lo >= -64 * sz && lo <= 63 * sz) {
                            u32 imm7 = (u32)(lo / sz) & 0x7fu;
                            u32 base_opc = sz == 8 ? (load ? 0xA9400000u : 0xA9000000u)
                                                   : (load ? 0x29400000u : 0x29000000u);
                            E(base_opc | imm7 << 15 | (u32)rt2 << 10 | (u32)in->base << 5 | (u32)rt1);
                            if (load) {
                                u16 clr = (u16)((1u << rt1) | (1u << rt2));
                                tr_zext8 &= (u16)~clr;
                                tr_zext16 &= (u16)~clr;
                            }
                            insn_arm[i + 1] = insn_arm[i];
                            block_stats_add_insn(sdelta, nx->op);
                            i++;
                            fused = true;
                        }
                    }
                }
            }
        }

        if (!fused) {
            switch (in->op) {
                case OP_JCC:
                case OP_JRCXZ:
                case OP_JMP:
                case OP_CALL:
                case OP_CALLI:
                case OP_JMPI:
                case OP_RET:
                    emit_control(in, i, live_after[i]);
                    if (in->op == OP_CALL || in->op == OP_CALLI || in->op == OP_JMPI || in->op == OP_RET) {
                        ext_ymm_cache_reset();
                        tr_ad_page = ~0ULL;
                        tr_zext8 = tr_zext16 = 0;
                    } else {
                        update_zext(in);
                    }
                    break;
                case OP_BAD:
                    emit_stop(in->rip, in->why);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_UD:
                    svc(SVC_UD);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_SSE:
                case OP_LDMXCSR:
                case OP_STMXCSR:
                    emit_sse(in, live_after[i]);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_X87:
                    emit_x87(in);
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_EXT:
                    emit_ext(in, live_after[i]);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_SYSCALL:
                case OP_SYSRET:
                case OP_IRET:
                case OP_INT:
                case OP_INT3:
                case OP_HLT:
                case OP_CPUID:
                case OP_RDTSC:
                case OP_RDTSCP:
                case OP_RDMSR:
                case OP_WRMSR:
                case OP_CLI:
                case OP_STI:
                case OP_MOV_CR:
                    if (i && in->reg == 3 && !in->to_reg && in->rm == 0 && !cfg.keep_kernel_tables && tr_cpl == 0) {
                        const Insn *p = &list[i - 1];
                        if (p->op == OP_MOV && p->seg == 2 && p->to_reg && p->reg == 0 && !p->rm_reg) {
                            if (p->disp == 0x138) {
                                tr_zext8 = tr_zext16 = 0;
                                break;
                            }
                            if (p->disp == 0x140) {
                                Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, cr3)};
                                ldst(1, 8, S0, a);
                                logic_sh(1, 2, 0, S0, S0, 0, 0, 0); // eor with rax
                                ubfx(1, S0, S0, 12, 40); // the tables, whatever the PCID
                                u32 *same = tr_cc;
                                cbz(0, 1, S0, 0);
                                emit_monitor_op(in);
                                *same |= ((u32)(tr_cc - same) & 0x7ffff) << 5;
                                ext_ymm_cache_reset();
                                tr_ad_page = ~0ULL;
                                tr_zext8 = tr_zext16 = 0;
                                break;
                            }
                        }
                    }
                    emit_monitor_op(in);
                    ext_ymm_cache_reset();
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_MOV_DR:
                case OP_INVLPG:
                case OP_INVPCID:
                case OP_WBINVD:
                case OP_CLTS:
                case OP_LGDT:
                case OP_LTR:
                case OP_MOV_SEG:
                case OP_IN:
                case OP_OUT:
                case OP_SWAPGS:
                case OP_SYSENTER:
                case OP_SYSEXIT:
                case OP_FAR:
                case OP_FXSAVE:
                case OP_FXRSTOR:
                    emit_monitor_op(in);
                    ext_ymm_cache_reset();
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                default:
                    emit_int(in, live_after[i]);
                    update_zext(in);
                    break;
            }
        }

        // Non-adjacent fallthrough inside Forward-DAG: emit exit/branch before next disconnected sub-block.
        const Insn *end_in = &list[i];
        if (i + 1 < count && !is_uncond_term_op(end_in->op) && end_in->next != list[i + 1].rip) {
            if (tr_aes_mask)
                sse_aes_flush_all();
            if (ext_ymm_cache_has_dirty())
                ext_ymm_cache_flush_dirty();
            u32 *back = internal_target(end_in->next, i);
            int fwd = back ? -1 : forward_target_idx(end_in->next, i);
            if (back) {
                b_rel((int)(back - tr_cc));
            } else if (fwd >= 0) {
                add_fwd(tr_cc, fwd, 1);
                b_rel(0);
            } else {
                u32 *direct = (!is_monitor_or_stop_op(end_in->op) && end_in->op != OP_POPF)
                                  ? try_direct_link_target(cur_key, cur_entry_rip, cur_page, end_in->next, cur_single)
                                  : NULL;
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
                    b_rel((int)dd);
                } else {
                    u32 *svc_site = tr_cc;
                    svc(SVC_DIRECT);
                    tr_aot_note_cc_branch(svc_site, AOT_CC_DIRECT, tr_cc);
                    stub_lit(end_in->next);
                }
            }
        }
    }
    if (tr_aes_mask)
        sse_aes_flush_all();
    if (ext_ymm_cache_has_dirty())
        ext_ymm_cache_flush_dirty();
    for (int i = 0; i < fwd_count; i++)
        fix(fwd_fix[i].site, fwd_fix[i].form, insn_arm[fwd_fix[i].target_idx]);
    u32 *tail = tr_cc;
    block_stats_add_words(sdelta, last->op, (u32)(tr_cc - insn_arm[count - 1]));
    if (fallthrough_end) {
        u32 *direct = (!is_monitor_or_stop_op(last->op) && last->op != OP_POPF)
                          ? try_direct_link_target(cur_key, cur_entry_rip, cur_page, last->next, cur_single)
                          : NULL;
        i64 dd = direct ? (direct - tr_cc) : 0;
        if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
            b_rel((int)dd);
        } else {
            u32 *svc_site = tr_cc;
            svc(SVC_DIRECT);
            tr_aot_note_cc_branch(svc_site, AOT_CC_DIRECT, tr_cc);
            stub_lit(last->next);
        }
    }
    for (int i = 0; i < pend_count; i++) {
        if (code_cap && (u32)(tr_cc - start) + 16U >= code_cap) {
            out->overflow = true;
            return;
        }
        u32 *stub = tr_cc;
        svc(pend[i].kind);
        u32 *lit_site = tr_cc;
        stub_lit(pend[i].target);
        if (pend[i].kind == SVC_JCC) {
            E((u32)(stub - pend[i].site));
            tr_aot_note_cc_branch(pend[i].site, AOT_CC_JCC, lit_site);
        } else if (pend[i].kind == SVC_CALL) {
            tr_aot_note_cc_branch(pend[i].site, AOT_CC_CALL, lit_site);
        }
        fix(pend[i].site, pend[i].form, stub);
    }
    for (int i = 0; i < l1_miss_count; i++) {
        if (code_cap && (u32)(tr_cc - start) + 16U >= code_cap) {
            out->overflow = true;
            return;
        }
        u32 *miss = tr_cc;
        tr_aot_note_stub(tr_cc, AOT_STUB_LOOKUP_L1, true);
        bl_rel((int)((i64)(STUB(PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1) - tr_cc_va(tr_cc)) / 4));
        if (l1_miss[i].is_call)
            b_rel((int)(l1_miss[i].blr_site - tr_cc));
        else
            br(S2);
        fix(l1_miss[i].site1, 0, miss);
        fix(l1_miss[i].site2, 0, miss);
    }
    sdelta->exit_words = (u32)(tr_cc - tail);
    out->pcm_written = pcm_n;
    out->t2_patch_off = t2_patch_off;

    if (t2_head_i >= 0 && t2_patch_off != ~0U) {
        int t2_tail_i = t2_head_i;
        for (int k = t2_head_i; k < count; k++) {
            u8 op = list[k].op;
            if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) && tgt_idx[k] == t2_head_i)
                t2_tail_i = k;
        }
        u64 lo = list[t2_head_i].rip, hi = list[t2_head_i].next;
        for (int k = t2_head_i; k <= t2_tail_i; k++) {
            if (list[k].rip < lo) lo = list[k].rip;
            if (list[k].next > hi) hi = list[k].next;
        }
        // Include up to 64 post-loop same-page bytes in the SMC-verified snapshot so Tier-2 can peek exit flag liveness!
        u64 snap_hi = hi + 64 <= page + 4096 ? hi + 64 : page + 4096;
        if ((lo & ~0xfffULL) == page && hi <= page + 4096 && hi > lo) {
            out->t2_head_rip = list[t2_head_i].rip;
            out->aot_t2_min_off = (u16)(lo & 0xfffULL);
            out->aot_t2_span_bytes = (u16)(snap_hi - lo);
            if (t2_patch_off < AOT_NO_PATCH_WORD) {
                out->aot_t2_patch_off = (u16)t2_patch_off;
                out->aot_t2_head_off = (u16)(list[t2_head_i].rip - rip);
            }
        }
    }
}

static void tr_aot_promote_t2_post(u32 idx, u32 part, u32 slot, u64 t2_ep, u32 key, u8 cpl, bool compat,
                                   u32 entry_word, u32 arm_words, u32 t2_patch_word,
                                   u64 rip, u64 head_rip, u64 frame0, u64 aot_page_hash,
                                   u16 t2_min_off, u16 t2_span_bytes) {
    bool promoted = false;
    if (tr_aot_active && aot_cache && t2_enabled() && t2_span_bytes > 0 &&
        (u32)t2_min_off + t2_span_bytes <= 4096U && frame0 < cfg.ram_size) {
        AotKey t2key = {
            .page_hash = aot_page_hash,
            .abi_hash = aot_abi_hash,
            .mode_bits = aot_build_mode_bits((int)cpl, compat, true),
            .page_offset = (u16)(head_rip & 0xFFFULL),
            .flags = 0,
        };
        AotBlockView t2view;
        if (aot_lookup(aot_cache, &t2key, &t2view) && t2view.code_words <= 8192U && t2view.pcm_count <= 1024U) {
            u32 t2_buf[8192];
            PcMap t2_pcm[1024];
            u32 bail_sites[32];
            u32 bail_cnt = 0;
            bool bail_ok = true;
            bool has_stub = false;
            for (u16 ri = 0; ri < t2view.reloc_count; ri++) {
                if (t2view.relocs[ri].type == AOT_RELOC_T2_BAILOUT) {
                    if (bail_cnt >= 32U) { bail_ok = false; break; }
                    bail_sites[bail_cnt++] = t2view.relocs[ri].word_off;
                } else if (t2view.relocs[ri].type == AOT_RELOC_STUB_BRANCH) {
                    has_stub = true;
                }
            }
            if (bail_ok && !has_stub) {
                u64 t2_base_va = tr_cc_va(cc_base + (part ? CC_T2_WORD1 : CC_T2_WORD0));
                if (aot_materialize(aot_cache, &t2view, head_rip, t2_buf, 8192U,
                                    t2_pcm, 1024U, t2_base_va, (int)part,
                                    &aot_stubs, NULL, NULL)) {
                    promoted = tr_t2_publish(idx, part, slot, t2_ep, head_rip,
                                             key, cpl, compat, t2_patch_word,
                                             frame0, t2_min_off, t2_span_bytes,
                                             ram_at(frame0 + t2_min_off),
                                             t2_buf, t2view.code_words, t2_pcm, t2view.pcm_count,
                                             t2view.has_t2_meta ? &t2view.t2_meta : NULL,
                                             bail_cnt ? bail_sites : NULL, bail_cnt);
                }
            }
        }
    }
    if (!promoted && t2_enabled() && t2_span_bytes > 0 && (u32)t2_min_off + t2_span_bytes <= 4096U &&
        !__atomic_load_n(&hblocks[idx].dead, __ATOMIC_ACQUIRE)) {
        t2_register_loop(idx, part, slot, entry_word, arm_words,
                         t2_patch_word, rip, head_rip, frame0,
                         t2_min_off, t2_span_bytes, key, cpl, compat, t2_ep);
        if (t2_eager_mode())
            t2_eager_promote_now(idx);
    }
}

static TrResult do_translate(int key, int view, u64 cr3, int cpl, u64 rip, bool single);

// 1-lock parallel fast entry: runs 100% UNLOCKED before acquiring translatorLock!
// Returns true iff a cold/AOT block was prepared in TLS and is ready for a single-lock tr_translate_commit().
// Returns false if likely a block_map fast-hit, dead-block reuse candidate, unprotected frame, or flush-needed.
bool tr_translate_try_prepare_unlocked(int view, uint64_t cr3, int cpl, bool compat, uint64_t rip) {
    if (!par_translate_enabled || cfg.profile_compile_inputs)
        return false;

    tr_compat = compat;
    tr_cpl = cpl;
    walk_cr3 = cr3;
    walk_cpl = cpl;
    fetch_cache_valid = false;
    __atomic_store_n(&tr_fpcr_merges_scalar, ctx_ptr->fpcr_merges_scalar != 0, __ATOMIC_RELAXED);

    int key = block_key(view, cr3, cpl, rip);
    // Advisory lock-free hit check (78% of Win2022 HC_TRANSLATE): skip prepare and let locked fast path return in ~150 ns.
    if (map_find_hint(key, rip))
        return false;
    // Advisory dead-block reuse hint: if a dead candidate exists at this exact slot, let locked fast path revive it.
    if (!cfg.no_reuse) {
        u32 it = __atomic_load_n(&dead_slots[dead_slot((u32)key, rip)], __ATOMIC_RELAXED);
        if (it && __atomic_load_n(&hblocks[it - 1].rip, __ATOMIC_RELAXED) == rip)
            return false;
    }

    u64 frame;
    if (frame_unprotected(rip, &frame))
        return false;

    ParScratch *ps = par_scratch_get();
    if (!ps)
        return false;

    ps->prep_ticks = 0;
    ps->view = view;
    ps->key = key;
    ps->cpl = cpl;
    ps->cr3 = cr3;
    ps->rip = rip;
    ps->page = rip & ~0xfffULL;
    ps->compat = compat;
    ps->fgen_flush = __atomic_load_n(&tr_flush_gen, __ATOMIC_ACQUIRE);
    ps->pt_gen = __atomic_load_n(&tr_pt_inval_gen, __ATOMIC_ACQUIRE);
    u64 gpa0 = 0;
    ps->have_gpa0 = fetch_walk(rip, &gpa0);
    ps->frame0 = ps->have_gpa0 ? (gpa0 & ~0xfffULL) : ~0ULL;
    ps->frame1 = ~0ULL;
    ps->fgen0 = (ps->frame0 != ~0ULL)
                    ? __atomic_load_n(&tr_frame_inval_gen[(ps->frame0 >> 12) & (PAR_GEN_BUCKETS - 1U)], __ATOMIC_ACQUIRE)
                    : 0ULL;
    ps->fgen1 = 0ULL;
    ps->res_kind = PAR_RES_NONE;
    ps->need_post = false;
    ps->par_overflow = false;

    tr_translate_prepare();
    return true;
}

bool tr_translate_fast(int view, uint64_t cr3, int cpl, bool compat, uint64_t rip, bool single, TrResult *out) {
    memset(out, 0, sizeof *out);
    out->frames[0] = out->frames[1] = ~0ULL;
    tr_compat = compat;
    tr_cpl = cpl;
    walk_cr3 = cr3;
    walk_cpl = cpl;
    fetch_cache_valid = false;
    __atomic_store_n(&tr_fpcr_merges_scalar, ctx_ptr->fpcr_merges_scalar != 0, __ATOMIC_RELAXED);
    if (view > view_high)
        view_high = view;
    int key = block_key(view, cr3, cpl, rip);
    out->key = key;

    if (!single) {
        u32 hit = map_find(key, rip);
        if (hit) {
            if ((key & KEY_DOMAIN) && cpl == 3)
                stats.shared_found++;
            stats.par_fast_hits++;
            out->arm = tr_cc_va(cc_base + hblocks[hit - 1].entry_word);
            out->frames[0] = hblocks[hit - 1].frame[0];
            out->frames[1] = hblocks[hit - 1].frame[1];
            return true;
        }
    }

    if (tr_needs_flush()) {
        if (ctx_ptr->vcpu_count > 1) {
            out->needs_flush = true;
            return true;
        }
        flush_everything();
        out->flushed = true;
    }
    cur_part = KEY_PART(key);
    if (cur_part == 0 && p0_ring_enabled && !p0_try_advance_empty_slot()) {
        if (ctx_ptr->vcpu_count > 1) {
            out->needs_flush = true;
            return true;
        }
        flush_everything();
        out->flushed = true;
    }

    if (!single) {
        u64 frame;
        if (frame_unprotected(rip, &frame)) {
            out->unprotected = frame + 1;
            return true;
        }
        u64 t_reuse = mach_absolute_time();
        if (reuse_dead_block((u32)key, cpl, rip, out)) {
            stats.translate_ns += mach_absolute_time() - t_reuse;
            return true;
        }
    }

    if (!par_translate_enabled || single || cfg.profile_compile_inputs) {
        *out = do_translate(key, view, cr3, cpl, rip, single);
        if ((key & KEY_DOMAIN) && cpl == 3 && !out->unprotected && !out->needs_flush)
            stats.shared_blocks++;
        out->key = key;
        return true;
    }

    ParScratch *ps = par_scratch_get();
    if (!ps) {
        *out = do_translate(key, view, cr3, cpl, rip, single);
        if ((key & KEY_DOMAIN) && cpl == 3 && !out->unprotected && !out->needs_flush)
            stats.shared_blocks++;
        out->key = key;
        return true;
    }

    // Snapshot concurrency tokens & frame0 BEFORE releasing translatorLock.
    ps->prep_ticks = 0;
    ps->view = view;
    ps->key = key;
    ps->cpl = cpl;
    ps->cr3 = cr3;
    ps->rip = rip;
    ps->page = rip & ~0xfffULL;
    ps->compat = compat;
    ps->fgen_flush = __atomic_load_n(&tr_flush_gen, __ATOMIC_ACQUIRE);
    ps->pt_gen = __atomic_load_n(&tr_pt_inval_gen, __ATOMIC_ACQUIRE);
    u64 gpa0 = 0;
    ps->have_gpa0 = fetch_walk(rip, &gpa0);
    ps->frame0 = ps->have_gpa0 ? (gpa0 & ~0xfffULL) : ~0ULL;
    ps->frame1 = ~0ULL;
    ps->fgen0 = (ps->frame0 != ~0ULL)
                    ? __atomic_load_n(&tr_frame_inval_gen[(ps->frame0 >> 12) & (PAR_GEN_BUCKETS - 1U)], __ATOMIC_ACQUIRE)
                    : 0ULL;
    ps->fgen1 = 0ULL;
    ps->res_kind = PAR_RES_NONE;
    ps->need_post = false;
    ps->par_overflow = false;
    return false;
}

void tr_translate_prepare(void) {
    ParScratch *ps = tls_par;
    if (!ps)
        return;
    u64 prep_t0 = mach_absolute_time();

    tr_compat = ps->compat;
    tr_cpl = ps->cpl;
    walk_cr3 = ps->cr3;
    walk_cpl = ps->cpl;
    cur_part = KEY_PART(ps->key);
    fetch_cache_valid = false;
    fetch_lo = ~0ULL;
    fetch_hi = 0;
    ps->stub_cnt = 0;
    ps->link_cnt = 0;
    ps->code_words = 0;
    ps->pcm_count = 0;
    ps->snap_off = 0;
    ps->snap_len = 0;
    ps->aot_have_page_hash = false;
    ps->aot_page_hash = 0;
    ps->aot_record_eligible = false;
    ps->t2_has_loop = false;
    ps->t2_patch_off = ~0U;
    ps->aot_t2_patch_off = AOT_NO_PATCH_WORD;
    ps->aot_t2_head_off = 0;
    ps->aot_t2_min_off = 0;
    ps->aot_t2_span_bytes = 0;
    ps->t2_head_rip = 0;
    ps->par_overflow = false;
    block_stats_reset(&ps->sdelta);

    // ---- Lock-Free Persistent AOT Fast Path (mmap disk + lock-free stage table + seqlock frame hash) ----
    if (tr_aot_active && aot_cache && ps->have_gpa0 && ps->frame0 < cfg.ram_size) {
        ps->aot_page_hash = aot_hash_frame_cached(aot_cache, ps->frame0, ram_at(ps->frame0));
        ps->aot_have_page_hash = true;
        AotKey akey = {
            .page_hash = ps->aot_page_hash,
            .abi_hash = aot_abi_hash,
            .mode_bits = aot_build_mode_bits(ps->cpl, ps->compat, false),
            .page_offset = (u16)(ps->rip & 0xFFFULL),
            .flags = 0,
        };
        AotBlockView aview;
        if (aot_lookup(aot_cache, &akey, &aview) &&
            aview.code_words > 0 && aview.code_words <= PAR_CODE_CAP &&
            aview.pcm_count <= PAR_PCM_CAP) {
            if (aot_materialize(aot_cache, &aview, ps->rip, ps->code, PAR_CODE_CAP,
                                ps->pcm, PAR_PCM_CAP, 0ULL, cur_part,
                                &aot_stubs, NULL, NULL)) {
                bool reloc_ok = true;
                for (u16 ri = 0; ri < aview.reloc_count; ri++) {
                    const AotReloc *r = &aview.relocs[ri];
                    if (r->type == AOT_RELOC_STUB_BRANCH) {
                        if (ps->stub_cnt >= PAR_MAX_STUBS) { reloc_ok = false; break; }
                        ps->stubs[ps->stub_cnt++] = (ParStubSite){
                            r->word_off, (u8)(r->aux & 0x7FU), (u8)((r->aux & 0x80U) ? 1u : 0u)
                        };
                    } else if (r->type == AOT_RELOC_CC_BRANCH) {
                        if (ps->link_cnt >= PAR_MAX_LINKS) { reloc_ok = false; break; }
                        ps->links[ps->link_cnt++] = (ParLinkSite){ r->word_off, r->aux2, r->aux };
                    }
                }
                if (reloc_ok) {
                    ps->code_words = aview.code_words;
                    ps->pcm_count = aview.pcm_count;
                    if (aview.read_bytes > 0 && (u32)aview.read_off + aview.read_bytes <= 4096U) {
                        ps->snap_off = aview.read_off;
                        ps->snap_len = aview.read_bytes;
                        memcpy(ps->snap_bytes, ram_at(ps->frame0 + aview.read_off), aview.read_bytes);
                    }
                    ps->t2_has_loop = (aview.t2_patch_word_off != AOT_NO_PATCH_WORD);
                    ps->t2_patch_off = ps->t2_has_loop ? aview.t2_patch_word_off : ~0U;
                    ps->aot_t2_patch_off = aview.t2_patch_word_off;
                    ps->aot_t2_head_off = aview.t2_head_off;
                    ps->aot_t2_min_off = aview.t2_min_off;
                    ps->aot_t2_span_bytes = aview.t2_span_bytes;
                    ps->t2_head_rip = ps->rip + aview.t2_head_off;
                    ps->res_kind = PAR_RES_AOT_HIT;
                    ps->prep_ticks = mach_absolute_time() - prep_t0;
                    return;
                }
                ps->stub_cnt = 0;
                ps->link_cnt = 0;
            }
        }
    }

    // ---- Lock-Free Cold Decode + Forward-DAG + Flag Liveness + ARM64 Emit into TLS Scratch ----
    Insn *list = ps->list;
    int count = 0;
    u64 pc = ps->rip, page = ps->page, gpa1 = ~0ULL, frame;
    do {
        if ((pc & 0xfff) > 4096 - 15) {
            if (!fetch_walk((pc | 0xfff) + 1, &gpa1))
                gpa1 = ~0ULL;
            if (gpa1 != ~0ULL && ps->frame1 == ~0ULL) {
                u64 f1 = gpa1 & ~0xfffULL;
                ps->fgen1 = __atomic_load_n(&tr_frame_inval_gen[(f1 >> 12) & (PAR_GEN_BUCKETS - 1U)], __ATOMIC_ACQUIRE);
            }
            if (frame_unprotected((pc | 0xfff) + 1, &frame)) {
                ps->res_kind = PAR_RES_UNPROTECTED;
                ps->fault_or_unprot = frame + 1;
                ps->prep_ticks = mach_absolute_time() - prep_t0;
                return;
            }
        }
        x86_decode(pc, &list[count]);
        Insn *in = &list[count];
        if (in->op == OP_BAD && in->why && !strcmp(in->why, "FETCH_FAULT")) {
            if (count == 0) {
                ps->res_kind = PAR_RES_FETCH_FAULT;
                ps->fault_or_unprot = in->target ? in->target : pc;
                ps->prep_ticks = mach_absolute_time() - prep_t0;
                return;
            }
            break;
        }
        if ((ps->key & KEY_USER_FRAME) == KEY_USER_FRAME && in->next > page + 4096 && count > 0)
            break;
        pc = in->next;
        count++;
        if (in->next > page + 4096 && count > 0) {
            if (gpa1 != ~0ULL)
                ps->frame1 = gpa1 & ~0xfffULL;
            break;
        }
        if (t2_enabled() && (in->op == OP_JCC || in->op == OP_JRCXZ) &&
            in->target >= ps->rip && in->target < in->rip)
            break;
    } while (count < MAX_INSNS && !insn_is_terminator(&list[count - 1]) && (pc & ~0xfffULL) == page);

    tr_par_active = true;
    tr_par_block_start = ps->code;
    ColdEmitOut ceo;
    emit_cold_block(ps->key, ps->rip, page, ps->cpl, false, ps->frame0, ps->frame1,
                    list, count, ps->code, PAR_CODE_CAP, ps->pcm, PAR_PCM_CAP,
                    &ps->sdelta, &ceo);
    u32 emitted_words = (u32)(tr_cc - ps->code);
    bool aot_ok = tr_aot_active && tr_aot_cap.active && !tr_aot_cap.ineligible &&
                  ps->frame1 == ~0ULL && ps->frame0 < cfg.ram_size &&
                  ps->aot_have_page_hash && fetch_lo >= page && fetch_hi <= page + 4096;
    if (aot_ok)
        ps->aot_cap = tr_aot_cap;
    tr_aot_cap.active = false;
    tr_par_active = false;

    if (ceo.overflow || ps->par_overflow || emitted_words == 0 || emitted_words > PAR_CODE_CAP) {
        ps->res_kind = PAR_RES_FALLBACK_LOCKED;
        ps->prep_ticks = mach_absolute_time() - prep_t0;
        return;
    }

    ps->code_words = emitted_words;
    ps->pcm_count = ceo.pcm_written;
    ps->aot_record_eligible = aot_ok;
    ps->t2_has_loop = (ceo.t2_head_i >= 0 && ceo.t2_patch_off != ~0U);
    ps->t2_patch_off = ceo.t2_patch_off;
    ps->aot_t2_patch_off = ceo.aot_t2_patch_off;
    ps->aot_t2_head_off = ceo.aot_t2_head_off;
    ps->aot_t2_min_off = ceo.aot_t2_min_off;
    ps->aot_t2_span_bytes = ceo.aot_t2_span_bytes;
    ps->t2_head_rip = ceo.t2_head_rip;

    if (ps->frame1 == ~0ULL && ps->frame0 < cfg.ram_size && fetch_lo < fetch_hi) {
        u64 page_end = page + 4096;
        u64 lo = fetch_lo < page ? page : fetch_lo;
        u64 hi = fetch_hi < page_end ? fetch_hi : page_end;
        if (hi > lo && hi - lo <= 4096ULL) {
            ps->snap_off = (u16)(lo & 0xfffULL);
            ps->snap_len = (u16)(hi - lo);
            memcpy(ps->snap_bytes, ram_at(ps->frame0 + ps->snap_off), ps->snap_len);
        }
    }
    ps->res_kind = PAR_RES_COLD;
    ps->prep_ticks = mach_absolute_time() - prep_t0;
}

int tr_translate_commit(TrResult *out) {
    u64 com_t0 = mach_absolute_time();
    memset(out, 0, sizeof *out);
    out->frames[0] = out->frames[1] = ~0ULL;
    ParScratch *ps = tls_par;
    if (!ps)
        return 0;

    tr_compat = ps->compat;
    tr_cpl = ps->cpl;
    walk_cr3 = ps->cr3;
    walk_cpl = ps->cpl;
    fetch_cache_valid = false;
    if (ps->view > view_high)
        view_high = ps->view;
    out->key = ps->key;
    ps->need_post = false;

    if (ps->res_kind == PAR_RES_FALLBACK_LOCKED) {
        stats.par_fallbacks++;
        *out = do_translate(ps->key, ps->view, ps->cr3, ps->cpl, ps->rip, false);
        if ((ps->key & KEY_DOMAIN) && ps->cpl == 3 && !out->unprotected && !out->needs_flush)
            stats.shared_blocks++;
        out->key = ps->key;
        return out->needs_flush ? -1 : 1;
    }

    if (tr_needs_flush()) {
        if (ctx_ptr->vcpu_count > 1) {
            out->needs_flush = true;
            return -1;
        }
        flush_everything();
        out->flushed = true;
    }

    // 1. Re-verify translation domain key & concurrent duplicate publication.
    int key_now = block_key(ps->view, ps->cr3, ps->cpl, ps->rip);
    out->key = key_now;
    if (key_now != ps->key) {
        stats.par_retries++;
        return 0;
    }
    u32 dup = map_find(key_now, ps->rip);
    if (dup) {
        stats.par_dups++;
        if ((key_now & KEY_DOMAIN) && ps->cpl == 3)
            stats.shared_found++;
        out->arm = tr_cc_va(cc_base + hblocks[dup - 1].entry_word);
        out->frames[0] = hblocks[dup - 1].frame[0];
        out->frames[1] = hblocks[dup - 1].frame[1];
        stats.translate_ns += ps->prep_ticks + (mach_absolute_time() - com_t0);
        return 1;
    }

    // 2. Verify flush / page-table / SMC generations unchanged since before decode!
    if (__atomic_load_n(&tr_flush_gen, __ATOMIC_ACQUIRE) != ps->fgen_flush) {
        stats.par_retries++;
        return 0;
    }
    if (ps->res_kind == PAR_RES_FETCH_FAULT || ps->res_kind == PAR_RES_UNPROTECTED) {
        if (__atomic_load_n(&tr_pt_inval_gen, __ATOMIC_ACQUIRE) != ps->pt_gen) {
            stats.par_retries++;
            return 0;
        }
        if (ps->res_kind == PAR_RES_FETCH_FAULT)
            out->fetch_fault_va = ps->fault_or_unprot;
        else
            out->unprotected = ps->fault_or_unprot;
        return 1;
    }

    u64 frame;
    if (frame_unprotected(ps->rip, &frame)) {
        out->unprotected = frame + 1;
        return 1;
    }
    if (ps->frame1 != ~0ULL && frame_unprotected(ps->page + 4096ULL, &frame)) {
        out->unprotected = frame + 1;
        return 1;
    }
    if (__atomic_load_n(&tr_pt_inval_gen, __ATOMIC_ACQUIRE) != ps->pt_gen) {
        u64 g0 = 0, g1 = 0;
        bool h0 = fetch_walk(ps->rip, &g0);
        u64 f0 = h0 ? (g0 & ~0xfffULL) : ~0ULL;
        if (h0 != ps->have_gpa0 || f0 != ps->frame0) {
            stats.par_retries++;
            return 0;
        }
        if (ps->frame1 != ~0ULL) {
            if (!fetch_walk(ps->page + 4096ULL, &g1) || (g1 & ~0xfffULL) != ps->frame1) {
                stats.par_retries++;
                return 0;
            }
        }
    }
    if (ps->frame0 != ~0ULL &&
        __atomic_load_n(&tr_frame_inval_gen[(ps->frame0 >> 12) & (PAR_GEN_BUCKETS - 1U)], __ATOMIC_ACQUIRE) != ps->fgen0) {
        stats.par_retries++;
        return 0;
    }
    if (ps->frame1 != ~0ULL &&
        __atomic_load_n(&tr_frame_inval_gen[(ps->frame1 >> 12) & (PAR_GEN_BUCKETS - 1U)], __ATOMIC_ACQUIRE) != ps->fgen1) {
        stats.par_retries++;
        return 0;
    }

    // 3. Allocate slot in code cache (p0 4-slot ring or linear part 0/1) & PcMap.
    cur_part = KEY_PART(key_now);
    u32 alloc_slot = 0;
    u32 *start;
    u32 *pcm_cursor_p;
    if (cur_part == 0 && p0_ring_enabled) {
        if (!p0_try_advance_empty_slot()) {
            if (ctx_ptr->vcpu_count > 1) {
                out->needs_flush = true;
                return -1;
            }
            flush_everything();
            out->flushed = true;
        }
        alloc_slot = p0_cur_slot;
        start = p0_cc_cursor[alloc_slot];
        // p0_try_advance_empty_slot() already ensured start <= 59 MiB (2 MiB headroom before true slot end at 61 MiB).
        u32 *slot_end = cc_base + engine_p0_slot_word_base(alloc_slot) + P0_SLOT_WORDS;
        if (start + ps->code_words > slot_end || p0_block_count[alloc_slot] >= P0_BLOCKS_PER_SLOT ||
            p0_pcmap_used[alloc_slot] + ps->pcm_count > (alloc_slot + 1U) * P0_PCMAP_PER_SLOT) {
            out->needs_flush = true;
            return -1;
        }
        pcm_cursor_p = &p0_pcmap_used[alloc_slot];
    } else {
        if (t2_enabled()) {
            u32 t2w = cur_part == 1 ? CC_T2_WORD1 : CC_T2_WORD0;
            if (cc_cursor[cur_part] + ps->code_words + 16U >= cc_base + t2w &&
                cc_cursor[cur_part] < cc_base + t2w + CC_T2_WORDS) {
                cc_cursor[cur_part] = cc_base + t2w + CC_T2_WORDS;
            }
        }
        start = cc_cursor[cur_part];
        u64 lim = cur_part == 1 && cc_limit >= CC_SIZE ? CC_SIZE - (4ULL << 20) : cc_limit;
        u32 *part_end = cc_base + cur_part * CC_PART2_WORD + (u32)(lim / 4ULL);
        u32 first_map = cur_part ? (OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap) : 0;
        if (start + ps->code_words > part_end || hblock_count[cur_part] >= PART_BLOCKS_MAX(cur_part) ||
            pc_map_used[cur_part] + ps->pcm_count > first_map + PCMAP_ENTRIES) {
            out->needs_flush = true;
            return -1;
        }
        pcm_cursor_p = &pc_map_used[cur_part];
    }

    // 4. Copy code + PcMap, patch stub B/BL relocations for final VA, apply live eager direct links.
    u32 map0 = *pcm_cursor_p;
    memcpy(start, ps->code, (size_t)ps->code_words * sizeof(u32));
    if (ps->pcm_count)
        memcpy(&pc_map[map0], ps->pcm, (size_t)ps->pcm_count * sizeof(PcMap));
    *pcm_cursor_p = map0 + ps->pcm_count;

    u64 base_va = tr_cc_va(start);
    for (u32 i = 0; i < ps->stub_cnt; i++) {
        u32 w = ps->stubs[i].word_off;
        if (w >= ps->code_words)
            continue;
        u64 pc_va = base_va + (u64)w * 4ULL;
        u64 stub_va = aot_resolve_stub_va(&aot_stubs, (AotStubId)ps->stubs[i].stub_id, cur_part, pc_va);
        i64 disp = (i64)(stub_va - pc_va) / 4;
        start[w] = (ps->stubs[i].is_bl ? 0x94000000u : 0x14000000u) | ((u32)disp & 0x3FFFFFFu);
    }
    u32 first_word_after_stubs = *start;

    if (aot_eager_link && ps->link_cnt > 0) {
        AotLinkCtx lc = { .key = key_now, .entry_rip = ps->rip, .page = ps->page };
        u64 linked = 0;
        for (u32 i = 0; i < ps->link_cnt; i++) {
            u32 w = ps->links[i].word_off;
            u32 lit_w = ps->links[i].lit_off;
            if (w >= ps->code_words || lit_w + 1U >= ps->code_words)
                continue;
            u64 target_rip = (u64)start[lit_w] | ((u64)start[lit_w + 1U] << 32);
            u64 target_arm = aot_resolve_live_target(&lc, target_rip);
            if (!target_arm)
                continue;
            i64 d = (i64)(target_arm - (base_va + (u64)w * 4ULL)) / 4;
            if (ps->links[i].kind == AOT_CC_DIRECT && d >= -(1LL << 25) && d < (1LL << 25)) {
                start[w] = 0x14000000u | ((u32)d & 0x3FFFFFFu);
                linked++;
            } else if (ps->links[i].kind == AOT_CC_CALL && d >= -(1LL << 25) && d < (1LL << 25)) {
                start[w] = 0x94000000u | ((u32)d & 0x3FFFFFFu);
                linked++;
            } else if (ps->links[i].kind == AOT_CC_JCC && d >= -(1LL << 18) && d < (1LL << 18)) {
                start[w] = (start[w] & ~(0x7FFFFu << 5)) | (((u32)d & 0x7FFFFu) << 5);
                linked++;
            }
        }
        if (ps->res_kind == PAR_RES_AOT_HIT && linked && aot_cache)
            aot_note_eager_links(aot_cache, linked);
    }

    // 5. Register HostBlock + BlockEntry + block_map + icache invalidate + code_published.
    if (ps->res_kind == PAR_RES_COLD && ps->compat && ps->cpl == 3)
        stats.i386_blocks++;

    u32 idx;
    if (cur_part == 0 && p0_ring_enabled) {
        idx = alloc_slot * P0_BLOCKS_PER_SLOT + p0_block_count[alloc_slot]++;
    } else {
        idx = BLOCK_BASE(cur_part) + hblock_count[cur_part]++;
    }
    HostBlock *hb = &hblocks[idx];
    memset(hb, 0, sizeof *hb);
    hb->rip = ps->rip;
    hb->view = (u32)key_now;
    hb->entry_word = (u32)(start - cc_base);
    hb->frame[0] = ps->frame0;
    hb->frame[1] = ps->frame1;
    hb->cpl = (u8)ps->cpl;
    hb->p0_slot = (u8)alloc_slot;
    hb->compat = ps->compat;
    hb->first_word = first_word_after_stubs;
    hb->t2_has_loop = ps->t2_has_loop;
    hb->t2_patch_word = (ps->t2_has_loop && ps->t2_patch_off != ~0U) ? (hb->entry_word + ps->t2_patch_off) : ~0U;
    hb->on_frame = hb->on_page = hb->on_view = true;
    if (!cfg.no_reuse && ps->frame1 == ~0ULL && ps->frame0 < cfg.ram_size && ps->snap_len > 0) {
        u32 end = cur_part ? READ_COPY_BYTES : READ_COPY_BYTES / 2;
        if (read_copied[cur_part] + ps->snap_len <= end) {
            memcpy(read_copies + read_copied[cur_part], ps->snap_bytes, ps->snap_len);
            hb->read_copy = read_copied[cur_part];
            hb->read_off = ps->snap_off;
            hb->read_bytes = ps->snap_len;
            read_copied[cur_part] += ps->snap_len;
        }
    }

    link_block(idx);
    BlockEntry *be = &block_table[cur_part ? BLOCK_TABLE_PART2 + (idx - BLOCK_BASE(1)) : idx];
    be->rip = ps->rip;
    be->key = hb->view;
    be->arm_off = hb->entry_word;
    be->arm_words = ps->code_words;
    be->map_off = map0;
    be->map_count = ps->pcm_count;
    map_insert((int)hb->view, ps->rip, idx + 1);
    hb->registered = true;

    u32 *end_cc = start + ps->code_words;
    if (cur_part == 0 && p0_ring_enabled) {
        __atomic_store_n(&ctx_ptr->p0_slot_block_count[alloc_slot], (u64)p0_block_count[alloc_slot], __ATOMIC_RELEASE);
        p0_cc_cursor[alloc_slot] = end_cc;
    } else {
        if (cur_part == 0)
            __atomic_store_n(&ctx_ptr->block_count, (u64)hblock_count[0], __ATOMIC_RELEASE);
        else
            __atomic_store_n(&ctx_ptr->block_count_private, (u64)hblock_count[1], __ATOMIC_RELEASE);
        cc_cursor[cur_part] = end_cc;
    }
    sys_icache_invalidate(start, (size_t)ps->code_words * sizeof(u32));
    code_published();

    // Register cold Tier-2 loop under commit lock (~80 ns, prevents kill_block vs register race);
    // heavy AOT record, Tier-2 AOT materialize/publish, and eager T2 promotion run unlocked in tr_translate_post()!
    u64 t2_ep = __atomic_load_n(&t2_epoch[cur_part], __ATOMIC_RELAXED);
    if (ps->res_kind == PAR_RES_COLD && hb->t2_has_loop && hb->t2_patch_word != ~0U && ps->aot_t2_span_bytes > 0) {
        t2_register_loop(idx, (u32)cur_part, alloc_slot, hb->entry_word, ps->code_words,
                         hb->t2_patch_word, ps->rip, ps->t2_head_rip, ps->frame0,
                         ps->aot_t2_min_off, ps->aot_t2_span_bytes, (u32)key_now, (u8)ps->cpl, ps->compat, t2_ep);
    }

    ps->pub_idx = idx;
    ps->pub_part = (u32)cur_part;
    ps->pub_slot = alloc_slot;
    ps->pub_entry_word = hb->entry_word;
    ps->pub_arm_words = ps->code_words;
    ps->pub_t2_patch_word = hb->t2_patch_word;
    ps->pub_read_off = hb->read_off;
    ps->pub_read_bytes = hb->read_bytes;
    ps->pub_t2_epoch = t2_ep;
    ps->post_is_aot_hit = (ps->res_kind == PAR_RES_AOT_HIT);
    ps->need_post = (ps->res_kind == PAR_RES_COLD && (ps->aot_record_eligible || (hb->t2_has_loop && t2_eager_mode()))) ||
                    (ps->res_kind == PAR_RES_AOT_HIT && hb->t2_has_loop && t2_enabled());

    if (ps->res_kind == PAR_RES_COLD)
        block_stats_merge(&ps->sdelta);
    stats.blocks++;
    stats.words += ps->code_words;
    stats.par_translates++;
    if ((key_now & KEY_DOMAIN) && ps->cpl == 3)
        stats.shared_blocks++;
    stats.translate_ns += ps->prep_ticks + (mach_absolute_time() - com_t0);

    out->arm = base_va;
    out->frames[0] = ps->frame0;
    out->frames[1] = ps->frame1;
    return 1;
}

void tr_translate_post(void) {
    ParScratch *ps = tls_par;
    if (!ps || !ps->need_post)
        return;
    ps->need_post = false;

    if (!ps->post_is_aot_hit) {
        // Cold block: record clean unlinked template in AOT cache outside translatorLock.
        if (ps->aot_record_eligible && tr_aot_active && aot_cache) {
            AotBlockView rec;
            memset(&rec, 0, sizeof rec);
            rec.key.page_hash = ps->aot_page_hash;
            rec.key.abi_hash = aot_abi_hash;
            rec.key.mode_bits = aot_build_mode_bits(ps->cpl, ps->compat, false);
            rec.key.page_offset = (u16)(ps->rip & 0xFFFULL);
            rec.orig_rip = ps->rip;
            rec.code = ps->code;
            rec.code_words = ps->code_words;
            rec.pcm = ps->pcm;
            rec.pcm_count = (u16)ps->pcm_count;
            rec.relocs = ps->aot_cap.relocs;
            rec.reloc_count = (u16)ps->aot_cap.count;
            rec.read_off = ps->pub_read_off;
            rec.read_bytes = ps->pub_read_bytes;
            rec.t2_patch_word_off = ps->aot_t2_patch_off;
            rec.t2_head_off = ps->aot_t2_head_off;
            rec.t2_min_off = ps->aot_t2_min_off;
            rec.t2_span_bytes = ps->aot_t2_span_bytes;
            rec.priority = AOT_PRIO_TIER1;
            if (aot_record(aot_cache, &rec))
                aot_maybe_save_async(aot_cache, 65536U);
        }
        if (ps->t2_has_loop && ps->pub_t2_patch_word != ~0U && ps->aot_t2_span_bytes > 0 && t2_eager_mode())
            t2_eager_promote_now(ps->pub_idx);
        return;
    }

    // Warm AOT hit with Tier-2 loop: instant Tier-2 promotion outside translatorLock!
    if (ps->t2_has_loop && ps->pub_t2_patch_word != ~0U) {
        tr_aot_promote_t2_post(ps->pub_idx, ps->pub_part, ps->pub_slot, ps->pub_t2_epoch,
                               (u32)ps->key, (u8)ps->cpl, ps->compat,
                               ps->pub_entry_word, ps->pub_arm_words, ps->pub_t2_patch_word,
                               ps->rip, ps->t2_head_rip, ps->frame0, ps->aot_page_hash,
                               ps->aot_t2_min_off, ps->aot_t2_span_bytes);
    }
}

static TrResult do_translate(int key, int view, u64 cr3, int cpl, u64 rip, bool single) {
    (void)view;
    u64 t0 = mach_absolute_time();
    TrResult res = {0};
    res.frames[0] = res.frames[1] = ~0ULL;
    Insn list[MAX_INSNS];
    walk_cr3 = cr3;
    walk_cpl = cpl;
    tr_cpl = cpl;
    fetch_cache_valid = false;
    if (tr_needs_flush()) {
        if (ctx_ptr->vcpu_count > 1) {
            res.needs_flush = true;
            return res;
        }
        flush_everything();
        res.flushed = true;
    }
    cur_part = KEY_PART(key);
    u32 alloc_slot = 0;
    if (cur_part == 0 && p0_ring_enabled) {
        if (!p0_try_advance_empty_slot()) {
            if (ctx_ptr->vcpu_count > 1) {
                res.needs_flush = true;
                return res;
            }
            flush_everything();
            res.flushed = true;
        }
        alloc_slot = p0_cur_slot;
        tr_cc = p0_cc_cursor[alloc_slot];
    } else {
        // Linear mode: skip over CC_T2 mid-cache island when Tier-2 is enabled so T1 never collides with CC_T2.
        if (t2_enabled()) {
            u32 t2w = cur_part == 1 ? CC_T2_WORD1 : CC_T2_WORD0;
            if (cc_cursor[cur_part] + 4096 >= cc_base + t2w &&
                cc_cursor[cur_part] < cc_base + t2w + CC_T2_WORDS) {
                cc_cursor[cur_part] = cc_base + t2w + CC_T2_WORDS;
            }
        }
        tr_cc = cc_cursor[cur_part];
    }
    int count = 0;
    u64 pc = rip, page = rip & ~0xfffULL, frame;
    if (!single) {
        if (frame_unprotected(rip, &frame)) {
            res.unprotected = frame + 1;
            return res;
        }
        if (reuse_dead_block((u32)key, cpl, rip, &res)) {
            stats.translate_ns += mach_absolute_time() - t0;
            return res;
        }
    }
    fetch_lo = ~0ULL;
    fetch_hi = 0;
    // Before the bytes, as the mapping can move at any time: a frame walked after them could be one the block was never read from, and looking again later would agree with it.
    u64 gpa0 = 0, gpa1 = ~0ULL;
    bool have_gpa0 = fetch_walk(rip, &gpa0);
    res.frames[0] = have_gpa0 ? (gpa0 & ~0xfffULL) : ~0ULL;

    // ---- Persistent AOT Cache Fast Path (single-page content hash lookup + ASLR relocation) ----
    u64 aot_page_hash = 0;
    bool aot_have_page_hash = false;
    if (tr_aot_active && aot_cache && !single && have_gpa0 && res.frames[0] < cfg.ram_size && !cfg.profile_compile_inputs) {
        aot_page_hash = aot_hash_frame_cached(aot_cache, res.frames[0], ram_at(res.frames[0]));
        aot_have_page_hash = true;
        AotKey akey = {
            .page_hash = aot_page_hash,
            .abi_hash = aot_abi_hash,
            .mode_bits = aot_build_mode_bits(cpl, tr_compat, false),
            .page_offset = (u16)(rip & 0xFFFULL),
            .flags = 0,
        };
        AotBlockView aview;
        if (aot_lookup(aot_cache, &akey, &aview)) {
            u32 *start = tr_cc;
            u32 *pcm_cursor_p = (cur_part == 0 && p0_ring_enabled) ? &p0_pcmap_used[alloc_slot] : &pc_map_used[cur_part];
            u32 map0 = *pcm_cursor_p;
            u32 cc_cap_words = 16384U;
            u32 pcm_cap = MAX_INSNS + 256U;
            AotLinkCtx lc = { .key = key, .entry_rip = rip, .page = page };
            if (aot_materialize(aot_cache, &aview, rip, start, cc_cap_words,
                                &pc_map[map0], pcm_cap, tr_cc_va(start), cur_part,
                                &aot_stubs, aot_resolve_live_target, &lc)) {
                tr_cc = start + aview.code_words;
                *pcm_cursor_p = map0 + aview.pcm_count;

                u32 idx;
                if (cur_part == 0 && p0_ring_enabled) {
                    idx = alloc_slot * P0_BLOCKS_PER_SLOT + p0_block_count[alloc_slot]++;
                } else {
                    idx = BLOCK_BASE(cur_part) + hblock_count[cur_part]++;
                }
                HostBlock *hb = &hblocks[idx];
                memset(hb, 0, sizeof *hb);
                hb->rip = rip;
                hb->view = (u32)key;
                hb->entry_word = (u32)(start - cc_base);
                hb->frame[0] = res.frames[0];
                hb->frame[1] = ~0ULL;
                hb->cpl = (u8)cpl;
                hb->p0_slot = (u8)alloc_slot;
                hb->compat = tr_compat;
                hb->first_word = *start;
                hb->t2_has_loop = (aview.t2_patch_word_off != AOT_NO_PATCH_WORD);
                hb->t2_patch_word = hb->t2_has_loop ? (hb->entry_word + aview.t2_patch_word_off) : ~0U;
                hb->on_frame = hb->on_page = hb->on_view = true;
                if (aview.read_bytes > 0) {
                    fetch_lo = page + aview.read_off;
                    fetch_hi = fetch_lo + aview.read_bytes;
                    remember_reads(hb, rip);
                }
                link_block(idx);

                BlockEntry *be = &block_table[cur_part ? BLOCK_TABLE_PART2 + (idx - BLOCK_BASE(1)) : idx];
                be->rip = rip;
                be->key = hb->view;
                be->arm_off = hb->entry_word;
                be->arm_words = aview.code_words;
                be->map_off = map0;
                be->map_count = aview.pcm_count;
                map_insert((int)hb->view, rip, idx + 1);
                hb->registered = true;

                __sync_synchronize();
                if (cur_part == 0 && p0_ring_enabled) {
                    __atomic_store_n(&ctx_ptr->p0_slot_block_count[alloc_slot], (u64)p0_block_count[alloc_slot], __ATOMIC_RELEASE);
                    p0_cc_cursor[alloc_slot] = tr_cc;
                } else {
                    if (cur_part == 0)
                        __atomic_store_n(&ctx_ptr->block_count, (u64)hblock_count[0], __ATOMIC_RELEASE);
                    else
                        __atomic_store_n(&ctx_ptr->block_count_private, (u64)hblock_count[1], __ATOMIC_RELEASE);
                    cc_cursor[cur_part] = tr_cc;
                }
                sys_icache_invalidate(start, (u8 *)tr_cc - (u8 *)start);
                code_published();

                // Warm Tier-2 instant promotion (0 ms warmup, preserving Round 2 T2TrapMeta & bailout sites!).
                if (hb->t2_has_loop && t2_enabled() && aview.t2_span_bytes > 0 &&
                    (u32)aview.t2_min_off + aview.t2_span_bytes <= 4096U) {
                    u64 head_rip = rip + aview.t2_head_off;
                    AotKey t2key = {
                        .page_hash = aot_page_hash,
                        .abi_hash = aot_abi_hash,
                        .mode_bits = aot_build_mode_bits(cpl, tr_compat, true),
                        .page_offset = (u16)(head_rip & 0xFFFULL),
                        .flags = 0,
                    };
                    AotBlockView t2view;
                    bool promoted = false;
                    if (aot_lookup(aot_cache, &t2key, &t2view) && t2view.code_words <= 8192U && t2view.pcm_count <= 1024U) {
                        u32 t2_buf[8192];
                        PcMap t2_pcm[1024];
                        u32 bail_sites[32];
                        u32 bail_cnt = 0;
                        bool bail_ok = true;
                        for (u16 ri = 0; ri < t2view.reloc_count; ri++) {
                            if (t2view.relocs[ri].type == AOT_RELOC_T2_BAILOUT) {
                                if (bail_cnt >= 32U) { bail_ok = false; break; }
                                bail_sites[bail_cnt++] = t2view.relocs[ri].word_off;
                            }
                        }
                        u64 t2_dest_va = tr_cc_va(cc_t2_cursor[cur_part]);
                        if (bail_ok && aot_materialize(aot_cache, &t2view, head_rip, t2_buf, 8192U,
                                                       t2_pcm, 1024U, t2_dest_va, cur_part,
                                                       &aot_stubs, NULL, NULL)) {
                            u64 ep = __atomic_load_n(&t2_epoch[cur_part], __ATOMIC_RELAXED);
                            promoted = tr_t2_publish(idx, (u32)cur_part, alloc_slot, ep, head_rip,
                                                     (u32)key, (u8)cpl, tr_compat, hb->t2_patch_word,
                                                     res.frames[0], aview.t2_min_off, aview.t2_span_bytes,
                                                     ram_at(res.frames[0] + aview.t2_min_off),
                                                     t2_buf, t2view.code_words, t2_pcm, t2view.pcm_count,
                                                     t2view.has_t2_meta ? &t2view.t2_meta : NULL,
                                                     bail_cnt ? bail_sites : NULL, bail_cnt);
                        }
                    }
                    if (!promoted) {
                        u64 ep = __atomic_load_n(&t2_epoch[cur_part], __ATOMIC_RELAXED);
                        t2_register_loop(idx, (u32)cur_part, alloc_slot, hb->entry_word, be->arm_words,
                                         hb->t2_patch_word, rip, head_rip, res.frames[0],
                                         aview.t2_min_off, aview.t2_span_bytes, (u32)key, (u8)cpl, tr_compat, ep);
                        if (t2_eager_mode())
                            t2_eager_promote_now(idx);
                    }
                }

                stats.blocks++;
                stats.words += aview.code_words;
                res.arm = tr_cc_va(start);
                stats.translate_ns += mach_absolute_time() - t0;
                return res;
            }
        }
    }

    compile_inputs_begin(rip, cpl, tr_compat, (u32)key, res.frames[0], single);
    do {
        if ((pc & 0xfff) > 4096 - 15) {
            if (!fetch_walk((pc | 0xfff) + 1, &gpa1))
                gpa1 = ~0ULL;
            if (!single && frame_unprotected((pc | 0xfff) + 1, &frame)) {
                compile_inputs_reject(TR_CI_UNPROTECTED);
                compile_inputs_finish();
                res.unprotected = frame + 1;
                return res;
            }
        }
        x86_decode(pc, &list[count]);
        Insn *in = &list[count];
        if (in->op == OP_BAD && in->why && !strcmp(in->why, "FETCH_FAULT")) {
            if (count == 0) {
                compile_inputs_reject(TR_CI_FETCH_FAULT);
                compile_inputs_finish();
                res.fetch_fault_va = in->target ? in->target : pc;
                return res;
            }
            break; // cut the block before it, the fault belongs to the instruction when it runs
        }
        if (in->op == OP_BAD || in->op == OP_UD)
            compile_inputs_reject(TR_CI_UNSUPPORTED);
        if ((key & KEY_USER_FRAME) == KEY_USER_FRAME && in->next > page + 4096 && count > 0)
            break;
        pc = in->next;
        count++;
        if (in->next > page + 4096 && count > 0) {
            // Crossed into the next page: record it, end here.
            if (gpa1 != ~0ULL)
                res.frames[1] = gpa1 & ~0xfffULL;
            break;
        }
        if (!single && t2_enabled() && (in->op == OP_JCC || in->op == OP_JRCXZ) &&
            in->target >= rip && in->target < in->rip)
            break;
    } while (!single && count < MAX_INSNS && !insn_is_terminator(&list[count - 1]) && (pc & ~0xfffULL) == page);

    Insn *last = &list[count - 1];
    bool fallthrough_end = !(last->op == OP_JMP || last->op == OP_JMPI || last->op == OP_RET || last->op == OP_BAD ||
                             last->op == OP_UD || last->op == OP_IRET);
    block_insns = list;
    insn_total = count;
    pend_count = 0;
    fwd_count = 0;
    tr_ad_page = ~0ULL;
    tr_aes_mask = 0;
    u32 *start = tr_cc;
    if (tr_aot_active && !single && res.frames[1] == ~0ULL && !cfg.profile_compile_inputs) {
        tr_aot_block_start = start;
        aot_capture_begin(&tr_aot_cap);
    } else {
        tr_aot_cap.active = false;
    }
    u32 *tail = tr_cc;
    u32 *pcm_cursor_p = (cur_part == 0 && p0_ring_enabled) ? &p0_pcmap_used[alloc_slot] : &pc_map_used[cur_part];
    u32 map0 = *pcm_cursor_p;
    if (native_dag && !single && count > 0 && count < MAX_INSNS && res.frames[1] == ~0ULL) {
        expand_forward_dag(list, &count, key, rip, page);
        insn_total = count;
        last = &list[count - 1];
        fallthrough_end = !is_uncond_term_op(last->op);
    }
    cur_key = key;
    cur_entry_rip = rip;
    cur_page = page;
    cur_single = single;
    l1_miss_count = 0;
    tr_zext8 = tr_zext16 = 0;
    ext_ymm_cache_reset();

    int tgt_idx[MAX_INSNS], fall_idx[MAX_INSNS];
    u8 ext_tgt_live[MAX_INSNS], ext_fall_live[MAX_INSNS];
    bool is_target[MAX_INSNS];
    memset(is_target, 0, (size_t)count * sizeof(bool));
    for (int i = 0; i < count; i++) {
        const Insn *in = &list[i];
        tgt_idx[i] = -1;
        fall_idx[i] = -1;
        ext_tgt_live[i] = 0;
        ext_fall_live[i] = 0;
        if (in->op == OP_JCC || in->op == OP_JRCXZ || in->op == OP_JMP) {
            int ti = idx_of_rip(list, count, in->target);
            tgt_idx[i] = ti;
            if (ti >= 0)
                is_target[ti] = true;
            else
                ext_tgt_live[i] = (u8)flags_needed_at(in->target, page, in->op == OP_JMP ? 3 : 2, list, count);
        } else if (in->op == OP_CALL) {
            if (!tr_precise_flags && (in->target & ~0xfffULL) == page && ((in->target + 15) & ~0xfffULL) == page)
                ext_tgt_live[i] = (u8)flags_needed_at(in->target, page, 2, list, count);
            else
                ext_tgt_live[i] = F_ARITH;
        }
        if (!is_uncond_term_op(in->op)) {
            if (i + 1 < count && list[i + 1].rip == in->next) {
                fall_idx[i] = i + 1;
            } else {
                int fi = idx_of_rip(list, count, in->next);
                fall_idx[i] = fi;
                if (fi >= 0)
                    is_target[fi] = true;
                else
                    ext_fall_live[i] = (u8)flags_needed_at(in->next, page, 3, list, count);
            }
        }
        if (i > 0 && (is_uncond_term_op(list[i - 1].op) || list[i - 1].next != in->rip))
            is_target[i] = true;
    }

    u8 live_after[MAX_INSNS];
    u8 live_before[MAX_INSNS];
    if (cfg.all_flags_live) {
        memset(live_after, F_ARITH, (size_t)count);
        memset(live_before, F_ARITH, (size_t)count);
    } else {
        memset(live_after, 0, (size_t)count);
        memset(live_before, 0, (size_t)count);
        bool changed = true;
        for (int pass = 0; pass < 8 && changed; pass++) {
            changed = false;
            for (int i = count - 1; i >= 0; i--) {
                const Insn *in = &list[i];
                u8 out;
                if (in->op == OP_JMP) {
                    out = tgt_idx[i] >= 0 ? live_before[tgt_idx[i]] : ext_tgt_live[i];
                } else if (is_uncond_term_op(in->op) || in->op == OP_CALLI) {
                    out = F_ARITH;
                } else if (in->op == OP_CALL) {
                    out = ext_tgt_live[i];
                } else if (in->op == OP_JCC || in->op == OP_JRCXZ) {
                    u8 f = fall_idx[i] >= 0 ? live_before[fall_idx[i]] : ext_fall_live[i];
                    u8 t = tgt_idx[i] >= 0 ? live_before[tgt_idx[i]] : ext_tgt_live[i];
                    out = f | t;
                } else {
                    out = fall_idx[i] >= 0 ? live_before[fall_idx[i]] : ext_fall_live[i];
                }
                u8 lb = (u8)((out & ~in->writes) | insn_reads(in));
                if (out != live_after[i] || lb != live_before[i]) {
                    live_after[i] = out;
                    live_before[i] = lb;
                    changed = true;
                }
            }
        }
        if (changed) {
            memset(live_after, F_ARITH, (size_t)count);
            memset(live_before, F_ARITH, (size_t)count);
        }
    }

    int t2_head_i = (!single && (cpl == 3 || t2_eager_mode()) &&
                     res.frames[1] == ~0ULL && res.frames[0] != ~0ULL && res.frames[0] < cfg.ram_size &&
                     !tr_precise_flags && !cfg.all_flags_live && !cfg.profile_compile_inputs)
                        ? t2_find_clean_loop(list, count, tgt_idx)
                        : -1;
    u32 t2_patch_word = ~0U;
    if (ctx_ptr->vcpu_count > 1 || t2_head_i == 0) // a kill or Tier-2 OSR patches the entry NOP atomically
        E(WORD_NOP);
    if (t2_head_i == 0)
        t2_patch_word = (u32)(start - cc_base);

    for (int i = 0; i < count; i++) {
        const Insn *in = &list[i];
        bool prev_fell_through = i > 0 && !is_uncond_term_op(list[i - 1].op) && list[i - 1].next == in->rip;
        if (is_target[i]) {
            if (prev_fell_through) {
                if (tr_aes_mask)
                    sse_aes_flush_all();
                if (ext_ymm_cache_has_dirty())
                    ext_ymm_cache_flush_dirty();
            }
            tr_ad_page = ~0ULL;
            tr_zext8 = tr_zext16 = 0;
            ext_ymm_cache_reset();
        }
        if (ext_ymm_cache_has_dirty() &&
            (in->op != OP_EXT || in->memop || is_target[i] || ext_ymm_cache_dirty_run() >= 8)) {
            ext_ymm_cache_flush_dirty();
        }
        if (in->op == OP_SSE || in->op == OP_LDMXCSR || in->op == OP_STMXCSR ||
            in->op == OP_X87 || (in->op == OP_POPCNT && !tr_hybrid)) {
            ext_ymm_cache_flush_and_drop();
        }
        if ((!tr_opt_peep && in->op != OP_SSE && in->op != OP_EXT) ||
            (tr_opt_peep && !ad_page_safe_op(in->op))) {
            tr_ad_page = ~0ULL;
        }
        if (in->op != OP_SSE && tr_aes_mask)
            sse_aes_flush_all();
        // Loop header OSR anchor: for head==0, backedges jump to start[0] (WORD_NOP);
        // for head>0, emit WORD_NOP right at insn_arm[head] (canonical flushed state).
        if (i == 0 && t2_head_i == 0) {
            insn_arm[0] = start;
        } else {
            insn_arm[i] = tr_cc;
        }
        if (i)
            stats.op_words[list[i - 1].op & 127] += (u64)(tr_cc - insn_arm[i - 1]);
        stats.op_insns[in->op & 127]++;
        if (!ext_ymm_cache_has_dirty()) {
            pc_map[*pcm_cursor_p].arm_word = (u16)(tr_cc - start);
            pc_map[*pcm_cursor_p].rip_off = (u16)(in->rip - rip);
            (*pcm_cursor_p)++;
        }
        if (i > 0 && i == t2_head_i) {
            t2_patch_word = (u32)(tr_cc - cc_base);
            E(WORD_NOP);
        }
        stats.insns++;

        // P1-A multi-instruction peephole fusions (adjacent non-target pairs).
        bool fused = false;
        if (tr_opt_peep && !single && i + 1 < count && !is_target[i + 1] && in->next == list[i + 1].rip) {
            const Insn *nx = &list[i + 1];
            // 1) TEST r, r / CMP r, 0 + JE/JNE -> CBZ/CBNZ (flags dead after Jcc).
            if (in->op == OP_ALU && nx->op == OP_JCC && (nx->cc == 4 || nx->cc == 5) && live_after[i + 1] == 0) {
                int cbz_reg = -1, cbz_sf = 0;
                if (in->sub == ALU_TEST && !in->has_imm && in->rm_reg && !in->high_reg && !in->high_rm && in->reg == in->rm) {
                    if (in->sz == 4 || in->sz == 8) {
                        cbz_reg = in->rm;
                        cbz_sf = in->sz == 8;
                    } else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    } else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    }
                } else if (in->sub == ALU_CMP && in->has_imm && in->imm == 0 && in->rm_reg && !in->high_rm) {
                    if (in->sz == 4 || in->sz == 8) {
                        cbz_reg = in->rm;
                        cbz_sf = in->sz == 8;
                    } else if (in->sz == 1 && (tr_zext8 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    } else if (in->sz == 2 && (tr_zext16 & (1u << in->rm))) {
                        cbz_reg = in->rm;
                        cbz_sf = 0;
                    }
                }
                if (cbz_reg >= 0) {
                    u32 *back = internal_target(nx->target, i);
                    int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                    u32 *direct = (back || fwd >= 0) ? NULL : try_direct_link_target(cur_key, cur_entry_rip, cur_page, nx->target, cur_single);
                    i64 dd = direct ? (direct - tr_cc) : 0;
                    if (direct && (dd < -(1 << 18) || dd >= (1 << 18)))
                        direct = NULL;
                    if (back) {
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, (int)(back - tr_cc));
                    } else if (fwd >= 0) {
                        add_fwd(tr_cc, fwd, 0);
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    } else if (direct) {
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, (int)dd);
                    } else {
                        add_pending(tr_cc, nx->target, SVC_JCC, 0);
                        cbz(nx->cc == 5, cbz_sf, cbz_reg, 0);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    stats.op_insns[nx->op & 127]++;
                    stats.insns++;
                    i++;
                    fused = true;
                } else if (in->sub == ALU_TEST && in->has_imm && (in->rm_reg || !tr_precise_flags)) {
                    // 2) TEST r/m, #(1<<k) + JE/JNE -> TBZ/TBNZ (STRICTLY internal targets only; 14-bit range!).
                    u64 m = in->sz == 1 ? (u8)in->imm : in->sz == 2 ? (u16)in->imm : in->sz == 4 ? (u32)in->imm : (u64)in->imm;
                    if (m != 0 && (m & (m - 1)) == 0) {
                        u32 *back = internal_target(nx->target, i);
                        int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                        bool back_ok = back && (back - tr_cc) >= -(1 << 13) + 16 && (back - tr_cc) < (1 << 13) - 16;
                        bool fwd_ok = false;
                        if (fwd >= 0 && (fwd - (i + 1)) <= 48 && fwd_count < MAX_FWD_FIX) {
                            fwd_ok = true;
                            for (int k = i + 2; k < fwd; k++) {
                                u8 kop = list[k].op;
                                if (kop == OP_EXT || kop == OP_SSE || kop == OP_X87 || kop == OP_STRING) {
                                    fwd_ok = false;
                                    break;
                                }
                            }
                        }
                        if (back_ok || fwd_ok) {
                            int bit = __builtin_ctzll(m);
                            int treg;
                            if (in->rm_reg) {
                                treg = gpr_phys(in->rm, in->high_rm);
                                if (in->high_rm)
                                    bit += 8;
                            } else {
                                ldst(1, in->sz, S0, addr_prepare(in, in->sz));
                                treg = S0;
                            }
                            if (back_ok) {
                                tbz(nx->cc == 5, treg, bit, (int)(back - tr_cc));
                            } else {
                                add_fwd(tr_cc, fwd, 2);
                                tbz(nx->cc == 5, treg, bit, 0);
                            }
                            insn_arm[i + 1] = insn_arm[i];
                            stats.op_insns[nx->op & 127]++;
                            stats.insns++;
                            i++;
                            fused = true;
                        }
                    }
                }
            } else if (in->op == OP_SETCC && in->rm_reg && !in->high_rm &&
                       nx->op == OP_MOVZX && nx->rm_reg && !nx->high_rm &&
                       nx->srcsz == 1 && nx->sz >= 4 && nx->rm == in->rm && nx->reg == in->rm) {
                // 2b) SETcc r8; MOVZX r32/64, r8 (same reg) -> single CSET (1 insn vs 3).
                int acc = x86_to_arm_cc[in->cc];
                if (acc < 0) {
                    emit_parity(nx->reg);
                    if (in->cc == 11)
                        E(0x52000000u | (u32)nx->reg << 5 | (u32)nx->reg); // eor wd, wd, #1
                } else {
                    cset(nx->reg, acc);
                }
                u16 bit = (u16)(1u << nx->reg);
                tr_zext8 |= bit;
                tr_zext16 |= bit;
                insn_arm[i + 1] = insn_arm[i];
                stats.op_insns[nx->op & 127]++;
                stats.insns++;
                i++;
                fused = true;
            } else if (in->op == OP_BT && in->sub == 0 && in->has_imm && in->rm_reg && in->sz >= 4 &&
                       nx->op == OP_JCC && (nx->cc == 2 || nx->cc == 3) && live_after[i + 1] == 0) {
                // 2c) BT r32/64, imm; JC/JNC -> TBNZ/TBZ (internal target within 14-bit range).
                u32 *back = internal_target(nx->target, i);
                int fwd = back ? -1 : forward_target_idx(nx->target, i + 1);
                bool back_ok = back && (back - tr_cc) >= -(1 << 13) + 16 && (back - tr_cc) < (1 << 13) - 16;
                bool fwd_ok = false;
                if (fwd >= 0 && (fwd - (i + 1)) <= 48 && fwd_count < MAX_FWD_FIX) {
                    fwd_ok = true;
                    for (int k = i + 2; k < fwd; k++) {
                        u8 kop = list[k].op;
                        if (kop == OP_EXT || kop == OP_SSE || kop == OP_X87 || kop == OP_STRING) {
                            fwd_ok = false;
                            break;
                        }
                    }
                }
                if (back_ok || fwd_ok) {
                    int bit = (int)((u64)in->imm & (u64)(in->sz * 8 - 1));
                    int nz = nx->cc == 2 ? 1 : 0; // cc==2 is JB/JC (CF==1 -> bit==1 -> TBNZ)
                    if (back_ok) {
                        tbz(nz, in->rm, bit, (int)(back - tr_cc));
                    } else {
                        add_fwd(tr_cc, fwd, 2);
                        tbz(nz, in->rm, bit, 0);
                    }
                    insn_arm[i + 1] = insn_arm[i];
                    stats.op_insns[nx->op & 127]++;
                    stats.insns++;
                    i++;
                    fused = true;
                }
            } else if (!tr_compat && !tr_precise_flags) {
                // 3) PUSH r1; PUSH r2 -> stp r2, r1, [x4, #-16]!
                if (in->op == OP_PUSH && in->rm_reg && !in->has_imm && !in->sub && in->sz == 8 && in->rm != 4 &&
                    nx->op == OP_PUSH && nx->rm_reg && !nx->has_imm && !nx->sub && nx->sz == 8 && nx->rm != 4) {
                    E(0xA9800000u | ((u32)-2 & 0x7fu) << 15 | (u32)in->rm << 10 | 4u << 5 | (u32)nx->rm);
                    tr_zext8 &= (u16)~(1u << 4);
                    tr_zext16 &= (u16)~(1u << 4);
                    insn_arm[i + 1] = insn_arm[i];
                    stats.op_insns[nx->op & 127]++;
                    stats.insns++;
                    i++;
                    fused = true;
                } else if (in->op == OP_POP && in->rm_reg && !in->sub && in->sz == 8 && in->rm != 4 &&
                           nx->op == OP_POP && nx->rm_reg && !nx->sub && nx->sz == 8 && nx->rm != 4 &&
                           in->rm != nx->rm) {
                    // 4) POP r1; POP r2 -> ldp r1, r2, [x4], #16 (r1 != r2 required by ARM64)
                    E(0xA8C00000u | (2u << 15) | (u32)nx->rm << 10 | 4u << 5 | (u32)in->rm);
                    u16 clr = (u16)((1u << 4) | (1u << in->rm) | (1u << nx->rm));
                    tr_zext8 &= (u16)~clr;
                    tr_zext16 &= (u16)~clr;
                    insn_arm[i + 1] = insn_arm[i];
                    stats.op_insns[nx->op & 127]++;
                    stats.insns++;
                    i++;
                    fused = true;
                } else if (in->op == OP_POP && in->rm_reg && !in->sub && in->sz == 8 && in->rm != 4 &&
                           nx->op == OP_RET && nx->imm == 0) {
                    // 5) POP r1; RET -> ldp r1, x16, [x4], #16 + 4-insn RAS return
                    bool use_l1_ret = use_native_l1(tr_cpl, tr_compat);
                    u64 ret_miss_va = use_l1_ret
                                          ? STUB(PRIV_VA + OFF_STUBS + STUB_OFF_RET_MISS_L1)
                                          : STUB(tr_stub_ret_miss);
                    AotStubId ret_sid = use_l1_ret ? AOT_STUB_RET_MISS_L1 : AOT_STUB_RET_MISS;
                    E(0xA8C00000u | (2u << 15) | (u32)S0 << 10 | 4u << 5 | (u32)in->rm);
                    if (cfg.no_return_prediction) {
                        tr_aot_note_stub(tr_cc, ret_sid, false);
                        b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
                    } else {
                        E(0xA8C00000u | 2 << 15 | LR << 10 | RS << 5 | S1); // ldp x17, x30, [x26], #16
                        logic_reg(1, 2, 0, S1, S1, S0);
                        cbz(1, 1, S1, 2);
                        ret_reg(LR);
                        tr_aot_note_stub(tr_cc, ret_sid, false);
                        b_rel((int)((i64)(ret_miss_va - tr_cc_va(tr_cc)) / 4));
                    }
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    ext_ymm_cache_reset();
                    insn_arm[i + 1] = insn_arm[i];
                    stats.op_insns[nx->op & 127]++;
                    stats.insns++;
                    i++;
                    fused = true;
                } else if (in->op == OP_MOV && nx->op == OP_MOV &&
                           !in->has_imm && !nx->has_imm && !in->rm_reg && !nx->rm_reg &&
                           in->sz == nx->sz && (in->sz == 4 || in->sz == 8) &&
                           in->to_reg == nx->to_reg &&
                           !in->riprel && !nx->riprel && !in->no_base && !nx->no_base &&
                           in->no_index && nx->no_index && in->seg == 0 && nx->seg == 0 &&
                           !in->addr32 && !nx->addr32 && !in->addr16 && !nx->addr16 &&
                           in->base == nx->base &&
                           (tr_cpl == 3 || in->base == 4 || in->base == 5)) {
                    // 6) Adjacent MOV [base+d], r1; MOV [base+d+sz], r2 -> STP/LDP (Win64 prologues/epilogues & struct spills).
                    int sz = in->sz;
                    bool fwd_pair = (nx->disp == in->disp + sz);
                    bool rev_pair = (in->disp == nx->disp + sz);
                    if (fwd_pair || rev_pair) {
                        i64 lo = fwd_pair ? in->disp : nx->disp;
                        int rt1 = fwd_pair ? in->reg : nx->reg;
                        int rt2 = fwd_pair ? nx->reg : in->reg;
                        bool load = in->to_reg;
                        bool ldp_ok = !load || (rt1 != rt2 && in->reg != in->base && nx->reg != in->base);
                        if (ldp_ok && (lo & (sz - 1)) == 0 && lo >= -64 * sz && lo <= 63 * sz) {
                            u32 imm7 = (u32)(lo / sz) & 0x7fu;
                            u32 base_opc = sz == 8 ? (load ? 0xA9400000u : 0xA9000000u)
                                                   : (load ? 0x29400000u : 0x29000000u);
                            E(base_opc | imm7 << 15 | (u32)rt2 << 10 | (u32)in->base << 5 | (u32)rt1);
                            if (load) {
                                u16 clr = (u16)((1u << rt1) | (1u << rt2));
                                tr_zext8 &= (u16)~clr;
                                tr_zext16 &= (u16)~clr;
                            }
                            insn_arm[i + 1] = insn_arm[i];
                            stats.op_insns[nx->op & 127]++;
                            stats.insns++;
                            i++;
                            fused = true;
                        }
                    }
                }
            }
        }

        if (!fused) {
            switch (in->op) {
                case OP_JCC:
                case OP_JRCXZ:
                case OP_JMP:
                case OP_CALL:
                case OP_CALLI:
                case OP_JMPI:
                case OP_RET:
                    emit_control(in, i, live_after[i]);
                    if (in->op == OP_CALL || in->op == OP_CALLI || in->op == OP_JMPI || in->op == OP_RET) {
                        ext_ymm_cache_reset();
                        tr_ad_page = ~0ULL;
                        tr_zext8 = tr_zext16 = 0;
                    } else {
                        update_zext(in);
                    }
                    break;
                case OP_BAD:
                    emit_stop(in->rip, in->why);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_UD:
                    svc(SVC_UD);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_SSE:
                case OP_LDMXCSR:
                case OP_STMXCSR:
                    emit_sse(in, live_after[i]);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_X87:
                    emit_x87(in);
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_EXT:
                    emit_ext(in, live_after[i]);
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_SYSCALL:
                case OP_SYSRET:
                case OP_IRET:
                case OP_INT:
                case OP_INT3:
                case OP_HLT:
                case OP_CPUID:
                case OP_RDTSC:
                case OP_RDTSCP:
                case OP_RDMSR:
                case OP_WRMSR:
                case OP_CLI:
                case OP_STI:
                case OP_MOV_CR:
                    if (i && in->reg == 3 && !in->to_reg && in->rm == 0 && !cfg.keep_kernel_tables && tr_cpl == 0) {
                        // XNU's Meltdown fix, from 10.13.2: a process runs on tables holding little of the kernel, and every way in and out changes between them and the full ones,
                        // through two fields of its per-CPU data. Those are the same to a process, which can't read the kernel's pages either way, so it stays on the full tables.
                        // Going out, that is nothing to do. Coming in, the tables may be the other ones if TCG took it out, so look.
                        const Insn *p = &list[i - 1];
                        if (p->op == OP_MOV && p->seg == 2 && p->to_reg && p->reg == 0 && !p->rm_reg) {
                            if (p->disp == 0x138) {
                                tr_zext8 = tr_zext16 = 0;
                                break;
                            }
                            if (p->disp == 0x140) {
                                Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, cr3)};
                                ldst(1, 8, S0, a);
                                logic_sh(1, 2, 0, S0, S0, 0, 0, 0); // eor with rax
                                ubfx(1, S0, S0, 12, 40); // the tables, whatever the PCID
                                u32 *same = tr_cc;
                                cbz(0, 1, S0, 0);
                                emit_monitor_op(in);
                                *same |= ((u32)(tr_cc - same) & 0x7ffff) << 5;
                                ext_ymm_cache_reset();
                                tr_ad_page = ~0ULL;
                                tr_zext8 = tr_zext16 = 0;
                                break;
                            }
                        }
                    }
                    emit_monitor_op(in);
                    ext_ymm_cache_reset();
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                case OP_MOV_DR:
                case OP_INVLPG:
                case OP_INVPCID:
                case OP_WBINVD:
                case OP_CLTS:
                case OP_LGDT:
                case OP_LTR:
                case OP_MOV_SEG:
                case OP_IN:
                case OP_OUT:
                case OP_SWAPGS:
                case OP_SYSENTER:
                case OP_SYSEXIT:
                case OP_FAR:
                case OP_FXSAVE:
                case OP_FXRSTOR:
                    emit_monitor_op(in);
                    ext_ymm_cache_reset();
                    tr_ad_page = ~0ULL;
                    tr_zext8 = tr_zext16 = 0;
                    break;
                default:
                    emit_int(in, live_after[i]);
                    update_zext(in);
                    break;
            }
        }

        // Non-adjacent fallthrough inside Forward-DAG: emit exit/branch before next disconnected sub-block.
        const Insn *end_in = &list[i];
        if (i + 1 < count && !is_uncond_term_op(end_in->op) && end_in->next != list[i + 1].rip) {
            if (tr_aes_mask)
                sse_aes_flush_all();
            if (ext_ymm_cache_has_dirty())
                ext_ymm_cache_flush_dirty();
            u32 *back = internal_target(end_in->next, i);
            int fwd = back ? -1 : forward_target_idx(end_in->next, i);
            if (back) {
                b_rel((int)(back - tr_cc));
            } else if (fwd >= 0) {
                add_fwd(tr_cc, fwd, 1);
                b_rel(0);
            } else {
                u32 *direct = (!is_monitor_or_stop_op(end_in->op) && end_in->op != OP_POPF)
                                  ? try_direct_link_target(cur_key, cur_entry_rip, cur_page, end_in->next, cur_single)
                                  : NULL;
                i64 dd = direct ? (direct - tr_cc) : 0;
                if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
                    b_rel((int)dd);
                } else {
                    u32 *svc_site = tr_cc;
                    svc(SVC_DIRECT);
                    tr_aot_note_cc_branch(svc_site, AOT_CC_DIRECT, tr_cc);
                    stub_lit(end_in->next);
                }
            }
        }
    }
    if (tr_aes_mask)
        sse_aes_flush_all();
    if (ext_ymm_cache_has_dirty())
        ext_ymm_cache_flush_dirty();
    for (int i = 0; i < fwd_count; i++)
        fix(fwd_fix[i].site, fwd_fix[i].form, insn_arm[fwd_fix[i].target_idx]);
    tail = tr_cc;
    stats.op_words[last->op & 127] += (u64)(tr_cc - insn_arm[count - 1]);
    if (fallthrough_end) {
        u32 *direct = (!is_monitor_or_stop_op(last->op) && last->op != OP_POPF)
                          ? try_direct_link_target(cur_key, cur_entry_rip, cur_page, last->next, cur_single)
                          : NULL;
        i64 dd = direct ? (direct - tr_cc) : 0;
        if (direct && dd >= -(1 << 25) && dd < (1 << 25)) {
            b_rel((int)dd);
        } else {
            u32 *svc_site = tr_cc;
            svc(SVC_DIRECT);
            tr_aot_note_cc_branch(svc_site, AOT_CC_DIRECT, tr_cc);
            stub_lit(last->next);
        }
    }
    for (int i = 0; i < pend_count; i++) {
        u32 *stub = tr_cc;
        svc(pend[i].kind);
        u32 *lit_site = tr_cc;
        stub_lit(pend[i].target);
        if (pend[i].kind == SVC_JCC) {
            E((u32)(stub - pend[i].site));
            tr_aot_note_cc_branch(pend[i].site, AOT_CC_JCC, lit_site);
        } else if (pend[i].kind == SVC_CALL) {
            tr_aot_note_cc_branch(pend[i].site, AOT_CC_CALL, lit_site);
        }
        fix(pend[i].site, pend[i].form, stub);
    }
    for (int i = 0; i < l1_miss_count; i++) {
        u32 *miss = tr_cc;
        tr_aot_note_stub(tr_cc, AOT_STUB_LOOKUP_L1, true);
        bl_rel((int)((i64)(STUB(PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1) - tr_cc_va(tr_cc)) / 4));
        if (l1_miss[i].is_call)
            b_rel((int)(l1_miss[i].blr_site - tr_cc));
        else
            br(S2);
        fix(l1_miss[i].site1, 0, miss);
        fix(l1_miss[i].site2, 0, miss);
    }
    compile_inputs_finish();
    (void)tail;

    // How many blocks a cache keyed by frame would have saved, and of those, 32-bit user code's: i386 libraries aren't in SHARED_USER_BASE.
    bool i386_user = tr_compat && cpl == 3;
    if (i386_user)
        stats.i386_blocks++;
    if (res.frames[0] != ~0ULL && res.frames[0] < cfg.ram_size) {
        u32 it = frame_head[res.frames[0] >> 12];
        for (int look = 0; it && look < 32; look++) {
            HostBlock *other = &hblocks[it - 1];
            if (other->rip == rip) {
                if (other->dead)
                    stats.again_dead++;
                else
                    stats.again_live++;
                if (i386_user)
                    stats.i386_again++;
                break;
            }
            it = other->frame[0] == res.frames[0] ? other->next_on_frame[0] : other->next_on_frame[1];
        }
    }

    // Registers the block.
    u32 idx;
    if (cur_part == 0 && p0_ring_enabled) {
        if (p0_block_count[alloc_slot] >= P0_BLOCKS_PER_SLOT) {
            fprintf(stderr, "engine stop: p0 slot %u block table overflow count=%u max=%u\n",
                    alloc_slot, p0_block_count[alloc_slot], P0_BLOCKS_PER_SLOT);
            abort();
        }
        idx = alloc_slot * P0_BLOCKS_PER_SLOT + p0_block_count[alloc_slot]++;
    } else {
        if (hblock_count[cur_part] >= PART_BLOCKS_MAX(cur_part)) {
            fprintf(stderr, "engine stop: block table overflow part=%d count=%u max=%llu\n",
                    cur_part, hblock_count[cur_part], (unsigned long long)PART_BLOCKS_MAX(cur_part));
            abort();
        }
        idx = BLOCK_BASE(cur_part) + hblock_count[cur_part]++;
    }
    HostBlock *hb = &hblocks[idx];
    memset(hb, 0, sizeof *hb);
    hb->rip = rip;
    hb->view = (u32)key; // one reading: the tables can change under this, and which part of the cache it's in goes by this
    hb->entry_word = (u32)(start - cc_base);
    hb->frame[0] = res.frames[0];
    hb->frame[1] = res.frames[1];
    hb->cpl = (u8)cpl;
    hb->p0_slot = (u8)alloc_slot;
    hb->compat = tr_compat;
    hb->first_word = *start;
    hb->t2_has_loop = t2_head_i >= 0;
    hb->t2_patch_word = t2_patch_word;
    hb->on_frame = hb->on_page = hb->on_view = true;
    if (!single)
        remember_reads(hb, rip);

    // Record clean unlinked template into persistent AOT cache BEFORE applying any live direct links.
    u16 aot_t2_patch_off = AOT_NO_PATCH_WORD;
    u16 aot_t2_head_off = 0, aot_t2_min_off = 0, aot_t2_span_bytes = 0;
    if (t2_head_i >= 0 && t2_patch_word != ~0U) {
        int t2_tail_i_pre = t2_head_i;
        for (int k = t2_head_i; k < count; k++) {
            u8 op = list[k].op;
            if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) && tgt_idx[k] == t2_head_i)
                t2_tail_i_pre = k;
        }
        u64 lo = list[t2_head_i].rip, hi = list[t2_head_i].next;
        for (int k = t2_head_i; k <= t2_tail_i_pre; k++) {
            if (list[k].rip < lo) lo = list[k].rip;
            if (list[k].next > hi) hi = list[k].next;
        }
        u64 snap_hi = hi + 64 <= page + 4096 ? hi + 64 : page + 4096;
        if ((lo & ~0xfffULL) == page && hi <= page + 4096 && hi > lo &&
            t2_patch_word >= hb->entry_word && (t2_patch_word - hb->entry_word) < AOT_NO_PATCH_WORD) {
            aot_t2_patch_off = (u16)(t2_patch_word - hb->entry_word);
            aot_t2_head_off = (u16)(list[t2_head_i].rip - rip);
            aot_t2_min_off = (u16)(lo & 0xfffULL);
            aot_t2_span_bytes = (u16)(snap_hi - lo);
        }
    }
    if (tr_aot_active && aot_cache && tr_aot_cap.active) {
        if (!tr_aot_cap.ineligible && !single && res.frames[1] == ~0ULL && res.frames[0] < cfg.ram_size &&
            aot_have_page_hash && fetch_lo >= page && fetch_hi <= page + 4096) {
            AotBlockView rec;
            memset(&rec, 0, sizeof rec);
            rec.key.page_hash = aot_page_hash;
            rec.key.abi_hash = aot_abi_hash;
            rec.key.mode_bits = aot_build_mode_bits(cpl, tr_compat, false);
            rec.key.page_offset = (u16)(rip & 0xFFFULL);
            rec.orig_rip = rip;
            rec.code = start;
            rec.code_words = (u32)(tr_cc - start);
            rec.pcm = &pc_map[map0];
            rec.pcm_count = (u16)(*pcm_cursor_p - map0);
            rec.relocs = tr_aot_cap.relocs;
            rec.reloc_count = (u16)tr_aot_cap.count;
            rec.read_off = hb->read_off;
            rec.read_bytes = hb->read_bytes;
            rec.t2_patch_word_off = aot_t2_patch_off;
            rec.t2_head_off = aot_t2_head_off;
            rec.t2_min_off = aot_t2_min_off;
            rec.t2_span_bytes = aot_t2_span_bytes;
            rec.priority = AOT_PRIO_TIER1;
            if (aot_record(aot_cache, &rec))
                aot_maybe_save_async(aot_cache, 65536U);
            AotLinkCtx lc = { .key = key, .entry_rip = rip, .page = page };
            aot_apply_live_eager_links(start, rec.code_words, &tr_aot_cap, &lc);
        }
        tr_aot_cap.active = false;
    }

    link_block(idx);
    BlockEntry *be = &block_table[cur_part ? BLOCK_TABLE_PART2 + (idx - BLOCK_BASE(1)) : idx];
    be->rip = rip;
    be->key = hb->view;
    be->arm_off = (u32)(start - cc_base);
    be->arm_words = (u32)(tr_cc - start);
    be->map_off = map0;
    be->map_count = *pcm_cursor_p - map0;
    if (!single) {
        map_insert((int)hb->view, rip, idx + 1);
        hb->registered = true;
    }
    __sync_synchronize();
    if (cur_part == 0 && p0_ring_enabled) {
        __atomic_store_n(&ctx_ptr->p0_slot_block_count[alloc_slot], (u64)p0_block_count[alloc_slot], __ATOMIC_RELEASE);
        p0_cc_cursor[alloc_slot] = tr_cc;
    } else {
        if (cur_part == 0)
            __atomic_store_n(&ctx_ptr->block_count, (u64)hblock_count[0], __ATOMIC_RELEASE);
        else
            __atomic_store_n(&ctx_ptr->block_count_private, (u64)hblock_count[1], __ATOMIC_RELEASE);
        cc_cursor[cur_part] = tr_cc;
    }
    sys_icache_invalidate(start, (u8 *)tr_cc - (u8 *)start);
    code_published();
    if (t2_head_i >= 0 && t2_patch_word != ~0U) {
        int t2_tail_i = t2_head_i;
        for (int k = t2_head_i; k < count; k++) {
            u8 op = list[k].op;
            if ((op == OP_JCC || op == OP_JRCXZ || op == OP_JMP) && tgt_idx[k] == t2_head_i)
                t2_tail_i = k;
        }
        u64 lo = list[t2_head_i].rip, hi = list[t2_head_i].next;
        for (int k = t2_head_i; k <= t2_tail_i; k++) {
            if (list[k].rip < lo) lo = list[k].rip;
            if (list[k].next > hi) hi = list[k].next;
        }
        // Include up to 64 post-loop same-page bytes in the SMC-verified snapshot so Tier-2 can peek exit flag liveness!
        u64 snap_hi = hi + 64 <= page + 4096 ? hi + 64 : page + 4096;
        if ((lo & ~0xfffULL) == page && hi <= page + 4096 && hi > lo) {
            u64 ep = __atomic_load_n(&t2_epoch[cur_part], __ATOMIC_RELAXED);
            t2_register_loop(idx, (u32)cur_part, alloc_slot, hb->entry_word, be->arm_words,
                             t2_patch_word, rip, list[t2_head_i].rip, res.frames[0],
                             (u16)(lo & 0xfffULL), (u16)(snap_hi - lo), (u32)key, (u8)cpl, tr_compat, ep);
            if (t2_eager_mode())
                t2_eager_promote_now(idx);
        }
    }
    stats.blocks++;
    stats.words += tr_cc - start;
    stats.exit_words += (u64)(tr_cc - tail);
    res.arm = tr_cc_va(start);
    stats.translate_ns += mach_absolute_time() - t0;
    return res;
}

// Privileged and rare instructions run in the monitor. Views are per CPL, so CPL-3 code simply takes #GP where it must.
static void emit_monitor_op(const Insn *in) {
    bool cpl0_only = false;
    switch (in->op) {
        case OP_HLT:
        case OP_RDMSR:
        case OP_WRMSR:
        case OP_MOV_CR:
        case OP_MOV_DR:
        case OP_INVLPG:
        case OP_INVPCID:
        case OP_WBINVD:
        case OP_CLTS:
        case OP_SYSRET:
        case OP_SYSEXIT:
        case OP_SWAPGS:
            cpl0_only = true;
            break;
        case OP_LGDT:
            cpl0_only = in->sub <= 1; // sgdt and sidt are not privileged without UMIP, which isn't offered: libsystem reads the CPU number with sidt
            break;
        case OP_LTR:
            cpl0_only = in->sub == 0 || in->sub == 2;
            break;
        default:
            break;
    }
    if (cpl0_only && tr_cpl != 0) {
        svc(SVC_GP0);
        return;
    }
    switch (in->op) {
        case OP_SYSCALL:
            svc(tr_compat ? SVC_UD : SVC_SYSCALL); // as Intel's parts, and TCG with Intel's vendor: no CSTAR path. In hybrid mode TCG decides anyway
            break;
        case OP_SYSENTER:
            svc(SVC_SYSENTER);
            break;
        case OP_SYSEXIT:
            movimm(S0, in->sz == 8);
            svc(SVC_SYSEXIT);
            break;
        case OP_FAR:
            if (in->has_imm) {
                movimm(S0, (u64)in->imm);
                movimm(S1, (u64)in->disp);
            } else if (in->sub) { // both words before anything changes
                Addr a = {.base = addr_reg(in)};
                ldst(1, in->sz, S0, a);
                a.imm = in->sz;
                ldst(1, 2, S1, a);
            }
            movimm(S2, (u64)in->sub | (u64)in->sz << 8 | (u64)(in->has_imm ? 0 : in->imm) << 16);
            svc(SVC_FAR);
            break;
        case OP_SYSRET:
            mov_reg(1, S0, XZR);
            movimm(S0, in->sz == 8);
            svc(SVC_SYSRET);
            break;
        case OP_IRET:
            movimm(S0, in->sz);
            svc(SVC_IRET);
            break;
        case OP_INT:
            movimm(S1, (u64)in->imm);
            svc(SVC_INT);
            break;
        case OP_INT3:
            svc(SVC_INT3);
            break;
        case OP_HLT:
            svc(SVC_HLT);
            break;
        case OP_CPUID: {
            u32 *to_slow1 = tr_cc;
            cbz(1, 0, 0, 0); // cbnz w0, slow
            Addr ac = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, cpuid0_cached)};
            ldst(1, 8, S0, ac);
            u32 *to_slow2 = tr_cc;
            cbz(0, 1, S0, 0); // cbz xS0, slow
            Addr a0 = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, cpuid0_eax_ebx)};
            ldst(1, 8, S0, a0);
            mov_reg(0, 0, S0);
            lsr_imm(1, 3, S0, 32);
            Addr a1 = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, cpuid0_ecx_edx)};
            ldst(1, 8, S0, a1);
            mov_reg(0, 1, S0);
            lsr_imm(1, 2, S0, 32);
            u32 *to_done = tr_cc;
            b_rel(0);
            u32 *slow = tr_cc;
            svc(SVC_CPUID);
            *to_slow1 |= ((u32)(slow - to_slow1) & 0x7ffffu) << 5;
            *to_slow2 |= ((u32)(slow - to_slow2) & 0x7ffffu) << 5;
            *to_done |= (u32)(tr_cc - to_done) & 0x3ffffffu;
            break;
        }
        case OP_RDTSC: {
            // tsc_now(), in place. CNTKCTL lets EL0 read the virtual counter, and nothing else runs ARM at EL0.
            Addr a = {.base = CTX};
            E(0xD53BE040u | S0); // mrs x, cntvct_el0
            a.imm = __builtin_offsetof(EngineCtx, tsc_counter_offset);
            ldst(1, 8, S1, a);
            add_reg(1, 0, 0, S0, S0, S1, 0);
            movimm(S1, 125);
            mul(1, S0, S0, S1);
            movimm(S1, 3);
            udiv(1, S0, S0, S1);
            a.imm = __builtin_offsetof(EngineCtx, tsc_ns_offset);
            ldst(1, 8, S1, a);
            add_reg(1, 0, 0, S0, S0, S1, 0);
            a.imm = __builtin_offsetof(EngineCtx, tsc_num);
            ldst(1, 8, S1, a);
            mul(1, S1, S0, S1);
            a.imm = __builtin_offsetof(EngineCtx, tsc_den);
            ldst(1, 8, S2, a);
            udiv(1, S1, S1, S2);
            add_reg(1, 0, 0, S0, S0, S1, 0);
            a.imm = __builtin_offsetof(EngineCtx, tsc_offset);
            ldst(1, 8, S1, a);
            add_reg(1, 0, 0, S0, S0, S1, 0);
            mov_reg(0, 0, S0);
            lsr_imm(1, 2, S0, 32);
            break;
        }
        case OP_RDTSCP:
            svc(SVC_RDTSCP);
            break;
        case OP_RDMSR:
            svc(SVC_RDMSR);
            break;
        case OP_WRMSR:
            svc(SVC_WRMSR);
            break;
        case OP_CLI: {
            if (tr_cpl != 0) { // IOPL is the monitor's to check
                svc(SVC_CLI);
                break;
            }
            Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, rflags_sys)};
            ldst(1, 8, S0, a);
            movimm(S1, 0x200);
            logic_sh(1, 0, 1, S0, S0, S1, 0, 0); // bic
            ldst(0, 8, S0, a);
            break;
        }
        case OP_STI: {
            if (tr_cpl != 0 || !tr_hybrid) { // the in-VM APIC's vectors are the monitor's to look at
                svc(SVC_STI);
                break;
            }
            // Opens IF and the one-instruction shadow, as the monitor does. With an interrupt already waiting it goes on through the monitor, whose
            // way out takes it; one that arrives later raises an IRQ here, which goes there too.
            Addr a = {.base = CTX, .imm = __builtin_offsetof(EngineCtx, rflags_sys)};
            ldst(1, 8, S0, a);
            u32 *already = tr_cc;
            E(0x37000000u | 9u << 19 | S0); // tbnz x, #9
            movimm(S1, 0x200);
            logic_sh(1, 1, 0, S0, S0, S1, 0, 0);
            ldst(0, 8, S0, a);
            tr_emit_movimm_gva(S1, in->next, tr_compat);
            a.imm = __builtin_offsetof(EngineCtx, irq_shadow_rip);
            ldst(0, 8, S1, a);
            a.imm = __builtin_offsetof(EngineCtx, irq_pending);
            ldst(1, 4, S0, a);
            a.imm = __builtin_offsetof(EngineCtx, irq_line);
            ldst(1, 4, S1, a);
            logic_sh(0, 1, 0, S0, S0, S1, 0, 0);
            u32 *quiet = tr_cc;
            cbz(0, 0, S0, 0);
            svc(SVC_STI);
            *already |= ((u32)(tr_cc - already) & 0x3fff) << 5;
            *quiet |= ((u32)(tr_cc - quiet) & 0x7ffff) << 5;
            break;
        }
        case OP_MOV_CR: {
            int cr = in->reg;
            if (cr != 0 && cr != 2 && cr != 3 && cr != 4 && cr != 8) {
                svc(SVC_UD);
                break;
            }
            if (in->to_reg) { // the context holds them, and only the monitor/fast-path changes them
                Addr a = {.base = CTX};
                if (cr == 8) {
                    a.imm = __builtin_offsetof(EngineCtx, apic_tpr);
                    ldst(1, 4, in->rm, a);
                    lsr_imm(1, in->rm, in->rm, 4);
                    break;
                }
                a.imm = cr == 0   ? __builtin_offsetof(EngineCtx, cr0)
                        : cr == 2 ? __builtin_offsetof(EngineCtx, cr2)
                        : cr == 3 ? __builtin_offsetof(EngineCtx, cr3)
                                  : __builtin_offsetof(EngineCtx, cr4);
                ldst(1, 8, in->rm, a);
                break;
            }
            if (cr == 8 && tr_hybrid) {
                // Fast-path CR8/TPR write without EL1 trap; preserves NZCV.
                ubfx(0, S1, in->rm, 0, 4);
                lsl_imm(0, S1, S1, 4);
                movimm(S2, __builtin_offsetof(EngineCtx, apic_tpr));
                add_reg(1, 0, 0, S2, CTX, S2, 0);
                ldst(1, 4, S0, (Addr){.base = S2});
                logic_reg(0, 2, 0, S3, S1, S0); // eor wS3, new_tpr, old_tpr
                u32 *same = tr_cc;
                cbz(0, 0, S3, 0);
                E(0x889FFC00u | (u32)S2 << 5 | (u32)S1); // stlr wS1, [xS2]
                add_reg(0, 1, 0, S3, S1, S0, 0); // sub wS3, new_tpr, old_tpr (bit 31 set iff new < old)
                u32 *raised = tr_cc;
                tbz(0, S3, 31, 0);
                movimm(S2, __builtin_offsetof(EngineCtx, apic_irr_prio));
                add_reg(1, 0, 0, S2, CTX, S2, 0);
                E(0x88DFFC00u | (u32)S2 << 5 | (u32)S0); // ldar wS0, [xS2]
                add_reg(0, 1, 0, S3, S1, S0, 0); // sub wS3, new_tpr, irr_prio (bit 31 set iff new < irr)
                u32 *masked = tr_cc;
                tbz(0, S3, 31, 0);
                movimm(S0, 8);
                mov_reg(1, S1, in->rm);
                svc(SVC_MOV_CR_WRITE);
                *same |= ((u32)(tr_cc - same) & 0x7ffffu) << 5;
                *raised |= ((u32)(tr_cc - raised) & 0x3fffu) << 5;
                *masked |= ((u32)(tr_cc - masked) & 0x3fffu) << 5;
                break;
            }
            movimm(S0, (u64)cr);
            mov_reg(1, S1, in->rm);
            svc(SVC_MOV_CR_WRITE);
            break;
        }
        case OP_MOV_DR:
            emit_stop(in->rip, "debug registers");
            break;
        case OP_INVLPG:
        case OP_WBINVD:
            // CPL != 0 already trapped #GP(0) above; shadow tables stay coherent via PT write-protection.
            break;
        case OP_INVPCID:
            svc(SVC_INVPCID);
            break;
        case OP_CLTS:
            svc(SVC_CLTS);
            break;
        case OP_LGDT: {
            // Limit word, then base. The load comes before anything changes.
            int r = addr_reg(in);
            if (in->sub <= 1) {
                Addr a = {.base = r};
                ldst(1, 2, S1, a);
                a.imm = 2;
                ldst(1, tr_compat ? 4 : 8, S0, a);
                if (tr_compat && in->sz == 2)
                    ubfx(0, S0, S0, 0, 24);
                svc(in->sub == 0 ? SVC_LGDT : SVC_LIDT);
            } else {
                mov_reg(1, S2, r);
                Addr from = {.base = CTX, .imm = in->sub == 2 ? __builtin_offsetof(EngineCtx, gdtr_base) : __builtin_offsetof(EngineCtx, idtr_base)};
                ldst(1, 8, S0, from);
                from.imm = in->sub == 2 ? __builtin_offsetof(EngineCtx, gdtr_limit) : __builtin_offsetof(EngineCtx, idtr_limit);
                ldst(1, 4, S1, from);
                Addr a = {.base = S2};
                ldst(0, 2, S1, a);
                a.imm = 2;
                ldst(0, tr_compat ? 4 : 8, S0, a);
            }
            break;
        }
        case OP_LTR: {
            int mode = in->sub; // 0 ltr, 1 str, 2 lldt, 3 sldt, 4 verr, 5 verw, 6 lsl
            if (mode == 6) {
                if (in->rm_reg)
                    mov_reg(0, S0, in->rm);
                else
                    ldst(1, 2, S0, addr_prepare(in, 2));
                svc(SVC_LSL);
                u32 *skip = tr_cc;
                b_cond(1, 0); // skip write if ZF == 0 (invalid selector)
                gpr_write(in->reg, in->sz, false, S0);
                *skip |= (u32)(tr_cc - skip) << 5;
                break;
            }
            bool read = mode == 1 || mode == 3;
            int svcn = mode == 0 ? SVC_LTR : mode == 1 ? SVC_STR : mode == 2 ? SVC_LLDT : mode == 3 ? SVC_SLDT : mode == 4 ? SVC_VERR : SVC_VERW;
            if (read) {
                svc(svcn);
                if (in->rm_reg)
                    gpr_write(in->rm, in->sz, false, S0);
                else
                    ldst(0, 2, S0, addr_prepare(in, 2));
            } else {
                if (in->rm_reg)
                    mov_reg(1, S0, in->rm);
                else
                    ldst(1, 2, S0, addr_prepare(in, 2));
                svc(svcn);
            }
            break;
        }
        case OP_MOV_SEG: {
            if (in->pop_seg) { // the monitor moves the stack, once the load is accepted
                movimm(S0, (u64)in->sub | (u64)in->sz << 8);
                Addr a = {.base = 4};
                ldst(1, in->sz, S1, a);
                svc(SVC_MOV_SEG_WRITE);
            } else if (in->to_reg) {
                Addr from = {.base = CTX, .imm = (i64)__builtin_offsetof(EngineCtx, seg_sel) + in->sub * 2};
                ldst(1, 2, S0, from);
                if (in->rm_reg)
                    gpr_write(in->rm, in->sz == 2 ? 2 : in->sz, false, S0);
                else
                    ldst(0, 2, S0, addr_prepare(in, 2));
            } else {
                movimm(S0, in->sub);
                if (in->rm_reg)
                    mov_reg(1, S1, in->rm);
                else
                    ldst(1, 2, S1, addr_prepare(in, 2));
                svc(SVC_MOV_SEG_WRITE);
            }
            break;
        }
        case OP_IN:
        case OP_OUT: {
            if (in->has_imm)
                movimm(S0, (u64)in->imm & 0xff);
            else
                ubfx(0, S0, 2, 0, 16);
            movimm(S1, in->sz);
            if (in->op == OP_IN) {
                svc(SVC_IN);
                gpr_write(0, in->sz, false, S0);
            } else {
                mov_reg(1, S2, 0);
                svc(SVC_OUT);
            }
            break;
        }
        case OP_SWAPGS:
            E(0xF9400000u | (u32)(__builtin_offsetof(EngineCtx, kernel_gs_base) / 8) << 10 | CTX << 5 | S0);
            E(0xF9000000u | (u32)(__builtin_offsetof(EngineCtx, kernel_gs_base) / 8) << 10 | CTX << 5 | R_GS);
            mov_reg(1, R_GS, S0);
            break;
        case OP_FXSAVE:
        case OP_FXRSTOR: {
            // Probe both ends, so the monitor's access cannot fault for the guest's reasons. TS comes before any fault, hence the NEON word first.
            E(0x4F00E400u | V_MEM);
            int r = addr_reg(in);
            if (r != S0)
                mov_reg(1, S0, r);
            movimm(S2, (u64)(in->sz == 8) | ((u64)in->sub << 1));
            ubfx(1, S1, S0, 0, in->sub ? 6 : 4);
            cbz(0, 1, S1, 2);
            svc(SVC_GP0);
            Addr a = {.base = S0};
            ldst(1, 8, XZR, a);
            a.imm = 504;
            ldst(1, 8, XZR, a);
            if (in->sub) {
                a.imm = 568;
                ldst(1, 8, XZR, a);
                u32 *no_ymm_r = tr_cc;
                tbz(0, 0, 2, 0); // skip probing offset 824 if EAX[2] == 0
                a.imm = 824;
                ldst(1, 8, XZR, a);
                *no_ymm_r |= ((u32)(tr_cc - no_ymm_r) & 0x3fff) << 5;
            }
            if (in->op == OP_FXSAVE) {
                a.imm = 0;
                ldst(1, 8, S1, a);
                ldst(0, 8, S1, a);
                a.imm = 504;
                ldst(1, 8, S1, a);
                ldst(0, 8, S1, a);
                if (in->sub) {
                    a.imm = 568;
                    ldst(1, 8, S1, a);
                    ldst(0, 8, S1, a);
                    u32 *no_ymm_w = tr_cc;
                    tbz(0, 0, 2, 0); // skip probing offset 824 if EAX[2] == 0
                    a.imm = 824;
                    ldst(1, 8, S1, a);
                    ldst(0, 8, S1, a);
                    *no_ymm_w |= ((u32)(tr_cc - no_ymm_w) & 0x3fff) << 5;
                }
            }
            svc(in->op == OP_FXSAVE ? SVC_FXSAVE : SVC_FXRSTOR);
            break;
        }
        default:
            emit_stop(in->rip, "no monitor path for this instruction");
    }
}

// Read-only profiler helper: fetch original x86 bytes of a block even after the guest process
// has exited and Windows has torn down its CR3 page tables (uses read_copies or HostBlock.frame[]).
size_t tr_profile_read_block_x86(uint32_t tbl_idx, bool priv_part, bool is_t2,
                                 uint64_t rip, uint8_t *dst, size_t max_len) {
    if (!dst || max_len == 0 || !hblocks)
        return 0;
    u32 hidx = ~0u;
    if (is_t2) {
        int part = priv_part ? 1 : 0;
        if (!t2_meta[part] || tbl_idx >= T2_BLOCKS_MAX)
            return 0;
        hidx = t2_meta[part][tbl_idx].t1_idx;
    } else {
        hidx = (priv_part ? (u32)BLOCKS_MAX : 0u) + tbl_idx;
    }
    if (hidx >= (u32)(BLOCKS_MAX + BLOCKS_MAX_PRIV))
        return 0;
    const HostBlock *hb = &hblocks[hidx];
    if (hb->read_bytes > 0 && read_copies && hb->rip <= rip) {
        u64 delta = rip - hb->rip;
        u32 base_off = (u32)(hb->rip & 0xfffu);
        if (delta < hb->read_bytes && base_off >= hb->read_off) {
            u32 rc_delta = (base_off - hb->read_off) + (u32)delta;
            if (rc_delta < hb->read_bytes) {
                size_t avail = (size_t)(hb->read_bytes - rc_delta);
                if (avail > max_len) avail = max_len;
                memcpy(dst, read_copies + hb->read_copy + rc_delta, avail);
                return avail;
            }
        }
    }
    if (hb->frame[0] != ~0ULL && cfg.ram) {
        size_t done = 0;
        u64 host_ram_bytes = cfg.ram_below_4g ? (cfg.ram_size - ((1ULL << 32) - cfg.ram_below_4g)) : cfg.ram_size;
        u64 off0 = rip & 0xfffULL;
        size_t chunk0 = (size_t)(4096ULL - off0);
        if (chunk0 > max_len) chunk0 = max_len;
        u64 gpa0 = hb->frame[0] + off0;
        bool in_hole0 = cfg.ram_below_4g && gpa0 >= cfg.ram_below_4g && gpa0 < (1ULL << 32);
        if (!in_hole0 && gpa0 + chunk0 <= cfg.ram_size) {
            u64 h0 = (cfg.ram_below_4g && gpa0 >= (1ULL << 32)) ? (gpa0 - (1ULL << 32) + cfg.ram_below_4g) : gpa0;
            if (h0 + chunk0 <= host_ram_bytes) {
                memcpy(dst, cfg.ram + h0, chunk0);
                done = chunk0;
            }
        }
        if (done < max_len && hb->frame[1] != ~0ULL) {
            size_t chunk1 = max_len - done;
            if (chunk1 > 4096U) chunk1 = 4096U;
            u64 gpa1 = hb->frame[1];
            bool in_hole1 = cfg.ram_below_4g && gpa1 >= cfg.ram_below_4g && gpa1 < (1ULL << 32);
            if (!in_hole1 && gpa1 + chunk1 <= cfg.ram_size) {
                u64 h1 = (cfg.ram_below_4g && gpa1 >= (1ULL << 32)) ? (gpa1 - (1ULL << 32) + cfg.ram_below_4g) : gpa1;
                if (h1 + chunk1 <= host_ram_bytes) {
                    memcpy(dst + done, cfg.ram + h1, chunk1);
                    done += chunk1;
                }
            }
        }
        return done;
    }
    return 0;
}

void tr_init(const TrConfig *config) {
    const char *no_dag = getenv("LC_HVDBT_NO_NATIVE_DAG");
    native_dag = !(no_dag && !strcmp(no_dag, "1"));
    const char *no_l1 = getenv("LC_HVDBT_NO_NATIVE_L1");
    native_l1 = !(no_l1 && !strcmp(no_l1, "1"));
    const char *no_peep = getenv("LC_HVDBT_NO_NATIVE_PEEP");
    tr_opt_peep = !(no_peep && !strcmp(no_peep, "1"));
    const char *no_ymm = getenv("LC_HVDBT_NO_YMM_CACHE");
    tr_opt_ymm_cache = !(no_ymm && !strcmp(no_ymm, "1"));
    const char *no_ring = getenv("LC_HVDBT_NO_P0_RING");
    const char *no_par = getenv("LC_HVDBT_NO_PAR_TRANSLATE");
    par_translate_enabled = !(no_par && !strcmp(no_par, "1"));
    no_uframe_share = getenv("LC_HVDBT_NO_USER_FRAME_SHARE") != NULL;
    x86_decode_init();
    memset(tr_frame_inval_gen, 0, sizeof tr_frame_inval_gen);
    tr_pt_inval_gen = 0;
    tr_flush_gen = 0;
    cfg = *config;
    if (cfg.profile_invalidations && (mach_timebase_info(&invalidation_timebase) != KERN_SUCCESS || !invalidation_timebase.denom))
        cfg.profile_invalidations = false;
    stats.invalidation_profile_enabled = cfg.profile_invalidations;
    cc_base = (u32 *)(cfg.priv + OFF_CC);
    tr_cc = cc_base;
    cc_cursor[0] = cc_base;
    cc_cursor[1] = cc_base + CC_PART2_WORD;
    block_table = (BlockEntry *)(cfg.priv + OFF_BLOCKS);
    pc_map = (PcMap *)(cfg.priv + OFF_PCMAP);
    t2_block_table = (BlockEntry *)(cfg.priv + OFF_BLOCKS_T2);
    t2_pc_map = (PcMap *)(cfg.priv + OFF_PCMAP_T2);
    t2_trap_meta = (T2TrapMeta *)(cfg.priv + OFF_META_T2);
    ctx_ptr = (EngineCtx *)(cfg.priv + OFF_CTX);
    compile_inputs_init();
    tr_precise_flags = cfg.precise_flags;
    tr_has_tzcnt = cfg.has_tzcnt_lzcnt;
    tr_hybrid = cfg.hybrid;
    tr_no_bulk_rep = cfg.no_bulk_rep;
    tr_profile_shift_helpers = cfg.profile_shift_helpers;
    tr_inline_shr8 = cfg.inline_shr8;
    tr_page_rep = cfg.page_rep;
    stats.inline_shr8_enabled = cfg.inline_shr8;
    stats.page_rep_enabled = cfg.page_rep;
    stats.retain_shared_dead_enabled = cfg.retain_shared_dead;
    stats.shared_dead_retained_hints = stats.shared_dead_cleared_hints = 0;
    cc_limit = cfg.code_cache_bytes ? cfg.code_cache_bytes : CC_SIZE;
    p0_ring_enabled = !(no_ring && !strcmp(no_ring, "1")) && cc_limit >= CC_SIZE;
    p0_cur_slot = 0;
    p0_gen_seq = 1;
    p0_ring_need_evict = false;
    p0_ring_evictions = 0;
    for (u32 s = 0; s < P0_SLOTS; s++) {
        p0_cc_cursor[s] = cc_base + engine_p0_slot_word_base(s);
        p0_block_count[s] = 0;
        p0_pcmap_used[s] = s * P0_PCMAP_PER_SLOT;
        p0_slot_gen[s] = s == 0 ? 1ULL : 0ULL;
        ctx_ptr->p0_slot_gen[s] = p0_slot_gen[s];
        ctx_ptr->p0_slot_block_count[s] = 0;
    }
    ctx_ptr->p0_ring_active = p0_ring_enabled ? 1U : 0U;
    cc_t2_cursor[0] = cc_base + CC_T2_WORD0;
    cc_t2_cursor[1] = cc_base + CC_T2_WORD1;
    t2_block_cnt[0] = t2_block_cnt[1] = 0;
    t2_pcmap_used[0] = 0;
    t2_pcmap_used[1] = T2_PCMAP_ENTRIES;
    t2_epoch[0] = t2_epoch[1] = 1;
    ctx_ptr->t2_block_count[0] = ctx_ptr->t2_block_count[1] = 0;
    free(t2_meta[0]);
    free(t2_meta[1]);
    t2_meta[0] = calloc(T2_BLOCKS_MAX, sizeof(T2BlockMeta));
    t2_meta[1] = calloc(T2_BLOCKS_MAX, sizeof(T2BlockMeta));
    tr_stub_lookup = cfg.stub_lookup;
    tr_stub_ret_miss = cfg.stub_ret_miss;
    tr_stub_pushf = cfg.stub_pushf;
    tr_stub_sahf = cfg.stub_sahf;
    // Always populate stub VA table so parallel stub relocation works with or without persistent AOT cache.
    aot_stubs.va[AOT_STUB_LOOKUP] = tr_stub_lookup;
    aot_stubs.va[AOT_STUB_RET_MISS] = tr_stub_ret_miss;
    aot_stubs.va[AOT_STUB_PUSHF] = tr_stub_pushf;
    aot_stubs.va[AOT_STUB_SAHF] = tr_stub_sahf;
    aot_stubs.va[AOT_STUB_LOOKUP_L1] = PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1;
    aot_stubs.va[AOT_STUB_LOOKUP_L1_FAST] = PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1_FAST;
    aot_stubs.va[AOT_STUB_RET_MISS_L1] = PRIV_VA + OFF_STUBS + STUB_OFF_RET_MISS_L1;
    aot_stubs.va[AOT_STUB_REP_MOVS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_MOVS;
    aot_stubs.va[AOT_STUB_REP_STOS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_STOS;
    aot_stubs.va[AOT_STUB_REP_PAGE_MOVS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_PAGE_MOVS;
    aot_stubs.va[AOT_STUB_REP_PAGE_STOS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_PAGE_STOS;
    stats.par_enabled = par_translate_enabled ? 1ULL : 0ULL;
    hblocks = calloc(BLOCKS_MAX + BLOCKS_MAX_PRIV, sizeof(HostBlock));
    free(shared_hash_views);
    shared_hash_views = NULL;
    stats.invalidation_membership_enabled = cfg.invalidation_membership;
    stats.invalidation_membership_reliable = 0;
    stats.invalidation_membership_bytes = 0;
    if (cfg.invalidation_membership) {
        shared_hash_views = calloc(BLOCKS_MAX, sizeof(*shared_hash_views));
        stats.invalidation_membership_reliable = shared_hash_views != NULL;
        stats.invalidation_membership_bytes = shared_hash_views ? BLOCKS_MAX * sizeof(*shared_hash_views) : 0;
        if (!shared_hash_views)
            stats.invalidation_membership_tracking_failures++;
    }
    block_map = calloc(MAP_SIZE, sizeof(MapEntry));
    frame_head = calloc(cfg.ram_size >> 12, sizeof(u32));
    dead_slots = calloc(DEAD_SLOTS, sizeof(u32));
    read_copies = mmap(NULL, READ_COPY_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    page_head = calloc(PAGE_HEADS, sizeof(u32));
    t2_init(&cfg, ctx_ptr, cc_base);
    tr_profile_init_mem(cfg.priv, cfg.ram, cfg.ram_size, cfg.ram_below_4g, ctx_ptr, NULL);

    // Optional persistent AOT code cache (default OFF unless LC_HVDBT_AOT_CACHE=<path>).
    if (aot_cache) {
        aot_close(aot_cache, true);
        aot_cache = NULL;
        tr_aot_active = false;
    }
    const char *aot_path = getenv("LC_HVDBT_AOT_CACHE");
    if (aot_path && aot_path[0]) {
        const char *max_mb_s = getenv("LC_HVDBT_AOT_MAX_MB");
        const char *stamp_s = getenv("LC_HVDBT_AOT_BUILD_STAMP");
        const char *no_eager = getenv("LC_HVDBT_AOT_NO_EAGER_LINK");
        u64 max_bytes = AOT_DEFAULT_MAX_BYTES;
        if (max_mb_s && max_mb_s[0]) {
            long mb = atol(max_mb_s);
            if (mb >= 1 && mb <= 4096)
                max_bytes = (u64)mb << 20;
        }
        u64 stamp = (stamp_s && stamp_s[0]) ? strtoull(stamp_s, NULL, 0) : 0ULL;
        aot_eager_link = !(no_eager && !strcmp(no_eager, "1"));
        aot_abi_hash = aot_compute_abi_hash(stamp);
        aot_stubs.va[AOT_STUB_LOOKUP] = tr_stub_lookup;
        aot_stubs.va[AOT_STUB_RET_MISS] = tr_stub_ret_miss;
        aot_stubs.va[AOT_STUB_PUSHF] = tr_stub_pushf;
        aot_stubs.va[AOT_STUB_SAHF] = tr_stub_sahf;
        aot_stubs.va[AOT_STUB_LOOKUP_L1] = PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1;
        aot_stubs.va[AOT_STUB_LOOKUP_L1_FAST] = PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1_FAST;
        aot_stubs.va[AOT_STUB_RET_MISS_L1] = PRIV_VA + OFF_STUBS + STUB_OFF_RET_MISS_L1;
        aot_stubs.va[AOT_STUB_REP_MOVS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_MOVS;
        aot_stubs.va[AOT_STUB_REP_STOS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_STOS;
        aot_stubs.va[AOT_STUB_REP_PAGE_MOVS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_PAGE_MOVS;
        aot_stubs.va[AOT_STUB_REP_PAGE_STOS] = PRIV_VA + OFF_STUBS + STUB_OFF_REP_PAGE_STOS;
        aot_cache = aot_open(aot_path, aot_abi_hash, max_bytes);
        tr_aot_active = (aot_cache != NULL);
        if (tr_aot_active && !aot_atexit_registered) {
            atexit(tr_aot_atexit_flush);
            aot_atexit_registered = true;
        }
    }
}

bool tr_t2_publish(u32 t1_idx, u32 part, u32 p0_slot, u64 start_epoch, u64 head_rip,
                   u32 key, u8 cpl, bool compat, u32 patch_word, u64 frame0,
                   u16 snap_off, u16 snap_len, const u8 *snap_bytes,
                   const u32 *code, u32 code_words, const PcMap *pcm, u32 pcm_count,
                   const T2TrapMeta *trap_meta, const u32 *bail_sites, u32 bail_cnt) {
    if (part > 1 || !code_words || !pcm_count || !t2_meta[part])
        return false;
    pthread_mutex_lock(&t2_pub_lock);
    bool ok = false;
    if (t2_epoch[part] != start_epoch)
        goto out;
    if (t1_idx >= BLOCKS_MAX + BLOCKS_MAX_PRIV)
        goto out;
    HostBlock *hb = &hblocks[t1_idx];
    if (hb->dead || hb->t2_entry_word != 0 || hb->t2_patch_word != patch_word || hb->frame[0] != frame0)
        goto out;
    if (frame0 >= cfg.ram_size || memcmp(ram_at(frame0 + snap_off), snap_bytes, snap_len) != 0)
        goto out;
    u32 t2_word_base = part ? CC_T2_WORD1 : CC_T2_WORD0;
    u32 *t2_end = cc_base + t2_word_base + CC_T2_WORDS;
    u32 *t2_start = cc_t2_cursor[part];
    u32 pcm_base = part ? T2_PCMAP_ENTRIES : 0;
    if (t2_start + code_words > t2_end ||
        t2_block_cnt[part] >= T2_BLOCKS_MAX ||
        t2_pcmap_used[part] + pcm_count > pcm_base + T2_PCMAP_ENTRIES)
        goto out;
    u32 *t1_patch = cc_base + patch_word;
    i64 dist = t2_start - t1_patch;
    if (dist < -(1 << 25) || dist >= (1 << 25))
        goto out;
    // Validate all alias-guard Tier-1 fallback branch sites fit in ±128 MiB relative B before publishing.
    u32 *t1_body = t1_patch + 1;
    for (u32 k = 0; k < bail_cnt; k++) {
        if (!bail_sites || bail_sites[k] >= code_words)
            goto out;
        i64 bdist = t1_body - (t2_start + bail_sites[k]);
        if (bdist < -(1 << 25) || bdist >= (1 << 25))
            goto out;
    }

    memcpy(t2_start, code, (size_t)code_words * 4);
    for (u32 k = 0; k < bail_cnt; k++) {
        u32 *site = t2_start + bail_sites[k];
        i64 bdist = t1_body - site;
        *site = 0x14000000u | ((u32)bdist & 0x3ffffffu);
    }
    u32 map0 = t2_pcmap_used[part];
    memcpy(&t2_pc_map[map0], pcm, (size_t)pcm_count * sizeof(PcMap));
    t2_pcmap_used[part] = map0 + pcm_count;

    u32 bi = t2_block_cnt[part];
    BlockEntry *be = &t2_block_table[part * T2_BLOCKS_MAX + bi];
    be->rip = head_rip;
    be->key = key;
    be->arm_off = (u32)(t2_start - cc_base);
    be->arm_words = code_words;
    be->map_off = map0;
    be->map_count = pcm_count;
    be->pad = 0;
    if (t2_trap_meta) {
        if (trap_meta)
            t2_trap_meta[part * T2_BLOCKS_MAX + bi] = *trap_meta;
        else
            memset(&t2_trap_meta[part * T2_BLOCKS_MAX + bi], 0, sizeof(T2TrapMeta));
    }

    T2BlockMeta *m = &t2_meta[part][bi];
    m->rip = head_rip;
    m->view = key;
    m->entry_word = be->arm_off;
    m->arm_words = code_words;
    m->t1_idx = t1_idx;
    m->cpl = cpl;
    m->p0_slot = (u8)p0_slot;
    m->compat = compat;
    m->dead = false;

    t2_block_cnt[part] = bi + 1U;
    cc_t2_cursor[part] = t2_start + code_words;
    __atomic_store_n(&ctx_ptr->t2_block_count[part], (u64)(bi + 1U), __ATOMIC_RELEASE);
    sys_icache_invalidate(t2_start, (size_t)code_words * 4);
    code_published();

    u32 expected = WORD_NOP;
    u32 branch = 0x14000000u | ((u32)dist & 0x3ffffffu);
    if (__atomic_compare_exchange_n(t1_patch, &expected, branch, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        sys_icache_invalidate(t1_patch, 4);
        hb->t2_entry_word = be->arm_off;
        hb->t2_meta_idx = bi + 1U;
        ok = true;
    } else {
        // Entry was killed concurrently; poison T2 entry so it stays dead.
        *t2_start = WORD_DEAD;
        sys_icache_invalidate(t2_start, 4);
        m->dead = true;
    }
out:
    pthread_mutex_unlock(&t2_pub_lock);
    return ok;
}

static TrResult translate(int view, uint64_t cr3, int cpl, bool compat, uint64_t rip, bool single) {
    // In single-vCPU / standalone C tests, also exercise the exact parallel prepare/reloc/commit/post pipeline
    // unless disabled (LC_HVDBT_NO_PAR_TRANSLATE=1) or single-stepping / profiling compile inputs.
    if (par_translate_enabled && !single && !cfg.profile_compile_inputs && ctx_ptr->vcpu_count <= 1) {
        TrResult r;
        bool flushed = false;
        for (int attempt = 0; attempt < 3; attempt++) {
            if (tr_translate_fast(view, cr3, cpl, compat, rip, false, &r)) {
                r.flushed = r.flushed || flushed;
                return r;
            }
            tr_translate_prepare();
            int st = tr_translate_commit(&r);
            flushed = flushed || r.flushed;
            if (st > 0) {
                r.flushed = flushed;
                tr_translate_post();
                return r;
            }
            if (st < 0) {
                flush_everything();
                flushed = true;
            }
        }
    }
    tr_compat = compat; // the asking vCPU's: ctx_ptr is the first vCPU's, whatever it is running
    __atomic_store_n(&tr_fpcr_merges_scalar, ctx_ptr->fpcr_merges_scalar != 0, __ATOMIC_RELAXED);
    if (view > view_high)
        view_high = view;
    int key = block_key(view, cr3, cpl, rip);
    if (!single) {
        u32 hit = map_find(key, rip);
        if (hit) {
            if ((key & KEY_DOMAIN) && cpl == 3)
                stats.shared_found++;
            TrResult r = {0};
            r.arm = tr_cc_va(cc_base + hblocks[hit - 1].entry_word);
            r.frames[0] = hblocks[hit - 1].frame[0];
            r.frames[1] = hblocks[hit - 1].frame[1];
            r.key = key;
            return r;
        }
    }
    TrResult r = do_translate(key, view, cr3, cpl, rip, single);
    if ((key & KEY_DOMAIN) && cpl == 3 && !r.unprotected && !r.needs_flush)
        stats.shared_blocks++;
    r.key = key;
    return r;
}

TrResult tr_translate(int view, uint64_t cr3, int cpl, bool compat, uint64_t rip, bool single) {
    return translate(view, cr3, cpl, compat, rip, single);
}

TrResult tr_translate_native(int view, uint64_t cr3, int cpl, bool compat, uint64_t rip, bool single) {
    return translate(view, cr3, cpl, compat, rip, single);
}

// The block holding a code cache word. Blocks only grow upward within each slot/part, so each slice is sorted.
static HostBlock *block_at(u32 word) {
    if (word >= CC_PART2_WORD) {
        u32 first = BLOCK_BASE(1), lo = first, hi = first + hblock_count[1];
        while (lo < hi) {
            u32 mid = (lo + hi) / 2;
            if (hblocks[mid].entry_word <= word)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo != first ? &hblocks[lo - 1] : 0;
    }
    if (p0_ring_enabled) {
        int slot = engine_p0_slot_of_word(word);
        if (slot < 0)
            return 0;
        u32 first = (u32)slot * P0_BLOCKS_PER_SLOT;
        u32 lo = first, hi = first + p0_block_count[slot];
        while (lo < hi) {
            u32 mid = (lo + hi) / 2;
            if (hblocks[mid].entry_word <= word)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo != first ? &hblocks[lo - 1] : 0;
    }
    u32 lo = 0, hi = hblock_count[0];
    while (lo < hi) {
        u32 mid = (lo + hi) / 2;
        if (hblocks[mid].entry_word <= word)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo != 0 ? &hblocks[lo - 1] : 0;
}

typedef struct {
    u64 rip;
    u32 view;
    u8 cpl;
    u8 p0_slot;
    u8 part;
    bool compat;
    bool valid;
} BlockInfo;

static BlockInfo block_info_at(u32 word) {
    BlockInfo bi = {0};
    if ((word >= CC_T2_WORD0 && word < CC_T2_WORD0 + CC_T2_WORDS) ||
        (word >= CC_T2_WORD1 && word < CC_T2_WORD1 + CC_T2_WORDS)) {
        u32 part = word >= CC_PART2_WORD ? 1U : 0U;
        u32 cnt = (u32)__atomic_load_n(&ctx_ptr->t2_block_count[part], __ATOMIC_ACQUIRE);
        T2BlockMeta *arr = t2_meta[part];
        if (arr && cnt > 0) {
            u32 lo = 0, hi = cnt;
            while (lo < hi) {
                u32 mid = (lo + hi) / 2;
                if (arr[mid].entry_word <= word)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo > 0) {
                T2BlockMeta *m = &arr[lo - 1];
                if (word < m->entry_word + m->arm_words) {
                    bi.rip = m->rip;
                    bi.view = m->view;
                    bi.cpl = m->cpl;
                    bi.p0_slot = m->p0_slot;
                    bi.part = (u8)part;
                    bi.compat = m->compat;
                    bi.valid = true;
                    return bi;
                }
            }
        }
    }
    HostBlock *hb = block_at(word);
    if (!hb)
        return bi;
    bi.rip = hb->rip;
    bi.view = hb->view;
    bi.cpl = hb->cpl;
    bi.p0_slot = hb->p0_slot;
    bi.part = word >= CC_PART2_WORD ? 1U : 0U;
    bi.compat = hb->compat;
    bi.valid = true;
    return bi;
}

// A cross-key or cross-generation transfer cannot bind permanently via direct B/BL.
// Give calls AND jumps an inline L1 / view lookup tail instead of trapping on every visit.
void tr_lookup_tail(uint64_t site, uint64_t target) {
    u32 *s = cc_host(site), was = *s;
    bool call = (was & 0xFC000000u) == 0x94000000u;
    bool jump = was == (0xD4000001u | SVC_DIRECT << 5) || was == (0xD4000001u | SVC_JCC << 5);
    if (!call && !jump)
        return;
    u32 s_word = (u32)(s - cc_base);
    BlockInfo from = block_info_at(s_word);
    cur_part = s_word >= CC_PART2_WORD; // in reach of its site
    u32 **cursor_p;
    u32 *limit_end;
    if (cur_part == 0 && p0_ring_enabled) {
        u32 slot = from.valid ? from.p0_slot : p0_cur_slot;
        cursor_p = &p0_cc_cursor[slot];
        limit_end = cc_base + engine_p0_slot_word_base(slot) + P0_SLOT_WORDS;
        // Lifetime-safe overflow fallback: if from's own slot tail headroom filled up,
        // spill into p0_cur_slot whenever p0_cur_slot outlives from and is within ±128 MiB of s.
        if (*cursor_p + 32 > limit_end && p0_cur_slot != slot &&
            p0_slot_gen[p0_cur_slot] >= p0_slot_gen[slot]) {
            u32 *alt_lim = cc_base + engine_p0_slot_word_base(p0_cur_slot) + P0_SLOT_WORDS;
            i64 alt_d = (i64)(p0_cc_cursor[p0_cur_slot] - s);
            if (p0_cc_cursor[p0_cur_slot] + 32 <= alt_lim &&
                alt_d >= -(1 << 25) && alt_d < (1 << 25)) {
                cursor_p = &p0_cc_cursor[p0_cur_slot];
                limit_end = alt_lim;
            }
        }
    } else {
        if (t2_enabled()) {
            u32 t2w = cur_part == 1 ? CC_T2_WORD1 : CC_T2_WORD0;
            if (cc_cursor[cur_part] + 64 >= cc_base + t2w &&
                cc_cursor[cur_part] < cc_base + t2w + CC_T2_WORDS) {
                cc_cursor[cur_part] = cc_base + t2w + CC_T2_WORDS;
            }
        }
        cursor_p = &cc_cursor[cur_part];
        u64 lim = cur_part == 1 && cc_limit >= CC_SIZE ? CC_SIZE - (4ULL << 20) : cc_limit;
        limit_end = cc_base + cur_part * CC_PART2_WORD + (u32)(lim / 4ULL);
    }
    i64 d_chk = (i64)(*cursor_p - s);
    if (d_chk < -(1 << 25) || d_chk >= (1 << 25))
        return;
    if (*cursor_p + 32 > limit_end)
        return;
    tr_cc = *cursor_p;
    u32 *tail = tr_cc;
    movimm(S0, target);
    if (from.valid && use_native_l1(from.cpl, from.compat) && target != 0) {
        // Constant-folded inline per-vCPU L1 probe: preserves NZCV and LR on hit.
        u32 slot = (u32)((target ^ (target >> 4)) & (FEX_L1_ENTRIES - 1));
        u32 byte_off = slot << 4;
        add_imm_lsl12(1, 0, 0, AD, CTX, 0x400u + (byte_off >> 12));
        u32 lo = byte_off & 0xfffu;
        if (lo <= 504) {
            E(0xA9400000u | ((lo / 8) & 0x7fu) << 15 | (u32)S3 << 10 | (u32)AD << 5 | (u32)S2);
        } else {
            add_imm(1, 0, 0, AD, AD, lo);
            E(0xA9400000u | (u32)S3 << 10 | (u32)AD << 5 | (u32)S2);
        }
        logic_reg(1, 2, 0, S3, S3, S0);
        u32 *miss = tr_cc;
        cbz(1, 1, S3, 0);
        br(S2);
        *miss |= ((u32)(tr_cc - miss) & 0x7ffffu) << 5;
        mov_reg(1, S4, LR);
        bl_rel((int)((i64)(STUB(PRIV_VA + OFF_STUBS + STUB_OFF_LOOKUP_L1) - tr_cc_va(tr_cc)) / 4));
        mov_reg(1, LR, S4);
        br(S2);
    } else {
        mov_reg(1, S4, LR);
        bl_rel((int)((i64)(STUB(tr_stub_lookup) - tr_cc_va(tr_cc)) / 4));
        mov_reg(1, LR, S4);
        br(S2);
    }
    *cursor_p = tr_cc;
    sys_icache_invalidate(tail, (u8 *)tr_cc - (u8 *)tail);
    code_published();
    i64 d = tail - s;
    if (__atomic_compare_exchange_n(s, &was, (call ? 0x94000000u : 0x14000000u) | ((u32)d & 0x3ffffff), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        sys_icache_invalidate(s, 4);
        if (call)
            stats.call_tails++;
        else
            stats.jump_tails++;
    }
}

void tr_link(uint64_t site, int kind, uint64_t arm, int key) {
    u32 *s = cc_host(site);
    // A link is forever, so it may only join blocks of one domain and safe ring generation order.
    BlockInfo from = block_info_at((u32)(s - cc_base));
    u32 was = *s;
    BlockInfo to = block_info_at((u32)(cc_host(arm) - cc_base));
    bool bad_ring_order = p0_ring_enabled && from.valid && to.valid && from.part == 0 && to.part == 0 &&
                          p0_slot_gen[from.p0_slot] > p0_slot_gen[to.p0_slot];
    if (from.valid && ((int)from.view != key || bad_ring_order) &&
        (kind == SVC_CALL || kind == SVC_DIRECT || kind == SVC_JCC)) {
        if (to.valid)
            tr_lookup_tail(site, to.rip);
        return;
    }
    if (!from.valid || (int)from.view != key || site == arm || (kind != SVC_CALL && (was & 0xFFE0001Fu) != 0xD4000001u))
        return;
    i64 d = ((i64)arm - (i64)site) >> 2;
    if (d < -(1 << 25) || d >= (1 << 25)) {
        if (to.valid && (kind == SVC_CALL || kind == SVC_DIRECT || kind == SVC_JCC))
            tr_lookup_tail(site, to.rip);
        return;
    }
    if (!__atomic_compare_exchange_n(s, &was, (kind == SVC_CALL ? 0x94000000u : 0x14000000u) | ((u32)d & 0x3ffffff), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    sys_icache_invalidate(s, 4);
    if (kind == SVC_JCC) { // the stub's fourth word locates the conditional branch: go straight to the block if it reaches
        u32 back = s[3];
        u32 *cond = s - back;
        i64 off = ((i64)arm - (i64)tr_cc_va(cond)) >> 2;
        if (off >= -(1 << 18) && off < (1 << 18)) {
            *cond = (*cond & ~(0x7ffffu << 5)) | ((u32)off & 0x7ffff) << 5;
            sys_icache_invalidate(cond, 4);
        }
    }
}

static u64 *hash_table(int view) { return (u64 *)(cfg.priv + OFF_HASH + (u64)view * HASH_BYTES); }

static inline u32 crc32c64(u64 v) {
    u32 c = 0;
    __asm__("crc32cx %w0, %w0, %x1" : "+r"(c) : "r"(v));
    return c;
}

// A lookup ends at an empty slot, in the stub and in the monitor, so a table must always have some. It is only a cache of the block map: when it fills, start again.
static u32 hash_slots_used[MAX_VIEWS];

// Keys only: another vCPU may be reading a slot, and a key it has seen stays good with the address beside it, if only for a block that may be dead by now, which is safe to enter.
static void hash_empty(int view) {
    u64 *t = hash_table(view);
    for (u32 i = 0; i < HASH_ENTRIES; i++)
        t[i * 2] = 0;
    __sync_synchronize();
}

// Called under the existing translator lock, before publishing any hash value/key. Unknown targets disable the
// optimization until a global hash flush: a missing membership bit must never silently leave an executable lookup.
static void membership_track(int view, u64 rip, u64 arm) {
    if (!stats.invalidation_membership_reliable)
        return;
    u64 base = PRIV_VA + OFF_CC;
    if (view < 0 || view >= MAX_VIEWS || arm < base || ((arm - base) & 3) != 0 || (arm - base) / 4 > UINT32_MAX)
        goto uncertain;
    u32 word = (u32)((arm - base) / 4);
    HostBlock *hb = block_at(word);
    if (!hb || hb->entry_word != word || hb->rip != rip || hb->dead || !hb->registered)
        goto uncertain;
    if (hb->view & KEY_DOMAIN) {
        u64 idx = (u64)(hb - hblocks);
        if (idx >= BLOCKS_MAX)
            goto uncertain;
        shared_hash_views[idx][view / 64] |= 1ULL << (view % 64);
        stats.invalidation_membership_inserts++;
    }
    return;
uncertain:
    stats.invalidation_membership_tracking_failures++;
    stats.invalidation_membership_reliable = 0;
}

static void membership_clear_all(void) {
    if (shared_hash_views) {
        memset(shared_hash_views, 0, BLOCKS_MAX * sizeof(*shared_hash_views));
        stats.invalidation_membership_reliable = 1; // Every hash is empty; future insertions can be tracked from scratch.
    }
}

void tr_hash_insert(int view, uint64_t rip, uint64_t arm) {
    if (cfg.invalidation_membership) {
        membership_track(view, rip, arm);
        __atomic_thread_fence(__ATOMIC_RELEASE); // Publish membership before an existing-key value as well as a new key.
    }
    u64 val = arm;
    {
        u64 base = PRIV_VA + OFF_CC;
        if (arm >= base && ((arm - base) & 3) == 0) {
            u32 *entry = cc_base + (u32)((arm - base) / 4);
            if (entry[1] == 0x10000050u)
                val |= 1ULL;
        }
    }
    u64 *t = hash_table(view);
    if (hash_slots_used[view] >= HASH_ENTRIES / 4 * 3) {
        hash_empty(view);
        hash_slots_used[view] = 0;
        stats.hash_refills++;
    }
    u32 h = crc32c64(rip) & (HASH_ENTRIES - 1);
    u32 tombstone = ~0u;
    for (u32 n = 0; n < HASH_ENTRIES; n++, h = (h + 1) & (HASH_ENTRIES - 1)) {
        u64 key = t[h * 2];
        if (key == rip) {
            t[h * 2 + 1] = val;
            return;
        }
        if (key == 1 && tombstone == ~0u)
            tombstone = h; // not yet: rip may be further on, and two entries for it would outlive a kill
        if (key == 0) {
            if (tombstone != ~0u)
                h = tombstone;
            else
                hash_slots_used[view]++;
            t[h * 2 + 1] = val;
            __sync_synchronize();
            t[h * 2] = rip;
            return;
        }
    }
}

static void hash_remove(int view, u64 rip) {
    u64 *t = hash_table(view);
    u32 h = crc32c64(rip) & (HASH_ENTRIES - 1);
    for (u32 n = 0; n < HASH_ENTRIES; n++, h = (h + 1) & (HASH_ENTRIES - 1)) {
        u64 key = t[h * 2];
        if (key == 0)
            return;
        if (key == rip) {
            t[h * 2] = 1; // tombstone: nonzero and never a valid target of a lookup that matters
            return;
        }
    }
}

// The disabled path continues to use hash_remove unchanged, with no per-probe profiling branch.
static void hash_remove_profiled(int view, u64 rip) {
    u64 *t = hash_table(view);
    u32 h = crc32c64(rip) & (HASH_ENTRIES - 1);
    u64 probes = 0, removed = 0;
    for (u32 n = 0; n < HASH_ENTRIES; n++, h = (h + 1) & (HASH_ENTRIES - 1)) {
        u64 key = t[h * 2];
        probes++;
        if (key == 0)
            break;
        if (key == rip) {
            t[h * 2] = 1;
            removed = 1;
            break;
        }
    }
    stats.invalidation_hash_view_visits++;
    stats.invalidation_hash_probes += probes;
    stats.invalidation_hash_removals += removed;
}

// Conservative bits can outlive a hash entry or view. Never remove a different block that now has the same RIP.
static void hash_remove_entry(int view, u64 rip, u64 arm, bool sample) {
    u64 *t = hash_table(view);
    u32 h = crc32c64(rip) & (HASH_ENTRIES - 1);
    u64 probes = 0, removed = 0;
    for (u32 n = 0; n < HASH_ENTRIES; n++, h = (h + 1) & (HASH_ENTRIES - 1)) {
        u64 key = t[h * 2];
        probes++;
        if (!key)
            break;
        if (key == rip) {
            if ((t[h * 2 + 1] & ~1ULL) == arm) {
                t[h * 2] = 1;
                removed = 1;
            }
            break;
        }
    }
    stats.invalidation_membership_view_visits++;
    stats.invalidation_membership_stale_visits += !removed;
    stats.invalidation_membership_removals += removed;
    if (sample) {
        stats.invalidation_hash_view_visits++;
        stats.invalidation_hash_probes += probes;
        stats.invalidation_hash_removals += removed;
    }
}

static void invalidate_shared_hashes(u32 idx, bool sample) {
    HostBlock *hb = &hblocks[idx];
    if (cfg.invalidation_membership && stats.invalidation_membership_reliable && idx < BLOCKS_MAX) {
        for (u32 group = 0; group < MAX_VIEWS / 64; group++) {
            u64 bits = shared_hash_views[idx][group];
            while (bits) {
                u32 bit = (u32)__builtin_ctzll(bits);
                bits &= bits - 1;
                hash_remove_entry((int)(group * 64 + bit), hb->rip, tr_cc_va(cc_base + hb->entry_word), sample);
            }
        }
    } else {
        if (cfg.invalidation_membership)
            stats.invalidation_membership_full_scans++;
        for (int view = 0; view <= view_high; view++) {
            if (sample)
                hash_remove_profiled(view, hb->rip);
            else
                hash_remove(view, hb->rip);
        }
    }
}

static void fex_l1_remove_rip(u64 rip);

static void kill_block(u32 idx) {
    HostBlock *hb = &hblocks[idx];
    if (hb->dead)
        return;
    bool sample = cfg.profile_invalidations && (++stats.invalidation_profiled_kills & 255) == 0;
    if (sample) {
        stats.invalidation_sampled_kills++;
        stats.invalidation_sampled_shared += (hb->view & KEY_DOMAIN) != 0;
    }
    hb->dead = true;
    if (hb->registered) {
        if (sample) {
            stats.invalidation_sampled_registered++;
            u64 began = mach_absolute_time();
            map_remove((int)hb->view, hb->rip);
            stats.invalidation_map_unlink_ns += invalidation_ns(mach_absolute_time() - began);
            began = mach_absolute_time();
            if (hb->view & KEY_DOMAIN)
                invalidate_shared_hashes(idx, true);
            else
                hash_remove_profiled((int)hb->view, hb->rip);
            stats.invalidation_hash_fanout_ns += invalidation_ns(mach_absolute_time() - began);
        } else {
            map_remove((int)hb->view, hb->rip);
            if (hb->view & KEY_DOMAIN)
                invalidate_shared_hashes(idx, false);
            else
                hash_remove((int)hb->view, hb->rip);
        }
    }
    if (shared_hash_views && idx < BLOCKS_MAX)
        memset(shared_hash_views[idx], 0, sizeof(*shared_hash_views));
    if (use_native_l1(hb->cpl, hb->compat))
        fex_l1_remove_rip(hb->rip);
    // Keep eager removal of every lookup to this entry and entry poisoning: SVC_DEAD starts with view_lookup,
    // and dead-block reuse can revive an entry. Conservative membership never makes lookup entries lazy.
    bool has_t2 = hb->t2_has_loop || hb->t2_entry_word != 0;
    if (has_t2)
        pthread_mutex_lock(&t2_pub_lock);
    u32 *entry = cc_base + hb->entry_word;
    *entry = WORD_DEAD;
    u32 *t1_patch = (hb->t2_has_loop && hb->t2_patch_word != ~0U && hb->t2_patch_word != hb->entry_word)
                        ? (cc_base + hb->t2_patch_word) : NULL;
    if (t1_patch)
        *t1_patch = WORD_DEAD;
    u32 *t2_entry = hb->t2_entry_word ? (cc_base + hb->t2_entry_word) : NULL;
    if (t2_entry) {
        *t2_entry = WORD_DEAD;
        u32 part = idx >= BLOCK_BASE(1) ? 1U : 0U;
        if (hb->t2_meta_idx && t2_meta[part] && hb->t2_meta_idx <= T2_BLOCKS_MAX)
            t2_meta[part][hb->t2_meta_idx - 1].dead = true;
        hb->t2_entry_word = 0;
        hb->t2_meta_idx = 0;
    }
    bool is_fex = entry[1] == 0x10000050u;
    if (is_fex)
        entry[3] = WORD_DEAD;
    if (sample) {
        u64 began = mach_absolute_time();
        sys_icache_invalidate(entry, 4);
        if (t1_patch) sys_icache_invalidate(t1_patch, 4);
        if (t2_entry) sys_icache_invalidate(t2_entry, 4);
        if (is_fex)
            sys_icache_invalidate(entry + 3, 4);
        stats.invalidation_icache_ns += invalidation_ns(mach_absolute_time() - began);
    } else {
        sys_icache_invalidate(entry, 4);
        if (t1_patch) sys_icache_invalidate(t1_patch, 4);
        if (t2_entry) sys_icache_invalidate(t2_entry, 4);
        if (is_fex)
            sys_icache_invalidate(entry + 3, 4);
    }
    if (has_t2) {
        pthread_mutex_unlock(&t2_pub_lock);
        t2_on_kill(idx);
    }
    stats.invalidated++;
    if (hb->read_bytes)
        dead_slots[dead_slot(hb->view, hb->rip)] = idx + 1;
}

int tr_invalidate_frame(uint64_t gpa, uint32_t why) {
    int n = 0;
    stats.kill_calls[why]++;
    __atomic_fetch_add(&tr_frame_inval_gen[(gpa >> 12) & (PAR_GEN_BUCKETS - 1U)], 1ULL, __ATOMIC_RELEASE);
    if (aot_cache)
        aot_invalidate_frame_hash(aot_cache, gpa);
    if (gpa >= cfg.ram_size)
        return 0;
    u32 fr = (u32)(gpa >> 12);
    u32 it = frame_head[fr];
    frame_head[fr] = 0;
    u64 frame = gpa & ~0xfffULL;
    while (it) {
        HostBlock *hb = &hblocks[it - 1];
        u32 next = hb->frame[0] == frame ? hb->next_on_frame[0] : hb->next_on_frame[1];
        hb->on_frame = false;
        if (!hb->dead) {
            kill_block(it - 1);
            n++;
        }
        it = next;
    }
    stats.killed[why] += (u64)n;
    return n;
}

static void fex_l1_remove_rip(u64 rip) {
    // Pairs with _stub_lookup_l1's stp -> dmb ish -> ldr [hash_slot] (Dekker):
    // ensures prior hash_remove store is visible before loading l1[idx + 1].
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    u32 vc = ctx_ptr->vcpu_count ? ctx_ptr->vcpu_count : 1;
    u64 slot = (rip ^ (rip >> 4)) & (FEX_L1_ENTRIES - 1);
    u64 idx = slot * 2;
    for (u32 vi = 0; vi < vc && vi < MAX_VCPUS; vi++) {
        u64 *l1 = (u64 *)(cfg.priv + OFF_FEX_L1 + (u64)vi * FEX_L1_BYTES);
        if (__atomic_load_n(&l1[idx + 1], __ATOMIC_RELAXED) == rip)
            __atomic_store_n(&l1[idx + 1], 0ULL, __ATOMIC_RELEASE);
    }
}

static void fex_l1_clear_view(int view) {
    u32 vc = ctx_ptr->vcpu_count ? ctx_ptr->vcpu_count : 1;
    if (vc > MAX_VCPUS)
        vc = MAX_VCPUS;
    // Pairs with monitor.c bumping fex_view_gen[view] (SEQ_CST) before HC_VIEW_FORGET
    // and view_activate_core storing c->view (SEQ_CST) before loading fex_view_gen.
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    for (u32 vi = 0; vi < vc; vi++) {
        const EngineCtx *vc_ctx = (const EngineCtx *)(cfg.priv + OFF_CTX + (u64)vi * CTX_SLOT);
        if (__atomic_load_n(&vc_ctx->view, __ATOMIC_ACQUIRE) != (u32)view)
            continue;
        u64 *l1 = (u64 *)(cfg.priv + OFF_FEX_L1 + (u64)vi * FEX_L1_BYTES);
        for (size_t i = 0; i < FEX_L1_ENTRIES; i++)
            __atomic_store_n(&l1[i * 2 + 1], 0ULL, __ATOMIC_RELEASE);
    }
}

void tr_view_forget(int view) {
    hash_empty(view);
    hash_slots_used[view] = 0;
    fex_l1_clear_view(view);
    stats.hash_forgets++;
}

int tr_invalidate_view_va(int view, uint64_t va, uint32_t why, bool unmoved, bool keeper) {
    int n = 0;
    __atomic_fetch_add(&tr_pt_inval_gen, 1ULL, __ATOMIC_RELEASE);
    bool unmapped_shared = false;
    u64 page = va & ~0xfffULL;
    u32 slot = page_slot(page);
    u32 *link = &page_head[slot];
    while (*link) {
        HostBlock *hb = &hblocks[*link - 1];
        u32 *next = &hb->next_on_page[(hb->rip >> 12 & (PAGE_HEADS - 1)) == slot ? 0 : 1];
        if (hb->dead) { // dead blocks leave here, which keeps the chains as short as the live blocks on these pages
            hb->on_page = false;
            *link = *next;
            continue;
        }
        u64 first = hb->rip & ~0xfffULL;
        bool here = first == page || (hb->frame[1] != ~0ULL && first + 4096 == page);
        if (here && (hb->view & KEY_DOMAIN) && ((hb->view & KEY_USER_FRAME) == KEY_USER_FRAME || !keeper)) { // only this view's sight of it changed: the block stands for the others
            hash_remove(view, hb->rip);
            fex_l1_remove_rip(hb->rip);
            unmapped_shared = true;
        } else if (here && ((int)hb->view == view || (hb->view & KEY_DOMAIN))) {
            kill_block(*link - 1);
            n++;
            hb->on_page = false;
            *link = *next;
            continue;
        }
        link = next;
    }
    if (unmapped_shared)
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    stats.kill_calls[why]++;
    stats.killed[why] += (u64)n;
    if (unmoved)
        stats.killed_unmoved[why] += (u64)n;
    return n;
}

void tr_view_drop(int view) {
    __atomic_fetch_add(&tr_pt_inval_gen, 1ULL, __ATOMIC_RELEASE);
    stats.kill_calls[KILL_VIEW_DROP]++;
    for (u32 it = view_head[view]; it; it = hblocks[it - 1].next_in_view) {
        hblocks[it - 1].on_view = false;
        if (!hblocks[it - 1].dead) {
            kill_block(it - 1);
            stats.killed[KILL_VIEW_DROP]++;
        }
    }
    view_head[view] = 0;
    memset(hash_table(view), 0, HASH_BYTES);
    hash_slots_used[view] = 0;
    fex_l1_clear_view(view);
    ctx_ptr->tr_flushed = 1;
}

void tr_flush_all(void) {
    if (ctx_ptr->vcpu_count > 1) { // other vCPUs are running in this cache: the host gets them out first, and calls tr_flush_parked
        fprintf(stderr, "engine stop: code cache reset with other vCPUs running\n");
        abort();
    }
    tr_flush_parked(false);
}

// After new code is written and its cache maintenance done, before any lookup or
// link can lead another vCPU there: see EngineCtx.code_gen. Safe from Tier-1 & Tier-2.
static void code_published(void) {
    static u64 generation;
    u64 g = __atomic_add_fetch(&generation, 1, __ATOMIC_RELAXED);
    u32 vc = ctx_ptr->vcpu_count ? ctx_ptr->vcpu_count : 1;
    if (vc > MAX_VCPUS)
        vc = MAX_VCPUS;
    for (u32 i = 0; i < vc; i++)
        __atomic_store_n(&((EngineCtx *)(cfg.priv + OFF_CTX + (u64)i * CTX_SLOT))->code_gen, g, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_SEQ_CST); // before a hash key or an existing key's new value
}

static void clear_code(u32 *from, u32 *to) {
    if (to <= from)
        return;
    memset(from, 0, (size_t)((u8 *)to - (u8 *)from));
    sys_icache_invalidate(from, (size_t)((u8 *)to - (u8 *)from));
}

void tr_flush_parked(bool everything) {
    __atomic_fetch_add(&tr_flush_gen, 1ULL, __ATOMIC_RELEASE);
    bool ring_evict = p0_ring_enabled && p0_ring_need_evict;
    if (ring_evict) {
        // Part 0 4-slot ring wrap: recycle ONLY the oldest 61 MiB slot (25%), keeping 3 newest slots (183 MiB) + live T2 hot loops!
        u32 victim = p0_next_ring_slot(p0_cur_slot);
        bool reset_p1 = part_full(1);
        pthread_mutex_lock(&t2_pub_lock);
        t2_epoch[0]++;
        if (reset_p1)
            t2_epoch[1]++;

        u32 v_word_base = engine_p0_slot_word_base(victim);
        fprintf(stderr, "[hvdbt-ring-evict] victim=%u gen=%llu->%llu evictions=%llu slot_cc=%ld/%llu slot_blk=%u/%u reset_p1=%d\n",
                victim, (unsigned long long)p0_slot_gen[victim], (unsigned long long)(p0_gen_seq + 1),
                (unsigned long long)(p0_ring_evictions + 1),
                (long)((u8 *)p0_cc_cursor[victim] - (u8 *)(cc_base + v_word_base)), (unsigned long long)P0_SLOT_BYTES,
                p0_block_count[victim], P0_BLOCKS_PER_SLOT, reset_p1 ? 1 : 0);

        ctx_ptr->tr_flushed = 1;
        memset(frame_head, 0, (cfg.ram_size >> 12) * sizeof(u32));
        memset(page_head, 0, PAGE_HEADS * sizeof(u32));
        memset(view_head, 0, sizeof view_head);
        memset(block_map, 0, MAP_SIZE * sizeof(MapEntry));
        memset(dead_slots, 0, DEAD_SLOTS * sizeof(u32));
        for (int v = 0; v <= view_high; v++)
            memset(hash_table(v), 0, HASH_BYTES);
        memset(cfg.priv + OFF_FEX_L1, 0, MAX_VCPUS * FEX_L1_BYTES);
        memset(hash_slots_used, 0, sizeof hash_slots_used);
        membership_clear_all();

        // Poison only CC_T2 part 0 entries belonging to the evicted victim slot; surviving slots' T2 loops stay hot.
        if (t2_meta[0]) {
            for (u32 k = 0; k < t2_block_cnt[0]; k++) {
                if (!t2_meta[0][k].dead && t2_meta[0][k].p0_slot == victim) {
                    u32 *e = cc_base + t2_meta[0][k].entry_word;
                    *e = WORD_DEAD;
                    sys_icache_invalidate(e, 4);
                    t2_meta[0][k].dead = true;
                }
            }
        }

        clear_code(cc_base + v_word_base, p0_cc_cursor[victim]);
        p0_cc_cursor[victim] = cc_base + v_word_base;
        p0_block_count[victim] = 0;
        p0_pcmap_used[victim] = victim * P0_PCMAP_PER_SLOT;
        __atomic_store_n(&ctx_ptr->p0_slot_block_count[victim], 0ULL, __ATOMIC_RELEASE);

        if (reset_p1) {
            clear_code(cc_base + CC_PART2_WORD, cc_cursor[1]);
            clear_code(cc_base + CC_T2_WORD1, cc_t2_cursor[1]);
            cc_cursor[1] = cc_base + CC_PART2_WORD;
            hblock_count[1] = 0;
            pc_map_used[1] = (OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap);
            read_copied[1] = READ_COPY_BYTES / 2;
            __atomic_store_n(&ctx_ptr->block_count_private, 0ULL, __ATOMIC_RELEASE);
            cc_t2_cursor[1] = cc_base + CC_T2_WORD1;
            t2_block_cnt[1] = 0;
            t2_pcmap_used[1] = T2_PCMAP_ENTRIES;
            __atomic_store_n(&ctx_ptr->t2_block_count[1], 0ULL, __ATOMIC_RELEASE);
            stats.cache_private_resets++;
        } else {
            for (u32 i = 0; i < hblock_count[1]; i++) {
                u32 idx = BLOCK_BASE(1) + i;
                HostBlock *hb = &hblocks[idx];
                if (hb->dead) {
                    hb->on_frame = hb->on_page = hb->on_view = false;
                    continue;
                }
                link_block(idx);
                if (hb->registered)
                    map_insert((int)hb->view, hb->rip, idx + 1);
            }
        }

        // Re-link all surviving live blocks from the other 3 p0 ring slots.
        for (u32 s = 0; s < P0_SLOTS; s++) {
            if (s == victim)
                continue;
            u32 base = s * P0_BLOCKS_PER_SLOT;
            for (u32 i = 0; i < p0_block_count[s]; i++) {
                u32 idx = base + i;
                HostBlock *hb = &hblocks[idx];
                if (hb->dead) {
                    hb->on_frame = hb->on_page = hb->on_view = false;
                    continue;
                }
                link_block(idx);
                if (hb->registered)
                    map_insert((int)hb->view, hb->rip, idx + 1);
            }
        }

        p0_cur_slot = victim;
        p0_slot_gen[victim] = ++p0_gen_seq;
        __atomic_store_n(&ctx_ptr->p0_slot_gen[victim], p0_slot_gen[victim], __ATOMIC_RELEASE);
        p0_ring_need_evict = false;
        p0_ring_evictions++;
        u64 ep0 = t2_epoch[0], ep1 = t2_epoch[1];
        pthread_mutex_unlock(&t2_pub_lock);
        t2_on_flush(false, (int)victim, ep0, ep1);
        if (reset_p1)
            t2_on_flush(false, -1, ep0, ep1);
        return;
    }

    everything = everything || (!p0_ring_enabled && part_full(0));
    u64 p0_cc_bytes = 0;
    u32 p0_blks = 0, p0_pcm = 0;
    if (p0_ring_enabled) {
        for (u32 s = 0; s < P0_SLOTS; s++) {
            p0_cc_bytes += (u64)((u8 *)p0_cc_cursor[s] - (u8 *)(cc_base + engine_p0_slot_word_base(s)));
            p0_blks += p0_block_count[s];
            p0_pcm += p0_pcmap_used[s] - s * P0_PCMAP_PER_SLOT;
        }
    } else {
        p0_cc_bytes = (u64)((u8 *)cc_cursor[0] - (u8 *)cc_base);
        p0_blks = hblock_count[0];
        p0_pcm = pc_map_used[0];
    }
    fprintf(stderr, "[hvdbt-flush] all=%d p0(cc=%llu/%llu blk=%u/%llu pcm=%u/%zu) p1(cc=%ld/%llu blk=%u/%llu pcm=%u/%zu)\n",
            everything ? 1 : 0,
            (unsigned long long)p0_cc_bytes, (unsigned long long)cc_limit,
            p0_blks, (unsigned long long)PART_BLOCKS_MAX(0), p0_pcm, (size_t)PCMAP_ENTRIES,
            (long)((u8 *)cc_cursor[1] - (u8 *)(cc_base + CC_PART2_WORD)), (unsigned long long)cc_limit,
            hblock_count[1], (unsigned long long)PART_BLOCKS_MAX(1),
            pc_map_used[1] - (u32)((OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap)), (size_t)PCMAP_ENTRIES);

    pthread_mutex_lock(&t2_pub_lock);
    t2_epoch[1]++;
    if (everything)
        t2_epoch[0]++;

    ctx_ptr->tr_flushed = 1;
    memset(frame_head, 0, (cfg.ram_size >> 12) * sizeof(u32));
    memset(page_head, 0, PAGE_HEADS * sizeof(u32));
    memset(view_head, 0, sizeof view_head);
    memset(block_map, 0, MAP_SIZE * sizeof(MapEntry));
    flush_dead_slots(everything);
    for (int v = 0; v <= view_high; v++)
        memset(hash_table(v), 0, HASH_BYTES);
    memset(cfg.priv + OFF_FEX_L1, 0, MAX_VCPUS * FEX_L1_BYTES);
    memset(hash_slots_used, 0, sizeof hash_slots_used);
    membership_clear_all(); // Both full and private-part flushes empty ALL view hashes before any block index is reused.
    // Return the reused span to zeros (UDF). A core may prefetch past the
    // cursor before the next translation lands there, and with old code left
    // in place that stale fetch would run silently. A zero traps, and the
    // monitor retries it once after synchronizing.
    clear_code(cc_base + CC_PART2_WORD, cc_cursor[1]);
    clear_code(cc_base + CC_T2_WORD1, cc_t2_cursor[1]);
    if (everything) {
        if (p0_ring_enabled) {
            for (u32 s = 0; s < P0_SLOTS; s++)
                clear_code(cc_base + engine_p0_slot_word_base(s), p0_cc_cursor[s]);
        } else {
            clear_code(cc_base, cc_cursor[0]);
        }
        clear_code(cc_base + CC_T2_WORD0, cc_t2_cursor[0]);
    }
    cc_cursor[1] = cc_base + CC_PART2_WORD;
    hblock_count[1] = 0;
    pc_map_used[1] = (OFF_PCMAP2 - OFF_PCMAP) / sizeof(PcMap);
    read_copied[1] = READ_COPY_BYTES / 2;
    __atomic_store_n(&ctx_ptr->block_count_private, 0ULL, __ATOMIC_RELEASE);
    cc_t2_cursor[1] = cc_base + CC_T2_WORD1;
    t2_block_cnt[1] = 0;
    t2_pcmap_used[1] = T2_PCMAP_ENTRIES;
    __atomic_store_n(&ctx_ptr->t2_block_count[1], 0ULL, __ATOMIC_RELEASE);
    if (everything) {
        if (aot_cache)
            aot_clear_frame_hashes(aot_cache);
        stats.cache_resets++;
        cc_cursor[0] = cc_base;
        hblock_count[0] = 0;
        pc_map_used[0] = 0;
        read_copied[0] = 1;
        __atomic_store_n(&ctx_ptr->block_count, 0ULL, __ATOMIC_RELEASE);
        p0_cur_slot = 0;
        p0_gen_seq = 1;
        p0_ring_need_evict = false;
        for (u32 s = 0; s < P0_SLOTS; s++) {
            p0_cc_cursor[s] = cc_base + engine_p0_slot_word_base(s);
            p0_block_count[s] = 0;
            p0_pcmap_used[s] = s * P0_PCMAP_PER_SLOT;
            p0_slot_gen[s] = s == 0 ? 1ULL : 0ULL;
            __atomic_store_n(&ctx_ptr->p0_slot_gen[s], p0_slot_gen[s], __ATOMIC_RELEASE);
            __atomic_store_n(&ctx_ptr->p0_slot_block_count[s], 0ULL, __ATOMIC_RELEASE);
        }
        cc_t2_cursor[0] = cc_base + CC_T2_WORD0;
        t2_block_cnt[0] = 0;
        t2_pcmap_used[0] = 0;
        __atomic_store_n(&ctx_ptr->t2_block_count[0], 0ULL, __ATOMIC_RELEASE);
        u64 ep0 = t2_epoch[0], ep1 = t2_epoch[1];
        pthread_mutex_unlock(&t2_pub_lock);
        t2_on_flush(true, -1, ep0, ep1);
        return;
    }
    stats.cache_private_resets++;
    // The shared part stays, and finds its way back onto what was just emptied. Its dead blocks stay off, as a kill took them.
    if (p0_ring_enabled) {
        for (u32 s = 0; s < P0_SLOTS; s++) {
            u32 base = s * P0_BLOCKS_PER_SLOT;
            for (u32 i = 0; i < p0_block_count[s]; i++) {
                u32 idx = base + i;
                HostBlock *hb = &hblocks[idx];
                if (hb->dead) {
                    hb->on_frame = hb->on_page = hb->on_view = false;
                    continue;
                }
                link_block(idx);
                if (hb->registered)
                    map_insert((int)hb->view, hb->rip, idx + 1);
            }
        }
    } else {
        for (u32 idx = 0; idx < hblock_count[0]; idx++) {
            HostBlock *hb = &hblocks[idx];
            if (hb->dead) {
                hb->on_frame = hb->on_page = hb->on_view = false;
                continue;
            }
            link_block(idx);
            if (hb->registered)
                map_insert((int)hb->view, hb->rip, idx + 1);
        }
    }
    u64 ep0 = t2_epoch[0], ep1 = t2_epoch[1];
    pthread_mutex_unlock(&t2_pub_lock);
    t2_on_flush(false, -1, ep0, ep1);
}

TrStats tr_stats(void) {
    TrStats out = stats;
    if (aot_cache) {
        AotStats as = aot_get_stats(aot_cache);
        out.aot_enabled = as.enabled;
        out.aot_loaded = as.loaded_from_disk;
        out.aot_loaded_t2 = as.loaded_t2_from_disk;
        out.aot_lookups = as.lookups;
        out.aot_hits_t1 = as.hits_t1;
        out.aot_hits_t2 = as.hits_t2;
        out.aot_misses = as.misses;
        out.aot_reloc_failures = as.reloc_failures;
        out.aot_eager_cc_links = as.eager_cc_links;
        out.aot_recorded_t1 = as.recorded_t1;
        out.aot_recorded_t2 = as.recorded_t2;
        out.aot_saves_ok = as.saves_ok;
        out.aot_saved_bytes = as.saved_bytes_last;
        out.aot_reloc_ns = as.reloc_ns_total;
    }
    return out;
}

