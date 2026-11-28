// Persistent Content-Addressed, ASLR-Relocatable AOT Code Cache for HVDBT CEngine.
//
// Design invariants:
//   1. Content-addressed key: (xxh64(x86_code_page_4096), page_offset, mode_bits, translator_abi_hash).
//      Any 1-byte SMC mutation anywhere on the 4 KiB page changes page_hash -> guaranteed AOT miss -> live JIT.
//   2. ASLR-relocatable blocks: every GVA constant, stub branch, cross-block branch, and Tier-2 alias-guard
//      bailout is captured as a compact 8-byte AotReloc and patched in place in nanoseconds without shifting PcMaps.
//   3. Zero-copy mmap read on startup + crash-safe atomic write (temp file + fsync + rename) on flush/exit.
//   4. Tier-2 hot-loop blocks (including Round 2 T2TrapMeta for memop-safe YMM8..15 & stack-slot residency)
//      are stored with AOT_PRIO_TIER2 so they survive disk caps first and promote immediately on warm boot (0 ms).
//   5. Default OFF unless LC_HVDBT_AOT_CACHE=<path> is configured (100% backward compatible, zero overhead).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "engine_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AOT_MAGIC_STR "HVDBTAOT"
#define AOT_FORMAT_VERSION 2U
#define AOT_EMITTER_EPOCH 4U
#define AOT_MAX_RELOCS_PER_BLOCK 512U
#define AOT_DEFAULT_MAX_BYTES (512ULL << 20) // 512 MiB default disk budget (covers full OS + heavy workloads)
#define AOT_NO_PATCH_WORD 0xFFFFU

// Content-addressed lookup key (24 bytes, deterministic layout across sessions).
typedef struct {
    uint64_t page_hash;    // XXH64 of the entire 4096-byte guest x86 code page
    uint64_t abi_hash;     // translator_abi_hash from aot_compute_abi_hash()
    uint32_t mode_bits;    // AOT_MODE_* bitmask (cpl, compat, is_t2, smp, flags, opts)
    uint16_t page_offset;  // entry_rip & 0xFFF (0..4095)
    uint16_t flags;        // reserved (must be 0)
} AotKey;

// Mode bit layout for AotKey.mode_bits.
#define AOT_MODE_CPL_MASK        0x00000003U // bits 0..1: guest CPL (0..3)
#define AOT_MODE_COMPAT          (1U << 2)   // 32-bit compatibility mode
#define AOT_MODE_IS_T2           (1U << 3)   // 1 = Tier-2 hot-loop block, 0 = Tier-1 block
#define AOT_MODE_SMP             (1U << 4)   // vcpu_count > 1 (leading WORD_NOP entry)
#define AOT_MODE_PRECISE_FLAGS   (1U << 5)   // tr_precise_flags
#define AOT_MODE_TZCNT_LZCNT     (1U << 6)   // tr_has_tzcnt
#define AOT_MODE_HYBRID          (1U << 7)   // tr_hybrid
#define AOT_MODE_FPCR_MERGE      (1U << 8)   // tr_fpcr_merges_scalar
#define AOT_MODE_NATIVE_DAG      (1U << 9)   // native_dag
#define AOT_MODE_NATIVE_L1       (1U << 10)  // native_l1
#define AOT_MODE_OPT_PEEP        (1U << 11)  // tr_opt_peep
#define AOT_MODE_OPT_YMM_CACHE   (1U << 12)  // tr_opt_ymm_cache
#define AOT_MODE_T2_ENABLED      (1U << 13)  // t2_enabled() (Tier-1 breaks at backedges & emits OSR anchor)
#define AOT_MODE_KEEP_KPTI       (1U << 14)  // cfg.keep_kernel_tables
#define AOT_MODE_NO_BULK_REP     (1U << 15)  // cfg.no_bulk_rep
#define AOT_MODE_NO_RET_PRED     (1U << 16)  // cfg.no_return_prediction
#define AOT_MODE_ALL_FLAGS_LIVE  (1U << 17)  // cfg.all_flags_live
#define AOT_MODE_PROF_SHIFT      (1U << 18)  // cfg.profile_shift_helpers
#define AOT_MODE_INLINE_SHR8     (1U << 19)  // cfg.inline_shr8
#define AOT_MODE_PAGE_REP        (1U << 20)  // cfg.page_rep
#define AOT_MODE_T2_NO_MEM_YMM   (1U << 21)  // LC_HVDBT_T2_NO_MEM_YMM=1
#define AOT_MODE_T2_NO_STK_PROMO (1U << 22)  // LC_HVDBT_T2_NO_STK_PROMO=1
#define AOT_MODE_T2_NO_UNROLL4   (1U << 23)  // LC_HVDBT_T2_NO_UNROLL4=1
#define AOT_MODE_ISA_DP          (1U << 24)  // tr_has_isa_dp (FEAT_DotProd: SDOT/UDOT)
#define AOT_MODE_ISA_I8MM        (1U << 25)  // tr_has_isa_i8mm (FEAT_I8MM: USDOT/SUDOT)
#define AOT_MODE_HW_POPCNT       (1U << 26)  // tr_opt_hw_popcnt (Ring-3 NEON CNT/ADDV POPCNT)
#define AOT_MODE_INLINE_PIC      (1U << 27)  // tr_opt_inline_pic (monomorphic inline PIC for JMPI/CALLI)
#define AOT_MODE_NO_DEAD_YMM_Z   (1U << 28)  // LC_HVDBT_NO_DEAD_YMM_ZERO=1
#define AOT_MODE_NO_SDOT_GEMM    (1U << 29)  // LC_HVDBT_NO_SDOT_GEMM=1
#define AOT_MODE_T2_MAX96        (1U << 30)  // LC_HVDBT_T2_MAX96=1
#define AOT_MODE_T2_NO_WIDE_POOL (1U << 31)  // LC_HVDBT_T2_NO_WIDE_POOL=1

// Record persistence priority (higher persisted first; Tier-2 never evicted by Tier-1).
typedef enum {
    AOT_PRIO_TIER1 = 1,
    AOT_PRIO_TIER2 = 2,
} AotPriority;

// Relocation kinds recorded in an AOT block template.
typedef enum {
    AOT_RELOC_NONE = 0,
    // 64-bit GVA literal in two adjacent u32 words [lo, hi] (SVC_DIRECT/JCC/CALL/X87).
    AOT_RELOC_GVA64_LIT = 1,
    // Fixed 2-word movz+movk (32-bit GVA: compat CALL/CALLI or 32-bit RIP-rel LEA).
    AOT_RELOC_MOVIMM32 = 2,
    // Fixed 3-word movz/movn + movk + movk (canonical 48-bit x86-64 user or kernel GVA).
    AOT_RELOC_MOVIMM48 = 3,
    // Fixed 4-word movz + 3x movk (arbitrary 64-bit GVA).
    AOT_RELOC_MOVIMM64 = 4,
    // 26-bit PC-relative B or BL to a fixed EL0 stub (resolved for dest partition & lower/upper stub copy).
    AOT_RELOC_STUB_BRANCH = 5,
    // Optional eager cross-block link hint at load time.
    AOT_RELOC_CC_BRANCH = 6,
    // Tier-2 Round 2 alias-guard bailout site: patched by tr_t2_publish() to B (t1_patch + 1).
    AOT_RELOC_T2_BAILOUT = 7,
    // Monomorphic inline PIC slot binding: word_off = ADD+LDR pair, aux2 = Lpic_init meta[2] word_off.
    AOT_RELOC_PIC_SLOT = 8,
} AotRelocType;

// Cross-block branch kinds for AOT_RELOC_CC_BRANCH.
typedef enum {
    AOT_CC_DIRECT = 0, // site holds SVC_DIRECT (patched to B target_arm if target is live)
    AOT_CC_JCC    = 1, // site holds B.cond/CBZ (patched directly to target_arm if within +-1 MiB)
    AOT_CC_CALL   = 2, // site holds BL stub (patched directly to BL target_arm if within +-128 MiB)
} AotCcBranchKind;

// Fixed EL0 stub identifiers for AOT_RELOC_STUB_BRANCH.
typedef enum {
    AOT_STUB_LOOKUP = 0,
    AOT_STUB_RET_MISS = 1,
    AOT_STUB_PUSHF = 2,
    AOT_STUB_SAHF = 3,
    AOT_STUB_LOOKUP_L1 = 4,
    AOT_STUB_LOOKUP_L1_FAST = 5,
    AOT_STUB_RET_MISS_L1 = 6,
    AOT_STUB_REP_MOVS = 7,
    AOT_STUB_REP_STOS = 8,
    AOT_STUB_REP_PAGE_MOVS = 9,
    AOT_STUB_REP_PAGE_STOS = 10,
    AOT_STUB_COUNT = 11,
} AotStubId;

// Compact 8-byte relocation entry.
typedef struct {
    uint16_t word_off; // ARM64 word offset from block start
    uint8_t  type;     // AotRelocType
    uint8_t  aux;      // rd, stub_id|(is_bl<<7), or AotCcBranchKind
    uint16_t aux2;     // AOT_RELOC_CC_BRANCH: word_off of target GVA64_LIT; else 0
    uint16_t pad;      // reserved (0)
} AotReloc;

_Static_assert(sizeof(AotReloc) == 8, "AotReloc must be exactly 8 bytes");
_Static_assert(sizeof(AotKey) == 24, "AotKey must be exactly 24 bytes");

// Runtime stub base addresses (in part-0 lower OFF_STUBS coordinates; relocator adds HI / PART2 offsets).
typedef struct {
    uint64_t va[AOT_STUB_COUNT];
} AotStubTable;

// Metadata + payloads describing one AOT block (used for both record and lookup views).
typedef struct {
    AotKey   key;
    uint64_t orig_rip;
    const uint32_t *code;
    uint32_t code_words;
    const PcMap *pcm;
    uint16_t pcm_count;
    const AotReloc *relocs;
    uint16_t reloc_count;
    uint16_t read_off;
    uint16_t read_bytes;
    uint16_t t2_patch_word_off; // offset from block start to WORD_NOP OSR anchor, or AOT_NO_PATCH_WORD
    uint16_t t2_head_off;       // head_rip - entry_rip (when t2_patch_word_off != AOT_NO_PATCH_WORD)
    uint16_t t2_min_off;        // loop snapshot min_off inside page
    uint16_t t2_span_bytes;     // loop snapshot span_bytes
    uint32_t priority;          // AOT_PRIO_TIER1 or AOT_PRIO_TIER2
    bool     has_t2_meta;       // true when t2_meta carries Tier-2 Round 2 T2TrapMeta
    T2TrapMeta t2_meta;         // Tier-2 trap reconstruction metadata (64B)
} AotBlockView;

// Telemetry counters (safe to read at any time via aot_get_stats()).
typedef struct {
    uint64_t enabled;
    uint64_t loaded_from_disk;      // total blocks in mmap'd file on open
    uint64_t loaded_t2_from_disk;   // Tier-2 blocks in mmap'd file on open
    uint64_t lookups;
    uint64_t hits_t1;
    uint64_t hits_t2;
    uint64_t misses;
    uint64_t reloc_failures;
    uint64_t eager_cc_links;        // AOT_RELOC_CC_BRANCH hits patched directly at load time
    uint64_t recorded_t1;
    uint64_t recorded_t2;
    uint64_t recorded_dups;
    uint64_t saves_ok;
    uint64_t saves_failed;
    uint64_t saved_blocks_last;
    uint64_t saved_t2_last;
    uint64_t saved_bytes_last;
    uint64_t corrupt_files_ignored;
    uint64_t abi_mismatches_ignored;
    uint64_t page_hash_cache_hits;
    uint64_t page_hash_cache_misses;
    uint64_t reloc_ns_total;        // cumulative nanoseconds spent in aot_materialize()
} AotStats;

// Optional callback for eager AOT_RELOC_CC_BRANCH linking during aot_materialize().
typedef uint64_t (*AotResolveTargetFn)(void *ctx, uint64_t target_rip);

// Thread-local relocation capture buffer used during live Tier-1 / Tier-2 emission.
typedef struct {
    AotReloc relocs[AOT_MAX_RELOCS_PER_BLOCK];
    uint32_t count;
    bool     active;
    bool     ineligible;
} AotCapture;

typedef struct AotCache AotCache;

// ---- Fast XXHash64 & Translator ABI Fingerprint ----
uint64_t aot_xxh64(const void *data, size_t len, uint64_t seed);
uint64_t aot_hash_page(const uint8_t page[4096]);
uint64_t aot_compute_abi_hash(uint64_t build_stamp);

// ---- Lifecycle ----
AotCache *aot_open(const char *path, uint64_t abi_hash, uint64_t max_bytes);
bool      aot_save(AotCache *cache);
void      aot_maybe_save_async(AotCache *cache, uint32_t min_dirty);
void      aot_close(AotCache *cache, bool save_dirty);

// ---- Per-Frame Page Hash Cache ----
uint64_t aot_hash_frame_cached(AotCache *cache, uint64_t frame_gpa, const uint8_t page[4096]);
void     aot_invalidate_frame_hash(AotCache *cache, uint64_t frame_gpa);
void     aot_clear_frame_hashes(AotCache *cache);

// ---- Lookup, Relocation & Recording ----
bool aot_lookup(AotCache *cache, const AotKey *key, AotBlockView *out_view);

bool aot_materialize(AotCache *cache,
                     const AotBlockView *view,
                     uint64_t new_rip,
                     uint32_t *dst_code,
                     uint32_t dst_code_cap_words,
                     PcMap *dst_pcm,
                     uint32_t dst_pcm_cap,
                     uint64_t dst_cc_va,
                     int dst_part,
                     const AotStubTable *stubs,
                     AotResolveTargetFn resolve_target,
                     void *resolve_ctx);

bool     aot_record(AotCache *cache, const AotBlockView *view);
void     aot_note_eager_links(AotCache *cache, uint64_t count);
AotStats aot_get_stats(const AotCache *cache);

// ---- Emitter Capture & Fixed-Width Relocatable Helpers ----
static inline void aot_capture_begin(AotCapture *cap) {
    cap->count = 0;
    cap->active = true;
    cap->ineligible = false;
}

static inline void aot_capture_mark_ineligible(AotCapture *cap) {
    if (cap && cap->active)
        cap->ineligible = true;
}

static inline void aot_capture_add(AotCapture *cap, uint32_t word_off, AotRelocType type,
                                   uint8_t aux, uint16_t aux2) {
    if (!cap || !cap->active)
        return;
    if (word_off > 0xFFFFU || cap->count >= AOT_MAX_RELOCS_PER_BLOCK) {
        cap->ineligible = true;
        return;
    }
    AotReloc *r = &cap->relocs[cap->count++];
    r->word_off = (uint16_t)word_off;
    r->type = (uint8_t)type;
    r->aux = aux;
    r->aux2 = aux2;
    r->pad = 0;
}

uint32_t aot_emit_movimm_gva(uint32_t **cc_p, uint32_t *block_start, AotCapture *cap,
                             int rd, uint64_t gva, bool force32);

uint64_t aot_resolve_stub_va(const AotStubTable *stubs, AotStubId id, int part, uint64_t pc_va);

#ifdef __cplusplus
}
#endif
