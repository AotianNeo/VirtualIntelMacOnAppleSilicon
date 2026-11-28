// Persistent Content-Addressed, ASLR-Relocatable AOT Code Cache for HVDBT CEngine.
#include "aot_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <pthread/qos.h>
#endif

// ---- Constants & On-Disk Layout ----

#define AOT_XXH_PAGE_SEED  0x4856444254504147ULL // "HVDBTPAG"
#define AOT_XXH_KEY_SEED   0x48564442544B4559ULL // "HVDBTKEY"
#define AOT_XXH_FILE_SEED  0x485644425446494CULL // "HVDBTFIL"
#define AOT_XXH_REC_SEED   0x4856444254524543ULL // "HVDBTREC"
#define AOT_XXH_ABI_SEED   0x4856444254414249ULL // "HVDBTABI"

#define AOT_FRAME_CACHE_SLOTS 16384U
#define AOT_STAGE_ARENA_CHUNK (4U << 20) // 4 MiB append-only staging chunks (stable pointers)
#define AOT_STAGE_INIT_BUCKETS 4096U

typedef struct {
    char     magic[8];        // "HVDBTAOT"
    uint32_t format_version;  // AOT_FORMAT_VERSION
    uint32_t header_bytes;    // sizeof(AotFileHeader) == 72
    uint64_t abi_hash;        // translator_abi_hash
    uint32_t index_buckets;   // power of 2
    uint32_t entry_count;     // total blocks persisted
    uint32_t t2_entry_count;  // Tier-2 blocks (placed first in payload)
    uint32_t flags;           // reserved (0)
    uint64_t index_offset;    // byte offset to AotIndexSlot[index_buckets]
    uint64_t payload_offset;  // byte offset to first AotDiskRecordHeader
    uint64_t file_bytes;      // exact file length
    uint64_t payload_xxh64;   // XXH64 of bytes [header_bytes .. file_bytes)
} AotFileHeader;

typedef struct {
    uint64_t key_hash; // nonzero XXH64(&key, sizeof(AotKey))
    uint64_t rec_off;  // byte offset from file start (0 = empty)
} AotIndexSlot;

typedef struct {
    AotKey   key;               // 24B
    uint64_t orig_rip;          // 8B
    uint32_t code_words;        // 4B
    uint16_t pcm_count;         // 2B
    uint16_t reloc_count;       // 2B
    uint16_t read_off;          // 2B
    uint16_t read_bytes;        // 2B
    uint16_t t2_patch_word_off; // 2B
    uint16_t t2_head_off;       // 2B
    uint16_t t2_min_off;        // 2B
    uint16_t t2_span_bytes;     // 2B
    uint32_t priority;          // 4B
    uint32_t rec_xxh32;         // 4B
    uint8_t  has_t2_meta;       // 1B (when nonzero, 64B T2TrapMeta follows relocs[])
    uint8_t  pad[3];            // 3B
} AotDiskRecordHeader;

_Static_assert(sizeof(AotFileHeader) == 72, "AotFileHeader must be 72 bytes");
_Static_assert(sizeof(AotIndexSlot) == 16, "AotIndexSlot must be 16 bytes");
_Static_assert(sizeof(AotDiskRecordHeader) == 64, "AotDiskRecordHeader must be 64 bytes");

// ---- Fast Portable XXHash64 (Zero External Dependencies) ----

#define XXH_P1 0x9E3779B185EBCA87ULL
#define XXH_P2 0xC2B2AE3D27D4EB4FULL
#define XXH_P3 0x165667B19E3779F9ULL
#define XXH_P4 0x85EBCA77C2B2AE63ULL
#define XXH_P5 0x27D4EB2F165667C5ULL

static inline uint64_t xxh_rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh_read64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline uint32_t xxh_read32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_P2;
    acc = xxh_rotl64(acc, 31);
    acc *= XXH_P1;
    return acc;
}

static inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    acc = acc * XXH_P1 + XXH_P4;
    return acc;
}

uint64_t aot_xxh64(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *end = p + len;
    uint64_t h64;

    if (len >= 32) {
        const uint8_t *limit = end - 32;
        uint64_t v1 = seed + XXH_P1 + XXH_P2;
        uint64_t v2 = seed + XXH_P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - XXH_P1;
        do {
            v1 = xxh_round(v1, xxh_read64(p));      p += 8;
            v2 = xxh_round(v2, xxh_read64(p));      p += 8;
            v3 = xxh_round(v3, xxh_read64(p));      p += 8;
            v4 = xxh_round(v4, xxh_read64(p));      p += 8;
        } while (p <= limit);

        h64 = xxh_rotl64(v1, 1) + xxh_rotl64(v2, 7) + xxh_rotl64(v3, 12) + xxh_rotl64(v4, 18);
        h64 = xxh_merge_round(h64, v1);
        h64 = xxh_merge_round(h64, v2);
        h64 = xxh_merge_round(h64, v3);
        h64 = xxh_merge_round(h64, v4);
    } else {
        h64 = seed + XXH_P5;
    }

    h64 += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k1 = xxh_round(0, xxh_read64(p));
        h64 ^= k1;
        h64 = xxh_rotl64(h64, 27) * XXH_P1 + XXH_P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)xxh_read32(p) * XXH_P1;
        h64 = xxh_rotl64(h64, 23) * XXH_P2 + XXH_P3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p++) * XXH_P5;
        h64 = xxh_rotl64(h64, 11) * XXH_P1;
    }

    h64 ^= h64 >> 33;
    h64 *= XXH_P2;
    h64 ^= h64 >> 29;
    h64 *= XXH_P3;
    h64 ^= h64 >> 32;
    return h64;
}

uint64_t aot_hash_page(const uint8_t page[4096]) {
    return aot_xxh64(page, 4096, AOT_XXH_PAGE_SEED);
}

static inline uint64_t aot_hash_key(const AotKey *k) {
    uint64_t h = aot_xxh64(k, sizeof(*k), AOT_XXH_KEY_SEED);
    return h ? h : 1ULL;
}

static inline bool aot_key_eq(const AotKey *a, const AotKey *b) {
    return a->page_hash == b->page_hash &&
           a->abi_hash == b->abi_hash &&
           a->mode_bits == b->mode_bits &&
           a->page_offset == b->page_offset &&
           a->flags == b->flags;
}

static inline size_t aot_align8(size_t n) {
    return (n + 7U) & ~(size_t)7U;
}

static size_t aot_record_disk_size(uint32_t code_words, uint16_t pcm_count,
                                   uint16_t reloc_count, uint8_t has_t2_meta) {
    size_t sz = sizeof(AotDiskRecordHeader)
              + (size_t)code_words * sizeof(uint32_t)
              + (size_t)pcm_count * sizeof(PcMap)
              + (size_t)reloc_count * sizeof(AotReloc);
    sz = aot_align8(sz);
    if (has_t2_meta)
        sz += sizeof(T2TrapMeta);
    return aot_align8(sz);
}

static uint32_t aot_compute_record_xxh32(const AotDiskRecordHeader *hdr,
                                         const uint32_t *code,
                                         const PcMap *pcm,
                                         const AotReloc *relocs,
                                         const T2TrapMeta *t2_meta) {
    AotDiskRecordHeader tmp = *hdr;
    tmp.rec_xxh32 = 0;
    tmp.pad[0] = tmp.pad[1] = tmp.pad[2] = 0;
    uint64_t h = aot_xxh64(&tmp, sizeof(tmp), AOT_XXH_REC_SEED);
    if (hdr->code_words)
        h = aot_xxh64(code, (size_t)hdr->code_words * sizeof(uint32_t), h);
    if (hdr->pcm_count)
        h = aot_xxh64(pcm, (size_t)hdr->pcm_count * sizeof(PcMap), h);
    if (hdr->reloc_count)
        h = aot_xxh64(relocs, (size_t)hdr->reloc_count * sizeof(AotReloc), h);
    if (hdr->has_t2_meta && t2_meta)
        h = aot_xxh64(t2_meta, sizeof(*t2_meta), h);
    return (uint32_t)(h ^ (h >> 32));
}

uint64_t aot_compute_abi_hash(uint64_t build_stamp) {
    const uint64_t sig[] = {
        AOT_FORMAT_VERSION,
        AOT_EMITTER_EPOCH,
        build_stamp,
        (uint64_t)sizeof(EngineCtx),
        (uint64_t)sizeof(PcMap),
        (uint64_t)sizeof(AotReloc),
        (uint64_t)sizeof(T2TrapMeta),
        (uint64_t)OFF_META_T2,
        (uint64_t)T2_MAX_STK_SLOTS,
        (uint64_t)PRIV_VA,
        (uint64_t)OFF_STUBS,
        (uint64_t)OFF_CC,
        (uint64_t)CC_SIZE,
        (uint64_t)OFF_STUBS_HI,
        (uint64_t)OFF_STUBS2,
        (uint64_t)OFF_CC2,
        (uint64_t)OFF_STUBS2_HI,
        (uint64_t)(OFF_FEX_L1 - OFF_CTX),
        (uint64_t)FEX_L1_ENTRIES,
        (uint64_t)STUB_OFF_REP_MOVS,
        (uint64_t)STUB_OFF_REP_STOS,
        (uint64_t)STUB_OFF_REP_PAGE_MOVS,
        (uint64_t)STUB_OFF_REP_PAGE_STOS,
        (uint64_t)offsetof(EngineCtx, df),
        (uint64_t)offsetof(EngineCtx, kernel_gs_base),
        (uint64_t)offsetof(EngineCtx, rs_top),
        (uint64_t)offsetof(EngineCtx, rflags_sys),
        (uint64_t)offsetof(EngineCtx, cr0),
        (uint64_t)offsetof(EngineCtx, cr2),
        (uint64_t)offsetof(EngineCtx, cr3),
        (uint64_t)offsetof(EngineCtx, cr4),
        (uint64_t)offsetof(EngineCtx, gdtr_base),
        (uint64_t)offsetof(EngineCtx, idtr_base),
        (uint64_t)offsetof(EngineCtx, seg_sel),
        (uint64_t)offsetof(EngineCtx, irq_shadow_rip),
        (uint64_t)offsetof(EngineCtx, irq_pending),
        (uint64_t)offsetof(EngineCtx, irq_line),
        (uint64_t)offsetof(EngineCtx, mm),
        (uint64_t)offsetof(EngineCtx, fp_top),
        (uint64_t)offsetof(EngineCtx, apic_tpr),
        (uint64_t)offsetof(EngineCtx, apic_irr_prio),
        (uint64_t)offsetof(EngineCtx, tsc_offset),
        (uint64_t)offsetof(EngineCtx, tsc_ns_offset),
        (uint64_t)offsetof(EngineCtx, tsc_num),
        (uint64_t)offsetof(EngineCtx, tsc_den),
        (uint64_t)offsetof(EngineCtx, tsc_counter_offset),
        (uint64_t)offsetof(EngineCtx, code_gen),
        (uint64_t)offsetof(EngineCtx, code_synced),
        (uint64_t)offsetof(EngineCtx, fp_d_valid),
        (uint64_t)offsetof(EngineCtx, fp_d),
        (uint64_t)offsetof(EngineCtx, ymm_hi),
        (uint64_t)offsetof(EngineCtx, cpuid0_cached),
        (uint64_t)offsetof(EngineCtx, cpuid0_eax_ebx),
        (uint64_t)offsetof(EngineCtx, cpuid0_ecx_edx),
        (uint64_t)SVC_DIRECT,
        (uint64_t)SVC_JCC,
        (uint64_t)SVC_CALL,
        (uint64_t)SVC_DEAD,
        (uint64_t)SVC_STOP,
        (uint64_t)SVC_X87,
        (uint64_t)SVC_H_SHIFT,
        (uint64_t)SVC_H_SHD,
        (uint64_t)SVC_H_MUL,
        (uint64_t)SVC_H_DIV,
        (uint64_t)SVC_H_PCMPISTRI,
        (uint64_t)SVC_LSL,
    };
    return aot_xxh64(sig, sizeof(sig), AOT_XXH_ABI_SEED);
}

// ---- Internal Structures ----

typedef struct AotArenaChunk {
    struct AotArenaChunk *next;
    size_t cap;
    size_t used;
    uint8_t data[];
} AotArenaChunk;

typedef struct {
    uint64_t key_hash;
    AotDiskRecordHeader *rec;
} AotStageSlot;

typedef struct {
    uint64_t frame_gpa;
    uint64_t page_hash;
    uint8_t  valid;
} AotFrameHashEntry;

struct AotCache {
    char    *path;
    uint64_t abi_hash;
    uint64_t max_bytes;

    uint8_t *disk_map;
    size_t   disk_size;
    const AotIndexSlot *disk_index;
    uint32_t disk_buckets;
    uint32_t disk_entry_count;
    uint32_t disk_t2_count;
    uint64_t disk_payload_off;

    pthread_mutex_t lock;
    AotArenaChunk  *arena_head;
    size_t          staged_bytes;
    size_t          staged_t2_bytes;
    AotStageSlot   *stage_table;
    uint32_t        stage_buckets;
    uint32_t        stage_count;
    uint32_t        stage_t2_count;
    uint32_t        dirty_since_save;
    uint64_t        save_seq;

    pthread_mutex_t save_lock;
    bool            async_save_running;
    pthread_t       async_save_tid;
    bool            async_save_joinable;
    uint64_t        last_save_mono_s;

    AotFrameHashEntry *frame_cache;

#ifdef __APPLE__
    mach_timebase_info_data_t tb;
#endif
    AotStats stats;
};

// ---- Helpers for Unpacking Disk/Staged Records ----

static bool aot_unpack_record(const uint8_t *base, size_t limit_bytes, uint64_t rec_off,
                              bool verify_crc, AotBlockView *out_view) {
    if ((rec_off & 7ULL) != 0 || rec_off + sizeof(AotDiskRecordHeader) > limit_bytes)
        return false;
    const AotDiskRecordHeader *hdr = (const AotDiskRecordHeader *)(base + rec_off);
    if (hdr->code_words == 0 || hdr->reloc_count > AOT_MAX_RELOCS_PER_BLOCK)
        return false;
    size_t need = aot_record_disk_size(hdr->code_words, hdr->pcm_count, hdr->reloc_count, hdr->has_t2_meta);
    if (rec_off + need > limit_bytes)
        return false;

    const uint8_t *rec_start = base + rec_off;
    size_t off = sizeof(AotDiskRecordHeader);
    const uint32_t *code = (const uint32_t *)(rec_start + off);
    off += (size_t)hdr->code_words * sizeof(uint32_t);
    const PcMap *pcm = (const PcMap *)(rec_start + off);
    off += (size_t)hdr->pcm_count * sizeof(PcMap);
    const AotReloc *relocs = (const AotReloc *)(rec_start + off);
    off += (size_t)hdr->reloc_count * sizeof(AotReloc);
    off = aot_align8(off);
    const T2TrapMeta *t2m = hdr->has_t2_meta ? (const T2TrapMeta *)(rec_start + off) : NULL;

    if (verify_crc) {
        uint32_t got = aot_compute_record_xxh32(hdr, code, pcm, relocs, t2m);
        if (got != hdr->rec_xxh32)
            return false;
    }

    out_view->key = hdr->key;
    out_view->orig_rip = hdr->orig_rip;
    out_view->code = code;
    out_view->code_words = hdr->code_words;
    out_view->pcm = pcm;
    out_view->pcm_count = hdr->pcm_count;
    out_view->relocs = relocs;
    out_view->reloc_count = hdr->reloc_count;
    out_view->read_off = hdr->read_off;
    out_view->read_bytes = hdr->read_bytes;
    out_view->t2_patch_word_off = hdr->t2_patch_word_off;
    out_view->t2_head_off = hdr->t2_head_off;
    out_view->t2_min_off = hdr->t2_min_off;
    out_view->t2_span_bytes = hdr->t2_span_bytes;
    out_view->priority = hdr->priority;
    out_view->has_t2_meta = hdr->has_t2_meta != 0;
    if (t2m)
        memcpy(&out_view->t2_meta, t2m, sizeof(T2TrapMeta));
    else
        memset(&out_view->t2_meta, 0, sizeof(T2TrapMeta));
    return true;
}

static void aot_view_from_hdr(const AotDiskRecordHeader *hdr, AotBlockView *out_view) {
    const uint8_t *rec_start = (const uint8_t *)hdr;
    size_t off = sizeof(AotDiskRecordHeader);
    const uint32_t *code = (const uint32_t *)(rec_start + off);
    off += (size_t)hdr->code_words * sizeof(uint32_t);
    const PcMap *pcm = (const PcMap *)(rec_start + off);
    off += (size_t)hdr->pcm_count * sizeof(PcMap);
    const AotReloc *relocs = (const AotReloc *)(rec_start + off);
    off += (size_t)hdr->reloc_count * sizeof(AotReloc);
    off = aot_align8(off);
    const T2TrapMeta *t2m = hdr->has_t2_meta ? (const T2TrapMeta *)(rec_start + off) : NULL;

    out_view->key = hdr->key;
    out_view->orig_rip = hdr->orig_rip;
    out_view->code = code;
    out_view->code_words = hdr->code_words;
    out_view->pcm = pcm;
    out_view->pcm_count = hdr->pcm_count;
    out_view->relocs = relocs;
    out_view->reloc_count = hdr->reloc_count;
    out_view->read_off = hdr->read_off;
    out_view->read_bytes = hdr->read_bytes;
    out_view->t2_patch_word_off = hdr->t2_patch_word_off;
    out_view->t2_head_off = hdr->t2_head_off;
    out_view->t2_min_off = hdr->t2_min_off;
    out_view->t2_span_bytes = hdr->t2_span_bytes;
    out_view->priority = hdr->priority;
    out_view->has_t2_meta = hdr->has_t2_meta != 0;
    if (t2m)
        memcpy(&out_view->t2_meta, t2m, sizeof(T2TrapMeta));
    else
        memset(&out_view->t2_meta, 0, sizeof(T2TrapMeta));
}

// ---- Staging Arena & Hash Table ----

static void *aot_arena_alloc_locked(AotCache *c, size_t bytes) {
    bytes = aot_align8(bytes);
    if (!c->arena_head || c->arena_head->cap - c->arena_head->used < bytes) {
        size_t chunk_cap = AOT_STAGE_ARENA_CHUNK;
        if (bytes > chunk_cap)
            chunk_cap = aot_align8(bytes);
        AotArenaChunk *ch = (AotArenaChunk *)malloc(sizeof(AotArenaChunk) + chunk_cap);
        if (!ch)
            return NULL;
        ch->next = c->arena_head;
        ch->cap = chunk_cap;
        ch->used = 0;
        c->arena_head = ch;
    }
    void *ptr = c->arena_head->data + c->arena_head->used;
    c->arena_head->used += bytes;
    return ptr;
}

static bool aot_stage_grow_locked(AotCache *c) {
    uint32_t new_cap = c->stage_buckets ? c->stage_buckets * 2U : AOT_STAGE_INIT_BUCKETS;
    AotStageSlot *nt = (AotStageSlot *)calloc(new_cap, sizeof(AotStageSlot));
    if (!nt)
        return false;
    uint32_t mask = new_cap - 1U;
    for (uint32_t i = 0; i < c->stage_buckets; i++) {
        if (!c->stage_table[i].rec)
            continue;
        uint32_t h = (uint32_t)c->stage_table[i].key_hash & mask;
        while (nt[h].rec)
            h = (h + 1U) & mask;
        nt[h] = c->stage_table[i];
    }
    free(c->stage_table);
    c->stage_table = nt;
    c->stage_buckets = new_cap;
    return true;
}

static const AotDiskRecordHeader *aot_stage_find_locked(const AotCache *c, const AotKey *key, uint64_t kh) {
    if (!c->stage_table || c->stage_buckets == 0)
        return NULL;
    uint32_t mask = c->stage_buckets - 1U;
    for (uint32_t h = (uint32_t)kh & mask, n = 0; n < c->stage_buckets; n++, h = (h + 1U) & mask) {
        const AotStageSlot *s = &c->stage_table[h];
        if (!s->rec)
            return NULL;
        if (s->key_hash == kh && aot_key_eq(&s->rec->key, key))
            return s->rec;
    }
    return NULL;
}

static bool aot_disk_find(const AotCache *c, const AotKey *key, uint64_t kh, AotBlockView *out_view) {
    if (!c->disk_map || !c->disk_index || c->disk_buckets == 0)
        return false;
    uint32_t mask = c->disk_buckets - 1U;
    for (uint32_t h = (uint32_t)kh & mask, n = 0; n < c->disk_buckets; n++, h = (h + 1U) & mask) {
        const AotIndexSlot *s = &c->disk_index[h];
        if (s->rec_off == 0)
            return false;
        if (s->key_hash != kh)
            continue;
        AotBlockView v;
        if (aot_unpack_record(c->disk_map, c->disk_size, s->rec_off, false, &v) &&
            aot_key_eq(&v.key, key)) {
            if (out_view)
                *out_view = v;
            return true;
        }
    }
    return false;
}

// ---- Lifecycle: Open / Close ----

static void aot_try_mmap_disk(AotCache *c) {
    int fd = open(c->path, O_RDONLY);
    if (fd < 0)
        return;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(AotFileHeader)) {
        if (st.st_size > 0)
            c->stats.corrupt_files_ignored++;
        close(fd);
        return;
    }
    size_t fsz = (size_t)st.st_size;
    void *m = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
        return;

    const uint8_t *base = (const uint8_t *)m;
    const AotFileHeader *hdr = (const AotFileHeader *)base;

    if (memcmp(hdr->magic, AOT_MAGIC_STR, 8) != 0 ||
        hdr->format_version != AOT_FORMAT_VERSION ||
        hdr->header_bytes != sizeof(AotFileHeader) ||
        hdr->file_bytes != (uint64_t)fsz) {
        c->stats.corrupt_files_ignored++;
        munmap(m, fsz);
        return;
    }
    if (hdr->abi_hash != c->abi_hash) {
        c->stats.abi_mismatches_ignored++;
        munmap(m, fsz);
        return;
    }
    if (hdr->index_buckets == 0 || (hdr->index_buckets & (hdr->index_buckets - 1U)) != 0) {
        c->stats.corrupt_files_ignored++;
        munmap(m, fsz);
        return;
    }
    uint64_t idx_bytes = (uint64_t)hdr->index_buckets * sizeof(AotIndexSlot);
    if (hdr->index_offset != sizeof(AotFileHeader) ||
        hdr->payload_offset != hdr->index_offset + idx_bytes ||
        hdr->payload_offset > fsz) {
        c->stats.corrupt_files_ignored++;
        munmap(m, fsz);
        return;
    }
    uint64_t calc = aot_xxh64(base + hdr->header_bytes, fsz - hdr->header_bytes, AOT_XXH_FILE_SEED);
    if (calc != hdr->payload_xxh64) {
        c->stats.corrupt_files_ignored++;
        munmap(m, fsz);
        return;
    }

    c->disk_map = (uint8_t *)m;
    c->disk_size = fsz;
    c->disk_index = (const AotIndexSlot *)(base + hdr->index_offset);
    c->disk_buckets = hdr->index_buckets;
    c->disk_entry_count = hdr->entry_count;
    c->disk_t2_count = hdr->t2_entry_count;
    c->disk_payload_off = hdr->payload_offset;
    c->stats.loaded_from_disk = hdr->entry_count;
    c->stats.loaded_t2_from_disk = hdr->t2_entry_count;
}

AotCache *aot_open(const char *path, uint64_t abi_hash, uint64_t max_bytes) {
    if (!path || !path[0])
        return NULL;
    AotCache *c = (AotCache *)calloc(1, sizeof(AotCache));
    if (!c)
        return NULL;
    c->path = strdup(path);
    if (!c->path) {
        free(c);
        return NULL;
    }
    c->abi_hash = abi_hash;
    c->max_bytes = max_bytes ? max_bytes : AOT_DEFAULT_MAX_BYTES;
    pthread_mutex_init(&c->lock, NULL);
    pthread_mutex_init(&c->save_lock, NULL);
    c->frame_cache = (AotFrameHashEntry *)calloc(AOT_FRAME_CACHE_SLOTS, sizeof(AotFrameHashEntry));
#ifdef __APPLE__
    mach_timebase_info(&c->tb);
    if (!c->tb.denom) {
        c->tb.numer = 1;
        c->tb.denom = 1;
    }
#endif
    c->stats.enabled = 1;
    struct timespec ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_now);
    c->last_save_mono_s = (uint64_t)ts_now.tv_sec;
    aot_try_mmap_disk(c);
    return c;
}

void aot_close(AotCache *c, bool save_dirty) {
    if (!c)
        return;
    pthread_mutex_lock(&c->save_lock);
    bool joinable = c->async_save_joinable;
    pthread_t tid = c->async_save_tid;
    c->async_save_joinable = false;
    pthread_mutex_unlock(&c->save_lock);
    if (joinable)
        pthread_join(tid, NULL);

    if (save_dirty) {
        pthread_mutex_lock(&c->lock);
        uint32_t dirty = c->dirty_since_save;
        pthread_mutex_unlock(&c->lock);
        if (dirty > 0)
            (void)aot_save(c);
    }

    if (c->disk_map)
        munmap(c->disk_map, c->disk_size);
    AotArenaChunk *ch = c->arena_head;
    while (ch) {
        AotArenaChunk *nx = ch->next;
        free(ch);
        ch = nx;
    }
    free(c->stage_table);
    free(c->frame_cache);
    free(c->path);
    pthread_mutex_destroy(&c->save_lock);
    pthread_mutex_destroy(&c->lock);
    free(c);
}

// ---- Per-Frame Page Hash Cache ----

static inline uint32_t aot_frame_slot(uint64_t frame_gpa) {
    uint64_t pfn = frame_gpa >> 12;
    return (uint32_t)((pfn * 0x9E3779B97F4A7C15ULL) >> (64 - 14)) & (AOT_FRAME_CACHE_SLOTS - 1U);
}

uint64_t aot_hash_frame_cached(AotCache *c, uint64_t frame_gpa, const uint8_t page[4096]) {
    if (!c || !c->frame_cache)
        return aot_hash_page(page);
    uint64_t aligned = frame_gpa & ~0xFFFULL;
    uint32_t slot = aot_frame_slot(aligned);
    pthread_mutex_lock(&c->lock);
    AotFrameHashEntry *e = &c->frame_cache[slot];
    if (e->valid && e->frame_gpa == aligned) {
        uint64_t h = e->page_hash;
        c->stats.page_hash_cache_hits++;
        pthread_mutex_unlock(&c->lock);
        return h;
    }
    pthread_mutex_unlock(&c->lock);

    uint64_t h = aot_hash_page(page);

    pthread_mutex_lock(&c->lock);
    e = &c->frame_cache[slot];
    e->frame_gpa = aligned;
    e->page_hash = h;
    e->valid = 1;
    c->stats.page_hash_cache_misses++;
    pthread_mutex_unlock(&c->lock);
    return h;
}

void aot_invalidate_frame_hash(AotCache *c, uint64_t frame_gpa) {
    if (!c || !c->frame_cache)
        return;
    uint64_t aligned = frame_gpa & ~0xFFFULL;
    uint32_t slot = aot_frame_slot(aligned);
    pthread_mutex_lock(&c->lock);
    if (c->frame_cache[slot].frame_gpa == aligned)
        c->frame_cache[slot].valid = 0;
    pthread_mutex_unlock(&c->lock);
}

void aot_clear_frame_hashes(AotCache *c) {
    if (!c || !c->frame_cache)
        return;
    pthread_mutex_lock(&c->lock);
    memset(c->frame_cache, 0, AOT_FRAME_CACHE_SLOTS * sizeof(AotFrameHashEntry));
    pthread_mutex_unlock(&c->lock);
}

// ---- Lookup & Recording ----

bool aot_lookup(AotCache *c, const AotKey *key, AotBlockView *out_view) {
    if (!c || !key || key->abi_hash != c->abi_hash)
        return false;
    uint64_t kh = aot_hash_key(key);
    __atomic_fetch_add(&c->stats.lookups, 1ULL, __ATOMIC_RELAXED);

    // Lock-free fast path: immutable read-only mmap'd disk index first!
    AotBlockView dv;
    if (aot_disk_find(c, key, kh, &dv)) {
        if (out_view)
            *out_view = dv;
        if (key->mode_bits & AOT_MODE_IS_T2)
            __atomic_fetch_add(&c->stats.hits_t2, 1ULL, __ATOMIC_RELAXED);
        else
            __atomic_fetch_add(&c->stats.hits_t1, 1ULL, __ATOMIC_RELAXED);
        return true;
    }

    if (__atomic_load_n(&c->stage_count, __ATOMIC_ACQUIRE) > 0) {
        pthread_mutex_lock(&c->lock);
        const AotDiskRecordHeader *staged = aot_stage_find_locked(c, key, kh);
        if (staged) {
            if (out_view)
                aot_view_from_hdr(staged, out_view);
            pthread_mutex_unlock(&c->lock);
            if (key->mode_bits & AOT_MODE_IS_T2)
                __atomic_fetch_add(&c->stats.hits_t2, 1ULL, __ATOMIC_RELAXED);
            else
                __atomic_fetch_add(&c->stats.hits_t1, 1ULL, __ATOMIC_RELAXED);
            return true;
        }
        pthread_mutex_unlock(&c->lock);
    }

    __atomic_fetch_add(&c->stats.misses, 1ULL, __ATOMIC_RELAXED);
    return false;
}

bool aot_record(AotCache *c, const AotBlockView *view) {
    if (!c || !view || !view->code || view->code_words == 0 ||
        view->reloc_count > AOT_MAX_RELOCS_PER_BLOCK ||
        view->key.abi_hash != c->abi_hash)
        return false;

    uint64_t kh = aot_hash_key(&view->key);
    bool is_t2 = (view->priority >= AOT_PRIO_TIER2) || ((view->key.mode_bits & AOT_MODE_IS_T2) != 0);
    uint8_t has_t2m = view->has_t2_meta ? 1U : 0U;
    size_t rec_sz = aot_record_disk_size(view->code_words, view->pcm_count, view->reloc_count, has_t2m);

    pthread_mutex_lock(&c->lock);

    if (aot_stage_find_locked(c, &view->key, kh) != NULL) {
        c->stats.recorded_dups++;
        pthread_mutex_unlock(&c->lock);
        return true;
    }
    if (aot_disk_find(c, &view->key, kh, NULL)) {
        c->stats.recorded_dups++;
        pthread_mutex_unlock(&c->lock);
        return true;
    }

    if (!is_t2 && c->staged_bytes + rec_sz > c->max_bytes) {
        pthread_mutex_unlock(&c->lock);
        return false;
    }
    if (is_t2 && c->staged_t2_bytes + rec_sz > c->max_bytes) {
        pthread_mutex_unlock(&c->lock);
        return false;
    }

    if (c->stage_buckets == 0 || (uint64_t)(c->stage_count + 1U) * 10ULL >= (uint64_t)c->stage_buckets * 7ULL) {
        if (!aot_stage_grow_locked(c)) {
            pthread_mutex_unlock(&c->lock);
            return false;
        }
    }

    uint8_t *buf = (uint8_t *)aot_arena_alloc_locked(c, rec_sz);
    if (!buf) {
        pthread_mutex_unlock(&c->lock);
        return false;
    }
    memset(buf, 0, rec_sz);

    AotDiskRecordHeader *hdr = (AotDiskRecordHeader *)buf;
    hdr->key = view->key;
    hdr->orig_rip = view->orig_rip;
    hdr->code_words = view->code_words;
    hdr->pcm_count = view->pcm_count;
    hdr->reloc_count = view->reloc_count;
    hdr->read_off = view->read_off;
    hdr->read_bytes = view->read_bytes;
    hdr->t2_patch_word_off = view->t2_patch_word_off;
    hdr->t2_head_off = view->t2_head_off;
    hdr->t2_min_off = view->t2_min_off;
    hdr->t2_span_bytes = view->t2_span_bytes;
    hdr->priority = is_t2 ? AOT_PRIO_TIER2 : AOT_PRIO_TIER1;
    hdr->has_t2_meta = has_t2m;

    size_t off = sizeof(AotDiskRecordHeader);
    memcpy(buf + off, view->code, (size_t)view->code_words * sizeof(uint32_t));
    off += (size_t)view->code_words * sizeof(uint32_t);
    if (view->pcm_count && view->pcm)
        memcpy(buf + off, view->pcm, (size_t)view->pcm_count * sizeof(PcMap));
    off += (size_t)view->pcm_count * sizeof(PcMap);
    if (view->reloc_count && view->relocs)
        memcpy(buf + off, view->relocs, (size_t)view->reloc_count * sizeof(AotReloc));
    off += (size_t)view->reloc_count * sizeof(AotReloc);
    off = aot_align8(off);
    if (has_t2m)
        memcpy(buf + off, &view->t2_meta, sizeof(T2TrapMeta));

    hdr->rec_xxh32 = aot_compute_record_xxh32(hdr, view->code, view->pcm, view->relocs,
                                              has_t2m ? &view->t2_meta : NULL);

    uint32_t mask = c->stage_buckets - 1U;
    uint32_t h = (uint32_t)kh & mask;
    while (c->stage_table[h].rec)
        h = (h + 1U) & mask;
    c->stage_table[h].key_hash = kh;
    c->stage_table[h].rec = hdr;

    __atomic_store_n(&c->stage_count, c->stage_count + 1U, __ATOMIC_RELEASE);
    c->staged_bytes += rec_sz;
    if (is_t2) {
        c->stage_t2_count++;
        c->staged_t2_bytes += rec_sz;
        c->stats.recorded_t2++;
    } else {
        c->stats.recorded_t1++;
    }
    c->dirty_since_save++;

    pthread_mutex_unlock(&c->lock);
    return true;
}

// ---- Fixed-Width Relocatable GVA Emitter & Stub Resolver ----

uint32_t aot_emit_movimm_gva(uint32_t **cc_p, uint32_t *block_start, AotCapture *cap,
                             int rd, uint64_t gva, bool force32) {
    uint32_t *cc = *cc_p;
    uint32_t w_off = (uint32_t)(cc - block_start);
    uint32_t reg = (uint32_t)rd & 31U;

    if (force32) {
        uint32_t v32 = (uint32_t)gva;
        cc[0] = 0xD2800000u | ((v32 & 0xFFFFu) << 5) | reg;
        cc[1] = 0xF2A00000u | (((v32 >> 16) & 0xFFFFu) << 5) | reg;
        *cc_p = cc + 2;
        aot_capture_add(cap, w_off, AOT_RELOC_MOVIMM32, (uint8_t)reg, 0);
        return 2;
    }

    uint64_t top = gva >> 48;
    if (top == 0ULL || top == 0xFFFFULL) {
        bool neg = (top == 0xFFFFULL);
        uint32_t h0 = (uint32_t)(gva & 0xFFFFu);
        uint32_t h1 = (uint32_t)((gva >> 16) & 0xFFFFu);
        uint32_t h2 = (uint32_t)((gva >> 32) & 0xFFFFu);
        cc[0] = (neg ? 0x92800000u : 0xD2800000u) | ((neg ? (~h0 & 0xFFFFu) : h0) << 5) | reg;
        cc[1] = 0xF2A00000u | (h1 << 5) | reg;
        cc[2] = 0xF2C00000u | (h2 << 5) | reg;
        *cc_p = cc + 3;
        aot_capture_add(cap, w_off, AOT_RELOC_MOVIMM48, (uint8_t)reg, 0);
        return 3;
    }

    cc[0] = 0xD2800000u | ((uint32_t)(gva & 0xFFFFu) << 5) | reg;
    cc[1] = 0xF2A00000u | ((uint32_t)((gva >> 16) & 0xFFFFu) << 5) | reg;
    cc[2] = 0xF2C00000u | ((uint32_t)((gva >> 32) & 0xFFFFu) << 5) | reg;
    cc[3] = 0xF2E00000u | ((uint32_t)((gva >> 48) & 0xFFFFu) << 5) | reg;
    *cc_p = cc + 4;
    aot_capture_add(cap, w_off, AOT_RELOC_MOVIMM64, (uint8_t)reg, 0);
    return 4;
}

uint64_t aot_resolve_stub_va(const AotStubTable *stubs, AotStubId id, int part, uint64_t pc_va) {
    uint64_t base = stubs->va[id];
#ifdef OFF_STUBS_HI
    bool hi = pc_va >= PRIV_VA + (part ? OFF_CC2 : OFF_CC) + (124ULL << 20);
    if (hi)
        return base + (part ? (OFF_STUBS2_HI - OFF_STUBS) : (OFF_STUBS_HI - OFF_STUBS));
#else
    (void)pc_va;
#endif
    return base + (part ? (OFF_STUBS2 - OFF_STUBS) : 0ULL);
}

// ---- Materialize + In-Place ASLR / Stub / Cross-Block Relocation ----

bool aot_materialize(AotCache *c,
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
                     void *resolve_ctx) {
    if (!view || !dst_code || !stubs)
        return false;
    if (view->code_words == 0 || view->code_words > dst_code_cap_words)
        return false;
    if (view->pcm_count > dst_pcm_cap)
        return false;
    if ((new_rip & 0xFFFULL) != (uint64_t)view->key.page_offset)
        goto reloc_fail;

#ifdef __APPLE__
    uint64_t t0 = mach_absolute_time();
#endif

    bool compat = (view->key.mode_bits & AOT_MODE_COMPAT) != 0;
    uint64_t delta = new_rip - view->orig_rip;

    memcpy(dst_code, view->code, (size_t)view->code_words * sizeof(uint32_t));
    if (view->pcm_count && dst_pcm && view->pcm)
        memcpy(dst_pcm, view->pcm, (size_t)view->pcm_count * sizeof(PcMap));

    // Pass 1: Apply all GVA and stub relocations.
    for (uint16_t i = 0; i < view->reloc_count; i++) {
        const AotReloc *r = &view->relocs[i];
        uint32_t w = r->word_off;
        switch (r->type) {
            case AOT_RELOC_GVA64_LIT: {
                if (w + 1U >= view->code_words)
                    goto reloc_fail;
                uint64_t old_v = (uint64_t)dst_code[w] | ((uint64_t)dst_code[w + 1U] << 32);
                uint64_t new_v = compat ? (uint64_t)(uint32_t)(old_v + delta) : (old_v + delta);
                dst_code[w] = (uint32_t)new_v;
                dst_code[w + 1U] = (uint32_t)(new_v >> 32);
                break;
            }
            case AOT_RELOC_MOVIMM32: {
                if (w + 1U >= view->code_words)
                    goto reloc_fail;
                uint32_t rd = (uint32_t)r->aux & 31U;
                uint64_t old_v = (uint64_t)((dst_code[w] >> 5) & 0xFFFFu) |
                                 ((uint64_t)((dst_code[w + 1U] >> 5) & 0xFFFFu) << 16);
                uint32_t new_v = (uint32_t)(old_v + delta);
                dst_code[w]      = 0xD2800000u | ((new_v & 0xFFFFu) << 5) | rd;
                dst_code[w + 1U] = 0xF2A00000u | (((new_v >> 16) & 0xFFFFu) << 5) | rd;
                break;
            }
            case AOT_RELOC_MOVIMM48: {
                if (w + 2U >= view->code_words)
                    goto reloc_fail;
                uint32_t rd = (uint32_t)r->aux & 31U;
                bool was_neg = ((dst_code[w] >> 30) & 1U) == 0U;
                uint64_t h0 = (dst_code[w] >> 5) & 0xFFFFu;
                if (was_neg)
                    h0 = (~h0) & 0xFFFFu;
                uint64_t h1 = (dst_code[w + 1U] >> 5) & 0xFFFFu;
                uint64_t h2 = (dst_code[w + 2U] >> 5) & 0xFFFFu;
                uint64_t old_v = h0 | (h1 << 16) | (h2 << 32) |
                                 (was_neg ? 0xFFFF000000000000ULL : 0ULL);
                uint64_t new_v = old_v + delta;
                uint64_t top = new_v >> 48;
                if (top != 0ULL && top != 0xFFFFULL)
                    goto reloc_fail;
                bool neg = (top == 0xFFFFULL);
                uint32_t nh0 = (uint32_t)(new_v & 0xFFFFu);
                dst_code[w]      = (neg ? 0x92800000u : 0xD2800000u) |
                                   ((neg ? (~nh0 & 0xFFFFu) : nh0) << 5) | rd;
                dst_code[w + 1U] = 0xF2A00000u | ((uint32_t)((new_v >> 16) & 0xFFFFu) << 5) | rd;
                dst_code[w + 2U] = 0xF2C00000u | ((uint32_t)((new_v >> 32) & 0xFFFFu) << 5) | rd;
                break;
            }
            case AOT_RELOC_MOVIMM64: {
                if (w + 3U >= view->code_words)
                    goto reloc_fail;
                uint32_t rd = (uint32_t)r->aux & 31U;
                uint64_t old_v = (uint64_t)((dst_code[w] >> 5) & 0xFFFFu) |
                                 ((uint64_t)((dst_code[w + 1U] >> 5) & 0xFFFFu) << 16) |
                                 ((uint64_t)((dst_code[w + 2U] >> 5) & 0xFFFFu) << 32) |
                                 ((uint64_t)((dst_code[w + 3U] >> 5) & 0xFFFFu) << 48);
                uint64_t new_v = old_v + delta;
                dst_code[w]      = 0xD2800000u | ((uint32_t)(new_v & 0xFFFFu) << 5) | rd;
                dst_code[w + 1U] = 0xF2A00000u | ((uint32_t)((new_v >> 16) & 0xFFFFu) << 5) | rd;
                dst_code[w + 2U] = 0xF2C00000u | ((uint32_t)((new_v >> 32) & 0xFFFFu) << 5) | rd;
                dst_code[w + 3U] = 0xF2E00000u | ((uint32_t)((new_v >> 48) & 0xFFFFu) << 5) | rd;
                break;
            }
            case AOT_RELOC_STUB_BRANCH: {
                if (w >= view->code_words)
                    goto reloc_fail;
                uint8_t stub_id = r->aux & 0x7FU;
                bool is_bl = (r->aux & 0x80U) != 0;
                if (stub_id >= AOT_STUB_COUNT)
                    goto reloc_fail;
                uint64_t pc_va = dst_cc_va + (uint64_t)w * 4ULL;
                uint64_t stub_va = aot_resolve_stub_va(stubs, (AotStubId)stub_id, dst_part, pc_va);
                int64_t disp = (int64_t)(stub_va - pc_va) / 4;
                if (disp < -(1LL << 25) || disp >= (1LL << 25))
                    goto reloc_fail;
                dst_code[w] = (is_bl ? 0x94000000u : 0x14000000u) | ((uint32_t)disp & 0x3FFFFFFu);
                break;
            }
            case AOT_RELOC_CC_BRANCH:
            case AOT_RELOC_T2_BAILOUT:
                break;
            default:
                goto reloc_fail;
        }
    }

    // Pass 2: Optional eager cross-block direct-linking.
    uint64_t linked = 0;
    if (resolve_target) {
        for (uint16_t i = 0; i < view->reloc_count; i++) {
            const AotReloc *r = &view->relocs[i];
            if (r->type != AOT_RELOC_CC_BRANCH)
                continue;
            uint32_t w = r->word_off;
            uint32_t lit_w = r->aux2;
            if (w >= view->code_words || lit_w + 1U >= view->code_words)
                goto reloc_fail;
            uint64_t target_rip = (uint64_t)dst_code[lit_w] | ((uint64_t)dst_code[lit_w + 1U] << 32);
            uint64_t target_arm = resolve_target(resolve_ctx, target_rip);
            if (!target_arm)
                continue;
            uint64_t site_va = dst_cc_va + (uint64_t)w * 4ULL;
            int64_t d = (int64_t)(target_arm - site_va) / 4;
            if (r->aux == AOT_CC_DIRECT) {
                if (d >= -(1LL << 25) && d < (1LL << 25)) {
                    dst_code[w] = 0x14000000u | ((uint32_t)d & 0x3FFFFFFu);
                    linked++;
                }
            } else if (r->aux == AOT_CC_CALL) {
                if (d >= -(1LL << 25) && d < (1LL << 25)) {
                    dst_code[w] = 0x94000000u | ((uint32_t)d & 0x3FFFFFFu);
                    linked++;
                }
            } else if (r->aux == AOT_CC_JCC) {
                if (d >= -(1LL << 18) && d < (1LL << 18)) {
                    dst_code[w] = (dst_code[w] & ~(0x7FFFFu << 5)) | (((uint32_t)d & 0x7FFFFu) << 5);
                    linked++;
                }
            }
        }
    }

    if (c) {
#ifdef __APPLE__
        uint64_t dt = mach_absolute_time() - t0;
        uint64_t ns = (uint64_t)(((__uint128_t)dt * c->tb.numer) / c->tb.denom);
#else
        uint64_t ns = 0;
#endif
        if (linked)
            __atomic_fetch_add(&c->stats.eager_cc_links, linked, __ATOMIC_RELAXED);
        __atomic_fetch_add(&c->stats.reloc_ns_total, ns, __ATOMIC_RELAXED);
    }

    return true;

reloc_fail:
    if (c)
        __atomic_fetch_add(&c->stats.reloc_failures, 1ULL, __ATOMIC_RELAXED);
    return false;
}

// ---- Crash-Safe Atomic Persistence (Tier-2 Prioritized First) ----

typedef struct {
    const uint8_t *raw;
    size_t         raw_size;
    uint64_t       key_hash;
    uint32_t       priority;
} AotEmitItem;

static bool aot_write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        p += (size_t)w;
        len -= (size_t)w;
    }
    return true;
}

static uint32_t aot_next_pow2(uint32_t v) {
    if (v < 64U)
        return 64U;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1U;
}

bool aot_save(AotCache *c) {
    if (!c || !c->path)
        return false;

    pthread_mutex_lock(&c->save_lock);

    pthread_mutex_lock(&c->lock);
    uint32_t cap = c->stage_count + c->disk_entry_count + 16U;
    AotEmitItem *t2_items = (AotEmitItem *)malloc((size_t)cap * sizeof(AotEmitItem));
    AotEmitItem *t1_items = (AotEmitItem *)malloc((size_t)cap * sizeof(AotEmitItem));
    uint32_t t2_n = 0, t1_n = 0;
    uint64_t seq = ++c->save_seq;
    uint32_t dirty_snapshot = c->dirty_since_save;

    if (!t2_items || !t1_items) {
        free(t2_items);
        free(t1_items);
        c->stats.saves_failed++;
        pthread_mutex_unlock(&c->lock);
        pthread_mutex_unlock(&c->save_lock);
        return false;
    }

    // 1. Snapshot ONLY staged pointers under short c->lock (arena chunks & disk_map are immutable!).
    for (uint32_t i = 0; i < c->stage_buckets; i++) {
        const AotDiskRecordHeader *hdr = c->stage_table[i].rec;
        if (!hdr)
            continue;
        size_t rsz = aot_record_disk_size(hdr->code_words, hdr->pcm_count, hdr->reloc_count, hdr->has_t2_meta);
        AotEmitItem it = { (const uint8_t *)hdr, rsz, c->stage_table[i].key_hash, hdr->priority };
        if (hdr->priority >= AOT_PRIO_TIER2)
            t2_items[t2_n++] = it;
        else
            t1_items[t1_n++] = it;
    }
    pthread_mutex_unlock(&c->lock);

    // 2. Collect surviving entries from immutable mmap'd disk file WITHOUT holding c->lock!
    // Note: aot_record() already skips keys present in disk_map, so staged and disk keys are disjoint.
    if (c->disk_map && c->disk_index) {
        for (uint32_t i = 0; i < c->disk_buckets; i++) {
            uint64_t off = c->disk_index[i].rec_off;
            if (!off)
                continue;
            AotBlockView dv;
            if (!aot_unpack_record(c->disk_map, c->disk_size, off, false, &dv))
                continue;
            uint64_t kh = c->disk_index[i].key_hash;
            size_t rsz = aot_record_disk_size(dv.code_words, dv.pcm_count, dv.reloc_count, dv.has_t2_meta ? 1U : 0U);
            AotEmitItem it = { c->disk_map + off, rsz, kh, dv.priority };
            if (dv.priority >= AOT_PRIO_TIER2)
                t2_items[t2_n++] = it;
            else
                t1_items[t1_n++] = it;
        }
    }

    uint32_t keep_t2 = 0, keep_t1 = 0;
    size_t payload_bytes = 0;
    size_t max_payload = (c->max_bytes > 4096ULL) ? (size_t)(c->max_bytes - 4096ULL) : (size_t)c->max_bytes;

    for (uint32_t i = 0; i < t2_n; i++) {
        if (payload_bytes + t2_items[i].raw_size > max_payload)
            break;
        payload_bytes += t2_items[i].raw_size;
        keep_t2++;
    }
    for (uint32_t i = 0; i < t1_n; i++) {
        if (payload_bytes + t1_items[i].raw_size > max_payload)
            break;
        payload_bytes += t1_items[i].raw_size;
        keep_t1++;
    }

    uint32_t total_keep = keep_t2 + keep_t1;
    uint32_t buckets = aot_next_pow2(total_keep * 2U + 16U);
    size_t idx_bytes = (size_t)buckets * sizeof(AotIndexSlot);

    while (keep_t1 > 0 && sizeof(AotFileHeader) + idx_bytes + payload_bytes > c->max_bytes) {
        keep_t1--;
        payload_bytes -= t1_items[keep_t1].raw_size;
        total_keep = keep_t2 + keep_t1;
        buckets = aot_next_pow2(total_keep * 2U + 16U);
        idx_bytes = (size_t)buckets * sizeof(AotIndexSlot);
    }

    size_t total_file_bytes = sizeof(AotFileHeader) + idx_bytes + payload_bytes;
    uint8_t *out_buf = (uint8_t *)calloc(1, total_file_bytes);
    if (!out_buf) {
        free(t2_items);
        free(t1_items);
        pthread_mutex_lock(&c->lock);
        c->stats.saves_failed++;
        pthread_mutex_unlock(&c->lock);
        pthread_mutex_unlock(&c->save_lock);
        return false;
    }

    AotFileHeader *fh = (AotFileHeader *)out_buf;
    memcpy(fh->magic, AOT_MAGIC_STR, 8);
    fh->format_version = AOT_FORMAT_VERSION;
    fh->header_bytes = sizeof(AotFileHeader);
    fh->abi_hash = c->abi_hash;
    fh->index_buckets = buckets;
    fh->entry_count = total_keep;
    fh->t2_entry_count = keep_t2;
    fh->flags = 0;
    fh->index_offset = sizeof(AotFileHeader);
    fh->payload_offset = sizeof(AotFileHeader) + idx_bytes;
    fh->file_bytes = (uint64_t)total_file_bytes;

    AotIndexSlot *idx_tbl = (AotIndexSlot *)(out_buf + fh->index_offset);
    uint32_t mask = buckets - 1U;
    uint64_t cur_off = fh->payload_offset;

    for (uint32_t i = 0; i < keep_t2; i++) {
        memcpy(out_buf + cur_off, t2_items[i].raw, t2_items[i].raw_size);
        uint32_t h = (uint32_t)t2_items[i].key_hash & mask;
        while (idx_tbl[h].rec_off != 0)
            h = (h + 1U) & mask;
        idx_tbl[h].key_hash = t2_items[i].key_hash;
        idx_tbl[h].rec_off = cur_off;
        cur_off += t2_items[i].raw_size;
    }
    for (uint32_t i = 0; i < keep_t1; i++) {
        memcpy(out_buf + cur_off, t1_items[i].raw, t1_items[i].raw_size);
        uint32_t h = (uint32_t)t1_items[i].key_hash & mask;
        while (idx_tbl[h].rec_off != 0)
            h = (h + 1U) & mask;
        idx_tbl[h].key_hash = t1_items[i].key_hash;
        idx_tbl[h].rec_off = cur_off;
        cur_off += t1_items[i].raw_size;
    }

    fh->payload_xxh64 = aot_xxh64(out_buf + fh->header_bytes,
                                  total_file_bytes - fh->header_bytes,
                                  AOT_XXH_FILE_SEED);

    free(t2_items);
    free(t1_items);

    size_t plen = strlen(c->path);
    char *tmp_path = (char *)malloc(plen + 64U);
    bool ok = false;
    if (tmp_path) {
        snprintf(tmp_path, plen + 64U, "%s.tmp.%ld.%llu",
                 c->path, (long)getpid(), (unsigned long long)seq);
        int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            bool wrote = aot_write_all(fd, out_buf, total_file_bytes);
            bool synced = wrote && (fsync(fd) == 0);
            bool closed = (close(fd) == 0);
            if (synced && closed && rename(tmp_path, c->path) == 0)
                ok = true;
            else
                unlink(tmp_path);
        }
        free(tmp_path);
    }
    free(out_buf);

    pthread_mutex_lock(&c->lock);
    if (ok) {
        c->stats.saves_ok++;
        c->stats.saved_blocks_last = total_keep;
        c->stats.saved_t2_last = keep_t2;
        c->stats.saved_bytes_last = total_file_bytes;
        if (c->dirty_since_save >= dirty_snapshot)
            c->dirty_since_save -= dirty_snapshot;
        else
            c->dirty_since_save = 0;
    } else {
        c->stats.saves_failed++;
    }
    pthread_mutex_unlock(&c->lock);
    pthread_mutex_unlock(&c->save_lock);
    return ok;
}

static void *aot_async_save_worker(void *arg) {
    AotCache *c = (AotCache *)arg;
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
    (void)aot_save(c);
    pthread_mutex_lock(&c->save_lock);
    c->async_save_running = false;
    pthread_mutex_unlock(&c->save_lock);
    return NULL;
}

void aot_maybe_save_async(AotCache *c, uint32_t min_dirty) {
    if (!c)
        return;
    uint32_t dirty = __atomic_load_n(&c->dirty_since_save, __ATOMIC_RELAXED);
    if (dirty < min_dirty)
        return;

    struct timespec ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_now);
    uint64_t now_s = (uint64_t)ts_now.tv_sec;
    // Rate-limit background checkpoints to at most once every 60s so boot & benchmarks see zero fsync stalls.
    if (now_s < c->last_save_mono_s + 60ULL)
        return;

    pthread_mutex_lock(&c->save_lock);
    if (c->async_save_running || now_s < c->last_save_mono_s + 60ULL) {
        pthread_mutex_unlock(&c->save_lock);
        return;
    }
    if (c->async_save_joinable) {
        pthread_t old_tid = c->async_save_tid;
        c->async_save_joinable = false;
        pthread_mutex_unlock(&c->save_lock);
        pthread_join(old_tid, NULL);
        pthread_mutex_lock(&c->save_lock);
        if (c->async_save_running) {
            pthread_mutex_unlock(&c->save_lock);
            return;
        }
    }
    c->last_save_mono_s = now_s;
    c->async_save_running = true;
    if (pthread_create(&c->async_save_tid, NULL, aot_async_save_worker, c) == 0) {
        c->async_save_joinable = true;
    } else {
        c->async_save_running = false;
    }
    pthread_mutex_unlock(&c->save_lock);
}

AotStats aot_get_stats(const AotCache *c) {
    AotStats s;
    memset(&s, 0, sizeof s);
    if (!c)
        return s;
    pthread_mutex_lock(&((AotCache *)c)->lock);
    s = c->stats;
    pthread_mutex_unlock(&((AotCache *)c)->lock);
    return s;
}
