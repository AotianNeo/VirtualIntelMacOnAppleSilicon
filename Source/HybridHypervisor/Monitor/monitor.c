// EL1 monitor. Runs inside the VM beside translated x86 code, which stays at EL0 in both x86 rings: guest kernel code never
// holds EL1, so a wild kernel store cannot reach this monitor's tables. Fast x86 paths (syscall, faults, iret, CR3, page-table
// writes) finish here; only translation misses, devices and unknown MSRs reach the host through mon_hvc.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "engine_abi.h"
#include "compat_segments.h"
#include "monitor-code-fault.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t i64;

register EngineCtx *c __asm__("x18");

void mon_hvc(void);
void mon_fp_save(void *);
void mon_fp_restore(void *);
void mon_t2_read_pool(void *);
void mon_enter_guest(void);
_Noreturn void mon_restart(void);

static inline void sysreg_isb(void) { __asm__ volatile("isb"); }
#define RSR(name) ({ u64 v_; __asm__ volatile("mrs %0, " #name : "=r"(v_)); v_; })
#define WSR(name, v) __asm__ volatile("msr " #name ", %0" : : "r"((u64)(v)))

void *memset(void *d, int v, unsigned long n) {
    u8 *p = d;
    while (n--)
        *p++ = (u8)v;
    return d;
}
void *memcpy(void *d, const void *s, unsigned long n) {
    u8 *p = d;
    const u8 *q = s;
    while (n--)
        *p++ = *q++;
    return d;
}

// ---- One vCPU's own state ----
// x18 picks the vCPU. EngineCtx is reachable from EL0, so what the monitor keeps for itself sits at a fixed distance from it, in EL1 memory, with the stack at the
// slot's top. The defines keep the old names: this state was statics, and a static in this file is now shared by every vCPU, so it must either move here or be locked.
typedef struct {
    bool mon_resync, walk_any_tables, evicting, event_delivered, code_cache_reset, t2_synced;
    u32 zap_why, deliver_depth;
    u64 pt_step_slot, *pt_step_pte, pt_step_va;
    u64 note_wait_cr3, note_wait_status;
    u64 rs_saved[MAX_VIEWS]; // where this vCPU's return stack stood when it left each view. 0: not been there yet
    u32 lock_depth, rs_epoch;
    u64 smc_held; // frame + 1: this vCPU's one-off store to a code frame hasn't run yet, so nobody may protect that frame again
    u64 shadow_step_rip;
    bool shadow_stepping; // stepping an x86 instruction in the interrupt shadow, not just its first ARM word
    bool tracked_new; // a fill just started protecting a guest table, so what it read of that table may predate the protection
    u32 pin[3]; // view index + 1 of the current address space, by CPL 0 or 3, then the CPL 0 one before: a kernel with its own page tables crosses between two at every syscall. Eviction leaves these alone, so none of those crossings needs the lock
    u32 xfl;
    u32 lock_site;
    u64 lock_from;
    u64 code_retry_pc;
    u32 code_retry_count;
    u64 fex_l1_gen;
    u64 mon_hold_accum[6];
    u64 mon_wait_accum;
    u32 mon_stat_ticks;
    u32 hybrid_apic_id_cached;
    u32 hybrid_apic_id_val;
    u32 hybrid_apic_ver_cached;
    u32 hybrid_apic_ver_val;
    u64 hv_hc_va_cached;
    u64 hv_hc_gpa_cached;
    u64 hvc_ticks_local;
    u8 fx_area[512] __attribute__((aligned(16))); // the x87 save formats, one at a time
} MonCpu;
_Static_assert(sizeof(MonCpu) <= CTX_SLOT / 2, "the rest of the slot is the stack");
_Static_assert(sizeof(EngineCtx) <= CTX_SLOT, "EngineCtx outgrew its slot");
_Static_assert(OFF_MON_CPU + MAX_VCPUS * CTX_SLOT <= OFF_STUBS, "per-vCPU slots run past monitor data");
_Static_assert(OFF_CTX + MAX_VCPUS * CTX_SLOT <= OFF_CTX_END, "contexts run past their region");
_Static_assert(OFF_RSTACKS + MAX_VCPUS * MAX_VIEWS * RSTACK_SLOT <= PRIV_SIZE, "return stacks run past private memory");
#define me ((MonCpu *)((u64)c + (OFF_MON_CPU - OFF_CTX)))
#define c0 ((EngineCtx *)(PRIV_VA + OFF_CTX)) // what the host publishes once for the VM lives in the first context
#define cpu_index() ((u32)(((u64)c - (PRIV_VA + OFF_CTX)) / CTX_SLOT))
#define mon_resync (me->mon_resync)
#define walk_any_tables (me->walk_any_tables)
#define evicting (me->evicting)
#define event_delivered (me->event_delivered)
#define code_cache_reset (me->code_cache_reset)
#define zap_why (me->zap_why)
#define deliver_depth (me->deliver_depth)
#define pt_step_slot (me->pt_step_slot)
#define pt_step_pte (me->pt_step_pte)
#define pt_step_va (me->pt_step_va)
#define note_wait_cr3 (me->note_wait_cr3)
#define note_wait_status (me->note_wait_status)
#define xfl (me->xfl)

// ---- Shared state ----
// One lock over views, the shadow pool, rmap, mirrors and page_info, with no cost beyond a load at one vCPU.
// Nested takers just count, and `stop` lets go, since it abandons the stack that took it.
// FIFO admission bounds reacquisition starvation: a busy vCPU must not keep
// overtaking the compositor's vCPU during cold translation/page-table bursts.
// At most MAX_VCPUS tickets are outstanding, so equality remains valid across
// u32 wrap. Reentrance and abandoned-stack release consume only the outer ticket.
static u32 mon_lock_word __attribute__((aligned(64)));
static u32 mon_lock_owner __attribute__((aligned(64)));
static inline void hostcall(u64 code);
#define smp (c0->vcpu_count > 1)
static inline void mon_flush_stats(void) {
    if (me->mon_wait_accum) {
        __atomic_fetch_add(&c0->stat_mon_wait, me->mon_wait_accum, __ATOMIC_RELAXED);
        me->mon_wait_accum = 0;
    }
    for (u32 i = 0; i < 6; i++) {
        if (me->mon_hold_accum[i]) {
            __atomic_fetch_add(&c0->stat_mon_hold[i], me->mon_hold_accum[i], __ATOMIC_RELAXED);
            me->mon_hold_accum[i] = 0;
        }
    }
}
static inline void mon_lock(void) {
    if (!smp || me->lock_depth++)
        return;
    // mon_lock_word is a TTAS seqlock: even = unlocked, odd = locked.
    u32 s = __atomic_load_n(&mon_lock_word, __ATOMIC_RELAXED) & ~1u;
    if (__builtin_expect(__atomic_compare_exchange_n(&mon_lock_word, &s, s + 1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED), 1)) {
        __atomic_store_n(&mon_lock_owner, cpu_index(), __ATOMIC_RELAXED);
        me->lock_from = RSR(cntvct_el0);
        me->lock_site = 0;
        return;
    }
    u64 waiting_from = RSR(cntvct_el0);
    for (u32 spins = 1;; spins++) {
        // Test-and-Test-and-Set avoids ticket-lock preemption convoys on 16 vCPUs when a waiter yields to macOS.
        s = __atomic_load_n(&mon_lock_word, __ATOMIC_RELAXED);
        if (!(s & 1u) &&
            __atomic_compare_exchange_n(&mon_lock_word, &s, s + 1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            __atomic_store_n(&mon_lock_owner, cpu_index(), __ATOMIC_RELAXED);
            u64 now = RSR(cntvct_el0);
            me->mon_wait_accum += now - waiting_from;
            me->lock_from = now;
            me->lock_site = 0;
            return;
        }
        __asm__ volatile("yield");
        if (!(spins & 0xFFFF)) {
            u32 owner = __atomic_load_n(&mon_lock_owner, __ATOMIC_RELAXED);
            MonCpu *om = (MonCpu *)(PRIV_VA + OFF_MON_CPU + (u64)owner * CTX_SLOT);
            c->req_a[0] = owner;
            c->req_a[1] = RSR(cntvct_el0) - waiting_from;
            c->req_a[2] = om->lock_site;
            hostcall(HC_YIELD);
        }
    }
}
static inline void mon_unlock(void) {
    if (smp && me->lock_depth && !--me->lock_depth) {
        u32 site = me->lock_site;
        u64 held = RSR(cntvct_el0) - me->lock_from;
        u32 next_seq = __atomic_load_n(&mon_lock_word, __ATOMIC_RELAXED) + 1;
        __atomic_store_n(&mon_lock_word, next_seq, __ATOMIC_RELEASE);
        me->mon_hold_accum[site] += held;
        if (!(++me->mon_stat_ticks & 63))
            mon_flush_stats();
    }
}
// The shadow pool and its lists go wrong silently without the lock, and show up much later as a loop that nothing can report.
#define LOCK_REQUIRED() \
    do { \
        if (smp && !me->lock_depth) \
            stop(STOP_MONITOR_FAULT, 0, 0x10cc, 0, c->elr); \
    } while (0)
static inline void mon_unlock_all(void) {
    if (smp && me->lock_depth) {
        me->lock_depth = 0;
        u32 next_seq = __atomic_load_n(&mon_lock_word, __ATOMIC_RELAXED) + 1;
        __atomic_store_n(&mon_lock_word, next_seq, __ATOMIC_RELEASE);
        mon_flush_stats();
    }
}
// Bumped when blocks die, so that every other vCPU drops its return stacks at its next way out of the monitor, which any x86 serialising event or interrupt passes through.
static u32 rs_epoch_shared __attribute__((aligned(64)));

// ---- Host calls ----
_Noreturn static void stop(u64 reason, u64 rip, u64 a, u64 b, u64 d);

static inline void hostcall(u64 code) {
    register u64 x0 __asm__("x0") = code;
    u64 h0 = c->profile_el1_ticks ? RSR(cntvct_el0) : 0;
    __asm__ volatile("hvc #1" : : "r"(x0) : "memory");
    if (c->profile_el1_ticks) {
        u64 dt = RSR(cntvct_el0) - h0;
        me->hvc_ticks_local += dt;
        c->stat_el1_hvc_ticks += dt;
    }
}

// ---- Layout ----
#define PRIV(off) ((void *)(PRIV_VA + (off)))
#define RAM(gpa) ((void *)(RAM_ALIAS_VA + (gpa)))
#define IPA_TO_VA(ipa) (PRIV_VA + ((ipa) - PRIV_IPA))
#define VA_TO_IPA(va) (PRIV_IPA + ((va) - PRIV_VA))
#define SPSR_NZCV_C (1ULL << 29)
#define SPSR_SS (1ULL << 21)

#define PTE_BLOCK 1ULL
#define PTE_VALID 3ULL
#define PTE_AF (1ULL << 10)
#define PTE_SH (3ULL << 8)
#define PTE_NG (1ULL << 11)
#define PTE_CONT (1ULL << 52)
#define PTE_UXN (1ULL << 54)
#define PTE_PXN (1ULL << 53)
#define AP_RW (1ULL << 6) // EL0 and EL1 read/write
#define AP_RO (3ULL << 6) // EL0 and EL1 read only
#define PTE_X_MARK (1ULL << 55) // software: code was translated from this page in this view. Valid in invalid entries too.
#define PTE_ADDR 0x0000FFFFFFFFF000ULL
static inline bool is_table_desc(u64 e) { return (e & 3ULL) == PTE_VALID; }
static inline bool is_block_desc(u64 e) { return (e & 3ULL) == PTE_BLOCK; }

#define X_P 1ULL
#define X_RW 2ULL
#define X_US 4ULL
#define X_A 0x20ULL
#define X_D 0x40ULL
#define X_PS 0x80ULL
#define X_NX (1ULL << 63)
#define X_ADDR 0x000FFFFFFFFFF000ULL

#define RFLAGS_FIXED 2ULL
#define RF_CF 1ULL
#define RF_PF 4ULL
#define RF_AF 0x10ULL
#define RF_ZF 0x40ULL
#define RF_SF 0x80ULL
#define RF_TF 0x100ULL
#define RF_IF 0x200ULL
#define RF_DF 0x400ULL
#define RF_OF 0x800ULL
#define RF_IOPL 0x3000ULL
#define RF_NT 0x4000ULL
#define RF_RF 0x10000ULL
#define RF_VM 0x20000ULL
#define RF_AC 0x40000ULL
#define RF_ARITH (RF_CF | RF_PF | RF_AF | RF_ZF | RF_SF | RF_OF)

// ---- Views: an x86 address space seen from one privilege level ----
typedef struct {
    u64 cr3; // masked: table address
    u32 cpl, valid; // cpl: 4 added in compatibility mode, whose code at any rip decodes differently
    u32 sees_shared, pad_shared; // sees_shared: its lookup table may hold blocks of a shared domain
    u64 root0, root1; // VA of shadow roots
    u64 ttbr0, ttbr1;
    u64 hash_va;
    u64 last_used;
    u32 keeper, pad;
} View;
static View views[MAX_VIEWS] __attribute__((aligned(64)));
static u32 view_count __attribute__((aligned(64)));
static u8 view_pin_cnt[MAX_VIEWS] __attribute__((aligned(64)));
// Each view has a return stack per vCPU: two vCPUs can run one address space at once.
static inline u64 view_rs_top(u32 idx) { return PRIV_VA + engine_off_rstacks(cpu_index()) + (u64)(idx + 1) * RSTACK_SLOT - 16; }

// ---- Shadow page pool ----
typedef struct {
    u64 guest_table; // GPA of the guest table this mirrors; ~0 for a fracture of a large page
    u64 va_base;
    u32 view;
    u32 level; // 3 root .. 0 leaf table
    u32 next_mirror; // pool index + 1: next shadow table mirroring the same guest frame
    u32 writes; // guest writes since the last walk through this table
    u64 *parent; // the shadow entry pointing here; null for roots
} ShadowDesc;

#define POOL_PAGES (SHADOW_POOL_BYTES / 4096)
#define POOL_VA (PRIV_VA + OFF_PT + PT_STATIC_BYTES)
_Static_assert(OFF_PT + PT_STATIC_BYTES + SHADOW_POOL_BYTES <= OFF_SHADOW_META, "a full pool would write over its own descriptors");
static ShadowDesc *const desc = (ShadowDesc *)PRIV(OFF_SHADOW_META);
#define MIRROR_HEAD_OFF (OFF_SHADOW_META + (16ULL << 20)) // u32 per guest frame
#define RMAP_HEAD_OFF (OFF_SHADOW_META + (48ULL << 20)) // u32 per guest frame
#define RMAP_NODES_OFF (OFF_SHADOW_META + (80ULL << 20))
#define RMAP_NODES ((64ULL << 20) / 8)
#define RMAP2M_HEAD_OFF (OFF_SHADOW_META + (144ULL << 20)) // u32 per 2 MiB chunk (32 GiB / 2 MiB = 16384 = 64 KiB)
#define RMAP2M_NODES_OFF (OFF_SHADOW_META + (145ULL << 20))
#define RMAP2M_NODES ((4ULL << 20) / 8)
_Static_assert(RMAP2M_NODES_OFF + RMAP2M_NODES * 8ULL <= OFF_PAGEINFO, "rmap2m overflows OFF_SHADOW_META");
typedef struct {
    u32 pte; // pool byte offset / 8
    u32 next;
} RmapNode;
static u32 *const mirror_head = (u32 *)PRIV(MIRROR_HEAD_OFF);
static u32 *const rmap_head = (u32 *)PRIV(RMAP_HEAD_OFF);
static RmapNode *const rmap_nodes = (RmapNode *)PRIV(RMAP_NODES_OFF);
static u32 *const rmap2m_head = (u32 *)PRIV(RMAP2M_HEAD_OFF);
static RmapNode *const rmap2m_nodes = (RmapNode *)PRIV(RMAP2M_NODES_OFF);
static u8 *const page_info = (u8 *)PRIV(OFF_PAGEINFO);
static u32 pool_next __attribute__((aligned(64))), pool_free_head, pool_free_count, rmap_next = 1, rmap_free, rmap_free_count;
static u32 rmap2m_next = 1, rmap2m_free, rmap2m_free_count;
static u64 hole_entry_priv, hole_entry_ram; // root-level descriptors for the private slots

static inline u64 *pool_page(u32 idx) { return (u64 *)(POOL_VA + (u64)idx * 4096); }
static inline u32 pool_index(const void *p) { return (u32)(((u64)p - POOL_VA) >> 12); }
// ram_size is the top of RAM: below it there can be holes, which belong to devices.
// Direct-mapped PCI BARs (e.g. VRAM/FIFO) may sit above ram_size on <=3 GB VMs or 64-bit BARs, up to PRIV_IPA (60 GB < 64 GB page_info).
static inline bool is_ram(u64 gpa) { return gpa < c->ram_size && (page_info[gpa >> 12] & PI_RAM); }
static inline bool is_direct(u64 gpa) { return gpa < PRIV_IPA && (page_info[gpa >> 12] & PI_DIRECT); }


static u32 pool_alloc(u32 view, u32 level, u64 guest_table, u64 va_base) {
    LOCK_REQUIRED();
    u32 idx;
    if (pool_free_head) {
        idx = pool_free_head - 1;
        pool_free_head = *(u32 *)pool_page(idx);
        pool_free_count--;
    } else if (pool_next < (c->pool_limit ? c->pool_limit : POOL_PAGES)) {
        idx = pool_next++;
    } else {
        stop(STOP_SHADOW_FULL, 0, 1, pool_next, 0); // pool_reclaim runs first on every path that allocates
    }
    memset(pool_page(idx), 0, 4096);
    ShadowDesc *d = &desc[idx];
    d->guest_table = guest_table;
    d->va_base = va_base;
    d->view = view;
    d->level = level;
    d->next_mirror = 0;
    d->writes = 0;
    d->parent = 0;
    if (guest_table != ~0ULL && guest_table < c->ram_size) {
        u32 fr = (u32)(guest_table >> 12);
        d->next_mirror = mirror_head[fr];
        mirror_head[fr] = idx + 1;
    }
    return idx;
}

// ---- TLB ----
static inline void tlbi_va(u32 asid, u64 va) {
    u64 op = (u64)asid << 48 | ((va >> 12) & 0xFFFFFFFFFFFULL);
    __asm__ volatile("dsb ishst\n tlbi vae1is, %0\n dsb ish\n isb" : : "r"(op));
}
static inline void tlbi_va_64k(u32 asid, u64 base_va) {
    u64 base = (u64)asid << 48 | ((base_va >> 12) & 0xFFFFFFFFFFFULL);
    __asm__ volatile("dsb ishst" : : : "memory");
    for (u64 k = 0; k < 16; k++) {
        u64 op = base + k;
        __asm__ volatile("tlbi vae1is, %0" : : "r"(op));
    }
    __asm__ volatile("dsb ish\n isb" : : : "memory");
}
static inline void tlbi_all(void) { __asm__ volatile("dsb ishst\n tlbi vmalle1is\n dsb ish\n isb"); }
static inline u32 view_asid(u32 v) { return v + 1; }

// ---- x86 page walk ----
typedef struct {
    bool ok;
    u32 fault_err; // x86 page-fault error code when !ok
    u64 gpa; // of the addressed byte
    u64 *leaf; // alias pointer to the leaf entry
    u64 *chain[4]; // alias pointers to the entries used, chain[3] is the PML4E
    u64 read[4]; // what each held
    int leaf_level;
    bool user, write, nx; // effective permissions
    bool dirty;
} Walk;

static inline bool canonical(u64 va) { return (u64)((i64)(va << 16) >> 16) == va; }

// Permission logic mirrors the SDM for CR0.WP=1 (enforced at CR0 writes) without SMAP (not advertised, stopped at CR4 writes).
// access: bit 1 write, bit 2 user, bit 4 fetch.
static void track_root(View *v, u64 va, const Walk *w);
// walk_any_tables: the tables may be a dead process's, freed and poisoned: a slot pointing outside RAM just ends the walk

static Walk x86_walk(u64 cr3, u64 va, u32 access, bool set_ad) {
    Walk w = {0};
    w.user = true;
    w.write = true;
    w.fault_err = access & (2 | 4 | 16);
    if (!canonical(va)) {
        w.fault_err = ~0u; // #GP, not #PF
        return w;
    }
    u64 table = cr3 & X_ADDR;
    bool nxe = c->efer & (1ULL << 11);
    for (int level = 3; level >= 0; level--) {
        if (!is_ram(table)) {
            if (walk_any_tables)
                return w;
            stop(STOP_UNSUPPORTED, 0, 1, table, va); // x86 page table outside RAM
        }
        u64 *ep = (u64 *)RAM(table + ((va >> (12 + 9 * level)) & 0x1ff) * 8);
        u64 e = *ep;
        w.chain[level] = ep;
        w.read[level] = e;
        if (!(e & X_P))
            return w;
        w.user = w.user && (e & X_US);
        w.write = w.write && (e & X_RW);
        if (nxe && (e & X_NX))
            w.nx = true;
        if (level == 0 || (level < 3 && (e & X_PS))) {
            u64 size = 1ULL << (12 + 9 * level);
            u64 base = level ? (e & X_ADDR & ~(size - 1) & ~(1ULL << 12)) : (e & X_ADDR);
            w.gpa = base | (va & (size - 1));
            w.leaf = ep;
            w.leaf_level = level;
            w.dirty = e & X_D;
            w.fault_err |= 1; // present from here on
            if (!(access & 4) && !(c->cr0 & (1ULL << 16)))
                w.write = true; // Ring 0 ignores R/W when CR0.WP=0 (UEFI/winload)
            if ((access & 4) && !w.user)
                return w;
            if ((access & 2) && !w.write)
                return w;
            if ((access & 16) && w.nx)
                return w;
            if ((access & 16) && !(access & 4) && w.user && (c->cr4 & (1ULL << 20)))
                return w; // SMEP
            if (set_ad) {
                for (int l = 3; l >= level; l--) {
                    u64 want = w.read[l] | X_A | (l == level && (access & 2) ? X_D : 0);
                    if (want != w.read[l] && !__atomic_compare_exchange_n(w.chain[l], &w.read[l], want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        return x86_walk(cr3, va, access, set_ad);
                }
                w.dirty = w.dirty || (access & 2);
            }
            w.ok = true;
            return w;
        }
        table = e & X_ADDR;
    }
    return w;
}

// Guest memory through the walker, for frames and descriptor tables. Returns false on a fault.
static bool guest_read(u64 va, void *out, u32 n, u32 access) {
    u8 *o = out;
    while (n) {
        Walk w = x86_walk(c->cr3, va, access, true);
        if (!w.ok || !is_ram(w.gpa))
            return false;
        u32 chunk = 4096 - (u32)(va & 0xfff);
        if (chunk > n)
            chunk = n;
        memcpy(o, RAM(w.gpa), chunk);
        o += chunk;
        va += chunk;
        n -= chunk;
    }
    return true;
}
static bool guest_write(u64 va, const void *in, u32 n, u32 access) {
    const u8 *o = in;
    while (n) {
        Walk w = x86_walk(c->cr3, va, access | 2, true);
        if (!w.ok || !is_ram(w.gpa))
            return false;
        u32 chunk = 4096 - (u32)(va & 0xfff);
        if (chunk > n)
            chunk = n;
        memcpy(RAM(w.gpa), o, chunk);
        o += chunk;
        va += chunk;
        n -= chunk;
    }
    return true;
}

// ---- rmap: who maps this frame writable ----
static void rmap_add(u64 gpa, u64 *pte) {
    LOCK_REQUIRED();
    u32 node;
    if (rmap_free) {
        node = rmap_free;
        rmap_free = rmap_nodes[node].next;
        rmap_free_count--;
    } else {
        if (rmap_next >= RMAP_NODES)
            stop(STOP_SHADOW_FULL, 0, 2, 0, 0);
        node = rmap_next++;
    }
    u32 fr = (u32)(gpa >> 12);
    rmap_nodes[node].pte = (u32)(((u64)pte - POOL_VA) >> 3);
    rmap_nodes[node].next = rmap_head[fr];
    rmap_head[fr] = node;
}
static void rmap_remove(u64 gpa, u64 *pte) {
    u32 fr = (u32)(gpa >> 12), want = (u32)(((u64)pte - POOL_VA) >> 3);
    u32 *link = &rmap_head[fr];
    while (*link) {
        RmapNode *n = &rmap_nodes[*link];
        if (n->pte == want) {
            u32 dead = *link;
            *link = n->next;
            n->next = rmap_free;
            rmap_free = dead;
            rmap_free_count++;
            return;
        }
        link = &n->next;
    }
}
static void rmap2m_add(u64 base_gpa, u64 *pte) {
    LOCK_REQUIRED();
    u32 node;
    if (rmap2m_free) {
        node = rmap2m_free;
        rmap2m_free = rmap2m_nodes[node].next;
        rmap2m_free_count--;
    } else {
        if (rmap2m_next >= RMAP2M_NODES)
            stop(STOP_SHADOW_FULL, 0, 2, 0, 0);
        node = rmap2m_next++;
    }
    u32 fr2m = (u32)(base_gpa >> 21);
    rmap2m_nodes[node].pte = (u32)(((u64)pte - POOL_VA) >> 3);
    rmap2m_nodes[node].next = rmap2m_head[fr2m];
    rmap2m_head[fr2m] = node;
}
static void rmap2m_remove(u64 base_gpa, u64 *pte) {
    u32 fr2m = (u32)(base_gpa >> 21), want = (u32)(((u64)pte - POOL_VA) >> 3);
    u32 *link = &rmap2m_head[fr2m];
    while (*link) {
        RmapNode *n = &rmap2m_nodes[*link];
        if (n->pte == want) {
            u32 dead = *link;
            *link = n->next;
            n->next = rmap2m_free;
            rmap2m_free = dead;
            rmap2m_free_count++;
            return;
        }
        link = &n->next;
    }
}

static inline u64 *rmap_pte(u32 v) { return (u64 *)(POOL_VA + (u64)v * 8); }
static u64 pte_va(const u64 *pte, const ShadowDesc *d) {
    u64 idx = ((u64)pte & 0xfff) >> 3;
    u64 va = d->va_base + (idx << (12 + 9 * d->level));
    return va;
}

// Break-Before-Make teardown of a 16-entry (64 KiB) PTE_CONT group in a level-0 shadow page.
static void break_cont_group(u64 *pte, const ShadowDesc *d, bool remove_self_rmap) {
    u64 *grp = (u64 *)((u64)pte & ~127ULL);
    u64 base_va = pte_va(grp, d);
    for (u32 k = 0; k < 16; k++) {
        u64 ge = grp[k];
        if (ge & 1) {
            if ((&grp[k] != pte || remove_self_rmap) && (ge & AP_RO) == AP_RW && (ge & PTE_ADDR) < c->ram_size)
                rmap_remove(ge & PTE_ADDR, &grp[k]);
            grp[k] = ge & PTE_X_MARK;
        }
    }
    tlbi_va_64k(view_asid(d->view), base_va);
    c->stat_hp_64k_breaks++;
}

// Clears writability of every shadow mapping of this frame (4 KiB, 64 KiB PTE_CONT, and 2 MiB L2 Block).
static void write_protect_frame(u64 gpa) {
    u32 fr = (u32)(gpa >> 12);
    u32 it = rmap_head[fr];
    rmap_head[fr] = 0;
    while (it) {
        RmapNode *n = &rmap_nodes[it];
        u64 *pte = rmap_pte(n->pte);
        u32 next = n->next;
        n->next = rmap_free;
        rmap_free = it;
        rmap_free_count++;
        ShadowDesc *d = &desc[pool_index(pte)];
        u64 e = *pte;
        if ((e & 1) && (e & PTE_CONT)) {
            break_cont_group(pte, d, false);
        } else if ((e & 1) && (e & AP_RO) == AP_RW) {
            *pte = (e & ~AP_RW) | AP_RO;
            tlbi_va(view_asid(d->view), pte_va(pte, d));
        }
        it = next;
    }
    // Zap-on-Split any writable 2 MiB L2 Block covering this frame.
    u32 fr2m = (u32)(gpa >> 21);
    u32 it2 = rmap2m_head[fr2m];
    rmap2m_head[fr2m] = 0;
    while (it2) {
        RmapNode *n = &rmap2m_nodes[it2];
        u64 *pte = rmap_pte(n->pte);
        u32 next = n->next;
        n->next = rmap2m_free;
        rmap2m_free = it2;
        rmap2m_free_count++;
        ShadowDesc *d = &desc[pool_index(pte)];
        u64 e = *pte;
        if (is_block_desc(e)) {
            *pte = 0;
            tlbi_va(view_asid(d->view), pte_va(pte, d));
            c->stat_hp_2m_splits++;
        }
        it2 = next;
    }
}

// ---- Boundary lookup: ARM pc to x86 rip ----
static BlockEntry *find_block(u64 pc) {
    u64 cc = PRIV_VA + OFF_CC;
    if ((pc < cc || pc >= cc + CC_SIZE) && (pc < PRIV_VA + OFF_CC2 || pc >= PRIV_VA + OFF_CC2 + CC_SIZE))
        return 0;
    u32 word = (u32)((pc - cc) >> 2);
    // Tier-2 mid-cache islands (part 0: [CC_T2_WORD0..+CC_T2_WORDS), part 1: [CC_T2_WORD1..+CC_T2_WORDS)).
    if ((word >= CC_T2_WORD0 && word < CC_T2_WORD0 + CC_T2_WORDS) ||
        (word >= CC_T2_WORD1 && word < CC_T2_WORD1 + CC_T2_WORDS)) {
        u32 part = word >= CC_PART2_WORD ? 1U : 0U;
        BlockEntry *t2 = (BlockEntry *)PRIV(OFF_BLOCKS_T2);
        u64 first = (u64)part * T2_BLOCKS_MAX;
        u64 lo = first, hi = first + __atomic_load_n(&c0->t2_block_count[part], __ATOMIC_ACQUIRE);
        while (lo < hi) {
            u64 mid = (lo + hi) / 2;
            if (t2[mid].arm_off <= word)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo > first) {
            BlockEntry *b = &t2[lo - 1];
            if (word < b->arm_off + b->arm_words)
                return b;
        }
    }
    BlockEntry *t = (BlockEntry *)PRIV(OFF_BLOCKS);
    bool private_part = word >= CC_PART2_WORD; // two tables, each sorted
    u64 first, count;
    if (private_part) {
        first = (OFF_BLOCKS2 - OFF_BLOCKS) / sizeof(BlockEntry);
        count = __atomic_load_n(&c0->block_count_private, __ATOMIC_ACQUIRE);
    } else if (c0->p0_ring_active) {
        int slot = engine_p0_slot_of_word(word);
        if (slot < 0)
            return 0;
        first = (u64)slot * P0_BLOCKS_PER_SLOT;
        count = __atomic_load_n(&c0->p0_slot_block_count[slot], __ATOMIC_ACQUIRE);
    } else {
        first = 0;
        count = __atomic_load_n(&c0->block_count, __ATOMIC_ACQUIRE);
    }
    u64 lo = first, hi = first + count;
    while (lo < hi) {
        u64 mid = (lo + hi) / 2;
        if (t[mid].arm_off <= word)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == first)
        return 0;
    BlockEntry *b = &t[lo - 1];
    return word < b->arm_off + b->arm_words ? b : 0;
}

// x86 rip of the instruction containing pc. exact is set when pc is its first word.
static inline u64 stubs_of(u64 pc) {
    if (pc >= PRIV_VA + OFF_STUBS2_HI)
        return PRIV_VA + OFF_STUBS2_HI;
    if (pc >= PRIV_VA + OFF_STUBS2)
        return PRIV_VA + OFF_STUBS2;
    if (pc >= PRIV_VA + OFF_STUBS_HI)
        return PRIV_VA + OFF_STUBS_HI;
    return PRIV_VA + OFF_STUBS;
} // of the copy in pc's part/half
static inline bool in_rep_stub(u64 pc) {
    u64 s = stubs_of(pc);
    return pc >= s + STUB_OFF_REP_MOVS && pc < s + STUB_OFF_REP_PAGE_END;
}
static bool rep_page_site(u64 pc, bool *movs) {
    u64 s = stubs_of(pc);
    if (pc == s + STUB_OFF_REP_PAGE_MOVS_HEAD) {
        *movs = true;
        return true;
    }
    if (pc == s + STUB_OFF_REP_PAGE_STOS_HEAD) {
        *movs = false;
        return true;
    }
    return false;
}

static u64 rip_of_pc(u64 pc, bool *exact, bool *found) {
    if (in_rep_stub(pc)) { // inside a rep instruction, called from its block; the loop heads are boundaries
        bool e;
        u64 s = stubs_of(pc), rip = rip_of_pc(c->x[R_LR] - 4, &e, found);
        *exact = pc == s + STUB_OFF_REP_MOVS_HEAD || pc == s + STUB_OFF_REP_STOS_HEAD ||
                 pc == s + STUB_OFF_REP_PAGE_MOVS_HEAD || pc == s + STUB_OFF_REP_PAGE_STOS_HEAD;
        return rip;
    }
    BlockEntry *b = find_block(pc);
    *exact = false;
    *found = b != 0;
    if (!b)
        return 0;
    bool is_t2 = (u64)b >= PRIV_VA + OFF_BLOCKS_T2 && (u64)b < PRIV_VA + OFF_PCMAP_T2;
    PcMap *m = (PcMap *)PRIV(is_t2 ? OFF_PCMAP_T2 : OFF_PCMAP) + b->map_off;
    u32 word = (u32)((pc - (PRIV_VA + OFF_CC)) >> 2) - b->arm_off;
    u32 best = 0;
    for (u32 i = 0; i < b->map_count; i++) {
        if (m[i].arm_word <= word)
            best = i;
        else
            break;
    }
    *exact = m[best].arm_word == word;
    return b->rip + m[best].rip_off;
}

static u64 rip_here(u64 pc) {
    bool e, f;
    u64 rip = rip_of_pc(pc, &e, &f);
    if (!f)
        stop(STOP_MONITOR_FAULT, 0, c->esr, c->far, pc);
    return rip;
}

// Reconstruct architectural ymm8..15 and dirty promoted stack slots when leaving a Tier-2 loop body on trap/fault/IRQ/SMC/stop.
// Zero cost on Tier-1 code and retryable shadow-PT fills (returns immediately unless c->elr is inside a T2 loop body).
static bool t2_sync_on_trap(void) {
    if (me->t2_synced)
        return false;
    u64 pc = c->elr;
    u64 cc = PRIV_VA + OFF_CC;
    if (pc < cc || pc >= PRIV_VA + OFF_CC2 + CC_SIZE)
        return false;
    u32 word = (u32)((pc - cc) >> 2);
    u32 part;
    if (word >= CC_T2_WORD0 && word < CC_T2_WORD0 + CC_T2_WORDS)
        part = 0U;
    else if (word >= CC_T2_WORD1 && word < CC_T2_WORD1 + CC_T2_WORDS)
        part = 1U;
    else
        return false;
    BlockEntry *t2 = (BlockEntry *)PRIV(OFF_BLOCKS_T2);
    u64 first = (u64)part * T2_BLOCKS_MAX;
    u64 lo = first, hi = first + __atomic_load_n(&c0->t2_block_count[part], __ATOMIC_ACQUIRE);
    while (lo < hi) {
        u64 mid = (lo + hi) / 2;
        if (t2[mid].arm_off <= word)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo <= first)
        return false;
    u64 idx = lo - 1;
    BlockEntry *b = &t2[idx];
    if (word >= b->arm_off + b->arm_words)
        return false;
    const T2TrapMeta *tm = &((const T2TrapMeta *)PRIV(OFF_META_T2))[idx];
    if (!(tm->flags & T2_META_FLAG_ACTIVE))
        return false;
    u32 rel = word - b->arm_off;
    if (rel < tm->body_start_word || rel >= tm->body_end_word)
        return false;
    me->t2_synced = true;
    if (tm->ymm_wr_mask) {
        u64 pool[4][2] __attribute__((aligned(16)));
        mon_t2_read_pool(pool);
        for (u32 s = 0; s < 4; s++) {
            if (!(tm->ymm_wr_mask & (1U << s)))
                continue;
            u32 reg = tm->ymm_reg[s];
            if (reg >= 8U && reg < 16U) {
                c->ymm_hi[reg][0] = pool[s][0];
                c->ymm_hi[reg][1] = pool[s][1];
            }
        }
    }
    if (tm->stk_cnt && tm->stk_dirty_mask && tm->stk_base < 16U) {
        u32 access = c->cpl == 3 ? 4U : 0U;
        u64 base_va = c->x[tm->stk_base];
        for (u32 k = 0; k < tm->stk_cnt && k < T2_MAX_STK_SLOTS; k++) {
            if (!(tm->stk_dirty_mask & (1U << k)))
                continue;
            u32 phys = tm->stk_phys[k];
            u32 sz = tm->stk_sz[k];
            if (phys < 31U && (sz == 4U || sz == 8U)) {
                u64 val = c->x[phys];
                guest_write(base_va + (u64)tm->stk_disp[k], &val, sz, access);
            }
        }
    }
    return true;
}

// ---- Stops ----
// A stop this instruction can come back from, in hybrid mode. The state is precise at every rip passed to stop: nothing guest-visible changes
// before an instruction's last access. Stops with no rip are about the engine's own limits, and stay final.
static void x87_flush_fast_cache(void);
static void irq_deliver_now(u64 rip, u32 apic_vector);
static void fex_unspill_state(void);
_Noreturn static void stop(u64 reason, u64 rip, u64 a, u64 b, u64 d) {
    t2_sync_on_trap();
    fex_unspill_state();
    if (reason == STOP_MONITOR_FAULT)
        c->req_a[5] = engine_capture_code_fault(d, PRIV_VA + OFF_CC, PRIV_VA + OFF_CC2, CC_SIZE);
    mon_unlock_all();
    bool resumable = c->hybrid && rip && (reason == STOP_UNTRANSLATED || reason == STOP_UNSUPPORTED || reason == STOP_FALLBACK);
    // A vector already taken from QEMU's APIC goes first. The instruction restarts on return from the handler, and the host never sits on
    // an interrupt it cannot see through a long stay on TCG.
    if (resumable && c->irq_pending && (c->rflags_sys & RF_IF) && rip != c->irq_shadow_rip) {
        irq_deliver_now(rip, 0);
        mon_restart();
    }
    c->req_a[0] = reason;
    c->req_a[1] = rip;
    c->req_a[2] = a;
    c->req_a[3] = b;
    c->req_a[4] = d;
    if (resumable) {
        c->req_r[0] = rip;
        x87_flush_fast_cache();
        hostcall(HC_FALLBACK);
        c->fp_d_valid = 0;
        mon_resync = true;
        mon_restart();
    }
    for (;;)
        hostcall(HC_STOP);
}

void mon_bug(void) {
    stop(STOP_MONITOR_FAULT, 0, RSR(esr_el1), RSR(far_el1), RSR(elr_el1));
}

// ---- RFLAGS ----
static inline u32 parity_even(u64 v) { return !(__builtin_popcountll(v & 0xff) & 1); }

static u64 rflags_get(void) {
    u64 nz = c->spsr, r = c->rflags_sys;
    if (!(nz & SPSR_NZCV_C))
        r |= RF_CF;
    if (nz & (1ULL << 30))
        r |= RF_ZF;
    if (nz & (1ULL << 31))
        r |= RF_SF;
    if (nz & (1ULL << 28))
        r |= RF_OF;
    if (parity_even(c->x[R_PF]))
        r |= RF_PF;
    if ((c->x[R_AF] ^ c->x[R_PF]) & 0x10)
        r |= RF_AF;
    if (c->df < 0)
        r |= RF_DF;
    return r;
}

static void rflags_set(u64 r) {
    u64 s = c->spsr & ~(0xFULL << 28);
    if (!(r & RF_CF))
        s |= SPSR_NZCV_C;
    if (r & RF_ZF)
        s |= 1ULL << 30;
    if (r & RF_SF)
        s |= 1ULL << 31;
    if (r & RF_OF)
        s |= 1ULL << 28;
    c->spsr = s;
    c->x[R_PF] = (r & RF_PF) ? 0 : 1;
    c->x[R_AF] = c->x[R_PF] ^ (r & RF_AF);
    c->df = (r & RF_DF) ? -1 : 1;
    c->rflags_sys = (r & ~(RF_ARITH | RF_DF)) | RFLAGS_FIXED;
}

// ---- Views ----
static u64 fex_view_gen[MAX_VIEWS] __attribute__((aligned(64)));
static u64 fex_gen_seq __attribute__((aligned(64)));

static void view_rs_prepare(u32 idx) {
    me->rs_saved[idx] = view_rs_top(idx);
    u64 *sent = (u64 *)view_rs_top(idx);
    sent[0] = ~0ULL; // never equals a canonical return address, so popping the sentinel misses
    sent[1] = 0;
}

static void view_build(u32 idx, u64 key, u32 cpl) {
    View *v = &views[idx];
    __atomic_store_n(&fex_view_gen[idx], __atomic_add_fetch(&fex_gen_seq, 1, __ATOMIC_ACQ_REL), __ATOMIC_RELEASE);
    v->cr3 = key;
    v->cpl = cpl;
    u32 r0 = pool_alloc(idx, 3, key & X_ADDR, 0);
    u32 r1 = pool_alloc(idx, 3, key & X_ADDR, 0xFFFF000000000000ULL); // upper half of the same guest table
    v->root0 = (u64)pool_page(r0);
    v->root1 = (u64)pool_page(r1);
    ((u64 *)v->root0)[(PRIV_VA >> 39) & 0x1ff] = hole_entry_priv;
    ((u64 *)v->root0)[(RAM_ALIAS_VA >> 39) & 0x1ff] = hole_entry_ram;
    v->ttbr0 = VA_TO_IPA(v->root0) | (u64)view_asid(idx) << 48;
    v->ttbr1 = VA_TO_IPA(v->root1);
    v->hash_va = PRIV_VA + OFF_HASH + (u64)idx * HASH_BYTES;
    view_rs_prepare(idx);
    // The monitor is about to fetch through this root. Unlike a guest access, a walk that sees the private slot before these stores reach memory can't refault
    // and retry: it is a fatal instruction abort at EL1.
    __asm__ volatile("dsb ishst");
    v->valid = 1;
}


static void zap_table(u32 idx);
static void reset_all_return_stacks(void);
static inline u32 view_asid(u32 v);

// Takes the least recently used view for reuse, or returns MAX_VIEWS if none may go. A view that keeps a shared kernel domain's marks stays.
static inline bool view_pinned(u32 idx) {
    return view_pin_cnt[idx] != 0;
}

static void view_drop(u32 best);

static u32 view_evict(void) {
    u32 best = MAX_VIEWS;
    for (u32 i = 0; i < view_count; i++)
        if (views[i].valid && !views[i].keeper && i != c->view && !view_pinned(i) && (best == MAX_VIEWS || views[i].last_used < views[best].last_used))
            best = i;
    if (best == MAX_VIEWS)
        return best;
    view_drop(best);
    return best;
}

static void view_drop(u32 best) {
    View *v = &views[best];
    c->stat_view_evictions++;
    // Code marks need no invalidation one by one: the host drops all the view's own blocks together.
    evicting = true;
    zap_table(pool_index((void *)v->root0));
    zap_table(pool_index((void *)v->root1));
    evicting = false;
    __asm__ volatile("dsb ishst\n tlbi aside1is, %0\n dsb ish\n isb" : : "r"((u64)view_asid(best) << 48));
    c->req_a[0] = best;
    hostcall(HC_VIEW_DROP);
    __atomic_store_n(&fex_view_gen[best], __atomic_add_fetch(&fex_gen_seq, 1, __ATOMIC_ACQ_REL), __ATOMIC_RELEASE);
    v->valid = 0;
    if (smp)
        reset_all_return_stacks();
}

// zap_why, for the tally: what the next zaps are for
static void zap_entry(u64 *e, const ShadowDesc *d);

// Every path that allocates shadow tables comes through here first, while it holds no pointer into them. Old views go first; if the running
// view and the keepers fill the pool alone, their shadows are dropped and refill on demand, and their translations go with their marks.
#define POOL_MARGIN 16
static void shadow_drop_all(void) {
    for (u32 i = 0; i < view_count; i++) {
        if (!views[i].valid)
            continue;
        ShadowDesc *d0 = &desc[pool_index((void *)views[i].root0)], *d1 = &desc[pool_index((void *)views[i].root1)];
        for (int n = 0; n < 256; n++) {
            u64 *r0 = (u64 *)views[i].root0 + n, *r1 = (u64 *)views[i].root1 + 256 + n;
            if (*r0)
                zap_entry(r0, d0);
            if (*r1)
                zap_entry(r1, d1);
        }
    }
    tlbi_all();
    reset_all_return_stacks();
}

static void pool_reclaim(void) {
    u32 limit = c->pool_limit ? c->pool_limit : POOL_PAGES;
    while (pool_free_count + (limit > pool_next ? limit - pool_next : 0) < POOL_MARGIN) {
        c->stat_pool_reclaim++;
        zap_why = KILL_RECLAIM;
        if (view_evict() != MAX_VIEWS)
            continue;
        shadow_drop_all();
        if (pool_free_count + (limit > pool_next ? limit - pool_next : 0) < POOL_MARGIN)
            stop(STOP_SHADOW_FULL, 0, 2, pool_free_count, 0);
    }
}

static View *view_find_locked(u64 cr3_key, u32 cpl) {
    static u16 hints[1024]; // the last view found under this hash, so that a guest crossing tables at every syscall doesn't scan them all under the lock
    u16 *hint = &hints[(cr3_key >> 12 ^ cr3_key >> 22 ^ cpl * 37) & 1023];
    View *hinted = &views[*hint];
    if (hinted->cr3 == cr3_key && hinted->cpl == cpl && hinted->valid)
        return hinted;
    for (u32 i = 0; i < view_count; i++)
        if (views[i].cr3 == cr3_key && views[i].cpl == cpl && views[i].valid) {
            *hint = (u16)i;
            return &views[i];
        }
    pool_reclaim();
    u32 idx = MAX_VIEWS;
    for (u32 i = 0; i < view_count; i++)
        if (!views[i].valid)
            idx = i;
    if (idx == MAX_VIEWS) {
        if (view_count < MAX_VIEWS)
            idx = view_count++;
        else if ((idx = view_evict()) == MAX_VIEWS)
            stop(STOP_UNSUPPORTED, 0, 2, 0, 0); // every view is a keeper or running
    }
    view_build(idx, cr3_key, cpl);
    return &views[idx];
}

// A pinned view can't go away, so a CPL change in the same address space touches nothing shared: syscalls and iret stay off the lock.
static View *view_find(u64 cr3_key, u32 cpl) {
    u32 *pin = &me->pin[(cpl & 3) == 3]; // cpl here has 4 added in compatibility mode
    if (*pin) {
        View *p = &views[*pin - 1];
        if (p->cr3 == cr3_key && p->cpl == cpl)
            return p;
    }
    bool kernel = (cpl & 3) != 3;
    if (kernel && me->pin[2]) {
        u32 before = me->pin[2];
        View *p = &views[before - 1];
        if (p->cr3 == cr3_key && p->cpl == cpl) {
            me->pin[2] = *pin;
            *pin = before;
            return p;
        }
    }
    mon_lock();
    me->lock_site = 1;
    u32 *leaving = kernel ? &me->pin[2] : pin; // out of this CPL's old address space. The other CPL's pin moves when that CPL next comes through
    if (*leaving) {
        static u64 unpin_stamp;
        views[*leaving - 1].last_used = ++unpin_stamp;
        view_pin_cnt[*leaving - 1]--;
    }
    if (kernel)
        me->pin[2] = *pin;
    View *v = view_find_locked(cr3_key, cpl);
    *pin = (u32)(v - views) + 1;
    view_pin_cnt[v - views]++;
    mon_unlock();
    return v;
}

// The view that keeps code marks for a shared kernel domain: whichever view first translated from it. It is not evicted, so a write to a
// kernel PTE always meets a mark, however many views have come and gone. Shortcut: a fixed table, more slots than kernels use.
static View *domain_keeper(u64 pml4e_target, View *requester) {
    static struct {
        u64 target;
        u32 view;
    } keepers[128];
    static u32 count;
    for (u32 i = 0; i < count; i++)
        if (keepers[i].target == pml4e_target)
            return &views[keepers[i].view];
    if (count == 128)
        stop(STOP_UNSUPPORTED, 0, 16, pml4e_target, 0);
    keepers[count].target = pml4e_target;
    keepers[count].view = (u32)(requester - views);
    count++;
    requester->keeper = 1;
    return requester;
}

static inline u32 view_mode(void) { return c->cpl | (c->compat ? 4 : 0); }
// The table alone: shadows are always in step, so a PCID separates nothing here, and XNU gives one address space a PCID per CPU, which would be a view per CPU.
static inline u64 cr3_key(u64 cr3) { return cr3 & X_ADDR; }

static View *view_current(void) { return &views[c->view]; }

static inline void fex_l1_sync_view(u32 view_idx, u32 cpl) {
    if ((cpl & 3) != 3)
        return;
    u64 vg = __atomic_load_n(&fex_view_gen[view_idx], __ATOMIC_ACQUIRE);
    if (vg != me->fex_l1_gen) {
        u64 *l1 = (u64 *)(PRIV_VA + OFF_FEX_L1 + (u64)cpu_index() * FEX_L1_BYTES);
        for (u32 i = 0; i < FEX_L1_ENTRIES; i++)
            l1[i * 2 + 1] = 0;
        me->fex_l1_gen = vg;
    }
}

static void view_activate_core(View *v) {
    View *old = view_current();
    if (old == v)
        return;
    if (((u64 *)v->root0)[(PRIV_VA >> 39) & 0x1ff] != hole_entry_priv) // caught here, a stomped root is a stop with a name and not an abort at EL1
        stop(STOP_UNSUPPORTED, 0, 17, v - views, ((u64 *)v->root0)[(PRIV_VA >> 39) & 0x1ff]);
    bool in_fex = (c->x[R_FS] == (u64)c + 32768ULL);
    u64 *fex_saved = (u64 *)((u8 *)c + 36864);
    me->rs_saved[old - views] = in_fex ? fex_saved[1] : c->x[R_RS];
    if (!me->rs_saved[v - views]) // built by another vCPU
        view_rs_prepare((u32)(v - views));
    c->rs_top = view_rs_top((u32)(v - views));
    if (in_fex) {
        fex_saved[1] = me->rs_saved[v - views];
        fex_saved[0] = v->hash_va;
        c->x[25] = c->rs_top;
        *(u64 *)c->rs_top = ~0ULL;
    } else {
        c->x[R_RS] = me->rs_saved[v - views];
        c->x[R_HASH] = v->hash_va;
    }
    __atomic_store_n(&c->view, (u32)(v - views), __ATOMIC_SEQ_CST);
    WSR(ttbr0_el1, v->ttbr0);
    WSR(ttbr1_el1, v->ttbr1);
    __asm__ volatile("isb");
}

// A cached CPL change leaves both views pinned and needs no shared lock.
// Replacing/rotating a pin is different: another CPU can retire the old root
// after view_find drops its pin, while this CPU still fetches monitor code
// through it. Keep selection, pin publication and the hardware TTBR switch in
// one retirement-critical section. The recursive inner lock remains balanced.
static void view_switch(u64 key, u32 mode) {
    u32 *pin = &me->pin[(mode & 3) == 3];
    if (*pin) {
        View *v = &views[*pin - 1];
        if (v->cr3 == key && v->cpl == mode) {
            view_activate_core(v);
            fex_l1_sync_view((u32)(v - views), mode);
            return;
        }
    }
    // Both me->pin[0] and me->pin[2] remain pinned in view_pin_cnt[], so rotating between
    // two kernel CR3s (e.g. KVA/KeStackAttachProcess ping-pong) is lock-free and retirement-safe.
    if ((mode & 3) != 3 && me->pin[2]) {
        u32 before = me->pin[2];
        View *v = &views[before - 1];
        if (v->cr3 == key && v->cpl == mode) {
            me->pin[2] = *pin;
            *pin = before;
            view_activate_core(v);
            fex_l1_sync_view((u32)(v - views), mode);
            return;
        }
    }
    mon_lock();
    View *v = view_find(key, mode);
    u32 vi = (u32)(v - views);
    view_activate_core(v);
    // Do not publish retirement eligibility before the MMU uses the new root.
    __asm__ volatile("isb" : : : "memory");
    mon_unlock();
    fex_l1_sync_view(vi, mode);
}

static void reset_my_return_stacks(void) {
    for (u32 i = 0; i < view_count; i++)
        if (me->rs_saved[i]) {
            me->rs_saved[i] = view_rs_top(i);
            *(u64 *)view_rs_top(i) = ~0ULL;
        }
    if (c->x[R_FS] == (u64)c + 32768ULL) {
        *(u64 *)((u8 *)c + 36864 + 8) = c->rs_top;
        c->x[25] = c->rs_top;
    } else {
        c->x[R_RS] = c->rs_top;
    }
    *(u64 *)c->rs_top = ~0ULL;
}
static void reset_all_return_stacks(void) {
    reset_my_return_stacks();
    if (smp)
        me->rs_epoch = __atomic_add_fetch(&rs_epoch_shared, 1, __ATOMIC_RELAXED);
}

// ---- Shadow fill ----
static inline bool chunk_2m_clean_ram(u64 base_gpa) {
    if ((base_gpa & 0x1fffffULL) != 0 || base_gpa + (2ULL << 20) > c->ram_size)
        return false;
    const u64 *p = (const u64 *)&page_info[base_gpa >> 12];
    const u64 mask = 0x0F0F0F0F0F0F0F0FULL;
    const u64 want = 0x0404040404040404ULL; // PI_RAM only (no PI_CODE, PI_PT, PI_DIRECT)
    for (u32 k = 0; k < 64; k++) {
        if ((p[k] & mask) != want)
            return false;
    }
    return true;
}

static inline bool chunk_64k_clean_ram(u64 base_gpa) {
    if ((base_gpa & 0xffffULL) != 0 || base_gpa + 65536ULL > c->ram_size)
        return false;
    const u64 *p = (const u64 *)&page_info[base_gpa >> 12];
    const u64 mask = 0x0F0F0F0F0F0F0F0FULL;
    const u64 want = 0x0404040404040404ULL;
    return (p[0] & mask) == want && (p[1] & mask) == want;
}

static u64 *shadow_slot(View *v, u64 va, const Walk *w, int leaf_level_target, u64 **parent_out) {
    bool high = va >> 47;
    u64 table_va = high ? v->root1 : v->root0;
    u32 vi = (u32)(v - views);
    (void)parent_out;
    for (int level = 3; level > leaf_level_target; level--) {
        u64 *t = (u64 *)table_va;
        u64 *e = &t[(va >> (12 + 9 * level)) & 0x1ff];
        if (level == 1 && is_block_desc(*e)) {
            // Break-Before-Make Zap-on-Split of an existing 2 MiB L2 Block before allocating L0 child table.
            u64 old_blk = *e;
            if ((old_blk & AP_RO) == AP_RW && (old_blk & PTE_ADDR) < c->ram_size)
                rmap2m_remove(old_blk & PTE_ADDR, e);
            *e = 0;
            tlbi_va(view_asid(vi), va & ~0x1fffffULL);
            c->stat_hp_2m_splits++;
        }
        if (!(*e & 1)) {
            int child = level - 1;
            // The child mirrors the guest table this entry points at. Below a large page there is none: it is a fracture.
            u64 gt = level > w->leaf_level ? (*w->chain[level] & X_ADDR) : ~0ULL;
            u64 mask = ~((1ULL << (12 + 9 * (child + 1))) - 1);
            u64 base = va & mask;
            u32 idx = pool_alloc(vi, (u32)child, gt, base);
            desc[idx].parent = e;
            desc[idx].writes = 0;
            if (gt != ~0ULL && gt < c->ram_size && !(page_info[gt >> 12] & PI_PT)) {
                page_info[gt >> 12] |= PI_PT;
                write_protect_frame(gt);
                me->tracked_new = true;
            }
            *e = VA_TO_IPA((u64)pool_page(idx)) | PTE_VALID;
        }
        table_va = IPA_TO_VA(*e & PTE_ADDR);
        desc[pool_index((void *)table_va)].writes = 0;
    }
    return &((u64 *)table_va)[(va >> (12 + 9 * leaf_level_target)) & 0x1ff];
}

static void mark_x_slot(View *v, u64 va, const Walk *w) {
    u64 *slot = shadow_slot(v, va, w, 0, 0);
    if ((*slot & 1) && (*slot & PTE_CONT))
        break_cont_group(slot, &desc[pool_index(slot)], true);
    *slot |= PTE_X_MARK;
}

static bool try_install_cont_64k(const Walk *w, u64 va, u64 *slot, u64 gpa, bool want_write) {
    u64 *grp = (u64 *)((u64)slot & ~127ULL);
    for (u32 k = 0; k < 16; k++)
        if (grp[k])
            return false;
    u32 idx_in_grp = (u32)(slot - grp);
    u64 base64_gpa = gpa - ((u64)idx_in_grp << 12);
    if (!chunk_64k_clean_ram(base64_gpa))
        return false;
    // Cold 64 KiB group: all 16 shadow slots are empty -> direct install with zero BBM/TLBI overhead.
    bool writable;
    if (w->leaf_level >= 1) {
        if (w->write && !w->dirty)
            return false;
        writable = w->write && w->dirty;
    } else {
        if (c->hugepages_disabled & 4u)
            return false;
        u64 *guest_l0 = (u64 *)((u64)w->leaf & ~0xfffULL);
        u32 grp_idx = (u32)(((va >> 12) & 0x1ffu) & ~15u);
        bool upper_write = ((w->read[3] & w->read[2] & w->read[1] & X_RW) != 0) ||
                           (c->cpl != 3 && !(c->cr0 & (1ULL << 16)));
        bool all_rw = upper_write;
        bool all_ro = true;
        for (u32 k = 0; k < 16; k++) {
            u64 ge = __atomic_load_n(&guest_l0[grp_idx + k], __ATOMIC_RELAXED);
            if ((ge & (X_P | X_A)) != (X_P | X_A))
                return false;
            if (c->cpl == 3 && !(ge & X_US))
                return false;
            if ((ge & X_ADDR) != base64_gpa + ((u64)k << 12))
                return false;
            bool page_w = upper_write && (ge & X_RW);
            if (page_w) {
                all_ro = false;
                if (!(ge & X_D))
                    return false;
            } else {
                all_rw = false;
            }
        }
        if (!all_rw && !all_ro)
            return false;
        if (want_write && !all_rw)
            return false;
        writable = all_rw;
    }
    if (writable && (u64)rmap_free_count + (RMAP_NODES - rmap_next) < 64)
        return false;
    u64 attr = PTE_VALID | PTE_CONT | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (writable ? AP_RW : AP_RO);
    for (u32 k = 0; k < 16; k++) {
        u64 kgpa = base64_gpa + ((u64)k << 12);
        grp[k] = kgpa | attr;
        if (writable)
            rmap_add(kgpa, &grp[k]);
    }
    c->stat_hp_64k_fills++;
    return true;
}

static void track_root(View *v, u64 va, const Walk *w);

// Installs a shadow mapping for va: 2 MiB L2 Block (for clean x86 large pages), 64 KiB PTE_CONT (for contiguous 16-page runs), or 4 KiB leaf.
static u64 *shadow_install(View *v, u64 va, const Walk *w, bool want_write) {
    track_root(v, va, w);
    if (w->leaf_level >= 1 && !(c->hugepages_disabled & 1u)) {
        u64 base_gpa = w->gpa & ~0x1fffffULL;
        if (chunk_2m_clean_ram(base_gpa)) {
            u64 *l1_slot = shadow_slot(v, va, w, 1, 0);
            // Re-verify after shadow_slot in case a newly tracked guest page table frame lies inside base_gpa..+2M.
            if (chunk_2m_clean_ram(base_gpa)) {
                u64 old1 = *l1_slot;
                if (!(old1 & 1) || is_block_desc(old1)) {
                    bool writable = w->write && w->dirty;
                    bool have_rmap2m = !writable || (u64)rmap2m_free_count + (RMAP2M_NODES - rmap2m_next) >= 16;
                    if (have_rmap2m) {
                        if (is_block_desc(old1) && (old1 & PTE_ADDR) == base_gpa &&
                            (!want_write || (old1 & AP_RO) == AP_RW))
                            return l1_slot;
                        if (is_block_desc(old1) && (old1 & PTE_ADDR) != base_gpa) {
                            if ((old1 & AP_RO) == AP_RW && (old1 & PTE_ADDR) < c->ram_size)
                                rmap2m_remove(old1 & PTE_ADDR, l1_slot);
                            *l1_slot = 0;
                            tlbi_va(view_asid((u32)(v - views)), va & ~0x1fffffULL);
                            old1 = 0;
                        }
                        bool was_rw = is_block_desc(old1) && (old1 & AP_RO) == AP_RW && (old1 & PTE_ADDR) == base_gpa;
                        *l1_slot = base_gpa | PTE_BLOCK | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (writable ? AP_RW : AP_RO);
                        if (writable && !was_rw)
                            rmap2m_add(base_gpa, l1_slot);
                        else if (!writable && was_rw)
                            rmap2m_remove(base_gpa, l1_slot);
                        if (!(old1 & 1))
                            c->stat_hp_2m_fills++;
                        return l1_slot;
                    }
                }
            }
        }
    }
    u64 *slot = shadow_slot(v, va, w, 0, 0);
    u64 old = *slot;
    u64 gpa = w->gpa & ~0xfffULL;
    if ((old & 1) && (old & PTE_CONT)) {
        if ((old & PTE_ADDR) == gpa && (!want_write || (old & AP_RO) == AP_RW))
            return slot;
        break_cont_group(slot, &desc[pool_index(slot)], true);
        old = *slot;
    }
    u8 pi = gpa < PRIV_IPA ? page_info[gpa >> 12] : 0;
    if (!old && !(c->hugepages_disabled & 2u) && !(pi & PI_DIRECT) && !(smp && me->tracked_new) &&
        try_install_cont_64k(w, va, slot, gpa, want_write))
        return slot;
    u64 pte = gpa | PTE_VALID | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (old & PTE_X_MARK);
    bool writable = w->write && w->dirty;
    if (writable && (pi & (PI_CODE | PI_PT)))
        writable = false;
    pte |= writable ? AP_RW : AP_RO;
    if ((old & 1) && (old & AP_RO) == AP_RW && (old & PTE_ADDR) != gpa)
        rmap_remove(old & PTE_ADDR, slot);
    bool was_rw = (old & 1) && (old & AP_RO) == AP_RW && (old & PTE_ADDR) == gpa;
    *slot = pte;
    if (writable && !was_rw && gpa < c->ram_size && !(pi & PI_DIRECT)) // nothing write-protects these, and there can be far more than rmap holds
        rmap_add(gpa, slot);
    return slot;
}

static void track_root(View *v, u64 va, const Walk *w) {
    (void)va;
    (void)w;
    u64 gt = v->cr3 & X_ADDR;
    if (gt < c->ram_size && !(page_info[gt >> 12] & PI_PT)) {
        page_info[gt >> 12] |= PI_PT;
        write_protect_frame(gt);
        me->tracked_new = true;
    }
}

// ---- Delivering x86 events ----
// For the tally: what processes start, how they end, and their faults. Rare, so a host call each. shortcut: one wait4 tracked per address space, and the next
// return in that space counts as its return, which another thread of the same process could get wrong.
static void note_process_syscall(u64 next) {
    u64 n = c->x[0];
    bool console_write = (n == 0x2000004 || n == 0x200018d) && c->x[7] <= 2;
    if (!c->hybrid || (!console_write && n != 0x2000001 && n != 0x2000007 && n != 0x2000190 && n != 0x200003b && n != 0x20000f4 && n != 0x2000037))
        return;
    c->req_a[0] = 1;
    c->req_a[1] = next;
    hostcall(HC_NOTE);
    if (n == 0x2000007 || n == 0x2000190) {
        note_wait_cr3 = c->cr3;
        note_wait_status = c->x[6];
    }
}
static void note_wait_returned(u64 rip, u64 user_flags) {
    if (!note_wait_cr3 || note_wait_cr3 != c->cr3)
        return;
    c->req_a[0] = 2;
    c->req_a[1] = rip;
    c->req_a[2] = note_wait_status;
    c->req_a[3] = user_flags & 1; // CF: Darwin's way of saying rax is an errno
    note_wait_cr3 = 0;
    hostcall(HC_NOTE);
}

static inline void note_event(u32 kind, u64 rip, u64 a, u32 vector) {
    if (!c->hybrid)
        return;
    EngineEvent *e = &c->events[c->event_next++ % EVENT_RING];
    e->kind = kind;
    e->vector = vector;
    e->rip = rip;
    e->a = a;
    e->cr3 = c->cr3;
}

static u64 translate_here(u64 rip, u64 site, u32 kind, bool single);
static void fex_unspill_state(void);
static void reset_all_return_stacks(void);
static void deliver(u32 vector, u32 err, bool has_err, u64 ret_rip, bool sw_int);

static void raise_gp(u32 err, u64 rip) { deliver(13, err, true, rip, false); }

static void shim_fault(u32 vector, u32 err, u64 rip, u64 cr2) {
    c->req_a[5] = cr2;
    stop(STOP_GUEST_FAULT_NO_IDT, rip, vector, err, cr2);
}

static void raise_pf(u64 va, u32 err, u64 rip) {
    if (!c->model_kernel)
        shim_fault(14, err, rip, va);
    c->cr2 = va;
    c->stat_pf_delivered++;
    deliver(14, err, true, rip, false);
}

// ---- Translation requests ----
// The view's indirect-branch table answers for control transfers made here too: a hit was validated at translation time, and any later
// change to that mapping invalidates the entry (PTE_X_MARK). No host exit, no walk.
static u64 view_lookup(u64 rip) {
    u64 *t = (u64 *)view_current()->hash_va;
    u32 h;
    __asm__ volatile(".arch_extension crc\n crc32cx %w0, wzr, %x1" : "=r"(h) : "r"(rip));
    h &= HASH_ENTRIES - 1;
    for (;;) {
        u64 key = t[h * 2];
        if (!key)
            return 0;
        if (key == rip) {
            u64 val = __atomic_load_n(&t[h * 2 + 1], __ATOMIC_ACQUIRE);
            if (__atomic_load_n(&t[h * 2], __ATOMIC_RELAXED) != rip)
                return 0;
            return val & ~1ULL;
        }
        h = (h + 1) & (HASH_ENTRIES - 1);
    }
}

static inline void code_changed(u32 *word) {
    __asm__ volatile("dc cvau, %0\n dsb ish\n ic ivau, %0\n dsb ish\n isb" : : "r"(word) : "memory");
}

// In Part 0 4-slot ring mode, a newer slot must never permanently direct-branch into an older slot
// (otherwise recycling the older slot leaves dangling direct B/BL/B.cond instructions in surviving slots).
static inline bool bad_ring_link(u64 site, u64 arm) {
    if (!c0->p0_ring_active)
        return false;
    u64 cc = PRIV_VA + OFF_CC;
    if (site < cc || site >= cc + CC_SIZE || arm < cc || arm >= cc + CC_SIZE)
        return false;
    int fs = engine_p0_slot_of_word((u32)((site - cc) >> 2));
    int ts = engine_p0_slot_of_word((u32)((arm - cc) >> 2));
    if (fs < 0 || ts < 0)
        return true;
    return __atomic_load_n(&c0->p0_slot_gen[fs], __ATOMIC_ACQUIRE) >
           __atomic_load_n(&c0->p0_slot_gen[ts], __ATOMIC_ACQUIRE);
}

// tr_link, for a target this view already knows, with no way out of the VM. Half of all host calls in a fork loop did only this.
static void link_here(u64 site, u32 kind, u64 arm) {
    BlockEntry *from = find_block(site), *to = find_block(arm);
    bool bad_ring = bad_ring_link(site, arm);
    if (from && to && (from->key != to->key || bad_ring) && (kind == SVC_CALL || kind == SVC_DIRECT || kind == SVC_JCC)) {
        c->req_a[0] = site;
        c->req_a[1] = to->rip;
        hostcall(HC_CALL_TAIL);
        return;
    }
    if (!from || !to || from->key != to->key || bad_ring)
        return; // a link is forever, so it may only join blocks of one domain and safe ring generation order
    u32 *s = (u32 *)site, was = *s;
    if (site == arm || (kind != SVC_CALL && (was & 0xFFE0001Fu) != 0xD4000001u)) // as in tr_link: a dead block's entry may have come back
        return;
    i64 d = ((i64)arm - (i64)site) >> 2;
    if (d < -(1 << 25) || d >= (1 << 25)) { // out of ±128 MiB reach: install inline lookup tail instead of trapping forever
        if (kind == SVC_CALL || kind == SVC_DIRECT || kind == SVC_JCC) {
            c->req_a[0] = site;
            c->req_a[1] = to->rip;
            hostcall(HC_CALL_TAIL);
        }
        return;
    }
    if (!__atomic_compare_exchange_n(s, &was, (kind == SVC_CALL ? 0x94000000u : 0x14000000u) | ((u32)d & 0x3ffffff), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    code_changed(s);
    if (kind == SVC_JCC) { // as in tr_link: straight from the conditional branch if it reaches
        u32 *cond = s - s[3];
        i64 off = ((i64)arm - (i64)cond) >> 2;
        if (off >= -(1 << 18) && off < (1 << 18)) {
            *cond = (*cond & ~(0x7ffffu << 5)) | ((u32)off & 0x7ffff) << 5;
            code_changed(cond);
        }
    }
    c->stat_links_here++;
}

// ---- Shared user translation domains ----
// A view may look up a shared block only while its own tables lead to the shared directory: zap_entry has it forget them when that slot changes. The directory's
// contents can change for every view at once, and only the keeper sees that: its shadow carries the code marks, and it stays while it maps the directory.
static struct {
    u64 directory, first_cr3;
    u32 first_view, keeper; // keeper is valid once shared
} shared_seen[SHARED_USER_DIRS + 8];
#define SHARED_SEEN (SHARED_USER_DIRS + 8)
static u32 shared_seen_next;

// Whether the rip's directory is shared, deciding it now if a second address space has just reached it. Returns its row, or -1.
static int shared_row(const Walk *w, const View *v, u64 rip) {
    bool compat = c->compat && engine_shared_compat_address(rip);
    if (!c->share_user || (c->cpl != 3 && !(compat && c->cpl == 0)) ||
        (!compat && (c->compat || rip < SHARED_USER_BASE || rip >= SHARED_USER_END - 4096)) ||
        w->leaf_level > (compat ? 0 : 1))
        return -1;
    u64 directory = compat ? engine_shared_compat_domain(rip, *w->chain[1] & X_ADDR) : *w->chain[2] & X_ADDR;
    if (compat && c->cpl == 0)
        directory |= SHARED_COMPAT_KERNEL;
    for (u32 i = 0; i < SHARED_SEEN; i++) {
        if (shared_seen[i].directory != directory)
            continue;
        for (u32 d = 0; d < c0->shared_dir_count; d++)
            if (c0->shared_dirs[d] == directory)
                return (int)i;
        if (shared_seen[i].first_cr3 == v->cr3 || c0->shared_dir_count == SHARED_USER_DIRS)
            return -1;
        c0->shared_dirs[c0->shared_dir_count++] = directory;
        View *first = &views[shared_seen[i].first_view];
        shared_seen[i].keeper = first->valid && first->cr3 == shared_seen[i].first_cr3 && first->cpl == v->cpl ? shared_seen[i].first_view : (u32)(v - views);
        views[shared_seen[i].keeper].keeper = 1;
        return (int)i;
    }
    for (u32 n = 0; n < SHARED_SEEN; n++, shared_seen_next = (shared_seen_next + 1) % SHARED_SEEN) { // a row in use as shared stays
        bool in_use = false;
        for (u32 d = 0; d < c0->shared_dir_count; d++)
            in_use = in_use || c0->shared_dirs[d] == shared_seen[shared_seen_next].directory;
        if (!in_use)
            break;
    }
    shared_seen[shared_seen_next].directory = directory;
    shared_seen[shared_seen_next].first_cr3 = v->cr3;
    shared_seen[shared_seen_next].first_view = (u32)(v - views);
    shared_seen_next = (shared_seen_next + 1) % SHARED_SEEN;
    return -1;
}

// Puts the code mark for a shared page on the keeper, choosing a new one if this one has stopped mapping the directory. Its zap then killed what it marked.
static void shared_mark(int row, u64 va, View *v) {
    u64 directory = shared_seen[row].directory;
    View *k = &views[shared_seen[row].keeper];
    walk_any_tables = true; // it may be a dead process's, with its tables freed and poisoned
    Walk own = x86_walk(k->cr3, va, 0, false);
    walk_any_tables = false;
    bool compat = directory & SHARED_COMPAT_TAG;
    u64 identity = !own.ok || own.leaf_level > (compat ? 0 : 1) ? 0 :
        (compat ? engine_shared_compat_domain(va, *own.chain[1] & X_ADDR) : *own.chain[2] & X_ADDR);
    if (identity && compat)
        identity |= directory & SHARED_COMPAT_KERNEL;
    if (!k->valid || identity != directory) {
        k->keeper = 0;
        shared_seen[row].keeper = (u32)(v - views);
        k = v;
        k->keeper = 1;
        own = x86_walk(k->cr3, va, 0, false);
        if (!own.ok || own.leaf_level > (compat ? 0 : 1) ||
            (compat ? engine_shared_compat_domain(va, *own.chain[1] & X_ADDR) | (directory & SHARED_COMPAT_KERNEL) : *own.chain[2] & X_ADDR) != directory)
            return; // the block's own page is gone from under it: its next use faults
    }
    track_root(k, va, &own);
    mark_x_slot(k, va, &own);
}

// The stack grows down from the top of this vCPU's slot toward its MonCpu, with nothing between. Going past would write over this vCPU's own state and show up as anything at all.
static inline void stack_check(void) {
    u64 sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    u64 used = (u64)me + CTX_SLOT - sp;
    if (used > c->stat_stack_deepest) {
        c->stat_stack_deepest = used;
        if (used > CTX_SLOT - sizeof(MonCpu) - 4096)
            stop(STOP_MONITOR_FAULT, 0, 0x57ACC, sp, used);
    }
}

static u64 translate_here(u64 rip, u64 site, u32 kind, bool single) {
    stack_check();
    u32 attempts = 0;
again:;
    View *v = view_current();
    if (!single) {
        u64 hit = view_lookup(rip);
        if (hit) {
            if (site && kind)
                link_here(site, kind, hit);
            return hit;
        }
        if (!site && !kind)
            kind = SVC_LOOKUP_MISS;
    }
    Walk w = x86_walk(c->cr3, rip, 16 | (c->cpl == 3 ? 4 : 0), true);
    if (!w.ok) {
        if (w.fault_err == ~0u) {
            if (!c->model_kernel)
                shim_fault(13, 0, rip, rip);
            raise_gp(0, rip);
        } else
            raise_pf(rip, w.fault_err, rip);
        return c->elr;
    }
    if (c->hybrid && !is_ram(w.gpa))
        stop(STOP_FALLBACK, rip, FB_FETCH_NOT_RAM, w.gpa, 0);
    int shared = -1;
    if (!single && c->share_user) {
        bool maybe_compat = c->compat && engine_shared_compat_address(rip);
        if ((c->cpl == 3 || (maybe_compat && c->cpl == 0)) &&
            (maybe_compat || (!c->compat && rip >= SHARED_USER_BASE && rip < SHARED_USER_END - 4096)) &&
            w.leaf_level <= (maybe_compat ? 0 : 1)) {
            mon_lock();
            me->lock_site = 2;
            shared = shared_row(&w, v, rip);
            mon_unlock();
        }
    }
    c->req_a[0] = c->view;
    c->req_a[1] = rip;
    c->req_a[2] = site;
    c->req_a[3] = kind;
    c->req_a[4] = single;
    c->req_a[5] = c->cr3;
    hostcall(HC_TRANSLATE);
    u64 fault = c->req_r[1];
    if (fault) {
        if (!c->model_kernel)
            shim_fault(14, 0x10, rip, fault - 1);
        Walk f = x86_walk(c->cr3, fault - 1, 16 | (c->cpl == 3 ? 4 : 0), false);
        if (c->hybrid && f.ok) // mapped, but the rest of the instruction is not in RAM
            stop(STOP_FALLBACK, rip, FB_FETCH_NOT_RAM, f.gpa, 0);
        raise_pf(fault - 1, f.fault_err, rip);
        return c->elr;
    }
    if (c->tr_flushed) {
        c->tr_flushed = 0;
        code_cache_reset = true;
        reset_all_return_stacks();
    }
    if (single)
        return c->req_r[0]; // runs once and stays unregistered: nothing to protect, and its page must stay writable
    mon_lock();
    me->lock_site = 3;
    // A vCPU writing a code frame gets to finish first. Otherwise another calling into that frame translates and protects it again before the one-off store runs, and the
    // store faults for ever. It lets go at its next monitor entry, one instruction away, so this is short and needs nothing from here.
    for (u32 cpu = 0; smp && cpu < c0->vcpu_count; cpu++) {
        MonCpu *m = (MonCpu *)(PRIV_VA + OFF_MON_CPU + cpu * CTX_SLOT);
        u64 held = __atomic_load_n(&m->smc_held, __ATOMIC_ACQUIRE);
        if (m != me && held && (held == (c->req_r[2] & ~1ULL) + 1 || held == (c->req_r[3] & ~1ULL) + 1)) {
            mon_unlock();
            __asm__ volatile("yield");
            if (c->stop_requested) // that vCPU may itself be waiting for this one to leave the code cache, which it can do here
                hostcall(HC_PARK);
            if (c->tr_flushed) { // and what the host translated above went with it
                c->tr_flushed = 0;
                code_cache_reset = true;
                reset_all_return_stacks();
                site = 0;
                kind = 0;
                goto again;
            }
            mon_lock();
            cpu = ~0u; // and look at them all again
        }
    }
    pool_reclaim();
    bool newly_code = false;
    // Every frame code was read from is write-protected, so a later store invalidates it, and every page-table entry on the
    // way is tracked by marking the leaf slot.
    for (int i = 2; i < 4; i++) {
        u64 fr = c->req_r[i];
        if (!(fr & 1))
            continue;
        u64 gpa = fr & ~1ULL;
        if (!(page_info[gpa >> 12] & PI_CODE)) {
            write_protect_frame(gpa); // then the mark: the translator reads a frame only once it sees that
            __atomic_store_n(&page_info[gpa >> 12], page_info[gpa >> 12] | PI_CODE, __ATOMIC_RELEASE);
            newly_code = true;
        }
    }
    if (!c->req_r[0]) { // the host refused until the frame was protected, which it is now
        mon_unlock();
        if (++attempts >= 8) { // a writer keeps opening it: one instruction from a one-off block is progress
            single = true;
            site = 0;
            kind = 0;
        }
        goto again;
    }
    track_root(v, rip, &w);
    if (shared >= 0) { // this view just needs its own path built, so that a change to its slot above the directory reaches zap_entry
        shadow_slot(v, rip, &w, 0, 0);
        v->sees_shared = 1;
        shared_mark(shared, rip, v);
        if (c->req_r[3])
            shared_mark(shared, (rip & ~0xfffULL) + 4096, v);
    } else if (c->share_user && c->cpl == 0 && (rip >> 47)) {
        // As for a shared user domain: only the keeper's marks kill, as this view may well be a process about to exit, which would take the kernel's translations with it.
        shadow_slot(v, rip, &w, 0, 0);
        v->sees_shared = 1;
        View *keeper = domain_keeper(*w.chain[3] & X_ADDR, v);
        mark_x_slot(keeper, rip, &w);
        if (c->req_r[3]) {
            u64 next = (rip & ~0xfffULL) + 4096;
            Walk w2 = x86_walk(c->cr3, next, 0, false);
            if (w2.ok)
                mark_x_slot(keeper, next, &w2);
        }
    } else {
        mark_x_slot(v, rip, &w);
        if (c->cpl == 0 && (rip >> 47)) { // shared with other views: the domain's keeper must see writes to this mapping too
            View *keeper = domain_keeper(*w.chain[3] & X_ADDR, v);
            if (keeper != v)
                mark_x_slot(keeper, rip, &w);
        }
        if (c->req_r[3]) { // a straddling instruction: a change to the next page's mapping must invalidate this block as well
            u64 next = (rip & ~0xfffULL) + 4096;
            Walk w2 = x86_walk(c->cr3, next, 0, false);
            if (w2.ok)
                mark_x_slot(v, next, &w2);
        }
    }
    if (smp) {
        // The host read tables with nothing protecting them, and a frame's mark may have gone since. From here a store or remap finds this block, and anything earlier
        // shows now: look again, and if it moved or the frame needed marking again, what was translated may be stale.
        Walk now = x86_walk(c->cr3, rip, 16 | (c->cpl == 3 ? 4 : 0), true);
        bool moved = !now.ok || (now.gpa & ~0xfffULL) != (c->req_r[2] & ~1ULL);
        if (c->req_r[3]) { // a block that runs onto the next page
            Walk next = x86_walk(c->cr3, (rip & ~0xfffULL) + 4096, 16 | (c->cpl == 3 ? 4 : 0), false);
            moved = moved || !next.ok || (next.gpa & ~0xfffULL) != (c->req_r[3] & ~1ULL);
        }
        if (newly_code || moved) {
            u64 r2 = c->req_r[2], r3 = c->req_r[3];
            mon_unlock();
            c->req_a[1] = KILL_STORE;
            c->req_a[0] = r2 & ~1ULL;
            hostcall(HC_SMC);
            if (r3 & 1) {
                c->req_a[0] = r3 & ~1ULL;
                hostcall(HC_SMC);
            }
            reset_all_return_stacks();
            // Over again, unless another vCPU keeps storing to this frame, as with code and busy data on one page. Then one instruction, from a one-off block no store can take
            // away, is progress however often that happens.
            site = 0;
            kind = 0;
            single = ++attempts >= 8;
            goto again;
        }
    }
    mon_unlock();
    return c->req_r[0];
}

// ---- Page-table writes ----


static inline void mmu_xmark_zap(u64 old, u32 view) {
    if (c->profile_mmu_observations && zap_why < KILL_KINDS)
        c->stat_mmu_xmark_zaps[zap_why * 4 + (views[view].keeper != 0) * 2 + (old & 1)]++;
}

static bool in_zap_table;

static void zap_entry(u64 *e, const ShadowDesc *d) {
    u64 v = *e;
    if (!v)
        return;
    if (d->level == 0 && (v & 1) && (v & PTE_CONT) && !evicting && !in_zap_table) {
        break_cont_group(e, d, true);
        return;
    }
    *e = 0;
    if (d->level == 0) {
        u64 va = pte_va(e, d);
        if (v & 1) {
            if ((v & AP_RO) == AP_RW && (v & PTE_ADDR) < c->ram_size)
                rmap_remove(v & PTE_ADDR, e);
            if (!evicting && !in_zap_table)
                tlbi_va(view_asid(d->view), va);
        }
        if ((v & PTE_X_MARK) && !evicting) {
            mmu_xmark_zap(v, d->view);
            walk_any_tables = true;
            Walk now = (v & 1) ? x86_walk(views[d->view].cr3, va, 0, false) : (Walk){0};
            walk_any_tables = false;
            c->req_a[0] = d->view;
            c->req_a[1] = va;
            c->req_a[2] = zap_why;
            c->req_a[3] = now.ok && (now.gpa & ~0xfffULL) == (v & PTE_ADDR); // the guest's mapping of this page has not moved
            c->req_a[4] = views[d->view].keeper || !c->share_user;
            hostcall(HC_VIEW_INVALIDATE_VA);
            reset_all_return_stacks();
        }
    } else if (d->level == 1 && is_block_desc(v)) {
        if ((v & AP_RO) == AP_RW && (v & PTE_ADDR) < c->ram_size)
            rmap2m_remove(v & PTE_ADDR, e);
        if (!evicting && !in_zap_table)
            tlbi_va(view_asid(d->view), pte_va(e, d));
    } else if (is_table_desc(v)) {
        bool kernel_slot = d->level == 3 && (((u64)e & 0xfff) >> 3) >= 256;
        bool compat_parent = false;
        // Follow the actual shared-table edge, not every low-address parent
        // mutation. XNU's based copy windows change unrelated kernel mappings
        // constantly; forgetting all translations there defeats kernel sharing.
        // Higher-level removal visits these level-1 edges through zap_table.
        if (d->level == 1 && views[d->view].sees_shared && !evicting &&
            (views[d->view].cpl & 4) && pte_va(e, d) < (1ULL << 32)) {
            const ShadowDesc *child = &desc[pool_index((void *)IPA_TO_VA(v & PTE_ADDR))];
            u64 domain = engine_shared_compat_domain(pte_va(e, d), child->guest_table);
            if (!(views[d->view].cpl & 3))
                domain |= SHARED_COMPAT_KERNEL;
            for (u32 i = 0; i < c0->shared_dir_count; i++) {
                if (c0->shared_dirs[i] == domain) {
                    compat_parent = true;
                    break;
                }
            }
        }
        if ((kernel_slot || compat_parent || (d->level == 2 && pte_va(e, d) >= SHARED_USER_BASE && pte_va(e, d) < SHARED_USER_END)) && views[d->view].sees_shared && !evicting) { // what its lookups mean may have changed
            views[d->view].sees_shared = 0;
            // Publish new generation BEFORE HC_VIEW_FORGET so any vCPU switching into d->view concurrently either
            // observes the new fex_view_gen in view_switch or is seen in vc_ctx->view by fex_l1_clear_view.
            __atomic_store_n(&fex_view_gen[d->view], __atomic_add_fetch(&fex_gen_seq, 1, __ATOMIC_ACQ_REL), __ATOMIC_SEQ_CST);
            c->req_a[0] = d->view;
            hostcall(HC_VIEW_FORGET);
            reset_all_return_stacks(); // a return needs no lookup, and it could land on one of those blocks
        }
        zap_table(pool_index((void *)IPA_TO_VA(v & PTE_ADDR)));
    }
}

static void zap_table(u32 idx) {
    LOCK_REQUIRED();
    ShadowDesc *d = &desc[idx];
    u64 *t = pool_page(idx);
    bool prev_in_zap = in_zap_table;
    bool had_leaf = false;
    bool batch_tlbi = !(c->hugepages_disabled & 8u);
    if (batch_tlbi)
        in_zap_table = true;
    for (int i = 0; i < 512; i++) {
        if (d->level == 3 && !d->va_base && (u64)i == ((PRIV_VA >> 39) & 0x1ff))
            continue; // the private slots of root0 are not shadows; root1 slots 480/482 are real kernel mappings
        if (d->level == 3 && !d->va_base && (u64)i == ((RAM_ALIAS_VA >> 39) & 0x1ff))
            continue;
        if (t[i]) {
            if (d->level == 0 || (d->level == 1 && is_block_desc(t[i])))
                had_leaf = true;
            zap_entry(&t[i], d);
        }
    }
    in_zap_table = prev_in_zap;
    // Ensure no core retains a stale TLB/walk entry pointing into this page before reusing slot 0 for pool_free_head.
    if (batch_tlbi && !evicting && had_leaf)
        __asm__ volatile("dsb ishst\n tlbi aside1is, %0\n dsb ish\n isb" : : "r"((u64)view_asid(d->view) << 48));
    if (d->guest_table != ~0ULL && d->guest_table < c->ram_size) {
        u32 *link = &mirror_head[d->guest_table >> 12];
        while (*link) {
            if (*link == idx + 1) {
                *link = d->next_mirror;
                break;
            }
            link = &desc[*link - 1].next_mirror;
        }
    }

    *(u32 *)t = pool_free_head;
    pool_free_head = idx + 1;
    pool_free_count++;
    __asm__ volatile("dsb ishst");
}

// A guest wrote a tracked table: drop every shadow entry that mirrors the slot. The next access refaults and refills.
// A frame written many times with no walk through it in between has stopped being a page table (a freed one reused as data, say). Unshadow
// it, or every store to it would trap for as long as the frame lives.
#define PT_FLOOD_WRITES 64
static inline bool pt_flood_decision(ShadowDesc *d, bool nobody_home) {
    if (!(d->level < 3 && d->parent))
        return false;
    bool threshold = ++d->writes > PT_FLOOD_WRITES; // Exactly the original eligibility and increment, including u32 wrap.
    if (c->keeper_inactive_guard && nobody_home && views[d->view].keeper) {
        if (!threshold) {
            c->stat_keeper_inactive_suppressed++;
            return false; // Keep tracking: the caller still zaps the written slot, including all affected execute marks.
        }
        c->stat_keeper_inactive_threshold_fallbacks++;
    }
    if (!(threshold || nobody_home))
        return false;
    if (c->profile_mmu_observations) {
        u32 cause = threshold ? (nobody_home ? 2 : 0) : 1;
        c->stat_mmu_flood_decisions[cause * 2 + (views[d->view].keeper != 0)]++;
    }
    return true;
}

#define X_ARCH_UPPER (X_P | X_RW | X_US | X_PS | X_NX | X_ADDR)
#define X_ARCH_LEAF  (X_P | X_RW | X_US | X_D  | X_NX | X_ADDR)
#define X_ARCH_ALL   (X_P | X_RW | X_US | X_D  | X_PS | X_NX | X_ADDR)
static inline bool pte_arch_same(u64 old, u64 next, u32 level) {
    if (!(old & X_P))
        return false;
    if (level == 0)
        return (old & X_ARCH_LEAF) == (next & X_ARCH_LEAF);
    if (old & X_PS)
        return (old & (X_ARCH_UPPER | X_D)) == (next & (X_ARCH_UPPER | X_D));
    return (old & X_ARCH_UPPER) == (next & X_ARCH_UPPER);
}

static void pt_slot_changed_impl(u64 gpa_slot, bool same64, bool has_old_next, u64 old_val, u64 next_val) {
    u32 fr = (u32)(gpa_slot >> 12), i = (u32)((gpa_slot & 0xfff) >> 3);
    bool populate_present = has_old_next && !(old_val & X_P) && (next_val & X_P) &&
                            ((next_val & X_ADDR) >= 0x1000ULL) && ((next_val & X_ADDR) < c->ram_size);
    zap_why = KILL_PT_LEAF;
    for (u32 it = mirror_head[fr], next; it; it = next) {
        ShadowDesc *d = &desc[it - 1];
        u64 *t = pool_page(it - 1);
        next = d->next_mirror;
        if (has_old_next && pte_arch_same(old_val, next_val, d->level))
            continue;
        // A user table of an address space no vCPU is in: it is being torn down or its frame reused, and every store would trap. It refills if that address space ever runs.
        bool nobody_home = d->view != c->view && d->va_base >> 47 == 0 && !view_pinned(d->view);
        if (!(populate_present && !nobody_home) && pt_flood_decision(d, nobody_home)) {
            u64 *parent = d->parent;
            ShadowDesc pd = desc[pool_index(parent)];
            u32 view = d->view;
            c->stat_pt_unshadow++;
            zap_why = KILL_PT_FLOOD;
            if (is_table_desc(*parent) && (*parent & PTE_ADDR) == VA_TO_IPA((u64)t))
                zap_entry(parent, &pd);
            else
                zap_table(it - 1);
            zap_why = KILL_PT_LEAF;
            __asm__ volatile("dsb ishst\n tlbi aside1is, %0\n dsb ish\n isb" : : "r"((u64)view_asid(view) << 48));
            next = mirror_head[fr]; // the list changed; anything still on it restarts, each has a bounded counter
            if (!next)
                page_info[fr] &= ~PI_PT;
            continue;
        }
        if (d->level == 3) {
            View *v = &views[d->view];
            // The user half of a root going to zero with nobody in it: the address space is gone and its frame is being cleared, one trap per slot. It rebuilds if it was not gone.
            if (nobody_home && i < 256 && !*(u64 *)RAM((u64)fr << 12 | (u64)i << 3) && !v->keeper) {
                view_drop(d->view);
                zap_why = KILL_PT_LEAF;
                next = mirror_head[fr];
                if (!next)
                    page_info[fr] &= ~PI_PT;
                continue;
            }
            if ((i < 256) != (d->va_base == 0))
                continue;
            u64 *r = (i < 256) ? (u64 *)v->root0 : (u64 *)v->root1;
            if (r[i]) {
                ShadowDesc rd = *d;
                rd.level = 3;
                zap_why = KILL_PT_TABLE;
                zap_entry(&r[i], &rd);
                zap_why = KILL_PT_LEAF;
                __asm__ volatile("dsb ishst\n tlbi aside1is, %0\n dsb ish\n isb" : : "r"((u64)view_asid(d->view) << 48));
            }
            continue;
        }
        if (t[i]) {
            // Only an exact pre-store value returned by the opt-in aligned64
            // atomic exchange supplies this witness. Still run every flood,
            // root-retirement and higher-level action above/as before: repeated
            // identical stores may be a freed table being overwritten as data.
            if (same64 && d->level == 0) {
                c->stat_pte_same64_leaf_skips++;
                if (t[i] & PTE_X_MARK)
                    c->stat_pte_same64_xmark_skips++;
                continue;
            }
            bool table = d->level > 0;
            zap_why = table ? KILL_PT_TABLE : KILL_PT_LEAF;
            zap_entry(&t[i], d);
            if (table) {
                __asm__ volatile("dsb ishst\n tlbi aside1is, %0\n dsb ish\n isb" : : "r"((u64)view_asid(d->view) << 48));
                c->stat_pt_writes_zap++;
            }
        }
    }
}

// All legacy callers retain their original invalidation and retirement behavior.
static void pt_slot_changed(u64 gpa_slot) { pt_slot_changed_impl(gpa_slot, false, false, 0, 0); }

// ---- ARM load and store decode, for the forms the translator emits ----
typedef struct {
    u8 bytes, rt, rn;
    bool load, sign, sign32, writeback;
    i64 wb_delta;
} ArmAccess;

static bool arm_decode(u32 w, ArmAccess *a) {
    if ((w & 0x3F000000u) != 0x38000000u && (w & 0x3F000000u) != 0x39000000u)
        return false; // bit 26 set is SIMD; pairs and exclusives live elsewhere
    u32 opc = (w >> 22) & 3, size = w >> 30;
    a->bytes = (u8)(1u << size);
    a->rt = w & 31;
    a->rn = (w >> 5) & 31;
    a->load = opc != 0;
    a->sign = opc >= 2;
    a->sign32 = opc == 3;
    a->writeback = false;
    a->wb_delta = 0;
    if (w & (1u << 24))
        return true; // unsigned immediate
    if (w & (1u << 21)) {
        // Atomics share this space with register offset: only option bits 10 = 2 means a register offset.
        return ((w >> 10) & 3) == 2;
    }
    u32 kind = (w >> 10) & 3;
    if (kind == 2)
        return false; // unprivileged forms are never emitted
    if (kind != 0) {
        a->writeback = true;
        a->wb_delta = (i64)(int32_t)(w << 11) >> 23;
    }
    return true;
}

// Finishes an emulated access: load value into the register, writeback, skip the word.
static void arm_finish(const ArmAccess *a, u64 r) {
    if (a->load && a->rt != 31) {
        if (a->sign) {
            u32 bits = a->bytes * 8u;
            r = (u64)((i64)(r << (64 - bits)) >> (64 - bits));
            if (a->sign32)
                r &= 0xffffffffULL;
        }
        c->x[a->rt] = r;
    }
    if (a->writeback)
        c->x[a->rn] += (u64)a->wb_delta;
    c->elr += 4;
}

// When only the second page refuses a split access, FAR is that page's start, so FAR can't say whether an access is split: work out its real address.
static u64 arm_address(u32 w, const ArmAccess *a) {
    u64 ea = c->x[a->rn]; // rn is never sp: the translator reaches guest memory through general registers
    if (w & (1u << 24)) {
        ea += (u64)((w >> 10) & 0xfff) << (w >> 30);
    } else if (w & (1u << 21)) {
        u32 rm = (w >> 16) & 31, option = (w >> 13) & 7;
        u64 off = rm == 31 ? 0 : c->x[rm];
        if (option == 2)
            off = (u32)off;
        else if (option == 6)
            off = (u64)(i64)(int32_t)off;
        ea += (w & (1u << 12)) ? off << (w >> 30) : off;
    } else if (((w >> 10) & 3) != 1) { // post-index accesses the base as it stands
        ea += (u64)((i64)(int32_t)(w << 11) >> 23);
    }
    return ea;
}
static bool arm_splits(u64 ea, const ArmAccess *a) { return (ea & 0xfff) + a->bytes > 4096; }

static bool is_ram(u64 gpa);
static void pt_slot_changed(u64 gpa);

// A store across a page boundary with a tracked frame on either side. Boots do hundreds of these, mostly file data landing on frames once
// used as tables. Both pages are checked first, so a fault leaves memory as it was.
static void pt_split_store(u64 ea, const ArmAccess *a, u64 val) {
    u64 here = rip_here(c->elr), second = (ea | 0xfff) + 1;
    u32 access = 2 | (c->cpl == 3 ? 4 : 0), first = 4096 - (u32)(ea & 0xfff);
    Walk w1 = x86_walk(c->cr3, ea, access, true);
    if (w1.ok) {
        Walk w2 = x86_walk(c->cr3, second, access, true);
        if (w2.ok && is_ram(w1.gpa) && is_ram(w2.gpa) && !((page_info[w1.gpa >> 12] | page_info[w2.gpa >> 12]) & PI_CODE)) {
            memcpy(RAM(w1.gpa), &val, first);
            memcpy(RAM(w2.gpa), (u8 *)&val + first, a->bytes - first);
            if (page_info[w1.gpa >> 12] & PI_PT)
                pt_slot_changed(w1.gpa & ~7ULL);
            if (page_info[w2.gpa >> 12] & PI_PT)
                pt_slot_changed(w2.gpa & ~7ULL);
            arm_finish(a, 0);
            return;
        }
        if (!w2.ok && w2.fault_err != ~0u) {
            raise_pf(second, w2.fault_err, here);
            return;
        }
    } else if (w1.fault_err != ~0u) {
        raise_pf(ea, w1.fault_err, here);
        return;
    }
    stop(STOP_UNSUPPORTED, here, 5, ea, c->esr); // code on one side, a device or a non-canonical address: TCG has the whole of it
}

// ---- Virtual local APIC ----
// The hot registers answer here without leaving the VM. Registers that need another CPU or a level-triggered EOI go to the host.
static inline bool vapic_owns(u64 gpa) { return c->apic_enabled && (gpa & ~0xfffULL) == c->apic_base; }

static inline u32 apic_highest(const u32 *bits) {
    for (int i = 7; i >= 0; i--)
        if (bits[i])
            return (u32)i * 32 + 31 - (u32)__builtin_clz(bits[i]);
    return 0;
}

static u32 apic_ppr(void) {
    u32 isr = apic_highest(c->apic_isr), tpr = c->apic_tpr;
    return (tpr >> 4) >= (isr >> 4) ? tpr : (isr & 0xf0);
}

static void apic_merge_host(void) {
    for (int i = 0; i < 8; i++) {
        if (c->apic_host_irr[i]) {
            u32 bits = __atomic_exchange_n(&c->apic_host_irr[i], 0, __ATOMIC_ACQ_REL);
            c->apic_irr[i] |= bits;
        }
    }
}

// The vector that may be delivered now, or 0.
static u32 apic_ready_vector(void) {
    if (!c->apic_enabled || !(c->apic_svr & 0x100))
        return 0;
    u32 v = apic_highest(c->apic_irr);
    if (!v || (v & 0xf0) <= (apic_ppr() & 0xf0))
        return 0;
    return v;
}

static inline void apic_set_irr(u32 vector) { c->apic_irr[vector / 32] |= 1u << (vector & 31); }

static inline u64 apic_cnt(void) { return RSR(cntvct_el0); }
// The APIC timer runs at 1 GHz before its divider, like the TSC here: 125/3 of the 24 MHz counter.
static inline u64 apic_ticks_to_cnt(u64 ticks) { return ticks * 3 / 125; }

static void apic_timer_arm(void) {
    if (c->apic_deadline_cnt) {
        WSR(cntv_cval_el0, c->apic_deadline_cnt);
        WSR(cntv_ctl_el0, 1);
    } else
        WSR(cntv_ctl_el0, 2);
}

static void apic_timer_start(void) {
    u32 div = c->apic_tdcr & 0xb;
    u32 shift = ((div & 3) | ((div >> 1) & 4));
    shift = shift == 7 ? 0 : shift + 1;
    c->apic_timer_shift = shift;
    c->apic_timer_start_cnt = apic_cnt();
    c->apic_deadline_cnt = c->apic_tmict ? c->apic_timer_start_cnt + (apic_ticks_to_cnt((u64)c->apic_tmict << shift) | 1) : 0;
    apic_timer_arm();
}

// Called when the counter passed the deadline. Sets the vector and rearms a periodic timer.
static void apic_timer_check(void) {
    if (!c->apic_deadline_cnt || apic_cnt() < c->apic_deadline_cnt)
        return;
    u32 lvt = c->apic_lvt_timer;
    if (!(lvt & 0x10000))
        apic_set_irr(lvt & 0xff);
    c->stat_apic_timer++;
    if (((lvt >> 17) & 3) == 1 && c->apic_tmict) {
        u64 period = apic_ticks_to_cnt((u64)c->apic_tmict << c->apic_timer_shift) | 1;
        u64 next = c->apic_deadline_cnt + period, now = apic_cnt();
        if (next <= now) // missed ticks do not queue, as on hardware a period only fires once per IRR
            next = now + period;
        c->apic_timer_start_cnt = next - period;
        c->apic_deadline_cnt = next;
    } else
        c->apic_deadline_cnt = 0;
    apic_timer_arm();
}

static u32 vapic_read(u32 off) {
    u32 reg = off & 0xff0;
    if (reg >= 0x100 && reg < 0x180)
        return c->apic_isr[(reg - 0x100) >> 4];
    if (reg >= 0x180 && reg < 0x200)
        return c->apic_tmr[(reg - 0x180) >> 4];
    if (reg >= 0x200 && reg < 0x280) {
        apic_merge_host();
        return c->apic_irr[(reg - 0x200) >> 4];
    }
    switch (reg) {
        case 0x20:
            return c->apic_id;
        case 0x30:
            return 0x00050014;
        case 0x80:
            return c->apic_tpr;
        case 0xA0:
            return apic_ppr();
        case 0xD0:
            return c->apic_ldr;
        case 0xE0:
            return c->apic_dfr;
        case 0xF0:
            return c->apic_svr;
        case 0x280:
            return 0;
        case 0x300:
            return c->apic_icr_lo;
        case 0x310:
            return c->apic_icr_hi;
        case 0x320:
            return c->apic_lvt_timer;
        case 0x330:
        case 0x340:
        case 0x350:
        case 0x360:
        case 0x370:
            return c->apic_lvt[(reg - 0x330) >> 4];
        case 0x380:
            return c->apic_tmict;
        case 0x390: {
            if (!c->apic_deadline_cnt)
                return 0;
            u64 now = apic_cnt();
            if (now >= c->apic_deadline_cnt)
                return 0;
            return (u32)((c->apic_deadline_cnt - now) * 125 / 3 >> c->apic_timer_shift);
        }
        case 0x3E0:
            return c->apic_tdcr;
        default:
            return 0;
    }
}

static void vapic_write(u32 off, u32 v) {
    u32 reg = off & 0xff0;
    switch (reg) {
        case 0x20:
            c->apic_id = v;
            break;
        case 0x80:
            c->apic_tpr = v & 0xff;
            break;
        case 0xB0: { // EOI
            u32 vec = apic_highest(c->apic_isr);
            if (vec) {
                c->apic_isr[vec / 32] &= ~(1u << (vec & 31));
                if (c->apic_tmr[vec / 32] & (1u << (vec & 31))) { // a level-triggered source lives in the host's IO-APIC
                    c->apic_tmr[vec / 32] &= ~(1u << (vec & 31));
                    c->req_a[0] = vec;
                    hostcall(HC_APIC_EOI);
                }
            }
            break;
        }
        case 0xD0:
            c->apic_ldr = v;
            break;
        case 0xE0:
            c->apic_dfr = v;
            break;
        case 0xF0:
            c->apic_svr = v;
            break;
        case 0x300: {
            c->apic_icr_lo = v & ~0x1000u; // delivery status always idle
            u32 mode = (v >> 8) & 7, shorthand = (v >> 18) & 3, vec = v & 0xff;
            bool self = shorthand == 1 || (shorthand == 0 && (c->apic_icr_hi >> 24) == (c->apic_id >> 24));
            if (mode == 0 && self) {
                apic_set_irr(vec);
                c->stat_ipi_self++;
            } else {
                c->req_a[0] = (u64)c->apic_icr_hi << 32 | v;
                hostcall(HC_APIC_ICR);
            }
            break;
        }
        case 0x310:
            c->apic_icr_hi = v;
            break;
        case 0x320:
            c->apic_lvt_timer = v;
            break;
        case 0x330:
        case 0x340:
        case 0x350:
        case 0x360:
        case 0x370:
            c->apic_lvt[(reg - 0x330) >> 4] = v;
            break;
        case 0x380:
            c->apic_tmict = v;
            apic_timer_start();
            break;
        case 0x3E0:
            c->apic_tdcr = v;
            break;
        default:
            break;
    }
}

// ---- Faults ----
static void step_on(void) {
    c->spsr |= SPSR_SS;
    if (!c->stepping) { // MDSCR_EL1.SS with SPSR.SS clear would trap at once, so it is only on while stepping
        WSR(mdscr_el1, 1);
        c->stepping = 1;
    }
}
static void step_off(void) {
    c->spsr &= ~SPSR_SS;
    if (c->stepping) {
        WSR(mdscr_el1, 0);
        c->stepping = 0;
    }
}

// pt_step_slot: nonzero while one store runs on a temporarily writable page-table page

static void shadow_write_finish(void);

// Timing wraps the two existing call sites, not x86_walk itself: recursive A/D retries remain inside their caller's sample.
// Counter reads are deliberately not serialized; these are sampled diagnostic ticks, not instruction-cycle measurements.
typedef struct {
    u64 began;
    bool sampled; // CNTVCT zero is a valid sample, not a disabled sentinel.
} MmuWalkSample;

static inline MmuWalkSample mmu_walk_begin(bool locked) {
    if (!c->profile_mmu_observations)
        return (MmuWalkSample){0, false};
    if (!c->stat_mmu_counter_hz)
        c->stat_mmu_counter_hz = RSR(cntfrq_el0);
    u64 count = locked ? ++c->stat_mmu_locked_rewalk_calls : ++c->stat_mmu_first_walk_calls;
    bool sample = (count & 255) == 0;
    return (MmuWalkSample){sample ? RSR(cntvct_el0) : 0, sample};
}

static inline void mmu_walk_end(bool locked, MmuWalkSample sample) {
    if (!sample.sampled)
        return;
    u64 elapsed = RSR(cntvct_el0) - sample.began;
    if (locked) {
        c->stat_mmu_locked_rewalk_samples++;
        c->stat_mmu_locked_rewalk_ticks += elapsed;
    } else {
        c->stat_mmu_first_walk_samples++;
        c->stat_mmu_first_walk_ticks += elapsed;
    }
}

static inline void mmu_observe_store(u32 kind, u64 before, u64 after) {
    if (!c->profile_mmu_observations)
        return;
    if (!(kind & 1)) { // The value actually stored by W-sized LSE/plain stores, including wraparound.
        before = (u32)before;
        after = (u32)after;
    }
    c->stat_mmu_store_observed[kind]++;
    c->stat_mmu_store_unchanged[kind] += before == after;
}

static inline bool walk_still_valid(const Walk *w) {
    for (int l = 3; l >= w->leaf_level; l--) {
        if ((*w->chain[l] & ~X_A) != (w->read[l] & ~X_A))
            return false;
    }
    return true;
}

static inline void hybrid_apic_tpr_write(u32 new_tpr) {
    u32 old_tpr = c->apic_tpr;
    new_tpr &= 0xffu;
    if (new_tpr != old_tpr)
        __atomic_store_n(&c->apic_tpr, new_tpr, __ATOMIC_SEQ_CST);
    if ((new_tpr & 0xf0u) <= (old_tpr & 0xf0u) &&
        __atomic_load_n(&c->apic_irr_prio, __ATOMIC_SEQ_CST) > (new_tpr & 0xf0u)) {
        c->req_a[0] = new_tpr;
        hostcall(HC_APIC_TPR);
    }
}

static void hybrid_apic_eoi_fast(bool *did_fast) {
    u32 isrv = apic_highest(c->apic_isr);
    if (!isrv) {
        if (did_fast)
            *did_fast = true;
        return;
    }
    bool level = (c->apic_tmr[isrv >> 5] >> (isrv & 31)) & 1u;
    bool fast_ok = false;
    if (!level && !c->hybrid_apic_vapic) {
        u32 wi = isrv >> 5, bm = 1u << (isrv & 31);
        c->apic_isr[wi] &= ~bm;
        u32 next_isrv = apic_highest(c->apic_isr);
        u32 tpr_p = c->apic_tpr & 0xf0u;
        u32 isr_p = next_isrv & 0xf0u;
        u32 new_ppr = tpr_p >= isr_p ? tpr_p : isr_p;
        __atomic_store_n(&c->hybrid_eoi_pending, 1u, __ATOMIC_SEQ_CST);
        u32 irr_prio = __atomic_load_n(&c->apic_irr_prio, __ATOMIC_SEQ_CST);
        if (irr_prio == 0 || irr_prio <= new_ppr) {
            fast_ok = true;
        } else {
            c->apic_isr[wi] |= bm;
        }
    }
    if (!fast_ok) {
        c->req_a[0] = isrv;
        hostcall(HC_APIC_EOI);
    }
    if (did_fast)
        *did_fast = fast_ok;
}

static void handle_data_abort(bool lower) {
    (void)lower;
    u64 far = c->far, esr = c->esr;
    u32 dfsc = esr & 0x3f;
    bool write = esr & (1 << 6);
    View *v = view_current();
    if ((dfsc & 0x3c) == 0x20 || dfsc == 0x21)
        stop(STOP_UNSUPPORTED, rip_here(c->elr), 3, far, esr); // an alignment fault: split locked access
    // The return-stack guard: drop all the entries and retry the one ARM load/store.
    u64 rs_base = PRIV_VA + engine_off_rstacks(cpu_index());
    if (far >= rs_base && far < rs_base + MAX_VIEWS * RSTACK_SLOT + RSTACK_BYTES) {
        if (c->x[R_FS] == (u64)c + 32768ULL)
            c->x[25] = c->rs_top;
        else
            c->x[R_RS] = c->rs_top;
        *(u64 *)c->rs_top = ~0ULL;
        return;
    }
    if (!canonical(far)) {
        if (!c->model_kernel)
            shim_fault(13, 0, rip_here(c->elr), far);
        raise_gp(0, rip_here(c->elr));
        return;
    }
    u32 access = (write ? 2 : 0) | (c->cpl == 3 ? 4 : 0);
    MmuWalkSample walk_sample = mmu_walk_begin(false);
    Walk w = x86_walk(c->cr3, far, access, true);
    mmu_walk_end(false, walk_sample);
    if (!w.ok) {
        raise_pf(far, w.fault_err, rip_here(c->elr));
        return;
    }
    if (!is_ram(w.gpa) && !is_direct(w.gpa)) { // device hole: emulate the ARM access itself, so all x86 forms of a move or RMW work
        if (in_rep_stub(c->elr)) { // string instruction on a device: back to the plain per-element loop
            c->elr = c->x[R_LR];
            return;
        }
        ArmAccess a;
        if (!arm_decode(*(u32 *)c->elr, &a))
            stop(STOP_UNSUPPORTED, rip_here(c->elr), 4, far, esr); // pair, atomic or SIMD access to a device
        if (arm_splits(arm_address(*(u32 *)c->elr, &a), &a))
            stop(STOP_UNSUPPORTED, rip_here(c->elr), 15, far, esr); // part of it is on another page
        u64 val = a.rt == 31 ? 0 : c->x[a.rt];
        u64 r = 0;
        u64 reg = (w.gpa & ~0xfffULL) == c->hybrid_apic_page && a.bytes == 4 ? w.gpa & 0xfff : 0;
        // Hybrid HPET main counter EL1 fast read (0xfed000f0 4B/8B, 0xfed000f4 4B): read-only c0 seqlock,
        // calibrated against QEMU HPET every 1s with monotonic floor, zero VM exit and zero BQL.
        if (c->hybrid && a.load &&
            ((w.gpa == 0xfed000f0ULL && (a.bytes == 4 || a.bytes == 8)) ||
             (w.gpa == 0xfed000f4ULL && a.bytes == 4))) {
            u32 s1 = __atomic_load_n(&c0->hpet_fast_seq, __ATOMIC_ACQUIRE);
            if (!(s1 & 1u) && __atomic_load_n(&c0->hpet_fast_enabled, __ATOMIC_RELAXED)) {
                i64 off_ns = c0->hpet_fast_offset_ns;
                u64 expire_ns = c0->hpet_fast_expire_ns;
                u64 floor_tick = c0->hpet_fast_floor_tick;
                __asm__ volatile("dmb ishld" : : : "memory");
                u32 s2 = __atomic_load_n(&c0->hpet_fast_seq, __ATOMIC_RELAXED);
                if (s1 == s2) {
                    u64 raw_ns = (RSR(cntvct_el0) + c->tsc_counter_offset) * 125 / 3;
                    if (raw_ns < expire_ns) {
                        i64 hns = (i64)raw_ns + off_ns;
                        u64 tick = hns > 0 ? (u64)hns / 10ULL : 0ULL;
                        if (tick < floor_tick)
                            tick = floor_tick;
                        r = w.gpa == 0xfed000f4ULL ? (tick >> 32) : (a.bytes == 4 ? (u32)tick : tick);
                        c->stat_mmio_vm++;
                        arm_finish(&a, r);
                        return;
                    }
                }
            }
        }
        if (reg == 0x20 && a.load && me->hybrid_apic_id_cached) {
            r = me->hybrid_apic_id_val;
            c->stat_mmio_vm++;
        } else if (reg == 0x30 && a.load && me->hybrid_apic_ver_cached) {
            r = me->hybrid_apic_ver_val;
            c->stat_mmio_vm++;
        } else if (reg == 0x80) { // hybrid APIC TPR MMIO: keep c->apic_tpr coherent with CR8 fast-path
            if (a.load) {
                r = c->apic_tpr;
            } else {
                hybrid_apic_tpr_write((u32)val & 0xffu);
            }
            c->stat_mmio_vm++;
        } else if (reg == 0xb0 && !a.load) { // hybrid LAPIC EOI fast path: defer edge-triggered EOI when no IRR unmasked
            hybrid_apic_eoi_fast(0);
            c->stat_mmio_vm++;
        } else if (reg == 0x300 || reg == 0x310) { // the kernel reads both and writes both for each IPI, and only the last write needs QEMU
            if (a.load)
                r = reg == 0x300 ? c->apic_icr_lo : c->apic_icr_hi;
            else if (reg == 0x310)
                c->apic_icr_hi = (u32)val;
            else {
                c->apic_icr_lo = (u32)val & ~0x1000u; // delivery status always idle
                c->req_a[0] = (u64)c->apic_icr_hi << 32 | (u32)val;
                hostcall(HC_APIC_ICR);
            }
            c->stat_mmio_vm++;
        } else if (vapic_owns(w.gpa)) {
            if (a.load)
                r = vapic_read((u32)(w.gpa & 0xfff));
            else
                vapic_write((u32)(w.gpa & 0xfff), (u32)val);
            c->stat_mmio_vm++;
        } else {
            if (reg == 0x20 && !a.load)
                me->hybrid_apic_id_cached = 0;
            c->req_a[0] = w.gpa;
            c->req_a[1] = a.bytes;
            c->req_a[2] = !a.load;
            c->req_a[3] = val;
            c->req_a[4] = (c->stat_mmio & 255) ? 0 : rip_here(c->elr); // a sample for the tally: finding the rip costs a search
            c->stat_mmio++;
            hostcall(HC_MMIO);
            r = c->req_r[0];
            if (reg == 0x20 && a.load) {
                me->hybrid_apic_id_val = (u32)r;
                me->hybrid_apic_id_cached = 1;
            } else if (reg == 0x30 && a.load) {
                me->hybrid_apic_ver_val = (u32)r;
                me->hybrid_apic_ver_cached = 1;
            }
        }
        arm_finish(&a, r);
        return;
    }
    // Lock-free fast path for aligned 64-bit guest page-table updates that do not invalidate any shadow mapping:
    // (A) Arch-bit-preserving updates on present PTEs (e.g. Windows clearing X_A or updating software bits)
    // (B) Populating a brand-new present RAM PTE into a previously non-present slot, verified via mon_lock_word seqlock.
    if (smp && write && is_ram(w.gpa) && !(w.gpa & 7ULL) && !c->profile_mmu_observations && !c->pte_same64 && !in_rep_stub(c->elr)) {
        u32 fr_fast = (u32)(w.gpa >> 12);
        u8 pi_fast = __atomic_load_n(&page_info[fr_fast], __ATOMIC_ACQUIRE);
        if ((pi_fast & PI_PT) && !(pi_fast & (PI_CODE | PI_NOTE))) {
            u32 word = *(u32 *)c->elr;
            ArmAccess a;
            if (arm_decode(word, &a) && !a.load && a.bytes == 8 && !arm_splits(arm_address(word, &a), &a)) {
                u64 val = a.rt == 31 ? 0 : c->x[a.rt];
                u64 *slot8 = (u64 *)RAM(w.gpa);
                u32 s1 = __atomic_load_n(&mon_lock_word, __ATOMIC_ACQUIRE);
                u64 peek = __atomic_load_n(slot8, __ATOMIC_ACQUIRE);
                bool arch_same = (peek & X_P) && (((peek ^ val) & X_ARCH_ALL) == 0);
                bool pop_new = !(peek & X_P) && (val & X_P) &&
                               ((val & X_ADDR) >= 0x1000ULL) && ((val & X_ADDR) < c->ram_size) &&
                               !(s1 & 1u) && !__atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE);
                if (arch_same || pop_new) {
                    u64 expected = peek;
                    if (__atomic_compare_exchange_n(slot8, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
                        if (pop_new &&
                            (__atomic_load_n(&mon_lock_word, __ATOMIC_ACQUIRE) != s1 ||
                             __atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE) ||
                             (__atomic_load_n(&page_info[fr_fast], __ATOMIC_ACQUIRE) & PI_NOTE))) {
                            mon_lock();
                            me->lock_site = 5;
                            pt_slot_changed_impl(w.gpa & ~7ULL, false, true, peek, val);
                            mon_unlock();
                        }
                        c->stat_pt_writes++;
                        arm_finish(&a, 0);
                        return;
                    }
                }
            } else if ((word & 0xFFE0FC00u) == 0xC8E0FC00u) { // 64-bit CAS
                u32 rs = (word >> 16) & 31, rt = word & 31;
                u64 rs_v = rs == 31 ? 0 : c->x[rs], rt_v = rt == 31 ? 0 : c->x[rt];
                u32 s1 = __atomic_load_n(&mon_lock_word, __ATOMIC_ACQUIRE);
                bool arch_same = (rs_v & X_P) && (((rs_v ^ rt_v) & X_ARCH_ALL) == 0);
                bool pop_new = !(rs_v & X_P) && (rt_v & X_P) &&
                               ((rt_v & X_ADDR) >= 0x1000ULL) && ((rt_v & X_ADDR) < c->ram_size) &&
                               !(s1 & 1u) && !__atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE);
                if (arch_same || pop_new) {
                    u64 expected = rs_v;
                    bool changed = __atomic_compare_exchange_n((u64 *)RAM(w.gpa), &expected, rt_v,
                        false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
                    if (changed && pop_new &&
                        (__atomic_load_n(&mon_lock_word, __ATOMIC_ACQUIRE) != s1 ||
                         __atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE) ||
                         (__atomic_load_n(&page_info[fr_fast], __ATOMIC_ACQUIRE) & PI_NOTE))) {
                        mon_lock();
                        me->lock_site = 5;
                        pt_slot_changed_impl(w.gpa & ~7ULL, false, true, expected, rt_v);
                        mon_unlock();
                    }
                    if (rs != 31)
                        c->x[rs] = expected;
                    c->elr += 4;
                    c->stat_pt_writes++;
                    return;
                }
            } else if ((word & 0xFFE0FC00u) == 0xF8E08000u) { // 64-bit SWP
                u32 rs = (word >> 16) & 31, rt = word & 31;
                u64 rs_v = rs == 31 ? 0 : c->x[rs];
                u64 *slot8 = (u64 *)RAM(w.gpa);
                u64 peek = __atomic_load_n(slot8, __ATOMIC_ACQUIRE);
                if ((peek & X_P) && (((peek ^ rs_v) & X_ARCH_ALL) == 0)) {
                    u64 expected = peek;
                    if (__atomic_compare_exchange_n(slot8, &expected, rs_v, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
                        if (rt != 31)
                            c->x[rt] = peek;
                        c->elr += 4;
                        c->stat_pt_writes++;
                        return;
                    }
                }
            } else if ((word & 0xFF20CC00u) == 0xF8200000u) { // 64-bit LSE LDADD/LDCLR/LDEOR/LDSET
                u32 rs = (word >> 16) & 31, rt = word & 31, op = (word >> 12) & 3;
                u64 v = rs == 31 ? 0 : c->x[rs];
                if (op != 0 && !(v & X_ARCH_ALL)) {
                    u64 *slot8 = (u64 *)RAM(w.gpa);
                    u64 old = op == 1 ? __atomic_fetch_and(slot8, ~v, __ATOMIC_SEQ_CST) :
                              op == 2 ? __atomic_fetch_xor(slot8, v, __ATOMIC_SEQ_CST) :
                                        __atomic_fetch_or(slot8, v, __ATOMIC_SEQ_CST);
                    if (rt != 31)
                        c->x[rt] = old;
                    c->elr += 4;
                    c->stat_pt_writes++;
                    return;
                }
            }
        }
    }
    mon_lock();
    me->lock_site = 4;
    pool_reclaim();
    if (smp && !walk_still_valid(&w)) { // Fast 4-load validation; rewalk only if a guest table entry changed.
        walk_sample = mmu_walk_begin(true);
        w = x86_walk(c->cr3, far, access, true);
        mmu_walk_end(true, walk_sample);
        if (!w.ok || (!is_ram(w.gpa) && !is_direct(w.gpa))) {
            mon_unlock();
            return; // retry the access, which sorts out which of the above it is now
        }
    }
    u64 gpa = w.gpa & ~0xfffULL;
    u8 pi = page_info[gpa >> 12];
    if (write && (pi & PI_CODE)) {
        // A store to a frame holding translated code. Kill those translations, run the store from a one-off block so that later
        // instructions in the same block see the new bytes, and leave the page writable until code runs from it again.
        c->req_a[0] = gpa;
        c->req_a[1] = KILL_STORE;
        c->stat_smc++;
        __atomic_store_n(&page_info[gpa >> 12], pi & ~PI_CODE, __ATOMIC_RELEASE); // before the kill: see frame_unprotected
        t2_sync_on_trap();
        hostcall(HC_SMC);
        fex_unspill_state();
        reset_all_return_stacks();
        shadow_install(v, far, &w, true);
        tlbi_va(view_asid(c->view), far);
        if (smp)
            __atomic_store_n(&me->smc_held, gpa + 1, __ATOMIC_RELEASE);
        mon_unlock();
        c->elr = translate_here(rip_here(c->elr), 0, 0, true);
        return;
    }
    if (write && (pi & PI_PT) && !mirror_head[gpa >> 12]) { // no shadow mirrors it any more: an old table now holding data, which need not trap
        pi &= ~PI_PT;
        page_info[gpa >> 12] = pi;
        c->stat_pt_stale++;
    }
    if (write && (pi & PI_PT)) {
        me->lock_site = 5;
        if (in_rep_stub(c->elr)) { // 32 bytes at a time is more slots than the one-step path below resyncs: on with the plain loop, one element per store
            c->elr = c->x[R_LR];
            mon_unlock();
            return;
        }
        c->stat_pt_writes++;
        ArmAccess a;
        if (arm_decode(*(u32 *)c->elr, &a) && !a.load) { // a stage-1 fault has no register syndrome here, so decode the word
            u64 val = a.rt == 31 ? 0 : c->x[a.rt];
            u32 n = a.bytes;
            u64 ea = arm_address(*(u32 *)c->elr, &a);
            if (arm_splits(ea, &a)) {
                pt_split_store(ea, &a, val);
                mon_unlock();
                return;
            }
            // Observation only: other walkers can CAS A/D bits outside this lock. Equality does NOT prove a future
            // skipped write/zap would be safe. Restrict the extra read to existing aligned 4/8-byte RAM store forms.
            bool observe_plain = c->profile_mmu_observations && ((n == 8 && !(w.gpa & 7)) || (n == 4 && !(w.gpa & 3)));
            u64 observed_old = 0;
            if (observe_plain)
                observed_old = n == 8 ? *(u64 *)RAM(w.gpa) : *(u32 *)RAM(w.gpa);
            bool same64 = false, has_old_next = false;
            u64 old_val = 0;
            if (n == 8 && !(w.gpa & 7) && is_ram(w.gpa)) {
                // Preserve the actual guest store. Unlike a separate old-value
                // read, exchange sees an intervening lock-free walker A/D CAS.
                old_val = __atomic_exchange_n((u64 *)RAM(w.gpa), val, __ATOMIC_SEQ_CST);
                has_old_next = true;
                if (c->pte_same64) {
                    c->stat_pte_same64_exchanges++;
                    same64 = old_val == val;
                    if (same64)
                        c->stat_pte_same64_equal++;
                    else
                        c->stat_pte_same64_changed++;
                }
            } else if (n == 8)
                *(u64 *)RAM(w.gpa) = val;
            else if (n == 4 && !(w.gpa & 3)) // one piece: another vCPU can read the table while this runs
                *(u32 *)RAM(w.gpa) = (u32)val;
            else
                memcpy(RAM(w.gpa), &val, n);
            if (observe_plain)
                mmu_observe_store(n == 8 ? 1 : 0, observed_old, val);
            pt_slot_changed_impl(w.gpa & ~7ULL, same64, has_old_next, old_val, val);
            if (((w.gpa & 7) + n) > 8)
                pt_slot_changed((w.gpa & ~7ULL) + 8);
            arm_finish(&a, 0);
            mon_unlock();
            return;
        }
        // Compare-and-swap and swap are how kernels update live PTEs. Both
        // W and X forms can hit a tracked frame (including either PTE half).
        // Lock-free walkers update A/D outside mon_lock, so the memory operation
        // itself must be atomic; a locked load followed by a store can lose A/D.
        u32 word = *(u32 *)c->elr;
        bool is_cas = (word & 0xBFE0FC00u) == 0x88E0FC00u;
        bool is_swap = (word & 0xBFE0FC00u) == 0xB8E08000u;
        bool wide_atomic = (word >> 30) & 1;
        if (!(w.gpa & (wide_atomic ? 7 : 3)) && (is_cas || is_swap)) {
            u32 rs = (word >> 16) & 31, rt = word & 31;
            u64 rs_v = rs == 31 ? 0 : c->x[rs], rt_v = rt == 31 ? 0 : c->x[rt];
            u64 old, next;
            if (wide_atomic) {
                u64 expected = rs_v;
                bool changed = is_cas && __atomic_compare_exchange_n((u64 *)RAM(w.gpa), &expected, rt_v,
                    false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
                old = is_cas ? expected : __atomic_exchange_n((u64 *)RAM(w.gpa), rs_v, __ATOMIC_SEQ_CST);
                next = is_cas ? (changed ? rt_v : old) : rs_v;
            } else {
                u32 expected = (u32)rs_v;
                bool changed = is_cas && __atomic_compare_exchange_n((u32 *)RAM(w.gpa), &expected, (u32)rt_v,
                    false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
                old = is_cas ? expected : __atomic_exchange_n((u32 *)RAM(w.gpa), (u32)rs_v, __ATOMIC_SEQ_CST);
                next = is_cas ? (changed ? (u32)rt_v : old) : (u32)rs_v;
            }
            if (next != old)
                pt_slot_changed_impl(w.gpa & ~7ULL, false, wide_atomic, old, next);
            if (is_cas) {
                if (rs != 31)
                    c->x[rs] = old;
            } else if (rt != 31)
                c->x[rt] = old;
            c->elr += 4;
            mon_unlock();
            return;
        }
        // LOCK AND/OR/XOR/ADD as LSE. Guest stores trap, but the independent
        // A/D walkers do not: use the corresponding atomic RMW on the RAM alias.
        if ((word & 0xBF20CC00u) == 0xB8200000u && !(far & ((word >> 30) & 1 ? 7 : 3))) {
            bool wide = (word >> 30) & 1;
            u32 rs = (word >> 16) & 31, rt = word & 31, op = (word >> 12) & 3;
            u64 *slot8 = (u64 *)RAM(w.gpa);
            u32 *slot4 = (u32 *)RAM(w.gpa);
            u64 v = rs == 31 ? 0 : c->x[rs];
            if (!wide) v = (u32)v;
            u64 old;
            if (wide)
                old = op == 0 ? __atomic_fetch_add(slot8, v, __ATOMIC_SEQ_CST) :
                    op == 1 ? __atomic_fetch_and(slot8, ~v, __ATOMIC_SEQ_CST) :
                    op == 2 ? __atomic_fetch_xor(slot8, v, __ATOMIC_SEQ_CST) :
                              __atomic_fetch_or(slot8, v, __ATOMIC_SEQ_CST);
            else
                old = op == 0 ? __atomic_fetch_add(slot4, (u32)v, __ATOMIC_SEQ_CST) :
                    op == 1 ? __atomic_fetch_and(slot4, ~(u32)v, __ATOMIC_SEQ_CST) :
                    op == 2 ? __atomic_fetch_xor(slot4, (u32)v, __ATOMIC_SEQ_CST) :
                              __atomic_fetch_or(slot4, (u32)v, __ATOMIC_SEQ_CST);
            u64 next = op == 0 ? old + v : op == 1 ? old & ~v : op == 2 ? old ^ v : old | v;
            if (!wide) next = (u32)next;
            mmu_observe_store(wide ? 3 : 2, old, next);
            if (next != old)
                pt_slot_changed_impl(w.gpa & ~7ULL, false, wide, old, next);
            if (rt != 31)
                c->x[rt] = old;
            c->elr += 4;
            mon_unlock();
            return;
        }
        // Other atomics and pairs: let the instruction itself run on a writable mapping for exactly one step, then resync.
        if (pt_step_slot) { // split across two tracked frames, and one step opens one: close it, or the first stays writable and its later writes go unseen
            shadow_write_finish();
            step_off();
            stop(STOP_UNSUPPORTED, rip_here(c->elr), 5, far, esr);
        }
        if (smp)
            stop(STOP_UNSUPPORTED, rip_here(c->elr), 5, far, esr); // shortcut: one step opens the page to every vCPU in this view; the monitor should emulate these atomics before a guest can reach it
        c->stat_pt_step++;
        Walk w2 = w;
        w2.write = true;
        u64 *slot = shadow_slot(v, far, &w, 0, 0);
        u64 old = *slot;
        if ((old & 1) && (old & PTE_CONT)) {
            break_cont_group(slot, &desc[pool_index(slot)], true);
            old = *slot;
        }
        *slot = (w.gpa & ~0xfffULL) | PTE_VALID | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (old & PTE_X_MARK) | AP_RW;
        tlbi_va(view_asid(c->view), far);
        pt_step_slot = w.gpa | 1;
        pt_step_pte = slot;
        pt_step_va = far;
        step_on();
        mon_unlock();
        return;
    }
    bool perm = (dfsc & 0x3c) == 0x0c;
    me->tracked_new = false;
    u64 *installed = shadow_install(v, far, &w, write);
    if (smp && me->tracked_new) { // Protected now: drop what was read before it and let the access fault again, which reads the table again.
        u64 old = *installed;
        if (is_block_desc(old)) {
            if ((old & AP_RO) == AP_RW && (old & PTE_ADDR) < c->ram_size)
                rmap2m_remove(old & PTE_ADDR, installed);
            *installed = 0;
            tlbi_va(view_asid(c->view), far & ~0x1fffffULL);
        } else if ((old & 1) && (old & PTE_CONT)) {
            break_cont_group(installed, &desc[pool_index(installed)], true);
        } else {
            if ((old & 1) && (old & AP_RO) == AP_RW)
                rmap_remove(old & PTE_ADDR, installed);
            *installed = old & PTE_X_MARK;
            tlbi_va(view_asid(c->view), far);
        }
        mon_unlock();
        return;
    }
    if (perm) {
        c->stat_dirty++;
        tlbi_va(view_asid(c->view), is_block_desc(*installed) ? (far & ~0x1fffffULL) : far);
    } else {
        // Batch-prefetch up to 31 already-accessed adjacent 4KB entries in the same L0 shadow page.
        if (!is_block_desc(*installed) && !(pi & PI_DIRECT) && (w.leaf_level == 0 || w.leaf_level == 1)) {
            u64 *shadow_l0 = (u64 *)((u64)installed & ~0xfffULL);
            u32 idx = (u32)((far >> 12) & 0x1ffu);
            u32 win_start = idx & ~31u, win_end = win_start + 32u;
            if (!(c->hugepages_disabled & 2u)) {
                // Also try coalescing the sibling 64 KiB half of this 128 KiB window into PTE_CONT before 4 KiB prefetch.
                u32 other = (idx & ~15u) ^ 16u;
                if (!shadow_l0[other]) {
                    u64 other_va = (far & ~0x1fffffULL) | ((u64)other << 12);
                    if (w.leaf_level == 0) {
                        u64 *guest_l0 = (u64 *)((u64)w.leaf & ~0xfffULL);
                        u64 ge = __atomic_load_n(&guest_l0[other], __ATOMIC_RELAXED);
                        if ((ge & (X_P | X_A)) == (X_P | X_A) && (c->cpl != 3 || (ge & X_US)))
                            try_install_cont_64k(&w, other_va, &shadow_l0[other], ge & X_ADDR, false);
                    } else {
                        u64 other_gpa = (w.gpa & ~0x1fffffULL) + ((u64)other << 12);
                        try_install_cont_64k(&w, other_va, &shadow_l0[other], other_gpa, false);
                    }
                }
            }
            if (w.leaf_level == 0) {
                u64 *guest_l0 = (u64 *)((u64)w.leaf & ~0xfffULL);
                bool upper_write = ((w.read[3] & w.read[2] & w.read[1] & X_RW) != 0) ||
                                   (c->cpl != 3 && !(c->cr0 & (1ULL << 16)));
                for (u32 j = win_start; j < win_end; j++) {
                    if (j == idx || shadow_l0[j])
                        continue;
                    u64 ge = __atomic_load_n(&guest_l0[j], __ATOMIC_RELAXED);
                    if ((ge & (X_P | X_A)) != (X_P | X_A))
                        continue;
                    if (c->cpl == 3 && !(ge & X_US))
                        continue;
                    u64 ngpa = ge & X_ADDR;
                    if (ngpa >= c->ram_size)
                        continue;
                    u8 npi = page_info[ngpa >> 12];
                    if (!(npi & PI_RAM))
                        continue;
                    bool nwritable = upper_write && (ge & X_RW) && (ge & X_D) && !(npi & (PI_CODE | PI_PT));
                    if (nwritable && !(npi & PI_DIRECT) && !rmap_free && rmap_next >= RMAP_NODES - 64)
                        nwritable = false;
                    shadow_l0[j] = ngpa | PTE_VALID | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (nwritable ? AP_RW : AP_RO);
                    if (nwritable && !(npi & PI_DIRECT))
                        rmap_add(ngpa, &shadow_l0[j]);
                }
            } else {
                u64 base_gpa = w.gpa & ~0x1fffffULL;
                bool lw = w.write && w.dirty;
                for (u32 j = win_start; j < win_end; j++) {
                    if (j == idx || shadow_l0[j])
                        continue;
                    u64 ngpa = base_gpa + ((u64)j << 12);
                    if (ngpa >= c->ram_size)
                        continue;
                    u8 npi = page_info[ngpa >> 12];
                    if (!(npi & PI_RAM))
                        continue;
                    bool nwritable = lw && !(npi & (PI_CODE | PI_PT));
                    if (nwritable && !(npi & PI_DIRECT) && !rmap_free && rmap_next >= RMAP_NODES - 64)
                        nwritable = false;
                    shadow_l0[j] = ngpa | PTE_VALID | PTE_AF | PTE_SH | PTE_NG | PTE_PXN | PTE_UXN | (nwritable ? AP_RW : AP_RO);
                    if (nwritable && !(npi & PI_DIRECT))
                        rmap_add(ngpa, &shadow_l0[j]);
                }
            }
        }
        c->stat_fills++;
        __asm__ volatile("dsb ishst");
    }
    mon_unlock();
}

static void shadow_write_finish(void) {
    u64 gpa = pt_step_slot & ~1ULL;
    pt_step_slot = 0;
    u64 e = *pt_step_pte;
    *pt_step_pte = (e & ~AP_RW) | AP_RO;
    tlbi_va(view_asid(c->view), pt_step_va);
    for (int i = 0; i < 3; i++) { // the slot, and any further slots a 16-byte store reached
        u64 g = (gpa & ~7ULL) + (u64)i * 8;
        if ((g >> 12) == (gpa >> 12))
            pt_slot_changed(g);
    }
}

// ---- Event delivery ----
typedef struct {
    u32 lo, hi;
} Desc32;

// XNU's 32-bit user selectors are in the LDT.
static bool desc_read(u32 sel, u64 *out) {
    u32 idx = sel & ~7u;
    if (idx + 7 > ((sel & 4) ? c->ldt_limit : c->gdtr_limit))
        return false;
    return guest_read(((sel & 4) ? c->ldt_base : c->gdtr_base) + idx, out, 8, 0);
}

static inline u64 desc_base(u64 d) { return ((d >> 16) & 0xffffff) | ((d >> 32) & 0xff000000ULL); }
static inline u64 desc_limit(u64 d) {
    u64 limit = (d & 0xffff) | ((d >> 32) & 0xf0000);
    return (d >> 55) & 1 ? limit << 12 | 0xfff : limit;
}
static inline u32 desc_dpl(u64 d) { return (u32)(d >> 45) & 3; }
static inline bool desc_flat(u64 d) { return !desc_base(d) && desc_limit(d) == 0xffffffffULL; }
#define DESC_P (1ULL << 47)
#define DESC_S (1ULL << 44)
#define DESC_CODE (1ULL << 43)
#define DESC_CONFORMING (1ULL << 42) // expand-down in a data segment
#define DESC_RW (1ULL << 41) // readable code, writable data
#define DESC_L (1ULL << 53)
#define DESC_D (1ULL << 54)
#define SEG_ES 0
#define SEG_SS 2
#define SEG_DS 3
#define SEG_FS 4
#define SEG_GS 5

// Translated 32-bit code uses es, ds, cs and ss as flat, and fs and gs by base alone, with no limit check: XNU's cthread segment is 8 KB and its users stay inside.
// A register holding anything else sets its bit in seg_bad, and while one is set compatibility mode runs on TCG, which also does the nulling of ring-0 selectors on
// the way out to ring 3. Noted at the load, as a CPU caches a descriptor, and from CPUX86State at each hand-back: hvdbt_seg_bad in hvdbt.c has to agree.
static void seg_note(u32 sr, u64 d, u16 selector) {
    u32 mask=(1u<<sr)|(1u<<(sr+8));
    c->seg_bad=(c->seg_bad&~mask)|engine_compat_segment_bad(sr,selector,(u32)(d>>32),desc_base(d),desc_limit(d));
}

static void set_mode(u32 cpl, bool compat) {
    if (compat) // what every 32-bit emitter counts on. The SDM leaves these undefined across the switch.
        for (int i = 0; i < 8; i++)
            c->x[i] = (u32)c->x[i];
    if (c->cpl == cpl && c->compat == compat)
        return;
    c->cpl = cpl;
    c->compat = compat;
    view_switch(cr3_key(c->cr3), view_mode());
}

// Whether entering compatibility mode with this CS and SS can stay here. False: it raised #GP or #SS, or it never returns because TCG takes the instruction at `here`.
static bool compat_entry_ok(u64 cd, u32 ss, u32 cpl, u64 here) {
    if ((cpl != 0 && cpl != 3) || !(cd & DESC_D) || !desc_flat(cd) || engine_compat_segments_bad(c->seg_bad,cpl))
        stop(STOP_UNSUPPORTED, here, 8, ss, cd);
    u64 sd;
    if (!(ss & ~3u)) {
        raise_gp(0, here);
        return false;
    }
    if (!desc_read(ss, &sd) || (ss & 3) != cpl || desc_dpl(sd) != cpl || !(sd & DESC_S) || (sd & DESC_CODE) || !(sd & DESC_RW)) {
        raise_gp(ss & ~3u, here);
        return false;
    }
    if (!(sd & DESC_P)) {
        deliver(12, ss & ~3u, true, here, false);
        return false;
    }
    if (!desc_flat(sd) || !(sd & DESC_D) || (sd & DESC_CONFORMING))
        stop(STOP_UNSUPPORTED, here, 8, ss, sd);
    return true;
}

static void fex_unspill_state(void) {
    if (c->x[R_FS] != (u64)c + 32768ULL)
        return;
    static const u8 fex_sra[16] = {4, 7, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 29};
    u64 g[16];
    for (int i = 0; i < 16; i++)
        g[i] = c->x[fex_sra[i]];
    for (int i = 0; i < 16; i++)
        c->x[i] = g[i];
    c->x[R_PF] = c->x[26] & 0xffULL;
    c->x[R_AF] = c->x[27] & 0xffULL;
    c->df = (i64)(*(const signed char *)((const u8 *)c + 32768 + 0x3fa));
    u64 fex_xmm_lo[8][2];
    for (int i = 0; i < 8; i++) {
        fex_xmm_lo[i][0] = c->ymm_hi[i][0];
        fex_xmm_lo[i][1] = c->ymm_hi[i][1];
    }
    const u64 *avx_hi = (const u64 *)((const u8 *)c + 32768 + 0xc0);
    for (int i = 0; i < 16; i++) {
        c->ymm_hi[i][0] = avx_hi[i * 2];
        c->ymm_hi[i][1] = avx_hi[i * 2 + 1];
    }
    __asm__ volatile(
        ".arch armv8-a+fp+simd\n"
        "ldr q0, [%0, #0]\n"
        "ldr q1, [%0, #16]\n"
        "ldr q2, [%0, #32]\n"
        "ldr q3, [%0, #48]\n"
        "ldr q4, [%0, #64]\n"
        "ldr q5, [%0, #80]\n"
        "ldr q6, [%0, #96]\n"
        "ldr q7, [%0, #112]\n"
        "mov v8.16b, v24.16b\n"
        "mov v9.16b, v25.16b\n"
        "mov v10.16b, v26.16b\n"
        "mov v11.16b, v27.16b\n"
        "mov v12.16b, v28.16b\n"
        "mov v13.16b, v29.16b\n"
        "mov v14.16b, v30.16b\n"
        "mov v15.16b, v31.16b\n"
        : : "r"(fex_xmm_lo) : "memory");
    const u64 *saved = (const u64 *)((const u8 *)c + 36864);
    c->x[19] = saved[0];
    c->x[26] = c->rs_top;
    *(u64 *)c->rs_top = ~0ULL;
    c->x[27] = saved[2];
    c->x[28] = saved[3];
    c->x[29] = saved[4];
    c->x[30] = saved[5];
}

static void deliver(u32 vector, u32 err, bool has_err, u64 ret_rip, bool sw_int) {
    t2_sync_on_trap();
    bool was_fex = (c->x[R_FS] == (u64)c + 32768ULL);
    fex_unspill_state();
    if (!c->model_kernel) {
        shim_fault(vector, err, ret_rip, c->far);
        return;
    }
    c->irq_shadow_rip = ~0ULL;
    me->shadow_stepping = false;
    event_delivered = true;
    c->stat_vector[vector & 255]++;
    c->vector_last_rip[vector & 255] = ret_rip;
    c->vector_last_err[vector & 255] = err;
    c->vector_last_cr3[vector & 255] = c->cr3;
    note_event('E', ret_rip, err, vector);
    if (c->hybrid && c->cpl == 3 && vector < 32 && vector != 7) {
        if (vector == 14) {
            EngineEvent *e = &c->user_pf[c->user_pf_next++ % 32];
            e->rip = ret_rip;
            e->a = c->cr2;
            e->vector = err;
            e->cr3 = c->cr3;
            if (c->cr2 < 0x10000ULL || (c->cr2 >> 47) != 0) {
                c->req_a[0] = 4;
                c->req_a[1] = ret_rip;
                c->req_a[2] = c->cr2;
                c->req_a[3] = err;
                c->req_a[4] = was_fex ? 1 : 0;
                hostcall(HC_NOTE);
            }
        } else if (vector < 20) {
            c->req_a[0] = 3;
            c->req_a[1] = ret_rip;
            c->req_a[2] = vector;
            c->req_a[3] = err;
            c->req_a[4] = was_fex ? 1 : 0;
            hostcall(HC_NOTE);
        }
    }
    if (deliver_depth >= 3)
        stop(STOP_TRIPLE_FAULT, ret_rip, vector, err, c->cr2);
    deliver_depth++;
    u64 gate[2];
    if ((vector * 16 + 15) > c->idtr_limit || !guest_read(c->idtr_base + vector * 16, gate, 16, 0)) {
        deliver(vector == 8 ? 8 : 8, 0, true, ret_rip, false); // an unreadable or short IDT goes to #DF, then triple
        deliver_depth--;
        return;
    }
    u32 attr = (u32)(gate[0] >> 32);
    u32 type = (attr >> 8) & 15, dpl = (attr >> 13) & 3, ist = attr & 7;
    if (!(attr & 0x8000) || (type != 14 && type != 15)) {
        deliver(13, vector * 8 + 2, true, ret_rip, false);
        deliver_depth--;
        return;
    }
    if (sw_int && dpl < c->cpl) {
        deliver(13, vector * 8 + 2, true, ret_rip, false);
        deliver_depth--;
        return;
    }
    u64 target = (gate[0] & 0xffff) | ((gate[0] >> 32) & 0xffff0000ULL) | (gate[1] << 32);
    u32 sel = (u32)(gate[0] >> 16) & 0xffff;
    u64 cdesc;
    if (!desc_read(sel, &cdesc) || !(cdesc & (1ULL << 47)) || !(cdesc & (1ULL << 53))) {
        deliver(13, sel & ~3u, true, ret_rip, false);
        deliver_depth--;
        return;
    }
    u32 new_cpl = (u32)(cdesc >> 45) & 3;
    u64 rsp = c->x[4], old_rsp = rsp;
    u64 rflags = rflags_get();
    u32 old_cs = c->seg_sel[1], old_ss = c->seg_sel[2];
    bool switched = new_cpl < c->cpl;
    if (new_cpl > c->cpl) {
        deliver(13, sel & ~3u, true, ret_rip, false);
        deliver_depth--;
        return;
    }
    if (switched || ist) {
        u32 off = ist ? 36 + 8 * (ist - 1) : 4 + 8 * new_cpl;
        if (off + 7 > c->tss_limit || !guest_read(c->tss_base + off, &rsp, 8, 0)) {
            deliver(10, c->tr & ~3u, true, ret_rip, false);
            deliver_depth--;
            return;
        }
    }
    rsp &= ~0xFULL;
    u64 frame[6];
    int n = 0;
    if (has_err)
        frame[n++] = err;
    frame[n++] = ret_rip;
    frame[n++] = old_cs;
    frame[n++] = rflags;
    frame[n++] = old_rsp;
    frame[n++] = old_ss;
    rsp -= (u64)n * 8;
    // Frames are written with supervisor rights on the handler's stack. If that faults, the event becomes a double fault.
    if (!guest_write(rsp, frame, (u32)n * 8, 0)) {
        if (vector == 8)
            stop(STOP_TRIPLE_FAULT, ret_rip, vector, err, rsp);
        c->cr2 = rsp;
        deliver(8, 0, true, ret_rip, false);
        deliver_depth--;
        return;
    }
    c->seg_sel[1] = (u16)((sel & ~3u) | new_cpl);
    if (switched)
        c->seg_sel[2] = (u16)new_cpl;
    c->x[4] = rsp;
    u64 keep = rflags & ~(RF_TF | RF_NT | RF_RF | RF_VM);
    if (type == 14)
        keep &= ~RF_IF;
    rflags_set(keep);
    set_mode(new_cpl, false);
    c->halted = 0;
    deliver_depth--;
    c->elr = translate_here(target, 0, 0, false);
}

static void irq_deliver_now(u64 rip, u32 apic_vector) {
    u32 vector;
    if (apic_vector) {
        vector = apic_vector;
        c->apic_irr[vector / 32] &= ~(1u << (vector & 31));
        c->apic_isr[vector / 32] |= 1u << (vector & 31);
        if (c->apic_host_level[vector / 32] & (1u << (vector & 31)))
            c->apic_tmr[vector / 32] |= 1u << (vector & 31);
    } else {
        vector = c->irq_pending - 1;
        c->irq_pending = 0;
    }
    c->stat_irq++;
    deliver(vector, 0, false, rip, false);
}

// Frames written behind the monitor's back. Same treatment as a store that traps, except that nothing has to be re-run.
static void note_apply(u32 fr) {
    u8 pi = __atomic_fetch_and(&page_info[fr], (u8)~PI_NOTE, __ATOMIC_ACQ_REL);
    if (pi & PI_CODE) {
        c->req_a[0] = (u64)fr << 12;
        c->req_a[1] = KILL_NOTE;
        c->stat_note_code++;
        __atomic_fetch_and(&page_info[fr], (u8)~PI_CODE, __ATOMIC_RELEASE); // before the kill: see frame_unprotected
        hostcall(HC_SMC);
        reset_all_return_stacks();
    }
    if (pi & PI_PT) {
        c->stat_note_pt++;
        for (u32 i = 0; i < 512; i++)
            pt_slot_changed(((u64)fr << 12) + i * 8);
    }
}

// One ring for the VM, in the first context, and any vCPU may drain it. A note is only looked at on the way out of the monitor, and what reaches the monitor first is not
// a guest that has acted on the write: that takes an interrupt or a serialising instruction, which come through here.
static void notes_drain(void) {
    if (__atomic_load_n(&c0->note_head, __ATOMIC_ACQUIRE) == __atomic_load_n(&c0->note_tail, __ATOMIC_ACQUIRE) &&
        !__atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE))
        return;
    u64 observed_at = c->profile_mmu_observations ? RSR(cntvct_el0) : 0;
    mon_lock();
    // Overflow: rescan. The write came before the note, so a producer still storing its slot is covered, and its note, seen later, costs nothing.
    if (__atomic_load_n(&c0->note_overflow, __ATOMIC_ACQUIRE)) {
        u64 before_code=c->stat_note_code, before_tables=c->stat_note_pt;
        if(observed_at)c->note_rescans.active_start_ticks=observed_at+c->tsc_counter_offset;
        __atomic_store_n(&c0->note_overflow, 0, __ATOMIC_RELEASE);
        for (u32 fr = 0; fr < c->ram_size >> 12; fr++)
            if (__atomic_load_n(&page_info[fr], __ATOMIC_ACQUIRE) & PI_NOTE)
                note_apply(fr);
        __atomic_store_n(&c0->note_tail, __atomic_load_n(&c0->note_head, __ATOMIC_ACQUIRE), __ATOMIC_RELEASE);
        mon_unlock();
        if(observed_at)engine_note_rescan_record(&c->note_rescans,
            observed_at+c->tsc_counter_offset,RSR(cntvct_el0)-observed_at,
            c->stat_note_code-before_code,c->stat_note_pt-before_tables);
        return;
    }
    u32 head = __atomic_load_n(&c0->note_head, __ATOMIC_ACQUIRE);
    while (c0->note_tail != head) {
        u32 v = __atomic_exchange_n(&c0->note_ring[c0->note_tail % 256], 0, __ATOMIC_ACQ_REL);
        if (!v)
            break; // reserved, not stored yet: its kick comes after the store
        note_apply(v - 1);
        __atomic_store_n(&c0->note_tail, c0->note_tail + 1, __ATOMIC_RELEASE);
    }
    mon_unlock();
}

static void fp_trap_sync(void);
static void mxcsr_apply(u32 v);

// The host may have run instructions: anything the monitor derives from x86 state is stale.
static void host_resync(void) {
    mxcsr_apply(c->resume_mxcsr);
    fp_trap_sync();
    if (c->compat) // TCG leaves the top halves as it found them
        for (int i = 0; i < 8; i++)
            c->x[i] = (u32)c->x[i];
    view_switch(cr3_key(c->cr3), view_mode());
    c->irq_shadow_rip = ~0ULL;
    me->shadow_stepping = false;
    deliver_depth = 0;
    if (c->tr_flushed) { // the code cache was reset while this vCPU was out: its return stacks hold old addresses
        c->tr_flushed = 0;
        code_cache_reset = true;
        reset_all_return_stacks();
    }
    c->elr = translate_here(c->req_r[0], 0, 0, false);
}

static void sync_host_irq(bool can_acknowledge) {
    c->req_a[0] = can_acknowledge;
    hostcall(HC_IRQ);
    if (c->req_r[0] != ~0ULL)
        c->irq_pending = (u32)c->req_r[0] + 1;
}

static void exit_check(bool after_sync_progress, bool refresh_host_irq);

// Reached through mon_restart with the monitor's stack reset: a stop abandons whatever called it.
void mon_after_stop(void) {
    if (mon_resync) {
        mon_resync = false;
        host_resync();
    }
    deliver_depth = 0;
    exit_check(false, false);
}

// Called on every way out of the monitor. If an interrupt is due and the pc is at an x86 boundary, take it. Otherwise step ARM
// instructions until it is: one trap per ARM word, only while an interrupt waits. Costs nothing when none does.
static void exit_check(bool after_sync_progress, bool refresh_host_irq) {
    stack_check();
again:
    if (__atomic_load_n(&rs_epoch_shared, __ATOMIC_RELAXED) != me->rs_epoch) { // shortcut: one load, and at one vCPU it never differs
        me->rs_epoch = __atomic_load_n(&rs_epoch_shared, __ATOMIC_RELAXED);
        reset_my_return_stacks();
    }
    step_off();
    if (pt_step_slot) {
        if (refresh_host_irq)
            sync_host_irq(false);
        step_on();
        return;
    }
    if (c->direct_flush) {
        mon_lock();
        c->direct_flush = 0;
        zap_why = KILL_DEVICE_MOVED;
        shadow_drop_all();
        mon_unlock();
    }
    if (c->hybrid)
        notes_drain();
    bool stop_now = c->stop_requested;
    u32 apic_vector = 0;
    bool if_open = c->rflags_sys & RF_IF;
    if (if_open)
        apic_vector = apic_ready_vector();
    bool waiting = if_open && (apic_vector || c->irq_pending || (c->hybrid && c->irq_line));
    if (!stop_now && !waiting && !me->shadow_stepping && !(if_open && refresh_host_irq)) {
        if (refresh_host_irq)
            sync_host_irq(false);
        return; // none waiting, or masked: sti, popf and iret come back through here
    }
    bool exact, found;
    u64 rip;
    const bool in_fex = c->x[R_FS] == (u64)c + 32768ULL;
    if (in_fex) {
        // The shared FEX body entry runs before any guest instruction. At
        // other ARM PCs an instruction can still have live JIT temporaries.
        const BlockEntry *block = find_block(c->elr);
        found = block != 0;
        exact = found && c->elr == PRIV_VA + OFF_CC + 4ULL * block->arm_off + 12 &&
                *(const u32 *)(c->elr - 8) == 0x10000050u;
        rip = exact ? block->rip : 0;
    } else {
        rip = rip_of_pc(c->elr, &exact, &found);
    }
    if (me->shadow_stepping) {
        if (me->shadow_step_rip != c->irq_shadow_rip) {
            me->shadow_stepping = false; // the protected instruction established a new shadow
        } else if (after_sync_progress && found && exact) {
            c->irq_shadow_rip = ~0ULL;
            me->shadow_stepping = false;
        }
    }
    // Do not dequeue an interrupt before a complete x86 boundary. In the
    // shadowed fallback case the pending vector must stay visible to TCG.
    bool can_acknowledge = !stop_now && if_open && found && exact && rip != c->irq_shadow_rip;
    if (refresh_host_irq || (c->hybrid && can_acknowledge && !c->irq_pending && c->irq_line)) {
        sync_host_irq(can_acknowledge);
        refresh_host_irq = false;
        stop_now = c->stop_requested;
    }
    waiting = if_open && (apic_vector || c->irq_pending || (c->hybrid && c->irq_line));
    if (!stop_now && !waiting)
        return;
    if (found && exact) {
        if (stop_now && rip != c->irq_shadow_rip) { // the host wants the vCPU at an x86 boundary, with all state in place. Not inside sti's shadow, which resuming would lose: the next boundary will do
            c->stop_requested = 0;
            c->req_a[0] = rip;
            c->req_r[0] = rip;
            bool from_t2 = t2_sync_on_trap();
            if (in_fex)
                fex_unspill_state();
            x87_flush_fast_cache();
            hostcall(HC_BOUNDARY);
            c->fp_d_valid = 0;
            if (c->hybrid)
                host_resync();
            else if (in_fex || from_t2 || c->req_r[0] != rip || c->tr_flushed) { // restart through native state after unspilling, leaving T2, moving RIP, or resetting the cache
                event_delivered = false;
                c->elr = translate_here(c->req_r[0], 0, 0, false);
            }
            after_sync_progress = false;
            goto again;
        }
        if (rip != c->irq_shadow_rip) {
            if (!waiting || (!apic_vector && !c->irq_pending))
                return;
            irq_deliver_now(rip, apic_vector);
            after_sync_progress = false;
            goto again;
        }
        me->shadow_stepping = true; // the instruction after sti runs first
        me->shadow_step_rip = rip;
    }
    c->stat_steps++;
    step_on();
}

// ---- SVC dispatch ----
static inline u64 *lit(u64 elr) { return (u64 *)elr; }

static inline u64 next_rip_lit(void) { return *lit(c->elr + 4); } // the fall-through stub's literal after the svc

static void x86_ud(void) {
    u64 rip = rip_here(c->elr - 4);
    deliver(6, 0, false, rip, false);
}
static void x86_gp0(void) { raise_gp(0, rip_here(c->elr - 4)); }


static void do_syscall(void) {
    u64 next = next_rip_lit();
    if (!c->model_kernel) {
        // User-mode shim: the host plays the kernel. rcx and r11 are clobbered as the instruction defines.
        c->x[1] = next;
        c->x[11] = rflags_get();
        c->req_a[0] = next;
        hostcall(HC_SYSCALL);
        return;
    }
    if (!(c->efer & 1)) {
        x86_ud();
        return;
    }
    c->stat_syscall++;
    note_event('S', next, c->x[0], 0);
    note_process_syscall(next);
    c->x[1] = next;
    u64 fl = rflags_get();
    c->x[11] = fl;
    rflags_set((fl & ~c->sfmask) & ~RF_RF);
    u16 cs = (u16)((c->star >> 32) & 0xfffc);
    c->seg_sel[1] = cs;
    c->seg_sel[2] = (u16)(cs + 8);
    set_mode(0, false);
    c->elr = translate_here(c->lstar, 0, 0, false);
}

// The 32-bit fast path XNU's libsystem takes. The hidden parts of CS and SS come from the instruction, not a table: hvdbt.c forces CS's size bits to match.
static void do_sysenter(void) {
    u64 here = rip_here(c->elr - 4);
    u16 cs = (u16)(c->sysenter_cs & 0xfffc);
    if (!cs) {
        raise_gp(0, here);
        return;
    }
    note_event('S', here, c->x[0], 1);
    c->rflags_sys &= ~(RF_IF | RF_RF | RF_VM);
    c->seg_sel[1] = cs;
    c->seg_sel[2] = (u16)(cs + 8);
    c->x[4] = c->sysenter_esp;
    set_mode(0, false);
    c->elr = translate_here(c->sysenter_eip, 0, 0, false);
}

// From a 64-bit kernel, normally to 32-bit code. The selectors follow from IA32_SYSENTER_CS, and no table is read.
static void do_sysexit(void) {
    u64 here = rip_here(c->elr - 4);
    bool op64 = c->x[R_S0];
    u16 cs = (u16)c->sysenter_cs;
    if (!(cs & 0xfffc)) {
        raise_gp(0, here);
        return;
    }
    if (!op64 && (c->seg_bad & (1u << SEG_ES | 1u << SEG_DS | 1u << SEG_FS | 1u << SEG_GS)))
        stop(STOP_UNSUPPORTED, here, 8, cs, 0);
    u64 rip = op64 ? c->x[2] : (u32)c->x[2], rsp = op64 ? c->x[1] : (u32)c->x[1];
    if (!canonical(rip)) {
        raise_gp(0, here);
        return;
    }
    note_event('R', rip, c->x[0], 1);
    note_wait_returned(rip, rflags_get());
    c->seg_sel[1] = (u16)(((cs + (op64 ? 32 : 16)) & 0xfffc) | 3);
    c->seg_sel[2] = (u16)(((cs + (op64 ? 40 : 24)) & 0xfffc) | 3);
    c->x[4] = rsp;
    set_mode(3, !op64);
    c->elr = translate_here(rip, 0, 0, false);
}

static void do_sysret(void) {
    u64 rip = c->x[1];
    u64 op64 = c->x[R_S0];
    if (!(c->efer & 1)) {
        x86_ud();
        return;
    }
    if (!op64 && (c->seg_bad & (1u << SEG_ES | 1u << SEG_DS | 1u << SEG_FS | 1u << SEG_GS)))
        stop(STOP_UNSUPPORTED, rip_here(c->elr - 4), 8, 0, 0);
    if (!op64)
        rip = (u32)rip;
    if (!canonical(rip)) {
        x86_gp0();
        return;
    }
    note_event('R', rip, c->x[0], 0);
    note_wait_returned(rip, c->x[11]);
    u64 fl = (c->x[11] & 0x3C7FD7ULL) | RFLAGS_FIXED;
    rflags_set(fl);
    u16 base = (u16)((c->star >> 48) & 0xffff);
    c->seg_sel[1] = (u16)((op64 ? base + 16 : base) | 3);
    c->seg_sel[2] = (u16)((base + 8) | 3);
    set_mode(3, !op64);
    c->elr = translate_here(rip, 0, 0, false);
}

static void do_iret(void) {
    u64 here = rip_here(c->elr - 4);
    if (c->compat && c->x[R_S0] == 4) { // 32-bit code's own: no change of ring or mode here, or TCG has it
        // XNU normally returns through its 64-bit interrupt trampoline.
        // Leave supervisor IRETD's privilege-changing forms to TCG rather
        // than interpreting their stack as the ring3-only frame below.
        if(c->cpl!=3)stop(STOP_UNSUPPORTED,here,7,4,0);
        u32 f[3];
        if (!guest_read(c->x[4], f, 12, 4)) {
            Walk w = x86_walk(c->cr3, c->x[4], 4, false);
            if (w.ok) // the second page of the frame
                w = x86_walk(c->cr3, (c->x[4] | 0xfff) + 1, 4, false), raise_pf((c->x[4] | 0xfff) + 1, w.fault_err, here);
            else
                raise_pf(c->x[4], w.fault_err, here);
            return;
        }
        u32 cs = f[1] & 0xffff;
        u64 cd;
        if (!(cs & ~3u) || !desc_read(cs, &cd) || !(cd & DESC_S) || !(cd & DESC_CODE) || (cs & 3) != 3 || desc_dpl(cd) != 3 || !(cd & DESC_P)) {
            raise_gp(cs & ~3u, here);
            return;
        }
        if ((cd & DESC_L) || !(cd & DESC_D) || !desc_flat(cd) || (f[2] & RF_VM))
            stop(STOP_UNSUPPORTED, here, 7, cs, cd);
        u64 cur = rflags_get(), mask = RF_ARITH | RF_DF | RF_TF | RF_NT | RF_RF | RF_AC | (1ULL << 21);
        if (((cur >> 12) & 3) == 3)
            mask |= RF_IF;
        u64 fl = (cur & ~mask) | (f[2] & mask) | RFLAGS_FIXED;
        if (fl & RF_TF)
            stop(STOP_UNSUPPORTED, here, 9, fl, 0);
        rflags_set(fl);
        c->x[4] = (u32)(c->x[4] + 12);
        c->seg_sel[1] = (u16)cs;
        c->elr = translate_here(f[0], 0, 0, false);
        return;
    }
    if (c->x[R_S0] != 8)
        stop(STOP_UNSUPPORTED, here, 7, c->x[R_S0], 0); // 16 or 32-bit iret
    c->stat_iret++;
    u64 f[5];
    if (!guest_read(c->x[4], f, 40, c->cpl == 3 ? 4 : 0)) {
        Walk w = x86_walk(c->cr3, c->x[4], c->cpl == 3 ? 4 : 0, false);
        raise_pf(c->x[4], w.fault_err, here);
        return;
    }
    u64 rip = f[0], rfl = f[2], rsp = f[3];
    note_event('I', here, rip, (u32)f[1] & 0xffff);
    u32 cs = (u32)f[1] & 0xffff, ss = (u32)f[4] & 0xffff;
    u32 new_cpl = cs & 3;
    if (new_cpl == 3)
        note_wait_returned(rip, rfl);
    if (new_cpl < c->cpl) {
        raise_gp(cs & ~3u, here);
        return;
    }
    u64 cd;
    if (!desc_read(cs, &cd) || !(cd & (1ULL << 47)) || !(cd & (1ULL << 44)) || !(cd & (1ULL << 43))) {
        raise_gp(cs & ~3u, here);
        return;
    }
    bool to_compat = !(cd & DESC_L);
    if (to_compat) {
        if (!compat_entry_ok(cd, ss, new_cpl, here))
            return;
        rip = (u32)rip;
    }
    if (!canonical(rip)) {
        raise_gp(0, here);
        return;
    }
    u64 mask = RF_ARITH | RF_DF | RF_TF | RF_NT | RF_RF | RF_AC | (1ULL << 21);
    u64 cur = rflags_get();
    u32 iopl = (u32)(cur >> 12) & 3;
    if (c->cpl == 0)
        mask |= RF_IOPL | RF_IF;
    else if (c->cpl <= iopl)
        mask |= RF_IF;
    u64 fl = (cur & ~mask) | (rfl & mask) | RFLAGS_FIXED;
    if (fl & RF_TF)
        stop(STOP_UNSUPPORTED, here, 9, fl, 0); // single-step trap flag
    rflags_set(fl);
    c->x[4] = rsp;
    c->seg_sel[1] = (u16)cs;
    c->seg_sel[2] = (u16)ss;
    if (new_cpl > c->cpl) {
        // Intel SDM IRET / QEMU validate_seg: returning to an outer privilege level clears
        // non-conforming ES/DS/FS/GS selectors with DPL < new_cpl, preserving 64-bit FS/GS bases.
        static const u8 dsregs[4] = {SEG_ES, SEG_DS, SEG_FS, SEG_GS};
        for (int i = 0; i < 4; i++) {
            u32 sr = dsregs[i];
            u16 s = c->seg_sel[sr];
            if (!(s & ~3u))
                continue;
            u64 d = 0;
            bool ok = desc_read(s, &d);
            bool conf_code = ok && (d & DESC_S) && (d & DESC_CODE) && (d & DESC_CONFORMING);
            if (!ok || !(d & DESC_S) || (!conf_code && desc_dpl(d) < new_cpl)) {
                c->seg_sel[sr] = 0;
                seg_note(sr, 0, 0);
            }
        }
    }
    set_mode(new_cpl, to_compat);
    c->elr = translate_here(rip, 0, 0, false);
}

// The page of [va, va + n) that refused the access, as the fault it is.
static void raise_range_fault(u64 va, u32 n, u32 access, u64 here) {
    Walk w = x86_walk(c->cr3, va, access, false);
    if (w.ok) {
        va = (va | 0xfff) + 1;
        w = x86_walk(c->cr3, va, access, false);
    }
    if (w.ok)
        stop(STOP_UNSUPPORTED, here, 19, va, n);
    if (w.fault_err == ~0u)
        raise_gp(0, here);
    else
        raise_pf(va, w.fault_err, here);
}

// Same-privilege ring 3 transfers and flat ring 0 jumps into/out of compatibility
// mode. Ring 0 calls/returns, gates, other rings and 16-bit forms remain on TCG.
static void do_far(void) {
    u64 here = rip_here(c->elr - 4), off = c->x[R_S0], arg = c->x[R_S2], sp = c->x[4];
    u32 kind = arg & 3, opsz = (arg >> 8) & 15, imm = (u32)(arg >> 16) & 0xffff, sel = (u32)c->x[R_S1] & 0xffff, access = c->cpl == 3 ? 4 : 0;
    u64 cd;
    u8 mem[16];
    u32 privilege = c->cpl;
    if (opsz == 2 || (privilege != 3 && !(privilege == 0 && kind == 2)))
        stop(STOP_UNSUPPORTED, here, 18, sel, opsz);
    if (kind == 0) { // retf: selector and offset from the stack, which moves last
        if (!guest_read(sp, mem, opsz * 2, access)) {
            raise_range_fault(sp, opsz * 2, access, here);
            return;
        }
        off = 0;
        memcpy(&off, mem, opsz);
        sel = *(u16 *)(mem + opsz);
    }
    if (!(sel & ~3u)) {
        raise_gp(0, here);
        return;
    }
    if (!desc_read(sel, &cd)) {
        raise_gp(sel & ~3u, here);
        return;
    }
    if (!(cd & DESC_S))
        stop(STOP_UNSUPPORTED, here, 18, sel, cd); // a call gate
    u32 rpl = sel & 3, dpl = desc_dpl(cd);
    bool conforming = cd & DESC_CONFORMING;
    if (privilege == 0 && conforming)
        stop(STOP_UNSUPPORTED, here, 18, sel, cd);
    bool allowed = privilege == 0 ? rpl == 0 && dpl == 0 :
        kind == 0 ? rpl == 3 && (conforming ? dpl <= 3 : dpl == 3) : (conforming ? dpl <= 3 : dpl == 3);
    if (!(cd & DESC_CODE) || !allowed) {
        raise_gp(sel & ~3u, here);
        return;
    }
    if (!(cd & DESC_P)) {
        deliver(11, sel & ~3u, true, here, false);
        return;
    }
    bool to_compat = !(cd & DESC_L);
    if (to_compat && !(cd & DESC_D))
        stop(STOP_UNSUPPORTED, here, 18, sel, cd);
    if (to_compat) {
        // A long-mode kernel may jump with a null SS and restore it in the
        // first compatibility instructions. This is legal, but not an Engine
        // stack model: let TCG execute that short selector-transition window,
        // rather than manufacture #GP in compat_entry_ok. Ordinary flat-stack
        // kernel jumps stay entirely within the Engine.
        if (privilege == 0 && !(c->seg_sel[SEG_SS] & ~3u))
            stop(STOP_UNSUPPORTED, here, 18, sel, cd);
        if (!compat_entry_ok(cd, c->seg_sel[SEG_SS], privilege, here))
            return;
        off = (u32)off;
    } else if (!canonical(off)) {
        raise_gp(0, here);
        return;
    }
    if (kind == 1) { // call: the old CS above the old rip, at the old width
        u64 at = sp - opsz * 2, ret = next_rip_lit(), old_cs = c->seg_sel[1];
        if (c->compat)
            at = (u32)at;
        memcpy(mem, &ret, opsz);
        memset(mem + opsz, 0, opsz);
        memcpy(mem + opsz, &old_cs, 2);
        if (!guest_write(at, mem, opsz * 2, access)) {
            raise_range_fault(at, opsz * 2, access | 2, here);
            return;
        }
        sp = at;
    } else if (kind == 0)
        sp += opsz * 2 + imm;
    c->x[4] = c->compat ? (u32)sp : sp;
    c->seg_sel[1] = (u16)((sel & ~3u) | privilege);
    set_mode(privilege, to_compat);
    c->elr = translate_here(off, 0, 0, false);
}

// mov to a segment register, or pop. Validates descriptors in both compatibility and 64-bit kernel mode
// per Intel SDM / QEMU seg_helper.c; fs and gs update their hidden base from the GDT/LDT (0 on null).
static void do_mov_seg(void) {
    u32 sr = (u32)c->x[R_S0] & 7, pop = (u32)(c->x[R_S0] >> 8) & 15, sel = (u32)c->x[R_S1] & 0xffff;
    u64 here = rip_here(c->elr - 4), d = 0;
    bool null = !(sel & ~3u);
    if (sr == 1) {
        x86_ud();
        return;
    }
    bool validate = c->compat || c->model_kernel;
    bool got = !null && desc_read(sel, &d);
    if (!null && !got && (validate || sr >= SEG_FS)) {
        raise_gp(sel & ~3u, here);
        return;
    }
    if (validate) {
        u32 rpl = sel & 3, dpl = desc_dpl(d), err = sel & ~3u;
        if (sr == SEG_SS) {
            if (null) {
                if (c->compat || c->cpl == 3 || rpl != c->cpl) {
                    raise_gp(0, here);
                    return;
                }
            } else {
                if (rpl != c->cpl || dpl != c->cpl || !(d & DESC_S) || (d & DESC_CODE) || !(d & DESC_RW)) {
                    raise_gp(err, here);
                    return;
                }
                if (!(d & DESC_P)) {
                    deliver(12, err, true, here, false);
                    return;
                }
                if (c->compat && (!desc_flat(d) || !(d & DESC_D) || (d & DESC_CONFORMING)))
                    stop(STOP_UNSUPPORTED, here, 13, sel, d);
            }
        } else if (!null) {
            bool code = d & DESC_CODE, conforming = code && (d & DESC_CONFORMING);
            if (!(d & DESC_S) || (code && !(d & DESC_RW)) || (!conforming && (dpl < c->cpl || dpl < rpl))) {
                raise_gp(err, here);
                return;
            }
            if (!(d & DESC_P)) {
                deliver(11, err, true, here, false);
                return;
            }
        }
        if (sr != SEG_SS) {
            seg_note(sr, null ? 0 : d, (u16)sel);
            if (c->compat && (engine_compat_segments_bad(c->seg_bad, c->cpl) & (1u << sr))) // TCG makes the load, and keeps the vCPU until the register changes
                stop(STOP_UNSUPPORTED, here, 13, sel, d);
        }
    } else if (sr != SEG_SS) {
        seg_note(sr, got ? d : 0, (u16)sel);
    }
    c->seg_sel[sr] = (u16)sel;
    // The exposed CPU is Intel: a null FS/GS load clears the hidden base
    // even in 64-bit code. Tiger expects this in its return trampoline.
    if (sr >= SEG_FS)
        c->x[sr == SEG_FS ? R_FS : R_GS] = null ? 0 : desc_base(d);
    if (pop)
        c->x[4] = c->compat ? (u32)(c->x[4] + pop) : c->x[4] + pop;
    if (sr == SEG_SS && c->compat)
        c->irq_shadow_rip = next_rip_lit(); // interrupts wait for the instruction after a load of ss
}

static void do_popf(void) {
    u64 here = rip_here(c->elr - 4);
    u64 v = c->x[R_S0], next_rsp = c->x[R_S1];
    u64 sz = c->x[R_S2];
    u64 cur = rflags_get();
    u32 iopl = (u32)(cur >> 12) & 3;
    u64 mask = RF_ARITH | RF_DF | RF_TF | RF_NT | RF_AC | (1ULL << 21);
    if (c->cpl == 0)
        mask |= RF_IOPL | RF_IF;
    else if (c->cpl <= iopl)
        mask |= RF_IF;
    if (sz == 2)
        mask &= 0xffff;
    u64 fl = (cur & ~mask) | (v & mask) | RFLAGS_FIXED;
    if (fl & RF_TF)
        stop(STOP_UNSUPPORTED, here, 9, fl, 0);
    rflags_set(fl);
    c->x[4] = next_rsp;
}

// Unsupported CR4 bits stop rather than half work. MCE, PCE and OSXSAVE ask nothing of this monitor: it raises no machine checks, and rdpmc and the
// xsave family stop anyway, where CR4 is checked by whoever runs them.
#define CR4_OK ((1ULL << 5) | (1ULL << 7) | (1ULL << 9) | (1ULL << 10) | (1ULL << 17) | (1ULL << 20) | (1ULL << 3) | (1ULL << 4) | (1ULL << 2) | (1ULL << 6) | (1ULL << 8) | (1ULL << 18))

static void do_mov_cr_write(void) {
    u64 cr = c->x[R_S0], v = c->x[R_S1];
    u64 here = rip_here(c->elr - 4);
    switch (cr) {
        case 0:
            if ((v & 0x80000001ULL) != 0x80000001ULL)
                stop(STOP_UNSUPPORTED, here, 10, v, 0); // paging or protection off
            if ((c->cr0 ^ v) & (1ULL << 16)) {
                c->cr0 = v;
                mon_lock();
                shadow_drop_all();
                mon_unlock();
            } else {
                c->cr0 = v;
            }
            fp_trap_sync();
            break;
        case 2:
            c->cr2 = v;
            break;
        case 3: {
            c->stat_cr3++;
            note_event('C', here, v, 0);
            c->cr3 = v & ~(1ULL << 63);
            // The shadow stays in step with x86 tables through write protection, so a flush needs no work: the x86 TLB
            // may always be emptier than software assumed, and here it simply never holds a stale entry.
            view_switch(cr3_key(c->cr3), view_mode());
            u64 next = next_rip_lit();
            c->elr = translate_here(next, 0, 0, false);
            break;
        }
        case 4:
            if (v & ~CR4_OK)
                stop(STOP_UNSUPPORTED, here, 11, v, 0);
            c->cr4 = v;
            break;
        case 8: {
            u32 old_tpr = c->apic_tpr;
            u32 new_tpr = (u32)(v & 15) << 4;
            if (new_tpr != old_tpr)
                __atomic_store_n(&c->apic_tpr, new_tpr, __ATOMIC_SEQ_CST);
            if (c->hybrid && new_tpr <= old_tpr &&
                __atomic_load_n(&c->apic_irr_prio, __ATOMIC_SEQ_CST) > new_tpr) {
                c->req_a[0] = new_tpr;
                hostcall(HC_APIC_TPR);
            }
            break;
        }
    }
}

#define MSR_EFER 0xC0000080u
#define MSR_STAR 0xC0000081u
#define MSR_LSTAR 0xC0000082u
#define MSR_CSTAR 0xC0000083u
#define MSR_SFMASK 0xC0000084u
#define MSR_FS 0xC0000100u
#define MSR_GS 0xC0000101u
#define MSR_KGS 0xC0000102u
#define MSR_TSC 0x10u
#define MSR_APIC_BASE 0x1Bu
#define MSR_CORE_THREAD_COUNT 0x35u
#define MSR_TSC_DEADLINE 0x6E0u
#define MSR_SYSENTER_CS 0x174u
#define MSR_SYSENTER_ESP 0x175u
#define MSR_SYSENTER_EIP 0x176u

// ---- Hyper-V Enlightenments (Microsoft TLFS, EL1 zero-VM-exit fast paths) ----
// Shared with Swift (Engine.swift / Tally.swift) at a dedicated 4 KiB page inside OFF_MON_DATA
// (after the 512 KiB handoff slot, well below OFF_MON_CPU at +1 MiB) so engine_abi.h stays untouched.
#define HV_SHARED_OFF (OFF_MON_DATA + 0x81000ULL)
#define HV_SHARED_MAGIC 0x4856454E4C494748ULL // "HVENLIGH"
#define HV_SHARED_VERSION 1u

#define HV_FEAT_TIME_REF_COUNT (1u << 1)
#define HV_FEAT_APIC_ACCESS (1u << 4)
#define HV_FEAT_HYPERCALL (1u << 5)
#define HV_FEAT_VP_INDEX (1u << 6)
#define HV_FEAT_REFERENCE_TSC (1u << 9)
#define HV_FEAT_ACCESS_FREQ_MSRS (1u << 11)

#define HV_MSR_GUEST_OS_ID 0x40000000u
#define HV_MSR_HYPERCALL 0x40000001u
#define HV_MSR_VP_INDEX 0x40000002u
#define HV_MSR_TIME_REF_COUNT 0x40000020u
#define HV_MSR_REFERENCE_TSC 0x40000021u
#define HV_MSR_TSC_FREQUENCY 0x40000022u
#define HV_MSR_APIC_FREQUENCY 0x40000023u
#define HV_MSR_EOI 0x40000070u
#define HV_MSR_ICR 0x40000071u
#define HV_MSR_TPR 0x40000072u
#define HV_MSR_APIC_ASSIST_PAGE 0x40000073u

#define HV_STATUS_SUCCESS 0ULL
#define HV_STATUS_INVALID_HYPERCALL_CODE 2ULL
#define HV_STATUS_INVALID_HYPERCALL_INPUT 3ULL
#define HV_STATUS_INVALID_ALIGNMENT 4ULL
#define HV_STATUS_INVALID_PARAMETER 5ULL

#define HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE 0x0002u
#define HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST 0x0003u
#define HVCALL_NOTIFY_LONG_SPIN_WAIT 0x0008u
#define HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE_EX 0x0013u
#define HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST_EX 0x0014u

typedef struct {
    u64 apic_assist_msr;
    u64 stat_msr_time_ref;
    u64 stat_msr_ref_tsc_w;
    u64 stat_msr_eoi;
    u64 stat_msr_eoi_fast;
    u64 stat_msr_tpr_r;
    u64 stat_msr_tpr_w;
    u64 stat_msr_icr_r;
    u64 stat_msr_icr_w;
    u64 stat_msr_apic_assist_w;
    u64 stat_msr_vp_index;
    u64 stat_msr_guest_os_id_w;
    u64 stat_msr_hypercall_w;
    u64 stat_hc_spinwait;
    u64 stat_hc_flush_space;
    u64 stat_hc_flush_list;
    u64 stat_hc_flush_space_ex;
    u64 stat_hc_flush_list_ex;
    u64 stat_hc_flush_gva_reps;
    u64 stat_hc_unknown;
    u64 last_unknown_code;
    u64 pad[3];
} HvCpuShared __attribute__((aligned(64)));

typedef struct {
    u64 magic;
    u32 version;
    u32 enabled;
    u32 feat_eax;
    u32 feat_edx;
    u32 enlighten_eax;
    u32 spinlock_attempts;
    u64 guest_os_id;
    u64 hypercall_msr;
    u64 ref_tsc_msr;
    u32 tsc_seq;
    u32 pad_hdr;
    HvCpuShared cpu[MAX_VCPUS];
} HvSharedBlock;

typedef struct {
    u32 tsc_sequence;
    u32 reserved1;
    u64 tsc_scale;
    i64 tsc_offset;
} HvReferenceTscPage;

_Static_assert(sizeof(HvCpuShared) == 192, "HvCpuShared must be 192 bytes (3 cache lines) to match Swift");
_Static_assert(__builtin_offsetof(HvSharedBlock, cpu) == 64, "HvSharedBlock.cpu offset must be 64 to match Swift");
_Static_assert(sizeof(HvSharedBlock) <= 4096, "HvSharedBlock must fit within one 4 KiB page");

#define hv_blk ((HvSharedBlock *)PRIV(HV_SHARED_OFF))
static inline bool hv_active(void) {
    return hv_blk->magic == HV_SHARED_MAGIC && hv_blk->enabled != 0;
}

static bool udiv128(u64 hi, u64 lo, u64 d, u64 *q, u64 *r);

// The 24 MHz counter in nanoseconds, then at the guest's TSC rate, in the same steps as QEMU's cpu_get_tsc so TCG and the engine agree.
static inline u64 hv_vm_ns_now(void) {
    return (RSR(cntvct_el0) + c->tsc_counter_offset) * 125 / 3 + c->tsc_ns_offset;
}
static inline u64 tsc_now(void) {
    u64 ns = hv_vm_ns_now();
    return ns + ns * c->tsc_num / c->tsc_den + c->tsc_offset;
}
static inline u64 hv_tsc_freq_hz(void) {
    u64 den = c->tsc_den ? c->tsc_den : 1;
    return 1000000000ULL + (1000000000ULL * c->tsc_num) / den;
}

static void hv_prepare_guest_page_write(u64 gpa) {
    LOCK_REQUIRED();
    u32 fr = (u32)(gpa >> 12);
    u8 pi = page_info[fr];
    if (pi & PI_CODE) {
        c->req_a[0] = gpa;
        c->req_a[1] = KILL_STORE;
        __atomic_store_n(&page_info[fr], (u8)(pi & ~PI_CODE), __ATOMIC_RELEASE);
        hostcall(HC_SMC);
        reset_all_return_stacks();
        pi = page_info[fr];
    }
    if (pi & PI_PT) {
        if (!mirror_head[fr]) {
            page_info[fr] = (u8)(pi & ~PI_PT);
        } else {
            for (u32 i = 0; i < 512; i++)
                pt_slot_changed(gpa + (u64)i * 8);
        }
    }
}

static bool hv_setup_hypercall_page(u64 val) {
    mon_lock();
    u64 os_id = __atomic_load_n(&hv_blk->guest_os_id, __ATOMIC_ACQUIRE);
    if (!(val & 1ULL) || !os_id) {
        __atomic_store_n(&hv_blk->hypercall_msr, val & ~1ULL, __ATOMIC_RELEASE);
        mon_unlock();
        return true;
    }
    u64 gpa = val & ~0xFFFULL;
    if (!is_ram(gpa)) {
        mon_unlock();
        return false;
    }
    hv_prepare_guest_page_write(gpa);
    u8 *pg = (u8 *)RAM(gpa);
    memset(pg, 0, 4096);
    // Standard Intel x86-64 Hyper-V hypercall stub: vmcall; ret
    pg[0] = 0x0F;
    pg[1] = 0x01;
    pg[2] = 0xC1;
    pg[3] = 0xC3;
    __asm__ volatile("dsb ishst" : : : "memory");
    __atomic_store_n(&hv_blk->hypercall_msr, gpa | 1ULL, __ATOMIC_RELEASE);
    mon_unlock();
    return true;
}

static bool hv_setup_ref_tsc_page(u64 val) {
    mon_lock();
    if (!(val & 1ULL)) {
        u64 old = __atomic_load_n(&hv_blk->ref_tsc_msr, __ATOMIC_ACQUIRE);
        if (old & 1ULL) {
            u64 old_gpa = old & ~0xFFFULL;
            if (is_ram(old_gpa)) {
                HvReferenceTscPage *opg = (HvReferenceTscPage *)RAM(old_gpa);
                __atomic_store_n(&opg->tsc_sequence, 0u, __ATOMIC_RELEASE);
            }
        }
        __atomic_store_n(&hv_blk->ref_tsc_msr, val, __ATOMIC_RELEASE);
        mon_unlock();
        return true;
    }
    u64 gpa = val & ~0xFFFULL;
    if (!is_ram(gpa)) {
        mon_unlock();
        return false;
    }
    u64 tsc_freq = hv_tsc_freq_hz();
    u64 scale = 0, rem = 0;
    if (!udiv128(10000000ULL, 0, tsc_freq, &scale, &rem)) {
        mon_unlock();
        return false;
    }
    if (rem * 2 >= tsc_freq)
        scale++;
    i64 off = 0;
    if (c->tsc_offset != 0 || c->tsc_num != 0) {
        u64 den = c->tsc_den ? c->tsc_den : 1;
        u64 now_ns = hv_vm_ns_now();
        u64 cur_ref = now_ns / 100ULL;
        u64 cur_tsc = now_ns + now_ns * c->tsc_num / den + c->tsc_offset;
        u64 scaled = (u64)(((__uint128_t)cur_tsc * scale) >> 64);
        off = (i64)(cur_ref - scaled);
    }
    hv_prepare_guest_page_write(gpa);
    u8 *raw = (u8 *)RAM(gpa);
    HvReferenceTscPage *pg = (HvReferenceTscPage *)raw;
    __atomic_store_n(&pg->tsc_sequence, 0u, __ATOMIC_RELEASE);
    __asm__ volatile("dsb ishst" : : : "memory");
    memset(raw + 4, 0, 4096 - 4);
    pg->tsc_scale = scale;
    pg->tsc_offset = off;
    u32 seq = hv_blk->tsc_seq + 1;
    if (seq == 0 || seq == 0xFFFFFFFFu)
        seq = 1;
    hv_blk->tsc_seq = seq;
    __asm__ volatile("dsb ishst" : : : "memory");
    __atomic_store_n(&pg->tsc_sequence, seq, __ATOMIC_RELEASE);
    __atomic_store_n(&hv_blk->ref_tsc_msr, val, __ATOMIC_RELEASE);
    mon_unlock();
    return true;
}

static bool hv_setup_apic_assist_page(u64 val) {
    if (val & 1ULL) {
        u64 gpa = val & ~0xFFFULL;
        if (!is_ram(gpa))
            return false;
        mon_lock();
        hv_prepare_guest_page_write(gpa);
        *(u32 *)RAM(gpa) = 0; // NO_EOI_REQUIRED = 0; EOI takes the EL1 HV_MSR_EOI fast path
        __asm__ volatile("dsb ishst" : : : "memory");
        mon_unlock();
    }
    hv_blk->cpu[cpu_index()].apic_assist_msr = val;
    return true;
}

static bool hv_rdmsr(u32 m, u64 *v) {
    u32 feat = hv_blk->feat_eax;
    HvCpuShared *hvc = &hv_blk->cpu[cpu_index()];
    switch (m) {
        case HV_MSR_GUEST_OS_ID:
            *v = __atomic_load_n(&hv_blk->guest_os_id, __ATOMIC_ACQUIRE);
            return true;
        case HV_MSR_HYPERCALL:
            *v = __atomic_load_n(&hv_blk->hypercall_msr, __ATOMIC_ACQUIRE);
            return true;
        case HV_MSR_VP_INDEX:
            if (!(feat & HV_FEAT_VP_INDEX))
                return false;
            hvc->stat_msr_vp_index++;
            *v = cpu_index();
            return true;
        case HV_MSR_TIME_REF_COUNT:
            if (!(feat & HV_FEAT_TIME_REF_COUNT))
                return false;
            hvc->stat_msr_time_ref++;
            *v = hv_vm_ns_now() / 100ULL;
            return true;
        case HV_MSR_REFERENCE_TSC:
            if (!(feat & HV_FEAT_REFERENCE_TSC))
                return false;
            *v = __atomic_load_n(&hv_blk->ref_tsc_msr, __ATOMIC_ACQUIRE);
            return true;
        case HV_MSR_TSC_FREQUENCY:
            if (!(feat & HV_FEAT_ACCESS_FREQ_MSRS))
                return false;
            *v = hv_tsc_freq_hz();
            return true;
        case HV_MSR_APIC_FREQUENCY:
            if (!(feat & HV_FEAT_ACCESS_FREQ_MSRS))
                return false;
            *v = 1000000000ULL;
            return true;
        case HV_MSR_EOI:
            return false; // Write-only per TLFS
        case HV_MSR_ICR:
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_icr_r++;
            *v = ((u64)c->apic_icr_hi << 32) | c->apic_icr_lo;
            return true;
        case HV_MSR_TPR:
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_tpr_r++;
            *v = c->apic_tpr;
            return true;
        case HV_MSR_APIC_ASSIST_PAGE:
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            *v = hvc->apic_assist_msr;
            return true;
        default:
            return false;
    }
}

static bool hv_wrmsr(u32 m, u64 v) {
    u32 feat = hv_blk->feat_eax;
    HvCpuShared *hvc = &hv_blk->cpu[cpu_index()];
    switch (m) {
        case HV_MSR_GUEST_OS_ID:
            hvc->stat_msr_guest_os_id_w++;
            __atomic_store_n(&hv_blk->guest_os_id, v, __ATOMIC_RELEASE);
            if (!v)
                __atomic_store_n(&hv_blk->hypercall_msr,
                                 __atomic_load_n(&hv_blk->hypercall_msr, __ATOMIC_RELAXED) & ~1ULL,
                                 __ATOMIC_RELEASE);
            return true;
        case HV_MSR_HYPERCALL:
            hvc->stat_msr_hypercall_w++;
            return hv_setup_hypercall_page(v);
        case HV_MSR_REFERENCE_TSC:
            if (!(feat & HV_FEAT_REFERENCE_TSC))
                return false;
            hvc->stat_msr_ref_tsc_w++;
            return hv_setup_ref_tsc_page(v);
        case HV_MSR_EOI: {
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_eoi++;
            if (c->hybrid) {
                bool fast = false;
                hybrid_apic_eoi_fast(&fast);
                if (fast)
                    hvc->stat_msr_eoi_fast++;
            } else {
                vapic_write(0xB0, (u32)v);
                hvc->stat_msr_eoi_fast++;
            }
            return true;
        }
        case HV_MSR_ICR: {
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_icr_w++;
            u32 lo = (u32)v;
            u32 hi = (u32)(v >> 32);
            c->apic_icr_hi = hi;
            c->apic_icr_lo = lo & ~0x1000u; // delivery status always idle
            if (c->hybrid) {
                c->req_a[0] = ((u64)hi << 32) | lo;
                hostcall(HC_APIC_ICR);
            } else {
                vapic_write(0x310, hi);
                vapic_write(0x300, lo);
            }
            return true;
        }
        case HV_MSR_TPR:
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_tpr_w++;
            if (c->hybrid) {
                hybrid_apic_tpr_write((u32)v & 0xffu);
            } else {
                c->apic_tpr = (u32)v & 0xffu;
            }
            return true;
        case HV_MSR_APIC_ASSIST_PAGE:
            if (!(feat & HV_FEAT_APIC_ACCESS))
                return false;
            hvc->stat_msr_apic_assist_w++;
            return hv_setup_apic_assist_page(v);
        default:
            return false; // Read-only or unadvertised Hyper-V MSR -> #GP(0)
    }
}

static bool try_hv_vmcall(u64 rip) {
    if (!hv_active() || c->cpl != 0)
        return false;
    u64 hc_msr = __atomic_load_n(&hv_blk->hypercall_msr, __ATOMIC_ACQUIRE);
    if (!(hc_msr & 1ULL))
        return false;
    u64 hc_gpa = hc_msr & ~0xFFFULL;
    bool is_vmcall = false;
    if (me->hv_hc_va_cached == rip && me->hv_hc_gpa_cached == hc_gpa && (rip & 0xFFFULL) <= 0xFFCu) {
        u32 insn = *(const u32 *)RAM(hc_gpa + (rip & 0xFFFULL));
        if ((insn & 0x00FFFFFFu) == 0x00C1010Fu)
            is_vmcall = true;
    }
    if (!is_vmcall) {
        Walk w = x86_walk(c->cr3, rip, 16, false);
        if (!w.ok || !is_ram(w.gpa))
            return false;
        u8 b[3];
        if ((w.gpa & 0xFFFULL) <= 0xFFDu) {
            const u8 *p = (const u8 *)RAM(w.gpa);
            b[0] = p[0];
            b[1] = p[1];
            b[2] = p[2];
        } else if (!guest_read(rip, b, 3, 0)) {
            return false;
        }
        if (b[0] != 0x0F || b[1] != 0x01 || b[2] != 0xC1)
            return false;
        if ((w.gpa & ~0xFFFULL) == hc_gpa && (w.gpa & 0xFFFULL) <= 0xFFCu) {
            me->hv_hc_va_cached = rip;
            me->hv_hc_gpa_cached = hc_gpa;
        }
    }

    u64 in_ctrl = c->x[1]; // RCX
    u64 in_param = c->x[2]; // RDX
    u32 code = (u32)(in_ctrl & 0xFFFFu);
    bool fast = ((in_ctrl >> 16) & 1u) != 0;
    u32 var_head = (u32)((in_ctrl >> 17) & 0x3FFu);
    u32 rep_count = (u32)((in_ctrl >> 32) & 0xFFFu);
    u32 rep_start = (u32)((in_ctrl >> 48) & 0xFFFu);
    HvCpuShared *hvc = &hv_blk->cpu[cpu_index()];
    u64 result = HV_STATUS_INVALID_HYPERCALL_CODE;

    if (in_ctrl & 0xF000F000F8000000ULL) {
        result = HV_STATUS_INVALID_HYPERCALL_INPUT;
    } else {
        switch (code) {
            case HVCALL_NOTIFY_LONG_SPIN_WAIT:
                if (rep_count != 0 || rep_start != 0 || var_head != 0) {
                    result = HV_STATUS_INVALID_HYPERCALL_INPUT;
                    break;
                }
                if (!fast) {
                    if (in_param & 7ULL) {
                        result = HV_STATUS_INVALID_ALIGNMENT;
                        break;
                    }
                    if (!is_ram(in_param)) {
                        result = HV_STATUS_INVALID_PARAMETER;
                        break;
                    }
                }
                hvc->stat_hc_spinwait++;
                __asm__ volatile("yield");
                if (smp) {
                    c->req_a[0] = 0;
                    c->req_a[1] = 0;
                    c->req_a[2] = 0;
                    hostcall(HC_YIELD);
                }
                result = HV_STATUS_SUCCESS;
                break;
            case HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE:
            case HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE_EX:
                if (rep_count != 0 || rep_start != 0 ||
                    (code == HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE && var_head != 0)) {
                    result = HV_STATUS_INVALID_HYPERCALL_INPUT;
                    break;
                }
                if (!fast) {
                    if (in_param & 7ULL) {
                        result = HV_STATUS_INVALID_ALIGNMENT;
                        break;
                    }
                    if (!is_ram(in_param)) {
                        result = HV_STATUS_INVALID_PARAMETER;
                        break;
                    }
                }
                // Shadow page tables stay synchronously coherent via PI_PT write-protection + tlbi vae1is/aside1is;
                // drain any pending DMA/TCG page-table notes and ensure inner-shareable TLBI visibility.
                if (c->hybrid)
                    notes_drain();
                __asm__ volatile("dsb ish" : : : "memory");
                if (code == HVCALL_FLUSH_VIRTUAL_ADDRESS_SPACE)
                    hvc->stat_hc_flush_space++;
                else
                    hvc->stat_hc_flush_space_ex++;
                result = HV_STATUS_SUCCESS;
                break;
            case HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST:
            case HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST_EX:
                if (rep_count == 0 || rep_start >= rep_count ||
                    (code == HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST && var_head != 0)) {
                    result = HV_STATUS_INVALID_HYPERCALL_INPUT;
                    break;
                }
                if (!fast) {
                    if (in_param & 7ULL) {
                        result = HV_STATUS_INVALID_ALIGNMENT;
                        break;
                    }
                    if (!is_ram(in_param)) {
                        result = HV_STATUS_INVALID_PARAMETER;
                        break;
                    }
                }
                if (c->hybrid)
                    notes_drain();
                __asm__ volatile("dsb ish" : : : "memory");
                if (code == HVCALL_FLUSH_VIRTUAL_ADDRESS_LIST)
                    hvc->stat_hc_flush_list++;
                else
                    hvc->stat_hc_flush_list_ex++;
                hvc->stat_hc_flush_gva_reps += (u64)(rep_count - rep_start);
                // TLFS 3.6.2: RepsCompleted (bits 43:32) equals total rep_count on full completion.
                result = ((u64)rep_count << 32) | HV_STATUS_SUCCESS;
                break;
            default:
                hvc->stat_hc_unknown++;
                hvc->last_unknown_code = in_ctrl;
                result = HV_STATUS_INVALID_HYPERCALL_CODE;
                break;
        }
    }
    c->x[0] = result;
    c->elr = translate_here(rip + 3, 0, 0, false);
    return true;
}

// Intel RAPL / thermal / platform / perf sensor MSRs verified to hit `default: val = 0;`
// in QEMU target/i386/tcg/system/misc_helper.c helper_rdmsr (excluding MCE 0x400+, x2APIC 0x800+,
// PERF_STATUS 0x198, SMI_COUNT 0x34, MTRRs, and MISC_ENABLE 0x1a0).
static bool msr_is_zero_read(u32 m) {
    switch (m) {
        case 0x17u: case 0x31u: case 0x39u:
        case 0xc1u: case 0xc2u: case 0xc3u: case 0xc4u:
        case 0xceu: case 0xe7u: case 0xe8u: case 0x150u:
        case 0x186u: case 0x187u:
        case 0x194u: case 0x196u: case 0x19au: case 0x19cu:
        case 0x1a2u: case 0x1a4u: case 0x1adu: case 0x1aeu:
        case 0x1b1u: case 0x1b2u: case 0x1fcu:
        case 0x30au: case 0x30bu: case 0x38du: case 0x38fu:
        case 0x601u: case 0x602u: case 0x606u:
        case 0x610u: case 0x611u: case 0x614u: case 0x615u: case 0x619u:
        case 0x638u: case 0x639u: case 0x641u:
        case 0x648u: case 0x649u: case 0x64au: case 0x64bu: case 0x64eu: case 0x650u:
        case 0x770u:
            return true;
        default:
            return false;
    }
}

// Sensor/perf MSRs verified to hit `default: break;` (no-op) in QEMU helper_wrmsr.
static bool msr_is_noop_write(u32 m) {
    return m == 0x150u || m == 0x38du || m == 0x38fu;
}

// The MSRs that live here in hybrid mode. Everything else is QEMU's.
static bool msr_is_monitors(u32 m) {
    return m == MSR_EFER || m == MSR_STAR || m == MSR_LSTAR || m == MSR_CSTAR || m == MSR_SFMASK || m == MSR_FS || m == MSR_GS || m == MSR_KGS || m == MSR_SYSENTER_CS || m == MSR_SYSENTER_ESP || m == MSR_SYSENTER_EIP;
}

static bool msr_is_hv_range(u32 m) {
    return hv_active() && m >= 0x40000000u && m <= 0x400001FFu;
}

static bool msr_is_monitors_read(u32 m) {
    return msr_is_monitors(m) || m == MSR_TSC ||
           (m == MSR_APIC_BASE && c->hybrid_apic_base_msr != 0) ||
           (m == MSR_CORE_THREAD_COUNT && c->msr_core_thread_count != 0) ||
           msr_is_hv_range(m) ||
           msr_is_zero_read(m);
}

static bool msr_is_monitors_write(u32 m) {
    return msr_is_monitors(m) || msr_is_hv_range(m) || msr_is_noop_write(m);
}

static bool rdmsr(u32 m, u64 *v) {
    if (msr_is_hv_range(m))
        return hv_rdmsr(m, v);
    switch (m) {
        case MSR_EFER:
            *v = c->efer;
            return true;
        case MSR_STAR:
            *v = c->star;
            return true;
        case MSR_LSTAR:
            *v = c->lstar;
            return true;
        case MSR_CSTAR:
            *v = c->cstar;
            return true;
        case MSR_SFMASK:
            *v = c->sfmask;
            return true;
        case MSR_FS:
            *v = c->x[R_FS];
            return true;
        case MSR_GS:
            *v = c->x[R_GS];
            return true;
        case MSR_KGS:
            *v = c->kernel_gs_base;
            return true;
        case MSR_SYSENTER_CS:
            *v = c->sysenter_cs;
            return true;
        case MSR_SYSENTER_ESP:
            *v = c->sysenter_esp;
            return true;
        case MSR_SYSENTER_EIP:
            *v = c->sysenter_eip;
            return true;
        case MSR_TSC:
            *v = tsc_now();
            return true;
        case MSR_APIC_BASE:
            *v = (c->hybrid && c->hybrid_apic_base_msr) ? c->hybrid_apic_base_msr
                                                        : (c->apic_base | (c->apic_enabled ? 1ULL << 11 : 0) | 1ULL << 8);
            return true;
        case MSR_CORE_THREAD_COUNT:
            if (c->hybrid && c->msr_core_thread_count) {
                *v = c->msr_core_thread_count;
                return true;
            }
            break;
        case MSR_TSC_DEADLINE:
            *v = c->apic_tsc_deadline;
            return true;
    }
    if (c->hybrid && msr_is_zero_read(m)) {
        *v = 0;
        return true;
    }
    c->req_a[0] = m;
    hostcall(HC_RDMSR);
    *v = c->req_r[0];
    return !c->req_r[1];
}

static bool wrmsr(u32 m, u64 v) {
    if (msr_is_hv_range(m))
        return hv_wrmsr(m, v);
    if (c->hybrid && msr_is_noop_write(m))
        return true;
    switch (m) {
        case MSR_EFER:
            c->efer = v;
            return true;
        case MSR_STAR:
            c->star = v;
            return true;
        case MSR_LSTAR:
            c->lstar = v;
            return true;
        case MSR_CSTAR:
            c->cstar = v;
            return true;
        case MSR_SFMASK:
            c->sfmask = v;
            return true;
        case MSR_FS:
            c->x[R_FS] = v;
            return true;
        case MSR_GS:
            c->x[R_GS] = v;
            return true;
        case MSR_KGS:
            c->kernel_gs_base = v;
            return true;
        case MSR_SYSENTER_CS:
            c->sysenter_cs = v;
            return true;
        case MSR_SYSENTER_ESP:
            c->sysenter_esp = v;
            return true;
        case MSR_SYSENTER_EIP:
            c->sysenter_eip = v;
            return true;
        case MSR_APIC_BASE:
            c->apic_base = v & 0xFFFFF000ULL;
            c->apic_enabled = (v >> 11) & 1;
            return true;
        case MSR_TSC_DEADLINE: {
            c->apic_tsc_deadline = v;
            if (((c->apic_lvt_timer >> 17) & 3) == 2) {
                u64 now = tsc_now();
                c->apic_deadline_cnt = !v ? 0 : apic_cnt() + (v > now ? (v - now) * 3 / 125 : 0) + 1;
                apic_timer_arm();
            }
            return true;
        }
    }
    c->req_a[0] = m;
    c->req_a[1] = v;
    hostcall(HC_WRMSR);
    return !c->req_r[1];
}

// FPCR: RMode from MXCSR.RC, FZ, FIZ from DAZ, and always AH and NEP.
static void mxcsr_apply(u32 v) {
    static const u8 rc_map[4] = {0, 2, 1, 3};
    u64 fpcr = 2 | 4;
    if (v & 0x40)
        fpcr |= 1;
    if (v & 0x8000)
        fpcr |= 1ULL << 24;
    fpcr |= (u64)rc_map[(v >> 13) & 3] << 22;
    WSR(S3_3_C4_C4_0, fpcr);
    u64 fpsr = 0;
    if (v & 1)
        fpsr |= 1;
    if (v & 4)
        fpsr |= 2;
    if (v & 8)
        fpsr |= 4;
    if (v & 16)
        fpsr |= 8;
    if (v & 32)
        fpsr |= 16;
    if (v & 2)
        fpsr |= 128;
    WSR(S3_3_C4_C4_1, fpsr);
    c->mxcsr = v & 0xFFC0;
}

static u32 mxcsr_read(void) {
    u64 s = RSR(S3_3_C4_C4_1);
    u32 v = c->mxcsr;
    if (s & 1)
        v |= 1;
    if (s & 2)
        v |= 4;
    if (s & 4)
        v |= 8;
    if (s & 8)
        v |= 16;
    if (s & 16)
        v |= 32;
    if (s & 128)
        v |= 2;
    return v;
}

// CR0.TS: lazy FP switching in a real kernel relies on #NM. FPEN traps EL0's SIMD to EL1, whichever x86 instruction it belongs to.
static void fp_trap_sync(void) {
    WSR(cpacr_el1, (c->cr0 & 8) ? 1ULL << 20 : 3ULL << 20);
    __asm__ volatile("isb");
}

// Reads the flags as data for a helper; PF and AF stay in their lazy form.
static inline u64 cur_nzcv(void) { return c->spsr & (0xFULL << 28); }
#define NZCV_N (1ULL << 31)
#define NZCV_Z (1ULL << 30)
#define NZCV_C (1ULL << 29)
#define NZCV_V (1ULL << 28)

// Result flags from a value at a width: N, Z, PF. CF and OF supplied, AF as given.
static void helper_flags(u64 r, u32 size, bool cf, bool of) {
    u64 nz = 0;
    u32 bits = size * 8;
    u64 mask = bits == 64 ? ~0ULL : (1ULL << bits) - 1;
    r &= mask;
    if (r >> (bits - 1))
        nz |= NZCV_N;
    if (!r)
        nz |= NZCV_Z;
    if (!cf)
        nz |= NZCV_C;
    if (of)
        nz |= NZCV_V;
    c->x[R_S2] = nz;
    c->x[R_S3] = r;
    c->x[R_S4] = r; // AF is undefined for these operations; report 0
}

static void helper_keep_flags(void) {
    c->x[R_S2] = cur_nzcv();
    c->x[R_S3] = c->x[R_PF];
    c->x[R_S4] = c->x[R_AF];
}

// pcmpistri, for any imm8. Arguments as an element of a, of b, are bytes or words, signed or not; the strings end at their first zero.
static void h_pcmpistri(void) {
    u64 a[2] = {c->x[R_S0], c->x[R_S1]}, b[2] = {c->x[R_S2], c->x[R_S3]};
    u32 imm = (u32)c->x[R_S4], words = imm & 1, n = words ? 8 : 16, agg = (imm >> 2) & 3, la = n, lb = n, res = 0;
    u32 bits = words ? 16 : 8, sign = imm & 2;
    int ea[16], eb[16];
    for (u32 i = n; i-- > 0;) {
        u32 x = (u32)(a[i * bits / 64] >> (i * bits % 64)) & ((1u << bits) - 1), y = (u32)(b[i * bits / 64] >> (i * bits % 64)) & ((1u << bits) - 1);
        if (!x)
            la = i;
        if (!y)
            lb = i;
        ea[i] = sign ? (int)(x << (32 - bits)) >> (32 - bits) : (int)x;
        eb[i] = sign ? (int)(y << (32 - bits)) >> (32 - bits) : (int)y;
    }
    for (u32 j = 0; j < n; j++) {
        bool hit = false;
        if (agg == 0) {
            for (u32 i = 0; i < la && j < lb; i++)
                hit = hit || ea[i] == eb[j];
        } else if (agg == 1) {
            for (u32 i = 0; i + 1 < la && j < lb; i += 2)
                hit = hit || (ea[i] <= eb[j] && eb[j] <= ea[i + 1]);
        } else if (agg == 2) {
            hit = j >= la || j >= lb ? j >= la && j >= lb : ea[j] == eb[j];
        } else {
            hit = true;
            for (u32 i = 0, k = j; i < n - j && i < la && hit; i++, k++)
                hit = k < lb && ea[i] == eb[k];
        }
        if (hit)
            res |= 1u << j;
    }
    u32 all = (1u << n) - 1, valid = (1u << lb) - 1;
    u32 polarity = (imm >> 4) & 3;
    if (polarity == 1)
        res ^= all;
    else if (polarity == 3)
        res ^= valid;
    c->x[R_S0] = res ? ((imm & 0x40) ? 31 - (u32)__builtin_clz(res) : (u32)__builtin_ctz(res)) : n;
    c->x[R_S2] = (lb < n ? NZCV_Z : 0) | (la < n ? NZCV_N : 0) | (res ? 0 : NZCV_C) | ((res & 1) ? NZCV_V : 0); // C holds CF inverted
    c->x[R_S3] = 1; // PF and AF clear
    c->x[R_S4] = 1;
}

static void count_shift_helper(u64 descriptor, u64 count) {
    if (!c->profile_shift_helpers)
        return;
    u32 size = (u32)(descriptor >> 8) & 0xff, width;
    switch (size) {
        case 1: width = 0; break;
        case 2: width = 1; break;
        case 4: width = 2; break;
        case 8: width = 3; break;
        default: c->stat_shift_helpers_unclassified++; return;
    }
    u64 allowed = 7 | (0xffULL << 8) | SHIFT_HELPER_PROFILE_MARK | SHIFT_HELPER_PROFILE_MEM | SHIFT_HELPER_PROFILE_CL;
    if (!(descriptor & SHIFT_HELPER_PROFILE_MARK) || (descriptor & ~allowed)) {
        c->stat_shift_helpers_unclassified++;
        return;
    }
    u32 memory = (descriptor & SHIFT_HELPER_PROFILE_MEM) != 0;
    u32 by_cl = (descriptor & SHIFT_HELPER_PROFILE_CL) != 0;
    u32 zero = (count & (size == 8 ? 63 : 31)) == 0;
    u32 bucket = (((((u32)descriptor & 7) * 4 + width) * 2 + memory) * 2 + by_cl) * 2 + zero;
    c->stat_shift_helpers[bucket]++;
}

static void h_shift(void) {
    u64 v = c->x[R_S0], cnt = c->x[R_S1], desc_ = c->x[R_S2];
    u32 kind = desc_ & 7, size = (u32)(desc_ >> 8) & 0xff;
    count_shift_helper(desc_, cnt);
    u32 bits = size * 8;
    u64 mask = bits == 64 ? ~0ULL : (1ULL << bits) - 1;
    cnt &= size == 8 ? 63 : 31;
    v &= mask;
    if (!cnt) {
        helper_keep_flags();
        c->x[R_S0] = v;
        return;
    }
    u64 nz = cur_nzcv();
    bool cf = !(nz & NZCV_C), of = nz & NZCV_V;
    u64 r = v;
    bool msb, setall = true;
    switch (kind) {
        case 4: // shl
            r = cnt >= 64 ? 0 : (v << cnt) & mask;
            cf = cnt <= bits ? (v >> (bits - cnt)) & 1 : 0;
            of = ((r >> (bits - 1)) & 1) ^ cf;
            break;
        case 5: // shr
            r = cnt >= bits ? 0 : v >> cnt;
            cf = cnt <= bits ? (v >> (cnt - 1)) & 1 : 0;
            of = (v >> (bits - 1)) & 1;
            break;
        case 7: { // sar
            i64 s = (i64)(v << (64 - bits)) >> (64 - bits);
            r = (u64)(cnt >= 64 ? s >> 63 : s >> cnt) & mask;
            cf = cnt >= bits ? (u64)s >> 63 : (u64)(s >> (cnt - 1)) & 1;
            of = 0;
            break;
        }
        default: {
            setall = false;
            if (kind < 2) { // rol, ror
                u32 k = cnt % bits;
                r = k ? ((kind == 0 ? (v << k) | (v >> (bits - k)) : (v >> k) | (v << (bits - k))) & mask) : v;
                msb = (r >> (bits - 1)) & 1;
                cf = kind == 0 ? r & 1 : msb;
                of = kind == 0 ? msb ^ cf : msb ^ ((r >> (bits - 2)) & 1);
            } else { // rcl, rcr, modulo width + 1
                u32 k = cnt % (bits + 1);
                for (u32 i = 0; i < k; i++) {
                    if (kind == 2) {
                        bool out = (r >> (bits - 1)) & 1;
                        r = ((r << 1) | cf) & mask;
                        cf = out;
                    } else {
                        bool out = r & 1;
                        r = (r >> 1) | ((u64)cf << (bits - 1));
                        cf = out;
                    }
                }
                msb = (r >> (bits - 1)) & 1;
                of = kind == 2 ? msb ^ cf : msb ^ ((r >> (bits - 2)) & 1);
            }
        }
    }
    c->x[R_S0] = r;
    if (setall)
        helper_flags(r, size, cf, of);
    else {
        c->x[R_S2] = (nz & (NZCV_N | NZCV_Z)) | (cf ? 0 : NZCV_C) | (of ? NZCV_V : 0);
        c->x[R_S3] = c->x[R_PF];
        c->x[R_S4] = c->x[R_AF];
    }
}

static void h_shd(void) {
    u64 d = c->x[R_S0], s = c->x[R_S1], cnt = c->x[R_S2];
    u32 size = (u32)c->x[R_S3] & 0xff;
    bool right = c->x[R_S3] >> 8;
    u32 bits = size * 8;
    u64 mask = bits == 64 ? ~0ULL : (1ULL << bits) - 1;
    cnt &= size == 8 ? 63 : 31;
    d &= mask;
    s &= mask;
    if (!cnt) {
        helper_keep_flags();
        c->x[R_S0] = d;
        return;
    }
    u64 r = 0;
    bool cf = false;
    if (size == 2 && cnt > 16) {
        // Undefined in the manual. As hardware does: the rotation goes through a 48-bit composite.
        u64 w = right ? (d | s << 16 | d << 32) : (d << 32 | s << 16 | d);
        r = right ? (w >> cnt) & 0xffff : (w >> (32 - cnt)) & 0xffff;
        cf = right ? (w >> (cnt - 1)) & 1 : (w >> (48 - cnt)) & 1;
    } else if (right) {
        r = ((d >> cnt) | (s << (bits - cnt))) & mask;
        cf = (d >> (cnt - 1)) & 1;
    } else {
        r = ((d << cnt) | (s >> (bits - cnt))) & mask;
        cf = (d >> (bits - cnt)) & 1;
    }
    c->x[R_S0] = r;
    helper_flags(r, size, cf, ((r ^ d) >> (bits - 1)) & 1);
}

// 128 by 64 unsigned division by shift-subtract. False on quotient overflow or zero divisor.
static bool udiv128(u64 hi, u64 lo, u64 d, u64 *q, u64 *r) {
    if (!d || hi >= d)
        return false;
    u64 rem = hi, quo = 0;
    for (int i = 63; i >= 0; i--) {
        bool top = rem >> 63;
        rem = rem << 1 | ((lo >> i) & 1);
        if (top || rem >= d) {
            rem -= d;
            quo |= 1ULL << i;
        }
    }
    *q = quo;
    *r = rem;
    return true;
}

static void h_div(void) {
    u64 lo = c->x[R_S0], hi = c->x[R_S1], d = c->x[R_S2];
    u32 size = (u32)c->x[R_S3] & 0xff;
    bool sign = c->x[R_S3] >> 8;
    u64 here = rip_here(c->elr - 4);
    u64 q = 0, r = 0;
    bool ok;
    if (size == 8) {
        if (!sign)
            ok = udiv128(hi, lo, d, &q, &r);
        else {
            bool nn = hi >> 63, nd = d >> 63;
            u64 alo = lo, ahi = hi;
            if (nn) {
                alo = -lo;
                ahi = ~hi + !lo;
            }
            u64 ad = nd ? -d : d;
            ok = udiv128(ahi, alo, ad, &q, &r);
            if (ok) {
                if (nn != nd) {
                    if (q > (1ULL << 63))
                        ok = false;
                    q = -q;
                } else if (q >> 63)
                    ok = false;
                if (nn)
                    r = -r;
            }
        }
    } else {
        // 8 and 16 bits: x16 holds ax or ax with x17 holding dx.
        u64 n = size == 2 ? ((hi & 0xffff) << 16 | (lo & 0xffff)) : (lo & 0xffff);
        u32 bits = size * 8;
        u64 mask = (1ULL << bits) - 1;
        d &= mask;
        if (!d)
            ok = false;
        else if (!sign) {
            q = n / d;
            r = n % d;
            ok = q <= mask;
        } else {
            i64 sn = size == 2 ? (i64)(int32_t)n : (i64)(int16_t)n;
            i64 sd = (i64)(d << (64 - bits)) >> (64 - bits);
            i64 sq = sn / sd, sr = sn % sd;
            i64 lim = 1LL << (bits - 1);
            ok = sq >= -lim && sq < lim;
            q = (u64)sq & mask;
            r = (u64)sr & mask;
        }
    }
    if (!ok) {
        deliver(0, 0, false, here, false);
        return;
    }
    c->x[R_S0] = q;
    c->x[R_S1] = r;
}

#include "x87.c"

static void do_cli_sti(bool sti) {
    u32 iopl = (u32)(c->rflags_sys >> 12) & 3;
    if (c->cpl > iopl) {
        x86_gp0();
        return;
    }
    if (sti) {
        if (!(c->rflags_sys & RF_IF)) {
            c->rflags_sys |= RF_IF;
            c->irq_shadow_rip = next_rip_lit();
        }
    } else
        c->rflags_sys &= ~RF_IF;
}

// ---- Experimental page REP service ----
// The final walks latch ordinary-RAM GPAs for one bounded prefix, like cached
// translations. They do not freeze page-table RAM: TCG/DMA may change it, and
// ordinary data writes may race. No hostcall, guest callback, IRQ or shootdown/
// HC_BOUNDARY acknowledgement occurs between that latch and the progress commit.
// Guest RAM backing is fixed while vCPUs run; direct/remappable/device mappings
// are excluded. mon_lock serializes PI_CODE/PT transitions and monitored writes.
// Rewalk the next prefix after exit_check has drained notes and handled stops.
// Physical aliases cannot become guest #PFs halfway through the committed copy.
_Static_assert(SVC_REP_PAGE == 71, "vectors.s embeds this internal SVC number");

static u64 rep_page_load(const void *p, u32 size) {
    u64 value;
    switch (size) {
        case 1: __asm__ volatile("ldarb %w0, [%1]" : "=r"(value) : "r"(p) : "memory"); break;
        case 2: __asm__ volatile("ldarh %w0, [%1]" : "=r"(value) : "r"(p) : "memory"); break;
        case 4: __asm__ volatile("ldar %w0, [%1]" : "=r"(value) : "r"(p) : "memory"); break;
        default: __asm__ volatile("ldar %0, [%1]" : "=r"(value) : "r"(p) : "memory"); break;
    }
    return value;
}

static void rep_page_store(void *p, u64 value, u32 size) {
    switch (size) {
        case 1: __asm__ volatile("stlrb %w0, [%1]" : : "r"(value), "r"(p) : "memory"); break;
        case 2: __asm__ volatile("stlrh %w0, [%1]" : : "r"(value), "r"(p) : "memory"); break;
        case 4: __asm__ volatile("stlr %w0, [%1]" : : "r"(value), "r"(p) : "memory"); break;
        default: __asm__ volatile("stlr %0, [%1]" : : "r"(value), "r"(p) : "memory"); break;
    }
}

static void rep_page_copy(u8 *dst, const u8 *src, u32 bytes, u32 size, bool movs, u64 pattern) {
    // Aligned integer units only: no pair-store abort/atomicity or EL1 TSO assumption.
    while (bytes >= 32 && !(((u64)dst | (movs ? (u64)src : 0)) & 7)) {
        for (u32 i = 0; i < 32; i += 8) {
            u64 value = movs ? rep_page_load(src + i, 8) : pattern;
            rep_page_store(dst + i, value, 8);
        }
        dst += 32;
        if (movs) src += 32;
        bytes -= 32;
    }
    while (bytes) {
        u64 value = movs ? rep_page_load(src, size) : pattern;
        rep_page_store(dst, value, size);
        dst += size;
        if (movs) src += size;
        bytes -= size;
    }
}

static bool rep_page_tracked(const Walk *w) {
    LOCK_REQUIRED();
    for (int level = 3; level >= w->leaf_level; level--) {
        u64 address = (u64)w->chain[level];
        if (address < RAM_ALIAS_VA)
            return false;
        u64 gpa = address - RAM_ALIAS_VA;
        if (!is_ram(gpa) || !(page_info[gpa >> 12] & PI_PT))
            return false;
        // Other walkers may set A/D without mon_lock; those changes preserve the mapping and permissions.
        if ((*w->chain[level] & ~(X_A | X_D)) != (w->read[level] & ~(X_A | X_D)))
            return false;
    }
    return true;
}

static void rep_page_fault(u64 va, u32 error, u64 rip) {
    // Every caller has released mon_lock, and no copied prefix remains uncommitted.
    if (error == ~0u)
        raise_gp(0, rip);
    else
        raise_pf(va, error, rip);
}

static void rep_page_service(bool movs) {
    bool site_movs;
    u64 head = c->elr - 4, elements = c->x[R_S1];
    if (!rep_page_site(head, &site_movs) || site_movs != movs ||
        (elements != 4 && elements != 8 && elements != 16 && elements != 32))
        stop(STOP_MONITOR_FAULT, 0, SVC_REP_PAGE, head, elements);
    u64 here = rip_here(head);
    c->elr = c->x[R_LR]; // Every zero-progress/completion/decline path runs the existing scalar fallthrough.
    u32 size = 32 / (u32)elements;
    u64 count = c->x[1], src = c->x[6], dst = c->x[7];
    if (!count || !c->model_kernel || (c->cpl != 0 && c->cpl != 3) || c->compat > 1 ||
        c->df != 1 || (c->rflags_sys & RF_TF) || me->smc_held || me->shadow_stepping ||
        c->irq_shadow_rip != ~0ULL || c->stepping || pt_step_slot || c->stop_requested)
        return;
    if (((dst | (movs ? src : 0)) & (size - 1)) ||
        (movs && ((dst - src) < 32 || (src - dst) < 32)))
        return;
    // Preserve the emitter's conservative whole-run compatibility-wrap exclusion.
    if (c->compat && (((count | dst | (movs ? src : 0)) >> 32) ||
        ((dst + count * size) >> 32) || (movs && ((src + count * size) >> 32))))
        return;

    mon_lock();
    Walk source = {0};
    u32 access = c->cpl == 3 ? 4 : 0;
    if (movs) {
        source = x86_walk(c->cr3, src, access, true); // source first, including its architectural A bit
        if (!source.ok) {
            mon_unlock();
            rep_page_fault(src, source.fault_err, here);
            return;
        }
        if (!is_ram(source.gpa) || is_direct(source.gpa) || !rep_page_tracked(&source)) {
            mon_unlock();
            return;
        }
    }
    Walk dest = x86_walk(c->cr3, dst, access | 2, true);
    if (!dest.ok) {
        mon_unlock();
        rep_page_fault(dst, dest.fault_err, here);
        return;
    }
    if (!is_ram(dest.gpa) || is_direct(dest.gpa) || !rep_page_tracked(&dest) ||
        (page_info[dest.gpa >> 12] & (PI_CODE | PI_PT)) ||
        (movs && (source.gpa >> 12) == (dest.gpa >> 12))) {
        mon_unlock();
        return;
    }
    // Complete any path preparation before the final walks/latch. No hostcall or
    // guest-visible acknowledgement is permitted after these and before commit.
    if (movs) {
        source = x86_walk(c->cr3, src, access, true);
        if (!source.ok) {
            mon_unlock();
            rep_page_fault(src, source.fault_err, here);
            return;
        }
        if (!is_ram(source.gpa) || is_direct(source.gpa) || !rep_page_tracked(&source)) {
            mon_unlock();
            return;
        }
    }
    dest = x86_walk(c->cr3, dst, access | 2, true);
    if (!dest.ok) {
        mon_unlock();
        rep_page_fault(dst, dest.fault_err, here);
        return;
    }
    if (!is_ram(dest.gpa) || is_direct(dest.gpa) || !rep_page_tracked(&dest) ||
        (page_info[dest.gpa >> 12] & (PI_CODE | PI_PT)) ||
        (movs && (source.gpa >> 12) == (dest.gpa >> 12))) {
        mon_unlock();
        return;
    }
    u64 prefix = (4096 - (dst & 0xfff)) / size;
    if (movs && prefix > (4096 - (src & 0xfff)) / size)
        prefix = (4096 - (src & 0xfff)) / size;
    if (prefix > count) prefix = count;
    if (!prefix || dest.gpa >= c->ram_size || prefix * size > c->ram_size - dest.gpa ||
        (movs && (source.gpa >= c->ram_size || prefix * size > c->ram_size - source.gpa))) {
        mon_unlock();
        return;
    }
    u32 bytes = (u32)(prefix * size);
    rep_page_copy(RAM(dest.gpa), movs ? RAM(source.gpa) : NULL, bytes, size, movs, c->x[R_S2]);
    __atomic_thread_fence(__ATOMIC_SEQ_CST); // All copy stores precede architectural progress publication, including at one vCPU.
    c->x[7] = c->compat ? (u32)(dst + bytes) : dst + bytes;
    if (movs) c->x[6] = c->compat ? (u32)(src + bytes) : src + bytes;
    c->x[1] = c->compat ? (u32)(count - prefix) : count - prefix;
    c->stat_rep_page_prefixes++;
    c->stat_rep_page_bytes += bytes;
    mon_unlock();
    if (c->x[1])
        c->elr = head; // Positive progress only: exit_check can now stop/deliver IRQ before another bounded prefix.
}
// ---- End experimental page REP service ----

#define HLT_POLL_TICKS 2400 // 100 us of the 24 MHz counter
static bool retry_unknown_code_fetch(void) {
    // See engine_code_fetch_can_retry: a stale fetch of translated code that
    // another vCPU has just written. Re-synchronize this executing alias and
    // run the same ARM PC again; repeating at that PC is a real fault.
    u64 pc = c->elr;
    u64 word = engine_capture_code_fault(pc, PRIV_VA + OFF_CC, PRIV_VA + OFF_CC2, CC_SIZE);
    if (!engine_code_fetch_can_retry(c->esr, word, pc, me->code_retry_pc))
        return false;
    me->code_retry_pc = pc;
    if (++me->code_retry_count <= ENGINE_CODE_FETCH_RETRY_LOGGED) {
        BlockEntry *block = find_block(pc);
        c->req_a[0] = pc;
        c->req_a[1] = block ? block->rip : 0;
        c->req_a[2] = me->code_retry_count;
        hostcall(HC_CODE_FETCH_RETRY);
    }
    code_changed((u32 *)pc);
    return true;
}

void mon_sync(void) {
    u64 prof_t0 = c->profile_el1_ticks ? RSR(cntvct_el0) : 0;
    u64 prof_hvc0 = c->profile_el1_ticks ? me->hvc_ticks_local : 0;
    me->t2_synced = false;
    u32 ec = (u32)(c->esr >> 26);
    u32 n = (u32)c->esr & 0xffff;
    if (ec == 0 && retry_unknown_code_fetch()) {
        if (c->profile_el1_ticks) {
            u64 total = RSR(cntvct_el0) - prof_t0;
            u64 hvc_dt = me->hvc_ticks_local - prof_hvc0;
            c->stat_el1_sync_svc_ticks += total > hvc_dt ? (total - hvc_dt) : 0;
        }
        return;
    }
    me->code_retry_pc = 0; // anything else between two faults at one PC means the first retry got past it
    bool page_movs = false;
    bool page_rep = ec == 0x15 && n == SVC_REP_PAGE;
    if (page_rep && !rep_page_site(c->elr - 4, &page_movs))
        stop(STOP_MONITOR_FAULT, 0, SVC_REP_PAGE, c->elr, 0);
    if (!page_rep && me->smc_held)
        __atomic_store_n(&me->smc_held, 0, __ATOMIC_RELEASE);
    if (ec == 0x24) {
        handle_data_abort(true);
    } else if (ec == 0x15) {
        c->stat_svc[n & 127]++;
        switch (n) {
            case SVC_REP_PAGE:
                rep_page_service(page_movs);
                break;
            case SVC_DIRECT:
            case SVC_JCC:
            case SVC_CALL:
            case SVC_SINGLE_DONE: {
                // The literal target sits right after the svc. The stub word (elr-4) is what gets patched.
                u64 stub = c->elr - 4;
                u64 target = *lit(c->elr);
                u64 site = stub;
                if (n == SVC_CALL)
                    site = c->x[R_LR] - 4;
                u64 arm = translate_here(target, site, n, false);
                c->elr = arm;
                break;
            }
            case SVC_LOOKUP_MISS: {
                u64 target = c->x[R_S0];
                event_delivered = false;
                code_cache_reset = false;
                u64 arm = translate_here(target, 0, SVC_LOOKUP_MISS, false);
                // The target did not fetch and a fault handler runs instead; or the cache was reset, and the code at x30 is gone.
                if (event_delivered || code_cache_reset) {
                    c->elr = arm;
                    break;
                }
                c->x[R_S2] = arm;
                c->elr = c->x[R_LR];
                break;
            }
            case SVC_DEAD: {
                // Entry of an invalidated block: find it and go to a fresh translation.
                bool from_fex = (c->x[R_FS] == (u64)c + 32768ULL);
                fex_unspill_state();
                u64 entry = c->elr - 4;
                u64 rip = rip_here(entry);
                event_delivered = false;
                code_cache_reset = false;
                u64 arm = translate_here(rip, from_fex ? 0 : entry, from_fex ? SVC_LOOKUP_MISS : SVC_DIRECT, false);
                if (from_fex && !event_delivered && !code_cache_reset && arm && arm != entry - 12 &&
                    *(const u32 *)(arm + 4) == 0x10000050u) {
                    BlockEntry *from = find_block(entry), *to = find_block(arm);
                    if (from && to && from->key == to->key && !bad_ring_link(entry, arm)) {
                        i64 d3 = ((i64)(arm + 12) - (i64)entry) >> 2;
                        i64 d0 = ((i64)arm - (i64)(entry - 12)) >> 2;
                        if (d3 >= -(1 << 25) && d3 < (1 << 25) && d0 >= -(1 << 25) && d0 < (1 << 25)) {
                            u32 *s0 = (u32 *)(entry - 12), w0 = *s0;
                            if (w0 == (0xD4000001u | (SVC_DEAD << 5))) {
                                __atomic_compare_exchange_n(s0, &w0, 0x14000000u | ((u32)d0 & 0x3ffffffu), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                                code_changed(s0);
                            }
                            u32 *s3 = (u32 *)entry, w3 = *s3;
                            if (w3 == (0xD4000001u | (SVC_DEAD << 5))) {
                                __atomic_compare_exchange_n(s3, &w3, 0x14000000u | ((u32)d3 & 0x3ffffffu), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                                code_changed(s3);
                            }
                        }
                    }
                }
                c->elr = arm;
                break;
            }
            case SVC_FEX_LINK: {
                u64 site = c->elr - 4;
                const u32 *ins = (const u32 *)c->elr;
                u64 target = ((u64)((ins[0] >> 5) & 0xffffu)) |
                             ((u64)((ins[1] >> 5) & 0xffffu) << 16) |
                             ((u64)((ins[2] >> 5) & 0xffffu) << 32) |
                             ((u64)((ins[3] >> 5) & 0xffffu) << 48);
                u64 arm = 0;
                {
                    const u64 *t = *(const u64 *const *)((const u8 *)c + 36864);
                    u32 h;
                    __asm__ volatile(".arch_extension crc\n crc32cx %w0, wzr, %x1" : "=r"(h) : "r"(target));
                    h &= HASH_ENTRIES - 1;
                    for (;;) {
                        u64 key = t[h * 2];
                        if (!key)
                            break;
                        if (key == target) {
                            u64 val = __atomic_load_n(&t[h * 2 + 1], __ATOMIC_ACQUIRE);
                            if (__atomic_load_n(&t[h * 2], __ATOMIC_RELAXED) == target)
                                arm = val & ~1ULL;
                            break;
                        }
                        h = (h + 1) & (HASH_ENTRIES - 1);
                    }
                }
                bool unspilled = false;
                if (!arm) {
                    fex_unspill_state();
                    unspilled = true;
                    event_delivered = false;
                    code_cache_reset = false;
                    arm = translate_here(target, 0, SVC_LOOKUP_MISS, false);
                    if (event_delivered || code_cache_reset) {
                        c->elr = arm;
                        break;
                    }
                } else if (c->tr_flushed) {
                    c->tr_flushed = 0;
                    fex_unspill_state();
                    reset_all_return_stacks();
                    c->elr = translate_here(target, 0, 0, false);
                    break;
                }
                BlockEntry *from = find_block(site), *to = find_block(arm);
                if (from && to && from->key == to->key && !bad_ring_link(site, arm) && *(const u32 *)(arm + 4) == 0x10000050u) {
                    i64 d = ((i64)(arm + 12) - (i64)site) >> 2;
                    if (d >= -(1 << 25) && d < (1 << 25)) {
                        u32 *s = (u32 *)site, was = *s;
                        if ((was & 0xFFE0001Fu) == 0xD4000001u) {
                            __atomic_compare_exchange_n(s, &was, 0x14000000u | ((u32)d & 0x3ffffffu), false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                            code_changed(s);
                        }
                        // Back-patch caller `b`/`b.cond` pointing at this thunk (word ins[6] = back = thunk_idx - caller_idx)
                        // to eliminate double jump `b Thunk -> b Target`.
                        u32 back = ins[6];
                        if (back > 0 && back < 4096 && back != 0xD503201Fu) {
                            u64 caller_va = site - (u64)back * 4;
                            u64 body_min = PRIV_VA + OFF_CC + 4ULL * from->arm_off + 12;
                            if (caller_va >= body_min && caller_va < site) {
                                u32 *caller = (u32 *)caller_va;
                                u32 cw = *caller;
                                if ((cw & 0x7C000000u) == 0x14000000u) {
                                    i64 orig_disp = (i64)((int32_t)(cw << 6) >> 6);
                                    if (caller + orig_disp == s) {
                                        i64 cd = ((i64)(arm + 12) - (i64)caller_va) >> 2;
                                        if (cd >= -(1 << 25) && cd < (1 << 25)) {
                                            __atomic_compare_exchange_n(caller, &cw, (cw & 0xFC000000u) | ((u32)cd & 0x3ffffffu),
                                                                        false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                                            code_changed(caller);
                                        }
                                    }
                                } else if ((cw & 0xFF000010u) == 0x54000000u) {
                                    i64 orig_disp = (i64)((int32_t)((cw >> 5) << 13) >> 13);
                                    if (caller + orig_disp == s) {
                                        i64 cd = ((i64)(arm + 12) - (i64)caller_va) >> 2;
                                        if (cd >= -(1 << 18) && cd < (1 << 18)) {
                                            __atomic_compare_exchange_n(caller, &cw, (cw & ~(0x7ffffu << 5)) | (((u32)cd & 0x7ffffu) << 5),
                                                                        false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                                            code_changed(caller);
                                        }
                                    }
                                }
                            }
                        }
                        c->elr = unspilled ? arm : (arm + 12);
                        break;
                    }
                }
                u32 *s = (u32 *)site, was = *s;
                if ((was & 0xFFE0001Fu) == 0xD4000001u) {
                    __atomic_compare_exchange_n(s, &was, 0xD503201Fu, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                    code_changed(s);
                }
                if (unspilled)
                    c->elr = arm;
                break;
            }
            case SVC_STOP: {
                u64 rip = rip_here(c->elr - 4);
                if (try_hv_vmcall(rip))
                    break;
                stop(STOP_UNTRANSLATED, rip, *(u32 *)c->elr, 0, 0);
            }
            case SVC_SYSCALL:
                do_syscall();
                break;
            case SVC_SYSRET:
                do_sysret();
                break;
            case SVC_SYSENTER:
                do_sysenter();
                break;
            case SVC_SYSEXIT:
                do_sysexit();
                break;
            case SVC_FAR:
                do_far();
                break;
            case SVC_IRET:
                do_iret();
                break;
            case SVC_INT:
                deliver((u32)c->x[R_S1], 0, false, next_rip_lit(), true);
                break;
            case SVC_INT3:
                deliver(3, 0, false, next_rip_lit(), true);
                break;
            case SVC_UD:
                if (c->hybrid)
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_UD, 0, 0);
                x86_ud();
                break;
            case SVC_DE:
                deliver(0, 0, false, rip_here(c->elr - 4), false);
                break;
            case SVC_GP0:
                x86_gp0();
                break;
            case SVC_HLT: {
                if (!c->model_kernel) {
                    stop(STOP_GUEST_HALT, rip_here(c->elr - 4), c->x[0], 0, 0);
                }
                u64 next = next_rip_lit();
                if (c->hybrid) {
                    // Waking through TCG and the host's condition variable costs tens of us, and threads that take turns across vCPUs do it each turn. So wait a
                    // little here first, as KVM's halt polling does: it costs an idle vCPU this long per hlt, and a stop still takes it straight to TCG.
                    if ((c->rflags_sys & RF_IF) && !c->irq_pending) {
                        u64 until = RSR(cntvct_el0) + HLT_POLL_TICKS;
                        while (!__atomic_load_n(&c->irq_line, __ATOMIC_ACQUIRE) && !__atomic_load_n(&c->stop_requested, __ATOMIC_ACQUIRE) && RSR(cntvct_el0) < until)
                            __asm__ volatile("yield");
                        if (__atomic_load_n(&c->irq_line, __ATOMIC_ACQUIRE)) {
                            sync_host_irq(true); // HLT retires before accepting the wake interrupt
                            if (c->irq_pending)
                                c->stat_hlt_polled++;
                        }
                    }
                    if (c->irq_pending && (c->rflags_sys & RF_IF)) { // sti; hlt with a vector already in hand: the hlt is over before it starts
                        c->elr = translate_here(next, 0, 0, false);
                        break;
                    }
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_HLT, 0, 0);
                }
                for (;;) {
                    apic_merge_host();
                    apic_timer_check();
                    bool ready = apic_ready_vector() || c->irq_pending;
                    if (ready && (c->rflags_sys & RF_IF))
                        break;
                    c->req_a[0] = next;
                    // How long the host may sleep: until the APIC timer, in counter ticks. 0 means no deadline.
                    c->req_a[1] = c->apic_deadline_cnt;
                    hostcall(HC_HLT); // returns when an interrupt may be pending
                    if (c->req_r[0] != ~0ULL)
                        c->irq_pending = (u32)c->req_r[0] + 1;
                    else if (c->req_r[1])
                        stop(STOP_GUEST_HALT, next, c->x[0], 0, 0);
                }
                c->elr = translate_here(next, 0, 0, false);
                break;
            }
            case SVC_CPUID: {
                u32 leaf = (u32)c->x[0];
                u32 subleaf = (u32)c->x[1];
                c->req_a[0] = leaf;
                c->req_a[1] = subleaf;
                hostcall(HC_CPUID);
                if (leaf == 0 && !c->cpuid0_cached) {
                    c->cpuid0_eax_ebx = ((u64)(u32)c->req_r[1] << 32) | (u32)c->req_r[0];
                    c->cpuid0_ecx_edx = ((u64)(u32)c->req_r[3] << 32) | (u32)c->req_r[2];
                    c->cpuid0_cached = 1;
                }
                if (leaf == 1 && (c->cr4 & (1ULL << 18)) && (c->req_r[2] & (1ULL << 26)))
                    c->req_r[2] |= (1ULL << 27); // CPUID.01H:ECX.OSXSAVE reflects live Monitor CR4.OSXSAVE
                if (leaf == 0xD && subleaf == 0 && (c->xcr0 & 4) && c->req_r[1] < c->req_r[2])
                    c->req_r[1] = c->req_r[2];   // CPUID.0DH.0:EBX reflects live Monitor XCR0 (AVX enabled -> 832 bytes)
                c->x[0] = c->req_r[0];
                c->x[3] = c->req_r[1];
                c->x[1] = c->req_r[2];
                c->x[2] = c->req_r[3];
                break;
            }
            case SVC_RDTSC:
            case SVC_RDTSCP: {
                u64 t = tsc_now();
                c->x[0] = (u32)t;
                c->x[2] = t >> 32;
                if (n == SVC_RDTSCP)
                    c->x[1] = 0;
                break;
            }
            case SVC_RDMSR: {
                if (c->cpl != 0) {
                    x86_gp0();
                    break;
                }
                u64 v;
                u32 msr = (u32)c->x[1];
                if (c->hybrid && !msr_is_monitors_read(msr))
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_RDMSR, msr, 0);
                if (!rdmsr(msr, &v)) {
                    x86_gp0();
                    break;
                }
                c->x[0] = (u32)v;
                c->x[2] = v >> 32;
                break;
            }
            case SVC_WRMSR:
                if (c->cpl != 0) {
                    x86_gp0();
                    break;
                }
                if (c->hybrid && !msr_is_monitors_write((u32)c->x[1])) {
                    if ((u32)c->x[1] == 0x6E0) { // the TSC deadline, written at every timer change: a call to QEMU's APIC is all that TCG would do for it
                        c->req_a[0] = 0x6E0;
                        c->req_a[1] = (u64)(u32)c->x[2] << 32 | (u32)c->x[0];
                        hostcall(HC_WRMSR);
                        if (!c->req_r[1])
                            break;
                    }
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_WRMSR, (u32)c->x[1], 0);
                }
                if (!wrmsr((u32)c->x[1], (u64)(u32)c->x[2] << 32 | (u32)c->x[0]))
                    x86_gp0();
                break;
            case SVC_CLI:
                do_cli_sti(false);
                break;
            case SVC_STI:
                do_cli_sti(true);
                break;
            case SVC_POPF:
                do_popf();
                break;
            case SVC_MOV_CR_READ:
                switch (c->x[R_S0]) {
                    case 0:
                        c->x[R_S0] = c->cr0;
                        break;
                    case 2:
                        c->x[R_S0] = c->cr2;
                        break;
                    case 3:
                        c->x[R_S0] = c->cr3;
                        break;
                    case 4:
                        c->x[R_S0] = c->cr4;
                        break;
                    case 8:
                        c->x[R_S0] = c->apic_tpr >> 4;
                        break;
                    default:
                        c->x[R_S0] = 0;
                }
                break;
            case SVC_MOV_CR_WRITE:
                do_mov_cr_write();
                break;
            case SVC_INVLPG:
            case SVC_INVPCID:
            case SVC_WBINVD:
                break; // the shadow is always in step: see the CR3 note
            case SVC_CLTS:
                if (c->clts_on_host)
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_CLTS, 0, 0);
                c->cr0 &= ~8ULL;
                fp_trap_sync();
                break;
            case SVC_LGDT:
                c->gdtr_base = c->x[R_S0];
                c->gdtr_limit = (u32)c->x[R_S1];
                break;
            case SVC_LIDT:
                c->idtr_base = c->x[R_S0];
                c->idtr_limit = (u32)c->x[R_S1];
                break;
            case SVC_SGDT:
                c->x[R_S0] = c->gdtr_base;
                c->x[R_S1] = c->gdtr_limit;
                break;
            case SVC_SIDT:
                c->x[R_S0] = c->idtr_base;
                c->x[R_S1] = c->idtr_limit;
                break;
            case SVC_LTR: {
                u64 d[2];
                u32 sel = (u32)c->x[R_S0] & 0xffff;
                u32 idx = sel & ~7u;
                if (idx + 15 > c->gdtr_limit || !guest_read(c->gdtr_base + idx, d, 16, 0)) {
                    raise_gp(sel & ~3u, rip_here(c->elr - 4));
                    break;
                }
                c->tr = (u16)sel;
                c->tss_base = ((d[0] >> 16) & 0xffffff) | ((d[0] >> 32) & 0xff000000ULL) | (d[1] << 32);
                c->tss_limit = (u32)((d[0] & 0xffff) | ((d[0] >> 32) & 0xf0000));
                break;
            }
            case SVC_STR:
                c->x[R_S0] = c->tr;
                break;
            case SVC_VERR:
            case SVC_VERW: {
                u32 sel = (u32)c->x[R_S0] & 0xffffu;
                u64 d = 0;
                bool ok = false;
                if ((sel & ~3u) && (!(sel & 4) || (c->ldtr & ~3u)) && desc_read(sel, &d) && (d & DESC_S)) {
                    u32 dpl = desc_dpl(d), rpl = sel & 3;
                    if (n == SVC_VERW) {
                        ok = !(d & DESC_CODE) && (d & DESC_RW) && c->cpl <= dpl && rpl <= dpl;
                    } else if (d & DESC_CODE) {
                        ok = (d & DESC_RW) && ((d & DESC_CONFORMING) || (c->cpl <= dpl && rpl <= dpl));
                    } else {
                        ok = c->cpl <= dpl && rpl <= dpl;
                    }
                }
                if (ok)
                    c->spsr |= (1ULL << 30);
                else
                    c->spsr &= ~(1ULL << 30);
                break;
            }
            case SVC_LSL: {
                u32 sel = (u32)c->x[R_S0] & 0xffffu;
                u64 d = 0;
                bool ok = false;
                if ((sel & ~3u) && (!(sel & 4) || (c->ldtr & ~3u)) && desc_read(sel, &d)) {
                    u32 dpl = desc_dpl(d), rpl = sel & 3;
                    if (d & DESC_S) {
                        ok = ((d & DESC_CODE) && (d & DESC_CONFORMING)) || (c->cpl <= dpl && rpl <= dpl);
                    } else {
                        u32 type = (u32)(d >> 40) & 0xfu;
                        ok = (type == 1 || type == 2 || type == 3 || type == 9 || type == 11) &&
                             c->cpl <= dpl && rpl <= dpl;
                    }
                }
                if (ok) {
                    u32 lim = (u32)((d & 0xffffULL) | ((d >> 32) & 0xf0000ULL));
                    if (d & (1ULL << 55))
                        lim = (lim << 12) | 0xfffu;
                    c->x[R_S0] = lim;
                    c->spsr |= (1ULL << 30);
                } else {
                    c->spsr &= ~(1ULL << 30);
                }
                break;
            }
            case SVC_LLDT: {
                u64 d[2] = {0, 0};
                u32 sel = (u32)c->x[R_S0] & 0xffff, idx = sel & ~7u;
                if ((sel & ~3u) && ((sel & 4) || idx + 15 > c->gdtr_limit || !guest_read(c->gdtr_base + idx, d, 16, 0) || ((d[0] >> 40) & 0x1f) != 2)) {
                    raise_gp(sel & ~3u, rip_here(c->elr - 4));
                    break;
                }
                c->ldtr = (u16)sel;
                c->ldt_base = desc_base(d[0]) | d[1] << 32;
                c->ldt_limit = (u32)desc_limit(d[0]);
                break;
            }
            case SVC_SLDT:
                c->x[R_S0] = c->ldtr;
                break;
            case SVC_MOV_SEG_READ:
                c->x[R_S0] = c->seg_sel[c->x[R_S0] & 7];
                break;
            case SVC_MOV_SEG_WRITE:
                do_mov_seg();
                break;
            case SVC_IN:
            case SVC_OUT: {
                u16 port = (u16)(c->x[R_S0] & 0xffff);
                if (c->hybrid && port == 0x5658) {
                    // VMware backdoor: permit Ring-3 (vm3dum64.dll / guest agents) when QEMU vmport exists
                    // and sync all 7 backdoor GPRs directly without a TCG fallback or TLB flush.
                    c->req_a[0] = port;
                    c->req_a[1] = c->x[R_S1];
                    c->req_a[2] = n == SVC_OUT;
                    hostcall(HC_VMPORT);
                    if (c->req_r[1]) {
                        if (n == SVC_IN)
                            c->x[R_S0] = c->req_r[0];
                        break;
                    }
                    stop(STOP_FALLBACK, rip_here(c->elr - 4), FB_VMPORT, 0, 0);
                }
                u32 iopl = (u32)(c->rflags_sys >> 12) & 3;
                if (c->cpl > iopl) {
                    x86_gp0();
                    break;
                }
                c->req_a[0] = c->x[R_S0];
                c->req_a[1] = c->x[R_S1];
                c->req_a[2] = n == SVC_OUT;
                c->req_a[3] = c->x[R_S2];
                if (c->hybrid) // who wrote a port, for the host to report
                    c->req_a[4] = rip_here(c->elr - 4);
                hostcall(HC_PORT);
                if (n == SVC_IN)
                    c->x[R_S0] = c->req_r[0];
                break;
            }
            case SVC_LDMXCSR: {
                u32 v = (u32)c->x[R_S0];
                if (c->cr0 & 8) {
                    c->stat_nm++;
                    deliver(7, 0, false, rip_here(c->elr - 4), false);
                    break;
                }
                if (v & 0xFFFF0000u) {
                    x86_gp0();
                    break;
                }
                if ((v & 0x1F80) != 0x1F80)
                    stop(STOP_UNSUPPORTED, rip_here(c->elr - 4), 14, v, 0); // unmasked FP exceptions cannot trap here
                mxcsr_apply(v);
                break;
            }
            case SVC_STMXCSR:
                if (c->cr0 & 8) {
                    c->stat_nm++;
                    deliver(7, 0, false, rip_here(c->elr - 4), false);
                    break;
                }
                c->x[R_S0] = mxcsr_read();
                break;
            case SVC_FXSAVE:
                do_fx(true);
                break;
            case SVC_FXRSTOR:
                do_fx(false);
                break;
            case SVC_X87:
                x87_op();
                break;
            case SVC_H_SHIFT:
                h_shift();
                break;
            case SVC_H_PCMPISTRI:
                h_pcmpistri();
                break;
            case SVC_H_SHD:
                h_shd();
                break;
            case SVC_H_DIV:
                h_div();
                break;
            default:
                stop(STOP_UNSUPPORTED, 0, 15, n, c->elr);
        }
    } else if (ec == 0x07) { // SIMD with CR0.TS set
        if (in_rep_stub(c->elr)) { // no x86 FP here, only rep stos's stub, which hybrid mode does not use: on with the plain loop at the call site
            c->elr = c->x[R_LR];
        } else {
            c->stat_nm++;
            deliver(7, 0, false, rip_here(c->elr), false);
        }
    } else if (ec == 0x32) { // software step
        if (pt_step_slot)
            shadow_write_finish();
    } else if (ec == 0x20 || ec == 0x22 || ec == 0x26) {
        stop(STOP_MONITOR_FAULT, 0, c->esr, c->far, c->elr); // translated code cannot fetch or align wrong: an engine bug
    } else {
        stop(STOP_MONITOR_FAULT, 0, c->esr, c->far, c->elr);
    }
    // An invalidated block's entry trap relinks the same x86 instruction;
    // it has not executed even if the new ARM PC is an exact x86 boundary.
    // A retryable data abort or internal REP service can also make no progress.
    exit_check(ec == 0x32 || (ec == 0x15 && !page_rep && n != SVC_DEAD), false);
    if (c->profile_el1_ticks) {
        u64 total = RSR(cntvct_el0) - prof_t0;
        u64 hvc_dt = me->hvc_ticks_local - prof_hvc0;
        u64 el1_dt = total > hvc_dt ? (total - hvc_dt) : 0;
        if (ec == 0x24)
            c->stat_el1_sync_dabt_ticks += el1_dt;
        else
            c->stat_el1_sync_svc_ticks += el1_dt;
    }
}

void mon_irq(void) {
    u64 prof_t0 = c->profile_el1_ticks ? RSR(cntvct_el0) : 0;
    u64 prof_hvc0 = c->profile_el1_ticks ? me->hvc_ticks_local : 0;
    me->t2_synced = false;
    if (me->smc_held)
        __atomic_store_n(&me->smc_held, 0, __ATOMIC_RELEASE);
    // Lockless 4-instruction Tier-2 hot-loop PC sample (zero EL0 instrumentation, zero false sharing).
    u64 sample_pc = c->elr;
    if (sample_pc >= PRIV_VA + OFF_CC && sample_pc < PRIV_VA + OFF_CC2 + CC_SIZE) {
        u32 sh = c->t2_sample_head;
        c->t2_sample_ring[sh & (T2_SAMPLE_RING - 1U)] = sample_pc;
        __atomic_store_n(&c->t2_sample_head, sh + 1U, __ATOMIC_RELEASE);
    }
    // The host raised the virtual IRQ line: a timer expiry, new vectors in the host IRR, an ExtINT, or a request to stop.
    // The timer rearms first, because the host unmasks the virtual timer in this call and an expired compare would fire again.
    // Always exit via HC_IRQ in hybrid mode so Hypervisor.framework drops the level-asserted virtual IRQ line before ERET.
    apic_timer_check();
    if (!c->hybrid)
        sync_host_irq(true);
    apic_merge_host();
    exit_check(false, c->hybrid);
    if (c->profile_el1_ticks) {
        u64 total = RSR(cntvct_el0) - prof_t0;
        u64 hvc_dt = me->hvc_ticks_local - prof_hvc0;
        c->stat_el1_irq_ticks += total > hvc_dt ? (total - hvc_dt) : 0;
    }
}

void mon_fiq(void) {
    me->t2_synced = false;
    stop(STOP_MONITOR_FAULT, 0, 0, 0, c->elr);
}

// Set by the host once: runs first, on the initial entry.
void mon_init(void) {
    zap_why = KILL_RECLAIM;
    c->irq_shadow_rip = ~0ULL;
    WSR(cntkctl_el1, 2); // EL0VCTEN: rdtsc in translated code reads the virtual counter
    u64 *desc_hole = (u64 *)PRIV(OFF_MON_DATA + (1ULL << 19));
    hole_entry_priv = desc_hole[0];
    hole_entry_ram = desc_hole[1];
    View *v = view_find(cr3_key(c->cr3), view_mode());
    c->view = (u32)(v - views);
    if (!me->rs_saved[c->view]) // built by another vCPU
        view_rs_prepare(c->view);
    c->x[R_RS] = me->rs_saved[c->view];
    c->rs_top = view_rs_top(c->view);
    c->x[R_HASH] = v->hash_va;
    c->x[R_CTX] = (u64)c;
    WSR(ttbr0_el1, v->ttbr0);
    WSR(ttbr1_el1, v->ttbr1);
    __asm__ volatile("isb");
    mxcsr_apply(c->hybrid ? c->resume_mxcsr : 0x1F80);
    if (!cpu_index()) {
        // Measure the FPCR Hypervisor actually exposes, not the host model.
        // All vCPUs in this VM use that feature set and mxcsr_apply's policy.
        c->fpcr_merges_scalar = (RSR(S3_3_C4_C4_0) & 4) != 0;
    }
    if (!c->hybrid && !c->fcw)
        c->fcw = 0x37F;
    fp_trap_sync();
    u64 entry = c->req_a[0];
    c->elr = translate_here(entry, 0, 0, false);
    exit_check(false, false);
    mon_enter_guest();
}

