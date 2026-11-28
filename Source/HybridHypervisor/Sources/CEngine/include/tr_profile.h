// HVDBT Full-Stack Profiler (EL0 Tier-1/Tier-2/Stub + EL1 Monitor + HVF Host Slicing).
// Activated ONLY when LC_HVDBT_PROFILE=<out_prefix> is set; zero runtime overhead when unset.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "engine_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-sample record captured on host in HV_EXIT_REASON_CANCELED (1-4 kHz) without injecting an EL1 IRQ.
typedef struct {
    uint64_t pc;   // HV_REG_PC at cancel exit
    uint64_t elr;  // HV_SYS_REG_ELR_EL1 when el != 0 (0 when el == 0)
    uint64_t lr;   // HV_REG_LR when el != 0 (0 when el == 0)
    uint64_t cr3;  // EngineCtx.cr3 at sample time
    uint32_t cpu;  // vCPU index
    uint16_t el;   // (CPSR >> 2) & 3 (0 = EL0, 1 = EL1)
    uint16_t cpl;  // EngineCtx.cpl (0 or 3)
} TrProfSample;

// Per-vCPU counter snapshot for exact per-workload delta slicing (EL0 vs EL1 vs HVF Host).
typedef struct {
    uint32_t cpu;
    uint32_t pad;
    uint64_t ticks_in_vcpu;       // Engine.ticksInVCPU (24 MHz host ticks)
    uint64_t el1_sync_svc_ticks;  // EngineCtx.stat_el1_sync_svc_ticks
    uint64_t el1_sync_dabt_ticks; // EngineCtx.stat_el1_sync_dabt_ticks
    uint64_t el1_irq_ticks;       // EngineCtx.stat_el1_irq_ticks
    uint64_t el1_hvc_ticks;       // EngineCtx.stat_el1_hvc_ticks
    uint64_t translator_wait_ticks;
    uint64_t vm_exits;
    uint64_t shadow_fills;
    uint64_t cr3_writes;
    uint64_t syscalls;
    uint64_t irets;
    uint64_t smc_events;
    uint64_t mon_wait_ticks;
    uint64_t hc_count[32];
    uint64_t hc_ticks[32];
    uint64_t svc_count[128];
} TrProfVcpuSnap;

// Tier-2 eligibility / rejection diagnosis for a hot Tier-1 block.
typedef enum {
    TR_PROF_T2_PROMOTED = 0,
    TR_PROF_T2_ELIGIBLE_NOT_PROMOTED,
    TR_PROF_T2_NO_BACKEDGE,
    TR_PROF_T2_HAS_CALL_RET,
    TR_PROF_T2_FORBIDDEN_OP,
    TR_PROF_T2_TOO_LARGE,
    TR_PROF_T2_CROSS_PAGE,
    TR_PROF_T2_NESTED_LOOP,
    TR_PROF_T2_UNREADABLE_GUEST,
} TrProfT2Diag;

// Initialize profiler subsystem from environment (LC_HVDBT_PROFILE, LC_HVDBT_PROFILE_HZ).
void tr_profile_init_mem(uint8_t *priv, uint8_t *ram, uint64_t ram_size, uint64_t ram_below_4g,
                         EngineCtx *ctx0, const char *explicit_prefix);
void tr_profile_shutdown(void);

// Zero-cost queries for Swift Engine / Monitor setup.
bool tr_profile_enabled(void);
uint32_t tr_profile_hz(void);
const char *tr_profile_prefix(void);

// Atomic flag & SIGUSR1 helpers (Swift 6 strict-concurrency friendly).
static inline void tr_profile_flag_set(uint32_t *p) {
    __atomic_store_n(p, 1U, __ATOMIC_RELEASE);
}
static inline bool tr_profile_flag_take(uint32_t *p) {
    return __atomic_exchange_n(p, 0U, __ATOMIC_ACQ_REL) != 0U;
}
void tr_profile_install_sigusr1(void);
bool tr_profile_take_sigusr1(void);

// Register EL1 Monitor text symbol (va in [PRIV_VA, PRIV_VA + OFF_MON_DATA)) from MachOImage.
void tr_profile_register_mon_sym(uint64_t va, const char *name);

// Lock-free per-vCPU sample append (called from vCPU thread inside HV_EXIT_REASON_CANCELED).
void tr_profile_record_sample(const TrProfSample *s);

// Clear accumulated sample ring & capture baseline snapshot for a fresh workload slice.
void tr_profile_reset(const TrProfVcpuSnap *baseline_snaps, uint32_t nvcpus);

// Dump full analysis for a workload slice `tag` (writes <prefix>.<tag>.{hotblocks.txt,flamegraph.folded,summary.json}).
int64_t tr_profile_dump(const char *tag, const TrProfVcpuSnap *cur_snaps, uint32_t nvcpus);

// Classify a single ARM64 instruction word into a concise human mnemonic.
void tr_profile_fmt_arm64(uint32_t word, char *buf, size_t cap);

// Human-readable name for TrProfT2Diag.
const char *tr_profile_t2_diag_str(TrProfT2Diag d);

#ifdef __cplusplus
}
#endif
