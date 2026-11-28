// EL1 exception vectors and entry and exit paths for the monitor, then the stubs translated code calls at EL0.
// Every entry comes from EL0, where x18 always holds EngineCtx. An exception taken at EL1 is a monitor bug.
.set CTX_ELR, 248
.set CTX_SPSR, 256
.set CTX_ESR, 264
.set CTX_FAR, 272
.set CTX_DF, 280
.set CTX_RS_TOP, 296
.set CTX_RFLAGS_SYS, 304
.set CTX_CODE_GEN, 30680 // engine_abi.h asserts both
.set CTX_CODE_SYNCED, 30688
.set CTX_YMM_HI, 30784 // ymm0-7 high halves live in v16-v23; spill them before C code
.set SVC_LOOKUP_MISS, 4
.set SVC_REP_PAGE, 71
.set HASH_MASK, 0xFFFF

.macro SAVE_ALL
    stp x0, x1, [x18, #0]
    stp x2, x3, [x18, #16]
    stp x4, x5, [x18, #32]
    stp x6, x7, [x18, #48]
    stp x8, x9, [x18, #64]
    stp x10, x11, [x18, #80]
    stp x12, x13, [x18, #96]
    stp x14, x15, [x18, #112]
    stp x16, x17, [x18, #128]
    str x19, [x18, #152]
    stp x20, x21, [x18, #160]
    stp x22, x23, [x18, #176]
    stp x24, x25, [x18, #192]
    stp x26, x27, [x18, #208]
    stp x28, x29, [x18, #224]
    str x30, [x18, #240]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x0, x1, [x18, #CTX_ELR]
    mrs x0, esr_el1
    mrs x1, far_el1
    stp x0, x1, [x18, #CTX_ESR]
    // ymm0-7 highs are resident in v16-v23. Spill before the C monitor uses those regs.
    movz x0, #CTX_YMM_HI
    add x0, x18, x0
    stp q16, q17, [x0, #0]
    stp q18, q19, [x0, #32]
    stp q20, q21, [x0, #64]
    stp q22, q23, [x0, #96]
.endm

.macro ENT target
    .p2align 7
    b \target
.endm

    .section __TEXT,__text,regular,pure_instructions
    .globl _mon_vectors
    .p2align 11
_mon_vectors:
    // Current EL with SP0: never used.
    ENT Lbug
    ENT Lbug
    ENT Lbug
    ENT Lbug
    // Current EL with SPx: a monitor fault.
    ENT Lbug
    ENT Lbug
    ENT Lbug
    ENT Lbug
    // Lower EL, AArch64.
    ENT Lsync
    ENT Lirq
    ENT Lfiq
    ENT Lserror
    // Lower EL, AArch32: not reachable.
    ENT Lbug
    ENT Lbug
    ENT Lbug
    ENT Lbug

Lbug:
    SAVE_ALL
    mrs x0, elr_el1
    bl _mon_bug
1:  b 1b

Lsync:
    SAVE_ALL
    bl _mon_sync
    b _mon_enter_guest

Lirq:
    SAVE_ALL
    bl _mon_irq
    b _mon_enter_guest

Lfiq:
    SAVE_ALL
    bl _mon_fiq
    b _mon_enter_guest

Lserror:
    SAVE_ALL
    bl _mon_bug
1:  b 1b

// The host starts every vCPU here, and every monitor path leaves here.
    .globl _mon_enter_guest
_mon_enter_guest:
    // Reload ymm0-7 highs. FXRSTOR and a TCG handoff may have rewritten ymm_hi.
    movz x0, #CTX_YMM_HI
    add x0, x18, x0
    ldp q16, q17, [x0, #0]
    ldp q18, q19, [x0, #32]
    ldp q20, q21, [x0, #64]
    ldp q22, q23, [x0, #96]
    ldp x0, x1, [x18, #CTX_ELR]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0, x1, [x18, #0]
    ldp x2, x3, [x18, #16]
    ldp x4, x5, [x18, #32]
    ldp x6, x7, [x18, #48]
    ldp x8, x9, [x18, #64]
    ldp x10, x11, [x18, #80]
    ldp x12, x13, [x18, #96]
    ldp x14, x15, [x18, #112]
    ldp x16, x17, [x18, #128]
    ldr x19, [x18, #152]
    ldp x20, x21, [x18, #160]
    ldp x22, x23, [x18, #176]
    ldp x24, x25, [x18, #192]
    ldp x26, x27, [x18, #208]
    ldp x28, x29, [x18, #224]
    ldr x30, [x18, #240]
    // ERET alone does not discard instructions an H15 core fetched before
    // another vCPU's write: relinks, lookups and stale-fetch retries return here.
    isb
    eret

// A stop that came back: start over on an empty stack, whatever depth the monitor stopped at.
    .globl _mon_restart
_mon_restart:
    mrs x0, tpidr_el1 // the host keeps this vCPU's stack top here
    mov sp, x0
    bl _mon_after_stop
    b _mon_enter_guest

// Host call: hvc #code. The host runs with the vCPU stopped here and resumes at the next instruction.
    .globl _mon_hvc
_mon_hvc:
    hvc #0
    ret

    .globl _mon_fp_save // x0: 256 bytes
_mon_fp_save:
    stp q0, q1, [x0, #0]
    stp q2, q3, [x0, #32]
    stp q4, q5, [x0, #64]
    stp q6, q7, [x0, #96]
    stp q8, q9, [x0, #128]
    stp q10, q11, [x0, #160]
    stp q12, q13, [x0, #192]
    stp q14, q15, [x0, #224]
    ret
    .globl _mon_fp_restore
_mon_fp_restore:
    ldp q0, q1, [x0, #0]
    ldp q2, q3, [x0, #32]
    ldp q4, q5, [x0, #64]
    ldp q6, q7, [x0, #96]
    ldp q8, q9, [x0, #128]
    ldp q10, q11, [x0, #160]
    ldp q12, q13, [x0, #192]
    ldp q14, q15, [x0, #224]
    ret

// Tier-2 resident YMM8..15 pool reader ({q24, q25, q26, q29, q30, q31} -> 96B buffer in x0).
// Index: 0=q24, 1=q25, 2=q26, 3=q29, 4=q30, 5=q31.
// Monitor C code is built with -mgeneral-regs-only and SAVE_ALL/mon_fp_save never touch q24..q31,
// so zero instructions are added to the normal trap path; called lazily only on Tier-2 body trap exit.
    .globl _mon_t2_read_pool
_mon_t2_read_pool:
    stp q24, q25, [x0, #0]
    stp q26, q29, [x0, #32]
    stp q30, q31, [x0, #64]
    ret

// ---- EL0 stubs. Fixed offsets, the translator calls them by address; see EngineLayout. ----
    .section __EL0,__stubs,regular,pure_instructions
    .p2align 12
    .globl _stub_base
_stub_base:
// x16: x86 target. Returns the translation in x22. Keeps NZCV. Clobbers x17, x20, x23.
    .globl _stub_lookup
_stub_lookup:
    crc32cx w17, wzr, x16
    and w17, w17, #HASH_MASK
1:  add x23, x19, x17, lsl #4
    ldp x20, x22, [x23]
    cbz x20, 2f
    eor x20, x20, x16
    cbnz x20, 3f
    cbz x22, 2f // a matching key whose value isn't there: let the monitor resolve it, never branch to 0
    eor x20, x22, x22
    ldr x20, [x23, x20]
    eor x20, x20, x16
    cbnz x20, 2f
    bic x22, x22, #1 // strip bit-0 FEX tag for native HVDBT callers
    // A hit may be code another vCPU or the host published after this vCPU last
    // synchronized. Only its own ISB makes its fetch of that code current: Apple M3
    // (H15) cores otherwise run instructions fetched before the write. The host
    // bumps code_gen before publishing; reading it with an address dependency on
    // the value keeps it from being read before the entry. No NZCV change.
    eor x23, x22, x22
    add x23, x18, x23
    ldr x17, [x23, #CTX_CODE_GEN]
    ldr x23, [x18, #CTX_CODE_SYNCED]
    eor x23, x23, x17
    cbnz x23, 4f
    ret
4:  str x17, [x18, #CTX_CODE_SYNCED]
    isb
    ret
3:  add w17, w17, #1
    and w17, w17, #HASH_MASK
    b 1b
2:  svc #SVC_LOOKUP_MISS
    // The monitor resumes at x30 with x22 set, so nothing follows.
    brk #0

    .org _stub_base + 0x80
// A return whose shadow entry did not match: drop the shadow stack, the guest stack stays authoritative.
    .globl _stub_ret_miss
_stub_ret_miss:
    ldr x26, [x18, #CTX_RS_TOP]
    bl _stub_lookup
    br x22

    .org _stub_base + 0x100
// x16 = RFLAGS. Keeps NZCV. Clobbers x17, x20.
    .globl _stub_pushf
_stub_pushf:
    mrs x17, nzcv
    ldr x16, [x18, #CTX_RFLAGS_SYS]
    ubfx x20, x17, #29, #1
    eor x20, x20, #1
    orr x16, x16, x20
    ubfx x20, x17, #30, #1
    orr x16, x16, x20, lsl #6
    ubfx x20, x17, #31, #1
    orr x16, x16, x20, lsl #7
    ubfx x20, x17, #28, #1
    orr x16, x16, x20, lsl #11
    eor w20, w21, w21, lsr #4
    and w20, w20, #0xf
    mov w17, #0x9669
    lsr w17, w17, w20
    and w17, w17, #1
    orr x16, x16, x17, lsl #2
    eor x20, x25, x21
    and x20, x20, #0x10
    orr x16, x16, x20
    ldr x20, [x18, #CTX_DF]
    tbz x20, #63, 1f
    orr x16, x16, #0x400
1:  ret

    .org _stub_base + 0x180
// x16: the byte for sahf. Keeps V. Clobbers x17, x20.
    .globl _stub_sahf
_stub_sahf:
    mrs x17, nzcv
    and x17, x17, #(1 << 28)
    ubfx x20, x16, #7, #1
    orr x17, x17, x20, lsl #31
    ubfx x20, x16, #6, #1
    orr x17, x17, x20, lsl #30
    ubfx x20, x16, #0, #1
    eor x20, x20, #1
    orr x17, x17, x20, lsl #29
    msr nzcv, x17
    ubfx x20, x16, #2, #1
    eor x21, x20, #1
    and x20, x16, #0x10
    eor x25, x21, x20
    ret
    .org _stub_base + 0x200
// rep movs, 32 bytes at a time while the copy runs forward and cannot see its own output inside a chunk. x17: elements per chunk.
// Registers move only after a chunk is stored, so a fault restarts the x86 instruction with exact state, and redoing a chunk is harmless
// because source and destination are at least a chunk apart both ways. Leaves the tail, and anything else, to the plain loop at the call
// site. No NEON: under CR0.TS it would trap for every element, and nothing in a plain copy would clear TS. Keeps NZCV. Clobbers x16, x20, x22, x23.
    .globl _stub_rep_movs
_stub_rep_movs:
    ldr x16, [x18, #CTX_DF]
    tbnz x16, #63, 2f
    sub x16, x7, x6
    lsr x20, x16, #5
    cbz x20, 2f
    sub x16, x6, x7
    lsr x20, x16, #5
    cbz x20, 2f
    b 1f
    .org _stub_base + 0x240 // the loop head is an x86 boundary: the monitor delivers interrupts here
1:  sub x20, x1, x17
    tbnz x20, #63, 2f
    ldp x22, x23, [x6]
    stp x22, x23, [x7]
    ldp x22, x23, [x6, #16]
    stp x22, x23, [x7, #16]
    add x6, x6, #32
    add x7, x7, #32
    mov x1, x20
    b 1b
2:  ret


    .org _stub_base + 0x280
// rep stos with the pattern already in x22. No NEON, as for movs: a kernel zeroes pages with CR0.TS set.
    .globl _stub_rep_stos
_stub_rep_stos:
    ldr x16, [x18, #CTX_DF]
    tbnz x16, #63, 2f
    b 1f
    .org _stub_base + 0x2C0 // boundary, as above
1:  sub x20, x1, x17
    tbnz x20, #63, 2f
    stp x22, x22, [x7]
    stp x22, x22, [x7, #16]
    add x7, x7, #32
    mov x1, x20
    b 1b
2:  ret

    .org _stub_base + 0x300
// Experimental page REP: the service either commits a page prefix or returns to
// the emitter's scalar loop. It never enters the original bulk stubs above.
    .globl _stub_rep_page_movs
_stub_rep_page_movs:
    ldr x16, [x18, #CTX_DF]
    tbnz x16, #63, 2f
    sub x16, x7, x6
    lsr x20, x16, #5
    cbz x20, 2f
    sub x16, x6, x7
    lsr x20, x16, #5
    cbz x20, 2f
    sub x20, x1, x17
    tbnz x20, #63, 2f
    b 1f
2:  ret
    .org _stub_base + 0x340
1:  svc #SVC_REP_PAGE
    ret

    .org _stub_base + 0x380
    .globl _stub_rep_page_stos
_stub_rep_page_stos:
    ldr x16, [x18, #CTX_DF]
    tbnz x16, #63, 2f
    sub x20, x1, x17
    tbnz x20, #63, 2f
    b 1f
2:  ret
    .org _stub_base + 0x3C0
1:  svc #SVC_REP_PAGE
    ret
    .org _stub_base + 0x400
// Shared FEX entry trampoline: x16 = entry_fex_body. Preserves NZCV.
    .globl _stub_fex_enter
_stub_fex_enter:
    add x17, x18, #36864
    stp x19, x26, [x17, #0]
    stp x27, x28, [x17, #16]
    stp x29, x30, [x17, #32]
    mov x20, x16
    mov x17, x28
    add x28, x18, #32768
    str x27, [x28, #0x3e0]
    str x17, [x28, #0x3e8]
    ldr x17, [x18, #312]
    add x17, x17, #1
    str x17, [x18, #312]
    add sp, x18, #16, lsl #12
    ldr x17, [x18, #CTX_DF]
    strb w17, [x28, #0x3fa]
    stp q16, q17, [x28, #0xc0]
    stp q18, q19, [x28, #0xe0]
    stp q20, q21, [x28, #0x100]
    stp q22, q23, [x28, #0x120]
    mov w17, #30912
    add x17, x18, x17
    ldp q24, q25, [x17, #0]
    stp q24, q25, [x28, #0x140]
    ldp q24, q25, [x17, #32]
    stp q24, q25, [x28, #0x160]
    ldp q24, q25, [x17, #64]
    stp q24, q25, [x28, #0x180]
    ldp q24, q25, [x17, #96]
    stp q24, q25, [x28, #0x1a0]
    mov v16.16b, v0.16b
    mov v17.16b, v1.16b
    mov v18.16b, v2.16b
    mov v19.16b, v3.16b
    mov v20.16b, v4.16b
    mov v21.16b, v5.16b
    mov v22.16b, v6.16b
    mov v23.16b, v7.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    mov v28.16b, v12.16b
    mov v29.16b, v13.16b
    mov v30.16b, v14.16b
    mov v31.16b, v15.16b
    mov w26, w21
    mov w27, w25
    ldr x25, [x18, #296]
    mvn x17, xzr
    str x17, [x25]
    mov x29, x15
    mov x19, x14
    mov x17, x13
    mov x16, x12
    mov x15, x11
    mov x14, x10
    mov x13, x9
    mov x12, x8
    mov x11, x7
    mov x10, x6
    mov x9, x5
    mov x8, x4
    mov x7, x1
    mov x6, x3
    mov x5, x2
    mov x4, x0
    br x20

    .org _stub_base + 0x5F8
    .globl _stub_fex_exit_linker
_stub_fex_exit_linker:
    ldr x0, [x30, #8]
    str x0, [x28, #0x18]

    .org _stub_base + 0x600
// Shared FEX exit trampoline with Stay-in-FEX + tear-free FEX_L1 fill. Preserves NZCV.
    .globl _stub_fex_exit
_stub_fex_exit:
    ldr x0, [x28, #0x18]
    ldr x1, [x28, #4096]
    crc32cx w2, wzr, x0
    and w2, w2, #HASH_MASK
Lfex_hash_loop:
    add x3, x1, x2, lsl #4
    ldp x20, x22, [x3]
    cbz x20, Lfex_exit_native
    eor x20, x20, x0
    cbnz x20, Lfex_hash_probe
    cbz x22, Lfex_exit_native
    eor x20, x22, x22
    ldr x20, [x3, x20]
    eor x20, x20, x0
    cbnz x20, Lfex_exit_native
    tbz x22, #0, Lfex_exit_native
    bic x22, x22, #1
    eor x23, x22, x22
    add x23, x18, x23
    ldr x20, [x23, #CTX_CODE_GEN]
    ldr x23, [x18, #CTX_CODE_SYNCED]
    eor x23, x23, x20
    cbnz x23, Lfex_code_sync
Lfex_stay_fill:
    add x22, x22, #12
    add x20, x18, #0x400, lsl #12
    ubfiz x21, x0, #4, #12
    add x20, x20, x21
    stp x22, x0, [x20]
    dmb ish
    ldr x21, [x3]
    eor x21, x21, x0
    cbnz x21, Lfex_l1_undo
    ldr x20, [x18, #320]
    add x20, x20, #1
    str x20, [x18, #320]
    add sp, x18, #16, lsl #12
    br x22

    .org _stub_base + 0x690 // STUB_OFF_FEX_EXIT_NATIVE
Lfex_exit_native:
    ldr x20, [x18, #328]
    add x20, x20, #1
    str x20, [x18, #328]
    mov v0.16b, v16.16b
    mov v1.16b, v17.16b
    mov v2.16b, v18.16b
    mov v3.16b, v19.16b
    mov v4.16b, v20.16b
    mov v5.16b, v21.16b
    mov v6.16b, v22.16b
    mov v7.16b, v23.16b
    mov v8.16b, v24.16b
    mov v9.16b, v25.16b
    mov v10.16b, v26.16b
    mov v11.16b, v27.16b
    mov v12.16b, v28.16b
    mov v13.16b, v29.16b
    mov v14.16b, v30.16b
    mov v15.16b, v31.16b
    ldp q16, q17, [x28, #0xc0]
    ldp q18, q19, [x28, #0xe0]
    ldp q20, q21, [x28, #0x100]
    ldp q22, q23, [x28, #0x120]
    mov w20, #30912
    add x20, x18, x20
    ldp q24, q25, [x28, #0x140]
    stp q24, q25, [x20, #0]
    ldp q24, q25, [x28, #0x160]
    stp q24, q25, [x20, #32]
    ldp q24, q25, [x28, #0x180]
    stp q24, q25, [x20, #64]
    ldp q24, q25, [x28, #0x1a0]
    stp q24, q25, [x20, #96]
    mov w21, w26
    mov w25, w27
    ldrsb x20, [x28, #0x3fa]
    str x20, [x18, #CTX_DF]
    ldr x22, [x28, #0x18]
    mov x0, x4
    mov x1, x7
    mov x2, x5
    mov x3, x6
    mov x4, x8
    mov x5, x9
    mov x6, x10
    mov x7, x11
    mov x8, x12
    mov x9, x13
    mov x10, x14
    mov x11, x15
    mov x12, x16
    mov x13, x17
    mov x14, x19
    mov x15, x29
    mov x16, x22
    add x17, x18, #36864
    ldp x19, x26, [x17, #0]
    ldr x26, [x18, #296]
    mvn x24, xzr
    str x24, [x26]
    ldp x29, x30, [x17, #32]
    ldp x27, x28, [x17, #16]
    mov x24, x30
    bl _stub_lookup
    mov x30, x24
    br x22

    .org _stub_base + 0x800
Lfex_hash_probe:
    add w2, w2, #1
    and w2, w2, #HASH_MASK
    b Lfex_hash_loop
Lfex_code_sync:
    str x20, [x18, #CTX_CODE_SYNCED]
    isb
    ldr x25, [x18, #296]
    mvn x20, xzr
    str x20, [x25]
    b Lfex_stay_fill
Lfex_l1_undo:
    str xzr, [x20, #8]
    b Lfex_exit_native

    .org _stub_base + 0x830
// Fast FEX_L1 lookup for unlinked/cross-page direct thunks (1 load, 0 stores on hit, 0 dmb).
// Entry: x0 = target_rip. Preserves NZCV, x25, x30.
    .globl _stub_fex_direct_exit
_stub_fex_direct_exit:
    cbz x0, Lfex_direct_miss
    add x20, x18, #0x400, lsl #12
    ubfiz x21, x0, #4, #12
    add x20, x20, x21
    ldp x22, x21, [x20]
    eor x21, x21, x0
    cbnz x21, Lfex_direct_miss
    br x22
Lfex_direct_miss:
    str x0, [x28, #0x18]
    b _stub_fex_exit

    .org _stub_base + 0x860
// Pure-native CPL=3 lookup with tear-free fill into per-vCPU OFF_FEX_L1 (x18 + 4 MiB).
// Entry: x16 = x86 target (0 -> SVC_LOOKUP_MISS). Returns translation in x22.
// Keeps NZCV, x16, x18, x19, x24, x26, x30. Clobbers x17, x20, x23.
    .globl _stub_lookup_l1
_stub_lookup_l1:
    cbz x16, 2f
    crc32cx w17, wzr, x16
    and w17, w17, #HASH_MASK
1:  add x23, x19, x17, lsl #4
    ldp x20, x22, [x23]
    cbz x20, 2f
    eor x20, x20, x16
    cbnz x20, 3f
    cbz x22, 2f
    eor x20, x22, x22
    ldr x20, [x23, x20]
    eor x20, x20, x16
    cbnz x20, 2f
    bic x22, x22, #1
    // Code publication ISB check (keeps x23 = hash slot ptr for tear-free L1 fill).
    eor x20, x22, x22
    add x20, x18, x20
    ldr x17, [x20, #CTX_CODE_GEN]
    ldr x20, [x18, #CTX_CODE_SYNCED]
    eor x20, x20, x17
    cbnz x20, 4f
5:  // Tear-free fill into per-vCPU OFF_FEX_L1 using (rip ^ (rip >> 4)) & 0xFFF.
    add x20, x18, #0x400, lsl #12
    eor x17, x16, x16, lsr #4
    ubfiz x17, x17, #4, #12
    add x20, x20, x17
    stp x22, x16, [x20]
    dmb ish
    ldr x17, [x23]
    eor x17, x17, x16
    cbnz x17, 6f
    ret
6:  str xzr, [x20, #8]
    b 2f
4:  str x17, [x18, #CTX_CODE_SYNCED]
    isb
    b 5b
3:  add w17, w17, #1
    and w17, w17, #HASH_MASK
    b 1b
2:  svc #SVC_LOOKUP_MISS
    brk #0

    .org _stub_base + 0x900
// Fast per-vCPU L1 probe stub for OP_CALLI (saves 10 ARM64 words / 40 B per indirect call site).
// Entry: x16 = x86 target. Returns translation in x22. Preserves NZCV, x16, x18, x19, x24, x26, x30.
    .globl _stub_lookup_l1_fast
_stub_lookup_l1_fast:
    cbz x16, _stub_lookup_l1
    add x20, x18, #0x400, lsl #12
    eor x17, x16, x16, lsr #4
    ubfiz x17, x17, #4, #12
    add x20, x20, x17
    ldp x22, x23, [x20]
    eor x23, x23, x16
    cbnz x23, _stub_lookup_l1
    ret

    .org _stub_base + 0x930
// CPL=3 64-bit return-stack miss: probe per-vCPU L1 before falling back to L2 hash.
    .globl _stub_ret_miss_l1
_stub_ret_miss_l1:
    ldr x26, [x18, #CTX_RS_TOP]
    bl _stub_lookup_l1_fast
    br x22
