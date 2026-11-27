# VirtualIntelMacOnAppleSilicon — Windows x86-64 Edition

> **This fork is specifically optimized for running x86-64 Windows (`Windows 10`, `Windows 11`, `Windows Server 2022`, and `WinPE`) on Apple Silicon via pure-software full-system binary translation (HVDBT, zero external FEX dependencies).**
>
> Derived from [`nfzerox/VirtualIntelMacOnAppleSilicon`](https://github.com/nfzerox/VirtualIntelMacOnAppleSilicon) (MIT License), pairing Apple `Hypervisor.framework` with an in-tree pure-C / ARM64 NEON dynamic binary translator and freestanding EL1 monitor.

---

## Overview

This repository adapts the [Hybrid Hypervisor](Source/HybridHypervisor) full-system x86-64 engine and QEMU HVDBT handoff layer for **x86-64 Windows guests** on Apple Silicon Macs:

- **100% Pure-Native & Self-Contained**: Pure C / ARM64 NEON x86-64 translation engine (`CEngine` + freestanding EL1 `Monitor` + Swift `Hypervisor.framework` runner) with **zero external FEXCore / fmt / xxhash dependencies** on `main`.
- **Windows Kernel & ISA Focus**: Targeted fast paths for NT kernel scheduling, CR3 context switches, lockless HPET reads, deferred edge-triggered LAPIC EOIs, 44 RAPL/thermal/platform MSRs, WoW64 segment transitions, and full AVX/AVX2/FMA3/BMI1/BMI2/ADX/AES-NI/SHA-NI user-mode workloads.
- **Branch Layout**:
  - `main`: Pure-native, self-contained Windows x86-64 optimized engine and QEMU HVDBT integration.
  - [`fex-frozen`](https://github.com/AotianNeo/VirtualIntelMacOnAppleSilicon/tree/fex-frozen): Frozen archive of the earlier experimental FEXCore JIT bridge exploration.

---

## Measured Benchmarks (Apple M5 Pro, 16-vCPU Windows Server 2022)

Measured inside a **16-vCPU Windows Server 2022** guest (`LC_HVDBT=1 LC_HVDBT_FAST_FLAGS=1`) on an **Apple M5 Pro** host:

### 7-Zip 26.06 Benchmark

| Configuration | Score (MIPS) | Comparison |
|---|---|---|
| **1-Thread** (`-mmt1 -md22`) | **`8,292 MIPS`** | **`+34.8%`** vs macOS Rosetta 2 (`6,152 MIPS`) · **`85.7%`** of native ARM64 (`9,680 MIPS`) |
| **16-Thread** (`-mmt16 -md24`) | **`110,011 MIPS`** | **`+29.1%`** vs macOS Rosetta 2 (`85,240 MIPS`) |

### Geekbench 7.1.0 Pro AVX2

| Metric | Historical Baseline | Optimized `main` | Improvement |
|---|---|---|---|
| **Single-Core** | `419` | **`1,611`** | **`3.84x`** (`16/16` valid workloads) |
| **Multi-Core** | `1,897` | **`11,768`** | **`6.20x`** (`8/8` valid workloads) |
| **Code-Cache Global Flushes (`all=1`)** | Frequent | **`0`** | Zero full-cache flushes across the entire SC + MC run |

---

## Core Architectural Optimizations for Windows

1. **Pure-C AVX/AVX2/FMA/BMI/ADX Translator & `YMM8..15_HI` Register Cache**
   - Vendored **Zydis v4.1.0** decoder ([`x86_decode.c`](Source/HybridHypervisor/Sources/CEngine/x86_decode.c)) paired with a pure-C ARM64/NEON emitter ([`emit_ext.c`](Source/HybridHypervisor/Sources/CEngine/emit_ext.c)) using MIT-licensed Box64 encoding primitives ([`arm64_emitter.h`](Source/HybridHypervisor/Sources/CEngine/third_party/box64/arm64_emitter.h)).
   - Resident `YMM0..7` upper 128-bit lanes in NEON `v16..v23` (spilled/reloaded across monitor transitions via `CTX_YMM_HI` in [`vectors.s`](Source/HybridHypervisor/Monitor/vectors.s)), paired with a 4-slot LRU dirty-tracking pool and `ymm_zero_mask` for `YMM8..15_HI`.
   - Complete `XSAVE`/`XRSTOR` 256-bit `YMM` state plumbing ([`x87.c`](Source/HybridHypervisor/Monitor/x87.c), [`EngineAPI.swift`](Source/HybridHypervisor/Sources/Engine/EngineAPI.swift)) and `log2sz(16)` fix in [`tr_internal.h`](Source/HybridHypervisor/Sources/CEngine/tr_internal.h) for full 128-bit memory loads on `VPMOV{S,Z}X* ymm, m128`.

2. **Forward-DAG, 9-Instruction Inline L1 Indirect Cache & `x26` Hardware RAS**
   - CPL-tiered Forward-DAG translation ([`translate.c`](Source/HybridHypervisor/Sources/CEngine/translate.c)) with compact codegen and EL0 inlining of narrow 8/16-bit shifts/rotates and 32/64-bit `ROL`/`ROR` by `CL` ([`emit_int.c`](Source/HybridHypervisor/Sources/CEngine/emit_int.c)), `LDMXCSR` fast-path skip, and pure-NEON EL0 `PCMPISTRI`/`VPCMPISTRI` ([`emit_sse.c`](Source/HybridHypervisor/Sources/CEngine/emit_sse.c)).
   - 9-instruction `_stub_lookup_l1_fast` (`0x900` in [`vectors.s`](Source/HybridHypervisor/Monitor/vectors.s)) probing per-vCPU 64 KB `OFF_FEX_L1` (`x18 + 4 MiB`), shrinking shared-partition block footprint from `520 B` to `289.7 B/block` (`-44.3%`) alongside the `x26` hardware return-address stack (`OFF_RSTACKS`).

3. **EL1 Lockless HPET / Deferred LAPIC EOI / CR3 / RAPL-MSR Fast Paths**
   - **Lockless HPET Main Counter**: Read-only EL1 `c0` seqlock fast path driven by `cntvct_el0` with monotonic floor enforcement, invalidated automatically on HPET writes/resets in [`hvdbt.c`](Source/vendor/qemu/target/i386/tcg/system/hvdbt.c).
   - **Deferred Edge-Triggered LAPIC EOI**: EL1 fast path updating `apic_isr` and `hybrid_eoi_pending` when `TMR=0` and `apic_irr_prio <= new_ppr`, synchronized prior to QEMU APIC reads (`hvdbt_sync_eoi_to_apic`).
   - **Platform & RAPL MSRs + CR3 Pinning**: Direct EL1 handling in [`monitor.c`](Source/HybridHypervisor/Monitor/monitor.c) for `MSR_IA32_TSC`, `MSR_APIC_BASE`, `MSR_CORE_THREAD_COUNT`, and 44 RAPL/thermal/performance MSRs (`CPL != 0 -> #GP(0)`), lockless pinned kernel-view switching (`me->pin[2]`), 64-byte-aligned TTAS `mon_lock`, and WoW64/64-bit `do_mov_seg` / `do_iret` segment validation fixes.

4. **248 MiB Split Code Cache & 128 MiB PcMap Layout**
   - Expanded per-partition code cache from `118 MiB` to `248 MiB` (`OFF_CC` shared + `OFF_CC2` private), `PCMAP_BYTES` from `44 MiB` to `128 MiB`, and `BLOCKS_MAX` to `1M` shared / `512K` private, with dual stub copies (`OFF_STUBS` / `OFF_STUBS_HI` and `OFF_STUBS2` / `OFF_STUBS2_HI`) so `+/-128 MiB` ARM64 `B`/`BL` branches stay in range across the entire code cache ([`engine_abi.h`](Source/HybridHypervisor/Sources/CEngine/include/engine_abi.h), [`Engine.swift`](Source/HybridHypervisor/Sources/Engine/Engine.swift)).

5. **45/45 Zero-Bit-Diff AVX2/YMM Differential Test Suite**
   - Bare-metal x86-64 Multiboot1 differential harness ([`test-ext-diff.py`](Source/HybridHypervisor/Tests/test-ext-diff.py)) verifying QEMU TCG (`LC_HVDBT=0`) vs HVDBT (`LC_HVDBT=1`) across 45 instruction suites (GPRs, RFLAGS, XMM0-15, YMM0-15) with **0 bit differences**, plus cross-page VEX decode coverage ([`page-edge-decode.c`](Source/HybridHypervisor/Tests/page-edge-decode.c)).

---

## Quick Start: Build, Test & Run Windows

### Prerequisites

- Apple Silicon Mac (macOS with `Hypervisor.framework` entitlement support and Xcode / Swift toolchain)
- Build & test tools (e.g., via Homebrew): `meson`, `ninja`, `glib`, `pixman`, `libslirp`, `nasm`, `lld` (`ld.lld`)

### 1. Build Hybrid Hypervisor (`libEngine.a`)

```zsh
./Source/HybridHypervisor/build.sh release --library-only
```

Sync sources and build `Source/Build/HybridEngine/Package` for QEMU linking:

```zsh
./Source/scripts/build-hybrid-engine Source/Build/HybridEngine --library-only
```

### 2. Build & Sign QEMU with HVDBT (`qemu-system-x86_64`)

```zsh
meson setup Source/Build/qemu-win-hvdbt Source/vendor/qemu \
  -Dhvf=disabled -Dcocoa=enabled -Dpixman=enabled -Dslirp=enabled \
  -Dpvg=disabled -Dguest_agent=disabled -Dwerror=false -Db_pie=false -Ddocs=disabled \
  -Dhvdbt_engine="$PWD/Source/Build/HybridEngine/Package"

ninja -C Source/Build/qemu-win-hvdbt -j"$(sysctl -n hw.ncpu)" qemu-system-x86_64

xattr -cr Source/Build/qemu-win-hvdbt/qemu-system-x86_64
codesign --force --sign - \
  --entitlements Source/vendor/qemu/accel/hvf/entitlements.plist \
  Source/Build/qemu-win-hvdbt/qemu-system-x86_64
```

### 3. Run the 45-Case AVX2/YMM Differential Test Suite

```zsh
python3 Source/HybridHypervisor/Tests/test-ext-diff.py --ymm
```

Expected result: `ALL 45 DIFFERENTIAL TESTS PASSED (0 bit differences)!`

### 4. Launch an x86-64 Windows Guest

```zsh
LC_HVDBT=1 LC_HVDBT_FAST_FLAGS=1 \
./Source/Build/qemu-win-hvdbt/qemu-system-x86_64 \
  -name Windows-x64-HVDBT \
  -M q35,accel=tcg,hpet=on \
  -cpu Westmere,level=0xd,+sse4.2,+popcnt,+aes,+pclmulqdq,+vaes,+vpclmulqdq,+movbe,+xsave,+avx,+avx2,+f16c,+fma,+bmi1,+bmi2,+abm,+adx,+sha-ni,+x2apic,+invtsc,model=151,stepping=2,model-id="12th Gen Intel(R) Core(TM) i9-12900K" \
  -smp 16,sockets=1,cores=16,threads=1 -m 16G \
  -drive if=pflash,format=raw,readonly=on,file=/path/to/OVMF_CODE.fd \
  -drive if=virtio,format=qcow2,file=/path/to/windows.qcow2 \
  -device VGA,vgamem_mb=64,edid=on,xres=1920,yres=1080 \
  -display cocoa,show-cursor=on \
  -device qemu-xhci -device usb-kbd -device usb-tablet \
  -netdev user,id=net0 -device virtio-net-pci,netdev=net0
```

> **Tip**: For rapid `CEngine` iteration on a running VM, [`Source/scripts/hot-reload-jit.sh`](Source/scripts/hot-reload-jit.sh) compiles the C translation emitters into a signed dylib for live reload. Keep hardware TSO enabled on Windows SMP guests (do not set `LC_HVDBT_NO_TSO=1`).

---

## Upstream Attribution & License

This project is a fork of [**Virtual Intel Mac on Apple silicon**](https://github.com/nfzerox/VirtualIntelMacOnAppleSilicon) (`MIT License`, Copyright (c) 2026 [nfzerox](https://github.com/nfzerox)). See [LICENSE](LICENSE) for full license terms.

Key upstream and third-party open-source projects powering this stack:

- **[VirtualIntelMacOnAppleSilicon](https://github.com/nfzerox/VirtualIntelMacOnAppleSilicon)** by `nfzerox` — original Virtual Intel Mac application, macOS guest graphics stack, and [Hybrid Hypervisor](Source/HybridHypervisor) (invented by Sam Henri Gold).
- **[QEMU](https://www.qemu.org)** — x86-64 platform/device emulation and [HVDBT handoff layer](Source/vendor/qemu/target/i386/tcg/system/hvdbt.c).
- **[Zydis](https://github.com/zyantific/zydis)** (`v4.1.0`, MIT License) — vendored x86-64/VEX/EVEX instruction decoder in [`third_party/zydis`](Source/HybridHypervisor/Sources/CEngine/third_party/zydis).
- **[Box64](https://github.com/ptitSeb/box64)** (MIT License) — ARM64/NEON emitter macros adapted in [`third_party/box64`](Source/HybridHypervisor/Sources/CEngine/third_party/box64).
