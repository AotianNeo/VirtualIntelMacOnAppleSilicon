# Windows 11 x64 HVDBT 适配与性能优化补丁集

- **基线版本**：`nfzerox/VirtualIntelMacOnAppleSilicon` commit `a361cc76`（`tag: 2.2.1`）
- **改动规模**：14 个源码文件，`+2,238 / -182` 行
- **测试平台**：Apple M5 Pro（18 核：6 超级大核 + 12 次级核，支持最高 16 vCPU），Windows 11 PE x64（Hikari PE V8.60）

---

## 补丁文件清单

| 补丁文件 | SHA-256 | 包含改动 |
|---|---|---|
| `win11-hvdbt-all-in-one.patch` | `e1204aa275f32b2e3069c2aca2722a43dfcca6f34ad6a29029dcbc03409773e4` | **完整单文件合集**（涵盖下述 0001–0004 全部 14 个文件） |
| `0001-qemu-win11-hvdbt-apic-tpr-memory-listener.patch` | `9b0e0d8ef084957d534be415dabbef7e084590fe912f7614219152340cb9f5cf` | QEMU Win11 适配、16 vCPU 扩展（`HVDBT_MAX_CPUS=16`）、零退出 CR8/APIC TPR 共享同步（Dekker 屏障）、延迟 `MemoryListener` 事务去重 |
| `0002-engine-monitor-lsl-verr-verw-tpr-x87-sync.patch` | `178436299d85af2aeeda8c888a5717036f9f267e41909588332b40feb3e0e840` | 16 vCPU 布局扩展（`MAX_VCPUS=16`、`OFF_MON_CPU`/`desc_hole` 重排）、Monitor 原生 `VERR`/`VERW`、`LSL (0F 03)`、CR8/TPR 快路径、精确 `FSW.ES` 检查、极速 x87 惰性刷回与 `fptan` 慢路径 |
| `0003-engine-inline-popcnt-narrow-shifts-rcl-rcr.patch` | `a40835c42f0da91f1b3cba2a795a4f4812aa781132bf75e0c2d4f9f4b92a8a6b` | SWAR `POPCNT`、8/16 位移位与 `ROL`/`ROR`、`RCL`/`RCR 1`、直写 `emit_lea`、`emit_alu` 负立即数/4096 倍数/内存立即数/窄位宽 `TEST`/`CMP` 快路径、内存 `INC`/`DEC`、直写 `CRC32`、单指令 `BSR` `EOR #63` |
| `0004-engine-extreme-fast-x87-and-fptan.patch` | `96f2675dd20f68bdd09bf2a283a6f81af52af3471fff34adfa581eaddd7c91bb` | 块内前向分支回填 + 精确 `live_before`、SSE 专项优化（消除 `align16()` 5 倍膨胀、`DUP`/`EXT`/`REV64`/`ZIP`/`UZP` 单指令洗牌、`FRSQRTE`/`FRECPE` 零流水线停顿、直写 `CVT*2SI`/`CMPPS`）、极速 x87 与 EL0 原生 `FPTAN` |

---

## 应用与编译方法

### 方式 A：一键应用整包补丁（推荐）

```bash
git apply --check patches/win11-hvdbt-all-in-one.patch
git apply patches/win11-hvdbt-all-in-one.patch
```

### 方式 B：按模块顺序应用拆分补丁

```bash
git apply patches/0001-qemu-win11-hvdbt-apic-tpr-memory-listener.patch
git apply patches/0002-engine-monitor-lsl-verr-verw-tpr-x87-sync.patch
git apply patches/0003-engine-inline-popcnt-narrow-shifts-rcl-rcr.patch
git apply patches/0004-engine-extreme-fast-x87-and-fptan.patch
```

### 重新编译与签名

```bash
ROOT="$(pwd)"
QEMU="$ROOT/Source/Build/qemu-win-hvdbt/qemu-system-x86_64"
zsh "$ROOT/Source/scripts/build-hybrid-engine" "$ROOT/Source/Build/HybridEngine" --library-only
rm -f "$QEMU" "$QEMU-unsigned"
PATH="/opt/homebrew/bin:$PATH" ninja -C "$ROOT/Source/Build/qemu-win-hvdbt" -j "$(sysctl -n hw.ncpu)" qemu-system-x86_64
xattr -cr "$QEMU"
codesign --force --sign - --entitlements "$ROOT/Source/vendor/qemu/accel/hvf/entitlements.plist" "$QEMU"
```

> **注意**：Windows 11 SMP 必须保持硬件 TSO 开启，切勿设置 `LC_HVDBT_NO_TSO=1`。

---

## 实测性能提升汇总（8 / 16 vCPU，带实时 GUI）

| 测试项 | 优化前 | 优化后（`prof18`） | 变化 |
|---|---|---|---|
| **Win11 PE 冷启动至桌面** | `54.40 秒` | `30.63–32.23 秒` | **约快 41%** |
| **7-Zip 25.01 x64（单核 `-mmt1 -md22`）** | `6,639 MIPS` | **`8,896 MIPS`**（解压 `9,847 MIPS`） | **+34.0%** |
| **7-Zip 25.01 x64（8 核 `-mmt8 -md22`）** | `53,868 MIPS` | **`70,414 MIPS`**（压缩 `70,934 MIPS`） | **+30.7%** |
| **7-Zip 25.01 x64（16 核 `-mmt16 -md22`）** | 原版上限 8 核 | **`114,320 MIPS`**（3 轮均值，峰值 **`118,363 MIPS`**，解压 **`130,193 MIPS / 1.49 GB/s`**） | **相比初版 8 核 +112%** |
| **SuperPI 16K（1.6 万位）** | `4.143 秒`（原版 >20 秒） | `0.750 秒` | **5.52× 提速** |
| **SuperPI 1M（100 万位）** | 推算 `~375 秒`（`prof13`: `224.078 秒`） | `35.029 秒`（校验通过） | **~10.7× 提速** |
| **Fritz Chess（8 核 WoW64）** | — | `17.44x / 8,372 kN/s` | 稳定通过 |
| **CPU-Z 17.01.64（单核 / 16 核）** | 单核 `277.2` / 8 核 `1,922.4` | **单核 `300.6` / 16 核 `4,689.8`**（`15.60×` 扩展） | **单核 +8.4% / 多核 +144%** |
| **`HC_APIC_TPR` 宿主机调用** | `2,280,112 次`（`5.41 秒`） | `~2,700 次`（`0.01 秒`） | **-99.9%** |
| **`fptan` / `SVC_X87` / `LSL` / `POPCNT` 回退** | 累计 `>1.5 亿次` | **`0 次`** | **100% 清零** |
| **`SVC_H_SHIFT` 移位陷入** | `116,731,356 次` | `1,742,029 次` | **-98.5%** |
