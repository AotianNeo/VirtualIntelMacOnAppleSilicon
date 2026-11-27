# VirtualIntelMacOnAppleSilicon — 实时任务清单 (TODO)

> 状态标识：`[x]` 已完成 | `[-]` 进行中 | `[ ]` 待执行

## 总览任务列表

- [x] **T1. 开源组件第一阶段迁移**（Zydis v4.1.1 + Box64 `arm64_emitter.h` + `Tests/test-ext-diff.py` 34/34 通过）
- [x] **T2. Alder Lake i9-12900K CPUID 拓扑 + 0 回退 TCG**（6P+10E QoS + 页末 14B VEX 修复）
- [x] **T3. Geekbench 6.7.1 阶段优化**（`557/3012` → `1515/8140`）
- [x] **T4. 采集当前版本 AIDA64 Business 8.40 全子项基线**（Step 1）
- [x] **T5. 实现 `dlopen` 可插拔 JIT 后端 + 引入 FEX-Emu `CodeEmitter`**（Step 2）
  - [x] 迁入 FEX `CodeEmitter` 与编写 `fex_shim.h`
  - [x] `translate.c` 状态收归与 `TrBackend` 函数指针表 + `tr_reload_backend`
  - [x] `Engine.swift` 接入 `/tmp/hvdbt-reload.path` 与 `CodeCacheGate` 热重载屏障
  - [x] 编写并验证 `Source/scripts/hot-reload-jit.sh`（0.36s 构建 dylib）
- [x] **T6. 直接复用 FEX-Emu 成熟组件优化 `emit_ext.c` / `fex_emit_ext.cpp`**（Step 3）
  - [x] `VLDP128_I7` 256 位对偶加载与直接偏移寻址
  - [x] `v16..v23` 常驻寄存器直通与自异或零开销清零
  - [x] 全系列 FMA3 原地 `VFMLA`/`VFMLS`
  - [x] 移植 FEX `AVX128_VPERMQ`、`VectorBlendImpl`/`VBLENDOpImpl`、`Single128Bit4ByteVectorShuffle`、`VPERMDIndices`、`VAESEnc/Dec` 及 `BSL` 移位
  - [x] 扩充 `Tests/test-ext-diff.py` 并 100% 0 比特差异通过（37/37 PASS）
- [x] **T7. 配置 QEMU & Win2022 Retina HiDPI (`2560x1600` @ 200%) + `+vaes,+vpclmulqdq`**（Step 4）
- [x] **T8. 在线热插拔实测验证 & AIDA64 全 14 子项前后对比**（Step 5：`libhvdbt_jit_1795487584.dylib` 零重启热替换成功，`2560x1600 @ 192 DPI` 生效）
- [-] **T9. 第二轮零重启热迭代（基于 `aida_opt1.tally` 与 Ray-Trace 热点）**
  - [x] 核对 `aida_opt1.tally`：`0x1408d0500` 的 SSE2 结论是页内偏移误匹配。FP32 内核 `TP256_AVX2_FMA3`（`0x1409505a0`）确认在跑；计时热点是 `cpuid; rdtsc; cpuid`（`0x1404414a5`）
  - [x] `VSQRTPS`/`VDIVPS` 已是单条 `FSQRT`/`FDIV`，不再改近似
  - [x] `VBLENDVPS`/`VBLENDVPD`/`VPBLENDVB` 移植 FEX `BSL`/`BIT`/`BIF`（`emit_ext.c`，别名安全，已热加载 `libhvdbt_jit_1795491281`，差分 38/38）
  - [x] 查清 6 倍缩放根因并彻底移除大小核区分（Hybrid `6C+10c` 在 Windows Server 2022 + AIDA64 异构掩码下触发单核/6 核限制；移除 Hybrid 位、Leaf 0x1A 与 E 核 `QOS_CLASS_UTILITY` 限制后，同进程 `F3RT,F6RT` 暴涨至 **7631.22 / 4064.01 KRay/s**）
  - [x] `CPUID` leaf 0 快路径（`EngineCtx` per-vCPU 缓存 + EL0 内联 8 条指令，零 VMEXIT）
  - [x] `ext_load256_pair`/`ext_store256_pair` 全面启用 `VLDP128_I7`/`VSTP128_I7` 256-bit 访存配对，以及 `VMASKMOV*` 全 0/全 1 向量快路径（差分 39/39 PASS，`F3RT,F6RT` 再涨至 **12315.58 / 4077.12 KRay/s**）
  - [ ] ACPI PMTimer 端口 `0x608` 内联快路径
- [-] **T10. 架构级大改：完整移植 FEX-Emu `FEXCore` 作为 HVDBT Ring-3 编译管线 + Geekbench 7.1.0 Pro 全子项优化**
  - [x] **Phase 0**: 拉取官方 FEX-Emu（`fex-emu-upstream`），补齐 `VPCMPISTRI`/`VMPSADBW`/`VMASKMOVDQU` 消除 88.7 万次 TCG 回退，采集 Geekbench 7.1.0 Pro 全子项基线（**SC=419 / MC=1897**，0 新增 `unknown opcode`；已定位 `202 Text Processing` validation 失败、`701 Video Encoder=45`、`603 Game Physics=197`、`501 Ray Tracer MC=871 < SC=1017` 等核心瓶颈）
  - [x] **Phase 1**: FEXCore 依赖审计与 macOS arm64 独立编译原型（50 个核心 `.cpp` 已编出 [libfexcore_jit.a](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/libfexcore_jit.a) 4.1 MB，[fex_standalone_test.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/fex_standalone_test.cpp) 端到端将 `vmovdqu+vfmadd231ps+vmovdqu+add+ret` 编译为 176B 原生 ARM64 机器码，退出码 0）
  - [x] **Phase 2**: HVDBT `EngineCtx` / `SVC` Target Adapter 桥接与全量差分测试（`sizeof(CpuStateFrame)=3648` 字节放入 `EngineCtx` 尾部；[hvdbt_fex_adapter.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/hvdbt_fex_adapter.cpp) Apple M5 Pro `MAP_JIT` 实机执行 0 bit diff；[libhvdbt_fex_bridge.a](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/libhvdbt_fex_bridge.a) 已静态链接入 [translate.c](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Sources/CEngine/translate.c) 与 `qemu-system-x86_64`，开启 `LC_HVDBT_FEX=2` 后 [test-ext-diff.py](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Tests/test-ext-diff.py) **40/40 全部 0 比特差异通过**）
  - [x] **Phase 3**: Ring-3 16 核 SMP 接入与稳定性修复（`x18` 保留、`SupportsAtomics`/`SupportsPMULL_128Bit`、`GuestSignal_SynchronousFault` 守卫、`fex_ctx_inited_count`、`reset_my_return_stacks` FEX 保护；`LC_HVDBT_FEX=2` 保持 40/40 PASS）
  - [x] **Phase 4**: Win2022 (`LC_HVDBT_FEX=1`) Geekbench 7.1.0 Pro 全量跑分验证（**SC: 419 → 487 (1.16x), MC: 1897 → 2434 (1.28x)**，`202 Text Processing` 修复为 `SC=304 / MC=2394`，实测 `[hvdbt-fex] compiled_blocks=61281742`）
  - [ ] **Phase 4b**: 排查 `LC_HVDBT_FEX=1` 下 Win2022 GUI 黑屏（`dwm.exe` / `LogonUI.exe` 存活但 `screendump` 全黑，对比 `LC_HVDBT_FEX=0`）
    - 已排除：QEMU `ar_index` 消隐（实测 `0x20`，图形模式，VBE 2560x1600x32 LFB）；FEX `stnp` 对主机别名不可见（桥接已把 `stnp` 改成 `stp`，`stnp_rewrites=2604`，`vram_ptr` 64MB 仍全 0）。见 PLAN Phase 4b。下一步查用户态合成为何是黑的，或 present 为何没写入 LFB。
  - [ ] **Phase 5**: 排查并消除 `compiled_blocks=61,281,742` 重编风暴与全局 `std::mutex` 编译锁竞争
  - [ ] **Phase 6**: 实现 Stay-in-FEX 架构（双入口 `entry_from_hvdbt`/`entry_fex_body` + FEX-to-FEX 块间直连 + 原生 `L1Pointer`/`CallRetStack` + `CONFIG_MULTIBLOCK=1`）
    - [x] Sprint 1 共享出入口、每线程编译器及 Stay-in-FEX 验证（219 秒进 2560×1600 桌面）。
    - [x] Sprint 2 多块翻译和 FEX→FEX 直接链接已接入。2026-11-25 续测重建成功，`LC_HVDBT_FEX=2` 差分 40/40 通过（0 比特差异）。
    - [x] 暂停 FEX 调用返回预测，调用与返回使用 view 块查找。多块开启和关闭各通过 41/41 项差分测试，包含 24 层直接和间接嵌套调用。
    - [ ] 对比多块开关下的 Win2022 冷启动、QGA、缓存重置及累计翻译时间，定位黑屏所在配置。
    - [ ] 验证 30 秒进桌面和 Geekbench 7.1.0 Pro 分数；原生 `L1Pointer` 的失效一致性仍需验证。

---

## Geekbench 7.1.0 Pro 全子项跑分记录（对称 16-Core 标尺）

> 基线产物：[gb7_base.json](file:///Users/admin/.gemini/antigravity/scratch/win2022-hvdbt/gb7_base.json)（热加载 `libhvdbt_jit_1795502624.dylib` 后跑测，0 新增 `unknown opcode` TCG 回退）
> FEXCore 产物：`C:\GB7\gb7_fex1.json`（`LC_HVDBT_FEX=1`，`compiled_blocks=61281742`）

| 模式 | ID | 子项 | 基线分数 | 优化后 (FEXCore) | 变化幅度 |
|---|---:|---|---:|---:|---|
| **总分** | — | **Single-Core Score** | **419** | **487** | **1.16x** |
| **总分** | — | **Multi-Core Score** | **1897** | **2434** | **1.28x** |
| SC | 202 | Text Processing | **0** *(validation failed)* | **304** | **PASS** |
| SC | 101 | File Compression | 257 | 319 | 1.24x |
| SC | 102 | Navigation | 453 | 627 | 1.38x |
| SC | 103 | HTML5 Browser | 320 | 484 | 1.51x |
| SC | 104 | PDF Renderer | 586 | 575 | 0.98x |
| SC | 105 | Photo Library | 471 | 600 | 1.27x |
| SC | 106 | Clang | 459 | 569 | 1.24x |
| SC | 107 | Asset Compression | 540 | 661 | 1.22x |
| SC | 301 | Object Detection | 279 | 281 | 1.01x |
| SC | 302 | Background Blur | 451 | 473 | 1.05x |
| SC | 401 | Horizon Detection | 725 | 707 | 0.98x |
| SC | 402 | Object Remover | 669 | 538 | 0.80x |
| SC | 403 | HDR | 412 | 548 | 1.33x |
| SC | 404 | Photo Filter | 516 | 575 | 1.11x |
| SC | 501 | Ray Tracer | 1017 | 923 | 0.91x |
| SC | 601 | Structure from Motion | 414 | 432 | 1.04x |
| MC | 202 | Text Processing | **0** *(validation failed)* | **2394** | **PASS** |
| MC | 101 | File Compression | 1152 | 1419 | 1.23x |
| MC | 102 | Navigation | 2124 | 3212 | 1.51x |
| MC | 103 | HTML5 Browser | 1280 | 2013 | 1.57x |
| MC | 104 | PDF Renderer | 2121 | 2343 | 1.10x |
| MC | 105 | Photo Library | 2678 | 3580 | 1.34x |
| MC | 106 | Clang | 1861 | 2331 | 1.25x |
| MC | 107 | Asset Compression | 2418 | 3013 | 1.25x |
| MC | 301 | Object Detection | 1219 | 1244 | 1.02x |
| MC | 302 | Background Blur | 2198 | 2483 | 1.13x |
| MC | 401 | Horizon Detection | 4787 | 5185 | 1.08x |
| MC | 402 | Object Remover | 2925 | 2481 | 0.85x |
| MC | 403 | HDR | 6869 | 7708 | 1.12x |
| MC | 404 | Photo Filter | 1882 | 2221 | 1.18x |
| MC | 501 | Ray Tracer | 871 | 1170 | 1.34x |
| MC | 601 | Structure from Motion | 1415 | 1663 | 1.18x |
---

## AIDA64 Business 8.40 全子项跑分记录

| 子项 | 基线（当前） | 优化后（FEX + 热插拔 + VAES） | 变化幅度 |
|---|---:|---:|---|
| Memory Read | 86808 MB/s | 83938 MB/s | -3.3% |
| Memory Write | 67393 MB/s | 65703 MB/s | -2.5% |
| Memory Copy | 170487 MB/s | 175739 MB/s | +3.1% |
| Memory Latency | 84.8 ns | 33.2 ns | -60.8% (**2.55x**) |
| CPU CheckMate (Queen) | 150 MP/s | 865 MP/s | +476.7% (**5.77x**) |
| CPU PhotoWorxx | 15164 MPixel/s | 55381 MPixel/s | +265.2% (**3.65x**) |
| CPU ZLib | 149.5 MB/s | 438.5 MB/s | +193.3% (**2.93x**) |
| CPU AES | 432 MB/s | 15968 MB/s | +3596.3% (**36.96x**) |
| CPU SHA3 | 567 MB/s | 2212 MB/s | +290.1% (**3.90x**) |
| FPU Julia | 5985 | 26152 | +337.0% (**4.37x**) |
| FPU Mandel | 2508 | 13012 | +418.8% (**5.19x**) |
| FPU SinJulia | 658 | 3150 | +378.7% (**4.79x**) |
| FP32 Ray-Trace | 991 KRay/s | 1326 KRay/s | +33.8% (**1.34x**) |
| FP64 Ray-Trace | 515 KRay/s | 576 KRay/s | +11.8% (**1.12x**) |

上表「优化后」仍是全套 `/BENCH`、blend 热插拔之前的数，不改。

blend 热插拔之后、不挂起、不读工作进程内存的射线单项（KRay/s）：

| 跑法 | FP32 | FP64 |
|---|---:|---:|
| Hybrid `6C+10c`：只跑 F3RT | 5740.01 | |
| Hybrid `6C+10c`：只跑 F6RT | | 2636.42 |
| Hybrid `6C+10c`：同进程 F3RT+F6RT | 956.96 | 439.28 |
| **对称 16-Core（已设为默认）：同进程 F3RT+F6RT** | **7631.22** | **4064.01** |
| **对称 16-Core + CPUID0/LDP-STP/VMASKMOV 优化** | **12315.58** | **4077.12** |

已验证：**取消大小核区分（移除 `Leaf 7 EDX[15]` Hybrid、`Leaf 0x1A` 与 `QOS_CLASS_UTILITY`）并全核 `QOS_CLASS_USER_INTERACTIVE` 后，同进程 `F3RT+F6RT` 直接吃满全部 16 核，FP32 暴涨 7.97x、FP64 暴涨 9.25x。**

`aida_bench64.dll` 的 CPU 类表在 `0x140c39890`。签名 `0x20090672`（前缀 `0x20`）第一次命中 class 356：

- FP32 下标 58：`Base256_AVX2_FMA3`，`0x14094f660`（678 条指令，含 12 条 `vblendvps`、38 条 `vfmadd231ps`、9 条 `vmaskmovps`）
- FP64 下标 55：`TP256_AVX_FMA3`，`0x1409084e0`

前缀 `0xB0` 会先命中 class 357 `Base512_AVX512BWDQ`。`0x14043a66c` 在型号掩码 `0x90670`、特性位 `0x69` 为空、且 `[cpu+0x26f0] >= 0x1a` 时把前缀改成 `0xB0`。不实现 AVX-512。6 倍缩放的除数还没在计分代码里定位。

### 2026-11-25 启动与跑分补记

- 隔离单块对照在 151 秒采样点确认进入桌面，91 秒采样点 QGA 已回应。隔离多块测试到 150 秒仍无 QGA 回应。各配置均重建自同一补丁和同一临时磁盘基底；并行负载下的单次结果用于排查，需串行复测评估耗时。
- Sprint 3 原实例的 Geekbench 主程序和 AVX2 工作进程启动时报 `0xC0000005`，没有取得可用于比较的新分数。
- [ ] 定位 FEX 路径上的 Geekbench `0xC0000005`。关闭 FEX 对照中 `--help` 正常退出，FEX 单块对照的完整跑分仍失败。

- [x] 修复 FEX 链接循环中的暂停饥饿，并新增 `Tests/test-fex-pause.py` 回归用例。
- [ ] 复测整机 FEX 时的 Geekbench 初始化。仅把主程序分流给原生 HVDBT 后，访问异常退出仍能复现。
