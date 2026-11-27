# VirtualIntelMacOnAppleSilicon — 架构升级与性能优化计划 (PLAN)

> 最后更新：2026-11-24 13:05 CST
> 目标：在 Apple M5 Pro（6 Super + 12 Efficiency Cores）上优化 Windows Server 2022 x64（模拟 Alder Lake i9-12900K）的 **AIDA64 Business 8.40** 全子项及 **Geekbench 6.7.1** 性能，同时实现 **JIT 后端零重启热插拔** 与 **macOS Retina HiDPI 显示**。

---

## 一、已完成基线

1. **开源组件第一阶段迁移（Zydis v4.1.1 + Box64 `arm64_emitter.h` + QEMU TCG 差分对拍）**
   - 接入 Zydis v4.1.1 解码器与 Box64 MIT `arm64_emitter.h`，建立 `Tests/test-ext-diff.py` 差分对拍平台（**34/34 项 0 比特差异通过**）。
2. **Alder Lake i9-12900K 拓扑与 0 回退 TCG**
   - `CPUID` 暴露 `Family 6 Model 151 Stepping 2`、混合架构 `Leaf 0x1A`（6P + 10E 绑定宿主 QoS）、1 GHz 恒定 TSC、`+abm`/`+f16c`/`+avx2`/`+fma`/`+bmi1`/`+bmi2`/`+adx`/`+sha-ni`。
   - 修复跨页取指与页末 14 字节 VEX `#UD` 整页回退 TCG 悬崖。
3. **Geekbench 6.7.1 阶段成果**：`557 / 3012` → **`1515 / 8140`**（`YMM0..7` 高半驻留 `v16..v23`、8/16 位 `DIV` 内联、`LDMXCSR`/`STMXCSR` EL0 内联、`VFMADD231` 原地累加）。

---

## 二、当前执行阶段

### Step 1：采集当前版本 AIDA64 Business 8.40 全子项基线
- 通过 QGA 运行 `C:\AIDA64\aida64.exe /R C:\aida_bench_base.txt /TEXT /BENCH /SILENT`，记录全部 14 项子项基准分数与 `tally` 状态。

### Step 2：实现 `dlopen` 可插拔 JIT 后端 + 接入 FEX-Emu `CodeEmitter`
1. **引入 FEX-Emu `CodeEmitter`**：将 `/tmp/fex/CodeEmitter/CodeEmitter/*.{h,inl}` 迁入 `CEngine/third_party/fex/CodeEmitter/`，并提供轻量 `fex_shim.h` 零依赖适配层。
2. **状态剥离与函数指针表**：将 `tr_ad_page`、`tr_aes_mask`、`tr_aes_key` 收归 `translate.c`，在 `translate.c` 中定义 9 入口函数指针表 `TrBackend tr_be`，新增 `tr_reload_backend(const char *dylib_path)`。
3. **`CodeCacheGate` 安全热插拔**：在 `Engine.swift` 监控 `/tmp/hvdbt-reload.path`，利用 `CodeCacheGate` 屏障将 16 个 vCPU 停在 x86 边界 → `dlopen` 新 `.dylib` → `tr_flush_parked(true)` 全量清空旧代码缓存 → 恢复运行。
4. **一键热替换脚本**：`Source/scripts/hot-reload-jit.sh` 在 0.4 秒内混合编译 C 与 C++20 后端为 `/tmp/libhvdbt_jit_<epoch>.dylib` 并触发在线重载。

### Step 3：直接复用 FEX-Emu 成熟组件优化 `emit_ext.c` / `fex_emit_ext.cpp`
- **256 位 `[base+disp]` 对偶加载/存储合并**：`VLDP128_I7` 单指令加载 32 字节 + 直接偏移 `VLDR128_U12`/`VSTR128_U12`。
- **常驻寄存器 `v16..v23` 直通**：消除 `is_lanewise_vex` 与 `VMOV*` 的多余 `VMOVQ` 拷贝，自异或清零直达 `VEORQ + ymm_zero_hi`。
- **全系列 FMA3 原地化**：覆盖 `132`/`213`/`231` 与 `FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`。
- **移植 FEX `AVX_128.cpp` / `Vector.cpp` / `EncryptionOps.cpp` 成熟降级**：
  - `AVX128_VPERMQ`（16 路单指令映射，针对 `CPU SHA3`）
  - `VectorBlendImpl` / `VBLENDOpImpl`（针对 `VPBLENDD`/`VBLENDPS`/`VPBLENDW`）
  - `Single128Bit4ByteVectorShuffle`（针对 `VPSHUFD`/`VPERMILPS`）
  - `VPERMDIndices`（针对 `CPU PhotoWorxx`）
  - `DEF_OP(VAESEnc/VAESEncLast/VAESDec/VAESDecLast)`（原地 AES，针对 `CPU AES`）
  - `ext_shift_var128` 使用 `BSL` 简化 64 位可变移位（针对 `CPU SHA3`）
- 更新 `Tests/test-ext-diff.py` 并全量验证 0 比特差异通过。

### Step 4：配置 QEMU & Windows Server 2022 Retina HiDPI（`2560x1600` @ 200% DPI）+ `+vaes,+vpclmulqdq`
- 更新 `start-win2022.sh`：`-device VGA,vgamem_mb=64,edid=on,xres=2560,yres=1600` 与 `+vaes,+vpclmulqdq`。
- 客户机设置 200% HiDPI（`LogPixels = 192`, `Win8DpiScaling = 1`）并安装驱动包。
- 重启切换到可插拔版本 QEMU（之后不再重启 VM）。

### Step 5：验证在线热插拔并实测 AIDA64 全子项对比（已完成）
- **在线热插拔实测**：在 16 vCPU 运行中的 Windows Server 2022 上运行 [hot-reload-jit.sh](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/scripts/hot-reload-jit.sh)，0.36s 完成构建并在零重启下热加载 `/tmp/hvdbt-jit-plugins/libhvdbt_jit_1795487584.dylib`。
- **Retina HiDPI 实测**：客户机分辨率 `2560x1600 @ 32bpp`，`LogPixels = 192`（200% DPI 缩放）已生效。
- **差分测试**：`Tests/test-ext-diff.py` 全部 **37/37 项 0 比特差异通过**。
- **AIDA64 Business 8.40 实测核心提升**：
  - `CPU AES`: `432` → **`15968 MB/s`**（**36.96x**）
  - `CPU CheckMate`: `150` → **`865 MP/s`**（**5.77x**）
  - `FPU Mandel`: `2508` → **`13012`**（**5.19x**）
  - `FPU SinJulia`: `658` → **`3150`**（**4.79x**）
  - `FPU Julia`: `5985` → **`26152`**（**4.37x**）
  - `CPU SHA3`: `567` → **`2212 MB/s`**（**3.90x**）
  - `CPU PhotoWorxx`: `15164` → **`55381 MPixel/s`**（**3.65x**）
  - `CPU ZLib`: `149.5` → **`438.5 MB/s`**（**2.93x**）
  - `Memory Latency`: `84.8` → **`33.2 ns`**（**2.55x**）

---

## 三、下一阶段路线（已移除大小核区分，全面转向对称 16 核优化）

`VSQRTPS`/`VDIVPS` 已经是一条 `FSQRT`/`FDIV`。`VBLENDVPS` 的 FEX `BSL`/`BIT`/`BIF` 已热加载，差分 38/38。

**6 倍掉分根因已彻底查清并解决**：
- 此前在 `hvdbt.c` 中暴露 `Hybrid (6C+10c)` + `Leaf 0x1A` 且将 E 核 QoS 限制为 `QOS_CLASS_UTILITY` 时，Windows Server 2022（无 Win11 Thread Director）与 AIDA64 异构线程规划器在同进程多项目下退化到单核（`956.96 / 439.28 KRay/s`），即便单跑也只能调度 6 个 P 核（`5740.01 / 2636.42 KRay/s`）。
- **彻底移除 `hvdbt.c` 全部大小核区分代码**（统一为对称 `16-Core` 与 `QOS_CLASS_USER_INTERACTIVE`）后，同进程 `F3RT,F6RT` 实测直接达到 **`7631.22 / 4064.01 KRay/s`**（暴涨 **7.97x / 9.25x**）。
- **继续叠加真实转译优化**（`EngineCtx` 内联 `CPUID leaf 0` 缓存、`ext_load256_pair`/`ext_store256_pair` `LDP`/`STP` 访存配对、`VMASKMOV*` 全 0/全 1 向量快路径，差分 **39/39 PASS**）后，同进程 `F3RT,F6RT` 进一步飙升至 **`12315.58 / 4077.12 KRay/s`**（FP32 再涨 **+61.4%**，较最初基线 `991` 暴涨 **12.43x**）。

下一项：

1. 在对称 `16-Core` 全优化版本上重跑全套 AIDA64 Business 8.40 `/BENCH`（更新 `TODO.md` 全 14 项）与 Geekbench 6.7.1。
2. 在确认的 FP32/FP64 核上继续收 `vfmadd231ps` 与 `ACPI PMTimer 0x608` 内联快路径。

---

## 四、架构级大改：完整移植 FEX-Emu `FEXCore` 作为 HVDBT Ring-3 编译管线

- **架构分工**：保留 HVDBT EL1 Monitor + 硬件影子页表 + Ring-0 翻译器；将官方 FEX-Emu `FEXCore`（`Frontend → OpcodeDispatcher → IR → 优化 Pass → RegisterAllocationPass → Arm64JITCore`）移植为 macOS arm64 独立编译后端并接入 Ring-3（CPL=3, 64-bit），彻底清零用户态 TCG 回退并获得块级 IR 优化（死标志位消除、CSE、寄存器分配）红利。
- **Phase 0（建立标尺 & 0 回退基线）**：
  - `git clone --depth 1` 官方 FEX-Emu 到 `/Users/admin/.gemini/antigravity/scratch/fex-emu-upstream`；
  - 先热补齐 `emit_ext.c` 剩余 VEX 缺口（`VPCMPISTRI` 88.7 万次 TCG 回退、`VMPSADBW`/`MPSADBW`、`VMASKMOVDQU`/`MASKMOVDQU`），跑通 0-TCG 回退的 **Geekbench 7.1.0 Pro** 全子项基线作为对比标尺。
- **Phase 1（已完成）**：
  - 全量编译官方 `FEXCore` 50 个核心 `.cpp`（`X86Tables + Frontend + OpcodeDispatcher + IR + Passes + RegisterAllocationPass + Arm64JITCore + Core/Context`）为 macOS arm64 静态库 [libfexcore_jit.a](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/libfexcore_jit.a)（4.1 MB）；
  - [fex_standalone_test.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/fex_standalone_test.cpp) 端到端将 `vmovdqu+vfmadd231ps+vmovdqu+add+ret` 编译为 176B 原生 ARM64 机器码（含 `LDP/STP` 256-bit、`FMLA`、内联 PF/AF、`CallRet` 预测栈）。
- **Phase 2（已完成）**：
  - 核对 `sizeof(CpuStateFrame)=3648` 字节、`x28 = STATE`、`x64::SRAFPR = v16..v31`；
  - [hvdbt_fex_adapter.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/hvdbt_fex_adapter.cpp) 在 Apple M5 Pro `MAP_JIT` 硬件上实机跑通完整桥接块（`RAX=1337`, 8 个 AVX2 FMA3 float 结果 0 bit diff, 0 宿主机 `BL`）；
  - [hvdbt_fex_bridge.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/hvdbt_fex_bridge.cpp) 已封装为 [libhvdbt_fex_bridge.a](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/libhvdbt_fex_bridge.a) 并接入 [translate.c](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Sources/CEngine/translate.c)（`LC_HVDBT_FEX=1/2`）及 `qemu-system-x86_64`，实测 `LC_HVDBT_FEX=2` 下 [test-ext-diff.py](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Tests/test-ext-diff.py) **40/40 全部 0 比特差异通过**！
- **Phase 3（已完成）**：
  - 解决 Win2022 16 核 SMP 接入 FEXCore 的 4 个深层根因：
    1. 在 [Arm64Emitter.cpp](file:///Users/admin/.gemini/antigravity/scratch/fex-emu-upstream/FEXCore/Source/Interface/Core/ArchHelpers/Arm64Emitter.cpp) 的 `x64::RA` 与 `PreserveAll_Dynamic` 中保留 `x18`（HVDBT 永久保留 `x18 = EngineCtx` 供 EL1 `vectors.s` `SAVE_ALL` 使用）；
    2. 在 [hvdbt_fex_bridge.cpp](file:///Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/hvdbt_fex_bridge.cpp) 中启用 `SupportsAtomics`（ARMv8.1 LSE，消除 `ldaxr`/`stlxr` 在 `step_on()` 下的活锁）、`SupportsTSOImm9`、`SupportsPMULL_128Bit`（消除 `PCLMULQDQ` 触发 `GuestSignal_SynchronousFault` 死循环）、`SupportsFRINTTS`、`SupportsFCMA`、`SupportsDotProd`，并将 `GuestSignal_SynchronousFault` 加入块拒绝守卫；
    3. 在 [translate.c](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Sources/CEngine/translate.c) 中将 FEX 上下文初始化从单次 `bool` 改为按 `vcpu_count` 增量同步（`fex_ctx_inited_count`）；
    4. 在 [monitor.c](file:///Users/admin/.gemini/antigravity/scratch/VirtualIntelMacOnAppleSilicon/Source/HybridHypervisor/Monitor/monitor.c) 的 `exit_check()` 中保护 FEX 块内的 `x26`（`R_RS` / `REG_PF`）不被 `reset_my_return_stacks()` 覆盖。
- **Phase 4（已完成 Geekbench 7.1.0 Pro 跑分，GUI 黑屏待排查）**：
  - 在 `LC_HVDBT_FEX=1` 下成功启动 16 核 Windows Server 2022（0 `monitor fault`，实测编译 `[hvdbt-fex] compiled_blocks=61281742`）；
  - **Geekbench 7.1.0 Pro 全量 CPU 对比（vs [gb7_base.json](file:///Users/admin/.gemini/antigravity/scratch/win2022-hvdbt/gb7_base.json)）**：Single-Core `419 → 487`（**1.16x**），Multi-Core `1897 → 2434`（**1.28x**），基线归零的 `[202] Text Processing` 恢复通过（`SC=304 / MC=2394`）；
  - **已知待查项**：Win2022 启动后 `Microsoft Basic Display Adapter (2560x1600)`、`dwm.exe` 与 `LogonUI.exe` 均在运行且无新崩溃，但 QMP `screendump` 与 Cocoa 窗口仍呈全黑（需进一步对比 `LC_HVDBT_FEX=0` 排查 Ring-3 DWM/帧缓冲渲染路径）。
- **Phase 4b（进行中：黑屏。STNP 与 QEMU 消隐都已排除）**：
  - `LC_HVDBT_FEX=0` 对照已有：`/tmp/win2022_nofex.ppm` 是 2560x1600 已登录桌面。黑屏只在 `LC_HVDBT_FEX=1`。
  - FEX 把 256-bit `MOVNT` 编成 `stnp`（`0xAC000000`），原生 JIT 是分配缓存的 `stp`（`0xAD000000`，只差 bit 24）。`hvdbt_fex_bridge.cpp` 的 `fex_allocating_pair` 在拷贝块时把 `stnp` w/s/d/x/q 改成对应 `stp` offset。
  - 实机 `LC_HVDBT_FEX=1`（tally `fex_stnp.tally`，截屏 `/tmp/win2022_fex_stnp.ppm`）：约 8 分钟时 `stnp_rewrites=2604`、`compiled_blocks≈2.7e7`。VGA 已是图形模式，不是消隐：`ar_index=0x20`，`graphic_mode=1`，VBE `2560x1600x32`，enable `0x41`（LFB），`vbe_start_addr=0`，BAR0 仍是 `0x80000000–0x83ffffff`。`vram_ptr` 整段 64MB 与该 GPA 全 0，PPM 全黑。
  - 因此像素没进线性帧缓冲。内核 present 若拷贝了非零表面，走的是原生 STP，会写进同一块 `vram_ptr`。下一步：用户态合成结果本身是 0，或 present 没有把表面拷进 LFB。QGA 在这次启动的 13 分钟内没有应答；客户机内存里能搜到 `dwm.exe` / `LogonUI` / `explorer.exe` / `qemu-ga` 字符串，用户态 syscall 与 CR3 切换在进行。
- **Phase 5（消除 `compiled_blocks=61,281,742` 重编风暴）**：
  - 排查代码缓存高频率冲刷/未命中根因及全局编译锁 `std::mutex` 竞争，消除重复编译，释放 16 核线性扩展度。
- **Phase 6（Stay-in-FEX 架构：双入口 + 块间直连 + 原生 `L1Pointer`/`CallRetStack` + `CONFIG_MULTIBLOCK=1`）**：
  - 暴露 `entry_from_hvdbt` 与 `entry_fex_body` 双入口，FEX→FEX 直接跳转原地 patch 为 `b target.entry_fex_body`；
  - 启用每 vCPU 原生 `L1Pointer` 哈希表与 `CallRetStack`，消除块进出 ~90 条寄存器搬运指令；
  - 打开 `CONFIG_MULTIBLOCK = 1`，释放 FEXCore 跨块 IR 优化与寄存器分配红利。

### 2026-11-25 · Sprint 2 续测

- 已实现共享 EL0 入口/出口、每线程 FEX 编译器和 FEX→FEX 双入口直连。Sprint 1 曾在 219 秒进入桌面；Sprint 2 默认启用多块翻译，差分测试通过后仍出现实机黑屏。
- 原 Sprint 2 长跑的 QMP 截图全黑。最终采样计有 33 次私有代码缓存重置、1,063 万次块失效及约 203 秒累计翻译时间。源码确认普通 `SVC_FEX_LINK` 冷未命中已通过 `translate_here` 进入哈希发布路径。
- 本轮把原生标志位活性分析移到 FEX 失败后的路径，避免成功的 FEX 块重复走原生分析。刷新代码缓存时继续发布查找未命中的新目标，旧源地址上的补丁仍跳过。
- 用同一个 Win2022 磁盘的临时写入层做 16 vCPU 启动对照。记录桌面截图、QGA、累计翻译时间及缓存重置。客户机响应后运行 Geekbench 7.1.0 Pro 离线跑分。

- FEX 桥接尚未移植上游带保护页的调用返回预测栈和失效清理。已在桥接 IR 中去掉调用与返回的预测提示，让它们继续通过当前 view 的块查找；保留 x86 调用栈语义、跨块翻译和直接跳转链接。
- 新增 24 层直接与间接嵌套调用用例。`LC_HVDBT_FEX=2` 在多块开启和关闭时分别通过 41/41 项差分测试。
- Antigravity 并行的 Sprint 3 记录显示，启用 SIMD/较长块/回跳分流、384 字节多块窗口、`LC_HVDBT_PTE_SAME64=1` 和 `LC_HVDBT_RETAIN_SHARED_DEAD=1` 时，一次启动在约 130 秒进入桌面。跑分和桥接补丁后的冷启动需要单独验证。

- Sprint 3 原实例实测桌面和 QGA 可用。通过 QGA 启动 Geekbench CPU 基准测试与 `geekbench7.exe --help` 均以 Windows `0xC0000005` 退出，无 JSON；`cmd /d /c echo QGA_OK` 对照成功。该实例早于本轮调用返回预测补丁启动。
- 同主机有并行 Antigravity 实例时，带本轮补丁的隔离多块测试到 150 秒 QGA 仍未响应，累计翻译 63.4 秒、私有代码缓存重置 18 次。因并行负载和临时磁盘层不同，该次耗时不作为 130 秒结果的直接比较。

- 同一隔离磁盘、同一补丁版本下关闭 FEX 多块作为对照。91 秒时 QGA 首次通过、画面为“请等候 本地会话管理器”；121 秒截屏尚未显示桌面，151 秒确认进入桌面（QGA 仍可用）。该次 151 秒是离散采样的上界。

- 本轮补丁的隔离单块实例也以 `0xC0000005` 结束 Geekbench CPU 跑分。相同磁盘基底和其他设置下，`LC_HVDBT_FEX=0` 的对照在第 46 秒采样点已进入桌面，`geekbench7.exe --help` 正常退出。跑分故障可缩小到启用 FEX 的运行路径，尚未定位出错指令。

- 在新的隔离单块实例内再次复现 `geekbench7.exe --help` 的退出状态 `0xC0000005`。WER 没有生成 minidump。Windows 调试器记录了正常的初始断点与模块加载，随后收到进程退出事件，但没有收到访问异常事件。最后记录的加载模块是 `amsi.dll`，仅作为启动阶段的时间线线索。

### Geekbench 的 FEX 路径隔离

- 8 项短差分探针覆盖正反向字符串操作、分支和标量 SIMD；另有一项探针检查用户指令在缺页恢复时不会重放前一条加法。这些探针均与 TCG 一致。
- `LC_HVDBT_FEX_NO_CHAIN=1` 让 FEX 退出时恢复 HVDBT 寄存器布局并使用原有查找路径，同时禁用编译期和延迟 FEX 直连。在关闭多块的 Win2022 隔离实例里，QGA 在 91 秒采样点响应，但 `geekbench7.exe --help` 仍以 `0xC0000005` 退出。
- `LC_HVDBT_FEX_NATIVE_IMAGE=geekbench7.exe` 是 Win64 按进程分流诊断，对识别到的目标进程的新块选用原生 HVDBT。相同的可执行文件以副本名继续走 FEX，便于在同一来宾中保持系统服务和启动状态相同。

- 同一启用 FEX 的 Win2022 实例内，日志确认原版 `geekbench7.exe` 被选为原生 HVDBT，副本使用 FEX。两者的 `--help` 都以 `0xC0000005` 退出。只分流 Geekbench 主进程不足以解决问题，仍需检查依赖进程和更早发生的系统状态变化。
- 另行复现了 FEX 链接块在循环中无法响应宿主暂停的问题。先前监视器在 FEX 模式下一律拒绝边界；现允许在 FEX 块体入口（首条客户机指令执行前）交接，并在宿主暂停前转换寄存器布局。双块无限互跳测试中，旧版 QMP `stop` 超时，新版连续三次暂停并恢复成功。
