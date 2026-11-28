import CEngine
import Foundation
import Hypervisor

/// What the embedding machine provides: x86 devices, MSRs and CPUID stay out here. QEMU's AccelOps layer is meant to implement
/// this protocol. Each call runs on the vCPU thread with the guest stopped at an x86 boundary or an ARM load.
public protocol EngineMachine: AnyObject {
    func mmioRead(address: UInt64, size: Int) -> UInt64
    func mmioWrite(address: UInt64, size: Int, value: UInt64)
    func portRead(port: UInt16, size: Int) -> UInt32
    func portWrite(port: UInt16, size: Int, value: UInt32)
    func readMSR(_ msr: UInt32) -> UInt64?
    func writeMSR(_ msr: UInt32, value: UInt64) -> Bool
    func cpuid(leaf: UInt32, subleaf: UInt32) -> (eax: UInt32, ebx: UInt32, ecx: UInt32, edx: UInt32)
    /// Returns the next external interrupt vector the local APIC model would deliver, if any.
    func acknowledgeInterrupt() -> UInt8?
    /// The guest ran `hlt`. Blocks until an interrupt may be pending. False: nothing can ever wake it, the run ends.
    func waitForInterrupt(deadlineNanos: UInt64) -> Bool
    /// EOI of a level-triggered vector, for the IO-APIC model.
    func apicEOI(vector: UInt8)
    /// An IPI the local APIC model cannot finish alone. High word in bits 63:32.
    func apicICR(_ icr: UInt64)
    /// Hybrid: CR8/TPR update forwarded directly to QEMU's APIC.
    func apicTPR(_ tpr: UInt8)
    /// Hybrid: VMware backdoor (0x5658) syncing [RAX,RCX,RDX,RBX,RBP,RSI,RDI] directly with QEMU vmport.
    func vmport(regs: inout [UInt64], port: UInt16, size: Int, isWrite: Bool) -> Bool
    /// True when an interrupt wants delivering; the engine then raises the guest's IRQ line on the vCPU thread.
    func hasInterruptWaiting() -> Bool
    /// End this run at a host-visible boundary, including a completed HLT.
    /// Read on the vCPU thread; a machine publishing this from another thread must synchronize it.
    var shouldStopRun: Bool { get }
    /// User-mode shim only: an x86 `syscall` with no kernel model. Registers are in `state`.
    func syscall(_ state: UnsafeMutablePointer<EngineCtx>, engine: Engine) throws
}

extension EngineMachine {
    public var shouldStopRun: Bool { false }
    public func apicTPR(_ tpr: UInt8) {}
    public func vmport(regs: inout [UInt64], port: UInt16, size: Int, isWrite: Bool) -> Bool { false }
}

public struct EngineOptions {
    public var hostRAM: UnsafeMutableRawPointer?
    public var ramBytes: UInt64 = 2 << 30
    public var preciseFlags = true
    public var useTSO = true
    public var kernelModel = false
    public var tzcntLzcnt = false
    public var codeCacheBytes: UInt64 = 0
    public var shadowPoolPages: UInt32 = 0
    /// Nonzero: this many bytes of RAM at GPA 0, the rest at GPA 4 GB.
    public var ramBelow4G: UInt64 = 0
    /// Another CPU emulator owns the machine and takes single instructions back: see `ENGINE_RUN_FALLBACK`.
    public var hybrid = false
    public init() {}
}

public enum RunOutcome: Equatable, Sendable {
    case boundary(rip: UInt64)
    case halted(rax: UInt64)
    case exited(code: Int32)
    case fallback(rip: UInt64)
}

/// Who is in the code cache. Resetting it needs them all out, each at an x86 boundary or out of the engine, and none back in until it is done.
final class CodeCacheGate: @unchecked Sendable {
    let condition = NSCondition()
    var inside = 0
    var flushing = false
    var flushGeneration: UInt64 = 0
    var engines: [Engine] = []
    var lastBackendPath = ""
    var lastBackendCheckTicks: UInt64 = 0

    func enter() {
        condition.lock()
        while flushing { condition.wait() }
        inside += 1
        condition.unlock()
    }

    func leave() {
        condition.lock()
        inside -= 1
        condition.broadcast()
        condition.unlock()
    }
}

/// One x86 vCPU, created and run on the calling thread. The first one makes the VM; `init(secondaryOf:index:)` adds another to it, on its own thread.
public final class Engine {
    public let options: EngineOptions
    public let cpuIndex: UInt32
    private let translatorLock: TranslationLock
    private let gate: CodeCacheGate
    public let ram: UnsafeMutableRawPointer
    let priv: UnsafeMutableRawPointer
    public let ctx: UnsafeMutablePointer<EngineCtx>
    public weak var machine: EngineMachine?
    var vcpu: hv_vcpu_t = 0
    var vcpuExit: UnsafeMutablePointer<hv_vcpu_exit_t>!
    public private(set) var exitCount: UInt64 = 0
    private nonisolated(unsafe) var watchdogFired = false
    private var trace = Int(ProcessInfo.processInfo.environment["ENGINE_TRACE"] ?? "") ?? 0
    private let nativeWindowsImages: Set<String> = {
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["LC_HVDBT_FEX_NATIVE_IMAGE"] {
            return Set(explicit.lowercased().split(separator: ",").map(String.init))
        }
        return []
    }()
    private var windowsProcessImages: [UInt64: (parameters: UInt64, isNative: Bool)] = [:]
    public var exitCode: Int32?
    public var requestStop = false
    public internal(set) var boundaryRIP: UInt64 = 0
    private var vtimerMasked = false
    private var vtimerExits = 0
    private var irqLine = false
    private var vtimerOffset: UInt64 = 0
    /// One past the highest RAM GPA. Below it there can be holes, which `PI_RAM` tells apart.
    public let ramTop: UInt64
    public var tally: Tally?
    // For stall reports, read from another thread. One word each, written around each exit.
    private nonisolated(unsafe) var inVCPURun = false
    private nonisolated(unsafe) var lastHostCall = 0
    private nonisolated(unsafe) var fallbackCount: UInt64 = 0
    private nonisolated(unsafe) var portCount: UInt64 = 0
    private var r3ExcLogged: Int = 0
    private var localBackendCheckTicks: UInt64 = 0
    // Explicit component-research diagnostic. Disabled in normal launches;
    // the host already times exits, so only a slow enabled event writes RAM.
    let latencyTracing = ProcessInfo.processInfo.environment["VIM_HYBRID_LATENCY_TRACE"] == "1"
    private static let latencyFloorTicks: UInt64 = {
        var scale = mach_timebase_info_data_t()
        mach_timebase_info(&scale)
        return (2_000_000 * UInt64(scale.denom) + UInt64(scale.numer) - 1) / UInt64(scale.numer)
    }()

    func recordLatency(kind: UInt32, code: UInt32, start: UInt64, end: UInt64,
                       wait: UInt64, detail0: UInt64, detail1: UInt64,
                       view: UInt32, cpl: UInt32) {
        // HLT is intentional idle time, not slow execution; it would evict
        // the interesting records. This diagnostic does not qualify wakeup.
        if kind == UInt32(ENGINE_LATENCY_HOST_CALL) && code == UInt32(HC_HLT) { return }
        if latencyTracing && end &- start >= Self.latencyFloorTicks {
            engine_latency_record(cpuIndex, kind, code, start, end &- start,
                                  wait, detail0, detail1, view, cpl)
        }
    }

    public init(options: EngineOptions) throws {
        if options.useTSO {
            guard #available(macOS 15.0, *) else {
                throw EngineError.stop("Hybrid hardware TSO requires macOS 15 or later")
            }
        }
        self.options = options
        cpuIndex = 0
        translatorLock = try TranslationLock()
        gate = CodeCacheGate()
        let below = options.ramBelow4G
        ramTop = below != 0 && options.ramBytes > below ? (1 << 32) + options.ramBytes - below : options.ramBytes
        // The monitor's per-frame tables hold 32 GB of frames; see MIRROR_HEAD_OFF and RMAP_HEAD_OFF.
        if ramTop > 32 << 30 { throw EngineError.stop("RAM reaches past GPA 32 GB") }
        try hvCheck("hv_vm_create") {
            if #available(macOS 13.0, *) {
                let config = hv_vm_config_create()
                if #available(macOS 26.0, *) {
                    _ = hv_vm_config_set_ipa_granule(config, HV_IPA_GRANULE_16KB)
                }
                return hv_vm_create(config)
            } else {
                return hv_vm_create(nil)
            }
        }
        let noHugePages = ProcessInfo.processInfo.environment["LC_HVDBT_NO_HUGEPAGES"] == "1"
        func anon(_ bytes: UInt64) throws -> UnsafeMutableRawPointer {
            let align: UInt64 = 2 << 20
            if !noHugePages && bytes >= align {
                let total = Int(bytes + align)
                let raw = mmap(nil, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
                guard let raw, raw != MAP_FAILED else { throw EngineError.stop("mmap failed") }
                let rawAddr = UInt64(UInt(bitPattern: raw))
                let alignedAddr = (rawAddr + align - 1) & ~(align - 1)
                let lead = Int(alignedAddr - rawAddr)
                let tail = total - lead - Int(bytes)
                if lead > 0 { munmap(raw, lead) }
                if tail > 0 { munmap(raw + lead + Int(bytes), tail) }
                return raw + lead
            }
            let p = mmap(nil, Int(bytes), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
            guard let p, p != MAP_FAILED else { throw EngineError.stop("mmap failed") }
            return p
        }
        ram = try options.hostRAM ?? anon(options.ramBytes)
        priv = try anon(kPrivSize)
        ctx = (priv + Int(kOffCtx)).bindMemory(to: EngineCtx.self, capacity: 1)
        let rwx = hv_memory_flags_t(HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC)
        if ramTop > options.ramBytes {
            try hvCheck("hv_vm_map ram below 4 GB") { hv_vm_map(ram, 0, Int(below), rwx) }
            try hvCheck("hv_vm_map ram above 4 GB") { hv_vm_map(ram + Int(below), 1 << 32, Int(options.ramBytes - below), rwx) }
        } else {
            try hvCheck("hv_vm_map ram") { hv_vm_map(ram, 0, Int(options.ramBytes), rwx) }
        }
        try hvCheck("hv_vm_map private") { hv_vm_map(priv, kPrivIPA, Int(kPrivSize), rwx) }
        try loadMonitor()
        var config = TrConfig()
        config.priv = priv.assumingMemoryBound(to: UInt8.self)
        config.ram = ram.assumingMemoryBound(to: UInt8.self)
        config.ram_size = ramTop
        config.ram_below_4g = ramTop > options.ramBytes ? below : 0
        config.hybrid = options.hybrid
        let environment = ProcessInfo.processInfo.environment
        config.no_bulk_rep = options.hybrid && environment["LC_HVDBT_NO_BULK"] == "1"
        config.no_reuse = environment["LC_HVDBT_NO_REUSE"] == "1"
        config.profile_invalidations = environment["LC_HVDBT_PROFILE_INVALIDATION"] == "1"
        // Track which views actually contain a shared block's lookup. The
        // translator retains its full scan if allocation/tracking is uncertain.
        config.invalidation_membership = true
        config.profile_shift_helpers = environment["LC_HVDBT_PROFILE_SHIFT_HELPERS"] == "1"
        if environment["VIM_HYBRID_HELPER_CENSUS"] == "1" { config.profile_shift_helpers = true }
        config.inline_shr8 = environment["LC_HVDBT_INLINE_SHR8"] == "1"
        config.retain_shared_dead = environment["LC_HVDBT_RETAIN_SHARED_DEAD"] == "1"
        config.profile_compile_inputs = environment["LC_HVDBT_PROFILE_COMPILE_INPUTS"] == "1"
        config.page_rep = environment["LC_HVDBT_PAGE_REP"] == "1"
        config.no_return_prediction = options.hybrid && environment["LC_HVDBT_NO_RETURN_PREDICTION"] == "1"
        config.all_flags_live = options.hybrid && environment["LC_HVDBT_ALL_FLAGS"] == "1"
        config.keep_kernel_tables = !options.hybrid || environment["LC_HVDBT_KEEP_KPTI"] == "1"
        config.precise_flags = options.preciseFlags
        config.has_tzcnt_lzcnt = options.tzcntLzcnt
        config.code_cache_bytes = options.codeCacheBytes
        let stubs = kPrivVA + kOffStubs
        config.stub_lookup = stubs
        config.stub_ret_miss = stubs + 0x80
        config.stub_pushf = stubs + 0x100
        config.stub_sahf = stubs + 0x180
        tr_init(&config)
        ctx.pointee.vcpu_count = 1
        gate.engines = [self]
        try setUpVCPU()
    }

    deinit {
        if cpuIndex == 0 { tr_aot_flush() }
    }

    /// Another vCPU in `first`'s VM. Call on the thread that will run it. `prepareCount` must have been called before the first vCPU ran; this one may start later.
    public init(secondaryOf first: Engine, index: UInt32) throws {
        options = first.options
        cpuIndex = index
        translatorLock = first.translatorLock
        gate = first.gate
        ram = first.ram
        priv = first.priv
        ramTop = first.ramTop
        bootTTBR = first.bootTTBR
        ctx = (priv + Int(kOffCtx + UInt64(index) * kCtxSlot)).bindMemory(to: EngineCtx.self, capacity: 1)
        try setUpVCPU()
        // One clock: the TSC and the APIC timer read this counter on every vCPU.
        vtimerOffset = first.vtimerOffset
        try hvCheck("set vtimer offset") { hv_vcpu_set_vtimer_offset(vcpu, vtimerOffset) }
        gate.condition.lock()
        gate.engines.append(self)
        gate.condition.unlock()
    }

    var firstContext: UnsafeMutablePointer<EngineCtx> { (priv + Int(kOffCtx)).assumingMemoryBound(to: EngineCtx.self) }
    private var severalVCPUs: Bool { firstContext.pointee.vcpu_count > 1 }

    /// Resets the code cache with every other vCPU out of it. This vCPU is inside, in a host call, with the translator lock not held. Its monitor holds no pointer into the
    /// cache that it uses again, and nor does a vCPU at a boundary: each resumes through translate_here.
    private func resetCodeCacheWithOthersOut(afterGeneration: UInt64? = nil, forceAll: Bool = false, forcePrivate: Bool = false) {
        gate.condition.lock()
        if gate.flushing { // another vCPU got here first: this one counts as out
            gate.inside -= 1
            gate.condition.broadcast()
            while gate.flushing { gate.condition.wait() }
            gate.inside += 1
            gate.condition.unlock()
            return
        }
        if let expected = afterGeneration, gate.flushGeneration != expected {
            gate.condition.unlock()
            return
        }
        gate.flushing = true
        while gate.inside > 1 {
            for engine in gate.engines where engine !== self { engine.requestBoundary() }
            gate.condition.wait(until: Date(timeIntervalSinceNow: 0.001))
        }
        gate.condition.unlock()
        translatorLock.lock()
        if tr_needs_flush() {
            tr_flush_parked(false)
        } else if forceAll {
            tr_flush_parked(true)
        } else if forcePrivate {
            tr_flush_parked(false)
        }
        for engine in gate.engines { engine.ctx.pointee.tr_flushed = 1 }
        translatorLock.unlock()
        gate.condition.lock()
        gate.flushGeneration &+= 1
        gate.flushing = false
        gate.condition.broadcast()
        gate.condition.unlock()
    }

    private func reloadBackendWithOthersOut(_ path: String) {
        if severalVCPUs {
            gate.condition.lock()
            if gate.flushing {
                gate.inside -= 1
                gate.condition.broadcast()
                while gate.flushing { gate.condition.wait() }
                gate.inside += 1
                gate.condition.unlock()
                return
            }
            gate.flushing = true
            while gate.inside > 1 {
                for engine in gate.engines where engine !== self { engine.requestBoundary() }
                gate.condition.wait(until: Date(timeIntervalSinceNow: 0.001))
            }
            gate.condition.unlock()
        }
        translatorLock.lock()
        let ok = path.withCString { tr_reload_backend($0) }
        if ok {
            tr_flush_parked(true)
            for engine in gate.engines { engine.ctx.pointee.tr_flushed = 1 }
            gate.lastBackendPath = path
        }
        translatorLock.unlock()
        if severalVCPUs {
            gate.condition.lock()
            gate.flushing = false
            gate.condition.broadcast()
            gate.condition.unlock()
        }
    }

    private func checkHotReloadBackend() {
        let now = mach_absolute_time()
        guard now &- localBackendCheckTicks >= 6_000_000 else { return } // ~0.25s at 24MHz, lock-free
        localBackendCheckTicks = now
        guard access("/tmp/hvdbt-reload.path", F_OK) == 0 else { return }
        gate.condition.lock()
        let prev = gate.lastBackendPath
        gate.condition.unlock()
        guard let raw = try? String(contentsOfFile: "/tmp/hvdbt-reload.path", encoding: .utf8) else { return }
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty && path != prev else { return }
        reloadBackendWithOthersOut(path)
    }

    /// On the first vCPU, before any vCPU runs: how many this VM will have.
    public func prepareCount(_ count: UInt32) { ctx.pointee.vcpu_count = count }

    // ---- Private memory ----
    private func loadMonitor() throws {
        let bytes = Array(UnsafeBufferPointer(start: engine_monitor_macho_ptr(), count: engine_monitor_macho_size()))
        let image = try MachOImage(bytes: bytes)
        image.copy(to: priv, origin: kPrivVA)
        let stubsBytes = Int(kOffCC - kOffStubs) // 2 MiB: covers all EL0 stubs (0x000..0xfff+)
        for dstOff in [kOffStubsHi, kOffStubs2, kOffStubs2Hi] {
            memcpy(priv + Int(dstOff), priv + Int(kOffStubs), stubsBytes) // position independent, within ±128 MiB of each half
            sys_icache_invalidate(priv + Int(dstOff), stubsBytes)
        }

        // Static tables for the private slots, 2 MB blocks, shared by every shadow root.
        var next = kOffPT
        func table() -> UInt64 {
            let at = next
            next += 4096
            return at
        }
        func slot(_ tableOff: UInt64, _ index: Int) -> UnsafeMutablePointer<UInt64> {
            (priv + Int(tableOff)).assumingMemoryBound(to: UInt64.self) + index
        }
        func ipa(_ off: UInt64) -> UInt64 { kPrivIPA + off }
        let valid: UInt64 = 3
        let af: UInt64 = 1 << 10
        let sh: UInt64 = 3 << 8
        let uxn: UInt64 = 1 << 54
        let pxn: UInt64 = 1 << 53
        let apEL1RW: UInt64 = 0 << 6
        let apRW: UInt64 = 1 << 6
        let apEL1RO: UInt64 = 2 << 6
        let apRO: UInt64 = 3 << 6
        let block = af | sh | 1

        // Boot root, used until the monitor builds the first view.
        let bootRoot = table()
        let privL1 = table()
        let privL2 = table()
        slot(bootRoot, Int((kPrivVA >> 39) & 0x1FF)).pointee = ipa(privL1) | valid
        slot(privL1, Int((kPrivVA >> 30) & 0x1FF)).pointee = ipa(privL2) | valid
        let privL2High = table()
        slot(privL1, Int((kPrivVA >> 30) & 0x1FF) + 1).pointee = ipa(privL2High) | valid
        func block2M(_ off: UInt64) -> UnsafeMutablePointer<UInt64> { slot(off >> 30 == 0 ? privL2 : privL2High, Int((off >> 21) & 0x1FF)) }
        var off: UInt64 = 0
        while off < kPrivSize {
            var attrs: UInt64
            switch off {
            case kOffMonText..<kOffMonData: attrs = apEL1RO | uxn  // monitor text: EL1 execute
            case kOffStubs..<(kOffCC + kCCSize), kOffStubs2..<(kOffCC2 + kCCSize): attrs = apEL1RW | pxn  // stubs and code: EL0 execute-only
            case kOffCtx..<kOffCtxEnd, kOffFexL1..<kOffFexL1End: attrs = apRW | uxn | pxn
            case kOffHash..<kOffRStacks: attrs = apRO | uxn | pxn
            default: attrs = apEL1RW | uxn | pxn
            }
            if off == kOffStubs || off == kOffStubsHi || off == kOffStubs2 || off == kOffStubs2Hi { attrs = apEL1RO | pxn }
            block2M(off).pointee = ipa(off) | block | attrs
            off += 2 << 20
        }
        // Return stacks grow down: each slot is an unmapped guard, then the writable part.
        for cpu in 0..<UInt32(MAX_VCPUS) {
            let stacks = engine_off_rstacks(cpu)
            var rstackL3: UInt64 = 0
            var page: UInt64 = 0
            while page < UInt64(MAX_VIEWS) * kRStackSlot {
                if page % (2 << 20) == 0 {
                    rstackL3 = table()
                    block2M(stacks + page).pointee = ipa(rstackL3) | valid
                }
                let inSlot = page % kRStackSlot
                if inSlot >= kRStackSlot - kRStackBytes {
                    slot(rstackL3, Int((page >> 12) & 0x1FF)).pointee =
                        ipa(stacks + page) | valid | af | sh | apRW | uxn | pxn
                }
                page += 4096
            }
        }
        // RAM alias for the monitor: 1 GB blocks where possible.
        let ramL1 = table()
        var gb: UInt64 = 0
        while gb < ramTop {
            slot(ramL1, Int((kRamAliasVA + gb) >> 30) & 0x1FF).pointee = gb | block | apEL1RW | uxn | pxn
            gb += 1 << 30
        }
        // Handed to the monitor: the root-level descriptors of its private slots.
        let handoff = (priv + Int(kOffMonData) + (1 << 19)).assumingMemoryBound(to: UInt64.self)
        handoff[0] = ipa(privL1) | valid
        handoff[1] = ipa(ramL1) | valid
        slot(bootRoot, Int((kRamAliasVA >> 39) & 0x1FF)).pointee = ipa(ramL1) | valid
        bootTTBR = ipa(bootRoot)
    }
    private var bootTTBR: UInt64 = 0

    private func setUpVCPU() throws {
        try hvCheck("hv_vcpu_create") { hv_vcpu_create(&vcpu, &vcpuExit, nil) }
        func sys(_ reg: hv_sys_reg_t, _ value: UInt64) throws {
            try hvCheck("set sys reg") { hv_vcpu_set_sys_reg(vcpu, reg, value) }
        }
        if #available(macOS 15.0, *) {
            var actlr: UInt64 = 0
            try hvCheck("read ACTLR") { hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_ACTLR_EL1, &actlr) }
            try sys(HV_SYS_REG_ACTLR_EL1, options.useTSO ? actlr | 2 : actlr & ~2)
            try hvCheck("verify ACTLR") { hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_ACTLR_EL1, &actlr) }
            if options.useTSO && actlr & 2 == 0 { throw EngineError.stop("Hardware TSO is unavailable") }
        } else if options.useTSO {
            throw EngineError.stop("Hybrid hardware TSO requires macOS 15 or later")
        }
        try sys(HV_SYS_REG_MAIR_EL1, 0xFF)
        // 48-bit VA both ways, 4 KB granule, 16-bit ASIDs from TTBR0, 36-bit IPA.
        let tcr: UInt64 =
            16 | 16 << 16 | 3 << 12 | 1 << 10 | 1 << 8 | 3 << 28 | 1 << 26 | 1 << 24 | 2 << 30 | 1 << 32 | 1 << 36
        try sys(HV_SYS_REG_TCR_EL1, tcr)
        try sys(HV_SYS_REG_TTBR0_EL1, bootTTBR)
        try sys(HV_SYS_REG_TTBR1_EL1, bootTTBR)
        try sys(HV_SYS_REG_VBAR_EL1, engine_sym_mon_vectors)
        // MMU, caches, nAA for LSE2, RES1 bits.
        try sys(HV_SYS_REG_SCTLR_EL1, 0x30D0_0800 | 1 | 1 << 2 | 1 << 12 | 1 << 6)
        try sys(HV_SYS_REG_CPACR_EL1, 3 << 20)
        let stackTop = kPrivVA + kOffMonCPU + (UInt64(cpuIndex) + 1) * kCtxSlot - 16
        try sys(HV_SYS_REG_SP_EL1, stackTop)
        try sys(HV_SYS_REG_TPIDR_EL1, stackTop)
        try hvCheck("set x18") { hv_vcpu_set_reg(vcpu, HV_REG_X18, kPrivVA + kOffCtx + UInt64(cpuIndex) * kCtxSlot) }
        hv_vcpu_get_vtimer_offset(vcpu, &vtimerOffset)
        try hvCheck("unmask vtimer") { hv_vcpu_set_vtimer_mask(vcpu, false) }
        ctx.pointee.apic_id = cpuIndex << 24
        ctx.pointee.ram_size = ramTop
        ctx.pointee.hybrid = options.hybrid ? 1 : 0
        let environment = ProcessInfo.processInfo.environment
        ctx.pointee.profile_mmu_observations = latencyTracing || environment["LC_HVDBT_MMU_OBSERVATIONS"] == "1" ? 1 : 0
        ctx.pointee.keeper_inactive_guard = environment["LC_HVDBT_KEEPER_INACTIVE_GUARD"] == "1" ? 1 : 0
        ctx.pointee.profile_shift_helpers = environment["LC_HVDBT_PROFILE_SHIFT_HELPERS"] == "1" ? 1 : 0
        if environment["VIM_HYBRID_HELPER_CENSUS"] == "1" { ctx.pointee.profile_shift_helpers = 1 }
        ctx.pointee.profile_x87_helpers = environment["VIM_HYBRID_HELPER_CENSUS"] == "1" ? 1 : 0
        ctx.pointee.pte_same64 = environment["LC_HVDBT_PTE_SAME64"] == "1" ? 1 : 0
        // hugepages_disabled bitmask:
        //   bit 0 (1): disable 2 MiB L2 Block
        //   bit 1 (2): disable 64 KiB PTE_CONT (fractured large pages)
        //   bit 2 (4): disable 64 KiB PTE_CONT on 4 KiB guest PTEs (default off to avoid 16x TLBI churn; opt-in via LC_HVDBT_HP_64K_4K=1)
        //   bit 3 (8): disable zap_table batched tlbi aside1is (for exact baseline A/B)
        var hpDisabled: UInt32 = environment["LC_HVDBT_HP_64K_4K"] == "1" ? 0 : 4
        if environment["LC_HVDBT_NO_HUGEPAGES"] == "1" { hpDisabled = 15 }
        if environment["LC_HVDBT_NO_HP_2M"] == "1" { hpDisabled |= 1 }
        if environment["LC_HVDBT_NO_HP_64K"] == "1" { hpDisabled |= 6 }
        ctx.pointee.hugepages_disabled = hpDisabled
        // ENGINE_SHARE: the keeper rule in a test with no fallbacks, as --smp is, so that it runs with vCPUs really side by side.
        ctx.pointee.share_user = (options.hybrid && environment["LC_HVDBT_NO_SHARE"] != "1") || environment["ENGINE_SHARE"] == "1" ? 1 : 0
        ctx.pointee.clts_on_host = options.hybrid && ProcessInfo.processInfo.environment["LC_HVDBT_CLTS_TCG"] == "1" ? 1 : 0
        ctx.pointee.tsc_den = 1
        ctx.pointee.resume_mxcsr = 0x1F80
        if cpuIndex == 0 {
            let pageInfo = priv + Int(kOffPageInfo)
            let lowFrames = Int(min(options.ramBelow4G != 0 ? options.ramBelow4G : options.ramBytes, options.ramBytes) >> 12)
            memset(pageInfo, Int32(PI_RAM), lowFrames)
            if ramTop > options.ramBytes { memset(pageInfo + (1 << 20), Int32(PI_RAM), Int((ramTop - (1 << 32)) >> 12)) }
        }
        ctx.pointee.df = 1
        ctx.pointee.rflags_sys = 0x202
        ctx.pointee.spsr = 1 << 29  // EL0t, C set: x86 CF clear
        ctx.pointee.model_kernel = options.kernelModel ? 1 : 0
        ctx.pointee.pool_limit = options.shadowPoolPages
        ctx.pointee.cr0 = 0x8001_0033
        ctx.pointee.cr4 = 0x620
        ctx.pointee.efer = 0xD01
        ctx.pointee.xcr0 = 1
    }

    // ---- Hyper-V Enlightenments shared block at OFF_MON_DATA + 0x81000 ----
    static let hvSharedOffset = Int(kOffMonData) + 0x81000
    static let hvSharedMagic: UInt64 = 0x4856_454E_4C49_4748 // "HVENLIGH"
    static let hvSharedVersion: UInt32 = 1
    static let hvCpuBaseOffset = 64
    static let hvCpuStride = 192

    private func initHyperVSharedBlock() {
        let base = priv + Self.hvSharedOffset
        if base.load(fromByteOffset: 0, as: UInt64.self) == Self.hvSharedMagic { return }
        guard let machine else { return }
        let iface = machine.cpuid(leaf: 0x4000_0001, subleaf: 0)
        memset(base, 0, 4096)
        if iface.eax == 0x3123_7648 { // "Hv#1"
            let feat = machine.cpuid(leaf: 0x4000_0003, subleaf: 0)
            let enl = machine.cpuid(leaf: 0x4000_0004, subleaf: 0)
            base.storeBytes(of: Self.hvSharedVersion, toByteOffset: 0x08, as: UInt32.self)
            base.storeBytes(of: feat.eax, toByteOffset: 0x10, as: UInt32.self)
            base.storeBytes(of: feat.edx, toByteOffset: 0x14, as: UInt32.self)
            base.storeBytes(of: enl.eax, toByteOffset: 0x18, as: UInt32.self)
            base.storeBytes(of: enl.ebx, toByteOffset: 0x1C, as: UInt32.self)
            base.storeBytes(of: UInt32(1), toByteOffset: 0x0C, as: UInt32.self)
            base.storeBytes(of: Self.hvSharedMagic, toByteOffset: 0x00, as: UInt64.self)
        } else {
            base.storeBytes(of: Self.hvSharedVersion, toByteOffset: 0x08, as: UInt32.self)
            base.storeBytes(of: UInt32(0), toByteOffset: 0x0C, as: UInt32.self)
            base.storeBytes(of: Self.hvSharedMagic, toByteOffset: 0x00, as: UInt64.self)
        }
    }

    private func resetHyperVSharedState() {
        let base = priv + Self.hvSharedOffset
        guard base.load(fromByteOffset: 0, as: UInt64.self) == Self.hvSharedMagic else { return }
        base.storeBytes(of: UInt64(0), toByteOffset: 0x20, as: UInt64.self) // guest_os_id
        base.storeBytes(of: UInt64(0), toByteOffset: 0x28, as: UInt64.self) // hypercall_msr
        base.storeBytes(of: UInt64(0), toByteOffset: 0x30, as: UInt64.self) // ref_tsc_msr
        base.storeBytes(of: UInt32(0), toByteOffset: 0x38, as: UInt32.self) // tsc_seq
        for cpu in 0..<Int(MAX_VCPUS) {
            base.storeBytes(of: UInt64(0), toByteOffset: Self.hvCpuBaseOffset + cpu * Self.hvCpuStride, as: UInt64.self)
        }
    }

    // ---- Running ----

    /// Starts the x86 CPU at `rip`. The caller filled registers in `ctx` and page tables in RAM.
    public func run(entry rip: UInt64) throws -> RunOutcome {
        if cpuIndex == 0 { initHyperVSharedBlock() }
        ctx.pointee.req_a.0 = rip
        if let seconds = ProcessInfo.processInfo.environment["ENGINE_WATCHDOG"].flatMap(Double.init) {
            let target = vcpu
            Thread.detachNewThread { [weak self] in
                Thread.sleep(forTimeInterval: seconds)
                self?.watchdogFired = true
                var v = target
                hv_vcpus_exit(&v, 1)
            }
        }
        try hvCheck("set pc") { hv_vcpu_set_reg(vcpu, HV_REG_PC, engine_sym_mon_init) }
        try hvCheck("set cpsr") { hv_vcpu_set_reg(vcpu, HV_REG_CPSR, 0x3C5) }
        return try runLoop()
    }

    // For the tally: ticks inside hv_vcpu_run, and per host call their number and the ticks from the exit to the end of handling.
    var ticksInVCPU: UInt64 = 0
    var pcAtKick: UInt64 = 0
    var lrAtKick: UInt64 = 0
    var hostCallCount = [UInt64](repeating: 0, count: 32)
    var hostCallTicks = [UInt64](repeating: 0, count: 32)

    // For the tally: how much of a translation is waiting for another vCPU's.
    var translatorWaits: UInt64 = 0
    var translatorWaitTicks: UInt64 = 0
    var invalidations: UInt64 = 0, invalidationWaitTicks: UInt64 = 0, invalidationTicks: UInt64 = 0 // HC_VIEW_INVALIDATE_VA, for the tally
    private func lockTranslator() {
        if translatorLock.tryLock() { return }
        let start = mach_absolute_time()
        translatorLock.lock()
        translatorWaits &+= 1
        translatorWaitTicks &+= mach_absolute_time() &- start
    }

    private func runLoop() throws -> RunOutcome {
        let counted = severalVCPUs
        if counted { gate.enter() }
        defer { if counted { gate.leave() } }
        while true {
            // The hypervisor forgets a pending interrupt at each exit, so the line is held here until the monitor takes it.
            if irqLine { hv_vcpu_set_pending_interrupt(vcpu, HV_INTERRUPT_TYPE_IRQ, true) }
            inVCPURun = true
            let entered = mach_absolute_time()
            let r = hv_vcpu_run(vcpu)
            let left = mach_absolute_time()
            ticksInVCPU &+= left &- entered
            inVCPURun = false
            if r != HV_SUCCESS { throw EngineError.hypervisor("hv_vcpu_run", r) }
            exitCount += 1
            switch vcpuExit.pointee.reason {
            case HV_EXIT_REASON_EXCEPTION:
                let syndrome = vcpuExit.pointee.exception.syndrome
                let ec = syndrome >> 26
                // Another vCPU's unmapDirect took this memory while this one's shadows still lead there. The kick path's IRQ drops them.
                if ec == 0x24, ctx.pointee.direct_flush != 0 {
                    irqLine = true
                    continue
                }
                if ec != 0x16 {
                    var pc: UInt64 = 0
                    hv_vcpu_get_reg(vcpu, HV_REG_PC, &pc)
                    throw EngineError.stop(
                        "unexpected exit: syndrome \(hex(syndrome)) pc \(hex(pc)) va \(hex(vcpuExit.pointee.exception.virtual_address))")
                }
                var code: UInt64 = 0
                hv_vcpu_get_reg(vcpu, HV_REG_X0, &code)
                if trace > 0 {
                    trace -= 1
                    FileHandle.standardError.write(
                        Data(
                            "hc \(code) a0 \(hex(ctx.pointee.req_a.0)) a1 \(hex(ctx.pointee.req_a.1)) a2 \(hex(ctx.pointee.req_a.2)) elr \(hex(ctx.pointee.elr)) esr \(hex(ctx.pointee.esr))\n"
                                .utf8))
                }
                lastHostCall = Int(code)
                let observed = latencyTracing ? (translatorWaitTicks, ctx.pointee.req_a.1,
                    ctx.pointee.req_a.2, ctx.pointee.view, ctx.pointee.cpl) : (0, 0, 0, 0, 0)
                let outcome = try hostCall(Int(code))
                let completed = mach_absolute_time()
                if code < 32 {
                    hostCallCount[Int(code)] &+= 1
                    hostCallTicks[Int(code)] &+= completed &- left
                }
                if latencyTracing {
                    recordLatency(kind: UInt32(ENGINE_LATENCY_HOST_CALL), code: UInt32(truncatingIfNeeded: code),
                                  start: left, end: completed, wait: translatorWaitTicks &- observed.0,
                                  detail0: observed.1, detail1: observed.2, view: observed.3, cpl: observed.4)
                }
                if let outcome { return outcome }
            case HV_EXIT_REASON_CANCELED:
                if watchdogFired {
                    var pc: UInt64 = 0
                    var elr: UInt64 = 0
                    var esr: UInt64 = 0
                    var far: UInt64 = 0
                    hv_vcpu_get_reg(vcpu, HV_REG_PC, &pc)
                    hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_ELR_EL1, &elr)
                    hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_ESR_EL1, &esr)
                    hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_FAR_EL1, &far)
                    let regs = withUnsafeBytes(of: &ctx.pointee.x) { Array($0.bindMemory(to: UInt64.self)) }
                    var ctl: UInt64 = 0
                    var cval: UInt64 = 0
                    hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_CNTV_CTL_EL0, &ctl)
                    hv_vcpu_get_sys_reg(vcpu, HV_SYS_REG_CNTV_CVAL_EL0, &cval)
                    FileHandle.standardError.write(
                        Data(
                            "vtimer ctl \(ctl) cval \(hex(cval)) deadline \(hex(ctx.pointee.apic_deadline_cnt)) now \(hex(mach_absolute_time() &- vtimerOffset)) lvt \(hex(UInt64(ctx.pointee.apic_lvt_timer))) vtexits \(vtimerExits) masked \(vtimerMasked) irqs \(ctx.pointee.stat_irq) apictimer \(ctx.pointee.stat_apic_timer) rf \(hex(ctx.pointee.rflags_sys))\n"
                                .utf8))
                    throw EngineError.stop(
                        "watchdog: pc \(hex(pc)) elr \(hex(elr)) esr \(hex(esr)) far \(hex(far)) exits \(exitCount) "
                            + "ctx.elr \(hex(ctx.pointee.elr)) x16 \(hex(regs[16])) x30 \(hex(regs[30]))")
                }
                // A kick means a device interrupt, a posted vector or a stop request: the monitor sorts out which.
                if tally != nil {
                    hv_vcpu_get_reg(vcpu, HV_REG_PC, &pcAtKick)  // the only live view of a vCPU that stays in the monitor, for the stall report
                    hv_vcpu_get_reg(vcpu, HV_REG_LR, &lrAtKick)
                }
                if options.hybrid {
                    let line = machine?.hasInterruptWaiting() ?? false
                    engine_ctx_set_irq_line(ctx, line)
                    let ifOpen = (ctx.pointee.rflags_sys & 0x200) != 0
                    let fc = firstContext.pointee
                    if (line && ifOpen) || ctx.pointee.stop_requested != 0 || ctx.pointee.direct_flush != 0 ||
                        (ctx.pointee.irq_pending != 0 && ifOpen) ||
                        fc.note_head != fc.note_tail || fc.note_overflow != 0 {
                        irqLine = true
                    }
                    continue
                }
                irqLine = true
                continue
            case HV_EXIT_REASON_VTIMER_ACTIVATED:
                // The APIC timer's compare fired. The monitor rearms it during the IRQ it now takes, then HC_IRQ unmasks.
                hv_vcpu_set_vtimer_mask(vcpu, true)
                vtimerMasked = true
                vtimerExits += 1
                irqLine = true
                continue
            default:
                throw EngineError.stop("unexpected exit reason \(vcpuExit.pointee.reason)")
            }
        }
    }

    public func stats() -> String {
        let t = tr_stats()
        var line =
            "exits \(exitCount) blocks \(t.blocks) x86-insns \(t.insns) arm-words \(t.words) "
            + "stack \(ctx.pointee.stat_stack_deepest) cache-resets \(t.cache_resets) invalidated \(t.invalidated) fills \(ctx.pointee.stat_fills) dirty \(ctx.pointee.stat_dirty) "
            + "smc \(ctx.pointee.stat_smc) pfs \(ctx.pointee.stat_pf_delivered) pt-writes \(ctx.pointee.stat_pt_writes) "
            + "steps \(ctx.pointee.stat_steps) irqs \(ctx.pointee.stat_irq) view-evictions \(ctx.pointee.stat_view_evictions) unshadowed \(ctx.pointee.stat_pt_unshadow) pool-reclaims \(ctx.pointee.stat_pool_reclaim)"
        if t.insns > 0 { line += String(format: " words/insn %.2f", Double(t.words) / Double(t.insns)) }
        line += "\nlookup tails (global translator, not per-vCPU): calls \(t.call_tails) jumps \(t.jump_tails)"
        if t.invalidation_membership_enabled != 0 {
            line += "\nmembership (global translator, not per-vCPU): enabled \(t.invalidation_membership_enabled) reliable \(t.invalidation_membership_reliable) bytes \(t.invalidation_membership_bytes) "
                + "inserts \(t.invalidation_membership_inserts) visits \(t.invalidation_membership_view_visits) stales \(t.invalidation_membership_stale_visits) removals \(t.invalidation_membership_removals) "
                + "trackFailures \(t.invalidation_membership_tracking_failures) fullScans \(t.invalidation_membership_full_scans)"
        }
        if ctx.pointee.keeper_inactive_guard != 0 {
            line += "\nkeeper inactive guard (this vCPU): enabled \(ctx.pointee.keeper_inactive_guard) "
                + "suppressed \(ctx.pointee.stat_keeper_inactive_suppressed) thresholdFallbacks \(ctx.pointee.stat_keeper_inactive_threshold_fallbacks) "
                + "sharedDirs \(firstContext.pointee.shared_dir_count)"
        }
        line += "\nshared dead hints (global translator; repeated, do not sum): enabled \(t.retain_shared_dead_enabled) "
            + "retained \(t.shared_dead_retained_hints) cleared \(t.shared_dead_cleared_hints)"
        line += "\ncompile inputs (inert global census; repeated, do not sum): requested \(t.compile_inputs_enabled) "
            + "available \(t.compile_inputs_available) bytes \(t.compile_inputs_bytes) attempts \(t.compile_inputs_attempts) "
            + "admitted \(t.compile_inputs_admitted) exact \(t.compile_inputs_exact) evictions \(t.compile_inputs_evictions)"
        line += "\npage REP: enabled \(t.page_rep_enabled) (global configuration; repeated), "
            + "this vCPU committed prefixes \(ctx.pointee.stat_rep_page_prefixes) bytes \(ctx.pointee.stat_rep_page_bytes)"
        return line
    }

    private func hostCall(_ code: Int) throws -> RunOutcome? {
        if code == Int(HC_TRANSLATE) || code == Int(HC_BOUNDARY) || code == Int(HC_HLT) {
            checkHotReloadBackend()
        }
        let c = ctx
        // The translator's tables are one set for the VM. Taken here, in the slow path of a host exit that costs far more.
        let translating = code == Int(HC_TRANSLATE) || code == Int(HC_SMC) || code == Int(HC_VIEW_INVALIDATE_VA) || code == Int(HC_VIEW_FORGET) || code == Int(HC_VIEW_DROP) || code == Int(HC_CALL_TAIL)
        var cacheWasReset = false
        if code == Int(HC_TRANSLATE) && severalVCPUs {
            lockTranslator()
            while tr_needs_flush() {
                translatorLock.unlock()
                resetCodeCacheWithOthersOut()
                cacheWasReset = true
                lockTranslator()
            }
        } else if translating {
            let start = code == Int(HC_VIEW_INVALIDATE_VA) ? mach_absolute_time() : 0
            lockTranslator()
            if start != 0 { invalidationWaitTicks &+= mach_absolute_time() &- start }
        }
        defer { if translating { translatorLock.unlock() } }
        switch code {
        case Int(HC_TRANSLATE):
            let view = Int32(c.pointee.req_a.0)
            let rip = c.pointee.req_a.1
            let site = c.pointee.req_a.2
            let kind = Int32(c.pointee.req_a.3)
            let coldBefore = tally != nil ? tr_stats().blocks : 0
            var result: TrResult
            while true {
                let flushGen = gate.flushGeneration
                let r = tr_translate(view, c.pointee.req_a.5, Int32(c.pointee.cpl), c.pointee.compat != 0, rip, c.pointee.req_a.4 != 0)
                if r.needs_flush && severalVCPUs {
                    let isSharedPart = (UInt32(bitPattern: r.key) & 0x4000_0000) != 0
                    translatorLock.unlock()
                    resetCodeCacheWithOthersOut(afterGeneration: flushGen, forceAll: isSharedPart, forcePrivate: !isSharedPart)
                    cacheWasReset = true
                    lockTranslator()
                    continue
                }
                result = r
                break
            }
            if let tally, c.pointee.cpl == 3, tr_stats().blocks != coldBefore { tally.coldUserBlock(rip: rip, cr3: c.pointee.req_a.5) }
            if result.fetch_fault_va != 0 {
                c.pointee.req_r.1 = result.fetch_fault_va + 1
                return nil
            }
            if result.unprotected != 0 {
                c.pointee.req_r.0 = 0
                c.pointee.req_r.1 = 0
                c.pointee.req_r.2 = (result.unprotected - 1) | 1
                c.pointee.req_r.3 = 0
                return nil
            }
            c.pointee.req_r.0 = result.arm
            c.pointee.req_r.1 = 0
            c.pointee.req_r.2 = result.frames.0 == ~0 ? 0 : result.frames.0 | 1
            c.pointee.req_r.3 = result.frames.1 == ~0 ? 0 : result.frames.1 | 1
            if kind == SVC_LOOKUP_MISS {
                // Only the source site went away on a flush. The new target can be looked up.
                tr_hash_insert(view, rip, result.arm)
            } else if !result.flushed && site != 0 {
                if !cacheWasReset { tr_link(site, kind, result.arm, result.key) } // or the site is in the old cache
                tr_hash_insert(view, rip, result.arm) // so the next site with this target links in the monitor
            }
        case Int(HC_SMC):
            _ = tr_invalidate_frame(c.pointee.req_a.0, UInt32(c.pointee.req_a.1))
        case Int(HC_VIEW_INVALIDATE_VA):
            let start = mach_absolute_time()
            defer { invalidations &+= 1; invalidationTicks &+= mach_absolute_time() &- start }
            _ = tr_invalidate_view_va(Int32(c.pointee.req_a.0), c.pointee.req_a.1, UInt32(c.pointee.req_a.2), c.pointee.req_a.3 != 0, c.pointee.req_a.4 != 0)
        case Int(HC_YIELD):
            let waitTicks = c.pointee.req_a.1
            if waitTicks > 5 * 24_000_000 && (waitTicks % (5 * 24_000_000)) < 500_000 {
                let owner = Int(c.pointee.req_a.0)
                let site = c.pointee.req_a.2
                let ownerCtx = (priv + Int(kOffCtx + UInt64(owner) * kCtxSlot)).assumingMemoryBound(to: EngineCtx.self).pointee
                let msg = "[hvdbt-monlock-stall] vCPU \(cpuIndex) waited \(waitTicks / 24_000_000)s for owner vCPU \(owner) lock_site=\(site) owner.elr=\(hex(ownerCtx.elr)) owner.esr=\(hex(ownerCtx.esr)) owner.far=\(hex(ownerCtx.far)) owner.cr3=\(hex(ownerCtx.cr3)) owner.cpl=\(ownerCtx.cpl)\n"
                _ = msg.withCString { fputs($0, stderr) }
            }
            sched_yield()
        case Int(HC_PARK):
            gate.leave()
            gate.enter()
        case Int(HC_CALL_TAIL):
            if !tr_needs_flush() { tr_lookup_tail(c.pointee.req_a.0, c.pointee.req_a.1) }
        case Int(HC_VIEW_FORGET):
            tr_view_forget(Int32(c.pointee.req_a.0))
        case Int(HC_VIEW_DROP):
            tr_view_drop(Int32(c.pointee.req_a.0))
        case Int(HC_SYSCALL):
            guard let machine else { throw EngineError.stop("syscall with no machine attached") }
            try machine.syscall(c, engine: self)
            if let code = exitCode { return .exited(code: code) }
        case Int(HC_MMIO):
            guard let machine else { throw EngineError.stop("device access with no machine attached") }
            tally?.mmio(c.pointee.req_a.0)
            if c.pointee.req_a.4 != 0 {
                tally?.mmioSample(rip: c.pointee.req_a.4, gpa: c.pointee.req_a.0, size: c.pointee.req_a.1, write: c.pointee.req_a.2 != 0)
            }
            if c.pointee.req_a.2 != 0 {
                machine.mmioWrite(address: c.pointee.req_a.0, size: Int(c.pointee.req_a.1), value: c.pointee.req_a.3)
            } else {
                c.pointee.req_r.0 = machine.mmioRead(address: c.pointee.req_a.0, size: Int(c.pointee.req_a.1))
            }
        case Int(HC_PORT):
            guard let machine else { throw EngineError.stop("port access with no machine attached") }
            tally?.port(UInt16(truncatingIfNeeded: c.pointee.req_a.0))
            portCount += 1
            if c.pointee.req_a.2 != 0 {
                machine.portWrite(
                    port: UInt16(truncatingIfNeeded: c.pointee.req_a.0), size: Int(c.pointee.req_a.1),
                    value: UInt32(truncatingIfNeeded: c.pointee.req_a.3))
            } else {
                c.pointee.req_r.0 = UInt64(
                    machine.portRead(port: UInt16(truncatingIfNeeded: c.pointee.req_a.0), size: Int(c.pointee.req_a.1)))
            }
            if requestStop { return .exited(code: exitCode ?? 0) }
        case Int(HC_RDMSR):
            let v = machine?.readMSR(UInt32(truncatingIfNeeded: c.pointee.req_a.0))
            c.pointee.req_r.0 = v ?? 0
            c.pointee.req_r.1 = v == nil ? 1 : 0
        case Int(HC_WRMSR):
            let ok =
                machine?.writeMSR(UInt32(truncatingIfNeeded: c.pointee.req_a.0), value: c.pointee.req_a.1) ?? false
            c.pointee.req_r.1 = ok ? 0 : 1
        case Int(HC_CPUID):
            let v =
                machine?.cpuid(
                    leaf: UInt32(truncatingIfNeeded: c.pointee.req_a.0),
                    subleaf: UInt32(truncatingIfNeeded: c.pointee.req_a.1)) ?? (0, 0, 0, 0)
            c.pointee.req_r = (UInt64(v.0), UInt64(v.1), UInt64(v.2), UInt64(v.3))
        case Int(HC_HLT):
            guard let machine else { throw EngineError.stop("hlt with no machine attached") }
            // The APIC deadline arrives in guest counter ticks. It stays in the virtual counter's terms, which track the host's.
            var deadlineNanos: UInt64 = 0
            if c.pointee.req_a.1 != 0 {
                let now = mach_absolute_time() &- vtimerOffset
                deadlineNanos = c.pointee.req_a.1 > now ? (c.pointee.req_a.1 - now) * 125 / 3 : 1
            }
            // Asleep it holds nothing in the code cache: it resumes through translate_here. A code cache reset need not wait for this vCPU's next interrupt.
            let asleep = severalVCPUs
            if asleep { gate.leave() }
            let wake = machine.waitForInterrupt(deadlineNanos: deadlineNanos)
            if asleep { gate.enter() }
            // With IF clear, the monitor can keep sleeping in the HLT host-call
            // loop without reaching HC_BOUNDARY. HLT has already completed here.
            if !options.hybrid && machine.shouldStopRun { return .halted(rax: c.pointee.x.0) }
            if let vector = machine.acknowledgeInterrupt() {
                c.pointee.req_r.0 = UInt64(vector)
            } else {
                c.pointee.req_r.0 = ~0
            }
            c.pointee.req_r.1 = wake || deadlineNanos != 0 ? 0 : 1
        case Int(HC_APIC_EOI):
            machine?.apicEOI(vector: UInt8(truncatingIfNeeded: c.pointee.req_a.0))
        case Int(HC_APIC_ICR):
            machine?.apicICR(c.pointee.req_a.0)
        case Int(HC_APIC_TPR):
            machine?.apicTPR(UInt8(truncatingIfNeeded: c.pointee.req_a.0))
        case Int(HC_VMPORT):
            guard let machine else { throw EngineError.stop("vmport access with no machine attached") }
            let port = UInt16(truncatingIfNeeded: c.pointee.req_a.0)
            let size = Int(c.pointee.req_a.1)
            let isWrite = c.pointee.req_a.2 != 0
            tally?.port(port)
            portCount += 1
            var regs: [UInt64] = [
                c.pointee.x.0, c.pointee.x.1, c.pointee.x.2, c.pointee.x.3,
                c.pointee.x.5, c.pointee.x.6, c.pointee.x.7,
            ]
            if machine.vmport(regs: &regs, port: port, size: size, isWrite: isWrite) {
                c.pointee.x.0 = regs[0]
                c.pointee.x.1 = regs[1]
                c.pointee.x.2 = regs[2]
                c.pointee.x.3 = regs[3]
                c.pointee.x.5 = regs[4]
                c.pointee.x.6 = regs[5]
                c.pointee.x.7 = regs[6]
                c.pointee.req_r.0 = regs[0]
                c.pointee.req_r.1 = 1
            } else {
                c.pointee.req_r.1 = 0
            }
        case Int(HC_BOUNDARY):
            if !options.hybrid && severalVCPUs && machine?.shouldStopRun != true { // a cache reset keeps this vCPU in the monitor; an explicit machine stop does not
                gate.leave()
                gate.enter()
                return nil
            }
            boundaryRIP = c.pointee.req_a.0
            tally?.boundary(rip: boundaryRIP, cpl: c.pointee.cpl, cr3: c.pointee.cr3)
            return .boundary(rip: boundaryRIP)
        case Int(HC_NOTE):
            tally?.processNote()
            let noteKind = c.pointee.req_a.0
            if (noteKind == 3 || noteKind == 4) && ProcessInfo.processInfo.environment["LC_HVDBT_LOG_R3_EXC"] != nil {
                if r3ExcLogged < 16 {
                    r3ExcLogged += 1
                    let rip = c.pointee.req_a.1
                    let b = bytes(at: rip, count: 16)
                    fputs("[hvdbt-r3-exc] cpu=\(cpuIndex) kind=\(noteKind) fex=\(c.pointee.req_a.4) rip=0x\(String(rip, radix: 16)) a2=0x\(String(c.pointee.req_a.2, radix: 16)) err=0x\(String(c.pointee.req_a.3, radix: 16)) cr3=0x\(String(c.pointee.cr3, radix: 16)) bytes=\(b)\n", stderr)
                }
            }
        case Int(HC_FALLBACK):
            boundaryRIP = c.pointee.req_a.1
            fallbackCount += 1
            if let tally {
                tally.fallback(
                    reason: fallbackReason(), rip: boundaryRIP, bytes: bytes(at: boundaryRIP, count: 4),
                    detail: c.pointee.req_a.3)
            }
            return .fallback(rip: boundaryRIP)
        case Int(HC_IRQ):
            irqLine = false
            if vtimerMasked {
                vtimerMasked = false
                hv_vcpu_set_vtimer_mask(vcpu, false)
            }
            if options.hybrid {
                // QEMU's APIC keeps a vector until it is taken, so take one only when it can go in at once. The monitor comes back here when IF opens.
                c.pointee.req_r.0 = ~0
                // When req_a.0 == 0, mon_irq only needed a host exit to drop Hypervisor.framework's level-asserted virtual IRQ line;
                // HV_EXIT_REASON_CANCELED already refreshed c->irq_line without extra BQL contention.
                if c.pointee.req_a.0 != 0 {
                    // Lower the line before asking, so a device's kick in between either sets it again or is seen by the question.
                    engine_ctx_set_irq_line(c, false)
                    let line = machine?.hasInterruptWaiting() ?? false
                    if line { engine_ctx_set_irq_line(c, true) }
                    if line && c.pointee.irq_pending == 0 && c.pointee.rflags_sys & 0x200 != 0 {
                        if let vector = machine?.acknowledgeInterrupt() { c.pointee.req_r.0 = UInt64(vector) }
                        engine_ctx_set_irq_line(c, false)
                        if machine?.hasInterruptWaiting() ?? false { engine_ctx_set_irq_line(c, true) }
                    }
                }
            } else if let vector = machine?.acknowledgeInterrupt() {
                c.pointee.req_r.0 = UInt64(vector)
            } else {
                c.pointee.req_r.0 = ~0
            }
        case Int(HC_STOP):
            return try stopReport()
        case Int(HC_CODE_FETCH_RETRY):
            let a = c.pointee.req_a
            let message = "hvdbt: vCPU \(cpuIndex) re-fetching translated code after a stale Unknown instruction; arm-pc \(hex(a.0)) x86-rip \(hex(a.1)) retry \(a.2)\n"
            FileHandle.standardError.write(Data(message.utf8))
        default:
            throw EngineError.stop("unknown host call \(code)")
        }
        return nil
    }

    private func fallbackReason() -> String {
        let a = ctx.pointee.req_a
        switch Int(a.0) {
        case Int(STOP_UNTRANSLATED):
            return String(cString: tr_stop_reason(UInt32(truncatingIfNeeded: a.2)))
        case Int(STOP_UNSUPPORTED):
            let names: [UInt64: String] = [
                3: "split locked access", 4: "mmio form: pair, atomic or SIMD to a device", 5: "split store on a page table",
                7: "16 or 32-bit iret", 8: "entering compatibility mode with a segment only TCG runs", 9: "TF set",
                10: "CR0 write: paging, protection or WP off", 11: "CR4 write: unsupported bit", 13: "segment load only TCG runs in compatibility mode",
                14: "unmasked SSE exceptions", 15: "split access across a device page", 18: "far transfer through a gate, to 16-bit code or with a 16-bit operand", 19: "stack or frame outside RAM", 16: "x87 with an exception unmasked or pending", 17: "x87 area on a device",
            ]
            return names[a.2] ?? "unsupported case \(a.2)"
        default:
            switch Int(a.2) {
            case Int(FB_UD): return "unknown opcode: #UD or a feature only TCG has"
            case Int(FB_HLT): return "hlt"
            case Int(FB_FX): return "fxsave, fxrstor: a device, a tracked frame, an unmasked MXCSR or no OSFXSR"
            case Int(FB_X87): return "x87 with an exception unmasked or pending"
            case Int(FB_CR8): return "mov cr8: the TPR is in QEMU's APIC"
            case Int(FB_RDMSR): return "rdmsr \(hex(a.3))"
            case Int(FB_WRMSR): return "wrmsr \(hex(a.3))"
            case Int(FB_FETCH_NOT_RAM): return "fetch from a page outside RAM"
            case Int(FB_CLTS): return "clts, on TCG by request"
            case Int(FB_VMPORT): return "vmport: its model uses the CPU's registers"
            default: return "fallback \(a.2)"
            }
        }
    }

    private func stopReport() throws -> RunOutcome? {
        let a = ctx.pointee.req_a
        let rip = a.1
        switch Int(a.0) {
        case Int(STOP_GUEST_HALT):
            return .halted(rax: a.2)
        case Int(STOP_UNTRANSLATED):
            let why = String(cString: tr_stop_reason(UInt32(truncatingIfNeeded: a.2)))
            throw EngineError.stop("\(why) at x86 rip \(hex(rip)): \(bytes(at: rip))")
        case Int(STOP_MONITOR_FAULT):
            // In any dump that follows, the registers are the monitor's at the fault, and the rip is the last boundary or fallback.
            let instruction = a.5 & (1 << 32) != 0 ?
                String(format: " arm-word 0x%08x", UInt32(truncatingIfNeeded: a.5)) : ""
            // The translated code's ARM registers, as saved on the faulting entry: where a bad branch came from (x30), the
            // lookup's result and probe (x22, x16, x17, x20, x23) and the shadow return stack (x26).
            let x = withUnsafeBytes(of: ctx.pointee.x) { Array($0.bindMemory(to: UInt64.self)) }
            let arm = [16, 17, 20, 22, 23, 26, 30].map { "x\($0) \(hex(x[$0]))" }.joined(separator: " ")
            throw EngineError.stop("monitor fault: esr \(hex(a.2)) far \(hex(a.3)) elr \(hex(a.4))\(instruction); arm \(arm), the registers below are the monitor's, not the guest's")
        case Int(STOP_UNSUPPORTED):
            throw EngineError.stop("unsupported case \(a.2) at x86 rip \(hex(rip)): \(hex(a.3)) \(hex(a.4))")
        case Int(STOP_TRIPLE_FAULT):
            throw EngineError.stop("triple fault at x86 rip \(hex(rip)): vector \(a.2) error \(hex(a.3)) \(hex(a.4))")
        case Int(STOP_GUEST_FAULT_NO_IDT):
            throw EngineError.stop(
                "x86 exception \(a.2) error \(hex(a.3)) at rip \(hex(rip)) address \(hex(a.4)): \(bytes(at: rip))")
        case Int(STOP_SHADOW_FULL):
            throw EngineError.stop("shadow pool exhausted (\(a.2))")
        default:
            throw EngineError.stop("stop reason \(a.0)")
        }
    }

    func hex(_ v: UInt64) -> String { "0x" + String(v, radix: 16) }

    // Opt-in Win64 diagnosis. Identify the image from the current thread's TEB,
    // and cache the classification only as long as that PEB parameter block is current.
    private func useNativeWindowsImage(at rip: UInt64) -> Bool {
        guard ctx.pointee.cpl == 3, ctx.pointee.compat == 0 else { return false }
        let cr3 = ctx.pointee.cr3
        func userPointer(_ address: UInt64) -> Bool { address >= 0x10000 && address < 0x0000_8000_0000_0000 }
        func pointer(_ address: UInt64) -> UInt64? {
            readGuest(address, count: 8)?.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        }
        let teb = withUnsafeBytes(of: ctx.pointee.x) { $0.bindMemory(to: UInt64.self)[27] }
        guard userPointer(teb), let peb = pointer(teb + 0x60), userPointer(peb),
              let parameters = pointer(peb + 0x20), userPointer(parameters)
        else { return false }
        if let cached = windowsProcessImages[cr3], cached.parameters == parameters {
            return cached.isNative
        }
        // RTL_USER_PROCESS_PARAMETERS.ImagePathName is a Win64 UNICODE_STRING.
        guard let descriptor = readGuest(parameters + 0x60, count: 16) else { return false }
        let count = Int(descriptor.withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) })
        let address = descriptor.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self) }
        guard count > 0, count <= 2048, count % 2 == 0, userPointer(address),
              let data = readGuest(address, count: count),
              let path = String(bytes: data, encoding: .utf16LittleEndian)
        else { return false }
        let image = path.split(separator: "\\").last?.lowercased() ?? ""
        let isNative = nativeWindowsImages.contains(image)
        if windowsProcessImages.count >= 512 { windowsProcessImages.removeAll(keepingCapacity: true) }
        windowsProcessImages[cr3] = (parameters, isNative)
        if isNative {
            let message = "hvdbt: vCPU \(cpuIndex) selecting native JIT for \(image) cr3 \(hex(cr3)) at x86-rip \(hex(rip))\n"
            _ = message.withCString { fputs($0, stderr) }
        }
        return isNative
    }

    /// Guest bytes at a VA for error messages, through the current CR3.
    public func bytes(at va: UInt64, count: Int = 12, cr3: UInt64? = nil) -> String {
        var out: [String] = []
        let cr3 = cr3 ?? ctx.pointee.cr3  // once: from another thread it changes mid-read, and a user page's bytes stop at the first switch
        for i in 0..<count {
            guard let gpa = translate(va + UInt64(i), cr3: cr3), let at = host(gpa) else { break }
            out.append(String(format: "%02x", at.load(as: UInt8.self)))
        }
        return out.joined(separator: " ")
    }

    /// The host address of a RAM GPA.
    public func host(_ gpa: UInt64) -> UnsafeMutableRawPointer? {
        guard gpa < ramTop, (priv + Int(kOffPageInfo)).load(fromByteOffset: Int(gpa >> 12), as: UInt8.self) & UInt8(PI_RAM) != 0 else { return nil }
        return ram + Int(ramTop > options.ramBytes && gpa >= 1 << 32 ? gpa - (1 << 32) + options.ramBelow4G : gpa)
    }

    public func mapDirect(gpa: UInt64, host: UnsafeMutableRawPointer, bytes: UInt64) -> Bool {
        // A frame must fit within page_info (covers [0, PRIV_IPA) = 60 GB < 64 GB table), and must not be RAM already.
        guard gpa <= kPrivIPA, bytes <= kPrivIPA - gpa else { return false }
        let pageInfo = (priv + Int(kOffPageInfo)).assumingMemoryBound(to: UInt8.self)
        for frame in stride(from: gpa >> 12, to: (gpa + bytes) >> 12, by: 1) where pageInfo[Int(frame)] & UInt8(PI_RAM) != 0 { return false }
        let rwx = hv_memory_flags_t(HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC)
        guard hv_vm_map(host, gpa, Int(bytes), rwx) == HV_SUCCESS else { return false }
        for frame in stride(from: gpa >> 12, to: (gpa + bytes) >> 12, by: 1) { pageInfo[Int(frame)] |= UInt8(PI_DIRECT) }
        return true
    }

    public func forgetAll() {
        translatorLock.lock()
        tr_flush_parked(true)
        resetHyperVSharedState()
        gate.condition.lock()
        for engine in gate.engines {
            engine.ctx.pointee.tr_flushed = 1
            engine.ctx.pointee.direct_flush = 1 // the next run drops its shadows too
        }
        gate.condition.unlock()
        translatorLock.unlock()
    }

    public func unmapDirect(gpa: UInt64, bytes: UInt64) {
        let pageInfo = (priv + Int(kOffPageInfo)).assumingMemoryBound(to: UInt8.self)
        let end = min(gpa &+ bytes, kPrivIPA)
        if gpa < end {
            for frame in stride(from: gpa >> 12, to: end >> 12, by: 1) { pageInfo[Int(frame)] &= ~UInt8(PI_DIRECT) }
        }
        gate.condition.lock()
        let engines = gate.engines
        for engine in engines { engine.ctx.pointee.direct_flush = 1 }
        gate.condition.unlock()
        for engine in engines { engine.interruptRequested() }
        hv_vm_unmap(gpa, Int(bytes))
    }

    /// Before the first run: a device answers for this range, whatever RAM lies behind it.
    public func markDevice(gpa: UInt64, bytes: UInt64) {
        let pageInfo = (priv + Int(kOffPageInfo)).assumingMemoryBound(to: UInt8.self)
        var at = gpa & ~0xFFF
        while at < min(gpa + bytes, ramTop) {
            pageInfo[Int(at >> 12)] &= ~UInt8(PI_RAM)
            at += 4096
        }
    }

    /// Any thread: guest RAM was written other than by this engine. See `engine_note_write`.
    public func noteWrite(gpa: UInt64, bytes: UInt64) {
        guard engine_ctx_note_write(firstContext, (priv + Int(kOffPageInfo)).assumingMemoryBound(to: UInt8.self), gpa, bytes) else { return }
        if severalVCPUs {
            gate.condition.lock()
            let engines = gate.engines
            gate.condition.unlock()
            for engine in engines { engine.interruptRequested() }
        } else {
            interruptRequested()
        }
    }

    /// Software walk with no permission checks.
    public func translate(_ va: UInt64, cr3: UInt64? = nil) -> UInt64? {
        var table = (cr3 ?? ctx.pointee.cr3) & 0x000F_FFFF_FFFF_F000
        var level = 3
        while level >= 0 {
            let index = (va >> UInt64(12 + 9 * level)) & 0x1FF
            guard let at = host(table + index * 8) else { return nil }
            let entry = at.load(as: UInt64.self)
            if entry & 1 == 0 { return nil }
            if level == 0 || (level < 3 && entry & 0x80 != 0) {
                let size = UInt64(1) << UInt64(12 + 9 * level)
                return (entry & 0x000F_FFFF_FFFF_F000 & ~(size - 1)) | (va & (size - 1))
            }
            table = entry & 0x000F_FFFF_FFFF_F000
            level -= 1
        }
        return nil
    }

    /// Reads guest virtual memory, for shims.
    public func readGuest(_ va: UInt64, count: Int) -> [UInt8]? {
        var out = [UInt8](repeating: 0, count: count)
        var done = 0
        while done < count {
            guard let gpa = translate(va + UInt64(done)) else { return nil }
            let chunk = min(4096 - Int((va + UInt64(done)) & 0xFFF), count - done)
            out.withUnsafeMutableBytes { dest in
                dest.baseAddress!.advanced(by: done).copyMemory(from: host(gpa) ?? ram, byteCount: chunk)
            }
            done += chunk
        }
        return out
    }

    /// Writes guest virtual memory, for shims. The caller must keep away from frames holding code or page tables.
    public func writeGuest(_ va: UInt64, bytes: [UInt8]) -> Bool {
        var done = 0
        while done < bytes.count {
            guard let gpa = translate(va + UInt64(done)) else { return false }
            let chunk = min(4096 - Int((va + UInt64(done)) & 0xFFF), bytes.count - done)
            bytes.withUnsafeBytes { src in
                (host(gpa) ?? ram).copyMemory(from: src.baseAddress!.advanced(by: done), byteCount: chunk)
            }
            done += chunk
        }
        return true
    }

    /// Any thread: forces a VM exit, after which the run loop asks the machine about pending interrupts.
    public func interruptRequested() {
        var v = vcpu
        hv_vcpus_exit(&v, 1)
    }

    /// Any thread: `run` returns `.boundary` at the next x86 instruction boundary.
    public func requestBoundary() {
        engine_ctx_set_stop(ctx)
        interruptRequested()
    }

    /// Current vCPU thread inside a host call: mark stop_requested without latching a self-cancel via hv_vcpus_exit.
    public func requestBoundaryLocal() {
        engine_ctx_set_stop(ctx)
    }

    /// Any thread: a vector for the virtual APIC's IRR, as an IO-APIC or MSI would deliver.
    public func postVector(_ vector: UInt8, level: Bool) {
        engine_ctx_post_vector(ctx, vector, level)
        interruptRequested()
    }

    /// Continues after `.boundary`. The rip in `req_r.0` is where execution resumes.
    public func resume() throws -> RunOutcome { try runLoop() }

    /// While stopped. See `engine_set_tsc`. The monitor's counter runs behind the host's by the VM's virtual timer offset.
    public func setTSC(nsOffset: UInt64, tscOffset: UInt64, num: UInt64, den: UInt64) {
        ctx.pointee.tsc_counter_offset = vtimerOffset
        ctx.pointee.tsc_ns_offset = nsOffset
        ctx.pointee.tsc_offset = tscOffset
        ctx.pointee.tsc_num = num
        ctx.pointee.tsc_den = den
    }

    func progress() -> UInt64 {
        let c = ctx.pointee
        return tr_stats().blocks &+ fallbackCount &+ portCount &+ c.stat_fills &+ c.stat_dirty &+ c.stat_smc &+ c.stat_pf_delivered
            &+ c.stat_pt_writes &+ c.stat_iret &+ c.stat_syscall &+ c.stat_mmio &+ c.stat_irq &+ c.stat_cr3
    }

    func debugText() -> String {
        let c = ctx.pointee
        let ns = ((mach_absolute_time() &- vtimerOffset) &+ c.tsc_counter_offset) &* 125 / 3 &+ c.tsc_ns_offset
        let tsc = ns &+ ns &* c.tsc_num / c.tsc_den &+ c.tsc_offset
        var o = "in hv_vcpu_run: \(inVCPURun), last host call \(lastHostCall), exits \(exitCount), progress \(progress())\n"
        o += "at the last kick: pc \(hex(pcAtKick)) lr \(hex(lrAtKick))\n"
        o += "monitor: elr \(hex(c.elr)) esr \(hex(c.esr)) far \(hex(c.far)) spsr \(hex(c.spsr)) cpl \(c.cpl) view \(c.view) cr0 \(hex(c.cr0)) cr3 \(hex(c.cr3)) cr4 \(hex(c.cr4))\n"
        o += "interrupts: rflags_sys \(hex(c.rflags_sys)) irq_line \(c.irq_line) irq_pending \(c.irq_pending) stop_requested \(c.stop_requested) stepping \(c.stepping) steps \(c.stat_steps) irqs \(c.stat_irq)\n"
        o += "engine TSC by its formula: \(hex(tsc))\n"
        // When the vCPU will not stop, this says what it last entered the monitor from. The monitor's own loops do not move elr.
        let codeBase = kPrivVA + UInt64(kOffCC)
        if c.elr >= codeBase + 32, c.elr < codeBase + UInt64(kCCSize) {
            let words = (priv + Int(kOffCC) + Int(c.elr - codeBase) - 32).assumingMemoryBound(to: UInt32.self)
            o += "code around elr, from elr-32: " + (0..<16).map { String(format: "%08x", words[$0]) }.joined(separator: " ") + "\n"
        }
        let x = withUnsafeBytes(of: c.x) { Array($0.bindMemory(to: UInt64.self)) }
        o += "registers at the last monitor entry, x0-x15 are rax rcx rdx rbx rsp rbp rsi rdi r8-r15:\n"
        for i in 0..<16 { o += hex(x[i]) + (i % 4 == 3 ? "\n" : " ") }
        return o
    }

    func mxcsr() -> UInt32 {
        var fpsr: UInt64 = 0
        hv_vcpu_get_reg(vcpu, HV_REG_FPSR, &fpsr)
        var v = ctx.pointee.mxcsr
        if fpsr & 1 != 0 { v |= 1 }
        if fpsr & 2 != 0 { v |= 4 }
        if fpsr & 4 != 0 { v |= 8 }
        if fpsr & 8 != 0 { v |= 16 }
        if fpsr & 16 != 0 { v |= 32 }
        if fpsr & 128 != 0 { v |= 2 }
        return v
    }

    func readXMM(into s: inout EngineX86State) {
        withUnsafeMutableBytes(of: &s.xmm) { out in
            for i in 0..<16 {
                var value = hv_simd_fp_uchar16_t()
                hv_vcpu_get_simd_fp_reg(vcpu, hv_simd_fp_reg_t(rawValue: HV_SIMD_FP_REG_Q0.rawValue + UInt32(i)), &value)
                withUnsafeBytes(of: value) { (out.baseAddress! + i * 16).copyMemory(from: $0.baseAddress!, byteCount: 16) }
            }
        }
    }

    func writeXMM(from s: EngineX86State) {
        var copy = s
        withUnsafeBytes(of: &copy.xmm) { input in
            for i in 0..<16 {
                let value = input.loadUnaligned(fromByteOffset: i * 16, as: hv_simd_fp_uchar16_t.self)
                hv_vcpu_set_simd_fp_reg(vcpu, hv_simd_fp_reg_t(rawValue: HV_SIMD_FP_REG_Q0.rawValue + UInt32(i)), value)
            }
        }
    }
}

func hvCheck(_ what: String, _ call: () -> hv_return_t) throws {
    let r = call()
    if r != HV_SUCCESS { throw EngineError.hypervisor(what, r) }
}
