// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Engine",
    platforms: [.macOS(.v11)],
    products: [
        .library(name: "Engine", type: .static, targets: ["Engine"]),
        .executable(name: "engine-run", targets: ["engine-run"]),
    ],
    targets: [
        .target(name: "CEngine",
            cSettings: [
                .headerSearchPath("third_party/zydis"),
                .headerSearchPath("third_party/box64"),
                .unsafeFlags([
                    "-march=armv8.5-a+crc", "-target", "arm64-apple-macosx11.0"
                ]),
            ],
            cxxSettings: [
                .headerSearchPath("third_party/fex"),
                .unsafeFlags([
                    "-march=armv8.5-a+crc", "-target", "arm64-apple-macosx11.0"
                ]),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
                .unsafeFlags([
                    "/Users/admin/.gemini/antigravity/scratch/fexcore-macos-build/libhvdbt_fex_bridge.a",
                    "/opt/homebrew/opt/fmt/lib/libfmt.a",
                    "/opt/homebrew/opt/xxhash/lib/libxxhash.a"
                ])
            ]
        ),
        .target(
            name: "Engine", dependencies: ["CEngine"],
            linkerSettings: [.linkedFramework("Hypervisor")]),
        .executableTarget(name: "engine-run", dependencies: ["Engine", "CEngine"]),
    ],
    cxxLanguageStandard: .cxx20
)
