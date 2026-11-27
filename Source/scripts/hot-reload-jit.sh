#!/bin/zsh
set -euo pipefail
REPO="${0:A:h:h:h}"
CENGINE="$REPO/Source/HybridHypervisor/Sources/CEngine"
ZYDIS_OBJ=""
for candidate in \
  "$REPO/Source/HybridHypervisor/.build/arm64-apple-macosx/release/CEngine.build/third_party/zydis/Zydis.c.o" \
  "$REPO/Source/Build/HybridEngine/Package/.build/arm64-apple-macosx/release/CEngine.build/third_party/zydis/Zydis.c.o"; do
  if [[ -f "$candidate" ]]; then
    ZYDIS_OBJ="$candidate"
    break
  fi
done
OUT_DIR=/tmp/hvdbt-jit-plugins
mkdir -p "$OUT_DIR"

if [[ -z "$ZYDIS_OBJ" ]]; then
  echo "Missing Zydis.c.o — run ./Source/HybridHypervisor/build.sh release --library-only or build-hybrid-engine first" >&2
  exit 1
fi

EPOCH=$(date +%s)
OUT_DYLIB="$OUT_DIR/libhvdbt_jit_${EPOCH}.dylib"
SDK=$(xcrun --sdk macosx --show-sdk-path)

clang -O3 -std=c11 -fPIC -march=armv8.5-a+crc -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -I"$CENGINE/include" -I"$CENGINE/third_party/zydis" -I"$CENGINE/third_party/box64" -I"$CENGINE" \
  -c "$CENGINE/x86_decode.c" -o "$OUT_DIR/x86_decode.o" &
clang -O3 -std=c11 -fPIC -march=armv8.5-a+crc -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -I"$CENGINE/include" -I"$CENGINE/third_party/zydis" -I"$CENGINE/third_party/box64" -I"$CENGINE" \
  -c "$CENGINE/emit_int.c" -o "$OUT_DIR/emit_int.o" &
clang -O3 -std=c11 -fPIC -march=armv8.5-a+crc -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -I"$CENGINE/include" -I"$CENGINE/third_party/zydis" -I"$CENGINE/third_party/box64" -I"$CENGINE" \
  -c "$CENGINE/emit_sse.c" -o "$OUT_DIR/emit_sse.o" &
clang -O3 -std=c11 -fPIC -march=armv8.5-a+crc -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -I"$CENGINE/include" -I"$CENGINE/third_party/zydis" -I"$CENGINE/third_party/box64" -I"$CENGINE" \
  -c "$CENGINE/emit_x87.c" -o "$OUT_DIR/emit_x87.o" &
clang -O3 -std=c11 -fPIC -march=armv8.5-a+crc -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -I"$CENGINE/include" -I"$CENGINE/third_party/zydis" -I"$CENGINE/third_party/box64" -I"$CENGINE" \
  -c "$CENGINE/emit_ext.c" -o "$OUT_DIR/emit_ext.o" &
wait

clang -dynamiclib -target arm64-apple-macosx11.0 -isysroot "$SDK" \
  -undefined dynamic_lookup \
  "$OUT_DIR/x86_decode.o" "$OUT_DIR/emit_int.o" "$OUT_DIR/emit_sse.o" \
  "$OUT_DIR/emit_x87.o" "$OUT_DIR/emit_ext.o" "$ZYDIS_OBJ" \
  -o "$OUT_DYLIB"
codesign -s - -f "$OUT_DYLIB" >/dev/null 2>&1 || true

if [[ "${1:-}" != "--no-trigger" ]]; then
  print -r -- "$OUT_DYLIB" > /tmp/hvdbt-reload.path
  echo "Triggered live hot-reload: $OUT_DYLIB"
else
  echo "Built plugin (no trigger): $OUT_DYLIB"
fi
