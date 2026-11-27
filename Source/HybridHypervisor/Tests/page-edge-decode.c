// Decoder-only check: a VEX opcode that starts in the last bytes of a page must
// not become a cached one-byte #UD when the next page is missing or not final.
#include <stdio.h>
#include <string.h>
#include "tr_internal.h"

bool tr_compat = false;
bool tr_precise_flags = false;
int tr_cpl = 3;

static u8 image[32];
static int image_n;
static int allow_cross;

int tr_fetch(u64 va, u8 *out, int n) {
    u64 off = va & 0xfff;
    int have = image_n;
    if (!allow_cross && (int)off < 0x1000)
        have = have < (int)(0x1000 - off) ? have : (int)(0x1000 - off);
    if (have < 0)
        have = 0;
    if (n > have)
        n = have;
    memcpy(out, image, (size_t)n);
    return n;
}

static int fail;

static void expect(const char *name, u64 rip, const u8 *bytes, int n, int cross, int want_len, int want_fault) {
    memset(image, 0, sizeof image);
    memcpy(image, bytes, (size_t)n);
    image_n = n;
    allow_cross = cross;
    Insn in;
    x86_decode(rip, &in);
    int fault = in.op == OP_BAD && in.why && !strcmp(in.why, "FETCH_FAULT");
    int ok = want_fault ? fault : want_len ? in.op == OP_EXT : in.op == OP_UD;
    if (!ok) {
        printf("FAIL %s op=%d why=%s next=%llx\n", name, in.op, in.why ? in.why : "-",
               (unsigned long long)in.next);
        fail++;
    } else if (want_len && in.next != rip + (u64)want_len) {
        printf("FAIL %s length got %llu want %d\n", name, (unsigned long long)(in.next - rip), want_len);
        fail++;
    } else {
        printf("PASS %s\n", name);
    }
}

int main(void) {
    const u8 fmadd[] = {0xc4, 0x42, 0x61, 0xb9, 0xe6};
    const u8 movss[] = {0xc5, 0xfa, 0x10, 0xac, 0x24, 0xc0, 0x02, 0x00, 0x00};
    const u8 movd[] = {0xc4, 0xc1, 0x79, 0x6e, 0xd1};
    const u8 pack[] = {0xc5, 0xcd, 0x6b, 0xcf};
    const u8 shlx[] = {0xc4, 0xe2, 0x41, 0xf7, 0xeb};
    const u8 zeros[] = {0xc4, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    u64 edge = 0x7ff6be4a0ffeULL;
    expect("fmadd full", edge, fmadd, 5, 1, 5, 0);
    expect("movss full", edge, movss, 9, 1, 9, 0);
    expect("movd full", edge, movd, 5, 1, 5, 0);
    expect("pack full", edge, pack, 4, 1, 4, 0);
    expect("shlx full", edge, shlx, 5, 1, 5, 0);
    expect("fmadd truncated", edge, fmadd, 2, 0, 0, 1);
    expect("prefix plus zeros", edge, zeros, 15, 1, 0, 1);
    expect("mid-page fmadd", 0x7ff6be4a0100ULL, fmadd, 5, 1, 5, 0);
    expect("mid-page zeros stay ud", 0x7ff6be4a0100ULL, zeros, 15, 1, 0, 0);
    if (fail) {
        printf("%d FAILED\n", fail);
        return 1;
    }
    printf("ALL PAGE-EDGE DECODE CHECKS PASSED\n");
    return 0;
}
