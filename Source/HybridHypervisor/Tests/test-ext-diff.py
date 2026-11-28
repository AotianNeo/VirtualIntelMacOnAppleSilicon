#!/usr/bin/env python3
"""
Differential ISA Test Harness: QEMU TCG Oracle (LC_HVDBT=0) vs hvdbt CEngine (LC_HVDBT=1).
Assembles a bare-metal Multiboot1 -> 64-bit high-half test kernel with NASM + ld.lld,
executes each instruction test case with deterministic initial state, dumps GPRs + RFLAGS + XMM/YMM
over port 0xE9 debugcon, exits via port 0xF4, and verifies 0 bit diffs and 0 unexpected fallbacks.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.environ.get(
    "VIM_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
QEMU = os.environ.get(
    "QEMU",
    os.path.join(ROOT, "Source", "Build", "qemu-win-hvdbt", "qemu-system-x86_64"),
)
NASM = os.environ.get("NASM") or shutil.which("nasm") or "/opt/homebrew/bin/nasm"
LLD = os.environ.get("LLD") or shutil.which("ld.lld") or "/opt/homebrew/bin/ld.lld"

BASELINE_TESTS = [
    ("nested_direct_and_indirect_calls", """
        xor eax, eax
        mov ecx, 24
        call .descend
        jmp .done
    .descend:
        test ecx, ecx
        jz .done_descending
        dec ecx
        lea r11, [rel .descend]
        test ecx, 1
        jz .direct
        call r11
        jmp .unwind
    .direct:
        call .descend
    .unwind:
        inc eax
    .done_descending:
        ret
    .done:
    """),
    ("smoke_add_adc", """
        add rax, rbx
        adc rcx, rdx
    """),
    ("smoke_aesenc", """
        aesenc xmm0, xmm1
        aesenc xmm0, xmm2
        aesenclast xmm0, xmm3
    """),
    ("stage3_xgetbv", """
        xor ecx, ecx
        xgetbv
    """),
    ("bmi1_andn_64_32", """
        andn r8, rax, rdx
        andn r9d, ebx, ecx
        andn r10, rbp, [rel scratch_buf + 16]
    """),
    ("bmi1_bextr_64_32_edge", """
        mov rcx, 0x1808
        bextr r8, rax, rcx
        mov ecx, 0x0c04
        bextr r9d, edx, ecx
        mov ecx, 0x0848 ; start > 64 -> 0
        bextr r10, [rel scratch_buf + 24], rcx
    """),
    ("bmi1_blsi_blsmsk_blsr_nonzero", """
        blsi r8, rax
        blsmsk r9, rdx
        blsr r10, rbx
        blsi r11d, [rel scratch_buf + 8]
    """),
    ("bmi1_blsi_zerosrc_cf", """
        xor eax, eax
        blsi r8, rax
    """),
    ("bmi1_blsr_zerosrc_cf", """
        xor eax, eax
        blsr r8, rax
    """),
    ("bmi2_bzhi_64_32_edge", """
        mov ecx, 29
        bzhi r8, rax, rcx
        mov ecx, 19
        bzhi r9d, edx, ecx
        mov ecx, 70 ; idx >= 64 -> CF=1
        bzhi r10, [rel scratch_buf + 32], rcx
    """),
    ("bmi2_mulx_rorx_shifts", """
        mulx r8, r9, rax
        mulx r10d, r11d, [rel scratch_buf + 16]
        rorx r12, rax, 19
        rorx r13d, ebx, 11
        mov ecx, 17
        shlx r14, rax, rcx
        shrx r15, rdx, rcx
        sarx rbx, rbx, rcx
    """),
    ("bmi2_pdep_pext_64_32", """
        pdep r8, rax, rbp
        pext r9, rdx, rbp
        pdep r10d, eax, esi
        pext r11d, edx, [rel scratch_buf + 40]
        xor ecx, ecx
        pdep r12, rax, rcx
        pext r13, rdx, rcx
    """),
    ("adx_adcx_adox_dual_chain", """
        ; Dual carry chains: CF via ADCX, OF via ADOX, preserving ZF/SF/PF/AF
        mov rax, -1
        add rax, 1      ; sets CF=1, ZF=1, OF=0
        mov r8, 0xffffffffffffffff
        mov r9, 0xffffffffffffffff
        adcx r8, rbx    ; uses CF=1 -> r8 wraps, CF=1
        adox r9, rbx    ; uses OF=0 -> r9 wraps, OF=1
        adcx r10, rdx
        adox r11, [rel scratch_buf + 48]
        adcx r12d, eax
        adox r13d, ecx
    """),
    ("sha_ni_sha1_all", """
        sha1nexte xmm0, xmm1
        sha1msg1 xmm2, xmm3
        sha1msg2 xmm4, [rel scratch_buf + 64]
        sha1rnds4 xmm5, xmm6, 0
        sha1rnds4 xmm7, xmm8, 1
        sha1rnds4 xmm9, xmm10, 2
        sha1rnds4 xmm11, [rel scratch_buf + 80], 3
    """),
    ("sha_ni_sha256_all", """
        sha256msg1 xmm1, xmm2
        sha256msg2 xmm3, [rel scratch_buf + 96]
        sha256rnds2 xmm4, xmm5, xmm0
        sha256rnds2 xmm6, [rel scratch_buf + 112], xmm0
    """),
    ("vex128_zeroes_ymm_hi", """
        vpaddd xmm0, xmm1, xmm2
        vpxor xmm3, xmm4, [rel scratch_buf + 32]
        vmovdqa xmm5, xmm6
        vmovd eax, xmm0
        vmovq xmm7, rax
        vpinsrd xmm8, xmm9, ecx, 2
        vpextrd r8d, xmm8, 2
    """),
    ("avx2_256_moves_broadcast_insert_extract", """
        vmovdqu ymm0, [rel scratch_buf + 64]
        vmovdqa ymm1, ymm2
        vpbroadcastb ymm3, xmm4
        vpbroadcastw ymm4, [rel scratch_buf + 16]
        vpbroadcastd ymm5, xmm6
        vpbroadcastq ymm6, [rel scratch_buf + 24]
        vbroadcasti128 ymm7, [rel scratch_buf + 32]
        vinserti128 ymm8, ymm0, xmm1, 1
        vinserti128 ymm9, ymm0, [rel scratch_buf + 48], 0
        vextracti128 xmm10, ymm8, 1
        vextracti128 xmm11, ymm8, 0
    """),
    ("avx2_256_int_arith_bitwise_cmp", """
        vpaddb ymm0, ymm1, ymm2
        vpaddw ymm3, ymm4, ymm5
        vpaddd ymm6, ymm7, [rel scratch_buf + 32]
        vpaddq ymm8, ymm9, ymm10
        vpsubb ymm11, ymm12, ymm13
        vpsubd ymm14, ymm15, ymm0
        vpand ymm1, ymm2, ymm3
        vpor ymm4, ymm5, ymm6
        vpxor ymm7, ymm8, ymm9
        vpandn ymm10, ymm11, ymm12
        vpcmpeqb ymm13, ymm1, ymm2
        vpcmpgtd ymm14, ymm6, ymm7
        vpminsd ymm15, ymm3, ymm4
        vpmaxud ymm0, ymm5, ymm8
        vpabsd ymm1, ymm14
        vpsignb ymm2, ymm3, ymm4
    """),
    ("avx2_256_int_mul_madd", """
        vpmullw ymm0, ymm1, ymm2
        vpmulld ymm3, ymm4, [rel scratch_buf + 64]
        vpmulhw ymm5, ymm6, ymm7
        vpmulhuw ymm8, ymm9, ymm10
        vpmulhrsw ymm11, ymm12, ymm13
        vpmuludq ymm14, ymm15, ymm0
        vpmuldq ymm1, ymm2, ymm3
        vpmaddwd ymm4, ymm5, ymm6
        vpmaddubsw ymm7, ymm8, ymm9
    """),
    ("avx2_256_shifts_imm_reg_var", """
        vpsllw ymm0, ymm1, 5
        vpsrld ymm2, ymm3, 7
        vpsrad ymm4, ymm5, 11
        vpsllq ymm6, ymm7, 19
        vpslldq ymm8, ymm9, 3
        vpsrldq ymm10, ymm11, 5
        vmovq xmm12, rcx ; rcx = 19
        vpsllq ymm13, ymm1, xmm12
        ; Variable shifts with small and >=32/64 lane counts
        vpand ymm14, ymm2, ymm3
        vpsllvd ymm15, ymm4, ymm14
        vpsrlvd ymm0, ymm5, ymm14
        vpsravd ymm1, ymm6, ymm14
        vpsllvq ymm2, ymm7, ymm14
        vpsrlvq ymm3, ymm8, ymm14
    """),
    ("avx2_256_shuffles_permutes_packs", """
        vpshufb ymm0, ymm1, ymm2
        vpshufd ymm3, ymm4, 0x1b
        vpshufhw ymm5, ymm6, 0x93
        vpshuflw ymm7, ymm8, 0x4e
        vpunpcklbw ymm9, ymm1, ymm2
        vpunpckhwd ymm10, ymm3, ymm4
        vpunpckldq ymm11, ymm5, ymm6
        vpunpckhqdq ymm12, ymm7, ymm8
        vpacksswb ymm13, ymm1, ymm2
        vpackusdw ymm14, ymm3, ymm4
        vpackssdw xmm0, xmm1, xmm2
        vpackssdw ymm5, ymm6, ymm7
        vpackssdw ymm8, ymm9, [rel scratch_buf]
        vpalignr ymm15, ymm5, ymm6, 7
        vperm2i128 ymm0, ymm1, ymm2, 0x21
        vperm2i128 ymm3, ymm4, ymm5, 0x83
        vpermq ymm6, ymm7, 0x1b
        vpermd ymm8, ymm9, ymm10
    """),
    ("avx2_256_blends_pmovsx_zx_mskb_ptest", """
        vpblendd ymm0, ymm1, ymm2, 0xa5
        vpblendw ymm3, ymm4, ymm5, 0x5a
        vpblendvb ymm6, ymm7, ymm8, ymm9
        vpmovsxbw ymm10, xmm1
        vpmovzxbd ymm11, xmm2
        vpmovsxwd ymm12, xmm3
        vpmovzxdq ymm13, xmm4
        vpmovmskb r8d, ymm0
        vpmovmskb r9d, xmm10
        vptest ymm0, ymm1
    """),
    ("avx_256_fp_and_fma3", """
        ; Prepare finite normalized FP32 / FP64 vectors by masking exponent bits
        mov rax, 0x3ff000003ff00000
        vmovq xmm0, rax
        vpbroadcastq ymm0, xmm0
        mov rax, 0x4000000040000000
        vmovq xmm1, rax
        vpbroadcastq ymm1, xmm1
        mov rax, 0x3f8000003f800000
        vmovq xmm2, rax
        vpbroadcastq ymm2, xmm2
        vaddps ymm3, ymm0, ymm1
        vmulpd ymm4, ymm0, ymm1
        vfmadd132ps ymm3, ymm1, ymm2
        vfmadd213pd ymm4, ymm0, ymm1
        vfmsub231ps ymm5, ymm3, ymm2
        vfnmadd132pd ymm6, ymm4, ymm0
        vfnmsub213ps ymm7, ymm2, ymm1
        vfmadd132ss xmm8, xmm1, xmm2
        vfmsub213sd xmm9, xmm0, xmm1
        vmovmskps r10d, ymm3
        vmovmskpd r11d, ymm4
        vzeroupper
    """),
    ("xsave_xrstor_256_ymm_roundtrip", """
        lea rdi, [rel xsave_buf]
        xor eax, eax
        mov ecx, 128
        rep stosq
        mov eax, 7
        xor edx, edx
        xsave64 [rel xsave_buf]
        vzeroall
        mov eax, 7
        xor edx, edx
        xrstor64 [rel xsave_buf]
        ; Also test Windows SwapContext rfbm=4 (YMM_Hi128 only) xsave/xrstor
        mov eax, 4
        xor edx, edx
        xsave [rel xsave_buf]
        vzeroupper
        mov eax, 4
        xor edx, edx
        xrstor [rel xsave_buf]
    """),
    ("avx2_cvt_vec", """
        mov rax, 0x00000007fffffffd
        vmovq xmm0, rax
        vpbroadcastq ymm0, xmm0
        vcvtdq2pd xmm1, xmm0
        vcvtdq2pd ymm2, xmm0
        vcvtpd2dq xmm3, xmm1
        vcvtpd2dq xmm4, ymm2
        vcvttpd2dq xmm5, xmm1
        vcvttpd2dq xmm6, ymm2
        vcvtdq2ps xmm7, xmm0
        vcvtdq2ps ymm8, ymm0
        vcvtps2dq xmm9, xmm7
        vcvtps2dq ymm10, ymm8
        vcvttps2dq xmm11, xmm7
        vcvttps2dq ymm12, ymm8
        vcvtps2pd xmm13, xmm7
        vcvtps2pd ymm14, xmm7
        vcvtpd2ps xmm15, ymm14
    """),
    ("avx_permil_round_insert_cmp_mask", """
        vpermilps xmm0, xmm1, 0x1b
        vpermilps ymm2, ymm3, 0x4e
        vpermilpd xmm4, xmm5, 0x2
        vpermilpd ymm6, ymm7, 0x9
        vpermilps xmm8, xmm9, xmm10
        vpermilps ymm11, ymm12, ymm13
        vpermilpd xmm14, xmm15, xmm1
        vpermilpd ymm0, ymm2, ymm4
        vroundss xmm1, xmm2, xmm3, 0
        vroundss xmm4, xmm5, dword [rel scratch_buf + 8], 1
        vroundsd xmm6, xmm7, xmm8, 3
        vroundsd xmm9, xmm10, qword [rel scratch_buf + 16], 4
        vinsertps xmm11, xmm12, xmm13, 0x1b
        vinsertps xmm14, xmm15, dword [rel scratch_buf + 4], 0x0d
        vextractps r8d, xmm1, 2
        vcmpss xmm0, xmm1, xmm2, 0
        vcmpss xmm3, xmm4, dword [rel scratch_buf], 1
        vcmpsd xmm5, xmm6, xmm7, 2
        vcmpps ymm8, ymm9, ymm10, 13
        vcmppd xmm11, xmm12, xmm13, 3
        vpsadbw xmm12, xmm13, xmm14
        vpsadbw ymm13, ymm14, ymm15
        vphminposuw xmm14, xmm15
        vpmaskmovd xmm15, xmm0, [rel scratch_buf]
        vpmaskmovq ymm1, ymm2, [rel scratch_buf + 32]
        vpmaskmovd [rel scratch_buf + 192], xmm3, xmm4
        vmovdqu xmm2, [rel scratch_buf + 192]
        mov dword [rel scratch_buf + 224], 0x1F80
        vldmxcsr [rel scratch_buf + 224]
        vstmxcsr [rel scratch_buf + 228]
        mov eax, [rel scratch_buf + 228]
    """),
    ("avx_f16c", """
        mov rax, 0xbc00380040003c00
        vmovq xmm1, rax
        mov rax, 0x000042003e004500
        vpinsrq xmm1, xmm1, rax, 1
        vcvtph2ps xmm0, xmm1
        vcvtph2ps ymm2, xmm1
        vcvtps2ph xmm3, xmm0, 0
        vcvtps2ph xmm4, ymm2, 4
        vcvtps2ph qword [rel scratch_buf + 256], xmm0, 0
        vmovq xmm5, qword [rel scratch_buf + 256]
    """),
    ("avx_scalar_fp", """
        mov eax, 0x3f800000
        vmovd xmm1, eax
        mov eax, 0x40000000
        vmovd xmm2, eax
        mov eax, 0x3f000000
        vmovd xmm3, eax
        mov eax, 0xc0400000
        vmovd xmm4, eax
        mov eax, 0x447a0000
        vmovd xmm5, eax
        mov rax, 0x3ff0000000000000
        vmovq xmm6, rax
        mov rax, 0x4000000000000000
        vmovq xmm7, rax
        mov rax, 0xc008000000000000
        vmovq xmm8, rax
        vaddss xmm9, xmm1, xmm2
        vsubss xmm10, xmm2, xmm3
        vmulss xmm11, xmm5, xmm2
        vdivss xmm12, xmm5, xmm3
        vminss xmm13, xmm4, xmm1
        vmaxss xmm14, xmm4, xmm1
        vaddss xmm15, xmm1, dword [rel scratch_buf]
        vaddsd xmm0, xmm6, xmm7
        vsubsd xmm1, xmm7, xmm8
        vmulsd xmm2, xmm6, xmm7
        vdivsd xmm3, xmm7, xmm6
        vminsd xmm4, xmm8, xmm6
        vmaxsd xmm5, xmm8, xmm6
        vaddsd xmm6, xmm6, qword [rel scratch_buf + 16]
        vsqrtss xmm7, xmm5, xmm2
        vsqrtsd xmm8, xmm0, xmm3
    """),
    ("avx2_gather", """
        mov dword [rel scratch_buf + 0], 0x11111111
        mov dword [rel scratch_buf + 4], 0x22222222
        mov dword [rel scratch_buf + 8], 0x33333333
        mov dword [rel scratch_buf + 12], 0x44444444
        mov dword [rel scratch_buf + 16], 0x55555555
        mov dword [rel scratch_buf + 20], 0x66666666
        mov dword [rel scratch_buf + 24], 0x77777777
        mov dword [rel scratch_buf + 28], 0x88888888
        mov eax, 0
        vmovd xmm1, eax
        mov eax, 1
        vpinsrd xmm1, xmm1, eax, 1
        mov eax, 2
        vpinsrd xmm1, xmm1, eax, 2
        mov eax, 0xffffffff
        vpinsrd xmm1, xmm1, eax, 3
        mov rax, 0x8000000000000000
        vmovq xmm2, rax
        mov rax, 0x0000000080000000
        vpinsrq xmm2, xmm2, rax, 1
        vgatherdps xmm0, [rel scratch_buf + xmm1*4], xmm2
        mov eax, 0
        vmovd xmm4, eax
        mov eax, 2
        vpinsrd xmm4, xmm4, eax, 1
        mov eax, 4
        vpinsrd xmm4, xmm4, eax, 2
        mov eax, 6
        vpinsrd xmm4, xmm4, eax, 3
        vpcmpeqd xmm5, xmm5, xmm5
        lea rax, [rel scratch_buf]
        vpgatherdd xmm3, [rax + xmm4*4 + 4], xmm5
        vpcmpeqd ymm6, ymm6, ymm6
        mov eax, 0
        vmovd xmm7, eax
        mov eax, 1
        vpinsrd xmm7, xmm7, eax, 1
        mov eax, 2
        vpinsrd xmm7, xmm7, eax, 2
        mov eax, 3
        vpinsrd xmm7, xmm7, eax, 3
        mov eax, 4
        vpinsrd xmm8, xmm7, eax, 0
        vinserti128 ymm7, ymm7, xmm8, 1
        vgatherdps ymm9, [rel scratch_buf + ymm7*4], ymm6
        mov eax, 0
        vmovd xmm10, eax
        mov eax, 1
        vpinsrd xmm10, xmm10, eax, 1
        vpcmpeqq xmm11, xmm11, xmm11
        vgatherdpd xmm12, [rel scratch_buf + xmm10*8], xmm11
    """),
    ("fma_addsub_dpps_maskmov", """
        ; lanes 1, 2, 4, 8 and 2, 2, 2, 2 and 3, 3, 3, 3
        mov eax, 0x3f800000
        vmovd xmm0, eax
        mov eax, 0x40000000
        vpinsrd xmm0, xmm0, eax, 1
        mov eax, 0x40800000
        vpinsrd xmm0, xmm0, eax, 2
        mov eax, 0x41000000
        vpinsrd xmm0, xmm0, eax, 3
        mov eax, 0x40000000
        vmovd xmm1, eax
        vpinsrd xmm1, xmm1, eax, 1
        vpinsrd xmm1, xmm1, eax, 2
        vpinsrd xmm1, xmm1, eax, 3
        mov eax, 0x40400000
        vmovd xmm2, eax
        vpinsrd xmm2, xmm2, eax, 1
        vpinsrd xmm2, xmm2, eax, 2
        vpinsrd xmm2, xmm2, eax, 3
        vmovaps xmm3, xmm0
        vfmaddsub231ps xmm3, xmm1, xmm2
        vmovaps xmm4, xmm0
        vfmsubadd231ps xmm4, xmm1, xmm2
        vmovaps xmm5, xmm0
        vfmaddsub213ps xmm5, xmm1, xmm2
        vmovaps xmm6, xmm1
        vfmaddsub132ps xmm6, xmm0, xmm2
        ; pd: 1.0, 2.0 and 3.0, 4.0
        mov rax, 0x3ff0000000000000
        vmovq xmm7, rax
        mov rax, 0x4000000000000000
        vpinsrq xmm7, xmm7, rax, 1
        mov rax, 0x4008000000000000
        vmovq xmm8, rax
        mov rax, 0x4010000000000000
        vpinsrq xmm8, xmm8, rax, 1
        vmovaps xmm9, xmm7
        vfmaddsub231pd xmm9, xmm8, xmm8
        vinsertf128 ymm0, ymm0, xmm0, 1
        vinsertf128 ymm1, ymm1, xmm1, 1
        vinsertf128 ymm2, ymm2, xmm2, 1
        vfmaddsub231ps ymm10, ymm1, ymm2
        vdpps xmm11, xmm0, xmm1, 0xFF
        vdpps xmm12, xmm0, xmm1, 0xF1
        vdpps ymm13, ymm0, ymm1, 0xFF
        vdppd xmm14, xmm7, xmm8, 0x31
        dpps xmm15, xmm0, 0xFF
        dppd xmm7, xmm8, 0x31
        mov dword [rel scratch_buf + 64], 0xffffffff
        mov dword [rel scratch_buf + 68], 0xffffffff
        mov dword [rel scratch_buf + 72], 0xffffffff
        mov dword [rel scratch_buf + 76], 0xffffffff
        mov eax, 0x80000000
        vmovd xmm8, eax
        mov eax, 0
        vpinsrd xmm8, xmm8, eax, 1
        mov eax, 0x80000000
        vpinsrd xmm8, xmm8, eax, 2
        mov eax, 0
        vpinsrd xmm8, xmm8, eax, 3
        lea rax, [rel scratch_buf + 64]
        vmaskmovps [rax], xmm8, xmm0
        vmaskmovps xmm2, xmm8, [rax]
    """),
    ("page_edge_vex", """
        ; Each target starts at page offset 0xffe, so the opcode crosses into the next page.
        align 4096
        jmp .edge_fmadd
        times 0xffe - 5 db 0x90
    .edge_fmadd:
        vfmadd231ss xmm3, xmm0, xmm1
        align 4096
        jmp .edge_movss
        times 0xffe - 5 db 0x90
    .edge_movss:
        vmovss xmm5, dword [rel scratch_buf]
        align 4096
        jmp .edge_movd
        times 0xffe - 5 db 0x90
    .edge_movd:
        vmovd xmm2, r9d
        align 4096
        jmp .edge_pack
        times 0xffe - 5 db 0x90
    .edge_pack:
        vpackssdw ymm1, ymm6, ymm7
        align 4096
        jmp .edge_shlx
        times 0xffe - 5 db 0x90
    .edge_shlx:
        shlx ebp, ebx, edi
    """),
    ("div_8_16", """
        mov eax, 0x1234
        xor edx, edx
        mov r8w, 0x10
        pushfq
        div r8w
        popfq
        mov r9d, eax
        mov r10d, edx
        mov eax, 0x00ff
        mov r8b, 7
        pushfq
        div r8b
        popfq
        mov r11d, eax
        mov eax, 0xffec
        mov edx, 0xffff
        mov r8w, 0xfffd
        pushfq
        idiv r8w
        popfq
        mov r12d, eax
        mov r13d, edx
        mov eax, 0xff9c
        mov r8b, 7
        pushfq
        idiv r8b
        popfq
        mov r14d, eax
    """),
    ("ymm_resident_chain", """
        vpcmpeqd ymm0, ymm0, ymm0
        vpand ymm1, ymm0, [rel scratch_buf]
        vperm2i128 ymm2, ymm0, ymm1, 0x20
        vpunpcklbw ymm3, ymm2, ymm1
        vpmullw ymm4, ymm3, ymm1
        vpaddw ymm5, ymm4, ymm2
        vextracti128 xmm6, ymm5, 1
    """),
    ("fma231_resident", """
        mov dword [rel scratch_buf], 0x3f800000
        vbroadcastss ymm0, [rel scratch_buf]
        vxorps ymm1, ymm1, ymm1
        vfmadd231ps ymm1, ymm0, ymm0
        vxorps ymm2, ymm2, ymm2
        vfmadd231ps ymm2, ymm2, ymm0
        vfmadd231ps ymm2, ymm0, ymm2
        vxorps ymm8, ymm8, ymm8
        vfmadd231ps ymm8, ymm0, ymm0
        vextractf128 xmm3, ymm1, 1
        vextractf128 xmm4, ymm2, 1
        vextractf128 xmm5, ymm8, 1
    """),
    ("mxcsr_round_up", """
        mov dword [rel scratch_buf], 0x5F80
        vldmxcsr [rel scratch_buf]
        mov eax, 0x3f800000
        vmovd xmm0, eax
        mov eax, 0x33800000
        vmovd xmm1, eax
        vxorps xmm2, xmm2, xmm2
        vaddss xmm2, xmm0, xmm1
        vstmxcsr [rel scratch_buf + 4]
        mov r10d, [rel scratch_buf + 4]
        mov dword [rel scratch_buf], 0x9FC0
        vldmxcsr [rel scratch_buf]
        vstmxcsr [rel scratch_buf + 4]
        mov r11d, [rel scratch_buf + 4]
        mov dword [rel scratch_buf], 0x1F80
        vldmxcsr [rel scratch_buf]
    """),
    ("fex_permq_blend_shuf_permd_mskb", """
        lea rsi, [rel scratch_buf]
        vmovdqu ymm0, [rsi]
        vmovdqu ymm8, [rsi + 32]
        vpermq ymm1, ymm0, 0x1e
        vpermq ymm2, ymm0, 0x39
        vpermq ymm3, ymm0, 0x93
        vpermq ymm9, ymm8, 0x4e
        vpblendd ymm4, ymm1, ymm2, 0xc0
        vpblendd ymm5, ymm2, ymm3, 0x0c
        vpblendd ymm6, ymm3, ymm0, 0x5a
        vpshufd ymm7, ymm4, 0x1b
        vpshufd ymm10, ymm9, 0x4e
        vpermd ymm11, ymm0, ymm5
        vpmovmskb r9d, ymm6
        vpmovmskb r10d, ymm10
        vextracti128 xmm12, ymm1, 1
        vextracti128 xmm13, ymm11, 1
    """),
    ("fex_fma3_all_orders", """
        lea rsi, [rel scratch_buf]
        vmovapd ymm0, [rsi]
        vmovapd ymm1, [rsi + 32]
        vmovapd ymm2, ymm0
        vmovapd ymm3, ymm1
        vfmsub231pd ymm2, ymm1, ymm0
        vfnmadd231pd ymm2, ymm0, ymm1
        vfmadd213pd ymm0, ymm1, ymm2
        vfnmsub213pd ymm1, ymm0, ymm2
        vfmadd132ps ymm3, ymm1, ymm2
        vextractf128 xmm4, ymm2, 1
        vextractf128 xmm5, ymm0, 1
        vextractf128 xmm6, ymm1, 1
        vextractf128 xmm7, ymm3, 1
    """),
    ("fex_vaes256_and_ldp256", """
        lea rbx, [rel scratch_buf]
        vmovdqa ymm0, [rbx]
        vmovdqa ymm1, [rbx + 32]
        vmovdqa ymm8, [rbx + 64]
        vaesenc ymm0, ymm0, ymm1
        vaesenclast ymm0, ymm0, ymm8
        vaesdec ymm1, ymm1, ymm0
        vaesdeclast ymm8, ymm8, ymm1
        vmovdqa [rbx + 96], ymm0
        vmovdqa [rbx + 128], ymm8
        vmovdqa ymm2, [rbx + 96]
        vmovdqa ymm9, [rbx + 128]
        vextracti128 xmm3, ymm2, 1
        vextracti128 xmm10, ymm9, 1
    """),
    ("blendv_bsl_alias", """
        mov dword [rel scratch_buf], 0x80000000
        mov dword [rel scratch_buf + 4], 0x3f800000
        mov dword [rel scratch_buf + 8], 0x40000000
        vbroadcastss ymm3, [rel scratch_buf]
        vbroadcastss ymm4, [rel scratch_buf + 4]
        vbroadcastss ymm14, [rel scratch_buf + 8]
        vpcmpeqd ymm5, ymm5, ymm5
        vpxor ymm6, ymm6, ymm6
        vblendvps ymm7, ymm4, ymm3, ymm5
        vblendvps ymm0, ymm4, ymm3, ymm6
        vmovaps ymm1, ymm4
        vblendvps ymm1, ymm1, ymm3, ymm5
        vmovaps ymm2, ymm3
        vblendvps ymm2, ymm4, ymm2, ymm5
        vmovaps ymm8, ymm5
        vblendvps ymm8, ymm4, ymm3, ymm8
        vblendvpd ymm9, ymm4, ymm14, ymm5
        vpblendvb ymm10, ymm4, ymm3, ymm6
        vblendvps ymm11, ymm8, ymm9, ymm10
        vextractf128 xmm12, ymm7, 1
        vextractf128 xmm13, ymm8, 1
        vextractf128 xmm15, ymm1, 1
    """),
    ("maskmov_fast_paths", """
        lea rbx, [rel scratch_buf]
        vmovdqu ymm0, [rbx]
        vmovdqu ymm1, [rbx + 32]
        vpxor ymm2, ymm2, ymm2
        vpcmpeqd ymm3, ymm3, ymm3
        ; mixed sign-bit mask: [-1, 0, -1, 0, 0, -1, 0, -1]
        mov dword [rbx + 64], 0x80000000
        mov dword [rbx + 68], 0x7fffffff
        mov dword [rbx + 72], 0x80000001
        mov dword [rbx + 76], 0x00000000
        mov dword [rbx + 80], 0x12345678
        mov dword [rbx + 84], 0xdeadbeef
        mov dword [rbx + 88], 0x00000001
        mov dword [rbx + 92], 0xffffffff
        vmovdqu ymm4, [rbx + 64]

        ; Preserve flags across fast paths
        mov eax, 5
        cmp eax, 10
        vmaskmovps ymm5, ymm2, [rbx]       ; all-zero load -> 0
        vmaskmovps ymm6, ymm3, [rbx]       ; all-ones load -> LDP
        vmaskmovps ymm7, ymm4, [rbx + 32]  ; partial load
        vmaskmovpd ymm8, ymm3, [rbx + 32]  ; all-ones 256b PD load
        vmaskmovps xmm9, xmm3, [rbx]       ; 128b all-ones load (zeros hi)
        vmaskmovps ymm10, ymm2, [0]        ; all-zero mask on unmapped addr must NOT fault!
        vmaskmovps [rbx + 128], ymm2, ymm0 ; all-zero store -> no-op
        vmaskmovps [rbx + 128], ymm3, ymm0 ; all-ones store -> STP
        vmaskmovpd [rbx + 160], ymm4, ymm1 ; partial store
        pushfq
        pop rcx
        vmovdqu ymm11, [rbx + 128]
        vmovdqu ymm12, [rbx + 160]
        vextractf128 xmm13, ymm6, 1
        vextractf128 xmm14, ymm7, 1
        vextractf128 xmm15, ymm12, 1
    """),
    ("vpcmpistri_mpsadbw_maskmovdqu", """
        lea rbx, [rel scratch_buf]
        ; Set up ASCII strings for VPCMPISTRI
        mov qword [rbx + 0],  0x6f6c6c6548007a61 ; "az\\0Hello"
        mov qword [rbx + 8],  0x0000000000000000
        mov qword [rbx + 16], 0x6f77206f6c6c6548 ; "Hello wo"
        mov qword [rbx + 24], 0x0000000000646c72 ; "rld\\0"
        vmovdqu xmm0, [rbx]
        vmovdqu xmm1, [rbx + 16]
        vpcmpistri xmm0, xmm1, 0x00
        mov r8d, ecx
        pushfq
        pop r9
        vpcmpistri xmm0, [rbx + 16], 0x0c ; equal ordered (substring)
        mov r10d, ecx
        pushfq
        pop r11
        vpcmpistri xmm1, xmm1, 0x18       ; equal each + negative polarity
        mov r13d, ecx
        pushfq
        pop r14

        ; MPSADBW (128b) & VMPSADBW (128b + 256b)
        vmovdqu ymm2, [rbx + 64]
        vmovdqu ymm3, [rbx + 96]
        vmovdqu ymm4, ymm2
        mpsadbw xmm4, xmm3, 0x05
        vmpsadbw xmm5, xmm2, xmm3, 0x02
        vmpsadbw ymm6, ymm2, ymm3, 0x2d
        vmpsadbw ymm7, ymm2, [rbx + 96], 0x1a

        ; VMASKMOVDQU / MASKMOVDQU
        lea rdi, [rbx + 192]
        vmovdqu xmm10, [rbx + 128]
        mov qword [rbx + 144], 0x8000808000800080
        mov qword [rbx + 152], 0x0080000080008000
        vmovdqu xmm11, [rbx + 144]
        mov qword [rdi],     0x1122334455667788
        mov qword [rdi + 8], 0x99aabbccddeeff00
        vmaskmovdqu xmm10, xmm11
        vmovdqu xmm12, [rdi]
        maskmovdqu xmm10, xmm11
        vmovdqu xmm13, [rdi]
        ; Restore rdi before exiting
        mov rdi, rbx
    """),
    ("avx2_pmovsx_zx_mem_128_256", """
        ; Regression test for GB7 workload 202 (Text Processing): 256-bit VPMOVSX/ZX{BW,WD,DQ}
        ; load 16 bytes from memory (m128); log2sz(16) must return 4 (LDR Q), not 0 (LDR B).
        lea rbx, [rel scratch_buf]
        vpmovzxbw ymm0, [rbx]
        vpmovsxbw ymm1, [rbx + 16]
        vpmovzxwd ymm2, [rbx + 32]
        vpmovsxwd ymm3, [rbx + 48]
        vpmovzxdq ymm4, [rbx + 64]
        vpmovsxdq ymm5, [rbx + 80]
        vpmovzxbd ymm6, [rbx + 96]
        vpmovsxbd ymm7, [rbx + 104]
        vpmovzxbq ymm8, [rbx + 112]
        vpmovsxbq ymm9, [rbx + 116]
        vpmovzxwq ymm10, [rbx + 120]
        vpmovsxwq ymm11, [rbx + 128]
        ; Also verify 128-bit memory operand forms (m64/m32/m16) zero upper 128 bits
        vpmovzxbw xmm12, [rbx + 8]
        vpmovzxbd xmm13, [rbx + 12]
        vpmovzxbq xmm14, [rbx + 14]
        vpmovsxdq xmm15, [rbx + 24]
    """),
    ("shift_inline_8_16_rot_cl", """
        lea rbx, [rel scratch_buf]
        ; 8-bit/16-bit shifts by CL with live flags (cnt=0, 1, 7, 8, 15, 16)
        mov rax, 0x123456789abcdef5
        mov rdx, 0xfedcba9876543281
        mov ecx, 0
        shl al, cl
        pushfq
        pop r8
        mov ecx, 1
        shl al, cl
        pushfq
        pop r9                          ; cnt=1: OF/CF/SF/ZF/PF all defined
        and r9, ~0x10                   ; AF is undefined on shifts
        mov ecx, 1
        shr dl, cl
        pushfq
        pop r10                         ; cnt=1: OF=MSB(src)=1, CF=LSB(src)=1 all defined
        and r10, ~0x10
        mov ecx, 7
        shr dl, cl
        pushfq
        pop r11
        and r11, ~0x810                 ; cnt>1: OF & AF undefined on x86 shifts
        shl r10, 16
        or r10, r11
        mov ecx, 8
        sar ah, cl
        pushfq
        pop r11
        and r11, ~0x810
        mov ecx, 16
        shl dx, cl
        pushfq
        pop r12
        and r12, ~0x810
        mov ecx, 5
        sar word [rbx + 128], cl
        movzx r13d, word [rbx + 128]
        ; 8/16-bit and 32/64-bit rotates by 1 (OF defined!) and by CL > 1 (CF defined, OF undefined)
        mov ecx, 1
        rol al, cl
        pushfq
        pop r14                         ; cnt=1: OF & CF defined
        mov ecx, 3
        rol al, cl
        pushfq
        pop r15
        and r15, ~0x800                 ; cnt>1: OF undefined on rotates
        shl r14, 16
        or r14, r15
        ror dx, 1
        rcl al, 1
        rcr dx, 1
        mov ecx, 1
        rol esi, cl
        pushfq
        pop r15                         ; cnt=1: OF & CF defined
        mov ecx, 11
        ror rdi, cl
        pushfq
        pop rcx
        and rcx, ~0x800                 ; cnt>1: OF undefined
        shl r15, 16
        or r15, rcx
        ; CL aliasing destination (shl cl, cl / ror ecx, cl)
        mov rcx, 0x0000000000000005
        shl cl, cl
        mov rcx, 0x123456789abcdef3
        ror ecx, cl
        ; Immediate >= width on 8-bit (clear undefined OF/AF before final dump)
        mov byte [rbx + 130], 0x85
        sar byte [rbx + 130], 9
        shl byte [rbx + 131], 8
        movzx ebp, word [rbx + 130]
        cmp ebp, ebp
    """),
    ("pcmpistri_el0_all_modes", """
        lea rbx, [rel scratch_buf]
        ; S0: "Hello, world!\\0\\0\\0"
        mov qword [rbx + 0],  0x77202c6f6c6c6548
        mov qword [rbx + 8],  0x00000021646c726f
        ; S1: "Hello, World!\\0\\0\\0" (differs at index 7)
        mov qword [rbx + 16], 0x57202c6f6c6c6548
        mov qword [rbx + 24], 0x00000021646c726f
        ; S2: 16 non-zero bytes "0123456789abcdef"
        mov qword [rbx + 32], 0x3736353433323130
        mov qword [rbx + 40], 0x6665646362613938
        ; S3: ranges "azAZ09\\0..."
        mov qword [rbx + 48], 0x000039305a417a61
        mov qword [rbx + 56], 0x0000000000000000
        ; S4: 16-bit words u"abcd\\0" vs u"abXd\\0"
        mov qword [rbx + 64], 0x0064006300620061
        mov qword [rbx + 72], 0x0000000000000000
        mov qword [rbx + 80], 0x0064005800620061
        mov qword [rbx + 88], 0x0000000000000000
        vmovdqu xmm0, [rbx + 0]
        vmovdqu xmm1, [rbx + 16]
        vmovdqu xmm2, [rbx + 32]
        vmovdqu xmm3, [rbx + 48]
        vmovdqu xmm4, [rbx + 64]
        vmovdqu xmm5, [rbx + 80]
        vpxor xmm6, xmm6, xmm6

        ; Equal Each (strlen / strcmp / wcscmp)
        pcmpistri xmm6, xmm0, 0x08      ; strlen(S0) -> 13
        mov r8d, ecx
        vpcmpistri xmm0, xmm1, 0x18     ; strcmp diff -> 7
        shl r8, 8
        or r8, rcx
        pcmpistri xmm0, xmm0, 0x38      ; equal strings masked-neg -> 16
        shl r8, 8
        or r8, rcx
        vpcmpistri xmm4, xmm5, 0x19     ; 16-bit word equal-each neg -> 2
        shl r8, 8
        or r8, rcx
        pcmpistri xmm2, xmm2, 0x48      ; MSB equal-each full 16 bytes -> 15
        shl r8, 8
        or r8, rcx
        pushfq
        pop r9

        ; Equal Any (strchr / strcspn / strspn)
        vpcmpistri xmm3, xmm0, 0x00     ; any of "azAZ09" in S0
        mov r10d, ecx
        pcmpistri xmm3, xmm2, 0x10      ; neg polarity
        shl r10, 8
        or r10, rcx
        vpcmpistri xmm6, xmm0, 0x30     ; empty needle, masked-neg -> 0
        shl r10, 8
        or r10, rcx
        pcmpistri xmm3, xmm0, 0x40      ; MSB equal-any
        shl r10, 8
        or r10, rcx
        pushfq
        pop r11

        ; Ranges (unsigned bytes: "a-z", "A-Z", "0-9")
        vpcmpistri xmm3, xmm0, 0x04     ; first alnum in S0 -> 0
        mov r12d, ecx
        pcmpistri xmm3, xmm0, 0x14      ; first non-alnum in S0 -> 5 (',')
        shl r12, 8
        or r12, rcx
        vpcmpistri xmm3, xmm0, 0x34     ; masked-neg ranges
        shl r12, 8
        or r12, rcx
        pcmpistri xmm3, xmm0, 0x44      ; MSB ranges
        shl r12, 8
        or r12, rcx
        pushfq
        pop r13
    """),
    ("avx2_ymm_cache_crosslane_and_shuf", """
        lea rbx, [rel scratch_buf]
        ; Exercise YMM8..15 cache coherence across cross-lane, widen, mask, cvt, and fast shuffles
        vmovdqu ymm8, [rbx]
        vmovdqu ymm9, [rbx + 32]
        vmovdqu ymm10, [rbx + 64]
        vmovdqu ymm11, [rbx + 96]
        vpaddd ymm12, ymm8, ymm9
        vpmovzxbw ymm13, xmm12
        vpmovzxbw ymm13, xmm13          ; self-alias d == s on ymm13
        vpmovsxdq ymm14, [rbx + 16]
        vinserti128 ymm15, ymm12, xmm13, 1
        vinserti128 ymm8, ymm15, xmm8, 0
        vextracti128 xmm0, ymm15, 1
        vperm2i128 ymm9, ymm12, ymm14, 0x21
        vpermq ymm10, ymm9, 0x4e
        vpermq ymm10, ymm10, 0x93       ; self-alias
        vpmovmskb r8d, ymm10
        vpmovmskb r9d, ymm13
        vcvtdq2ps ymm11, ymm12
        vcvtps2dq ymm12, ymm11
        ; Fast VPSHUFLW / VPSHUFHW / VSHUFPD / VPERMILPD patterns
        vpshuflw ymm1, ymm8, 0x00
        vpshufhw ymm2, ymm9, 0x1b
        vpshuflw ymm3, ymm10, 0xb1
        vpshufhw ymm4, ymm11, 0x4e
        vshufpd ymm5, ymm8, ymm9, 0x5
        vshufpd ymm6, ymm10, ymm12, 0x9
        vpermilpd ymm7, ymm13, 0x6
        pshuflw xmm0, xmm1, 0x1b
        pshufhw xmm0, xmm0, 0xaa
    """),
    ("box64_int_shd_cl_bt_peep", """
        lea rbx, [rel scratch_buf]
        mov qword [rbx + 0], 0x1122334455667788
        mov qword [rbx + 8], 0x99aabbccddeeff00
        mov qword [rbx + 16], 0xfedcba9876543210
        mov qword [rbx + 24], 0x0123456789abcdef
        ; Inline EL0 SHLD/SHRD by CL (64-bit & 32-bit, reg & mem, cl=0/1/17/31/33/63/64, live & dead flags)
        mov r8, 0x0123456789abcdef
        mov r9, 0xfedcba9876543210
        xor ecx, ecx
        shld r8, r9, cl                 ; cl=0 -> untouched, flags untouched
        pushfq
        pop r10
        mov ecx, 1
        shld r8, r9, cl                 ; cl=1 (OF defined!)
        pushfq
        pop r11
        mov ecx, 17
        shrd r8, r9, cl                 ; cl=17 live flags (OF/AF undefined for cnt>1)
        pushfq
        pop r12
        and r12, 0x0c5
        mov ecx, 63
        shld r9, r8, cl                 ; dead flags
        mov ecx, 33
        shrd r9d, r8d, cl               ; 32-bit cl&31 == 1 (OF defined!)
        pushfq
        pop r13
        and r13, 0x8c5
        mov ecx, 19
        shld dword [rbx + 0], r8d, cl   ; mem32 live flags (OF/AF undefined for cnt>1)
        pushfq
        pop r14
        and r14, 0x0c5
        mov ecx, 29
        shrd qword [rbx + 8], r9, cl    ; mem64 dead flags
        ; Aliasing: rcx as dest and as src
        mov rcx, 0x1234567800000007
        shld rcx, r8, cl
        mov r15, rcx
        mov rcx, 0x876543210000000d
        shrd r8, rcx, cl

        ; Fast BT/BTS/BTR/BTC imm + BT+JC/JNC TBZ/TBNZ fusion
        mov rax, 0x8000000100010005
        bt rax, 63
        jc .bt_hi_set
        xor eax, eax
    .bt_hi_set:
        bt eax, 31
        jnc .bt_31_clr
        or r15, 0x100
    .bt_31_clr:
        bts rax, 5
        btr rax, 0
        btc rax, 32
        bts dword [rbx + 16], 3
        btr qword [rbx + 24], 40
        btc qword [rbx + 24], 63

        ; SETcc + MOVZX fusion + XOR r,r live flags
        cmp r8, r9
        setb dl
        movzx edx, dl
        shl r15, 1
        or r15, rdx
        cmp r8, r8
        sete sil
        movzx rsi, sil
        shl r15, 1
        or r15, rsi
        xor rdi, rdi
        pushfq
        pop rdx
        xor r10, rdx

        ; Adjacent MOV LDP/STP pairing (forward, reverse, 32/64, base-overwrite guard)
        mov [rbx + 32], r8
        mov [rbx + 40], r9
        mov [rbx + 56], r11
        mov [rbx + 48], r10
        mov r8, [rbx + 32]
        mov r9, [rbx + 40]
        mov r11, [rbx + 56]
        mov r10, [rbx + 48]
        mov eax, [rbx + 32]
        mov edx, [rbx + 36]
        ; Base overwrite safety: dst1 == base must NOT fuse erroneously
        mov rcx, rbx
        mov rcx, [rcx + 32]
        mov rdi, [rbx + 40]
        xor rax, rcx
        xor rax, rdi
        xor rax, [rbx + 0]
        xor rax, [rbx + 8]
        xor rax, [rbx + 16]
        xor rax, [rbx + 24]
    """),
    ("box64_sse_shuf_mskb_addsub_blend_dp_ptest", """
        lea rbx, [rel scratch_buf]
        ; Prepare finite FP32/FP64 test vectors
        mov rax, 0x402000003fc00000     ; 1.5f, 2.5f
        movq xmm0, rax
        mov rax, 0xc060000040600000     ; 3.5f, -3.5f
        movq xmm1, rax
        punpcklqdq xmm0, xmm1
        mov rax, 0x3f000000bf800000     ; -1.0f, 0.5f
        movq xmm1, rax
        mov rax, 0x40800000c0000000     ; -2.0f, 4.0f
        movq xmm2, rax
        punpcklqdq xmm1, xmm2
        mov rax, 0x3ff8000000000000     ; 1.5
        movq xmm2, rax
        mov rax, 0xc004000000000000     ; -2.5
        movq xmm3, rax
        punpcklqdq xmm2, xmm3
        mov rax, 0x400c000000000000     ; 3.5
        movq xmm3, rax
        mov rax, 0x3fe0000000000000     ; 0.5
        movq xmm4, rax
        punpcklqdq xmm3, xmm4
        movdqu [rbx + 0], xmm0
        movdqu [rbx + 16], xmm1
        movdqu [rbx + 32], xmm2
        movdqu [rbx + 48], xmm3

        ; Fast PMOVMSKB, MOVMSKPS, MOVMSKPD
        pmovmskb r8d, xmm0
        movmskps r9d, xmm1
        movmskpd r10d, xmm2
        movdqa xmm5, xmm0
        movdqa xmm6, xmm1
        movdqa xmm7, xmm2
        movdqa xmm8, xmm3

        ; Fast ADDSUBPS / ADDSUBPD (reg, self-alias, mem)
        addsubps xmm5, xmm1
        addsubps xmm6, xmm6
        addsubpd xmm7, xmm3
        addsubpd xmm8, [rbx + 48]

        ; Exhaustive SHUFPS fast-table coverage
        movdqa xmm9, xmm0
        shufps xmm9, xmm1, 0x88
        shufps xmm9, xmm1, 0xdd
        shufps xmm9, xmm0, 0x44
        shufps xmm9, xmm1, 0xee
        shufps xmm9, xmm9, 0xb1
        shufps xmm9, xmm9, 0x4e
        shufps xmm9, xmm9, 0x39
        shufps xmm9, xmm9, 0x93
        shufps xmm9, xmm0, 0x22
        shufps xmm9, xmm1, 0x77
        shufps xmm9, xmm0, 0x99
        shufps xmm9, xmm9, 0x1b
        shufps xmm9, xmm1, 0x33
        shufps xmm9, xmm0, 0x14
        shufps xmm9, xmm1, 0xe2

        ; Fast PBLENDW / BLENDPS / BLENDPD
        movdqa xmm10, xmm0
        pblendw xmm10, xmm1, 0x0f
        pblendw xmm10, xmm5, 0xf0
        pblendw xmm10, xmm6, 0x55
        pblendw xmm10, xmm7, 0xaa
        pblendw xmm10, xmm8, 0x33
        pblendw xmm10, xmm9, 0xcc
        pblendw xmm10, xmm2, 0x03
        blendps xmm10, xmm0, 0x05
        blendps xmm10, xmm1, 0x0a
        blendpd xmm10, xmm3, 0x01

        ; Fast DPPS / DPPD
        movdqa xmm11, xmm0
        dpps xmm11, xmm1, 0x71
        movdqa xmm12, xmm0
        dpps xmm12, xmm1, 0x7f
        movdqa xmm13, xmm0
        dpps xmm13, xmm1, 0xcf
        movdqa xmm14, xmm2
        dppd xmm14, xmm3, 0x31
        movdqa xmm15, xmm2
        dppd xmm15, xmm3, 0x33

        ; Fast PTEST (d==s zero/nonzero, d!=s, JZ/JC fast path & full flags)
        xor r11, r11
        pxor xmm4, xmm4
        ptest xmm4, xmm4
        jz .pt_z1
        or r11, 1
    .pt_z1:
        ptest xmm0, xmm0
        jnz .pt_nz1
        or r11, 2
    .pt_nz1:
        ptest xmm0, xmm1
        jc .pt_c1
        or r11, 4
    .pt_c1:
        ptest xmm0, xmm0
        pushfq
        pop r12
        ptest xmm0, xmm1
        pushfq
        pop r13
    """),
    ("box64_avx2_fast_vex_scalar_ymm_cache", """
        lea rbx, [rel scratch_buf]
        ; Load clean finite FP32/FP64 into ymm0..ymm15 (exercising ymm8..15 cache!)
        mov rax, 0x3ff4000000000000     ; 1.25
        vmovq xmm0, rax
        vpbroadcastq ymm8, xmm0
        mov rax, 0x4004000000000000     ; 2.5
        vmovq xmm1, rax
        vpbroadcastq ymm9, xmm1
        mov rax, 0xbfd0000000000000     ; -0.25
        vmovq xmm2, rax
        vpbroadcastq ymm10, xmm2
        mov rax, 0x4010000000000000     ; 4.0
        vmovq xmm3, rax
        vpbroadcastq ymm11, xmm3

        ; Keep ymm8..11 dirty in cache while running scalar VEX instructions!
        vaddpd ymm12, ymm8, ymm9
        vmulpd ymm13, ymm10, ymm11
        vaddsd xmm0, xmm0, xmm1         ; d == n scalar merge
        vmulsd xmm0, xmm0, xmm2
        vsubsd xmm1, xmm0, xmm3
        vdivsd xmm2, xmm1, xmm3
        vminsd xmm3, xmm2, xmm0
        vmaxsd xmm4, xmm3, xmm1
        vsqrtsd xmm4, xmm4, xmm3
        mov eax, 42
        vcvtsi2sd xmm5, xmm4, eax
        vcvtsd2si r8, xmm5
        vcvtss2sd xmm5, xmm5, xmm0
        vcvtsd2ss xmm5, xmm5, xmm5
        vroundsd xmm6, xmm5, xmm1, 0x01
        vcmpsd xmm6, xmm6, xmm0, 0x01
        vucomisd xmm0, xmm1
        pushfq
        pop r9
        vpinsrd xmm7, xmm6, eax, 2
        vpextrd r10d, xmm7, 2
        vinsertps xmm7, xmm7, xmm0, 0x10
        vmovd r11d, xmm7
        vmovq r12, xmm6
        ; Now use dirty ymm12/ymm13 + ymm8..11 to verify scalar VEX preserved cache!
        vaddsubpd ymm14, ymm12, ymm13
        vaddsubps ymm15, ymm12, ymm12
        vfmaddsub132pd ymm12, ymm8, ymm9
        vfmsubadd213pd ymm13, ymm10, ymm11
        vfmaddsub231ps ymm14, ymm8, ymm9
        vfmsubadd132ps ymm15, ymm10, ymm11

        ; Fast VSHUFPS / VPBLENDW / VDPPS / VPMOVMSKB / VMOVMSKPS / VMOVMSKPD / VPTEST
        vshufps ymm0, ymm12, ymm13, 0x88
        vshufps ymm1, ymm14, ymm15, 0xdd
        vshufps ymm2, ymm0, ymm1, 0x22
        vshufps ymm3, ymm1, ymm2, 0x77
        vshufps ymm4, ymm2, ymm3, 0x99
        vpblendw ymm5, ymm12, ymm13, 0xaa
        vpblendw ymm6, ymm14, ymm15, 0x33
        vdpps ymm7, ymm8, ymm9, 0x71
        vpmovmskb r13d, ymm12
        vmovmskps r14d, ymm14
        vmovmskpd r15d, ymm13
        vptest ymm12, ymm12
        pushfq
        pop rcx
        vptest ymm12, ymm13
        pushfq
        pop rdx
        xor r15, rcx
        xor r15, rdx
        vmovdqu [rbx + 0], ymm12
        vmovdqu [rbx + 32], ymm13
        vmovdqu [rbx + 64], ymm14
        vmovdqu [rbx + 96], ymm15
    """),
    ("tier2_hot_loop_unroll_ymm_licm_flags", """
        ; 1) Odd-iter (7) AVX2 FMA3 + integer YMM8..11 loop (RO ymm8/ymm9, RW ymm10/ymm11)
        lea rbx, [rel scratch_buf]
        vmovdqu ymm8,  [rbx + 0]
        vmovdqu ymm9,  [rbx + 32]
        vmovdqu ymm10, [rbx + 64]
        vmovdqu ymm11, [rbx + 96]
        mov ecx, 7
    .t2_fma_odd:
        vfmadd231ps ymm10, ymm9, ymm8
        vpaddd ymm11, ymm11, ymm8
        dec ecx
        jnz .t2_fma_odd

        ; 2) Even-iter (8) with mid-body early exit & live dirty ymm10/ymm11
        mov ecx, 8
    .t2_early_loop:
        vpaddq ymm10, ymm10, ymm9
        cmp ecx, 5
        je .t2_early_done
        vpxor ymm11, ymm11, ymm10
        dec ecx
        jnz .t2_early_loop
    .t2_early_done:

        ; 3) Live-out CF preservation across dec rcx; jnz backedge (t2_peek_exit_flags)
        xor eax, eax
        stc
        mov ecx, 9
    .t2_cf_live:
        add rdx, 3
        dec ecx
        jnz .t2_cf_live
        ; CF MUST still be 0 (Wait: add rdx,3 overwrote CF; test stc before a pure dec loop!)
        mov r8, 0x1234
        stc
        mov ecx, 5
    .t2_pure_dec:
        lea r8, [r8 + rcx*2 + 1]
        dec ecx
        jnz .t2_pure_dec
        adc rax, 0x100

        ; 4) RIP-relative LICM + head_i > 0 OSR loop
        mov r9, 0x55
        mov ecx, 6
    .t2_riprel_loop:
        add r9, [rel scratch_buf + 8]
        xor r9, [rel scratch_buf + 16]
        dec ecx
        jnz .t2_riprel_loop

        ; 5) Multi-branch inner loop (forward branch inside loop + backedge)
        mov r10, 0
        mov ecx, 11
    .t2_branch_loop:
        test ecx, 1
        jz .t2_even
        add r10, rcx
        jmp .t2_next
    .t2_even:
        sub r10, 3
    .t2_next:
        dec ecx
        jnz .t2_branch_loop

        ; Verify final ymm10/ymm11 spilled accurately to memory too
        vmovdqu [rbx + 128], ymm10
        vmovdqu [rbx + 160], ymm11
        add rax, [rbx + 128]
        xor rax, [rbx + 160]
    """),
]


def build_asm(tests, use_ymm=False):
    lines = [
        "bits 32",
        "section .text",
        "align 4",
        "dd 0x1BADB002",
        "dd 0x00000000",
        "dd -(0x1BADB002 + 0x00000000)",
        "",
        "global _start",
        "_start:",
        "    cli",
        "    mov esp, 0x180000",
        "    ; Build 64-bit page tables at 0x190000 (PML4), 0x191000 (PDPT), 0x192000 (PD)",
        "    mov edi, 0x190000",
        "    xor eax, eax",
        "    mov ecx, 3072",
        "    rep stosd",
        "    mov dword [0x190000], 0x191003         ; PML4[0] -> PDPT",
        "    mov dword [0x190000 + 256*8], 0x191003 ; PML4[256] (0xffff800000000000) -> PDPT",
        "    mov dword [0x191000], 0x192003         ; PDPT[0] -> PD",
        "    mov dword [0x192000], 0x000083         ; PD[0] -> 0..2MB (2MB page)",
        "    mov dword [0x192008], 0x200083         ; PD[1] -> 2..4MB (2MB page)",
        "",
        "    lgdt [gdt_desc]",
        "    mov eax, cr4",
        "    or eax, (1 << 5) | (1 << 9) | (1 << 10) | (1 << 18) ; PAE | OSFXSR | OSXMMEXCPT | OSXSAVE",
        "    mov cr4, eax",
        "    mov eax, 0x190000",
        "    mov cr3, eax",
        "    mov ecx, 0xC0000080 ; EFER",
        "    rdmsr",
        "    or eax, (1 << 8)    ; LME",
        "    wrmsr",
        "    mov eax, cr0",
        "    and eax, ~(1 << 2)  ; clear EM",
        "    or eax, (1 << 31) | (1 << 1) | (1 << 0) ; PG | MP | PE",
        "    mov cr0, eax",
        "    jmp 0x08:long_entry",
        "",
        "bits 64",
        "long_entry:",
        "    mov ax, 0x10",
        "    mov ds, ax",
        "    mov es, ax",
        "    mov ss, ax",
        "    xor ax, ax",
        "    mov fs, ax",
        "    mov gs, ax",
        "    ; Enable x87 + SSE + AVX in XCR0 (7)",
        "    xor ecx, ecx",
        "    mov eax, 7",
        "    xor edx, edx",
        "    xsetbv",
        "    ; Mask all SSE exceptions in MXCSR (0x1F80)",
        "    mov dword [0x180000], 0x1F80",
        "    ldmxcsr [0x180000]",
        "    ; Jump to canonical negative high-half VA so hvdbt takes over immediately!",
        "    mov rax, 0xffff800000000000",
        "    lea rdx, [rel high_start]",
        "    or rax, rdx",
        "    mov rsp, 0xffff800000180000",
        "    jmp rax",
        "",
        "high_start:",
    ]

    for idx, (name, body) in enumerate(tests):
        lines += [
            f"test_case_{idx}:",
            "    call init_test_state",
            "    push qword 0x2",
            "    popfq",
            body,
            "    pushfq",
            "    pop qword [rel saved_rflags]",
            "    mov [rel saved_gprs + 0*8], rax",
            "    mov [rel saved_gprs + 1*8], rcx",
            "    mov [rel saved_gprs + 2*8], rdx",
            "    mov [rel saved_gprs + 3*8], rbx",
            "    mov [rel saved_gprs + 4*8], rsp",
            "    mov [rel saved_gprs + 5*8], rbp",
            "    mov [rel saved_gprs + 6*8], rsi",
            "    mov [rel saved_gprs + 7*8], rdi",
            "    mov [rel saved_gprs + 8*8], r8",
            "    mov [rel saved_gprs + 9*8], r9",
            "    mov [rel saved_gprs + 10*8], r10",
            "    mov [rel saved_gprs + 11*8], r11",
            "    mov [rel saved_gprs + 12*8], r12",
            "    mov [rel saved_gprs + 13*8], r13",
            "    mov [rel saved_gprs + 14*8], r14",
            "    mov [rel saved_gprs + 15*8], r15",
            f"    mov rdi, {idx}",
            "    call dump_state",
        ]

    lines += [
        "    xor eax, eax",
        "    out 0xf4, eax",
        "    hlt",
        "",
        "init_test_state:",
        "    lea rdi, [rel scratch_buf]",
        "    mov rcx, 64",
        "    mov rax, 0x0123456789abcdef",
        "    mov rdx, 0x1111111122222222",
        ".fill_loop:",
        "    mov [rdi], rax",
        "    add rax, rdx",
        "    ror rax, 13",
        "    add rdi, 8",
        "    dec rcx",
        "    jnz .fill_loop",
        "    lea rdi, [rel scratch_buf]",
    ]
    for r in range(16):
        if use_ymm:
            lines.append(f"    vmovdqu ymm{r}, [rdi + {(r * 16) % 256}]")
        else:
            lines.append(f"    movdqu xmm{r}, [rdi + {(r * 16) % 256}]")
    lines += [
        "    mov rax, 0x13579bdf2468ace0",
        "    mov rcx, 0x0000000000000013",
        "    mov rdx, 0xfedcba9876543210",
        "    mov rbx, 0x8000000000000001",
        "    mov rbp, 0x5555aaaa5555aaaa",
        "    mov rsi, 0x0f0f0f0f0f0f0f0f",
        "    mov rdi, 0xf0f0f0f0f0f0f0f0",
        "    mov r8,  0x123456789abcdef0",
        "    mov r9,  0x0fedcba987654321",
        "    mov r10, 0xdeadbeefcafebabe",
        "    mov r11, 0x00001f0800000703",
        "    lea r12, [rel scratch_buf]",
        "    mov r13, 0x7fffffffffffffff",
        "    mov r14, 0x8000000000000000",
        "    mov r15, 0xffffffffffffffff",
        "    ret",
        "",
        "dump_state:",
        "    lea rbx, [rel saved_vecs]",
    ]
    for r in range(16):
        if use_ymm:
            lines.append(f"    vmovdqu [rbx + {r * 32}], ymm{r}")
        else:
            lines.append(f"    movdqu [rbx + {r * 32}], xmm{r}")

    lines += [
        "    mov al, 'T'",
        "    out 0xe9, al",
        "    mov rax, rdi",
        "    call print_hex16",
        "    mov al, ':'",
        "    out 0xe9, al",
        "    mov rax, [rel saved_rflags]",
        "    and rax, 0x8D5",
        "    call print_hex16",
        "    xor rsi, rsi",
        ".gpr_loop:",
        "    cmp rsi, 4",
        "    je .gpr_next",
        "    mov al, ' '",
        "    out 0xe9, al",
        "    lea rbx, [rel saved_gprs]",
        "    mov rax, [rbx + rsi*8]",
        "    call print_hex64",
        ".gpr_next:",
        "    inc rsi",
        "    cmp rsi, 16",
        "    jb .gpr_loop",
        "    xor rsi, rsi",
        ".vec_loop:",
        "    mov al, ' '",
        "    out 0xe9, al",
        "    lea rbx, [rel saved_vecs]",
        "    mov rax, rsi",
        "    shl rax, 5",
    ]
    if use_ymm:
        lines += [
            "    push rax",
            "    mov rax, [rbx + rax + 24]",
            "    call print_hex64",
            "    mov rax, [rsp]",
            "    mov rax, [rbx + rax + 16]",
            "    call print_hex64",
            "    pop rax",
        ]
    lines += [
        "    push rax",
        "    mov rax, [rbx + rax + 8]",
        "    call print_hex64",
        "    pop rax",
        "    mov rax, [rbx + rax + 0]",
        "    call print_hex64",
        "    inc rsi",
        "    cmp rsi, 16",
        "    jb .vec_loop",
        "    mov al, 10",
        "    out 0xe9, al",
        "    ret",
        "",
        "print_hex16:",
        "    mov ecx, 4",
        "    shl rax, 48",
        "    jmp print_hex_n",
        "print_hex64:",
        "    mov ecx, 16",
        "print_hex_n:",
        "    rol rax, 4",
        "    mov edx, eax",
        "    and edx, 0xf",
        "    cmp dl, 10",
        "    jb .d",
        "    add dl, 'a' - 10",
        "    jmp .o",
        ".d:",
        "    add dl, '0'",
        ".o:",
        "    push rax",
        "    mov al, dl",
        "    out 0xe9, al",
        "    pop rax",
        "    dec ecx",
        "    jnz print_hex_n",
        "    ret",
        "",
        "align 16",
        "gdt_start:",
        "    dq 0x0000000000000000",
        "    dq 0x00af9a000000ffff ; 0x08: 64-bit code",
        "    dq 0x00cf92000000ffff ; 0x10: data",
        "gdt_desc:",
        "    dw gdt_desc - gdt_start - 1",
        "    dd gdt_start",
        "",
        "section .bss",
        "align 64",
        "scratch_buf: resb 512",
        "xsave_buf: resb 1024",
        "saved_rflags: resq 1",
        "saved_gprs: resq 16",
        "saved_vecs: resb 512",
    ]
    return "\n".join(lines) + "\n"


def run_qemu(elf_path, hvdbt, tally_path=None):
    env = os.environ.copy()
    if hvdbt:
        env["LC_HVDBT"] = "1"
        env["LC_HVDBT_FAST_FLAGS"] = "1"
        if tally_path:
            env["LC_HVDBT_TALLY"] = tally_path
    else:
        env.pop("LC_HVDBT", None)
        env.pop("LC_HVDBT_TALLY", None)

    cmd = [
        QEMU,
        "-M", "q35,accel=tcg",
        "-cpu", "Skylake-Client-v3,+avx,+avx2,+fma,+bmi1,+bmi2,+adx,+sha-ni,+aes,+vaes,+pclmulqdq,+vpclmulqdq,+movbe,+x2apic",
        "-smp", "2",
        "-m", "256M",
        "-display", "none",
        "-no-reboot",
        "-chardev", "stdio,id=dbg",
        "-device", "isa-debugcon,iobase=0xe9,chardev=dbg",
        "-device", "isa-debug-exit,iobase=0xf4,iosize=4",
        "-kernel", elf_path,
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20)
    out_lines = [l.strip() for l in res.stdout.splitlines() if re.match(r"^T[0-9a-f]{4}:", l.strip())]
    return out_lines, res.stderr


def main():
    use_ymm = "--ymm" in sys.argv
    tests = list(BASELINE_TESTS)
    with tempfile.TemporaryDirectory(prefix="ext-diff-") as tmp:
        asm_path = os.path.join(tmp, "test.asm")
        obj_path = os.path.join(tmp, "test.o")
        ld_path = os.path.join(tmp, "test.ld")
        elf_path = os.path.join(tmp, "test.elf")
        tally_path = os.path.join(tmp, "dut.tally")

        with open(asm_path, "w") as f:
            f.write(build_asm(tests, use_ymm=use_ymm))

        with open(ld_path, "w") as f:
            f.write(
                "ENTRY(_start)\n"
                "SECTIONS {\n"
                "  . = 0x100000;\n"
                "  .text : { *(.text*) *(.rodata*) *(.data*) }\n"
                "  .bss : { *(.bss*) *(COMMON) }\n"
                "}\n"
            )

        subprocess.run([NASM, "-f", "elf32", asm_path, "-o", obj_path], check=True)
        subprocess.run([LLD, "-m", "elf_i386", "-N", "-T", ld_path, obj_path, "-o", elf_path], check=True)

        tcg_out, tcg_err = run_qemu(elf_path, hvdbt=False)
        dut_out, dut_err = run_qemu(elf_path, hvdbt=True, tally_path=tally_path)

        if "the engine takes over" not in dut_err:
            print("FAIL: hvdbt did not take over! stderr:\n" + dut_err)
            sys.exit(1)

        if len(tcg_out) != len(tests) or len(dut_out) != len(tests):
            print(f"FAIL: expected {len(tests)} lines, got TCG={len(tcg_out)} DUT={len(dut_out)}")
            print("TCG err:", tcg_err)
            print("DUT err:", dut_err)
            sys.exit(1)

        diffs = 0
        for i, (name, _) in enumerate(tests):
            if tcg_out[i] != dut_out[i]:
                print(f"MISMATCH in [{i}] {name}:")
                print(f"  TCG: {tcg_out[i]}")
                print(f"  DUT: {dut_out[i]}")
                diffs += 1
            else:
                print(f"PASS [{i:02d}] {name}")

        if os.path.exists(tally_path):
            tally_txt = open(tally_path).read()
            m = re.search(r"== stops and fallbacks.*?\n(.*?)\n\n", tally_txt, re.S)
            if m:
                print("Tally stops/fallbacks:\n" + m.group(1))
            # A page-edge VEX that is already implemented must stay in the engine.
            # Falling back matches TCG bit-for-bit, so the line compare above cannot see it.
            edge_fb = []
            for line in tally_txt.splitlines():
                if "unknown opcode" not in line and "fetch from a page outside RAM" not in line:
                    continue
                rip_m = re.search(r"@([0-9a-fA-F]+)", line)
                if rip_m and (int(rip_m.group(1), 16) & 0xFFF) >= 0xFF2:
                    edge_fb.append(line.strip())
            if edge_fb:
                print("FAIL: page-edge instruction left the engine")
                for line in edge_fb:
                    print("  " + line)
                diffs += len(edge_fb)

        if os.environ.get("LC_HVDBT_T2_LOG"):
            t2_lines = [l for l in dut_err.splitlines() if "[hvdbt-t2]" in l]
            print(f"Tier-2 promoted logs ({len(t2_lines)}):")
            for l in t2_lines[:20]:
                print("  " + l)

        if diffs:
            sys.exit(1)
        print(f"ALL {len(tests)} DIFFERENTIAL TESTS PASSED (0 bit differences)!")


if __name__ == "__main__":
    main()
