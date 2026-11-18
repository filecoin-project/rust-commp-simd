/**
 * extrude_avx512.S - FR32 extrusion (127B -> 128B per quad), x86-64 AVX512-VBMI2
 *
 * Converted from Plan 9 extrudinator/extrude_avx512.s to GAS AT&T syntax.
 * C ABI: rdi=buf, rsi=count
 *
 * Uses: VPSHLDQ (fused double-precision shift-left) + VALIGNQ (qword align)
 *      In constrast with three SSE instructions, VPSHLDQ accomplishes the same with a single instruciton
 *
 * Requires: AVX512-VBMI2 (Ice Lake+, Tiger Lake+, Zen4+).
 *
 * Per quad: 128B loaded as 4×YMM (16 uint64 limbs LE).
 *   Node 3: VALIGNQ+VPSHLDQ $6 + VPANDQ
 *   Node 2: VALIGNQ+VPSHLDQ $4 + VPANDQ
 *   Node 1: VALIGNQ+VPSHLDQ $2 + VPANDQ
 *   Node 0: VPANDQ only
 *
 * padMask: lanes 0-2 = all ones, lane 3 = 0x3FFFFFFFFFFFFFFF
 */

    .text
    .p2align 4

/* ================================================================
 * extrude_quads_avx512_core - 1-quad-at-a-time
 *
 * void extrude_quads_avx512_core(uint8_t *buf, uint64_t count)
 *   rdi = buf, rsi = count
 * ================================================================ */
    .globl extrude_quads_avx512_core
    .type  extrude_quads_avx512_core, @function
extrude_quads_avx512_core:
    testq   %rsi, %rsi
    jz      .Ldone_1x

    vmovdqu .Lpad_mask(%rip), %ymm5

    leaq    -1(%rsi), %rax        // rax = count - 1

    movq    %rax, %r8
    shlq    $7, %r8
    addq    %rdi, %r8             // r8 = buf + (count-1)*128 (dst_last)

    movq    %r8, %r9
    subq    %rax, %r9             // r9 = buf + (count-1)*127 (src_last)

.Lloop_1x:
    vmovdqu    (%r9), %ymm0      // load 32 bytes of source from (%r9) into YMM0
    vmovdqu  32(%r9), %ymm1
    vmovdqu  64(%r9), %ymm2
    vmovdqu  96(%r9), %ymm3

    /*
      ========
       NODE 3
      ========

      In short:
        L12 shifts left 6. Top 6 bits -> caught, land in L13's bottom 6
        L13 shifts left 6. Top 6 bits -> caught, land in L14's bottom 6
        L14 shifts left 6. Top 6 bits -> caught, land in L15's bottom 6
        L15 shifts left 6. Top 6 bits -> abyss (cleared by FR32 padding mask).

      All 4 limbs of node 3 live in YMM3 (256 bits = 4 x 64-bit qwords):
      ┌────────────┬────────────┬────────────┬────────────┐
      │   lane 3   │   lane 2   │   lane 1   │   lane 0   │
      │    L15     │    L14     │    L13     │    L12     │
      └────────────┴────────────┴────────────┴────────────┘
      255       192 191      128 127       64 63          0

      YMM2 holds node 2 (L8..L11) immediately below:
      ┌────────────┬────────────┬────────────┬────────────┐
      │   lane 3   │   lane 2   │   lane 1   │   lane 0   │
      │    L11     │    L10     │     L9     │     L8     │
      └────────────┴────────────┴────────────┴────────────┘
      255       192 191      128 127       64 63          0

      When we shift L13 left by 6, bits 58..63 of L13 get caught and land
      in bits 0..5 of L14's result

      L12 itself needs NEW bits 0..5 <- those come from L11's top 6 bits
                                         (L11 is the top lane of YMM2)
    */
    valignq $3, %ymm2, %ymm3, %ymm4    // YMM4 = [L14 | L13 | L12 | L11]
    vpshldq $6, %ymm4, %ymm3, %ymm3    // YMM3 = [L15<<6|L14[63:58] | L14<<6|L13[63:58] | L13<<6|L12[63:58] | L12<<6|L11[63:58]]
    vpandq  %ymm5, %ymm3, %ymm3        // YMM3 = [L15<<6|L14[63:58] & 0x3FFF... | L14<<6|L13[63:58] | L13<<6|L12[63:58] | L12<<6|L11[63:58]]

    /* Node 2: shift left 4 */
    valignq $3, %ymm1, %ymm2, %ymm4
    vpshldq $4, %ymm4, %ymm2, %ymm2
    vpandq  %ymm5, %ymm2, %ymm2

    /* Node 1: shift left 2 */
    valignq $3, %ymm0, %ymm1, %ymm4
    vpshldq $2, %ymm4, %ymm1, %ymm1
    vpandq  %ymm5, %ymm1, %ymm1

    /* Node 0: just mask */
    vpandq  %ymm5, %ymm0, %ymm0

    vmovdqu %ymm0,    (%r8)
    vmovdqu %ymm1,  32(%r8)
    vmovdqu %ymm2,  64(%r8)
    vmovdqu %ymm3,  96(%r8)

    subq    $127, %r9
    subq    $128, %r8
    decq    %rsi
    jnz     .Lloop_1x

.Ldone_1x:
    vzeroupper
    ret
    .size extrude_quads_avx512_core, .-extrude_quads_avx512_core


/* ================================================================
 * extrude_quads_4x_avx512_core - 4-quad unrolled
 *
 * void extrude_quads_4x_avx512_core(uint8_t *dst, const uint8_t *src, uint64_t count)
 *   rdi = dst, rsi = src, rdx = count (MUST be multiple of 4)
 *
 * Register allocation (21 of 32 YMM):
 *   Q0: ymm0-ymm3     Q1: ymm4-ymm7
 *   Q2: ymm8-ymm11    Q3: ymm12-ymm15
 *   ymm16 = padMask
 *   ymm17-ymm20 = carry temps
 *
 * Port scheduling (Tiger Lake / Ice Lake):
 *   VALIGNQ  -> port 5 only, lat 1, tput 1
 *   VPSHLDQ  -> port 0/1, lat 1, tput 1
 *   VPANDQ   -> port 0/5, lat 1, tput 0.5
 * Interleaving across quads saturates ports 0, 1, 5 in parallel.
 * ================================================================ */
    .p2align 4
    .globl extrude_quads_4x_avx512_core
    .type  extrude_quads_4x_avx512_core, @function
extrude_quads_4x_avx512_core:
    testq   %rsi, %rsi
    jz      .Ldone_4x

    vmovdqu64 .Lpad_mask(%rip), %ymm16

    leaq    -1(%rsi), %rax        // rax = count - 1
    movq    %rax, %r8
    shlq    $7, %r8
    addq    %rdi, %r8             // r8 = buf + (count-1)*128
    movq    %r8, %r9
    subq    %rax, %r9             // r9 = buf + (count-1)*127

    subq    $384, %r8             // r8 -= 3*128 -> points to start of last 4-quad batch dst
    subq    $381, %r9             // r9 -= 3*127 -> points to start of last 4-quad batch src

.Lloop_4x:
    /* ===== LOAD 4 quads (16 YMM loads) ===== */
    /* Q0: src[0..127] */
    vmovdqu    (%r9), %ymm0
    vmovdqu  32(%r9), %ymm1
    vmovdqu  64(%r9), %ymm2
    vmovdqu  96(%r9), %ymm3
    /* Q1: src[127..254] */
    vmovdqu 127(%r9), %ymm4
    vmovdqu 159(%r9), %ymm5
    vmovdqu 191(%r9), %ymm6
    vmovdqu 223(%r9), %ymm7
    /* Q2: src[254..381] */
    vmovdqu 254(%r9), %ymm8
    vmovdqu 286(%r9), %ymm9
    vmovdqu 318(%r9), %ymm10
    vmovdqu 350(%r9), %ymm11
    /* Q3: src[381..508] */
    vmovdqu 381(%r9), %ymm12
    vmovdqu 413(%r9), %ymm13
    vmovdqu 445(%r9), %ymm14
    vmovdqu 477(%r9), %ymm15

    /* ===== NODE 3: shift left 6, interleaved across quads ===== */
    valignq $3, %ymm2,  %ymm3,  %ymm17
    valignq $3, %ymm6,  %ymm7,  %ymm18
    vpshldq $6, %ymm17, %ymm3,  %ymm3
    vpshldq $6, %ymm18, %ymm7,  %ymm7
    valignq $3, %ymm10, %ymm11, %ymm19
    valignq $3, %ymm14, %ymm15, %ymm20
    vpshldq $6, %ymm19, %ymm11, %ymm11
    vpshldq $6, %ymm20, %ymm15, %ymm15
    vpandq  %ymm16, %ymm3,  %ymm3
    vpandq  %ymm16, %ymm7,  %ymm7
    vpandq  %ymm16, %ymm11, %ymm11
    vpandq  %ymm16, %ymm15, %ymm15

    /* ===== NODE 2: shift left 4, interleaved ===== */
    valignq $3, %ymm1,  %ymm2,  %ymm17
    valignq $3, %ymm5,  %ymm6,  %ymm18
    vpshldq $4, %ymm17, %ymm2,  %ymm2
    vpshldq $4, %ymm18, %ymm6,  %ymm6
    valignq $3, %ymm9,  %ymm10, %ymm19
    valignq $3, %ymm13, %ymm14, %ymm20
    vpshldq $4, %ymm19, %ymm10, %ymm10
    vpshldq $4, %ymm20, %ymm14, %ymm14
    vpandq  %ymm16, %ymm2,  %ymm2
    vpandq  %ymm16, %ymm6,  %ymm6
    vpandq  %ymm16, %ymm10, %ymm10
    vpandq  %ymm16, %ymm14, %ymm14

    /* ===== NODE 1: shift left 2, interleaved ===== */
    valignq $3, %ymm0,  %ymm1,  %ymm17
    valignq $3, %ymm4,  %ymm5,  %ymm18
    vpshldq $2, %ymm17, %ymm1,  %ymm1
    vpshldq $2, %ymm18, %ymm5,  %ymm5
    valignq $3, %ymm8,  %ymm9,  %ymm19
    valignq $3, %ymm12, %ymm13, %ymm20
    vpshldq $2, %ymm19, %ymm9,  %ymm9
    vpshldq $2, %ymm20, %ymm13, %ymm13
    vpandq  %ymm16, %ymm1,  %ymm1
    vpandq  %ymm16, %ymm5,  %ymm5
    vpandq  %ymm16, %ymm9,  %ymm9
    vpandq  %ymm16, %ymm13, %ymm13

    /* ===== NODE 0: just mask ===== */
    vpandq  %ymm16, %ymm0,  %ymm0
    vpandq  %ymm16, %ymm4,  %ymm4
    vpandq  %ymm16, %ymm8,  %ymm8
    vpandq  %ymm16, %ymm12, %ymm12

    /* ===== STORE 4 quads (16 YMM stores) ===== */
    vmovdqu %ymm0,     (%r8)
    vmovdqu %ymm1,   32(%r8)
    vmovdqu %ymm2,   64(%r8)
    vmovdqu %ymm3,   96(%r8)
    vmovdqu %ymm4,  128(%r8)
    vmovdqu %ymm5,  160(%r8)
    vmovdqu %ymm6,  192(%r8)
    vmovdqu %ymm7,  224(%r8)
    vmovdqu %ymm8,  256(%r8)
    vmovdqu %ymm9,  288(%r8)
    vmovdqu %ymm10, 320(%r8)
    vmovdqu %ymm11, 352(%r8)
    vmovdqu %ymm12, 384(%r8)
    vmovdqu %ymm13, 416(%r8)
    vmovdqu %ymm14, 448(%r8)
    vmovdqu %ymm15, 480(%r8)

    /* Advance backward: src -= 4*127 = 508, dst -= 4*128 = 512 */
    subq    $508, %r9
    subq    $512, %r8
    subq    $4, %rsi
    jnz     .Lloop_4x

.Ldone_4x:
    vzeroupper
    ret
    .size extrude_quads_4x_avx512_core, .-extrude_quads_4x_avx512_core


/* ================================================================
 * Read-only data
 * ================================================================ */
    .section .rodata,"a",@progbits
    .align 32

/* padMask: lanes 0-2 = all ones, lane 3 = mask62 (top 2 bits zero) */
.Lpad_mask:
    .quad 0xFFFFFFFFFFFFFFFF
    .quad 0xFFFFFFFFFFFFFFFF
    .quad 0xFFFFFFFFFFFFFFFF
    .quad 0x3FFFFFFFFFFFFFFF

    .section .note.GNU-stack,"",@progbits