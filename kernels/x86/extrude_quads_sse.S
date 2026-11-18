# Register allocation (13 of 16 XMM):
#   XMM0  = [L0,  L1 ]  Node 0
#   XMM1  = [L2,  L3 ]  Node 0
#   XMM2  = [L4,  L5 ]  Node 1
#   XMM3  = [L6,  L7 ]  Node 1
#   XMM4  = [L8,  L9 ]  Node 2
#   XMM5  = [L10, L11]  Node 2
#   XMM6  = [L12, L13]  Node 3
#   XMM7  = [L14, L15]  Node 3
#   XMM8  = carry temp lo
#   XMM9  = carry temp hi
#   XMM12 = padMask [0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF]
#                   (clears bits 62-63 of the HIGH qword ONLY)

    .text
    .p2align 4

/* ================================================================
 * extrude_quads_sse_core - 1-quad-at-a-time, in-place
 *
 * void extrude_quads_sse_core(uint8_t *buf, uint64_t count)
 *   rdi = buf, rsi = count
 * ================================================================ */
    .globl extrude_quads_sse_core
    .type  extrude_quads_sse_core, @function
extrude_quads_sse_core:
    test   %rsi, %rsi
    jz     .Ldone_1x

    leaq    .Lpad_mask(%rip), %rax
    movdqu  (%rax), %xmm12

    /*
      1) count - 1 into a scratch reg
      2) compute dst_last = buf + (count-1) * 128
      3) compute src_last = buf + (count-1) * 127

      REG: rax = caller-saved [FREE]
    */

    // (1)
    leaq    -1(%rsi), %rax  // rax = rsi - 1 (count - 1)

    // (2)
    movq   %rax, %r8
    shlq   $7, %r8
    addq   %rdi, %r8       // R8 = buf + (count-1) * 128 [DST]

    // (3)
    movq   %r8, %r9
    subq   %rax, %r9       // R9 = buf + (count-1) * 127 [SRC]

.Lloop_1x:
    /*
      load 128 bytes from [SRC] into 8 XMM registers.
      one 16-byte chunk at a time (offsets 0, 16, 32, 48, 64, 80, 96, 112)\
      L0..L15 -> 15 x 8-byte "limbs"(qword) of a quad
    */
    movdqu    (%r9), %xmm0      // [L0  | L1]  ┐ node 0
    movdqu  16(%r9), %xmm1      // [L2  | L3]  ┘

    movdqu  32(%r9), %xmm2      // [L4  | L5]  ┐ node 1
    movdqu  48(%r9), %xmm3      // [L6  | L7]  ┘

    movdqu  64(%r9), %xmm4      // [L8  |  L9] ┐ node 2
    movdqu  80(%r9), %xmm5      // [L10 | L11] ┘

    movdqu  96(%r9), %xmm6      // [L12 | L13] ┐ node 3
    movdqu 112(%r9), %xmm7      // [L14 | L15] ┘

    /*
      ========
       NODE 3
      ========

      In short:
        L12 shifts left 6. Top 6 bits -> caught, land in L13's bottom 6
        L13 shifts left 6. Top 6 bits -> caught, land in L14's bottom 6
        L14 shifts left 6. Top 6 bits -> caught, land in L15's bottom 6
        L15 shifts left 6. Top 6 bits -> abyss (cleared by FR32 padding mask).

      L13 lives in the high qword of XMM6 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │         L13          │          L12         │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      XMM7 holds L14 and L15 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │         L15          │          L14         │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      When we shift L13 left by 6, bits 58..63 of L13 get caught and land
      in bits 0..5 of L14's result

      L13 itself needs NEW bits 0..5 <- those come from L12's top 6 bits
    */
                                //         hi    lo
    movdqu   %xmm6, %xmm8       // XMM6 = [L13 | L12]
    palignr  $8, %xmm5, %xmm8   // XMM8 = [L12 | L11] (save carry lo)
    movdqu   %xmm7, %xmm9       // XMM7 = [L15 | L14]
    palignr  $8, %xmm6, %xmm9   // XMM9 = [L14 | L13] (save carry hi)

    psllq    $6,  %xmm6         // XMM6 = [L13 | L12] << 6  (both hi falls off)
    psrlq    $58, %xmm8         // XMM8 = [L12 | L11] >> 58 (both lo falls off)
    por      %xmm8, %xmm6       // XMM6 = sandwich them together
    //   xmm6=  [ L13 shifted left 6, bits 0..5 = old top 6 of L12
    //          | L12 shifted left 6, bits 0..5 = old top 6 of L11 ]

    psllq    $6,  %xmm7         // XMM7 = [L15 | L14] << 6  (both hi falls off)
    psrlq    $58, %xmm9         // XMM9 = [L14 | L13] >> 58 (both lo falls off)
    por      %xmm9, %xmm7       // XMM7 = sandwich them together
    //   xmm7=  [ L15 shifted left 6, bits 0..5 = old top 6 of L14
    //          | L14 shifted left 6, bits 0..5 = old top 6 of L13 ]

    pand     %xmm12, %xmm7      // clear bits 62-63 of L15

/*
      ========
       NODE 2
      ========

      In short:
        L8  shifts left 4. Top 4 bits -> caught, land in L9's bottom 4
        L9  shifts left 4. Top 4 bits -> caught, land in L10's bottom 4
        L10 shifts left 4. Top 4 bits -> caught, land in L11's bottom 4
        L11 shifts left 4. Top 4 bits -> abyss (cleared by FR32 padding mask).

      L9 lives in the high qword of XMM4 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L9          │          L8          │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      XMM5 holds L10 and L11 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L11         │          L10         │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      When we shift L9 left by 4, bits 60..63 of L9 get caught and land
      in bits 0..3 of L10's result

      L9 itself needs NEW bits 0..3 <- those come from L8's top 4 bits
    */
                                //         hi    lo
    movdqu   %xmm4, %xmm8       // XMM4 = [L9  | L8]
    palignr  $8, %xmm3, %xmm8   // XMM8 = [L8  | L7] (save carry lo)
    movdqu   %xmm5, %xmm9       // XMM5 = [L11 | L10]
    palignr  $8, %xmm4, %xmm9   // XMM9 = [L10 | L9] (save carry hi)

    psllq    $4,  %xmm4         // XMM4 = [L9  | L8] << 4   (both hi falls off)
    psrlq    $60, %xmm8         // XMM8 = [L8  | L7] >> 60  (both lo falls off)
    por      %xmm8, %xmm4       // XMM4 = sandwich them together
    //   xmm4=  [ L9 shifted left 4, bits 0..3 = old top 4 of L8
    //          | L8 shifted left 4, bits 0..3 = old top 4 of L7 ]

    psllq    $4,  %xmm5         // XMM5 = [L11 | L10] << 4  (both hi falls off)
    psrlq    $60, %xmm9         // XMM9 = [L10 | L9] >> 60  (both lo falls off)
    por      %xmm9, %xmm5       // XMM5 = sandwich them together
    //   xmm5=  [ L11 shifted left 4, bits 0..3 = old top 4 of L10
    //          | L10 shifted left 4, bits 0..3 = old top 4 of L9  ]

    pand     %xmm12, %xmm5      // clear bits 62-63 of L11

    /*
      ========
       NODE 1
      ========

      In short:
        L4 shifts left 2. Top 2 bits -> caught, land in L5's bottom 2
        L5 shifts left 2. Top 2 bits -> caught, land in L6's bottom 2
        L6 shifts left 2. Top 2 bits -> caught, land in L7's bottom 2
        L7 shifts left 2. Top 2 bits -> abyss (cleared by FR32 padding mask).

      L5 lives in the high qword of XMM2 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L5          │          L4          │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      XMM3 holds L6 and L7 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L7          │          L6          │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      When we shift L5 left by 2, bits 62..63 of L5 get caught and land
      in bits 0..1 of L6's result

      L5 itself needs NEW bits 0..1 <- those come from L4's top 2 bits
    */
                                //         hi    lo
    movdqu   %xmm2, %xmm8       // XMM2 = [L5 | L4]
    palignr  $8, %xmm1, %xmm8   // XMM8 = [L4 | L3] (save carry lo)
    movdqu   %xmm3, %xmm9       // XMM3 = [L7 | L6]
    palignr  $8, %xmm2, %xmm9   // XMM9 = [L6 | L5] (save carry hi)

    psllq    $2,  %xmm2         // XMM2 = [L5 | L4] << 2    (both hi falls off)
    psrlq    $62, %xmm8         // XMM8 = [L4 | L3] >> 62   (both lo falls off)
    por      %xmm8, %xmm2       // XMM2 = sandwich them together
    //   xmm2=  [ L5 shifted left 2, bits 0..1 = old top 2 of L4
    //          | L4 shifted left 2, bits 0..1 = old top 2 of L3 ]

    psllq    $2,  %xmm3         // XMM3 = [L7 | L6] << 2    (both hi falls off)
    psrlq    $62, %xmm9         // XMM9 = [L6 | L5] >> 62   (both lo falls off)
    por      %xmm9, %xmm3       // XMM3 = sandwich them together
    //   xmm3=  [ L7 shifted left 2, bits 0..1 = old top 2 of L6
    //          | L6 shifted left 2, bits 0..1 = old top 2 of L5 ]

    pand     %xmm12, %xmm3      // clear bits 62-63 of L7

    /*
      ========
       NODE 0
      ========

      Node 0 doesn't shift - it's the bottom of the quad, no carry-in from below.
      Just mask bits 62-63 of L3 for FR32 padding.

      L1 lives in the high qword of XMM0 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L1          │          L0          │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0

      XMM1 holds L2 and L3 (128 bits):
      ┌──────────────────────┬──────────────────────┐
      │      high qword      │      low qword       │
      │          L3          │          L2          │
      └──────────────────────┴──────────────────────┘
      127                   64 63                    0
    */

    pand     %xmm12, %xmm1      // clear bits 62-63 of L3

    /* STORE: 8 × 128-bit */
    movdqu %xmm0,    (%r8)
    movdqu %xmm1,  16(%r8)
    movdqu %xmm2,  32(%r8)
    movdqu %xmm3,  48(%r8)
    movdqu %xmm4,  64(%r8)
    movdqu %xmm5,  80(%r8)
    movdqu %xmm6,  96(%r8)
    movdqu %xmm7, 112(%r8)

    // Advance ptrs (backward): src -= 127, dst -= 128
    subq   $127, %r9
    subq   $128, %r8
    decq   %rsi
    jnz    .Lloop_1x

.Ldone_1x:
    ret

/* ================================================================
 * Read-only data
 * ================================================================ */
    .section .rodata,"a",@progbits
    .align 16

/* padMask: low qword = all ones (preserve), high qword = 0x3FFFFFFFFFFFFFFF (clear top 2 bits) */
.Lpad_mask:
    .quad 0xFFFFFFFFFFFFFFFF
    .quad 0x3FFFFFFFFFFFFFFF

    .section .note.GNU-stack,"",@progbits