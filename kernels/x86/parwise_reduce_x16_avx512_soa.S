# parwise_reduce_x16_avx512_soa.S - Breadth-first SoA kernel
#
# Drop-in replacement: same signature as pairwise_reduce_buffer_x16.
#   void pairwise_reduce_buffer_x16_soa(uint8_t *buf, size_t byte_len, size_t out_pair_count)
#
# Level 0: AoS input via sha256_x16_avx512, store results in SoA (no untranspose)
# Level 1+: SoA input via direct PROCESS_LOOP, store SoA (no transpose/untranspose)
# Exit: extract root, bswap to AoS, store 32 bytes

#include "sha256_reg_aliases.h"
#include "sha256_round_avx512.h"
#include "transpose16_avx512.h"
#include "sha256_commp_macros.h"
#include "sha256_inline_block1.h"

#define IS_LEVEL0    664
#define LEVEL_NPAIRS 672

/* ---- SoA-specific macros ---- */

#define STORE_SOA(dst) \
    vmovdqu32  A, 0*64(dst)  ;\
    vmovdqu32  B, 1*64(dst)  ;\
    vmovdqu32  C, 2*64(dst)  ;\
    vmovdqu32  D, 3*64(dst)  ;\
    vmovdqu32  E, 4*64(dst)  ;\
    vmovdqu32  F, 5*64(dst)  ;\
    vmovdqu32  G, 6*64(dst)  ;\
    vmovdqu32  H, 7*64(dst)

#define DEINTERLEAVE_SOA_TWO_BLOCKS(base) \
    leaq .LS_EVEN_IDX(%rip), %rax; vmovdqa32 (%rax), TMP4                          ;\
    leaq .LS_ODD_IDX(%rip), %rax; vmovdqa32 (%rax), TMP5                          ;\
    vmovdqu32  0*64(base), TMP0;  vmovdqu32  0*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W0                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W8                          ;\
    vmovdqu32  1*64(base), TMP0;  vmovdqu32  1*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W1                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W9                          ;\
    vmovdqu32  2*64(base), TMP0;  vmovdqu32  2*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W2                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W10                         ;\
    vmovdqu32  3*64(base), TMP0;  vmovdqu32  3*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W3                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W11                         ;\
    vmovdqu32  4*64(base), TMP0;  vmovdqu32  4*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W4                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W12                         ;\
    vmovdqu32  5*64(base), TMP0;  vmovdqu32  5*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W5                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W13                         ;\
    vmovdqu32  6*64(base), TMP0;  vmovdqu32  6*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W6                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W14                         ;\
    vmovdqu32  7*64(base), TMP0;  vmovdqu32  7*64+512(base), TMP1 ;\
    vpermd     TMP0, TMP4, TMP2;  vpermd     TMP1, TMP4, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W7                          ;\
    vpermd     TMP0, TMP5, TMP2;  vpermd     TMP1, TMP5, TMP3    ;\
    vinserti64x4 $1, %ymm12, %zmm11, W15

#define DEINTERLEAVE_SOA_1BLOCK(base) \
    leaq .LS_EVEN_IDX(%rip), %rax; vmovdqa32 (%rax), TMP4  ;\
    leaq .LS_ODD_IDX(%rip), %rax; vmovdqa32 (%rax), TMP5  ;\
    vmovdqu32  0*64(base), TMP0;  vpermd TMP0, TMP4, W0;  vpermd TMP0, TMP5, W8  ;\
    vmovdqu32  1*64(base), TMP0;  vpermd TMP0, TMP4, W1;  vpermd TMP0, TMP5, W9  ;\
    vmovdqu32  2*64(base), TMP0;  vpermd TMP0, TMP4, W2;  vpermd TMP0, TMP5, W10 ;\
    vmovdqu32  3*64(base), TMP0;  vpermd TMP0, TMP4, W3;  vpermd TMP0, TMP5, W11 ;\
    vmovdqu32  4*64(base), TMP0;  vpermd TMP0, TMP4, W4;  vpermd TMP0, TMP5, W12 ;\
    vmovdqu32  5*64(base), TMP0;  vpermd TMP0, TMP4, W5;  vpermd TMP0, TMP5, W13 ;\
    vmovdqu32  6*64(base), TMP0;  vpermd TMP0, TMP4, W6;  vpermd TMP0, TMP5, W14 ;\
    vmovdqu32  7*64(base), TMP0;  vpermd TMP0, TMP4, W7;  vpermd TMP0, TMP5, W15

#define UNTRANSPOSE_SOA_TO_AOS \
    vunpcklps  B, A, TMP0 ;\
    vunpckhps  B, A, TMP1 ;\
    vunpcklps  D, C, TMP2 ;\
    vunpckhps  D, C, TMP3 ;\
    vunpcklps  F, E, TMP4 ;\
    vunpckhps  F, E, TMP5 ;\
    vunpcklps  H, G, TMP6 ;\
    vunpckhps  H, G, T1   ;\
    vshufps  $0x44, TMP2, TMP0, A ;\
    vshufps  $0x44, TMP6, TMP4, B ;\
    vshufps  $0xEE, TMP2, TMP0, C ;\
    vshufps  $0xEE, TMP6, TMP4, D ;\
    vshufps  $0x44, TMP3, TMP1, E ;\
    vshufps  $0x44, T1,   TMP5, F ;\
    vshufps  $0xEE, TMP3, TMP1, G ;\
    vshufps  $0xEE, T1,   TMP5, H ;\
    vshuff32x4  $0x44, B, A, TMP0 ;\
    vshuff32x4  $0xEE, B, A, TMP1 ;\
    vshuff32x4  $0x44, D, C, TMP2 ;\
    vshuff32x4  $0xEE, D, C, TMP3 ;\
    vshuff32x4  $0x44, F, E, TMP4 ;\
    vshuff32x4  $0xEE, F, E, TMP5 ;\
    vshuff32x4  $0x44, H, G, TMP6 ;\
    vshuff32x4  $0xEE, H, G, T1   ;\
    vshuff32x4  $0x88, TMP2, TMP0, A ;\
    vshuff32x4  $0x88, TMP6, TMP4, B ;\
    vshuff32x4  $0xDD, TMP2, TMP0, C ;\
    vshuff32x4  $0xDD, TMP6, TMP4, D ;\
    vshuff32x4  $0x88, TMP3, TMP1, E ;\
    vshuff32x4  $0x88, T1,   TMP5, F ;\
    vshuff32x4  $0xDD, TMP3, TMP1, G ;\
    vshuff32x4  $0xDD, T1,   TMP5, H

#define BYTESWAP_ALL \
    leaq .LS_BYTESWAP(%rip), %rax; vbroadcasti32x4 (%rax), TMP1 ;\
    vpshufb TMP1, A, A ;\
    vpshufb TMP1, B, B ;\
    vpshufb TMP1, C, C ;\
    vpshufb TMP1, D, D ;\
    vpshufb TMP1, E, E ;\
    vpshufb TMP1, F, F ;\
    vpshufb TMP1, G, G ;\
    vpshufb TMP1, H, H

/* ---- Function ---- */

.section .text
.globl pairwise_reduce_buffer_x16_soa
.type  pairwise_reduce_buffer_x16_soa, @function
.align 64

pairwise_reduce_buffer_x16_soa:
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15

    movq    %rsp, %rbp
    movq    %rdx, %r15
    movq    %rdi, %rbx
    movq    %rsi, %r13
    shrq    $6,   %r13
    movq    %rdi, %r12
    movq    %rdi, %r14

    subq    $768, %rsp
    andq    $-64, %rsp

    movq    $1, IS_LEVEL0(%rsp)

.Llevel_loop:
    cmpq    %r15, %r13
    jbe     .Lexit_convert
    movq    %rbx, %r12
    movq    %rbx, %r14
    movq    %r13, LEVEL_NPAIRS(%rsp)

    cmpq    $0, IS_LEVEL0(%rsp)
    je      .Lbatch_loop_soa

// ========================================================================
// LEVEL 0: AoS input, sha256_x16_avx512 for block 1
// ========================================================================

.Lbatch_loop_l0:
    cmpq    $16, %r13
    jb      .Lbatch_tail_l0

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    SCATTER_PTRS(%r12)
    LOAD_IV_STACK

    // movq    %rsp, %rdi
    // movl    $1,   %esi
    // movl    $0xFFFF, %edx
    INLINE_SHA256_BLOCK1

    // Block 2: KW2 inlined
    LOAD_STATE_STACK
    SAVE_STATE_W
    SHA_ROUNDS_KW2
    ADDBACK_W
    TRUNC254

    movq    SAVE_R14(%rsp), %r14
    STORE_SOA(%r14)

    movq    SAVE_R12(%rsp), %r12
    movq    SAVE_R13(%rsp), %r13
    addq    $1024, %r12
    addq    $512,  %r14
    subq    $16,   %r13
    jmp     .Lbatch_loop_l0

.Lbatch_tail_l0:
    testq   %r13, %r13
    jz      .Llevel_done

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    movq    %r12, 512+0*8(%rsp)
.irp lane,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    leaq    \lane*64(%r12), %rax
    cmpq    $\lane, %r13
    cmovbe  %r12, %rax
    movq    %rax, 512+\lane*8(%rsp)
.endr

    movl    $1, %eax
    movl    %r13d, %ecx
    shll    %cl, %eax
    subl    $1, %eax
    movl    %eax, %edx

    LOAD_IV_STACK

    movq    %rsp, %rdi
    movl    $1,   %esi
    // edx = partial mask
    INLINE_SHA256_BLOCK1

    // Block 2: KW2 inlined (tail)
    LOAD_STATE_STACK
    SAVE_STATE_W
    SHA_ROUNDS_KW2
    ADDBACK_W
    TRUNC254

    movq    SAVE_R14(%rsp), %r14
    STORE_SOA(%r14)

    movq    SAVE_R13(%rsp), %r13
    addq    $512, %r14
    jmp     .Llevel_done

// ========================================================================
// LEVEL 1+: SoA input, direct PROCESS_LOOP (no transpose)
// ========================================================================

.Lbatch_loop_soa:
    cmpq    $16, %r13
    jb      .Lbatch_tail_soa

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    DEINTERLEAVE_SOA_TWO_BLOCKS(%r12)

    LOAD_IV_REGS
    SAVE_STATE_STACK
    SHA_ROUNDS_BLOCK1
    ADDBACK_STACK
    SAVE_STATE_W
    SHA_ROUNDS_KW2
    ADDBACK_W
    TRUNC254

    movq    SAVE_R14(%rsp), %r14
    STORE_SOA(%r14)

    movq    SAVE_R12(%rsp), %r12
    movq    SAVE_R13(%rsp), %r13
    addq    $1024, %r12
    addq    $512,  %r14
    subq    $16,   %r13
    jmp     .Lbatch_loop_soa

.Lbatch_tail_soa:
    testq   %r13, %r13
    jz      .Llevel_done

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    DEINTERLEAVE_SOA_1BLOCK(%r12)

    LOAD_IV_REGS
    SAVE_STATE_STACK
    SHA_ROUNDS_BLOCK1
    ADDBACK_STACK
    SAVE_STATE_W
    SHA_ROUNDS_KW2
    ADDBACK_W
    TRUNC254

    movq    SAVE_R14(%rsp), %r14
    STORE_SOA(%r14)

    addq    $512, %r14
    jmp     .Llevel_done

.Llevel_done:
    movq    LEVEL_NPAIRS(%rsp), %r13
    shrq    $1, %r13
    movq    $0, IS_LEVEL0(%rsp)
    jmp     .Llevel_loop

.Lexit_convert:
    cmpq    $0, IS_LEVEL0(%rsp)
    jne     .Lexit_done

    testq   %r13, %r13
    jz      .Lexit_root

    // Partial reduction: r13 results in SoA at %rbx -> convert to AoS
    movq    %rbx, %r12
    movq    %rbx, %r14
    movq    LEVEL_NPAIRS(%rsp), %rcx

.Lconvert_block:
    vmovdqu32  0*64(%r12), A
    vmovdqu32  1*64(%r12), B
    vmovdqu32  2*64(%r12), C
    vmovdqu32  3*64(%r12), D
    vmovdqu32  4*64(%r12), E
    vmovdqu32  5*64(%r12), F
    vmovdqu32  6*64(%r12), G
    vmovdqu32  7*64(%r12), H

    UNTRANSPOSE_SOA_TO_AOS
    BYTESWAP_ALL

    vmovdqu32  A, 0*64(%r14)
    vmovdqu32  B, 1*64(%r14)
    vmovdqu32  C, 2*64(%r14)
    vmovdqu32  D, 3*64(%r14)
    vmovdqu32  E, 4*64(%r14)
    vmovdqu32  F, 5*64(%r14)
    vmovdqu32  G, 6*64(%r14)
    vmovdqu32  H, 7*64(%r14)

    addq    $512, %r12
    addq    $512, %r14
    subq    $16, %rcx
    ja      .Lconvert_block
    jmp     .Lexit_done

.Lexit_root:
    // Full reduction (r13=0): single root in lane 0 of SoA
    movl  0*64(%rbx), %eax;  bswapl %eax;  movl %eax, 0*4(%rbx)
    movl  1*64(%rbx), %eax;  bswapl %eax;  movl %eax, 1*4(%rbx)
    movl  2*64(%rbx), %eax;  bswapl %eax;  movl %eax, 2*4(%rbx)
    movl  3*64(%rbx), %eax;  bswapl %eax;  movl %eax, 3*4(%rbx)
    movl  4*64(%rbx), %eax;  bswapl %eax;  movl %eax, 4*4(%rbx)
    movl  5*64(%rbx), %eax;  bswapl %eax;  movl %eax, 5*4(%rbx)
    movl  6*64(%rbx), %eax;  bswapl %eax;  movl %eax, 6*4(%rbx)
    movl  7*64(%rbx), %eax;  bswapl %eax;  movl %eax, 7*4(%rbx)
    andb  $0x3F, 31(%rbx)

.Lexit_done:
    vzeroupper
    movq    %rbp, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret

.size pairwise_reduce_buffer_x16_soa, .-pairwise_reduce_buffer_x16_soa

.section .rodata,"a",@progbits
#include "sha256_commp_rodata.h"
.section .note.GNU-stack,"",@progbits