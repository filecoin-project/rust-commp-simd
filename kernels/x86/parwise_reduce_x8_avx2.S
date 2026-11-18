#include "sha256_reg_aliases_avx2.h"
#include "sha256_round_avx2.h"
#include "transpose8_avx2.h"
#include "sha256_commp_macros_avx2.h"

/*
  The x8 stack is different (than x16)

  It needs:
    - W slots: 16 x 32 = 512 bytes
    - Digests: 8 x 32 = 256 bytes
    - GPR saves
    - IS_LEVEL0
    - LEVEL_NPAIRS

   rsp
    │
    ├─ 0x000 ─ W[ 0]           ┐
    ├─ 0x020 ─ W[ 1]           │ 16 W slots
    ├─  ...                    │ (message schedule)
    ├─ 0x1E0 ─ W[15]           ┘
    ├─ 0x200 ─ digest A save   ┐
    ├─ 0x220 ─ digest B save   │ 8 state saves
    ├─  ...                    │ (Merkle-Damgard add-back)
    ├─ 0x2E0 ─ digest H save   ┘
    ├─ 0x300 ─ saved r12       ┐
    ├─ 0x308 ─ saved r13       │ GPR saves
    ├─ 0x310 ─ saved r14       ┘
    ├─ 0x318 ─ IS_LEVEL0
    ├─ 0x320 ─ LEVEL_NPAIRS
    └─ 0x328 ─ total: 808 bytes (aligned to 832 = 0x340)
*/

#define W_AREA        0
#define DIGEST_SAVE   0x200
#define SAVE_R12      0x300
#define SAVE_R13      0x308
#define SAVE_R14      0x310
#define IS_LEVEL0     0x318
#define LEVEL_NPAIRS  0x320
#define FRAMESZ       0x340

/* ---- Function ---- */
.section .text
.globl pairwise_reduce_buffer_x8_avx2
.type  pairwise_reduce_buffer_x8_avx2, @function
.align 32 // AVX2 max width is 32 byte

pairwise_reduce_buffer_x8_avx2:
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15

    movq    %rsp, %rbp  // save rsp
    movq    %rdx, %r15  // r15 = out_pair_count (arg3)
    movq    %rdi, %rbx  // rbx = buf (arg1, preserved across loops)
    movq    %rsi, %r13  // r13 = byte_len (arg2)
    shrq    $6,   %r13  // r13 = byte_len / 64 (pair count)
    movq    %rdi, %r12  // r12 = read ptr (starts at buf)
    movq    %rdi, %r14  // r14 = write ptr (starts at buf)

    // reserve stack
    subq    $FRAMESZ, %rsp
    andq    $-32, %rsp

    /*
     * Level-0 marker used to differentiate
     * Aos/SoA compute paths
     * only Level-0 is Aos
     * every subsequent level reads SoA
     */
    movq    $1, IS_LEVEL0(%rsp)

.Llevel_loop:
    cmpq    %r15, %r13        // compare n_pairs(r13)/out_pair_count(r15)
    jbe     .Lexit_convert    // n_pairs <= out_pair_count, done

    /*
     * Reset read/write ptrs to the buffer base
     * for this level, save the pair count for later
     */
    movq   %rbx, %r12   // r12 = read ptr  (back to buf start)
    movq   %rbx, %r14   // r14 = write ptr (back to buf start)

    movq   %r13, LEVEL_NPAIRS(%rsp)  // save n_pairs for this level

    cmpq    $0, IS_LEVEL0(%rsp)
    je      .Lbatch_loop_soa         // if not level 0 -> SoA path

.Lbatch_loop_l0:
    cmpq    $8, %r13              // enough pairs for full batch?
    jb      .Lbatch_tail_l0       // less than 8 -> go to tail

    /*
     * Save loop state before batch processing
     * because the batch processing needs all availablle registers
     * for SHA256 states and intermediates. Restore after batch completion
     */
    movq    %r12, SAVE_R12(%rsp)  // save read ptr
    movq    %r13, SAVE_R13(%rsp)  // save n_pairs
    movq    %r14, SAVE_R14(%rsp)  // save write ptr - 0x310(%rsp)

    LOAD_AOS_FIRST8(%r12)                        /* load first 32B of each of 8 pairs */
    TRANSPOSE8(A, B, C, D, E, F, G, H, T1, TMP0) /* AoS -> SoA (dwords 0-7) */
    BYTESWAP_AH                                  /* byte-swap LE -> BE for SHA256 */
    STORE_W_FIRST8                               /* W[0-7] from A-H to stack */

    LOAD_AOS_SECOND8(%r12)                       /* load second 32B of each of 8 pairs */
    TRANSPOSE8(A, B, C, D, E, F, G, H, T1, TMP0) /* AoS -> SoA (dwords 8-15) */
    BYTESWAP_AH                                  /* byte-swap LE -> BE for SHA256 */
    STORE_W_SECOND8                              /* W[8-15] from A-H to stack */

    SHA256_FULL_BLOCK

    movq    SAVE_R12(%rsp), %r12                 /* restore read ptr */
    movq    SAVE_R13(%rsp), %r13                 /* restore n_pairs */
    addq    $512, %r12                           /* advance read ptr: 8 pairs * 64 bytes */
    addq    $256, %r14                           /* advance write ptr: 8 results * 32 bytes SoA */
    subq    $8, %r13                             /* n_pairs -= 8 */

    jmp     .Lbatch_loop_l0

.Lbatch_tail_l0:
    testq  %r13, %r13    /* any pairs left? */
    jz .Llevel_done      /* no pairs = level complete */

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)  // r14 = write ptr (back to buf start)

    LOAD_AOS_TAIL(%r12, 0, %r13)       /* first 32B */
    TRANSPOSE8(A, B, C, D, E, F, G, H, T1, TMP0)
    BYTESWAP_AH
    STORE_W_FIRST8

    /*
     * A-H hold the second 32B only. The first 32B is gone
     * second LOAD_AOS_TAIL overwrote A-H.
     */
    LOAD_AOS_TAIL(%r12, 32, %r13)      /* second 32B */
    TRANSPOSE8(A, B, C, D, E, F, G, H, T1, TMP0)
    BYTESWAP_AH
    STORE_W_SECOND8

    SHA256_FULL_BLOCK

    /*
     * vmovdqu  A, 0*32(%r14) <-  0..31
     * vmovdqu  B, 1*32(%r14) <- 32..63
     * vmovdqu  C, 2*32(%r14) <- 64..95
     * vmovdqu  D, 3*32(%r14) <- 96..127
     * vmovdqu  E, 4*32(%r14) <- 128..159
     * vmovdqu  F, 5*32(%r14) <- 160..191
     * vmovdqu  G, 6*32(%r14) <- 192..223
     * vmovdqu  H, 7*32(%r14) <- 224..255
     *
     * 256 bytes of output per batch. addq $256, %r14 advances the write pointer
     * past what was just written, so the next batch writes after it
     * TODO: but Llevel_done resets r14 to buf start anyway>?
     */
    addq    $256, %r14
    jmp     .Llevel_done

/*
 * Key difference from level0 batch loop is that input is already in SoA layout
 * previous level wrote it via STORE_SOA
 * no transpose, no byteswap, no loading (already loaded)
 */
.Lbatch_loop_soa:
    cmpq    $8, %r13
    jb      .Lbatch_tail_soa

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    /*
     * Now in level 0 the next step was LOAD_AOS_FIRST8.
     * But we're in SoA territory now. The data is already vertical 8 registers
     * worth of state words sitting contiguously in memory.
     */
    DEINTERLEAVE_SOA_2BLOCK(%r12)

    SHA256_FULL_BLOCK

    movq    SAVE_R12(%rsp), %r12
    movq    SAVE_R13(%rsp), %r13
    addq    $512, %r12  // advance read ptr: 2 SoA blocks × 256 bytes
    addq    $256, %r14  // advance write ptr: 1 SoA block × 256 bytes
    subq    $8,   %r13  // n_pairs -= 8
    jmp     .Lbatch_loop_soa

.Lbatch_tail_soa:
    testq   %r13, %r13
    jz      .Llevel_done

    movq    %r12, SAVE_R12(%rsp)
    movq    %r13, SAVE_R13(%rsp)
    movq    %r14, SAVE_R14(%rsp)

    DEINTERLEAVE_SOA_1BLOCK(%r12)

    SHA256_FULL_BLOCK

    addq    $256, %r14
    jmp     .Llevel_done

.Llevel_done:
    movq    LEVEL_NPAIRS(%rsp), %r13
    shrq    $1, %r13
    movq    $0, IS_LEVEL0(%rsp)
    jmp     .Llevel_loop

.Lexit_convert:
    /*
     * Arrived here because n_pairs <= out_pair_count (level loop done).
     * If IS_LEVEL0 is still set, no level completed -> STORE_SOA was never called.
     * Buffer is still in original AoS format from the caller. Nothing to convert.
     * This happens when out_pair_count >= initial n_pairs (a legal no-op request ?)
     */
    cmpq    $0, IS_LEVEL0(%rsp)
    jne     .Lexit_done

    testq   %r13, %r13
    jz      .Lexit_root

    // Partial reduction: r13 results in SoA at %rbx -> convert to AoS
    movq    %rbx, %r12   // r12 = read ptr (SoA source)
    movq    %rbx, %r14   // r14 = write ptr (AoS destination, same buffer)

    /* LEVEL_NPAIRSS - pair count from the last level that actually ran batches
     *
     * The jump to .Lexit_convert happens BEFORE %r13 gets saved to LEVEL_NPAIRS
     * So at .Lexit_convert:
     *   %r13 = the pair count that triggered the exit
     *          (already halved by .Llevel_done)
     *
     *   LEVEL_NPAIRS on stack = the pair count from the previous level
     *          (the last one that actually produced SoA output)
     */
    movq    LEVEL_NPAIRS(%rsp), %rcx

/*
 * The conversion loop needs LEVEL_NPAIRS because
 * that tells it how many SoA results are sitting in the buffer
 */
.Lconvert_block:
    vmovdqu  0*32(%r12), A
    vmovdqu  1*32(%r12), B
    vmovdqu  2*32(%r12), C
    vmovdqu  3*32(%r12), D
    vmovdqu  4*32(%r12), E
    vmovdqu  5*32(%r12), F
    vmovdqu  6*32(%r12), G
    vmovdqu  7*32(%r12), H

    TRANSPOSE8(A, B, C, D, E, F, G, H, T1, TMP0)
    BYTESWAP_AH

    vmovdqu  A, 0*32(%r14)
    vmovdqu  B, 1*32(%r14)
    vmovdqu  C, 2*32(%r14)
    vmovdqu  D, 3*32(%r14)
    vmovdqu  E, 4*32(%r14)
    vmovdqu  F, 5*32(%r14)
    vmovdqu  G, 6*32(%r14)
    vmovdqu  H, 7*32(%r14)

    addq    $256, %r12
    addq    $256, %r14
    subq    $8, %rcx
    ja      .Lconvert_block
    jmp     .Lexit_done

/*
 * The root's 8 state words are spread across the SoA buffer
 * one DWORD at position 0 of each row
 *
 * Need to extract A0, B0, C0, D0, E0, F0, G0, H0, byteswap them,
 * and write 32 contiguous bytes
 *
 * The *64 vs *32 is because x16's STORE_SOA writes ZMMs
 *                (64 bytes per row, 16 lanes × 4 bytes)
 * x8's STORE_SOA writes YMMs (32 bytes per row, 8 lanes × 4 bytes).
 * Different row spacing, same extraction:
 *          "grab the first dword from each row"
 *
 * Not "two results in one"
 * ⮑ the x16 ZMM holds 16 lanes (one dword per lane),
 *     the x8 YMM holds 8 lanes  (Same structure, wider register)
 */
.Lexit_root:
    // Full reduction (r13=0): single root in lane 0 of SoA
    movl  0*32(%rbx), %eax;  bswapl %eax;  movl %eax, 0*4(%rbx)
    movl  1*32(%rbx), %eax;  bswapl %eax;  movl %eax, 1*4(%rbx)
    movl  2*32(%rbx), %eax;  bswapl %eax;  movl %eax, 2*4(%rbx)
    movl  3*32(%rbx), %eax;  bswapl %eax;  movl %eax, 3*4(%rbx)
    movl  4*32(%rbx), %eax;  bswapl %eax;  movl %eax, 4*4(%rbx)
    movl  5*32(%rbx), %eax;  bswapl %eax;  movl %eax, 5*4(%rbx)
    movl  6*32(%rbx), %eax;  bswapl %eax;  movl %eax, 6*4(%rbx)
    movl  7*32(%rbx), %eax;  bswapl %eax;  movl %eax, 7*4(%rbx)
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

.size pairwise_reduce_buffer_x8_avx2, .- pairwise_reduce_buffer_x8_avx2

.section .rodata,"a",@progbits
#include "sha256_commp_rodata.h"
.section .note.GNU-stack,"",@progbits