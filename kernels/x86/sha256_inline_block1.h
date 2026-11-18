/* sha256_inline_block1.h - inline scattered-load + SHA256 block 1
 *
 * Replaces `call sha256_x16_avx512` for single-block processing.
 * Uses only r9-r12 as scratch (r13-r15, rbp, rbx preserved).
 *
 * Requires: sha256_reg_aliases.h, sha256_round_avx512.h, transpose16_avx512.h,
 *           sha256_commp_macros.h, sha256_commp_rodata.h
 *
 * Preconditions:
 *   - Digest IV already in stack slots 0-7*64(%rsp) via LOAD_IV_STACK
 *   - Data pointers at 512+lane*8(%rsp) via SCATTER_PTRS
 *   - k1 set to lane mask (0xFFFF for batch, partial for tail)
 *
 * Postcondition:
 *   - Block-1 result in stack slots 0-7*64(%rsp)
 *   - A-H contain the result (loaded from stack)
 */

#ifndef SHA256_INLINE_BLOCK1_H
#define SHA256_INLINE_BLOCK1_H

/* ---- LOAD_SCATTERED_W: load 16 scattered 64B blocks into W0-W15 ----
 * Reads pointers from 512+lane*8(base), loads 32B halves,
 * uses vinserti64x4 for lanes 8-15 upper halves.
 * Clobbers: r9, r10, r11, r12 only. */
#define LOAD_SCATTERED_W(base) \
    /* Lanes 0-3: first 32B -> YW0-YW3, second 32B -> YW8-YW11 */ \
    movq 512+0*8(base), %r9  ;\
    movq 512+1*8(base), %r10 ;\
    movq 512+2*8(base), %r11 ;\
    movq 512+3*8(base), %r12 ;\
    vmovups (%r9),  YW0  ;\
    vmovups (%r10), YW1  ;\
    vmovups (%r11), YW2  ;\
    vmovups (%r12), YW3  ;\
    vmovups 0x20(%r9),  YW8  ;\
    vmovups 0x20(%r10), YW9  ;\
    vmovups 0x20(%r11), YW10 ;\
    vmovups 0x20(%r12), YW11 ;\
    /* Lanes 4-7 */ \
    movq 512+4*8(base), %r9  ;\
    movq 512+5*8(base), %r10 ;\
    movq 512+6*8(base), %r11 ;\
    movq 512+7*8(base), %r12 ;\
    vmovups (%r9),  YW4  ;\
    vmovups (%r10), YW5  ;\
    vmovups (%r11), YW6  ;\
    vmovups (%r12), YW7  ;\
    vmovups 0x20(%r9),  YW12 ;\
    vmovups 0x20(%r10), YW13 ;\
    vmovups 0x20(%r11), YW14 ;\
    vmovups 0x20(%r12), YW15 ;\
    /* Lanes 8-11: insert upper 256 bits */ \
    movq 512+8*8(base),  %r9  ;\
    movq 512+9*8(base),  %r10 ;\
    movq 512+10*8(base), %r11 ;\
    movq 512+11*8(base), %r12 ;\
    vinserti64x4 $1, (%r9),  W0, W0   ;\
    vinserti64x4 $1, (%r10), W1, W1   ;\
    vinserti64x4 $1, (%r11), W2, W2   ;\
    vinserti64x4 $1, (%r12), W3, W3   ;\
    vinserti64x4 $1, 0x20(%r9),  W8,  W8   ;\
    vinserti64x4 $1, 0x20(%r10), W9,  W9   ;\
    vinserti64x4 $1, 0x20(%r11), W10, W10  ;\
    vinserti64x4 $1, 0x20(%r12), W11, W11  ;\
    /* Lanes 12-15 */ \
    movq 512+12*8(base), %r9  ;\
    movq 512+13*8(base), %r10 ;\
    movq 512+14*8(base), %r11 ;\
    movq 512+15*8(base), %r12 ;\
    vinserti64x4 $1, (%r9),  W4, W4   ;\
    vinserti64x4 $1, (%r10), W5, W5   ;\
    vinserti64x4 $1, (%r11), W6, W6   ;\
    vinserti64x4 $1, (%r12), W7, W7   ;\
    vinserti64x4 $1, 0x20(%r9),  W12, W12  ;\
    vinserti64x4 $1, 0x20(%r10), W13, W13  ;\
    vinserti64x4 $1, 0x20(%r11), W14, W14  ;\
    vinserti64x4 $1, 0x20(%r12), W15, W15

/* ---- BYTESWAP_W: byte-swap W0-W15 from LE to BE ---- */
#define BYTESWAP_W \
    leaq .LS_BYTESWAP(%rip), %rax; vbroadcasti32x4 (%rax), TMP2 ;\
    vpshufb TMP2, W0,  W0  ;\
    vpshufb TMP2, W1,  W1  ;\
    vpshufb TMP2, W2,  W2  ;\
    vpshufb TMP2, W3,  W3  ;\
    vpshufb TMP2, W4,  W4  ;\
    vpshufb TMP2, W5,  W5  ;\
    vpshufb TMP2, W6,  W6  ;\
    vpshufb TMP2, W7,  W7  ;\
    vpshufb TMP2, W8,  W8  ;\
    vpshufb TMP2, W9,  W9  ;\
    vpshufb TMP2, W10, W10 ;\
    vpshufb TMP2, W11, W11 ;\
    vpshufb TMP2, W12, W12 ;\
    vpshufb TMP2, W13, W13 ;\
    vpshufb TMP2, W14, W14 ;\
    vpshufb TMP2, W15, W15

/* ---- INLINE_SHA256_BLOCK1: full block-1 processing, no call ----
 * Precondition: SCATTER_PTRS + LOAD_IV_STACK already done.
 * Postcondition: block-1 result in stack slots and A-H regs.
 * Clobbers: r9-r12, all ZMM regs, TBL (%rdx), IDX (%rcx). */
#define INLINE_SHA256_BLOCK1 \
    LOAD_STATE_STACK                                                    ;\
    LOAD_SCATTERED_W(%rsp)                                              ;\
    TRANSPOSE16(W0,W1,W2,W3,W4,W5,W6,W7,                              \
                W8,W9,W10,W11,W12,W13,W14,W15,                         \
                TMP0,TMP1,TMP4,TMP5)                                    ;\
    BYTESWAP_W                                                          ;\
    SAVE_STATE_STACK                                                    ;\
    SHA_ROUNDS_BLOCK1                                                   ;\
    ADDBACK_STACK                                                       ;\
    SAVE_STATE_STACK

#endif /* SHA256_INLINE_BLOCK1_H */
