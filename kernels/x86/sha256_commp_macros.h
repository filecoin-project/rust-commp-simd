/*
 * sha256_commp_macros.h - shared macros for x86 AVX512 CommP kernels
 *
 * Requires: sha256_reg_aliases.h included before this file.
 * Requires: sha256_commp_rodata.h in .rodata section (for .LS_* labels).
 *
 * Usage:
 *   #include "sha256_reg_aliases.h"
 *   #include "sha256_commp_macros.h"
 */

#ifndef SHA256_COMMP_MACROS_H
#define SHA256_COMMP_MACROS_H

/* ---- PROCESS_LOOP_CONST: one SHA256 round with precomputed KW2 ----
 * TMP3 must hold current round's K+W on entry.
 * Prefetches next round's K+W from .LS_KW2 into TMP3. */
#define PROCESS_LOOP_CONST(_ROUND, _A, _B, _C, _D, _E, _F, _G, _H) \
    vpaddd      TMP3, _H, T1                                        ;\
    vmovdqa32   _E, TMP0                                            ;\
    vprord      $6,  _E, TMP1                                       ;\
    vprord      $11, _E, TMP2                                       ;\
    vprord      $25, _E, TMP3                                       ;\
    vpternlogd  $0xCA, _G, _F, TMP0                                 ;\
    vpternlogd  $0x96, TMP3, TMP2, TMP1                             ;\
    vpaddd      TMP0, T1, T1                                        ;\
    vpaddd      TMP1, T1, T1                                        ;\
    vpaddd      T1, _D, _D                                          ;\
    vprord      $2,  _A, _H                                         ;\
    vprord      $13, _A, TMP2                                       ;\
    vprord      $22, _A, TMP3                                       ;\
    vmovdqa32   _A, TMP0                                            ;\
    vpternlogd  $0xE8, _C, _B, TMP0                                 ;\
    vpternlogd  $0x96, TMP3, TMP2, _H                               ;\
    vpaddd      TMP0, _H, _H                                        ;\
    vpaddd      T1, _H, _H                                          ;\
    vpbroadcastd ((_ROUND)+1)*4(TBL), TMP3

/* ---- SHA_ROUNDS_KW2: full 64 rounds with precomputed KW2 constants ----
 * State in A-H on entry. W0-W7 hold saved state for add-back.
 * Primes TMP3 with .LS_KW2[0] then unrolls all 64 rounds. */
#define SHA_ROUNDS_KW2 \
    leaq .LS_KW2(%rip), TBL; vpbroadcastd 0(TBL), TMP3 ;\
    PROCESS_LOOP_CONST( 0, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST( 1, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST( 2, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST( 3, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST( 4, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST( 5, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST( 6, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST( 7, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST( 8, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST( 9, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(10, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(11, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(12, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(13, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(14, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(15, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(16, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(17, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(18, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(19, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(20, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(21, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(22, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(23, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(24, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(25, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(26, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(27, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(28, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(29, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(30, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(31, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(32, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(33, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(34, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(35, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(36, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(37, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(38, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(39, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(40, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(41, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(42, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(43, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(44, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(45, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(46, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(47, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(48, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(49, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(50, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(51, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(52, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(53, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(54, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(55, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP_CONST(56, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP_CONST(57, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP_CONST(58, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP_CONST(59, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP_CONST(60, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP_CONST(61, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP_CONST(62, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP_CONST(63, B,C,D,E,F,G,H,A)

/* ---- STORE_IV_STATE: broadcast SHA256 IV into stack slots [0..7]*64 ---- */
#define STORE_IV_STATE \
    leaq .LS_IV(%rip), %rax; vbroadcastss 0(%rax), %zmm0;  vmovdqa32 %zmm0, 0*64(%rsp) ;\
    vbroadcastss 4(%rax), %zmm0;  vmovdqa32 %zmm0, 1*64(%rsp) ;\
    vbroadcastss 8(%rax), %zmm0;  vmovdqa32 %zmm0, 2*64(%rsp) ;\
    vbroadcastss 12(%rax), %zmm0;  vmovdqa32 %zmm0, 3*64(%rsp) ;\
    vbroadcastss 16(%rax), %zmm0;  vmovdqa32 %zmm0, 4*64(%rsp) ;\
    vbroadcastss 20(%rax), %zmm0;  vmovdqa32 %zmm0, 5*64(%rsp) ;\
    vbroadcastss 24(%rax), %zmm0;  vmovdqa32 %zmm0, 6*64(%rsp) ;\
    vbroadcastss 28(%rax), %zmm0;  vmovdqa32 %zmm0, 7*64(%rsp)
#define LOAD_IV_STACK STORE_IV_STATE

/* ---- LOAD_STATE_STACK: reload A-H from stack slots [0..7]*64 ---- */
#define LOAD_STATE_STACK \
    vmovdqa32  0*64(%rsp), A ;\
    vmovdqa32  1*64(%rsp), B ;\
    vmovdqa32  2*64(%rsp), C ;\
    vmovdqa32  3*64(%rsp), D ;\
    vmovdqa32  4*64(%rsp), E ;\
    vmovdqa32  5*64(%rsp), F ;\
    vmovdqa32  6*64(%rsp), G ;\
    vmovdqa32  7*64(%rsp), H

/* ---- SAVE_STATE_W: copy A-H into W0-W7 for Merkle-Damgård add-back ---- */
#define SAVE_STATE_W \
    vmovdqa32  A, W0 ;\
    vmovdqa32  B, W1 ;\
    vmovdqa32  C, W2 ;\
    vmovdqa32  D, W3 ;\
    vmovdqa32  E, W4 ;\
    vmovdqa32  F, W5 ;\
    vmovdqa32  G, W6 ;\
    vmovdqa32  H, W7

/* ---- ADDBACK_W: A-H += W0-W7 (Merkle-Damgård feed-forward) ---- */
#define ADDBACK_W \
    vpaddd  W0, A, A ;\
    vpaddd  W1, B, B ;\
    vpaddd  W2, C, C ;\
    vpaddd  W3, D, D ;\
    vpaddd  W4, E, E ;\
    vpaddd  W5, F, F ;\
    vpaddd  W6, G, G ;\
    vpaddd  W7, H, H

/* ---- TRUNC254: clear top 2 bits of H (byte 31 of each 32B result) ---- */
#define TRUNC254 \
    leaq .LS_TRUNC254(%rip), %rax; vpbroadcastd (%rax), TMP0 ;\
    vpandd  TMP0, H, H

/* ---- SCATTER_PTRS_RBX: write 16 pointers at stride 64 from %rbx ----
 * Stores into stack at offset 512 + lane*8 (sha256_x16_avx512 data_ptr layout). */
#define SCATTER_PTRS_RBX \
    leaq    0*64(%rbx), %rax;   movq %rax, 512+ 0*8(%rsp) ;\
    leaq    1*64(%rbx), %rax;   movq %rax, 512+ 1*8(%rsp) ;\
    leaq    2*64(%rbx), %rax;   movq %rax, 512+ 2*8(%rsp) ;\
    leaq    3*64(%rbx), %rax;   movq %rax, 512+ 3*8(%rsp) ;\
    leaq    4*64(%rbx), %rax;   movq %rax, 512+ 4*8(%rsp) ;\
    leaq    5*64(%rbx), %rax;   movq %rax, 512+ 5*8(%rsp) ;\
    leaq    6*64(%rbx), %rax;   movq %rax, 512+ 6*8(%rsp) ;\
    leaq    7*64(%rbx), %rax;   movq %rax, 512+ 7*8(%rsp) ;\
    leaq    8*64(%rbx), %rax;   movq %rax, 512+ 8*8(%rsp) ;\
    leaq    9*64(%rbx), %rax;   movq %rax, 512+ 9*8(%rsp) ;\
    leaq   10*64(%rbx), %rax;   movq %rax, 512+10*8(%rsp) ;\
    leaq   11*64(%rbx), %rax;   movq %rax, 512+11*8(%rsp) ;\
    leaq   12*64(%rbx), %rax;   movq %rax, 512+12*8(%rsp) ;\
    leaq   13*64(%rbx), %rax;   movq %rax, 512+13*8(%rsp) ;\
    leaq   14*64(%rbx), %rax;   movq %rax, 512+14*8(%rsp) ;\
    leaq   15*64(%rbx), %rax;   movq %rax, 512+15*8(%rsp)

/* ---- SCATTER_PTRS(base): parameterized version of SCATTER_PTRS_RBX ---- */
#define SCATTER_PTRS(base) \
    leaq   0*64(base), %rax;  movq %rax, 512+ 0*8(%rsp) ;\
    leaq   1*64(base), %rax;  movq %rax, 512+ 1*8(%rsp) ;\
    leaq   2*64(base), %rax;  movq %rax, 512+ 2*8(%rsp) ;\
    leaq   3*64(base), %rax;  movq %rax, 512+ 3*8(%rsp) ;\
    leaq   4*64(base), %rax;  movq %rax, 512+ 4*8(%rsp) ;\
    leaq   5*64(base), %rax;  movq %rax, 512+ 5*8(%rsp) ;\
    leaq   6*64(base), %rax;  movq %rax, 512+ 6*8(%rsp) ;\
    leaq   7*64(base), %rax;  movq %rax, 512+ 7*8(%rsp) ;\
    leaq   8*64(base), %rax;  movq %rax, 512+ 8*8(%rsp) ;\
    leaq   9*64(base), %rax;  movq %rax, 512+ 9*8(%rsp) ;\
    leaq  10*64(base), %rax;  movq %rax, 512+10*8(%rsp) ;\
    leaq  11*64(base), %rax;  movq %rax, 512+11*8(%rsp) ;\
    leaq  12*64(base), %rax;  movq %rax, 512+12*8(%rsp) ;\
    leaq  13*64(base), %rax;  movq %rax, 512+13*8(%rsp) ;\
    leaq  14*64(base), %rax;  movq %rax, 512+14*8(%rsp) ;\
    leaq  15*64(base), %rax;  movq %rax, 512+15*8(%rsp)

/* ---- LANE_REARRANGE: deinterleave 16 results into 8 left + 8 right ----
 * Input: A-H (SoA digest, 16 lanes each)
 * Output: W0-W7 = even lanes (left nodes), W8-W15 = odd lanes (right nodes) */
#define LANE_REARRANGE \
    leaq .LS_EVEN_IDX(%rip), %rax; vmovdqa32 (%rax), TMP4 ;\
    leaq .LS_ODD_IDX(%rip), %rax; vmovdqa32 (%rax), TMP5 ;\
    vpermd  A, TMP4, W0 ;  vpermd  A, TMP5, W8  ;\
    vpermd  B, TMP4, W1 ;  vpermd  B, TMP5, W9  ;\
    vpermd  C, TMP4, W2 ;  vpermd  C, TMP5, W10 ;\
    vpermd  D, TMP4, W3 ;  vpermd  D, TMP5, W11 ;\
    vpermd  E, TMP4, W4 ;  vpermd  E, TMP5, W12 ;\
    vpermd  F, TMP4, W5 ;  vpermd  F, TMP5, W13 ;\
    vpermd  G, TMP4, W6 ;  vpermd  G, TMP5, W14 ;\
    vpermd  H, TMP4, W7 ;  vpermd  H, TMP5, W15

/* ---- Stack frame offsets for callee-saved GPRs ---- */
#define SAVE_R12  640
#define SAVE_R13  648
#define SAVE_R14  656

/* ---- LOAD_IV_REGS: broadcast SHA256 IV into A-H registers ---- */
#define LOAD_IV_REGS \
    leaq .LS_IV(%rip), %rax; vbroadcastss 0(%rax), A ;\
    vbroadcastss 4(%rax), B ;\
    vbroadcastss 8(%rax), C ;\
    vbroadcastss 12(%rax), D ;\
    vbroadcastss 16(%rax), E ;\
    vbroadcastss 20(%rax), F ;\
    vbroadcastss 24(%rax), G ;\
    vbroadcastss 28(%rax), H

/* ---- SAVE_STATE_STACK: store A-H to stack slots [0..7]*64 ---- */
#define SAVE_STATE_STACK \
    vmovdqa32  A, 0*64(%rsp) ;\
    vmovdqa32  B, 1*64(%rsp) ;\
    vmovdqa32  C, 2*64(%rsp) ;\
    vmovdqa32  D, 3*64(%rsp) ;\
    vmovdqa32  E, 4*64(%rsp) ;\
    vmovdqa32  F, 5*64(%rsp) ;\
    vmovdqa32  G, 6*64(%rsp) ;\
    vmovdqa32  H, 7*64(%rsp)

/* ---- ADDBACK_STACK: A-H += stack[0..7]*64 (Merkle-Damgård from stack) ---- */
#define ADDBACK_STACK \
    vpaddd  0*64(%rsp), A, A ;\
    vpaddd  1*64(%rsp), B, B ;\
    vpaddd  2*64(%rsp), C, C ;\
    vpaddd  3*64(%rsp), D, D ;\
    vpaddd  4*64(%rsp), E, E ;\
    vpaddd  5*64(%rsp), F, F ;\
    vpaddd  6*64(%rsp), G, G ;\
    vpaddd  7*64(%rsp), H, H

/* ---- SHA_ROUNDS_BLOCK1: full 64-round Block 1 with message schedule ----
 * Requires: sha256_round_avx512.h (PROCESS_LOOP, MSG_SCHED_ROUND_16_63).
 * W0-W15 must hold transposed message words on entry. */
#define SHA_ROUNDS_BLOCK1 \
    leaq    .LS_K256(%rip), TBL ;\
    vpbroadcastd (TBL), TMP3 ;\
    PROCESS_LOOP(W0,  0, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W0, W1, W9, W14) ;\
    PROCESS_LOOP(W1,  1, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W1, W2, W10, W15) ;\
    PROCESS_LOOP(W2,  2, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W2, W3, W11, W0) ;\
    PROCESS_LOOP(W3,  3, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W3, W4, W12, W1) ;\
    PROCESS_LOOP(W4,  4, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W4, W5, W13, W2) ;\
    PROCESS_LOOP(W5,  5, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W5, W6, W14, W3) ;\
    PROCESS_LOOP(W6,  6, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W6, W7, W15, W4) ;\
    PROCESS_LOOP(W7,  7, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W7, W8, W0, W5) ;\
    PROCESS_LOOP(W8,  8, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W8, W9, W1, W6) ;\
    PROCESS_LOOP(W9,  9, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W9, W10, W2, W7) ;\
    PROCESS_LOOP(W10, 10, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W10, W11, W3, W8) ;\
    PROCESS_LOOP(W11, 11, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W11, W12, W4, W9) ;\
    PROCESS_LOOP(W12, 12, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W12, W13, W5, W10) ;\
    PROCESS_LOOP(W13, 13, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W13, W14, W6, W11) ;\
    PROCESS_LOOP(W14, 14, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W14, W15, W7, W12) ;\
    PROCESS_LOOP(W15, 15, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W15, W0, W8, W13) ;\
    PROCESS_LOOP(W0, 16, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W0, W1, W9, W14) ;\
    PROCESS_LOOP(W1, 17, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W1, W2, W10, W15) ;\
    PROCESS_LOOP(W2, 18, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W2, W3, W11, W0) ;\
    PROCESS_LOOP(W3, 19, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W3, W4, W12, W1) ;\
    PROCESS_LOOP(W4, 20, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W4, W5, W13, W2) ;\
    PROCESS_LOOP(W5, 21, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W5, W6, W14, W3) ;\
    PROCESS_LOOP(W6, 22, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W6, W7, W15, W4) ;\
    PROCESS_LOOP(W7, 23, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W7, W8, W0, W5) ;\
    PROCESS_LOOP(W8, 24, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W8, W9, W1, W6) ;\
    PROCESS_LOOP(W9, 25, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W9, W10, W2, W7) ;\
    PROCESS_LOOP(W10, 26, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W10, W11, W3, W8) ;\
    PROCESS_LOOP(W11, 27, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W11, W12, W4, W9) ;\
    PROCESS_LOOP(W12, 28, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W12, W13, W5, W10) ;\
    PROCESS_LOOP(W13, 29, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W13, W14, W6, W11) ;\
    PROCESS_LOOP(W14, 30, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W14, W15, W7, W12) ;\
    PROCESS_LOOP(W15, 31, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W15, W0, W8, W13) ;\
    PROCESS_LOOP(W0, 32, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W0, W1, W9, W14) ;\
    PROCESS_LOOP(W1, 33, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W1, W2, W10, W15) ;\
    PROCESS_LOOP(W2, 34, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W2, W3, W11, W0) ;\
    PROCESS_LOOP(W3, 35, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W3, W4, W12, W1) ;\
    PROCESS_LOOP(W4, 36, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W4, W5, W13, W2) ;\
    PROCESS_LOOP(W5, 37, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W5, W6, W14, W3) ;\
    PROCESS_LOOP(W6, 38, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W6, W7, W15, W4) ;\
    PROCESS_LOOP(W7, 39, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W7, W8, W0, W5) ;\
    PROCESS_LOOP(W8, 40, A,B,C,D,E,F,G,H) ;\
    MSG_SCHED_ROUND_16_63(W8, W9, W1, W6) ;\
    PROCESS_LOOP(W9, 41, H,A,B,C,D,E,F,G) ;\
    MSG_SCHED_ROUND_16_63(W9, W10, W2, W7) ;\
    PROCESS_LOOP(W10, 42, G,H,A,B,C,D,E,F) ;\
    MSG_SCHED_ROUND_16_63(W10, W11, W3, W8) ;\
    PROCESS_LOOP(W11, 43, F,G,H,A,B,C,D,E) ;\
    MSG_SCHED_ROUND_16_63(W11, W12, W4, W9) ;\
    PROCESS_LOOP(W12, 44, E,F,G,H,A,B,C,D) ;\
    MSG_SCHED_ROUND_16_63(W12, W13, W5, W10) ;\
    PROCESS_LOOP(W13, 45, D,E,F,G,H,A,B,C) ;\
    MSG_SCHED_ROUND_16_63(W13, W14, W6, W11) ;\
    PROCESS_LOOP(W14, 46, C,D,E,F,G,H,A,B) ;\
    MSG_SCHED_ROUND_16_63(W14, W15, W7, W12) ;\
    PROCESS_LOOP(W15, 47, B,C,D,E,F,G,H,A) ;\
    MSG_SCHED_ROUND_16_63(W15, W0, W8, W13) ;\
    PROCESS_LOOP(W0, 48, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP(W1, 49, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP(W2, 50, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP(W3, 51, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP(W4, 52, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP(W5, 53, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP(W6, 54, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP(W7, 55, B,C,D,E,F,G,H,A) ;\
    PROCESS_LOOP(W8, 56, A,B,C,D,E,F,G,H) ;\
    PROCESS_LOOP(W9, 57, H,A,B,C,D,E,F,G) ;\
    PROCESS_LOOP(W10, 58, G,H,A,B,C,D,E,F) ;\
    PROCESS_LOOP(W11, 59, F,G,H,A,B,C,D,E) ;\
    PROCESS_LOOP(W12, 60, E,F,G,H,A,B,C,D) ;\
    PROCESS_LOOP(W13, 61, D,E,F,G,H,A,B,C) ;\
    PROCESS_LOOP(W14, 62, C,D,E,F,G,H,A,B) ;\
    PROCESS_LOOP(W15, 63, B,C,D,E,F,G,H,A)

#endif /* SHA256_COMMP_MACROS_H */