/* ---- SHA_ROUNDS_BLOCK1: full 64-round Block 1 for x8 AVX2 ----
 *
 * Precondition: W[0..15] stored in stack slots 0..15 (X8_SLOT * i).
 *   State A-H must hold initial IV (or previous block result for chained calls).
 *
 * For rounds 0-15: load W from stack, call PROCESS_LOOP.
 * For rounds 16-63: MSG_SCHED_ROUND_16_63 loads W from stack, computes new W,
 *                   leaves it in T1, then PROCESS_LOOP consumes T1.
 *
 * Depends on:  PROCESS_LOOP, MSG_SCHED_ROUND_16_63, PRORD (in sha256_round_avx2.h)
 *
 * Clobbers: all 16 YMMs, stack slots 0..1E0 (W area), TBL (%rdx) if TBL load inside.
 *
 * x8 version difference to x16:
 *   For each round we:
 *     - Load W from stack -> REG
 *       (for rounds 0-15, W is already there from initial transpose)
 *     - Pass REG -> PROCESS_LOOP
 */
#define SHA_ROUNDS_BLOCK1 \
    /* (Round 0) loads W[0] from slot 0 (stack y) */ \
    leaq    .LS_K256(%rip), TBL            ;  /* TBL = K table */ \
    /* Rounds 0-15: W already on stack, load and run PROCESS_LOOP */ \
    vmovdqa 0*X8_SLOT(%rsp), T1            ;  /* (Round 0) loads  W[0]  from slot 0 */ \
    PROCESS_LOOP(T1, 0, A,B,C,D,E,F,G,H)   ;  \
    vmovdqa 1*X8_SLOT(%rsp), T1            ;  /* (Round 1) loads  W[1]  from slot 1 */ \
    PROCESS_LOOP(T1, 1, H,A,B,C,D,E,F,G)   ;  \
    vmovdqa 2*X8_SLOT(%rsp), T1            ;  /* (Round 2) loads  W[2]  from slot 2 */ \
    PROCESS_LOOP(T1, 2, G,H,A,B,C,D,E,F)   ;  \
    vmovdqa 3*X8_SLOT(%rsp), T1            ;  /* (Round 3) loads  W[3]  from slot 3 */ \
    PROCESS_LOOP(T1, 3, F,G,H,A,B,C,D,E)   ;  \
    vmovdqa 4*X8_SLOT(%rsp), T1            ;  /* (Round 4) loads  W[4]  from slot 4 */ \
    PROCESS_LOOP(T1, 4, E,F,G,H,A,B,C,D)   ;  \
    vmovdqa 5*X8_SLOT(%rsp), T1            ;  /* (Round 5) loads  W[5]  from slot 5 */ \
    PROCESS_LOOP(T1, 5, D,E,F,G,H,A,B,C)   ;  \
    vmovdqa 6*X8_SLOT(%rsp), T1            ;  /* (Round 6) loads  W[6]  from slot 6 */ \
    PROCESS_LOOP(T1, 6, C,D,E,F,G,H,A,B)   ;  \
    vmovdqa 7*X8_SLOT(%rsp), T1            ;  /* (Round 7) loads  W[7]  from slot 7 */ \
    PROCESS_LOOP(T1, 7, B,C,D,E,F,G,H,A)   ;  \
    vmovdqa 8*X8_SLOT(%rsp), T1            ;  /* (Round 8) loads  W[8]  from slot 8 */ \
    PROCESS_LOOP(T1, 8, A,B,C,D,E,F,G,H)   ;  \
    vmovdqa 9*X8_SLOT(%rsp), T1            ;  /* (Round 9) loads  W[9]  from slot 9 */ \
    PROCESS_LOOP(T1, 9, H,A,B,C,D,E,F,G)   ;  \
    vmovdqa 10*X8_SLOT(%rsp), T1           ;  /* (Round 10) loads W[10] from slot 10 */ \
    PROCESS_LOOP(T1, 10, G,H,A,B,C,D,E,F)  ;  \
    vmovdqa 11*X8_SLOT(%rsp), T1           ;  /* (Round 11) loads W[11] from slot 11 */ \
    PROCESS_LOOP(T1, 11, F,G,H,A,B,C,D,E)  ;  \
    vmovdqa 12*X8_SLOT(%rsp), T1           ;  /* (Round 12) loads W[12] from slot 12 */ \
    PROCESS_LOOP(T1, 12, E,F,G,H,A,B,C,D)  ;  \
    vmovdqa 13*X8_SLOT(%rsp), T1           ;  /* (Round 13) loads W[13] from slot 13 */ \
    PROCESS_LOOP(T1, 13, D,E,F,G,H,A,B,C)  ;  \
    vmovdqa 14*X8_SLOT(%rsp), T1           ;  /* (Round 14) loads W[14] from slot 14 */ \
    PROCESS_LOOP(T1, 14, C,D,E,F,G,H,A,B)  ;  \
    vmovdqa 15*X8_SLOT(%rsp), T1           ;  /* (Round 15) loads W[15] from slot 15 */ \
    PROCESS_LOOP(T1, 15, B,C,D,E,F,G,H,A)  ;  \
    /* Rounds 16-63. MSG_SCHED_ROUND_16_63 computes W directly into T1                  \
     * No more stack loads -> then PROCESS_LOOP consumes it.                            \
     */                                                                                 \
    MSG_SCHED_ROUND_16_63(T1, 16)           ;  /* (Round 16) W[16] from W[0,1,9,14] */ \
    PROCESS_LOOP(T1, 16, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 17)           ;  /* (Round 17) W[17] from W[1,2,10,15] */ \
    PROCESS_LOOP(T1, 17, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 18)           ;  /* (Round 18) W[18] from W[2,3,11,0] */ \
    PROCESS_LOOP(T1, 18, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 19)           ;  /* (Round 19) W[19] from W[3,4,12,1] */ \
    PROCESS_LOOP(T1, 19, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 20)           ;  /* (Round 20) W[20] from W[4,5,13,2] */ \
    PROCESS_LOOP(T1, 20, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 21)           ;  /* (Round 21) W[21] from W[5,6,14,3] */ \
    PROCESS_LOOP(T1, 21, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 22)           ;  /* (Round 22) W[22] from W[6,7,15,4] */ \
    PROCESS_LOOP(T1, 22, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 23)           ;  /* (Round 23) W[23] from W[7,8,0,5] */ \
    PROCESS_LOOP(T1, 23, B,C,D,E,F,G,H,A)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 24)           ;  /* (Round 24) W[24] from W[8,9,1,6] */ \
    PROCESS_LOOP(T1, 24, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 25)           ;  /* (Round 25) W[25] from W[9,10,2,7] */ \
    PROCESS_LOOP(T1, 25, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 26)           ;  /* (Round 26) W[26] from W[10,11,3,8] */ \
    PROCESS_LOOP(T1, 26, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 27)           ;  /* (Round 27) W[27] from W[11,12,4,9] */ \
    PROCESS_LOOP(T1, 27, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 28)           ;  /* (Round 28) W[28] from W[12,13,5,10] */ \
    PROCESS_LOOP(T1, 28, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 29)           ;  /* (Round 29) W[29] from W[13,14,6,11] */ \
    PROCESS_LOOP(T1, 29, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 30)           ;  /* (Round 30) W[30] from W[14,15,7,12] */ \
    PROCESS_LOOP(T1, 30, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 31)           ;  /* (Round 31) W[31] from W[15,0,8,13] */ \
    PROCESS_LOOP(T1, 31, B,C,D,E,F,G,H,A)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 32)           ;  /* (Round 32) W[32] from W[0,1,9,14] */ \
    PROCESS_LOOP(T1, 32, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 33)           ;  /* (Round 33) W[33] from W[1,2,10,15] */ \
    PROCESS_LOOP(T1, 33, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 34)           ;  /* (Round 34) W[34] from W[2,3,11,0] */ \
    PROCESS_LOOP(T1, 34, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 35)           ;  /* (Round 35) W[35] from W[3,4,12,1] */ \
    PROCESS_LOOP(T1, 35, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 36)           ;  /* (Round 36) W[36] from W[4,5,13,2] */ \
    PROCESS_LOOP(T1, 36, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 37)           ;  /* (Round 37) W[37] from W[5,6,14,3] */ \
    PROCESS_LOOP(T1, 37, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 38)           ;  /* (Round 38) W[38] from W[6,7,15,4] */ \
    PROCESS_LOOP(T1, 38, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 39)           ;  /* (Round 39) W[39] from W[7,8,0,5] */ \
    PROCESS_LOOP(T1, 39, B,C,D,E,F,G,H,A)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 40)           ;  /* (Round 40) W[40] from W[8,9,1,6] */ \
    PROCESS_LOOP(T1, 40, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 41)           ;  /* (Round 41) W[41] from W[9,10,2,7] */ \
    PROCESS_LOOP(T1, 41, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 42)           ;  /* (Round 42) W[42] from W[10,11,3,8] */ \
    PROCESS_LOOP(T1, 42, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 43)           ;  /* (Round 43) W[43] from W[11,12,4,9] */ \
    PROCESS_LOOP(T1, 43, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 44)           ;  /* (Round 44) W[44] from W[12,13,5,10] */ \
    PROCESS_LOOP(T1, 44, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 45)           ;  /* (Round 45) W[45] from W[13,14,6,11] */ \
    PROCESS_LOOP(T1, 45, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 46)           ;  /* (Round 46) W[46] from W[14,15,7,12] */ \
    PROCESS_LOOP(T1, 46, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 47)           ;  /* (Round 47) W[47] from W[15,0,8,13] */ \
    PROCESS_LOOP(T1, 47, B,C,D,E,F,G,H,A)   ;                                             \
    /* Rounds 48-63: still need MSG_SCHED                                                 \
     * (x8 computes W on demand, not 16 ahead like x16)                                   \
     * because all 16 YMM registers are already occupied,                                 \
     * no register to hold w[48] for 16 rounds while                                      \
     * waiting for round 48 to arrive                                                     \
     * x8 needs to park somewhere  (eg. w[48]) somewhere for 16 rounds                    \
     * and stack is the only option                                                       \
     */                                                                                   \
    MSG_SCHED_ROUND_16_63(T1, 48)           ;  /* (Round 48) W[48] from W[32,33,41,46] */ \
    PROCESS_LOOP(T1, 48, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 49)           ;  /* (Round 49) W[49] from W[33,34,42,47] */ \
    PROCESS_LOOP(T1, 49, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 50)           ;  /* (Round 50) W[50] from W[34,35,43,48] */ \
    PROCESS_LOOP(T1, 50, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 51)           ;  /* (Round 51) W[51] from W[35,36,44,49] */ \
    PROCESS_LOOP(T1, 51, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 52)           ;  /* (Round 52) W[52] from W[36,37,45,50] */ \
    PROCESS_LOOP(T1, 52, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 53)           ;  /* (Round 53) W[53] from W[37,38,46,51] */ \
    PROCESS_LOOP(T1, 53, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 54)           ;  /* (Round 54) W[54] from W[38,39,47,52] */ \
    PROCESS_LOOP(T1, 54, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 55)           ;  /* (Round 55) W[55] from W[39,40,48,53] */ \
    PROCESS_LOOP(T1, 55, B,C,D,E,F,G,H,A)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 56)           ;  /* (Round 56) W[56] from W[40,41,49,54] */ \
    PROCESS_LOOP(T1, 56, A,B,C,D,E,F,G,H)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 57)           ;  /* (Round 57) W[57] from W[41,42,50,55] */ \
    PROCESS_LOOP(T1, 57, H,A,B,C,D,E,F,G)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 58)           ;  /* (Round 58) W[58] from W[42,43,51,56] */ \
    PROCESS_LOOP(T1, 58, G,H,A,B,C,D,E,F)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 59)           ;  /* (Round 59) W[59] from W[43,44,52,57] */ \
    PROCESS_LOOP(T1, 59, F,G,H,A,B,C,D,E)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 60)           ;  /* (Round 60) W[60] from W[44,45,53,58] */ \
    PROCESS_LOOP(T1, 60, E,F,G,H,A,B,C,D)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 61)           ;  /* (Round 61) W[61] from W[45,46,54,59] */ \
    PROCESS_LOOP(T1, 61, D,E,F,G,H,A,B,C)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 62)           ;  /* (Round 62) W[62] from W[46,47,55,60] */ \
    PROCESS_LOOP(T1, 62, C,D,E,F,G,H,A,B)   ;  \
    MSG_SCHED_ROUND_16_63(T1, 63)           ;  /* (Round 63) W[63] from W[47,48,56,61] */ \
    PROCESS_LOOP(T1, 63, B,C,D,E,F,G,H,A)


#define LOAD_IV_REGS \
    leaq .LS_IV(%rip), %rax     ;\
    vbroadcastss  0(%rax), A    ;\
    vbroadcastss  4(%rax), B    ;\
    vbroadcastss  8(%rax), C    ;\
    vbroadcastss 12(%rax), D    ;\
    vbroadcastss 16(%rax), E    ;\
    vbroadcastss 20(%rax), F    ;\
    vbroadcastss 24(%rax), G    ;\
    vbroadcastss 28(%rax), H

/*
 * save A-H digest to stack (rsp+0x200..)
 */
#define SAVE_STATE_STACK \
    vmovdqa  A, DIGEST_SAVE+0*X8_SLOT(%rsp)     ;\
    vmovdqa  B, DIGEST_SAVE+1*X8_SLOT(%rsp)     ;\
    vmovdqa  C, DIGEST_SAVE+2*X8_SLOT(%rsp)     ;\
    vmovdqa  D, DIGEST_SAVE+3*X8_SLOT(%rsp)     ;\
    vmovdqa  E, DIGEST_SAVE+4*X8_SLOT(%rsp)     ;\
    vmovdqa  F, DIGEST_SAVE+5*X8_SLOT(%rsp)     ;\
    vmovdqa  G, DIGEST_SAVE+6*X8_SLOT(%rsp)     ;\
    vmovdqa  H, DIGEST_SAVE+7*X8_SLOT(%rsp)

/*
 * Add A-H values that SAVE_STATE_STACK stored at
 * DIGEST_SAVE (rsp+0x200..) to re-load the hot tracks
 */
#define ADDBACK_STACK \
    vpaddd  DIGEST_SAVE+0*X8_SLOT(%rsp), A, A   ;\
    vpaddd  DIGEST_SAVE+1*X8_SLOT(%rsp), B, B   ;\
    vpaddd  DIGEST_SAVE+2*X8_SLOT(%rsp), C, C   ;\
    vpaddd  DIGEST_SAVE+3*X8_SLOT(%rsp), D, D   ;\
    vpaddd  DIGEST_SAVE+4*X8_SLOT(%rsp), E, E   ;\
    vpaddd  DIGEST_SAVE+5*X8_SLOT(%rsp), F, F   ;\
    vpaddd  DIGEST_SAVE+6*X8_SLOT(%rsp), G, G   ;\
    vpaddd  DIGEST_SAVE+7*X8_SLOT(%rsp), H, H

/*
 * Clear top 2 bits of H (254-bit truncate for BLS12-381)
 * .LS_TRUNC254 = 0xFFFFFF3F broadcasts t oall 8 lanes
 * then AND with H zeroes bits 254-255 of each 32-byte result
 */
#define TRUNC_254 \
    leaq .LS_TRUNC254(%rip), %rax               ;\
    vpbroadcastd (%rax), TMP0                    ;\
    vpand  TMP0, H, H

/*
 * Store A-H to MEM in SoA layout (8 lanes x 32 bytes each)
 *
 * vmovdqu - unaligned becuase dst is the output buffer not aligned stack
 */
#define STORE_SOA(dst) \
    vmovdqu  A, 0*X8_SLOT(dst)                        ;\
    vmovdqu  B, 1*X8_SLOT(dst)                        ;\
    vmovdqu  C, 2*X8_SLOT(dst)                        ;\
    vmovdqu  D, 3*X8_SLOT(dst)                        ;\
    vmovdqu  E, 4*X8_SLOT(dst)                        ;\
    vmovdqu  F, 5*X8_SLOT(dst)                        ;\
    vmovdqu  G, 6*X8_SLOT(dst)                        ;\
    vmovdqu  H, 7*X8_SLOT(dst)

/*
 * Load first 32 bytes of each of 8 pairs into A-H (used as temp data regs)
 * Pairs are 64-byte apart
 * IV not loaded yet so A-H are free
 * each mov loads 32 bytes (dwords 0-7) from one pair
 * 2nd half will need +32 to get dwords 8-15
 */
#define LOAD_AOS_FIRST8(base) \
    vmovdqu  0*64(base), A                            ;\
    vmovdqu  1*64(base), B                            ;\
    vmovdqu  2*64(base), C                            ;\
    vmovdqu  3*64(base), D                            ;\
    vmovdqu  4*64(base), E                            ;\
    vmovdqu  5*64(base), F                            ;\
    vmovdqu  6*64(base), G                            ;\
    vmovdqu  7*64(base), H

#define LOAD_AOS_SECOND8(base) \
    vmovdqu  0*64+32(base), A                         ;\
    vmovdqu  1*64+32(base), B                         ;\
    vmovdqu  2*64+32(base), C                         ;\
    vmovdqu  3*64+32(base), D                         ;\
    vmovdqu  4*64+32(base), E                         ;\
    vmovdqu  5*64+32(base), F                         ;\
    vmovdqu  6*64+32(base), G                         ;\
    vmovdqu  7*64+32(base), H

/*
 * Byte-swap all DWORD(s) in A-H (LE -> BE)
 * vbroadcasti128 loads the 16-byte swap mask and
 * duplicates it to both 128-bit halves of TMP0.
 *
 *
 * vshufb reverses bytes wthin each DWORD across all 8 lane.
 *   .LS_BYTESWAP = .long 0x00010203 (LE: 03 02 01 00)
 *   byte position:  0  1  2  3 │ 4  5  6  7 │  8  9 10 11 │ 12 13 14 15
 *   gets from:      3  2  1  0 │ 7  6  5  4 │ 11 10  9  8 │ 15 14 13 12
 *
 */
#define BYTESWAP_AH \
    leaq .LS_BYTESWAP(%rip), %rax                     ;\
    vbroadcasti128 (%rax), TMP0                       ;\
    vpshufb TMP0, A, A                                ;\
    vpshufb TMP0, B, B                                ;\
    vpshufb TMP0, C, C                                ;\
    vpshufb TMP0, D, D                                ;\
    vpshufb TMP0, E, E                                ;\
    vpshufb TMP0, F, F                                ;\
    vpshufb TMP0, G, G                                ;\
    vpshufb TMP0, H, H

/*
 * STORE(s) transposed+byteswapped W values from A-H -> W stack slots
 * Per batch of 8 pairs, called once:
 *
 *   LOAD_AOS_FIRST8   ─┐
 *   TRANSPOSE8         │ pass 1: W[0-7] through A-H onto stack
 *   BYTESWAP           │ (A-H are used as temps to move W values onto stack)
 *   STORE_W_FIRST8    ─┘
 *
 *   LOAD_AOS_SECOND8  ─┐
 *   TRANSPOSE8         │ pass 2: W[8-15] through A-H onto stack
 *   BYTESWAP           │ (A-H are used as temps to move W values onto stack)
 *   STORE_W_SECOND8   ─┘
 *
 *   LOAD_IV_REGS        A-H become state, W[0-15] ready on stack
 *   SAVE_STATE_STACK
 *   SHA_ROUNDS_BLOCK1   64 rounds reading W from stack
 *   ADDBACK_STACK
 *   ...
 */

/* Must be saved, before LOAD_AOS_SECOND8 overwrites A-H */
#define STORE_W_FIRST8 \
    vmovdqa  A, 0*X8_SLOT(%rsp)                       ;\
    vmovdqa  B, 1*X8_SLOT(%rsp)                       ;\
    vmovdqa  C, 2*X8_SLOT(%rsp)                       ;\
    vmovdqa  D, 3*X8_SLOT(%rsp)                       ;\
    vmovdqa  E, 4*X8_SLOT(%rsp)                       ;\
    vmovdqa  F, 5*X8_SLOT(%rsp)                       ;\
    vmovdqa  G, 6*X8_SLOT(%rsp)                       ;\
    vmovdqa  H, 7*X8_SLOT(%rsp)

/* Must be saved before LOAD_IV_REGS overwrites A-H with IV */
#define STORE_W_SECOND8 \
    vmovdqa  A, 8*X8_SLOT(%rsp)                       ;\
    vmovdqa  B, 9*X8_SLOT(%rsp)                       ;\
    vmovdqa  C, 10*X8_SLOT(%rsp)                      ;\
    vmovdqa  D, 11*X8_SLOT(%rsp)                      ;\
    vmovdqa  E, 12*X8_SLOT(%rsp)                      ;\
    vmovdqa  F, 13*X8_SLOT(%rsp)                      ;\
    vmovdqa  G, 14*X8_SLOT(%rsp)                      ;\
    vmovdqa  H, 15*X8_SLOT(%rsp)

/* SHA_ROUNDS_KW2 - 64 rounds for block 2 (CommP constant padding block)
 *
 * No MSG_SCHED - all 64 KW2 values are precomputed in .LS_KW2
 * (K[i] + W[i] for the fixed padding block, baked into hex at build time)
 *
 * Round 63's PROCESS_LOOP_CONST prefetches KW2[64] which is one past
 * the table. .LS_KW2 has a safety pad dword for this:
 *   .long 0x00000000  (sha256_commp_rodata.h line 67)
 */
#define SHA_ROUNDS_KW2 \
    leaq .LS_KW2(%rip), TBL                 ;\
    vpbroadcastd 0(TBL), TMP6               ;   /* prime KW2[0] for first round */ \
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

/*
 * PROCESS_LOOP_CONST - SHA256 round with pre-comptued K+W (block 2 padding)
 *
 * KW2 values from .LS_KW2 table, broadcast per round
 * Same SIGMA/CH/MAJ as PROCESS_LOOP, just different input source
 * TMP6 preped to .LS_KW2 (KW2[_i] for all 8 lanes)
 *
 * SHA256 spec defines: T1 = H + SIGMA1(E) + CH(E,F,G) + K[t] + W[t]
 * so we begin with H and use it as an accumulator
 */
#define PROCESS_LOOP_CONST(_i, _A, _B, _C, _D, _E, _F, _G, _H) \
    vpaddd TMP6, _H, _H                 ;   /* H+ = KW[_i] */ \
    CH_SIGMA1(_E, _F, _G)               ;   /* TMP5 = CH(E,F,G), TMP3 = SIGMA1(E) */ \
    vpaddd  TMP5, _H, _H                ;   /* H += CH(E,F,G) */ \
    vpaddd  TMP3, _H, _H                ;   /* H += SIGMA1(E) */ \
    vpaddd  _H, _D, _D                  ;   /* D += H (new E = D + T1) */ \
    SIGMA0(_A)                          ;   /* TMP5 = SIGMA0(A) */ \
    MAJ(_A, _B, _C, T1)                 ;   /* TMP4 = MAJ(A,B,C) */ \
    vpaddd  TMP4, _H, _H                ;   /* H += MAJ */  \
    vpaddd  TMP5, _H, _H                ;   /* H += SIGMA0 (H is now new A) */ \
    vpbroadcastd ((_i)+1)*4(TBL), TMP6      /* prefetch KW2[_i+1] */


/* LOAD_AOS_TAIL - load 32B from up to 8 pairs, invalid lanes read pair 0
 *
 * _base:   GPR pointing to start of pairs in buffer (e.g. %r12)
 * _offset: 0 for first 32B, 32 for second 32B of each 64-byte pair
 * _npairs: GPR holding valid pair count (e.g. %r13)
 *
 * Lanes where lane_index >= _npairs read pair 0 + _offset instead
 * Clobbers: %rax, %rcx */
#define LOAD_AOS_TAIL(_base, _offset, _npairs) \
    leaq    _offset(_base), %rcx                          ;  /* pair 0 + offset (redirect target) */ \
    vmovdqu _offset(_base), A                             ;  /* lane 0: always valid */ \
    leaq    1*64+_offset(_base), %rax                     ;  /* lane 1 */ \
    cmpq    $1, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), B                                     ;  \
    leaq    2*64+_offset(_base), %rax                     ;  /* lane 2 */ \
    cmpq    $2, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), C                                     ;  \
    leaq    3*64+_offset(_base), %rax                     ;  /* lane 3 */ \
    cmpq    $3, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), D                                     ;  \
    leaq    4*64+_offset(_base), %rax                     ;  /* lane 4 */ \
    cmpq    $4, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), E                                     ;  \
    leaq    5*64+_offset(_base), %rax                     ;  /* lane 5 */ \
    cmpq    $5, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), F                                     ;  \
    leaq    6*64+_offset(_base), %rax                     ;  /* lane 6 */ \
    cmpq    $6, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), G                                     ;  \
    leaq    7*64+_offset(_base), %rax                     ;  /* lane 7 */ \
    cmpq    $7, _npairs                                   ;  \
    cmovbe  %rcx, %rax                                    ;  \
    vmovdqu (%rax), H


/* SHA256_FULL_BLOCK - IV -> block 1 -> block 2 -> trunc -> store
 *
 * Precondition: W[0-15] on stack, SAVE_R14 holds write ptr
 * This is the shared core between full batch and tail paths.
 * Only the loading (before) and looping (after) differ. */
#define SHA256_FULL_BLOCK \
    /* All 16 W values now on stack. A-H are free */                                                   \
    LOAD_IV_REGS                                  ;   /* A-H = SHA256 IV (broadcast to 8 lanes) */     \
    /*                                                                                                 \
     * SHA_ROUNDS_BLOCK1 will modify A-H across 64 rounds.                                             \
     * After those rounds, ADDBACK_STACK needs the ORIGINAL values (IV) to add back                    \
     */                                                                                                \
    SAVE_STATE_STACK                              ;                                                    \
    SHA_ROUNDS_BLOCK1                             ;   /* 64 rounds on block 1 (real input) */          \
    ADDBACK_STACK                                 ;   /* A-H += saved IV (block 1) */                  \
    SAVE_STATE_STACK                              ;   /* save block 1 result for block 2 add-back */   \
    SHA_ROUNDS_KW2                                ;   /* 64 rounds on block 2 (constant padding) */    \
    ADDBACK_STACK                                 ;   /* A-H += saved block1 (block 2) */              \
    TRUNC_254                                     ;   /* clear top 2 bits of H (254-bit truncation) */ \
    movq  SAVE_R14(%rsp), %r14                    ;   /* restore write ptr */                          \
    STORE_SOA(%r14)                                   /* write A-H in SoA to output */

/*
 * DEINTERLEAVE_SOA_2BLOCK
 *
 * base - at base address sit two consecutive 256-byte SoA groups
 *        (written by two previous STORE_SOA calls)
 *        The W stack area
 *        Same 16 slots at 0x000-0x1FF that STORE_W_FIRST8 and STORE_W_SECOND8 wrote to at level 0.
 *        That's where SHA256_FULL_BLOCK reads from
 *
 */
#define DEINTERLEAVE_SOA_2BLOCK(_base) \
    leaq .LS_EVEN_IDX(%rip), %rax; vmovdqu (%rax), TMP2   ;   /* TMP2 = even index */                     \
    leaq .LS_ODD_IDX(%rip),  %rax; vmovdqu (%rax), TMP3   ;   /* TMP3 = odd index */                      \
    /* A - Rows (W[0] and W[8]) */                                                                        \
    vmovdqu  0*32(_base), TMP0          ;  /* TMP0 = [A0 A1  A2  A3  A4  A5  A6  A7 ] block 1 */          \
    vmovdqu  0*32+256(_base), TMP1      ;  /* TMP1 = [A8 A9  A10 A11 A12 A13 A14 A15] block 2 */          \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [A0 A2  A4  A6  A0  A2  A4  A6 ] block 1: evens */   \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [A8 A10 A12 A14 A8  A10 A12 A14] block 2: evens */   \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [A0 A2  A4  A6  A8  A10 A12 A14] full evens row A */ \
    vmovdqa TMP4, 0*X8_SLOT(%rsp)       ;  /* W[0] = evens of row A */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [A1  A3  A5  A7  A1 A3  A5  A7 ] blk 1 odds  */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [A9  A11 A13 A15 A9 A11 A13 A15] blk 2 odds  */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [A1  A3  A5  A7  A9 A11 A13 A15]             */      \
    vmovdqa TMP4, 8*X8_SLOT(%rsp)       ;  /* W[8] = row A odds (rights)                          */      \
    /* B - Rows (W[1] and W[9]) */                                                                        \
    vmovdqu  1*32(_base), TMP0          ;  /* TMP0 = [B0  B1  B2  B3  B4  B5  B6  B7 ] block 1 */         \
    vmovdqu  1*32+256(_base), TMP1      ;  /* TMP1 = [B8  B9  B10 B11 B12 B13 B14 B15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [B0  B2  B4  B6  B0  B2  B4  B6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [B8  B10 B12 B14 B8  B10 B12 B14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [B0  B2  B4  B6  B8  B10 B12 B14] full evens row B */\
    vmovdqa TMP4, 1*X8_SLOT(%rsp)       ;  /* W[1] = evens of row B */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [B1  B3  B5  B7  B1  B3  B5  B7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [B9  B11 B13 B15 B9  B11 B13 B15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [B1  B3  B5  B7  B9  B11 B13 B15]        */          \
    vmovdqa TMP4, 9*X8_SLOT(%rsp)       ;  /* W[9] = row B odds (rights)                      */          \
    /* C - Rows (W[2] and W[10]) */                                                                       \
    vmovdqu  2*32(_base), TMP0          ;  /* TMP0 = [C0  C1  C2  C3  C4  C5  C6  C7 ] block 1 */         \
    vmovdqu  2*32+256(_base), TMP1      ;  /* TMP1 = [C8  C9  C10 C11 C12 C13 C14 C15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [C0  C2  C4  C6  C0  C2  C4  C6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [C8  C10 C12 C14 C8  C10 C12 C14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [C0  C2  C4  C6  C8  C10 C12 C14] full evens row C */\
    vmovdqa TMP4, 2*X8_SLOT(%rsp)       ;  /* W[2] = evens of row C */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [C1  C3  C5  C7  C1  C3  C5  C7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [C9  C11 C13 C15 C9  C11 C13 C15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [C1  C3  C5  C7  C9  C11 C13 C15]        */          \
    vmovdqa TMP4, 10*X8_SLOT(%rsp)      ;  /* W[10] = row C odds (rights)                     */          \
    /* D - Rows (W[3] and W[11]) */                                                                       \
    vmovdqu  3*32(_base), TMP0          ;  /* TMP0 = [D0  D1  D2  D3  D4  D5  D6  D7 ] block 1 */         \
    vmovdqu  3*32+256(_base), TMP1      ;  /* TMP1 = [D8  D9  D10 D11 D12 D13 D14 D15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [D0  D2  D4  D6  D0  D2  D4  D6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [D8  D10 D12 D14 D8  D10 D12 D14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [D0  D2  D4  D6  D8  D10 D12 D14] all evens row D */ \
    vmovdqa TMP4, 3*X8_SLOT(%rsp)       ;  /* W[3] = evens of row D */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [D1  D3  D5  D7  D1  D3  D5  D7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [D9  D11 D13 D15 D9  D11 D13 D15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [D1  D3  D5  D7  D9  D11 D13 D15]        */          \
    vmovdqa TMP4, 11*X8_SLOT(%rsp)      ;  /* W[11] = row D odds (rights)                     */          \
    /* E - Rows (W[4] and W[12]) */                                                                       \
    vmovdqu  4*32(_base), TMP0          ;  /* TMP0 = [E0  E1  E2  E3  E4  E5  E6  E7 ] block 1 */         \
    vmovdqu  4*32+256(_base), TMP1      ;  /* TMP1 = [E8  E9  E10 E11 E12 E13 E14 E15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [E0  E2  E4  E6  E0  E2  E4  E6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [E8  E10 E12 E14 E8  E10 E12 E14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [E0  E2  E4  E6  E8  E10 E12 E14] all evens row E */ \
    vmovdqa TMP4, 4*X8_SLOT(%rsp)       ;  /* W[4] = evens of row E */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [E1  E3  E5  E7  E1  E3  E5  E7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [E9  E11 E13 E15 E9  E11 E13 E15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [E1  E3  E5  E7  E9  E11 E13 E15]        */          \
    vmovdqa TMP4, 12*X8_SLOT(%rsp)      ;  /* W[12] = row E odds (rights)                     */          \
    /* F - Rows (W[5] and W[13]) */                                                                       \
    vmovdqu  5*32(_base), TMP0          ;  /* TMP0 = [F0  F1  F2  F3  F4  F5  F6  F7 ] block 1 */         \
    vmovdqu  5*32+256(_base), TMP1      ;  /* TMP1 = [F8  F9  F10 F11 F12 F13 F14 F15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [F0  F2  F4  F6  F0  F2  F4  F6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [F8  F10 F12 F14 F8  F10 F12 F14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [F0  F2  F4  F6  F8  F10 F12 F14] all evens row F */ \
    vmovdqa TMP4, 5*X8_SLOT(%rsp)       ;  /* W[5] = evens of row F */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [F1  F3  F5  F7  F1  F3  F5  F7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [F9  F11 F13 F15 F9  F11 F13 F15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [F1  F3  F5  F7  F9  F11 F13 F15]        */          \
    vmovdqa TMP4, 13*X8_SLOT(%rsp)      ;  /* W[13] = row F odds (rights)                     */          \
    /* G - Rows (W[6] and W[14]) */                                                                       \
    vmovdqu  6*32(_base), TMP0          ;  /* TMP0 = [G0  G1  G2  G3  G4  G5  G6  G7 ] block 1 */         \
    vmovdqu  6*32+256(_base), TMP1      ;  /* TMP1 = [G8  G9  G10 G11 G12 G13 G14 G15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [G0  G2  G4  G6  G0  G2  G4  G6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [G8  G10 G12 G14 G8  G10 G12 G14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [G0  G2  G4  G6  G8  G10 G12 G14] full evens row G */\
    vmovdqa TMP4, 6*X8_SLOT(%rsp)       ;  /* W[6] = evens of row G */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [G1  G3  G5  G7  G1  G3  G5  G7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [G9  G11 G13 G15 G9  G11 G13 G15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [G1  G3  G5  G7  G9  G11 G13 G15]        */          \
    vmovdqa TMP4, 14*X8_SLOT(%rsp)      ;  /* W[14] = row G odds (rights)                     */          \
    /* H - Rows (W[7] and W[15]) */                                                                       \
    vmovdqu  7*32(_base), TMP0          ;  /* TMP0 = [H0  H1  H2  H3  H4  H5  H6  H7 ] block 1 */         \
    vmovdqu  7*32+256(_base), TMP1      ;  /* TMP1 = [H8  H9  H10 H11 H12 H13 H14 H15] block 2 */         \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [H0  H2  H4  H6  H0  H2  H4  H6 ] block 1: evens */  \
    vpermd TMP1, TMP2, TMP5             ;  /* TMP5 = [H8  H10 H12 H14 H8  H10 H12 H14] block 2: evens */  \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [H0  H2  H4  H6  H8  H10 H12 H14] full evens row H */\
    vmovdqa TMP4, 7*X8_SLOT(%rsp)       ;  /* W[7] = evens of row H */                                    \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [H1  H3  H5  H7  H1  H3  H5  H7 ] blk 1 odds */      \
    vpermd TMP1, TMP3, TMP5             ;  /* TMP5 = [H9  H11 H13 H15 H9  H11 H13 H15] blk 2 odds */      \
    vperm2f128 $0x02, TMP4, TMP5, TMP4  ;  /* TMP4 = [H1  H3  H5  H7  H9  H11 H13 H15]        */          \
    vmovdqa TMP4, 15*X8_SLOT(%rsp)      ;  /* W[15] = row H odds (rights)                     */

/*
 * DEINTERLEAVE_SOA_1BLOCK
 *
 * _base - one 256-byte SoA group (from one STORE_SOA call)
 *         8 results -> 4 pairs -> 4 active lanes (0-3)
 *         Lanes 4-7 are duplicates of 0-3 (garbage, never consumed)
 */
#define DEINTERLEAVE_SOA_1BLOCK(_base) \
    leaq .LS_EVEN_IDX(%rip), %rax; vmovdqu (%rax), TMP2   ;   /* TMP2 = even index */            \
    leaq .LS_ODD_IDX(%rip),  %rax; vmovdqu (%rax), TMP3   ;   /* TMP3 = odd index */             \
    /* A - Rows (W[0] and W[8]) */                                                               \
    vmovdqu  0*32(_base), TMP0          ;  /* TMP0 = [A0 A1  A2  A3  A4  A5  A6  A7 ]        */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [A0 A2  A4  A6  A0  A2  A4  A6 ] evens  */  \
    vmovdqa TMP4, 0*X8_SLOT(%rsp)       ;  /* W[0] = evens of row A                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [A1 A3  A5  A7  A1  A3  A5  A7 ] odds   */  \
    vmovdqa TMP4, 8*X8_SLOT(%rsp)       ;  /* W[8] = odds of row A                           */  \
    /* B - Rows (W[1] and W[9]) */                                                               \
    vmovdqu  1*32(_base), TMP0          ;  /* TMP0 = [B0  B1  B2  B3  B4  B5  B6  B7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [B0  B2  B4  B6  B0  B2  B4  B6 ] evens */  \
    vmovdqa TMP4, 1*X8_SLOT(%rsp)       ;  /* W[1] = evens of row B                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [B1  B3  B5  B7  B1  B3  B5  B7 ] odds  */  \
    vmovdqa TMP4, 9*X8_SLOT(%rsp)       ;  /* W[9] = odds of row B                           */  \
    /* C - Rows (W[2] and W[10]) */                                                              \
    vmovdqu  2*32(_base), TMP0          ;  /* TMP0 = [C0  C1  C2  C3  C4  C5  C6  C7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [C0  C2  C4  C6  C0  C2  C4  C6 ] evens */  \
    vmovdqa TMP4, 2*X8_SLOT(%rsp)       ;  /* W[2] = evens of row C                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [C1  C3  C5  C7  C1  C3  C5  C7 ] odds  */  \
    vmovdqa TMP4, 10*X8_SLOT(%rsp)      ;  /* W[10] = odds of row C                          */  \
    /* D - Rows (W[3] and W[11]) */                                                              \
    vmovdqu  3*32(_base), TMP0          ;  /* TMP0 = [D0  D1  D2  D3  D4  D5  D6  D7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [D0  D2  D4  D6  D0  D2  D4  D6 ] evens */  \
    vmovdqa TMP4, 3*X8_SLOT(%rsp)       ;  /* W[3] = evens of row D                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [D1  D3  D5  D7  D1  D3  D5  D7 ] odds  */  \
    vmovdqa TMP4, 11*X8_SLOT(%rsp)      ;  /* W[11] = odds of row D                          */  \
    /* E - Rows (W[4] and W[12]) */                                                              \
    vmovdqu  4*32(_base), TMP0          ;  /* TMP0 = [E0  E1  E2  E3  E4  E5  E6  E7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [E0  E2  E4  E6  E0  E2  E4  E6 ] evens */  \
    vmovdqa TMP4, 4*X8_SLOT(%rsp)       ;  /* W[4] = evens of row E                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [E1  E3  E5  E7  E1  E3  E5  E7 ] odds  */  \
    vmovdqa TMP4, 12*X8_SLOT(%rsp)      ;  /* W[12] = odds of row E                          */  \
    /* F - Rows (W[5] and W[13]) */                                                              \
    vmovdqu  5*32(_base), TMP0          ;  /* TMP0 = [F0  F1  F2  F3  F4  F5  F6  F7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [F0  F2  F4  F6  F0  F2  F4  F6 ] evens */  \
    vmovdqa TMP4, 5*X8_SLOT(%rsp)       ;  /* W[5] = evens of row F                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [F1  F3  F5  F7  F1  F3  F5  F7 ] odds  */  \
    vmovdqa TMP4, 13*X8_SLOT(%rsp)      ;  /* W[13] = odds of row F                          */  \
    /* G - Rows (W[6] and W[14]) */                                                              \
    vmovdqu  6*32(_base), TMP0          ;  /* TMP0 = [G0  G1  G2  G3  G4  G5  G6  G7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [G0  G2  G4  G6  G0  G2  G4  G6 ] evens */  \
    vmovdqa TMP4, 6*X8_SLOT(%rsp)       ;  /* W[6] = evens of row G                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [G1  G3  G5  G7  G1  G3  G5  G7 ] odds  */  \
    vmovdqa TMP4, 14*X8_SLOT(%rsp)      ;  /* W[14] = odds of row G                          */  \
    /* H - Rows (W[7] and W[15]) */                                                              \
    vmovdqu  7*32(_base), TMP0          ;  /* TMP0 = [H0  H1  H2  H3  H4  H5  H6  H7 ]       */  \
    vpermd TMP0, TMP2, TMP4             ;  /* TMP4 = [H0  H2  H4  H6  H0  H2  H4  H6 ] evens */  \
    vmovdqa TMP4, 7*X8_SLOT(%rsp)       ;  /* W[7] = evens of row H                          */  \
    vpermd TMP0, TMP3, TMP4             ;  /* TMP4 = [H1  H3  H5  H7  H1  H3  H5  H7 ] odds  */  \
    vmovdqa TMP4, 15*X8_SLOT(%rsp)      ;  /* W[15] = odds of row H                          */
