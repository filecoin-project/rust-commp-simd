# =========================================================================
# PRORD - Packed Rotate Right Dword (vprord in AVX2)
#
# AVX-512 has vprord: 1 instruction.
# AVX2 doesn't. This macro does it with 3 instructions:
#
#   rotate right by N (32-bit dword):
#
#   original:    [ A  A  A  A  A  A | B  B  B  B ]
#                 <--  32-N bits --> <-- N bits-->
#
#   vpslld 32-N: [ B  B  B  B  0  0   0  0  0  0 ]  B bits pushed left  (TMP6)
#   vpsrld N:    [ 0  0  0  0  A  A   A  A  A  A ]  A bits pushed right (_reg)
#   vpor:        [ B  B  B  B  A  A   A  A  A  A ]  rotated, no zeros
#                                   ↑
#                          original split point
# Two variants:
#   PRORD(reg, imm)          - destructive: result overwrites reg, uses TMP6
#   PRORD_nd(dst, src, imm)  - non-destructive: src preserved, result in dst, uses TMP6
#
# TMP6 (ymm15) is always clobbered.
#
# Used by: PROCESS_LOOP (SIGMA0, SIGMA1, CH, MAJ)
#          MSG_SCHED    (sigma0, sigma1)
# =========================================================================
#define PRORD(_reg, _imm) \
    vpslld  $(32-(_imm)), _reg, TMP6             ;   /* TMP6 = B bits (left shift) */ \
    vpsrld  $(_imm), _reg, _reg                  ;   /* reg  = A bits (right shift, original gone) */ \
    vpor    TMP6, _reg, _reg                        /* reg  = rotated result (A and B rejoined) */

#define PRORD_nd(_dst, _src, _imm) \
    vpslld  $(32-(_imm)), _src, TMP6             ;   /* TMP6 = B bits */ \
    vpsrld  $(_imm), _src, _dst                  ;   /* dst  = A bits (src preserved) */ \
    vpor    TMP6, _dst, _dst                        /* dst  = rotated result (A and B rejoined) */


#define CH_SIGMA1(_E, _F, _G) \
    PRORD_nd(TMP3, _E, (11-6))                   ;   /* a0 = ROR_5(E) (trcik chain into ROR_11 ^ ROR_6) */ \
    vpxor   _F, _G, TMP5                         ;   /* TMP5 = F ^ G */ \
    vpand   _E, TMP5, TMP5                       ;   /* TMP5 = (F ^ G) & E */ \
    vpxor   _G, TMP5, TMP5                       ;   /* TMP5 = ((F^G)&E) ^ G = CH(E,F,G) */ \
    PRORD_nd(TMP4, _E, 25)                       ;   /* TMP4 = ROR_25(E) */ \
    vpxor   _E, TMP3, TMP3                       ;   /* TMP3 = ROR_5(E) ^ E */ \
    PRORD(TMP3, 6)                               ;   /* TMP3 = ROR_6(ROR_5(E)^E) = ROR_11(E) ^ ROR_6(E) */ \
    vpxor   TMP4, TMP3, TMP3                     ;   /* TMP3 = ROR_11 ^ ROR_6 ^ ROR_25 = SIGMA1(E) */

/* SIGMA0(A) - shared by PROCESS_LOOP and PROCESS_LOOP_CONST
 * Input:  _A (state register)
 * Output: TMP5 = SIGMA0(A)
 * Clobbers: TMP4, TMP6 (via PRORD) */
#define SIGMA0(_A) \
    PRORD_nd(TMP5, _A, 11)                        ;   /* TMP5 = ROR_11(A) (trick: chains into ROR_13 ^ ROR_2) */ \
    PRORD_nd(TMP4, _A, 22)                        ;   /* TMP4 = ROR_22(A) */ \
    vpxor   _A, TMP5, TMP5                        ;   /* TMP5 = ROR_11(A) ^ A */ \
    PRORD(TMP5, 2)                                ;   /* TMP5 = ROR_2(ROR_11(A)^A) = ROR_13 ^ ROR_2 */ \
    vpxor   TMP4, TMP5, TMP5                      ;   /* TMP5 = ROR_13 ^ ROR_2 ^ ROR_22 = SIGMA0(A) */

/* MAJ(A,B,C) - shared by PROCESS_LOOP and PROCESS_LOOP_CONST
 * Input:  _A, _B, _C (state registers)
 * Output: TMP4 = MAJ(A,B,C)
 * _scratch: any free register (T1 in both callers)
 * Clobbers: _scratch, TMP4 */
#define MAJ(_A, _B, _C, _scratch) \
    vpxor   _A, _C, _scratch                      ;   /* scratch = A ^ C */ \
    vpand   _B, _scratch, _scratch                ;   /* scratch = (A ^ C) & B */ \
    vpand   _A, _C, TMP4                          ;   /* TMP4 = A & C */ \
    vpor    _scratch, TMP4, TMP4                      /* TMP4 = (A&C) | ((A^C)&B) = MAJ(A,B,C) */

# =========================================================================
# PROCESS_LOOP - one SHA256 compression round (8 lanes in parallel)
# a.k.a -> the part that updates state A-H using one W value
#
# Adapted from intel-ipsec-mb sha256_oct_avx2 ROUND_00_15.
#
# Fixed register assignments (must match sha256_reg_aliases_avx2.h):
#    T1:    ymm8     (round temp, also carries current W on entry)
#    TMP3:  ymm12    SIGMA
#    TMP4:  ymm13    CH
#    TMP5:  ymm14    MAJ
#    TMP6:  ymm15    (implicit scratch inside PRORD macro)
#    TBL:   rdx      (pointer to K256 table, 64 entries x 4 bytes each)
#
#   _T1:    YMM register holding the current W value (loaded by caller)
#           Unlike x16 where W0-W15 live in zmm16-31, x8 has no room -
#           W values live on stack. Caller loads W into _T1 before calling.
#
#   _i:     round number (0-63), used for stack slot: X8_SLOT*(_i & 0xf)
#           16 W entries on stack, each X8_SLOT (32) bytes apart.
#           _i & 0xf maps any round to slot 0-15 (circular reuse):
#
#           64 rounds, 16 slots.
#           Each slot gets written 4 times total (rounds 0-15, 16-31, 32-47, 48-63):
#                 ┌───┐
#           slot: │ 0 │ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
#           r0:   │ W │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#           r1:   │ . │ W  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#           ...   │   │
#           r15:  │ . │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  W
# - - - - - r16:  │ W │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  <- slot 0 again - - - -
#           r17:  │ . │ W  .  .  .  .  .  .  .  .  .  .  .  .  .  .
#           ...   │   │
#           r31:  │ . │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  W
# - - - - - r32:  │ W │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  <- slot 0 again - - - -
#           ...   │   │
# - - - - - r48:  │ W │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  <- slot 0 again - - - -
#           ...   │   │
#           r63:  │ . │ .  .  .  .  .  .  .  .  .  .  .  .  .  .  W
#                 └───┘
#
#   _A.._H: the 8 state word registers in current rotation order
#
#   After the macro, _H holds the new A, _D holds the new E.
#   The caller rotates by shifting argument positions each round.
# =========================================================================
#define PROCESS_LOOP(_T1, _i, _A, _B, _C, _D, _E, _F, _G, _H) \
    vmovdqa _T1, X8_SLOT*(_i & 0xf)(%rsp)         ;   /* save W to stack slot */ \
    vpbroadcastd  ((_i)*4)(TBL), TMP6             ;   /* broadcast K[_i] to all 8 lanes */ \
    vpaddd TMP6, _T1, _T1                         ;   /* T1 = 8 W's + same K[_i] in each lane */ \
    /* CH(E,F,G) and SIGMA1(E) */                 \
    CH_SIGMA1(_E, _F, _G)                         ;   /* TMP5 = CH(E,F,G), TMP3 = ROR_6(ROR_5(E)^E) = ROR_11(E) ^ ROR_6(E) */  \
    /* T1 accumulate: H += CH + (W+K) + SIGMA1 */ \
    vpaddd  TMP5, _H, _H                          ;   /* H += CH(E,F,G) */ \
    vpaddd  _T1, _H, _H                           ;   /* H += W+K (T1 holds W+K from earlier) */ \
    vpaddd  TMP3, _H, _H                          ;   /* H += SIGMA1(E) */ \
    /* D = D + T1 (new E) */                      \
    vpaddd  _H, _D, _D                            ;   /* D += H (new E = D + T1) */ \
    /* SIGMA0 */                                  \
    SIGMA0(_A)                                    ;   /* TMP5 = SIGMA0(A) */ \
    /* MAJ(A,B,C) */                              \
    MAJ(_A, _B, _C, _T1)                          ;   /* TMP4 = MAJ(A,B,C) */ \
    /* H = SIGMA0 + MAJ + T1 (new A) */           \
    vpaddd  TMP4, _H, _H                          ;   /* H += MAJ */  \
    vpaddd  TMP5, _H, _H

# =========================================================================
# MSG_SCHED_ROUND_16_63 - message schedule for rounds 16-63 (8 lanes)
#
# Computes new Wt from four older W values on the stack:
#   Wt = sigma1(Wt-2) + Wt-7 + sigma0(Wt-15) + Wt-16
#
# sigma0(x) = ROR_7  ^ ROR_18 ^ SHR_3
# sigma1(x) = ROR_17 ^ ROR_19 ^ SHR_10
#
# (lowercase sigma = message schedule, different from
#  uppercase SIGMA in PROCESS_LOOP which is compression)
#
# Reads 4 W values from stack (Wt-15, Wt-2, Wt-16, Wt-7)
# using X8_SLOT*(((_i)-N)&0xf) addressing (&0xf = mod 16 circular slot)
#
# Interleaves sigma0 and sigma1 steps so CPU can work on both chains
# Uses same PRORD chaining trick as PROCESS_LOOP:
#   sigma0: ROR_11 then ROR_7  gives ROR_18 ^ ROR_7   (18-7=11)
#   sigma1: ROR_2  then ROR_17 gives ROR_19 ^ ROR_17  (19-17=2)
#
#   _T1:    output register - receives the new Wt (ready for PROCESS_LOOP)
#   _i:     round number (16-63)
#
# Clobbers: TMP3, TMP4, TMP5, TMP6 (via PRORD)
# =========================================================================
#define MSG_SCHED_ROUND_16_63(_T1, _i) \
    /* load Wt-15 and Wt-2 from stack */               \
    vmovdqa X8_SLOT*(((_i)-15)&0xf)(%rsp), _T1         ; /* T1 = Wt-15 from stack */ \
    vmovdqa X8_SLOT*(((_i)-2) &0xf)(%rsp), TMP4        ; /* TMP4 = Wt-2 from stack */ \
    /* sigma0(Wt-15) = ROR_18 ^ ROR_7 ^ SHR_3 */       \
    vmovdqa _T1, TMP3                                  ; /* TMP3 = copy of Wt-15 (PRORD will destroy T1) */ \
    PRORD(_T1, 11)                                     ; /* T11 = ROR_11(Wt-15) (trick: 18-7=11, chains into ROR_18 ^ ROR_7) */ \
    /* sigma1(Wt-2) */                                 \
    vmovdqa TMP4, TMP5                                 ; /* TMP5 = copy of Wt-2 (save for sigma1's SHR_10 later) */ \
    PRORD(TMP4, 2)                                     ; /* TMP4 = ROR_2(Wt-2) (trick: 19-17=2, chains into ROR_19 ^ ROR_17) */ \
    /* sigma0(Wt-15) - finis */                        \
    vpxor TMP3, _T1, T1                                ; /* T1 = ROR_11(Wt-15) ^ Wt-15 */ \
    PRORD(_T1, 7)                                      ; /* T1 = ROR_7(ROR_11^Wt-15) = ROR_18 ^ ROR_7 */ \
    vpsrld $3, TMP3, TMP3                              ; /* TMP3 = SHR_3(Wt-15) (sigma uses SHR (shift right) for the third term) */ \
    vpxor TMP3, _T1, _T1                               ; /* T1 = ROR_18 ^ ROR_7 ^ SHR_3 = sigma0(Wt-15) done */ \
    /* sigma1(Wt-2) - finis */                         \
    vpxor TMP5, TMP4, TMP4                             ; /* TMP4 = ROR_2(Wt-2) ^ Wt-2 */ \
    PRORD(TMP4, 17)                                    ; /* TMP4 = ROR_17(ROR_2 ^ Wt-2) = ROR_19 ^ ROR_17 */ \
    vpsrld $10, TMP5, TMP5                             ; /* TMP5 = SHR_10(Wt-2) (shift not rotate) */ \
    vpxor TMP5, TMP4, TMP4                             ; /* TMP4 = ROR_19 ^ ROR_17 ^ SHR_10 = sigma1(Wt-2) done */ \
    /* Wt = Wt-16 + sigma0 + Wt-7 + sigma1 */          \
    vpaddd X8_SLOT*(((_i)-16)&0xf)(%rsp), _T1, _T1     ; /* T1   = Wt-16 + sigma0(T1)   (Wt-16 from stack) */ \
    vpaddd X8_SLOT*(((_i)- 7)&0xf)(%rsp), TMP4, TMP4   ; /* TMP4 = Wt-7  + sigma1(TMP4) (Wt-7 from the stack) */ \
    vpaddd TMP4, _T1, _T1                                /* T1 = new Wt (Wt-16 + sigma0 + Wt-7 + sigma1) */

