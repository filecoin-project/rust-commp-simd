# sha256_round_avx512.h - SHA256 round macros for 16-lane AVX-512
#
# Adapted from intel-ipsec-mb PROCESS_LOOP / MSG_SCHED_ROUND_16_63.
#
# Fixed register assignments (must match caller):
#   T1   = zmm8     (round temp)
#   TMP0 = zmm9     TMP1 = zmm10    TMP2 = zmm11    TMP3 = zmm12
#   TMP4 = zmm13    TMP5 = zmm14    TMP6 = zmm15
#   TBL  = rdx      (pointer to K256 table, 4-byte entries)
#
# TMP3 (zmm12) carries the current round constant Kt across calls.
# It must be pre-loaded before the first PROCESS_LOOP invocation.

# =========================================================================
# PROCESS_LOOP - one SHA256 compression round (16 lanes in parallel)
#
#   _WT:    message schedule word register (zmm16-31)
#   _ROUND: round number (0-63), used to load K[ROUND+1] at end
#   _A.._H: the 8 state word registers in current rotation order
#
#   After the macro, _H holds the new A, _D holds the new E.
#   The caller rotates by shifting argument positions each round.
# =========================================================================
#define PROCESS_LOOP(_WT, _ROUND, _A, _B, _C, _D, _E, _F, _G, _H) \
    /* T1 = H + Kt */                                              \
    vpaddd TMP3, _H, T1                                      ;\
    /* CH(E,F,G) and SIGMA1(E) */                                   \
    vmovdqa32 _E, TMP0                                           ;\
    vprord $6, _E, TMP1                                       ;\
    vprord $11, _E, TMP2                                      ;\
    vprord $25, _E, TMP3                                      ;\
    vpternlogd $0xCA, _G, _F, TMP0  /* CH(E,F,G) */     ;\
    vpaddd _WT, T1, T1  /* T1 += Wt */      ;\
    vpternlogd $0x96, TMP3, TMP2, TMP1  /* SIGMA1(E) */     ;\
    vpaddd TMP0, T1, T1  /* T1 += CH */      ;\
    vpaddd TMP1, T1, T1  /* T1 += SIGMA1 */  ;\
    /* D = D + T1  (new E) */                                       \
    vpaddd T1, _D, _D                                        ;\
    /* SIGMA0(A) and MAJ(A,B,C) */                                  \
    vprord $2, _A, _H                                         ;\
    vprord $13, _A, TMP2                                      ;\
    vprord $22, _A, TMP3                                      ;\
    vmovdqa32 _A, TMP0                                           ;\
    vpternlogd $0xE8, _C, _B, TMP0  /* MAJ(A,B,C) */    ;\
    vpternlogd $0x96, TMP3, TMP2, _H  /* SIGMA0(A) */     ;\
    vpaddd TMP0, _H, _H  /* T2 = SIGMA0+MAJ */;\
    /* H = T2 + T1  (new A) */                                      \
    vpaddd T1, _H, _H                                        ;\
    /* Load K[ROUND+1] for next round */                            \
    vpbroadcastd ((_ROUND)+1)*4(TBL), TMP3

# =========================================================================
# MSG_SCHED_ROUND_16_63 - message schedule update Wt (16 lanes)
#
#   _WT:    register to update (Wt-16, becomes new Wt)
#   _WTp1:  Wt-15 register
#   _WTp9:  Wt-7 register
#   _WTp14: Wt-2 register
#
#   Wt = Wt-16 + sigma1(Wt-2) + Wt-7 + sigma0(Wt-15)
# =========================================================================
#define MSG_SCHED_ROUND_16_63(_WT, _WTp1, _WTp9, _WTp14) \
    /* sigma1(Wt-2) */                                     \
    vprord $17, _WTp14, TMP4                          ;\
    vprord $19, _WTp14, TMP5                          ;\
    vpsrld $10, _WTp14, TMP6                          ;\
    vpternlogd $0x96, TMP6, TMP5, TMP4                    ;\
    /* Wt += sigma1(Wt-2) + Wt-7 */                       \
    vpaddd TMP4, _WT, _WT                            ;\
    vpaddd _WTp9, _WT, _WT                           ;\
    /* sigma0(Wt-15) */                                    \
    vprord $7, _WTp1, TMP4                            ;\
    vprord $18, _WTp1, TMP5                           ;\
    vpsrld $3, _WTp1, TMP6                            ;\
    vpternlogd $0x96, TMP6, TMP5, TMP4                    ;\
    /* Wt += sigma0(Wt-15) */                              \
    vpaddd TMP4, _WT, _WT

# =========================================================================
# MSG_SCHED_ROUND_00_15 - load next block data into Wt register
#
# Split into FIRST8 (lanes 0-7, first 32B of each 64B block) and
# LAST8 (lanes 8-15, second 32B).  The split-load + vinserti64x4
# interleaves lanes 0-7 with 8-15, which the TRANSPOSE16 macro
# expects as its "preloaded" input format.
#
#   _WT:   zmm register for this W word (zmm16-31)
#   _YWT:  corresponding ymm register (ymm16-31)
#   _LANE: lane index (0-7 for FIRST8, 8-15 for LAST8)
#
# Fixed registers: rdi=STATE, rcx=IDX, r9/r10=scratch
# Data pointers at STATE + 0x200 + lane*8
# =========================================================================
#define MSG_SCHED_ROUND_00_15_FIRST8(_WT, _YWT, _LANE)                     \
    mov (0x200 + (_LANE)*8)(STATE), %r9                   ;\
    mov (0x240 + (_LANE)*8)(STATE), %r10                   ;\
    vmovups (%r9,IDX), _YWT                                   ;\
    vinserti64x4 $0x01, (%r10,IDX), _WT, _WT

#define MSG_SCHED_ROUND_00_15_LAST8(_WT, _YWT, _LANE)                     \
    mov (0x200 + ((_LANE)-8)*8)(STATE), %r9              ;\
    mov (0x240 + ((_LANE)-8)*8)(STATE), %r10              ;\
    vmovups 0x20(%r9,IDX), _YWT                           ;\
    vinserti64x4 $0x01, 0x20(%r10,IDX), _WT, _WT
# OPERANDS: AT&T order (src, dst)
