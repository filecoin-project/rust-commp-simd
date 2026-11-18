# transpose16_avx512.h - 16x16 u32 in-register transpose (AVX-512)
#
# Adapted from intel-ipsec-mb TRANSPOSE16_U32_PRELOADED.
# This is the "preloaded" variant: data is already in zmm registers
# (loaded via the split ymm + vinserti64x4 pattern that interleaves
# lanes 0-7 with lanes 8-15).  This eliminates the final vshuff64x2
# phase that the non-preloaded transpose requires.
#
# Usage:
#   TRANSPOSE16(r0,r1,...,r15, t0,t1,t2,t3)
#
#   r0-r15: 16 zmm registers holding the input (modified in place)
#   t0-t3:  4 zmm scratch registers (clobbered)
#   Loads PSHUFFLE_TRANSPOSE_MASK1/MASK2 internally via t2/t3.
#
# After transpose, register rN holds dword element N from all
# 16 original input registers packed as 16 x u32.

#define TRANSPOSE16(_r0,_r1,_r2,_r3,_r4,_r5,_r6,_r7,            \
                    _r8,_r9,_r10,_r11,_r12,_r13,_r14,_r15,      \
                    _t0,_t1,_t2,_t3)                            \
                                                                \
    /* input  r0  = {a15 a14 a13 a12   a11 a10  a9  a8   a7  a6  a5  a4   a3  a2  a1  a0} */ \
    /*        r1  = {b15 b14 b13 b12   b11 b10  b9  b8   b7  b6  b5  b4   b3  b2  b1  b0} */ \
    /*        ...                                                                          */ \
    /*        r15 = {p15 p14 p13 p12   p11 p10  p9  p8   p7  p6  p5  p4   p3  p2  p1  p0} */ \
    /*                                                                                     */ \
    /* output r0  = { p0  o0  n0  m0    l0  k0  j0  i0   h0  g0  f0  e0   d0  c0  b0  a0} */ \
    /*        r1  = { p1  o1  n1  m1    l1  k1  j1  i1   h1  g1  f1  e1   d1  c1  b1  a1} */ \
    /*        ...                                                                          */ \
    /*        r15 = {p15 o15 n15 m15   l15 k15 j15 i15  h15 g15 f15 e15  d15 c15 b15 a15} */ \
                                                                \
    /* ---- Phase 1: vshufps 4x4 blocks ---- */                 \
    /* group r0-r3 */                                           \
    vshufps $0x44, _r1, _r0, _t0  /* t0 = {b13 b12 a13 a12  b9 b8 a9 a8  b5 b4 a5 a4  b1 b0 a1 a0} */ ;\
    vshufps $0xEE, _r1, _r0, _r0  /* r0 = {b15 b14 a15 a14  b11 b10 a11 a10  b7 b6 a7 a6  b3 b2 a3 a2} */ ;\
    vshufps $0x44, _r3, _r2, _t1                                ;\
    vshufps $0xEE, _r3, _r2, _r2                                ;\
    vshufps $0xDD, _t1, _t0, _r3  /* r3 = {d13 c13 b13 a13  d9 c9 b9 a9  d5 c5 b5 a5  d1 c1 b1 a1} */ ;\
    vshufps $0x88, _r2, _r0, _r1  /* r1 = {d14 c14 b14 a14  ...  d2 c2 b2 a2} */ ;\
    vshufps $0xDD, _r2, _r0, _r0  /* r0 = {d15 c15 b15 a15  ...  d3 c3 b3 a3} */ ;\
    vshufps $0x88, _t1, _t0, _t0  /* t0 = {d12 c12 b12 a12  ...  d0 c0 b0 a0} */ ;\
                                                                \
    /* load permute masks (slotted here to hide latency) */     \
    leaq PSHUFFLE_TRANSPOSE_MASK1(%rip), %rax; vmovdqu64 (%rax), _t2  ;\
    leaq PSHUFFLE_TRANSPOSE_MASK2(%rip), %rax; vmovdqu64 (%rax), _t3  ;\
                                                                \
    /* group r4-r7 (use r2 in place of t0) */                   \
    vshufps $0x44, _r5, _r4, _r2                             ;\
    vshufps $0xEE, _r5, _r4, _r4                             ;\
    vshufps $0x44, _r7, _r6, _t1                             ;\
    vshufps $0xEE, _r7, _r6, _r6                             ;\
    vshufps $0xDD, _t1, _r2, _r7                             ;\
    vshufps $0x88, _r6, _r4, _r5                             ;\
    vshufps $0xDD, _r6, _r4, _r4                             ;\
    vshufps $0x88, _t1, _r2, _r2                             ;\
                                                                \
    /* group r8-r11 (use r6 in place of t0) */                  \
    vshufps $0x44, _r9, _r8, _r6                             ;\
    vshufps $0xEE, _r9, _r8, _r8                             ;\
    vshufps $0x44, _r11, _r10, _t1                             ;\
    vshufps $0xEE, _r11, _r10, _r10                             ;\
    vshufps $0xDD, _t1, _r6, _r11                             ;\
    vshufps $0x88, _r10, _r8, _r9                             ;\
    vshufps $0xDD, _r10, _r8, _r8                             ;\
    vshufps $0x88, _t1, _r6, _r6                             ;\
                                                                \
    /* group r12-r15 (use r10 in place of t0) */                \
    vshufps $0x44, _r13, _r12, _r10                             ;\
    vshufps $0xEE, _r13, _r12, _r12                             ;\
    vshufps $0x44, _r15, _r14, _t1                             ;\
    vshufps $0xEE, _r15, _r14, _r14                             ;\
    vshufps $0xDD, _t1, _r10, _r15                             ;\
    vshufps $0x88, _r14, _r12, _r13                             ;\
    vshufps $0xDD, _r14, _r12, _r12                             ;\
    vshufps $0x88, _t1, _r10, _r10                             ;\
                                                                \
    /* At this point interesting data is in:                 */ \
    /* t0, r3, r1, r0, r2, r7, r5, r4, r6, r11, r9, r8, r10, r15, r13, r12 */ \
                                                                \
    /* Can use t1 and r14 as scratch registers               */ \
                                                                \
    /* ---- Phase 2: vpermi2q cross-lane permutes ---- */       \
    vmovdqu32 _t2, _t1                                        ;\
    vpermi2q _r13, _r9, _t1                                  ;\
    vmovdqu32 _t3, _r14                                        ;\
    vpermi2q _r13, _r9, _r14                                  ;\
    vmovdqu32 _t2, _r9                                        ;\
    vpermi2q _r15, _r11, _r9                                 ;\
    vmovdqu32 _t3, _r13                                        ;\
    vpermi2q _r15, _r11, _r13                                 ;\
    vmovdqu32 _t2, _r11                                        ;\
    vpermi2q _r12, _r8, _r11                                  ;\
    vmovdqu32 _t3, _r15                                        ;\
    vpermi2q _r12, _r8, _r15                                  ;\
    vmovdqu32 _t2, _r8                                        ;\
    vpermi2q _r10, _r6, _r8                                  ;\
    vmovdqu32 _t3, _r12                                        ;\
    vpermi2q _r10, _r6, _r12                                  ;\
    vmovdqu32 _t1, _r10                                        ;\
    vmovdqu32 _t2, _t1                                        ;\
    vpermi2q _r5, _r1, _t1                                   ;\
    vmovdqu32 _t3, _r6                                        ;\
    vpermi2q _r5, _r1, _r6                                   ;\
    vmovdqu32 _t2, _r1                                        ;\
    vpermi2q _r7, _r3, _r1                                   ;\
    vmovdqu32 _t3, _r5                                        ;\
    vpermi2q _r7, _r3, _r5                                   ;\
    vmovdqu32 _t2, _r3                                        ;\
    vpermi2q _r4, _r0, _r3                                   ;\
    vmovdqu32 _t3, _r7                                        ;\
    vpermi2q _r4, _r0, _r7                                   ;\
    vmovdqu32 _t2, _r0                                        ;\
    vpermi2q _r2, _t0, _r0                                   ;\
    vmovdqu32 _t3, _r4                                        ;\
    vpermi2q _r2, _t0, _r4                                   ;\
    vmovdqu32 _t1, _r2
# OPERANDS: AT&T order (src, dst)
