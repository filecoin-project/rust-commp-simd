# 10 args: 8 data regs (in-place), 2 temps
#
#   Input: 8 regs, each holds 8 dwords from one message block
#     r0 = | a7 a6 a5 a4 | a3 a2 a1 a0 |  message a
#     r1 = | b7 b6 b5 b4 | b3 b2 b1 b0 |  message b
#     r2 = | c7 c6 c5 c4 | c3 c2 c1 c0 |  message c
#     r3 = | d7 d6 d5 d4 | d3 d2 d1 d0 |  message d
#     r4 = | e7 e6 e5 e4 | e3 e2 e1 e0 |  message e
#     r5 = | f7 f6 f5 f4 | f3 f2 f1 f0 |  message f
#     r6 = | g7 g6 g5 g4 | g3 g2 g1 g0 |  message g
#     r7 = | h7 h6 h5 h4 | h3 h2 h1 h0 |  message h
#              upper128      lower128

# ===[ Phase 1: vshufps 0x44/0xEE - interleave dword pairs ]===
#
#   0x44 picks dwords 0,1 from each source
#   0xEE picks dwords 2,3 from each source
#   4 reg-pairs: (r0,r1) (r2,r3) (r4,r5) (r6,r7)
#
#   Phase 1 results (upper128 | lower128):
#     t0 = | b5 b4 a5 a4 | b1 b0 a1 a0 |
#     r0 = | b7 b6 a7 a6 | b3 b2 a3 a2 |
#     t1 = | d5 d4 c5 c4 | d1 d0 c1 c0 |
#     r2 = | d7 d6 c7 c6 | d3 d2 c3 c2 |
#     r2 = | f5 f4 e5 e4 | f1 f0 e1 e0 |  (r2 reused as temp)
#     r4 = | f7 f6 e7 e6 | f3 f2 e3 e2 |
#     t1 = | h5 h4 g5 g4 | h1 h0 g1 g0 |  (t1 reused)
#     r6 = | h7 h6 g7 g6 | h3 h2 g3 g2 |
#              upper128       lower128
#
# ===[ Phase 2: vshufps 0x88/0xDD - pick every-other dword ]===
#
#   0x88 picks dwords 0,2 from each source
#   0xDD picks dwords 1,3 from each source
#   4 reg-pairs: (t0,t1) (r0,r2) (r2,t1) (r4,r6)
#
#   Phase 2 results (upper128 | lower128):
#     t0 = | d4 c4 b4 a4 | d0 c0 b0 a0 |
#     r3 = | d5 c5 b5 a5 | d1 c1 b1 a1 |
#     r1 = | d6 c6 b6 a6 | d2 c2 b2 a2 |
#     r0 = | d7 c7 b7 a7 | d3 c3 b3 a3 |
#     t1 = | h4 g4 f4 e4 | h0 g0 f0 e0 |
#     r7 = | h5 g5 f5 e5 | h1 g1 f1 e1 |
#     r5 = | h6 g6 f6 e6 | h2 g2 f2 e2 |
#     r4 = | h7 g7 f7 e7 | h3 g3 f3 e3 |
#              upper128       lower128
#
#   Each Phase 2 result holds two rows
#   packed into one register - one in lower128, one in upper128
#   (next phase 3 pulls them apart)
#      t0 = | d4 c4 b4 a4 | d0 c0 b0 a0 |
#             row 4  ^^^^    row 0 ^^^^
#
#      r1 = | d6 c6 b6 a6 | d2 c2 b2 a2 |
#             row 6  ^^^^    row 2 ^^^^
#
#
# ===[ Phase 3: vperm2f128 0x02/0x13 - combine 128-bit halves ]===
#
#   0x02 joins lower128 of both: abcd + efgh -> full 8-wide row
#   0x13 joins upper128 of both: abcd + efgh -> full 8-wide row
#   4 reg-pairs: (t0,t1) (r3,r7) (r1,r5) (r0,r4)
#
#   Phase 3 results = TRANSPOSE DONE:
#     r0 = | h0 g0 f0 e0 | d0 c0 b0 a0 |  row 0
#     r1 = | h1 g1 f1 e1 | d1 c1 b1 a1 |  row 1
#     r2 = | h2 g2 f2 e2 | d2 c2 b2 a2 |  row 2
#     r3 = | h3 g3 f3 e3 | d3 c3 b3 a3 |  row 3
#     r4 = | h4 g4 f4 e4 | d4 c4 b4 a4 |  row 4
#     r5 = | h5 g5 f5 e5 | d5 c5 b5 a5 |  row 5
#     r6 = | h6 g6 f6 e6 | d6 c6 b6 a6 |  row 6
#     r7 = | h7 g7 f7 e7 | d7 c7 b7 a7 |  row 7
#              upper128       lower128
#define TRANSPOSE8(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _t0, _t1) \
    /* ---- Phase 1: vshufps 2x2 blocks ---- */                                \
    /* Group: r0-r3                                                            \
                                                                               \
    */                                                                         \
    /* phase 1 ops (AT&T syntax: vshufps $imm, src2, src1, dst) */             \
    vshufps $0x44, _r1, _r0, _t0  ;   /* t0 = {b5 b4 a5 a4 | b1 b0 a1 a0} */   \
    vshufps $0xEE, _r1, _r0, _r0  ;   /* r0 = {b7 b6 a7 a6 | b3 b2 a3 a2} */   \
    vshufps $0x44, _r3, _r2, _t1  ;   /* t1 = {d5 d4 c5 c4 | d1 d0 c1 c0} */   \
    vshufps $0xEE, _r3, _r2, _r2  ;   /* r2 = {d7 d6 c7 c6 | d3 d2 c3 c2} */   \
    /* phase 2 ops */                                                          \
    vshufps $0xDD, _t1, _t0, _r3  ;   /* r3 = {d5 c5 b5 a5 | d1 c1 b1 a1} */   \
    vshufps $0x88, _r2, _r0, _r1  ;   /* r1 = {d6 c6 b6 a6 | d2 c2 b2 a2} */   \
    vshufps $0xDD, _r2, _r0, _r0  ;   /* r0 = {d7 c7 b7 a7 | d3 c3 b3 a3} */   \
    vshufps $0x88, _t1, _t0, _t0  ;   /* t0 = {d4 c4 b4 a4 | d0 c0 b0 a0} */   \
    /* Group: r4-r7                                                            \
                                                                               \
      "Busy" registers:                                                        \
                            [ lo ] and [hi]                                    \
        t0 - occupied (holds rows 0 and 4)                                     \
        r3 - occupied (holds rows 1 and 5)                                     \
        r1 - occupied (holds rows 2 and 6)                                     \
        r0 - occupied (holds rows 3 and 7)                                     \
    */                                                                         \
    /* phase 1 ops (AT&T syntax: vshufps $imm, src2, src1, dst) */             \
    vshufps $0x44, _r5, _r4, _r2   ;   /* r2 = {f5 f4 e5 e4 | f1 f0 e1 e0} */  \
    vshufps $0xEE, _r5, _r4, _r4   ;   /* r4 = {f7 f6 e7 e6 | f3 f2 e3 e2} */  \
    vshufps $0x44, _r7, _r6, _t1   ;   /* t1 = {h5 h4 g5 g4 | h1 h0 g1 g0} */  \
    vshufps $0xEE, _r7, _r6, _r6   ;   /* r6 = {h7 h6 g7 g6 | h3 h2 g3 g2} */  \
    /* phase 2 ops */                                                          \
    vshufps $0xDD, _t1, _r2, _r7   ;   /* r7 = {h5 g5 f5 e5 | h1 g1 f1 e1} */  \
    vshufps $0x88, _r6, _r4, _r5   ;   /* r5 = {h6 g6 f6 e6 | h2 g2 f2 e2} */  \
    vshufps $0xDD, _r6, _r4, _r4   ;   /* r4 = {h7 g7 f7 e7 | h3 g3 f3 e3} */  \
    vshufps $0x88, _t1, _r2, _t1   ;   /* t1 = {h4 g4 f4 e4 | h0 g0 f0 e0} */  \
    /* Phase 3: vperm2f128 - stitch 128-bit halves into full 8-wide rows       \
                                                                               \
      "Busy" registers (holding Phase 2 results):                              \
        _t0 - rows 0,4 {d4 c4 b4 a4 | d0 c0 b0 a0} (top group)                 \
        _t1 - rows 0,4 {h4 g4 f4 e4 | h0 g0 f0 e0} (bottom group)              \
        _r3 - rows 1,5 {d5 c5 b5 a5 | d1 c1 b1 a1} (top)                       \
        _r7 - rows 1,5 {h5 g5 f5 e5 | h1 g1 f1 e1} (bottom)                    \
        _r1 - rows 2,6 {d6 c6 b6 a6 | d2 c2 b2 a2} (top)                       \
        _r5 - rows 2,6 {h6 g6 f6 e6 | h2 g2 f2 e2} (bottom)                    \
        _r0 - rows 3,7 {d7 c7 b7 a7 | d3 c3 b3 a3} (top)                       \
        _r4 - rows 3,7 {h7 g7 f7 e7 | h3 g3 f3 e3} (bottom)                    \
                                                                               \
      FREE: r2, r6                                                             \
                                                                               \
      Wwrite to free regs first                                                \
        1. (_r5, _r1) -> _r6, _r2   then _r5, _r1 are free                     \
        2. (_r7, _r3) -> _r5, _r1   then _r7, _r3 are free                     \
        3. (_r4, _r0) -> _r7, _r3   then _r4, _r0 are free                     \
        4. (_t1, _t0) -> _r4, _r0   done                                       \
    */                                                                         \
    vperm2f128 $0x13, _r1, _r5, _r6 ; /* _r6 = {h6 g6 f6 e6 | d6 c6 b6 a6}  */ \
    vperm2f128 $0x02, _r1, _r5, _r2 ; /* _r2 = {h2 g2 f2 e2 | d2 c2 b2 a2 } */ \
    vperm2f128 $0x13, _r3, _r7, _r5 ; /* _r5 = {h5 g5 f5 e5 | d5 c5 b5 a5 } */ \
    vperm2f128 $0x02, _r3, _r7, _r1 ; /* _r1 = {h1 g1 f1 e1 | d1 c1 b1 a1}  */ \
    vperm2f128 $0x13, _r0, _r4, _r7 ; /* _r7 = {h7 g7 f7 e7 | d7 c7 b7 a7}  */ \
    vperm2f128 $0x02, _r0, _r4, _r3 ; /* _r3 = {h3 g3 f3 e3 | d3 c3 b3 a3}  */ \
    vperm2f128 $0x13, _t0, _t1, _r4 ; /* _r4 = {h4 g4 f4 e4 | d4 c4 b4 a4}  */ \
    vperm2f128 $0x02, _t0, _t1, _r0   /* _r0 = {h0 g0 f0 e0 | d0 c0 b0 a0}  */


