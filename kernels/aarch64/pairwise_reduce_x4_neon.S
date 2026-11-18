/**
 * pairwise_reduce_x4_neon.S - CommP pairwise SHA256 tree reduction, ARM64
 *
 * void pairwise_reduce_buffer_x4_neon(uint8_t *buf, size_t byte_len, size_t out_pair_count)
 *
 *   buf            x0 - AoS buffer [L0|R0|L1|R1|…], each node 32B, pair 64B
 *   byte_len       x1 - buffer size; must equal n_pairs * 64
 *   out_pair_count x2 - stop when n_pairs <= this (0 = full tree reduction)
 *
 * STRATEGY:
 *   Full tree reduction in a single .S call - no Rust roundtrip per level.
 *   Per batch of 4 pairs:
 *     Block 1 - load AoS pairs, rev32 to BE, 64 SHA256 rounds w/ msg schedule
 *     Block 2 - INLINED: W[0..15] constant (SHA256 padding for 64B message).
 *               KW2[i] = K[i]+W2[i] precomputed in .LS_KW2, eliminating
 *               the second function call entirely.
 *   Results written compact (4 × 32B = 128B per batch).
 *
 * STACK FRAME (256B, 16B aligned):
 *   [sp +   0 ..  63]  callee-saved d8-d15      (AAPCS64)
 *   [sp +  64 .. 191]  Merkle-Damgård save:     8 × 16B (4 lanes × abcd+efgh)
 *   [sp + 192 .. 247]  callee-saved x19-x24     (6 × 8B)
 *   [sp + 248 .. 255]  padding
 */

/* ================================================================
 * Cross-platform glue (Apple / Linux)
 * ================================================================ */
#ifdef __APPLE__
#define GLBL(name)      .globl _##name
#define ENTRY(name)     _##name:
#define SIZE(name)
#define ADRP_IV(r)      adrp r, LS_IV@PAGE
#define ADD_IV(r)       add  r, r, LS_IV@PAGEOFF
#define ADRP_K(r)       adrp r, LS_K256@PAGE
#define ADD_K(r)        add  r, r, LS_K256@PAGEOFF
#define ADRP_KW2(r)     adrp r, LS_KW2@PAGE
#define ADD_KW2(r)      add  r, r, LS_KW2@PAGEOFF
#define ADRP_MASK(r)    adrp r, LS_TRUNC254@PAGE
#define ADD_MASK(r)     add  r, r, LS_TRUNC254@PAGEOFF
#else
#define GLBL(name)      .globl name ; .type name, %function
#define ENTRY(name)     name:
#define SIZE(name)      .size name, .-name
#define ADRP_IV(r)      adrp r, LS_IV
#define ADD_IV(r)       add  r, r, :lo12:LS_IV
#define ADRP_K(r)       adrp r, LS_K256
#define ADD_K(r)        add  r, r, :lo12:LS_K256
#define ADRP_KW2(r)     adrp r, LS_KW2
#define ADD_KW2(r)      add  r, r, :lo12:LS_KW2
#define ADRP_MASK(r)    adrp r, LS_TRUNC254
#define ADD_MASK(r)     add  r, r, :lo12:LS_TRUNC254
#endif

    .arch armv8-a+crypto
    .text
    .p2align 3

/* ================================================================
 * Register aliases
 * ================================================================ */

.macro declare_vr  name:req, reg:req
    \name\()_q  .req q\reg
    \name\()_v  .req v\reg
.endm

/* SHA256 working state - 4 independent lanes */
declare_vr  l0_abcd,  0      /* lane 0: {A,B,C,D} */
declare_vr  l0_efgh,  1      /* lane 0: {E,F,G,H} */
declare_vr  l1_abcd,  2
declare_vr  l1_efgh,  3
declare_vr  l2_abcd,  4
declare_vr  l2_efgh,  5
declare_vr  l3_abcd,  6
declare_vr  l3_efgh,  7

/* Per-lane K+W input for sha256h - callee-saved (d8-d11) */
declare_vr  l0_kw,    8
declare_vr  l1_kw,    9
declare_vr  l2_kw,   10
declare_vr  l3_kw,   11

/* Scratch for sha256h2 abcd save - callee-saved (d12-d13) */
declare_vr  sv0,     12      /* save slot 0 */
declare_vr  sv1,     13      /* save slot 1 */

/* Round constant and trunc254 mask - callee-saved (d14-d15) */
declare_vr  rk,      14      /* current 4-round constant vector */
declare_vr  t254,    15      /* trunc254 mask (0x3F at byte 15) */

/* Message schedule - 4 lanes × 4 × 16B = 256B (v16-v31) */
declare_vr  l0_w0,   16
declare_vr  l0_w1,   17
declare_vr  l0_w2,   18
declare_vr  l0_w3,   19
declare_vr  l1_w0,   20
declare_vr  l1_w1,   21
declare_vr  l1_w2,   22
declare_vr  l1_w3,   23
declare_vr  l2_w0,   24
declare_vr  l2_w1,   25
declare_vr  l2_w2,   26
declare_vr  l2_w3,   27
declare_vr  l3_w0,   28
declare_vr  l3_w1,   29
declare_vr  l3_w2,   30
declare_vr  l3_w3,   31

/* GPR aliases */
    buf         .req x19    /* constant buffer base */
    n_pairs     .req x20    /* decreasing per tree level */
    out_count   .req x21    /* constant out_pair_count */
    read_ptr    .req x22    /* read head within level */
    write_ptr   .req x23    /* write head within level */
    tbl_ptr     .req x24    /* current position in K / KW2 table */

/* ================================================================
 * Macro: LOAD_IV
 *
 * Load SHA256 IV into all 4 lanes.
 * Uses x9 as temporary; clobbers v0-v7.
 * ================================================================ */
.macro LOAD_IV
    ADRP_IV(x9)
    ADD_IV(x9)
    ldr     l0_abcd_q, [x9]
    ldr     l0_efgh_q, [x9, #16]
    mov     l1_abcd_v.16b, l0_abcd_v.16b
    mov     l1_efgh_v.16b, l0_efgh_v.16b
    mov     l2_abcd_v.16b, l0_abcd_v.16b
    mov     l2_efgh_v.16b, l0_efgh_v.16b
    mov     l3_abcd_v.16b, l0_abcd_v.16b
    mov     l3_efgh_v.16b, l0_efgh_v.16b
.endm

/* ================================================================
 * Macro: SAVE_STATE
 *
 * Spill current abcd/efgh to stack[64..191] for Merkle-Damgård add-back.
 * ================================================================ */
.macro SAVE_STATE
    stp     l0_abcd_q, l0_efgh_q, [sp, #64]
    stp     l1_abcd_q, l1_efgh_q, [sp, #96]
    stp     l2_abcd_q, l2_efgh_q, [sp, #128]
    stp     l3_abcd_q, l3_efgh_q, [sp, #160]
.endm

/* ================================================================
 * Macro: ADDBACK_STATE
 *
 * Reload saved state and add back (Merkle-Damgård feed-forward).
 * Uses sv0/sv1 as scratch.
 * ================================================================ */
.macro ADDBACK_STATE
    ldp     sv0_q, sv1_q, [sp, #64]
    add     l0_abcd_v.4s, l0_abcd_v.4s, sv0_v.4s
    add     l0_efgh_v.4s, l0_efgh_v.4s, sv1_v.4s
    ldp     sv0_q, sv1_q, [sp, #96]
    add     l1_abcd_v.4s, l1_abcd_v.4s, sv0_v.4s
    add     l1_efgh_v.4s, l1_efgh_v.4s, sv1_v.4s
    ldp     sv0_q, sv1_q, [sp, #128]
    add     l2_abcd_v.4s, l2_abcd_v.4s, sv0_v.4s
    add     l2_efgh_v.4s, l2_efgh_v.4s, sv1_v.4s
    ldp     sv0_q, sv1_q, [sp, #160]
    add     l3_abcd_v.4s, l3_abcd_v.4s, sv0_v.4s
    add     l3_efgh_v.4s, l3_efgh_v.4s, sv1_v.4s
.endm

/* ================================================================
 * Macro: LOAD_4_PAIRS
 *
 * Load 4 × 64B AoS pairs from read_ptr, byte-swap to BE.
 * Advances read_ptr by 256B.
 * Clobbers l{0-3}_w{0-3} (message schedule regs).
 * ================================================================ */
.macro LOAD_4_PAIRS
    ld1     {l0_w0_v.4s-l0_w3_v.4s}, [read_ptr], #64
    ld1     {l1_w0_v.4s-l1_w3_v.4s}, [read_ptr], #64
    ld1     {l2_w0_v.4s-l2_w3_v.4s}, [read_ptr], #64
    ld1     {l3_w0_v.4s-l3_w3_v.4s}, [read_ptr], #64

    /* byte-swap: ARM LE -> SHA256 BE word format */
    REV32_ALL_LANES
.endm

/* ================================================================
 * Macro: SHA_4_ROUNDS_HIGH  wa
 *
 * PIPELINED: 4 SHA256 compression rounds across all 4 lanes.
 *
 * Step 1: sha256h/sha256h2 using the CURRENT l{X}_kw (computed by
 *         the PREVIOUS call or the bootstrap in SHA_ROUNDS_BLOCK1).
 * Step 2: compute NEXT l{X}_kw = K_next + wa  (for the next call).
 *
 * This matches the pipeline trick in the working sha256_x4_aarch64.S
 * where l{X}_tmp0 carries K+W one round group ahead.
 * ================================================================ */
.macro SHA_4_ROUNDS_HIGH  wa:req
    /* --- compress using PREVIOUSLY computed kw --- */
    /* lanes 0-1 interleaved to hide sha256h 2-cycle latency */
    mov     sv0_v.16b, l0_abcd_v.16b
    mov     sv1_v.16b, l1_abcd_v.16b
    sha256h     l0_abcd_q, l0_efgh_q, l0_kw_v.4s
    sha256h     l1_abcd_q, l1_efgh_q, l1_kw_v.4s
    sha256h2    l0_efgh_q, sv0_q, l0_kw_v.4s
    sha256h2    l1_efgh_q, sv1_q, l1_kw_v.4s
    /* lanes 2-3 */
    mov     sv0_v.16b, l2_abcd_v.16b
    mov     sv1_v.16b, l3_abcd_v.16b
    sha256h     l2_abcd_q, l2_efgh_q, l2_kw_v.4s
    sha256h     l3_abcd_q, l3_efgh_q, l3_kw_v.4s
    sha256h2    l2_efgh_q, sv0_q, l2_kw_v.4s
    sha256h2    l3_efgh_q, sv1_q, l3_kw_v.4s
    /* --- compute NEXT kw for the following round group --- */
    ldr     rk_q, [tbl_ptr], #16
    add     l0_kw_v.4s, l0_\wa\()_v.4s, rk_v.4s
    add     l1_kw_v.4s, l1_\wa\()_v.4s, rk_v.4s
    add     l2_kw_v.4s, l2_\wa\()_v.4s, rk_v.4s
    add     l3_kw_v.4s, l3_\wa\()_v.4s, rk_v.4s
.endm

/* ================================================================
 * Macro: SHA_4_ROUNDS_LAST
 *
 * Terminal round group (rounds 60-63): uses current l{X}_kw,
 * does NOT compute a next kw (no more rounds follow).
 * ================================================================ */
.macro SHA_4_ROUNDS_LAST
    mov     sv0_v.16b, l0_abcd_v.16b
    mov     sv1_v.16b, l1_abcd_v.16b
    sha256h     l0_abcd_q, l0_efgh_q, l0_kw_v.4s
    sha256h     l1_abcd_q, l1_efgh_q, l1_kw_v.4s
    sha256h2    l0_efgh_q, sv0_q, l0_kw_v.4s
    sha256h2    l1_efgh_q, sv1_q, l1_kw_v.4s
    mov     sv0_v.16b, l2_abcd_v.16b
    mov     sv1_v.16b, l3_abcd_v.16b
    sha256h     l2_abcd_q, l2_efgh_q, l2_kw_v.4s
    sha256h     l3_abcd_q, l3_efgh_q, l3_kw_v.4s
    sha256h2    l2_efgh_q, sv0_q, l2_kw_v.4s
    sha256h2    l3_efgh_q, sv1_q, l3_kw_v.4s
.endm

/* ================================================================
 * Macro: SHA_4_ROUNDS_LOW  wa wb wc wd
 *
 * One group of 4 rounds WITH message schedule update.
 * sha256su0/sha256su1 updates wa in-place for the next 16 rounds.
 * ================================================================ */
.macro SHA_4_ROUNDS_LOW  wa:req, wb:req, wc:req, wd:req
    sha256su0   l0_\wa\()_v.4s, l0_\wb\()_v.4s
    sha256su0   l1_\wa\()_v.4s, l1_\wb\()_v.4s
    sha256su0   l2_\wa\()_v.4s, l2_\wb\()_v.4s
    sha256su0   l3_\wa\()_v.4s, l3_\wb\()_v.4s
    SHA_4_ROUNDS_HIGH  \wb
    sha256su1   l0_\wa\()_v.4s, l0_\wc\()_v.4s, l0_\wd\()_v.4s
    sha256su1   l1_\wa\()_v.4s, l1_\wc\()_v.4s, l1_\wd\()_v.4s
    sha256su1   l2_\wa\()_v.4s, l2_\wc\()_v.4s, l2_\wd\()_v.4s
    sha256su1   l3_\wa\()_v.4s, l3_\wc\()_v.4s, l3_\wd\()_v.4s
.endm

/* ================================================================
 * Macro: SHA_ROUNDS_BLOCK1
 *
 * Full 64-round SHA256 Block 1 compression across 4 lanes.
 * tbl_ptr must point to LS_K256 on entry; advanced by 16×16=256B.
 * Message schedule in l{0-3}_w{0-3} on entry (from LOAD_4_PAIRS).
 *
 * PIPELINE BOOTSTRAP: pre-compute K[0..3]+W0 into l{X}_kw before
 * the first SHA_4_ROUNDS_LOW. Each subsequent SHA_4_ROUNDS_HIGH
 * uses the CURRENT kw and computes the NEXT one.
 * ================================================================ */
.macro SHA_ROUNDS_BLOCK1
    ADRP_K(tbl_ptr)
    ADD_K(tbl_ptr)
    /* Bootstrap: l{X}_kw = K[0..3] + w0  (consumed by first SHA_4_ROUNDS_HIGH) */
    ldr     rk_q, [tbl_ptr], #16
    add     l0_kw_v.4s, l0_w0_v.4s, rk_v.4s
    add     l1_kw_v.4s, l1_w0_v.4s, rk_v.4s
    add     l2_kw_v.4s, l2_w0_v.4s, rk_v.4s
    add     l3_kw_v.4s, l3_w0_v.4s, rk_v.4s
    /* Rounds  0-47: with message schedule (sha256su0/su1) */
    /* Rounds  0- 3 */  SHA_4_ROUNDS_LOW  w0, w1, w2, w3
    /* Rounds  4- 7 */  SHA_4_ROUNDS_LOW  w1, w2, w3, w0
    /* Rounds  8-11 */  SHA_4_ROUNDS_LOW  w2, w3, w0, w1
    /* Rounds 12-15 */  SHA_4_ROUNDS_LOW  w3, w0, w1, w2
    /* Rounds 16-19 */  SHA_4_ROUNDS_LOW  w0, w1, w2, w3
    /* Rounds 20-23 */  SHA_4_ROUNDS_LOW  w1, w2, w3, w0
    /* Rounds 24-27 */  SHA_4_ROUNDS_LOW  w2, w3, w0, w1
    /* Rounds 28-31 */  SHA_4_ROUNDS_LOW  w3, w0, w1, w2
    /* Rounds 32-35 */  SHA_4_ROUNDS_LOW  w0, w1, w2, w3
    /* Rounds 36-39 */  SHA_4_ROUNDS_LOW  w1, w2, w3, w0
    /* Rounds 40-43 */  SHA_4_ROUNDS_LOW  w2, w3, w0, w1
    /* Rounds 44-47 */  SHA_4_ROUNDS_LOW  w3, w0, w1, w2
    /* Rounds 48-63: no message schedule, pipeline continues.
     * After rounds 44-47, kw = K[48..51]+w0  (computed inside last LOW).
     * Each HIGH uses current kw and prepares the next from the given word. */
    /* Rounds 48-51 */  SHA_4_ROUNDS_HIGH  w1
    /* Rounds 52-55 */  SHA_4_ROUNDS_HIGH  w2
    /* Rounds 56-59 */  SHA_4_ROUNDS_HIGH  w3
    /* Rounds 60-63 */  SHA_4_ROUNDS_LAST
.endm

/* ================================================================
 * Macro: SHA_4_ROUNDS_KW2
 *
 * One group of 4 Block-2 rounds using precomputed KW2 constant.
 * KW2[i] = K[i] + W2[i] - no per-lane message schedule needed.
 * tbl_ptr (pointing into LS_KW2) advances by 16B.
 * ================================================================ */
.macro SHA_4_ROUNDS_KW2
    ldr     rk_q, [tbl_ptr], #16    /* rk = KW2[round_group]: same for all lanes */
    /* lanes 0-1 interleaved */
    mov     l0_kw_v.16b, rk_v.16b
    mov     l1_kw_v.16b, rk_v.16b
    mov     sv0_v.16b, l0_abcd_v.16b
    mov     sv1_v.16b, l1_abcd_v.16b
    sha256h     l0_abcd_q, l0_efgh_q, l0_kw_v.4s
    sha256h     l1_abcd_q, l1_efgh_q, l1_kw_v.4s
    sha256h2    l0_efgh_q, sv0_q, l0_kw_v.4s
    sha256h2    l1_efgh_q, sv1_q, l1_kw_v.4s
    /* lanes 2-3 */
    mov     l2_kw_v.16b, rk_v.16b
    mov     l3_kw_v.16b, rk_v.16b
    mov     sv0_v.16b, l2_abcd_v.16b
    mov     sv1_v.16b, l3_abcd_v.16b
    sha256h     l2_abcd_q, l2_efgh_q, l2_kw_v.4s
    sha256h     l3_abcd_q, l3_efgh_q, l3_kw_v.4s
    sha256h2    l2_efgh_q, sv0_q, l2_kw_v.4s
    sha256h2    l3_efgh_q, sv1_q, l3_kw_v.4s
.endm

/* ================================================================
 * Macro: SHA_ROUNDS_KW2
 *
 * Full 64-round Block 2 using precomputed KW2 constants.
 * 16 groups × 4 rounds = 64 rounds.
 * ================================================================ */
.macro SHA_ROUNDS_KW2
    ADRP_KW2(tbl_ptr)
    ADD_KW2(tbl_ptr)
    SHA_4_ROUNDS_KW2    /* rounds  0- 3 */
    SHA_4_ROUNDS_KW2    /* rounds  4- 7 */
    SHA_4_ROUNDS_KW2    /* rounds  8-11 */
    SHA_4_ROUNDS_KW2    /* rounds 12-15 */
    SHA_4_ROUNDS_KW2    /* rounds 16-19 */
    SHA_4_ROUNDS_KW2    /* rounds 20-23 */
    SHA_4_ROUNDS_KW2    /* rounds 24-27 */
    SHA_4_ROUNDS_KW2    /* rounds 28-31 */
    SHA_4_ROUNDS_KW2    /* rounds 32-35 */
    SHA_4_ROUNDS_KW2    /* rounds 36-39 */
    SHA_4_ROUNDS_KW2    /* rounds 40-43 */
    SHA_4_ROUNDS_KW2    /* rounds 44-47 */
    SHA_4_ROUNDS_KW2    /* rounds 48-51 */
    SHA_4_ROUNDS_KW2    /* rounds 52-55 */
    SHA_4_ROUNDS_KW2    /* rounds 56-59 */
    SHA_4_ROUNDS_KW2    /* rounds 60-63 */
.endm

/* ================================================================
 * Macro: TRUNC254
 *
 * Clear top 2 bits of SHA256 byte 31 (CommP 254-bit truncation).
 * In ARM BE word storage: H is efgh[3], byte 31 = byte 15 of efgh register.
 * t254 must hold LS_TRUNC254 mask (loaded once at batch entry).
 * ================================================================ */
.macro TRUNC254
    and     l0_efgh_v.16b, l0_efgh_v.16b, t254_v.16b
    and     l1_efgh_v.16b, l1_efgh_v.16b, t254_v.16b
    and     l2_efgh_v.16b, l2_efgh_v.16b, t254_v.16b
    and     l3_efgh_v.16b, l3_efgh_v.16b, t254_v.16b
.endm

/* ================================================================
 * Macro: STORE_4_RESULTS
 *
 * Convert state to BE bytes and write 4 × 32B compact to write_ptr.
 * Applies rev32 (SHA state LE words -> CommP BE bytes).
 * Advances write_ptr by 128B.
 * Clobbers abcd/efgh registers.
 * ================================================================ */
.macro STORE_4_RESULTS
    rev32   l0_abcd_v.16b, l0_abcd_v.16b
    rev32   l0_efgh_v.16b, l0_efgh_v.16b
    rev32   l1_abcd_v.16b, l1_abcd_v.16b
    rev32   l1_efgh_v.16b, l1_efgh_v.16b
    rev32   l2_abcd_v.16b, l2_abcd_v.16b
    rev32   l2_efgh_v.16b, l2_efgh_v.16b
    rev32   l3_abcd_v.16b, l3_abcd_v.16b
    rev32   l3_efgh_v.16b, l3_efgh_v.16b
    stp     l0_abcd_q, l0_efgh_q, [write_ptr]       /* H0 */
    stp     l1_abcd_q, l1_efgh_q, [write_ptr, #32]  /* H1 - paired with H0 */
    stp     l2_abcd_q, l2_efgh_q, [write_ptr, #64]  /* H2 */
    stp     l3_abcd_q, l3_efgh_q, [write_ptr, #96]  /* H3 - paired with H2 */
    add     write_ptr, write_ptr, #128
.endm

/* ================================================================
 * Macro: REV32_ALL_LANES
 * Byte-swap all 16 message-schedule vectors (LE -> SHA256 BE).
 * ================================================================ */
.macro REV32_ALL_LANES
    rev32   l0_w0_v.16b, l0_w0_v.16b
    rev32   l0_w1_v.16b, l0_w1_v.16b
    rev32   l0_w2_v.16b, l0_w2_v.16b
    rev32   l0_w3_v.16b, l0_w3_v.16b
    rev32   l1_w0_v.16b, l1_w0_v.16b
    rev32   l1_w1_v.16b, l1_w1_v.16b
    rev32   l1_w2_v.16b, l1_w2_v.16b
    rev32   l1_w3_v.16b, l1_w3_v.16b
    rev32   l2_w0_v.16b, l2_w0_v.16b
    rev32   l2_w1_v.16b, l2_w1_v.16b
    rev32   l2_w2_v.16b, l2_w2_v.16b
    rev32   l2_w3_v.16b, l2_w3_v.16b
    rev32   l3_w0_v.16b, l3_w0_v.16b
    rev32   l3_w1_v.16b, l3_w1_v.16b
    rev32   l3_w2_v.16b, l3_w2_v.16b
    rev32   l3_w3_v.16b, l3_w3_v.16b
.endm

/* ================================================================
 * Macro: LOAD_TAIL_PAIRS
 *
 * Load 1 or 2 pairs from read_ptr. x10 = active count (1 or 2).
 * Inactive lanes get lane 0's data (result discarded).
 * Applies REV32_ALL_LANES.
 * ================================================================ */
.macro LOAD_TAIL_PAIRS
    /* lane 0: always load */
    ld1     {l0_w0_v.4s-l0_w3_v.4s}, [read_ptr]
    /* lanes 2-3: always dup lane 0 (max 2 active pairs) */
    mov     l2_w0_v.16b, l0_w0_v.16b
    mov     l2_w1_v.16b, l0_w1_v.16b
    mov     l2_w2_v.16b, l0_w2_v.16b
    mov     l2_w3_v.16b, l0_w3_v.16b
    mov     l3_w0_v.16b, l0_w0_v.16b
    mov     l3_w1_v.16b, l0_w1_v.16b
    mov     l3_w2_v.16b, l0_w2_v.16b
    mov     l3_w3_v.16b, l0_w3_v.16b
    /* lane 1: real pair if x10 == 2, else dup lane 0 */
    cmp     x10, #2
    blt     1f
    add     x9, read_ptr, #64
    ld1     {l1_w0_v.4s-l1_w3_v.4s}, [x9]
    b       2f
1:
    mov     l1_w0_v.16b, l0_w0_v.16b
    mov     l1_w1_v.16b, l0_w1_v.16b
    mov     l1_w2_v.16b, l0_w2_v.16b
    mov     l1_w3_v.16b, l0_w3_v.16b
2:
    REV32_ALL_LANES
.endm

/* ================================================================
 * Macro: STORE_TAIL_RESULTS
 *
 * Store 1 or 2 results from x10 active lanes.
 * Applies rev32 (SHA state -> CommP BE bytes).
 * Advances write_ptr by x10 * 32.
 * ================================================================ */
.macro STORE_TAIL_RESULTS
    rev32   l0_abcd_v.16b, l0_abcd_v.16b
    rev32   l0_efgh_v.16b, l0_efgh_v.16b
    stp     l0_abcd_q, l0_efgh_q, [write_ptr]
    cmp     x10, #2
    blt     1f
    rev32   l1_abcd_v.16b, l1_abcd_v.16b
    rev32   l1_efgh_v.16b, l1_efgh_v.16b
    stp     l1_abcd_q, l1_efgh_q, [write_ptr, #32]
1:
    add     write_ptr, write_ptr, x10, lsl #5
.endm

.macro SHA_CORE
    LOAD_IV
    SAVE_STATE
    SHA_ROUNDS_BLOCK1
    ADDBACK_STATE
    SAVE_STATE
    SHA_ROUNDS_KW2
    ADDBACK_STATE
    TRUNC254
.endm

/* ================================================================
 * pairwise_reduce_buffer_x4_neon - main entry
 * ================================================================ */
    .p2align 2
GLBL(pairwise_reduce_buffer_x4_neon)
ENTRY(pairwise_reduce_buffer_x4_neon)

    /* Early exit: nothing to do */
    cbz     x1, .Lexit_noop
    lsr     x9, x1, #6          /* x9 = n_pairs = byte_len / 64 */
    cmp     x9, x2
    bls     .Lexit_noop

    /* ---- Prologue: 256B frame ---- */
    sub     sp, sp, #256
    stp     d8,  d9,  [sp,  #0]
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    stp     x19, x20, [sp, #192]
    stp     x21, x22, [sp, #208]
    stp     x23, x24, [sp, #224]
    str     x30, [sp, #240]

    /* Load trunc254 mask once into t254 (callee-saved d15) */
    ADRP_MASK(x9)
    ADD_MASK(x9)
    ldr     t254_q, [x9]

    /* Set up level-loop GPRs */
    mov     buf,       x0
    lsr     n_pairs,   x1, #6   /* n_pairs = byte_len / 64 */
    mov     out_count, x2

/* ================================================================
 * .Llevel_loop - outer loop: one iteration = one tree level
 * ================================================================ */
.Llevel_loop:
    mov     read_ptr,  buf
    mov     write_ptr, buf
    cmp     n_pairs, out_count
    bls     .Lexit_restore

    /* full_batches = n_pairs / 4 */
    lsr     x11, n_pairs, #2      /* x11 = full batches */
    cbz     x11, .Ltail_only

/* ================================================================
 * .Lbatch_loop - inner loop: 4 pairs per iteration
 * ================================================================ */
.Lbatch_loop:
    LOAD_4_PAIRS
    SHA_CORE
    STORE_4_RESULTS
    subs    x11, x11, #1
    bne     .Lbatch_loop

    /* ---- Tail: remaining < 4 pairs ---- */
.Ltail_only:
    and     x10, n_pairs, #3    /* x10 = n_pairs % 4 */
    cbz     x10, .Llevel_done
    LOAD_TAIL_PAIRS
    SHA_CORE
    STORE_TAIL_RESULTS

.Llevel_done:
    lsr     n_pairs, n_pairs, #1    /* n_pairs /= 2 */
    b       .Llevel_loop

/* ================================================================
 * Epilogue
 * ================================================================ */
.Lexit_restore:
    ldp     d8,  d9,  [sp,  #0]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     x19, x20, [sp, #192]
    ldp     x21, x22, [sp, #208]
    ldp     x23, x24, [sp, #224]
    ldr     x30, [sp, #240]
    add     sp, sp, #256
.Lexit_noop:
    ret

SIZE(pairwise_reduce_buffer_x4_neon)

/* ================================================================
 * Read-only data
 * ================================================================ */
#ifdef __APPLE__
    .section __TEXT,__const
#else
    .section .rodata,"a",@progbits
#endif
    .align 4

/* SHA256 IV (FIPS 180-4 §5.3.3) - big-endian u32 */
LS_IV:
    .word 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a   /* A B C D */
    .word 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19   /* E F G H */

/* SHA256 round constants K[0..63] - big-endian (used with rev32 input) */
    .align 4
LS_K256:
    .word 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5
    .word 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5
    .word 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3
    .word 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174
    .word 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc
    .word 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da
    .word 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7
    .word 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967
    .word 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13
    .word 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85
    .word 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3
    .word 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070
    .word 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5
    .word 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3
    .word 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208
    .word 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2

/* KW2[i] = K[i] + W2[i] - precomputed Block-2 constants.
 * W2 is the SHA256 message schedule of the constant padding block:
 *   W2[0]=0x80000000  W2[1..14]=0  W2[15]=0x00000200
 *   W2[16..63] derived by standard expansion.
 * Stored as big-endian u32 (same format as LS_K256). */
    .align 4
LS_KW2:
    .word 0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5   /* rounds  0- 3 */
    .word 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5   /* rounds  4- 7 */
    .word 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3   /* rounds  8-11 */
    .word 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374   /* rounds 12-15 */
    .word 0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254   /* rounds 16-19 */
    .word 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa   /* rounds 20-23 */
    .word 0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7   /* rounds 24-27 */
    .word 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0   /* rounds 28-31 */
    .word 0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd   /* rounds 32-35 */
    .word 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16   /* rounds 36-39 */
    .word 0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537   /* rounds 40-43 */
    .word 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37   /* rounds 44-47 */
    .word 0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7   /* rounds 48-51 */
    .word 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890   /* rounds 52-55 */
    .word 0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c   /* rounds 56-59 */
    .word 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76   /* rounds 60-63 */

/* trunc254 mask: 0x3F at byte 15 of 128-bit vector (clears top 2 bits of H).
 * In ARM SIMD, efgh register has H at S[3] (bytes 12-15), MSByte at byte 15. */
    .align 4
LS_TRUNC254:
    .byte 0xff,0xff,0xff,0xff, 0xff,0xff,0xff,0xff
    .byte 0xff,0xff,0xff,0xff, 0x3f,0xff,0xff,0xff

#ifndef __APPLE__
    .section .note.GNU-stack,"",@progbits
#endif