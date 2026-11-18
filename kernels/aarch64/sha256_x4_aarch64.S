/**
 * sha256_x4_aarch64.S - 4-lane SHA256 using ARM Crypto Extensions
 *
 * Adapted from ISA-L sha256_mb_x4_ce.S (Copyright 2019 Arm Corporation)
 * Original: 4 independent SHA256 hashes interleaved to hide sha256h latency.
 *
 * Simplified ABI for CommP pairwise reduction:
 *   void sha256_x4_aarch64(args: *mut Sha256x4Args, num_blocks: u64)
 *     x0 = pointer to Args struct
 *     x1 = number of 64-byte blocks to process (must be >= 1)
 *
 * Args struct layout (160 bytes, 16-byte aligned):
 *   0x00: lane 0 abcd  (4 × u32, 16 bytes) - SHA256 state {A,B,C,D}
 *   0x10: lane 0 efgh  (4 × u32, 16 bytes) - SHA256 state {E,F,G,H}
 *   0x20: lane 1 abcd
 *   0x30: lane 1 efgh
 *   0x40: lane 2 abcd
 *   0x50: lane 2 efgh
 *   0x60: lane 3 abcd
 *   0x70: lane 3 efgh
 *   0x80: data_ptr[0]  (8 bytes) - pointer to lane 0 input
 *   0x88: data_ptr[1]
 *   0x90: data_ptr[2]
 *   0x98: data_ptr[3]
 *
 * After return, digest[] contains updated state and data_ptr[] are advanced
 * by num_blocks * 64 bytes each.
 *
 * License: BSD-3-Clause (ISA-L original)
 */

/* ---- Cross-platform glue ---- */
#ifdef __APPLE__
#define FUNC_ENTRY(name)    \
    .globl _##name          \
    _##name:
#define FUNC_END(name)
#define ADRP_KEY(reg)       adrp reg, KEY@PAGE
#define ADD_KEY(reg)        add  reg, reg, KEY@PAGEOFF
#else
#define FUNC_ENTRY(name)    \
    .globl name             \
    .type name, %function   \
    name:
#define FUNC_END(name)      .size name, .-name
#define ADRP_KEY(reg)       adrp reg, KEY
#define ADD_KEY(reg)        add  reg, reg, :lo12:KEY
#endif

    .arch armv8-a+crypto
    .text
    .align 2
    .p2align 3

/* ================================================================
 * Macros - verbatim from ISA-L (register alias helper)
 * ================================================================ */

.macro declare_var_vector_reg name:req, reg:req
    \name\()_q  .req q\reg
    \name\()_v  .req v\reg
    \name\()_s  .req s\reg
.endm

/**
 * sha256_4_rounds_high - rounds 48-63 (no message schedule)
 * Processes all 4 lanes for one set of 4 rounds.
 */
.macro sha256_4_rounds_high msg:req, tmp0:req
    /* Lanes 0-1 */
    ldr     key_q, [tmp]
    mov     tmp0_v.16b, l0_\tmp0\()_v.16b
    mov     tmp1_v.16b, l1_\tmp0\()_v.16b
    add     l0_\tmp0\()_v.4s, l0_\msg\()_v.4s, key_v.4s
    add     l1_\tmp0\()_v.4s, l1_\msg\()_v.4s, key_v.4s
    mov     tmp2_v.16b, l0_abcd_v.16b
    mov     tmp3_v.16b, l1_abcd_v.16b
    sha256h     l0_abcd_q, l0_efgh_q, tmp0_v.4s
    sha256h     l1_abcd_q, l1_efgh_q, tmp1_v.4s
    sha256h2    l0_efgh_q, tmp2_q, tmp0_v.4s
    sha256h2    l1_efgh_q, tmp3_q, tmp1_v.4s

    /* Lanes 2-3 */
    ldr     key_q, [tmp]
    mov     tmp0_v.16b, l2_\tmp0\()_v.16b
    mov     tmp1_v.16b, l3_\tmp0\()_v.16b
    add     tmp, tmp, 16
    add     l2_\tmp0\()_v.4s, l2_\msg\()_v.4s, key_v.4s
    add     l3_\tmp0\()_v.4s, l3_\msg\()_v.4s, key_v.4s
    mov     tmp2_v.16b, l2_abcd_v.16b
    mov     tmp3_v.16b, l3_abcd_v.16b
    sha256h     l2_abcd_q, l2_efgh_q, tmp0_v.4s
    sha256h     l3_abcd_q, l3_efgh_q, tmp1_v.4s
    sha256h2    l2_efgh_q, tmp2_q, tmp0_v.4s
    sha256h2    l3_efgh_q, tmp3_q, tmp1_v.4s
.endm

/**
 * sha256_4_rounds_low - rounds 0-47 (with message schedule)
 */
.macro sha256_4_rounds_low msg0:req, msg1:req, msg2:req, msg3:req, tmp0:req
    sha256su0   l0_\msg0\()_v.4s, l0_\msg1\()_v.4s
    sha256su0   l1_\msg0\()_v.4s, l1_\msg1\()_v.4s
    sha256su0   l2_\msg0\()_v.4s, l2_\msg1\()_v.4s
    sha256su0   l3_\msg0\()_v.4s, l3_\msg1\()_v.4s
    sha256_4_rounds_high \msg1, \tmp0
    sha256su1   l0_\msg0\()_v.4s, l0_\msg2\()_v.4s, l0_\msg3\()_v.4s
    sha256su1   l1_\msg0\()_v.4s, l1_\msg2\()_v.4s, l1_\msg3\()_v.4s
    sha256su1   l2_\msg0\()_v.4s, l2_\msg2\()_v.4s, l2_\msg3\()_v.4s
    sha256su1   l3_\msg0\()_v.4s, l3_\msg2\()_v.4s, l3_\msg3\()_v.4s
.endm


/* ================================================================
 * Register aliases
 * ================================================================ */

/* Round key (loaded from table each set of 4 rounds) */
    declare_var_vector_reg  key, 15

/* Digest state - 4 lanes × {abcd, efgh} */
    declare_var_vector_reg  l0_abcd, 0
    declare_var_vector_reg  l0_efgh, 1
    declare_var_vector_reg  l1_abcd, 2
    declare_var_vector_reg  l1_efgh, 3
    declare_var_vector_reg  l2_abcd, 4
    declare_var_vector_reg  l2_efgh, 5
    declare_var_vector_reg  l3_abcd, 6
    declare_var_vector_reg  l3_efgh, 7

/* Saved digest (aliased with msg0/msg1 - only used after round 48) */
    declare_var_vector_reg  l0_abcd_saved, 20
    declare_var_vector_reg  l0_efgh_saved, 21
    declare_var_vector_reg  l1_abcd_saved, 16
    declare_var_vector_reg  l1_efgh_saved, 17
    declare_var_vector_reg  l2_abcd_saved, 24
    declare_var_vector_reg  l2_efgh_saved, 25
    declare_var_vector_reg  l3_abcd_saved, 28
    declare_var_vector_reg  l3_efgh_saved, 29

/* Per-lane temporaries (key+msg sum for sha256h input) */
    declare_var_vector_reg  l0_tmp0, 8
    declare_var_vector_reg  l1_tmp0, 9
    declare_var_vector_reg  l2_tmp0, 10
    declare_var_vector_reg  l3_tmp0, 11

/* Scratch registers */
    declare_var_vector_reg  tmp0, 12
    declare_var_vector_reg  tmp1, 13
    declare_var_vector_reg  tmp2, 14
    declare_var_vector_reg  tmp3, 15

/* Message schedule - 4 lanes × 4 words (aliased with saved digest) */
    declare_var_vector_reg  l0_msg0, 16
    declare_var_vector_reg  l0_msg1, 17
    declare_var_vector_reg  l0_msg2, 18
    declare_var_vector_reg  l0_msg3, 19
    declare_var_vector_reg  l1_msg0, 20
    declare_var_vector_reg  l1_msg1, 21
    declare_var_vector_reg  l1_msg2, 22
    declare_var_vector_reg  l1_msg3, 23
    declare_var_vector_reg  l2_msg0, 24
    declare_var_vector_reg  l2_msg1, 25
    declare_var_vector_reg  l2_msg2, 26
    declare_var_vector_reg  l2_msg3, 27
    declare_var_vector_reg  l3_msg0, 28
    declare_var_vector_reg  l3_msg1, 29
    declare_var_vector_reg  l3_msg2, 30
    declare_var_vector_reg  l3_msg3, 31


/* GPR aliases */
    l0_data     .req x5
    l1_data     .req x6
    l2_data     .req x7
    l3_data     .req x8
    tmp         .req x9
    args_save   .req x10
    len         .req w11


/* ================================================================
 * Function entry
 * ================================================================ */

#ifdef __APPLE__
    .globl _sha256_x4_aarch64
    .p2align 2
_sha256_x4_aarch64:
#else
    .globl sha256_x4_aarch64
    .type sha256_x4_aarch64, %function
    .p2align 2
sha256_x4_aarch64:
#endif
    /* Early exit if num_blocks == 0 */
    cbz     x1, .Lexit_noop

    /* Save callee-saved SIMD regs d8-d15 (AAPCS64) */
    stp     d8, d9, [sp, -64]!
    stp     d10, d11, [sp, 16]
    stp     d12, d13, [sp, 32]
    stp     d14, d15, [sp, 48]

    /* Save args pointer in caller-saved x10 */
    mov     args_save, x0

    /* Move num_blocks to our len register */
    mov     len, w1

    /* Load data pointers from Args + 0x80 */
    ldr     l0_data, [args_save, 0x80]
    ldr     l1_data, [args_save, 0x88]
    ldr     l2_data, [args_save, 0x90]
    ldr     l3_data, [args_save, 0x98]

    /* Load initial digest state from Args */
    ldr     l0_abcd_q, [args_save, 0x00]
    ldr     l0_efgh_q, [args_save, 0x10]
    ldr     l1_abcd_q, [args_save, 0x20]
    ldr     l1_efgh_q, [args_save, 0x30]
    ldr     l2_abcd_q, [args_save, 0x40]
    ldr     l2_efgh_q, [args_save, 0x50]
    ldr     l3_abcd_q, [args_save, 0x60]
    ldr     l3_efgh_q, [args_save, 0x70]


/* ================================================================
 * Main block loop - processes one 64-byte block per lane per iteration
 * ================================================================ */
.Lstart_loop:
    /* Load SHA256 round constants table address */
    ADRP_KEY(tmp)
    ADD_KEY(tmp)

    /* Load 64 bytes of input per lane and byte-swap to big-endian */
    ld1     {l0_msg0_v.4s-l0_msg3_v.4s}, [l0_data]
    ld1     {l1_msg0_v.4s-l1_msg3_v.4s}, [l1_data]
    ld1     {l2_msg0_v.4s-l2_msg3_v.4s}, [l2_data]
    ld1     {l3_msg0_v.4s-l3_msg3_v.4s}, [l3_data]
    ldr     key_q, [tmp]
    add     tmp, tmp, 16

    /* Advance data pointers */
    add     l0_data, l0_data, 64
    add     l1_data, l1_data, 64
    add     l2_data, l2_data, 64
    add     l3_data, l3_data, 64

    /* Decrement block count */
    sub     len, len, #1
    cmp     len, 0

    /* Byte-swap: ARM is little-endian, SHA256 is big-endian */
    rev32   l0_msg0_v.16b, l0_msg0_v.16b
    rev32   l0_msg1_v.16b, l0_msg1_v.16b
    add     l0_tmp0_v.4s, l0_msg0_v.4s, key_v.4s
    rev32   l0_msg2_v.16b, l0_msg2_v.16b
    rev32   l0_msg3_v.16b, l0_msg3_v.16b

    rev32   l1_msg0_v.16b, l1_msg0_v.16b
    rev32   l1_msg1_v.16b, l1_msg1_v.16b
    add     l1_tmp0_v.4s, l1_msg0_v.4s, key_v.4s
    rev32   l1_msg2_v.16b, l1_msg2_v.16b
    rev32   l1_msg3_v.16b, l1_msg3_v.16b

    rev32   l2_msg0_v.16b, l2_msg0_v.16b
    rev32   l2_msg1_v.16b, l2_msg1_v.16b
    add     l2_tmp0_v.4s, l2_msg0_v.4s, key_v.4s
    rev32   l2_msg2_v.16b, l2_msg2_v.16b
    rev32   l2_msg3_v.16b, l2_msg3_v.16b

    rev32   l3_msg0_v.16b, l3_msg0_v.16b
    rev32   l3_msg1_v.16b, l3_msg1_v.16b
    add     l3_tmp0_v.4s, l3_msg0_v.4s, key_v.4s
    rev32   l3_msg2_v.16b, l3_msg2_v.16b
    rev32   l3_msg3_v.16b, l3_msg3_v.16b


    /* Rounds 0-47: message schedule + compression (12 × 4 rounds) */
    sha256_4_rounds_low msg0, msg1, msg2, msg3, tmp0    /* rounds  0- 3 */
    sha256_4_rounds_low msg1, msg2, msg3, msg0, tmp0    /* rounds  4- 7 */
    sha256_4_rounds_low msg2, msg3, msg0, msg1, tmp0    /* rounds  8-11 */
    sha256_4_rounds_low msg3, msg0, msg1, msg2, tmp0    /* rounds 12-15 */
    sha256_4_rounds_low msg0, msg1, msg2, msg3, tmp0    /* rounds 16-19 */
    sha256_4_rounds_low msg1, msg2, msg3, msg0, tmp0    /* rounds 20-23 */
    sha256_4_rounds_low msg2, msg3, msg0, msg1, tmp0    /* rounds 24-27 */
    sha256_4_rounds_low msg3, msg0, msg1, msg2, tmp0    /* rounds 28-31 */
    sha256_4_rounds_low msg0, msg1, msg2, msg3, tmp0    /* rounds 32-35 */
    sha256_4_rounds_low msg1, msg2, msg3, msg0, tmp0    /* rounds 36-39 */
    sha256_4_rounds_low msg2, msg3, msg0, msg1, tmp0    /* rounds 40-43 */
    sha256_4_rounds_low msg3, msg0, msg1, msg2, tmp0    /* rounds 44-47 */


    /* Rounds 48-51 */
    sha256_4_rounds_high msg1, tmp0

    /*
     * After round 51, msg0/msg1 registers (v16-v17, v20-v21, v24-v25, v28-v29)
     * are free. Load saved digest into those aliased registers for the
     * final add-back.
     */
    ldr     l0_abcd_saved_q, [args_save, 0x00]
    ldr     l1_abcd_saved_q, [args_save, 0x20]
    ldr     l2_abcd_saved_q, [args_save, 0x40]
    ldr     l3_abcd_saved_q, [args_save, 0x60]
    ldr     l0_efgh_saved_q, [args_save, 0x10]
    ldr     l1_efgh_saved_q, [args_save, 0x30]
    ldr     l2_efgh_saved_q, [args_save, 0x50]
    ldr     l3_efgh_saved_q, [args_save, 0x70]

    /* Rounds 52-55, 56-59 */
    sha256_4_rounds_high msg2, tmp0
    sha256_4_rounds_high msg3, tmp0

    /* Rounds 60-63 - inline, no message schedule */
    mov     tmp2_v.16b, l0_abcd_v.16b
    sha256h     l0_abcd_q, l0_efgh_q, l0_tmp0_v.4s
    sha256h2    l0_efgh_q, tmp2_q, l0_tmp0_v.4s

    mov     tmp2_v.16b, l1_abcd_v.16b
    sha256h     l1_abcd_q, l1_efgh_q, l1_tmp0_v.4s
    sha256h2    l1_efgh_q, tmp2_q, l1_tmp0_v.4s

    mov     tmp2_v.16b, l2_abcd_v.16b
    sha256h     l2_abcd_q, l2_efgh_q, l2_tmp0_v.4s
    sha256h2    l2_efgh_q, tmp2_q, l2_tmp0_v.4s

    mov     tmp2_v.16b, l3_abcd_v.16b
    sha256h     l3_abcd_q, l3_efgh_q, l3_tmp0_v.4s
    sha256h2    l3_efgh_q, tmp2_q, l3_tmp0_v.4s

    /* Add saved state to compression output */
    add     l0_abcd_v.4s, l0_abcd_v.4s, l0_abcd_saved_v.4s
    add     l0_efgh_v.4s, l0_efgh_v.4s, l0_efgh_saved_v.4s
    add     l1_abcd_v.4s, l1_abcd_v.4s, l1_abcd_saved_v.4s
    add     l1_efgh_v.4s, l1_efgh_v.4s, l1_efgh_saved_v.4s
    add     l2_abcd_v.4s, l2_abcd_v.4s, l2_abcd_saved_v.4s
    add     l2_efgh_v.4s, l2_efgh_v.4s, l2_efgh_saved_v.4s
    add     l3_abcd_v.4s, l3_abcd_v.4s, l3_abcd_saved_v.4s
    add     l3_efgh_v.4s, l3_efgh_v.4s, l3_efgh_saved_v.4s

    /* Store updated digest to Args (needed for next iteration's saved reload) */
    str     l0_abcd_q, [args_save, 0x00]
    str     l0_efgh_q, [args_save, 0x10]
    str     l1_abcd_q, [args_save, 0x20]
    str     l1_efgh_q, [args_save, 0x30]
    str     l2_abcd_q, [args_save, 0x40]
    str     l2_efgh_q, [args_save, 0x50]
    str     l3_abcd_q, [args_save, 0x60]
    str     l3_efgh_q, [args_save, 0x70]

    bgt     .Lstart_loop


/* ================================================================
 * Loop done - store advanced data pointers back to Args
 * ================================================================ */
    str     l0_data, [args_save, 0x80]
    str     l1_data, [args_save, 0x88]
    str     l2_data, [args_save, 0x90]
    str     l3_data, [args_save, 0x98]

    /* Restore callee-saved SIMD regs */
    ldp     d10, d11, [sp, 16]
    ldp     d12, d13, [sp, 32]
    ldp     d14, d15, [sp, 48]
    ldp     d8, d9, [sp], 64
    ret

.Lexit_noop:
    ret

#ifndef __APPLE__
    .size sha256_x4_aarch64, .-sha256_x4_aarch64
#endif


/* ================================================================
 * SHA256 round constants K[0..63] (FIPS 180-4 §4.2.2)
 * Big-endian u32 - consumed by add v.4s after rev32 byte-swap.
 * ================================================================ */
#ifdef __APPLE__
    .section __TEXT,__const
#else
    .section .rodata,"a",@progbits
#endif
    .align 4
KEY:
    .word 0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5  /* K[ 0.. 3] */
    .word 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5  /* K[ 4.. 7] */
    .word 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3  /* K[ 8..11] */
    .word 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174  /* K[12..15] */
    .word 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC  /* K[16..19] */
    .word 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA  /* K[20..23] */
    .word 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7  /* K[24..27] */
    .word 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967  /* K[28..31] */
    .word 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13  /* K[32..35] */
    .word 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85  /* K[36..39] */
    .word 0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3  /* K[40..43] */
    .word 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070  /* K[44..47] */
    .word 0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5  /* K[48..51] */
    .word 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3  /* K[52..55] */
    .word 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208  /* K[56..59] */
    .word 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2  /* K[60..63] */