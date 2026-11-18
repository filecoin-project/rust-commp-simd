/*
 * sha256_commp_rodata.h - shared read-only constants for x86 CommP kernels
 *
 * Include at the bottom of each .S file (after .text), inside or just
 * before the .rodata section. All labels use .L prefix (GAS-local) so
 * multiple .o files can include without linker conflictss
 *
 * Usage:
 *   .section .rodata,"a",@progbits
 *   #include "sha256_commp_rodata.h"
 *   .section .note.GNU-stack,"",@progbits
 */

#ifndef SHA256_COMMP_RODATA_H
#define SHA256_COMMP_RODATA_H

/* ---- SHA256 initial hash values (FIPS 180-4) ---- */
.align 64
.LS_IV:
    .long 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a
    .long 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19

/* ---- SHA256 round constants K[0..63] (FIPS 180-4) ---- */
.align 64
.LS_K256:
    .long 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5
    .long 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5
    .long 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3
    .long 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174
    .long 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc
    .long 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da
    .long 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7
    .long 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967
    .long 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13
    .long 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85
    .long 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3
    .long 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070
    .long 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5
    .long 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3
    .long 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208
    .long 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2

/* ---- KW2[i] = K[i] + W2[i] for CommP 64-byte padding block ----
 * W2 is the SHA256 message schedule of the constant padding block:
 *   W2[0]      = 0x80000000
 *   W2[1..14]  = 0
 *   W2[15]     = 0x00000200
 *   W2[16..63] = (standard rounds) */
.align 64
.LS_KW2:
    .long 0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5  /* rounds  0- 3 */
    .long 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5  /* rounds  4- 7 */
    .long 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3  /* rounds  8-11 */
    .long 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374  /* rounds 12-15 */
    .long 0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254  /* rounds 16-19 */
    .long 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa  /* rounds 20-23 */
    .long 0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7  /* rounds 24-27 */
    .long 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0  /* rounds 28-31 */
    .long 0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd  /* rounds 32-35 */
    .long 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16  /* rounds 36-39 */
    .long 0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537  /* rounds 40-43 */
    .long 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37  /* rounds 44-47 */
    .long 0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7  /* rounds 48-51 */
    .long 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890  /* rounds 52-55 */
    .long 0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c  /* rounds 56-59 */
    .long 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76  /* rounds 60-63 */
    .long 0x00000000  /* safety pad for round-63 prefetch */

/* ---- SHA256 padding block for a 64-byte message ----
 * byte 64 = 0x80, bytes 65-119 = 0x00, bytes 120-127 = bit length (512) */
.align 64
.LS_PAD:
    .long 0x00000080, 0x00000000, 0x00000000, 0x00000000
    .long 0x00000000, 0x00000000, 0x00000000, 0x00000000
    .long 0x00000000, 0x00000000, 0x00000000, 0x00000000
    .long 0x00000000, 0x00000000, 0x00000000, 0x00020000

/* ---- 254-bit truncation masks ---- */

/* Scalar broadcast version: vpbroadcastd clears top 2 bits of every dword 7 */
.LS_TRUNC254:
    .long 0xFFFFFF3F

/* Vector version: 0x3FFFFFFF at dword 7 and 15 (byte 31 of each 32B node) */
.align 64
.LS_TRUNC254_VEC:
    .long 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    .long 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x3FFFFFFF
    .long 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    .long 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x3FFFFFFF

/* ---- Lane deinterleave indices (SoA even/odd split) ---- */
.align 64
.LS_EVEN_IDX:
    .long 0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0
.LS_ODD_IDX:
    .long 1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0

/* ---- Dword byte-reversal mask (LE ↔ BE) ---- */
.align 16
.LS_BYTESWAP:
    .long 0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F

.align 64
PSHUFFLE_TRANSPOSE_MASK1:
        .quad  0x0000000000000000, 0x0000000000000001
        .quad  0x0000000000000008, 0x0000000000000009
        .quad  0x0000000000000004, 0x0000000000000005
        .quad  0x000000000000000C, 0x000000000000000D

.align 64
PSHUFFLE_TRANSPOSE_MASK2:
        .quad  0x0000000000000002, 0x0000000000000003
        .quad  0x000000000000000A, 0x000000000000000B
        .quad  0x0000000000000006, 0x0000000000000007
        .quad  0x000000000000000E, 0x000000000000000F

.align  16
PSHUFFLE_BYTE_FLIP_MASK:
        .int   0x00010203, 0x04050607
        .int   0x08090A0B, 0x0C0D0E0F

#endif /* SHA256_COMMP_RODATA_H */