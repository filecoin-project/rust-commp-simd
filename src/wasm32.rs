// SHA256 x4 block kernel using WASM SIMD v128 intrinsics
//
// Port of sha256_4way.rs (browser WASM)
// Same SHA256 round function, same transpose, same bswap

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// ---
// SHA256 round constants
// ---

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// SHA-256 initial hash values H[0..7]
pub(crate) const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// ---
// WASM SIMD primitives (equivalent of AVX512 single-instruction ops)
// ---

// Rotate right each i32 lane by n bits
// WASM has no v128 rotr - emulate: (x >> n) | (x << (32 - n))
// Maps to AVX512 vprord
#[inline(always)]
fn rotr(x: v128, n: u32) -> v128 {
    v128_or(u32x4_shr(x, n), u32x4_shl(x, 32 - n))
}

// Byte-swap each i32 lane: LE <-> BE
// Maps to AVX512 vpshufb with PSHUFFLE_BYTE_FLIP_MASK
#[inline(always)]
pub(crate) fn bswap(v: v128) -> v128 {
    i8x16_shuffle::<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>(v, v)
}

// CH(e, f, g) = (e & f) ^ (~e & g) = g ^ (e & (f ^ g))
// Maps to AVX512 vpternlogd 0xCA
#[inline(always)]
fn ch(e: v128, f: v128, g: v128) -> v128 {
    v128_xor(g, v128_and(e, v128_xor(f, g)))
}

// MAJ(a, b, c) = (a & b) ^ (a & c) ^ (b & c) = (a & b) ^ (c & (a ^ b))
// Maps to AVX512 vpternlogd 0xE8
#[inline(always)]
fn maj(a: v128, b: v128, c: v128) -> v128 {
    v128_xor(v128_and(a, b), v128_and(c, v128_xor(a, b)))
}

// SIGMA0(a) = ROTR(2) ^ ROTR(13) ^ ROTR(22)
#[inline(always)]
fn big_sigma0(a: v128) -> v128 {
    v128_xor(v128_xor(rotr(a, 2), rotr(a, 13)), rotr(a, 22))
}

// SIGMA1(e) = ROTR(6) ^ ROTR(11) ^ ROTR(25)
#[inline(always)]
fn big_sigma1(e: v128) -> v128 {
    v128_xor(v128_xor(rotr(e, 6), rotr(e, 11)), rotr(e, 25))
}

// sigma0(w) = ROTR(7) ^ ROTR(18) ^ SHR(3)
#[inline(always)]
fn small_sigma0(w: v128) -> v128 {
    v128_xor(v128_xor(rotr(w, 7), rotr(w, 18)), u32x4_shr(w, 3))
}

// sigma1(w) = ROTR(17) ^ ROTR(19) ^ SHR(10)
#[inline(always)]
fn small_sigma1(w: v128) -> v128 {
    v128_xor(v128_xor(rotr(w, 17), rotr(w, 19)), u32x4_shr(w, 10))
}

// ---
// 4x4 transpose (WASM analog of TRANSPOSE16)
// ---

// Transpose a 4x4 matrix of i32 values across four v128 registers
// 8 shuffles total (2-phase interleave)
//
// input:  r0 = {a0, a1, a2, a3}  (lane 0, words i..i+3)
//         r1 = {b0, b1, b2, b3}  (lane 1)
//         r2 = {c0, c1, c2, c3}  (lane 2)
//         r3 = {d0, d1, d2, d3}  (lane 3)
// output: t0 = {a0, b0, c0, d0}  (word i from all 4 lanes)
//         t1 = {a1, b1, c1, d1}
//         t2 = {a2, b2, c2, d2}
//         t3 = {a3, b3, c3, d3}
#[inline(always)]
pub(crate) fn transpose_4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
    // Phase 1: interleave pairs
    let p0 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(r0, r1);
    let p1 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(r0, r1);
    let p2 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(r2, r3);
    let p3 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(r2, r3);

    // Phase 2: merge across pairs
    let t0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(p0, p2);
    let t1 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(p0, p2);
    let t2 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(p1, p3);
    let t3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(p1, p3);

    (t0, t1, t2, t3)
}

// -----------------------------------------------------------------------
// SHA256 block processing (4 lanes in parallel)
// -----------------------------------------------------------------------

// One SHA256 round on 4 lanes. The names rotate instead of the values:
// the new E lands in d, the new A in h, and the next round is called with
// the names shifted by one (h, a, b, c, d, e, f, g)
macro_rules! round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $kw:expr) => {{
        // T1 = H + SIGMA1(E) + CH(E,F,G) + K[i] + W[i]
        let t1 = i32x4_add(i32x4_add($h, big_sigma1($e)), i32x4_add(ch($e, $f, $g), $kw));
        // T2 = SIGMA0(A) + MAJ(A,B,C)
        let t2 = i32x4_add(big_sigma0($a), maj($a, $b, $c));
        $d = i32x4_add($d, t1);
        $h = i32x4_add(t1, t2);
    }};
}

// 8 rounds: after 8 shifts the names are back in place
macro_rules! rounds8 {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $kw:ident, $i:expr) => {
        round!($a, $b, $c, $d, $e, $f, $g, $h, $kw!($i));
        round!($h, $a, $b, $c, $d, $e, $f, $g, $kw!($i + 1));
        round!($g, $h, $a, $b, $c, $d, $e, $f, $kw!($i + 2));
        round!($f, $g, $h, $a, $b, $c, $d, $e, $kw!($i + 3));
        round!($e, $f, $g, $h, $a, $b, $c, $d, $kw!($i + 4));
        round!($d, $e, $f, $g, $h, $a, $b, $c, $kw!($i + 5));
        round!($c, $d, $e, $f, $g, $h, $a, $b, $kw!($i + 6));
        round!($b, $c, $d, $e, $f, $g, $h, $a, $kw!($i + 7));
    };
}

/// Process one 64-byte SHA256 block across 4 lanes simultaneously.
/// `state`: 8 × v128, each holding one state word for all 4 messages.
/// `w`: 16 × v128, the message schedule words (modified in-place for rounds 16+).
#[inline(always)]
pub(crate) fn sha256_block(state: &mut [v128; 8], w: &mut [v128; 16]) {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    // Save initial state for final addition
    let (a0, b0, c0, d0) = (a, b, c, d);
    let (e0, f0, g0, h0) = (e, f, g, h);

    // Round constant broadcast to all 4 lanes, plus W[i]
    macro_rules! kw_msg {
        ($i:expr) => {
            i32x4_add(u32x4_splat(K[$i]), w[($i) & 15])
        };
    }

    // 64 rounds of SHA256, doing 4 hashes: 8 passes of 8 rounds
    for i in (0..64).step_by(8) {
        // Message schedule: compute the 8 new W[j%16] of the pass for rounds 16+
        if i >= 16 {
            for j in i..i + 8 {
                let s0 = small_sigma0(w[(j - 15) & 15]);
                let s1 = small_sigma1(w[(j - 2) & 15]);
                w[j & 15] = i32x4_add(i32x4_add(w[j & 15], s0), i32x4_add(w[(j - 7) & 15], s1));
            }
        }

        rounds8!(a, b, c, d, e, f, g, h, kw_msg, i);
    }

    // Add back initial state (same as AVX512's final vpaddd)
    state[0] = i32x4_add(a0, a);
    state[1] = i32x4_add(b0, b);
    state[2] = i32x4_add(c0, c);
    state[3] = i32x4_add(d0, d);
    state[4] = i32x4_add(e0, e);
    state[5] = i32x4_add(f0, f);
    state[6] = i32x4_add(g0, g);
    state[7] = i32x4_add(h0, h);
}

// BlockFn-compatible wrapper for reduce_loop tests
// AoS Sha256x4Args -> transpose -> sha256_block -> transpose back -> AoS
pub(crate) unsafe extern "C" fn sha256_x4_wasm(args_ptr: *mut u8, block_count: u64) {
    let args = &mut *(args_ptr as *mut crate::reduce_loop::Sha256x4Args);

    for _ in 0..block_count {
        let d0_lo = v128_load(args.digest[0].as_ptr().add(0) as *const v128);
        let d1_lo = v128_load(args.digest[1].as_ptr().add(0) as *const v128);
        let d2_lo = v128_load(args.digest[2].as_ptr().add(0) as *const v128);
        let d3_lo = v128_load(args.digest[3].as_ptr().add(0) as *const v128);
        let (s0, s1, s2, s3) = transpose_4x4(d0_lo, d1_lo, d2_lo, d3_lo);

        let d0_hi = v128_load(args.digest[0].as_ptr().add(4) as *const v128);
        let d1_hi = v128_load(args.digest[1].as_ptr().add(4) as *const v128);
        let d2_hi = v128_load(args.digest[2].as_ptr().add(4) as *const v128);
        let d3_hi = v128_load(args.digest[3].as_ptr().add(4) as *const v128);
        let (s4, s5, s6, s7) = transpose_4x4(d0_hi, d1_hi, d2_hi, d3_hi);

        let mut state = [s0, s1, s2, s3, s4, s5, s6, s7];

        let p0 = args.data_ptr[0];
        let p1 = args.data_ptr[1];
        let p2 = args.data_ptr[2];
        let p3 = args.data_ptr[3];

        let mut w = [u32x4_splat(0); 16];
        for group in 0..4 {
            let off = group * 16;
            let r0 = v128_load(p0.add(off) as *const v128);
            let r1 = v128_load(p1.add(off) as *const v128);
            let r2 = v128_load(p2.add(off) as *const v128);
            let r3 = v128_load(p3.add(off) as *const v128);
            let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
            w[group * 4 + 0] = bswap(t0);
            w[group * 4 + 1] = bswap(t1);
            w[group * 4 + 2] = bswap(t2);
            w[group * 4 + 3] = bswap(t3);
        }

        sha256_block(&mut state, &mut w);

        let (d0_lo, d1_lo, d2_lo, d3_lo) = transpose_4x4(state[0], state[1], state[2], state[3]);
        let (d0_hi, d1_hi, d2_hi, d3_hi) = transpose_4x4(state[4], state[5], state[6], state[7]);

        v128_store(args.digest[0].as_mut_ptr().add(0) as *mut v128, d0_lo);
        v128_store(args.digest[0].as_mut_ptr().add(4) as *mut v128, d0_hi);
        v128_store(args.digest[1].as_mut_ptr().add(0) as *mut v128, d1_lo);
        v128_store(args.digest[1].as_mut_ptr().add(4) as *mut v128, d1_hi);
        v128_store(args.digest[2].as_mut_ptr().add(0) as *mut v128, d2_lo);
        v128_store(args.digest[2].as_mut_ptr().add(4) as *mut v128, d2_hi);
        v128_store(args.digest[3].as_mut_ptr().add(0) as *mut v128, d3_lo);
        v128_store(args.digest[3].as_mut_ptr().add(4) as *mut v128, d3_hi);
    }
}

// ---
// Public API for Hugo's repo (imported as sha256_4way)
// ---

/// Node size constant
const NODE_SIZE: usize = 32;

// -----------------------------------------------------------------------
// hash_4_blocks: hash 4 independent 64-byte blocks in parallel
// Used by leaf hashing where each FR32-padded quad produces two 64-byte blocks
// -----------------------------------------------------------------------

// Transpose a 4x4 matrix of u32 held in 4 rows, as u32 lane shuffles
#[inline(always)]
fn transpose_rows(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let t0 = u32x4_shuffle::<0, 4, 1, 5>(r0, r1);
    let t1 = u32x4_shuffle::<2, 6, 3, 7>(r0, r1);
    let t2 = u32x4_shuffle::<0, 4, 1, 5>(r2, r3);
    let t3 = u32x4_shuffle::<2, 6, 3, 7>(r2, r3);
    [
        u32x4_shuffle::<0, 1, 4, 5>(t0, t2),
        u32x4_shuffle::<2, 3, 6, 7>(t0, t2),
        u32x4_shuffle::<0, 1, 4, 5>(t1, t3),
        u32x4_shuffle::<2, 3, 6, 7>(t1, t3),
    ]
}

/// Hash 4 independent 64-byte blocks in parallel, returning 4 truncated results.
///
/// Input: `blocks` - 4 × 64 bytes, each an independent SHA256 input.
/// Output: 4 × 32-byte truncated (254-bit) hashes.
///
/// Same kernel as hash_4_pairs but takes raw 64-byte blocks instead of
/// node pairs. This is the hot path for leaf generation.
#[inline(always)]
pub fn hash_4_blocks(blocks: &[[u8; 64]; 4]) -> [[u8; NODE_SIZE]; 4] {
    // Initialize state: H_INIT broadcast to all 4 lanes
    let mut state = [
        u32x4_splat(H_INIT[0]),
        u32x4_splat(H_INIT[1]),
        u32x4_splat(H_INIT[2]),
        u32x4_splat(H_INIT[3]),
        u32x4_splat(H_INIT[4]),
        u32x4_splat(H_INIT[5]),
        u32x4_splat(H_INIT[6]),
        u32x4_splat(H_INIT[7]),
    ];

    // w[k] = [ blk0 word k | blk1 word k | blk2 word k | blk3 word k ]
    let mut w = [u32x4_splat(0); 16];
    for g in 0..4 {
        // byte swap each loaded row, then transpose
        let [r0, r1, r2, r3] = core::array::from_fn(|m| unsafe {
            // turn to BE
            bswap(v128_load(blocks[m].as_ptr().add(16 * g) as *const v128))
        });
        w[4 * g..4 * g + 4].copy_from_slice(&transpose_rows(r0, r1, r2, r3));
    }

    sha256_block(&mut state, &mut w);

    // Block 2: precomputed K[i]+W[i] - no message schedule
    sha256_block_padding(&mut state);

    // transpose back, then byte swap on the way out
    // These 8 words are the hash itself: the 8 state words A to H after all 64
    // rounds, which become h0 to h7. They go back into memory as 4 hashes of 32 bytes
    let lo = transpose_rows(state[0], state[1], state[2], state[3]);
    let hi = transpose_rows(state[4], state[5], state[6], state[7]);

    // where to words end up: 4 hashes of 32-bytes
    let mut out = [[0u8; NODE_SIZE]; 4];

    // 4 resulting nodes (out of 8 input nodes)
    for m in 0..4 {
        let node = &mut out[m];
        unsafe {
            // new node bytes  0..15   words h0 h1 h2 h3
            v128_store(node.as_mut_ptr() as *mut v128, bswap(lo[m]));
            // new node bytes 16..31   words h4 h5 h6 h7
            v128_store(node.as_mut_ptr().add(16) as *mut v128, bswap(hi[m]));
        }
        node[NODE_SIZE - 1] &= 0x3F;
    }

    out
}

#[cfg(not(target_arch = "wasm32"))]
pub fn hash_4_blocks(blocks: &[[u8; 64]; 4]) -> [[u8; NODE_SIZE]; 4] {
    use sha2::{Digest, Sha256};
    let mut out = [[0u8; NODE_SIZE]; 4];
    for i in 0..4 {
        let mut hasher = Sha256::new();
        hasher.update(&blocks[i]);
        let result: [u8; 32] = hasher.finalize().into();
        out[i] = result;
        out[i][31] &= 0x3F;
    }
    out
}

// -----------------------------------------------------------------------
// hash_4_pairs: hash 4 pairs of 32-byte nodes in parallel
// Used in tree reduction loop
// -----------------------------------------------------------------------

/// Hash 4 pairs of 32-byte nodes in parallel, returning 4 truncated (254-bit) results.
///
/// Input: `nodes` slice with at least 8 elements.
///   - Pair 0: (nodes[0], nodes[1])
///   - Pair 1: (nodes[2], nodes[3])
///   - Pair 2: (nodes[4], nodes[5])
///   - Pair 3: (nodes[6], nodes[7])
///
/// Each pair is concatenated (left || right = 64 bytes) and SHA256'd.
/// Output byte[31] has top 2 bits cleared (254-bit truncation for CommP).
pub fn hash_4_pairs(nodes: &[[u8; NODE_SIZE]]) -> [[u8; NODE_SIZE]; 4] {
    debug_assert!(nodes.len() >= 8);

    // Initialize sha256 state: put to all 4 lanes ---
    let mut state = [
        u32x4_splat(H_INIT[0]),
        u32x4_splat(H_INIT[1]),
        u32x4_splat(H_INIT[2]),
        u32x4_splat(H_INIT[3]),
        u32x4_splat(H_INIT[4]),
        u32x4_splat(H_INIT[5]),
        u32x4_splat(H_INIT[6]),
        u32x4_splat(H_INIT[7]),
    ];

    // Block 1: Load data -> transpose -> byte-swap -> process
    // Each message = left_node[32] || right_node[32] = 64 bytes = 16 u32 words
    // Words 0-3:   from left[0..16]   (first half of left node)
    // Words 4-7:   from left[16..32]  (second half of left node)
    // Words 8-11:  from right[0..16]  (first half of right node)
    // Words 12-15: from right[16..32] (second half of right node)
    let mut w = [u32x4_splat(0); 16];

    unsafe {
        // For each group of 4 words,
        //     - load from all 4 messages
        //     - transpose
        // Group 0:
        // nodes[0], nodes[2], nodes[4], nodes[6]
        //   are the left nodes for msg0..msg3
        {
            let r0 = v128_load(nodes[0].as_ptr() as *const v128);
            let r1 = v128_load(nodes[2].as_ptr() as *const v128);
            let r2 = v128_load(nodes[4].as_ptr() as *const v128);
            let r3 = v128_load(nodes[6].as_ptr() as *const v128);
            let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
            w[0] = bswap(t0);
            w[1] = bswap(t1);
            w[2] = bswap(t2);
            w[3] = bswap(t3);
        }

        // Group 1: words 4-7 from left nodes bytes 16..32
        {
            let r0 = v128_load(nodes[0].as_ptr().add(16) as *const v128);
            let r1 = v128_load(nodes[2].as_ptr().add(16) as *const v128);
            let r2 = v128_load(nodes[4].as_ptr().add(16) as *const v128);
            let r3 = v128_load(nodes[6].as_ptr().add(16) as *const v128);
            let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
            w[4] = bswap(t0);
            w[5] = bswap(t1);
            w[6] = bswap(t2);
            w[7] = bswap(t3);
        }

        // Group 2: words 8-11 from right nodes bytes 0..16
        {
            let r0 = v128_load(nodes[1].as_ptr() as *const v128);
            let r1 = v128_load(nodes[3].as_ptr() as *const v128);
            let r2 = v128_load(nodes[5].as_ptr() as *const v128);
            let r3 = v128_load(nodes[7].as_ptr() as *const v128);
            let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
            w[8] = bswap(t0);
            w[9] = bswap(t1);
            w[10] = bswap(t2);
            w[11] = bswap(t3);
        }

        // Group 3: words 12-15 from right nodes bytes 16..32
        {
            let r0 = v128_load(nodes[1].as_ptr().add(16) as *const v128);
            let r1 = v128_load(nodes[3].as_ptr().add(16) as *const v128);
            let r2 = v128_load(nodes[5].as_ptr().add(16) as *const v128);
            let r3 = v128_load(nodes[7].as_ptr().add(16) as *const v128);
            let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
            w[12] = bswap(t0);
            w[13] = bswap(t1);
            w[14] = bswap(t2);
            w[15] = bswap(t3);
        }
    }

    sha256_block(&mut state, &mut w);

    // Block 2: precomputed K[i]+W[i] - no message schedule
    sha256_block_padding(&mut state);

    // Extract results: un-transpose state and bswap to Be
    // state[0..4] -> first 16 bytes of each hash
    // state[4..8] -> last 16 bytes of each hash
    let (t0_lo, t1_lo, t2_lo, t3_lo) = transpose_4x4(
        bswap(state[0]),
        bswap(state[1]),
        bswap(state[2]),
        bswap(state[3]),
    );
    let (t0_hi, t1_hi, t2_hi, t3_hi) = transpose_4x4(
        bswap(state[4]),
        bswap(state[5]),
        bswap(state[6]),
        bswap(state[7]),
    );

    let mut out = [[0u8; NODE_SIZE]; 4];
    unsafe {
        v128_store(out[0].as_mut_ptr() as *mut v128, t0_lo);
        v128_store(out[0].as_mut_ptr().add(16) as *mut v128, t0_hi);
        v128_store(out[1].as_mut_ptr() as *mut v128, t1_lo);
        v128_store(out[1].as_mut_ptr().add(16) as *mut v128, t1_hi);
        v128_store(out[2].as_mut_ptr() as *mut v128, t2_lo);
        v128_store(out[2].as_mut_ptr().add(16) as *mut v128, t2_hi);
        v128_store(out[3].as_mut_ptr() as *mut v128, t3_lo);
        v128_store(out[3].as_mut_ptr().add(16) as *mut v128, t3_hi);
    }

    // 254-bit truncation: clear top 2 bits of last byte
    out[0][31] &= 0x3F;
    out[1][31] &= 0x3F;
    out[2][31] &= 0x3F;
    out[3][31] &= 0x3F;

    out
}

// -----------------------------------------------------------------------
// Scalar fallback for non-wasm32 targets (testing only)
// -----------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn hash_4_pairs(nodes: &[[u8; NODE_SIZE]]) -> [[u8; NODE_SIZE]; 4] {
    use sha2::{Digest, Sha256};
    let mut out = [[0u8; NODE_SIZE]; 4];
    for i in 0..4 {
        let mut hasher = Sha256::new();
        hasher.update(&nodes[i * 2]);
        hasher.update(&nodes[i * 2 + 1]);
        let result: [u8; 32] = hasher.finalize().into();
        out[i] = result;
        out[i][31] &= 0x3F;
    }
    out
}

// -----------------------------------------------------------------------
// process_4_quads: FR32 extrude + leaf hash for 4 quads batched
// Convenience export so Hugo imports this instead of reimplementing
// -----------------------------------------------------------------------

pub fn process_4_quads(
    sources: [&[u8]; 4],
    pad_bufs: &mut [[u8; 128]; 4],
) -> [[u8; NODE_SIZE]; 8] {
    use crate::arch_wasm32::extrude_wasm32::extrude_one_quad;

    // Phase 1: extrude all 4 quads (SIMD bit shifts) - in-place per buf
    for i in 0..4 {
        pad_bufs[i][..127].copy_from_slice(&sources[i][..127]);
        extrude_one_quad(&mut pad_bufs[i]);
    }

    // Phase 2: batch hash first halves (bytes 0..64 of each padded quad)
    let first_halves: [&[u8; 64]; 4] = [
        pad_bufs[0][..64].try_into().unwrap(),
        pad_bufs[1][..64].try_into().unwrap(),
        pad_bufs[2][..64].try_into().unwrap(),
        pad_bufs[3][..64].try_into().unwrap(),
    ];
    let first_results = hash_4_blocks(&[
        *first_halves[0], *first_halves[1], *first_halves[2], *first_halves[3],
    ]);

    // Phase 3: batch hash second halves (bytes 64..128 of each padded quad)
    let second_halves: [&[u8; 64]; 4] = [
        pad_bufs[0][64..].try_into().unwrap(),
        pad_bufs[1][64..].try_into().unwrap(),
        pad_bufs[2][64..].try_into().unwrap(),
        pad_bufs[3][64..].try_into().unwrap(),
    ];
    let second_results = hash_4_blocks(&[
        *second_halves[0], *second_halves[1], *second_halves[2], *second_halves[3],
    ]);

    // Interleave: quad0_leafA, quad0_leafB, quad1_leafA, quad1_leafB, ...
    [
        first_results[0], second_results[0],
        first_results[1], second_results[1],
        first_results[2], second_results[2],
        first_results[3], second_results[3],
    ]
}

// -----------------------------------------------------------------------
// Precomputed K[i]+W[i] for SHA256 padding block (64-byte input)
//
// W[0]=0x80000000 W[1..14]=0 W[15]=0x00000200 W[16..63]=schedule(W)
// KW_PAD[i] = K[i] + W[i]
//
// Same values as LS_KW2 in pairwise_reduce_x4_neon.S
// Eliminates message schedule (48 sigma expansions + 64 K splats)
// -----------------------------------------------------------------------

const KW_PAD: [u32; 64] = [
    0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,   // rounds  0- 3
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,   // rounds  4- 7
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,   // rounds  8-11
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,   // rounds 12-15
    0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254,   // rounds 16-19
    0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,   // rounds 20-23
    0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7,   // rounds 24-27
    0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,   // rounds 28-31
    0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd,   // rounds 32-35
    0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,   // rounds 36-39
    0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537,   // rounds 40-43
    0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,   // rounds 44-47
    0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7,   // rounds 48-51
    0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,   // rounds 52-55
    0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c,   // rounds 56-59
    0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76,   // rounds 60-63
];

// SHA256 works on 64-byte blocks (every message gets padded with 0x80)
// W[i] of the padding is always the same, so KW_PAD[i] = K[i] + W[i] is
// precomputed: KW_PAD[i] = K[i] + W[i] of the padding block
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn sha256_block_padding(state: &mut [v128; 8]) {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    let (a0, b0, c0, d0) = (a, b, c, d);
    let (e0, f0, g0, h0) = (e, f, g, h);

    // KW_PAD[i] of round i, one splat of a constant
    macro_rules! kw_pad {
        ($i:expr) => {
            u32x4_splat(KW_PAD[$i])
        };
    }

    // 8 passes of 8 rounds
    for i in (0..64).step_by(8) {
        rounds8!(a, b, c, d, e, f, g, h, kw_pad, i);
    }

    state[0] = i32x4_add(a0, a);
    state[1] = i32x4_add(b0, b);
    state[2] = i32x4_add(c0, c);
    state[3] = i32x4_add(d0, d);
    state[4] = i32x4_add(e0, e);
    state[5] = i32x4_add(f0, f);
    state[6] = i32x4_add(g0, g);
    state[7] = i32x4_add(h0, h);
}