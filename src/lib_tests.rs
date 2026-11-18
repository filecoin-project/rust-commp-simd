use super::*;
use std::path::PathBuf;

const MIN_PAYLOAD:  usize = 65;

pub const KNOWN: [u8; 32] = [
    0xf5, 0xa5, 0xfd, 0x42, 0xd1, 0x6a, 0x20, 0x30,
    0x27, 0x98, 0xef, 0x6e, 0xd3, 0x09, 0x97, 0x9b,
    0x43, 0x00, 0x3d, 0x23, 0x20, 0xd9, 0xf0, 0xe8,
    0xea, 0x98, 0x31, 0xa9, 0x27, 0x59, 0xfb, 0x0b,
];

pub fn known_pair() -> [u8; 64] {
    let mut pair = [0u8; 64];
    pair[..32].copy_from_slice(&KNOWN);
    pair[32..].copy_from_slice(&KNOWN);
    pair
}

// minimum buffer size the production kernel touches regardless of pair count
//
// x86: x16 AVX512 untranspose writes 8 x 64 = 512 bytes (all 16 lanes)
//      even for 1 pair, the kernel writes all 512 bytes
//
// aarch64: NEON x4 writes exactly n_pairs * 32 bytes, no overwrite
//          minimum is 1 pair = 64 bytes (input size)
#[cfg(target_arch = "x86_64")]
pub const PROD_KERNEL_MIN_BUF: usize = 512;

#[cfg(target_arch = "aarch64")]
pub const PROD_KERNEL_MIN_BUF: usize = 64;

#[cfg(target_arch = "wasm32")]
pub const PROD_KERNEL_MIN_BUF: usize = 64;

// make_known_pairs with buffer padded for prod kernel safety
pub fn make_padded_pairs(n_pairs: usize, seed: usize) -> Vec<u8> {
    let base = make_known_pairs(n_pairs, seed);
    let len = base.len().max(PROD_KERNEL_MIN_BUF);
    let mut buf = vec![0u8; len];
    buf[..base.len()].copy_from_slice(&base);
    buf
}

// fill buf with position-dependent deterministic bytes
// 6700417 = factor of 5th Fermat num (F5 = 641 × 6700417)
// gives decent byte spread (any large prime works here)
//
// wrapping_mul: i × 6700417, wraps on overflow
//
//             dec        product         >> 8           as u8
//
//   [i=1]:  6700417 = 0x_0066_3D81 -> 0x_0066_3D  ->  0x3D ( 61)
//   [i=2]: 13400834 = 0x_00CC_7B02 -> 0x_00CC_7B  ->  0x7B (123)
//   [i=3]: 20101251 = 0x_0132_B883 -> 0x_0132_B8  ->  0xB8 (184)
//                          ^^              ^^
// >> 8: slide right 8 bits, the ^^ byte becomes the output
//
// at each i moves by ~61 within 0-255
// hits all 256 values in 1024 positions, before the sequence starts repeating
// only 5 positions unchanged in 0..1023: [0, 732, 787, 842, 897]
//
//   pos:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
//   dec:  0  61 123 184 246  51 113 174 236  41 103 164 226  31  93 154
pub fn fill_pattern(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        // NOTE: 274177 also not so bad scatter with narrower jumps between
        *b = (i.wrapping_mul(6700417) >> 8) as u8;
    }
}

pub fn fill_zero(_buf: &mut [u8]) { /* already zeroed by vec![0u8; ...*/ }
pub fn fill_0xcc(buf: &mut [u8])  { buf.fill(0xCC) }

// sha256_trunc254 each 64-byte pair independently, return results
pub fn hashtrunc_each_pair(input: &[u8], n_pairs: usize) -> Vec<[u8; 32]> {
    (0..n_pairs).map(|p| {
        let mut pair = [0u8; 64];
        pair.copy_from_slice(&input[p * 64..(p + 1) * 64]);
        sha256_trunc254(&pair)
    }).collect()
}

// (n) identical pairs of [KNOWN + KNOWN]
pub fn make_known_pairs(n: usize, min_buf: usize) -> Vec<u8> {
    let size = min_buf.max(n * 64);
    // assert!(size >= 512, "buffer must be >= 512 (SoA kernel minimum)");

    let mut buf = vec![0u8; size]; // min 256 for hash_4_pairs

    for i in 0..n {
        let off = i * 64;
        buf[off..off + 32].copy_from_slice(&KNOWN);      // left  [0..32)
        buf[off + 32..off + 64].copy_from_slice(&KNOWN); // right [32..64)
    }

    buf
}

// compare (n) 32-byte results
// print first 4 bytes of each on mismatch
pub fn assert_results_match(a: &[u8], b: &[u8], n: usize, label: &str) {
    for i in 0..n {
        let off = i * 32;
        let sa = &a[off..off + 32];
        let sb = &b[off..off + 32];

        // for debugging
        // eprintln!("[{i}] old={:02x?}", &sa[..4]);
        // eprintln!("[{i}] new={:02x?}", &sb[..4]);

        assert_eq!(sa, sb, "{label} result {i}");
    }
}

// full tree: scalar pairwise_reduce_buffer vs kernel
// assert: roots match
pub fn assert_root_matches(
    buf: &[u8],
    kernel: impl Fn(&mut Vec<u8>),
    label: &str,
) {
    let mut scalar = buf.to_vec();
    pairwise_reduce_buffer_scalar(&mut scalar);

    let mut kern = buf.to_vec();
    kern.resize(kern.len().max(PROD_KERNEL_MIN_BUF), 0);
    kernel(&mut kern);

    eprintln!("{label} scalar: {:02x?}", &scalar[..32]);
    eprintln!("{label} kernel: {:02x?}", &kern[..32]);

    assert_eq!(&scalar[..32], &kern[..32], "{label}");
}

/// Build a buffer of `n` 32-byte nodes filled with a deterministic pattern.
pub fn make_slab(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n * 32];
    fill_pattern(&mut buf);
    buf
}

/// Round up to next power of two, minimum 64.
pub fn pad_to_pow2(mut buf: Vec<u8>) -> Vec<u8> {
    let target = buf.len().next_power_of_two().max(64);
    buf.resize(target, 0);
    buf
}

// CommP pipeline - FR32 padding + pairwise tree reduction
pub fn commp(input: &[u8]) -> Result<[u8; NODE_SIZE], &'static str> {
    if input.len() < MIN_PAYLOAD {
        return Err("input shorter than minimum payload (65 bytes)");
    }

    let n_quads = (input.len() + IN_PER_QUAD - 1) / IN_PER_QUAD;
    let mut quad_buf = vec![0u8; n_quads * IN_PER_QUAD];
    quad_buf[..input.len()].copy_from_slice(input);
    let mut buf = fr32_pad_slab(&quad_buf);

    let n_nodes = 4 * n_quads;
    let target = n_nodes.next_power_of_two();
    buf.resize(target * NODE_SIZE, 0);   // zero bytes = zero_comm(0)
    // minimum 512 bytes for SoA kernel
    if buf.len() < 512 { buf.resize(512, 0); }
    unsafe {
        pairwise_reduce_buffer(buf.as_mut_ptr(), target * NODE_SIZE, 0);
    }

    let mut root = [0u8; NODE_SIZE];
    root.copy_from_slice(&buf[..NODE_SIZE]);
    Ok(root)
}

// commp() with scalar reduce (no SIMD)
//
// input:  raw payload bytes (≥ 65)
// output: 32-byte commP root
//
// Appends zero bytes until the node count is a power of 2, then reduces
// e.g. 3 nodes -> pad-to-4
//      5 nodes -> pad-to-8
//
// Padding once at level 0 is likeo Go's perLevel zero_comm
// We know the full input before we start, so we can pad once
// each appended node is at level 0: [0u8; 32] (raw zero) = zero_comm(0)
pub fn commp_scalar(input: &[u8]) -> Result<[u8; NODE_SIZE], &'static str> {
    if input.len() < MIN_PAYLOAD {
        return Err("input shorter than minimum payload (65 bytes)");
    }

    // pad input to fill complete 127-byte quads
    // eg. 200 bytes -> 254 bytes (2 x 127) last 54 bytes zero filled
    let n_quads = (input.len() + IN_PER_QUAD - 1) / IN_PER_QUAD;
    let mut quad_buf = vec![0u8; n_quads * IN_PER_QUAD];
    quad_buf[..input.len()].copy_from_slice(input);

    // FR32 pad, then round up to power-of-2 node count
    // eg. 2 quads -> 8 nodes (already pow2),
    //     3 quads -> 12 nodes + pad to 16
    let mut buf = fr32_pad_slab(&quad_buf);
    let n_nodes = 4 * n_quads;
    let target = n_nodes.next_power_of_two();
    buf.resize(target * NODE_SIZE, 0);   // zero-pad leaves = zero_comm(0)

    pairwise_reduce_buffer_scalar(&mut buf);

    let mut root = [0u8; NODE_SIZE];
    root.copy_from_slice(&buf[..NODE_SIZE]);

    Ok(root)
}

// reference extrusion - one quad at a time, used as ground truth
pub fn scalar_extrude(src: &[u8]) -> Vec<u8> {
    let count = src.len() / IN_PER_QUAD;
    let mut out = vec![0u8; count * OUT_PER_QUAD];

    for q in 0..count {
        let mut quad = [0u8; OUT_PER_QUAD];

        fr32_pad_quad(
            &src[q * IN_PER_QUAD..(q + 1) * IN_PER_QUAD],
            &mut quad,
        );

        out[q * OUT_PER_QUAD..(q + 1) * OUT_PER_QUAD].copy_from_slice(&quad);
    }
    out
}

fn assert_slabs_equal(nq: usize, got: &[u8], expected: &[u8]) {
    for i in 0..got.len() {
        if got[i] != expected[i] {
            panic!(
                "nq={nq}: mismatch at byte {i} (quad {}, offset {}): got {:02x} want {:02x}",
                i / OUT_PER_QUAD, i % OUT_PER_QUAD, got[i], expected[i]
            );
        }
    }
}

// dst starts as 0xFF - any surviving 0xFF means "SIMD skipped a byte"
//
// dst before:  [FF FF FF FF FF FF ...]
// input:       [xx xx xx xx xx xx ...]
// dst after:   [xx xx xx xx xx xx ...]  <- any FF left = bug
pub fn assert_extrude_matches_scalar(
    nqs: &[usize],
    fill: fn(&mut [u8]),
    extrude: fn(&mut [u8], usize),
) {
    for &nq in nqs {
        let mut buf = vec![0u8; nq * OUT_PER_QUAD];
        fill(&mut buf[..nq * IN_PER_QUAD]);

        let expected = scalar_extrude(&buf[..nq * IN_PER_QUAD]);
        extrude(&mut buf, nq);

        assert_slabs_equal(nq, &buf, &expected);
    }
}

// input:  extruded dst buf (nq x 128 bytes)
// assert: top 2 bits at bytes 31, 63, 95, 127 of each quad are zero
//
//  byte:  [data x 31] [XX] [data x 31] [XX] [data x 31] [XX] [data x 31] [XX]
//                      ^^                                                 ^^
//                      top 2 bits forced to 0 by FR32
//
//  catches: mask bugs, shift-amount off-by-one
pub fn assert_padding_bits_clear(nq: usize, dst: &[u8]) {
    for q in 0..nq {
        for &node_end in &[31, 63, 95, 127] {
            let idx = q * 128 + node_end;
            let top2 = dst[idx] >> 6;

            assert_eq!(top2, 0, "nq={nq} quad {q} byte {node_end}: top 2 bits = {top2:02b}");
        }
    }
}

pub fn testdata_path(dir: &str, filename: &str) -> PathBuf {
    format!("{dir}/{filename}").into()
}

pub struct ExtrudeTestCase {
    pub quad_count: usize,
    pub output_hash: [u8; 32],
}

// input:  path to the extrude testdata (CSV in .txt)
// format: quad_count, hex_sha256_hash
// output: parsed test cases, panics if file missing or malformed
pub fn load_test_cases(path: &std::path::Path) -> Vec<ExtrudeTestCase> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut cases = Vec::new();
    for line in content.lines() {
        let line = line.trim();

        // skip comments, empty lines
        if line.starts_with('#') || line.is_empty() { continue; }

        let parts: Vec<&str> = line.splitn(2, ',').collect();
        if parts.len() != 2 { continue }

        let nq: usize = parts[0].parse().unwrap();
        let hash_bytes = hex::decode(parts[1]).unwrap();
        assert_eq!(hash_bytes.len(), 32);

        let mut h = [0u8; 32];
        h.copy_from_slice(&hash_bytes);

        cases.push(ExtrudeTestCase { quad_count: nq, output_hash: h });
    }
    cases
}

// read test vector binary into buf
pub fn fill_random(buf: &mut [u8], path: &std::path::Path) {
    let data = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let n = buf.len().min(data.len());
    buf[..n].copy_from_slice(&data[..n]);
}

// Run extrude test vectors
//
// For each vector:
//   - fill src with pattern
//   - extrude (via asm)
//   - sha256 the output
//   - verify sha256 of output matches expected
pub fn run_extrude_vectors(
    file: &str,
    fill: fn(&mut [u8]),
    extrude: fn(&mut [u8], usize),
) {
    let path = testdata_path("testdata/extrude", file);
    let cases = load_test_cases(&path);

    for tc in &cases {
        let mut buf = vec![0u8; tc.quad_count * OUT_PER_QUAD];
        fill(&mut buf[..tc.quad_count * IN_PER_QUAD]);
        let src_copy = buf[..tc.quad_count * IN_PER_QUAD].to_vec();

        extrude(&mut buf, tc.quad_count);

        use sha2::{Sha256, Digest};
        let got: [u8; 32] = Sha256::digest(&buf).into();
        assert_eq!(
            got, tc.output_hash,
            "quads={}: hash mismatch\n  got  {}\n  want {}",
            tc.quad_count, hex::encode(got), hex::encode(tc.output_hash),
        );

        // scalar sanity check
        let scalar_dst = scalar_extrude(&src_copy);
        let scalar_hash: [u8; 32] = Sha256::digest(&scalar_dst).into();
        assert_eq!(
            scalar_hash, tc.output_hash,
            "scalar: quads={} hash mismatch",
            tc.quad_count,
        );
    }
}

/// Core helper: run SIMD kernel and scalar with identical inputs, compare root.
fn simd_matches_scalar(nodes: usize) {
    let slab = pad_to_pow2(make_slab(nodes));

    let mut scalar_buf = slab.clone();
    pairwise_reduce_buffer_scalar(&mut scalar_buf);

    let mut simd_buf = slab;
    unsafe {
        pairwise_reduce_buffer(simd_buf.as_mut_ptr(), simd_buf.len(), 0);
    }

    assert_eq!(
        &scalar_buf[..32], &simd_buf[..32],
        "SIMD root does not match scalar for {nodes} nodes"
    );
}

#[test]
fn simd_matches_scalar_16_nodes() {
    simd_matches_scalar(16);
}

#[test]
fn simd_matches_scalar_1024_nodes() {
    simd_matches_scalar(1024);
}

// single-level reduce: n_pairs in, n_pairs results out
// each result compared against scalar sha256_trunc254
fn assert_single_level(n_pairs: usize) {
    let mut buf = make_slab(n_pairs * 2);  // 2 nodes per pair
    let expected = hashtrunc_each_pair(&buf, n_pairs);

    unsafe {
        pairwise_reduce_buffer(buf.as_mut_ptr(), n_pairs * 64, n_pairs / 2);
    }

    for i in 0..n_pairs {
        assert_eq!(
            &buf[i * 32..(i + 1) * 32],
            expected[i].as_slice(),
            "pair {i} mismatch"
        );
    }
}

#[test]
// 16 pairs, one level: out_pair_count=8, 8 > 8 false, stop
fn single_level_16_pairs() { assert_single_level(16); }

#[test]
// 32 pairs, one level: out_pair_count=16, 16 > 16 false, stop
fn single_level_32_pairs() { assert_single_level(32); }
