//! Scalar:
//!     - SHA256-trunc254
//!     - FR32 padding
//!     - pairwise reduce
//!     - zero_comm (null padding hashes for odd tree levels)
//!
//! Pure Rust, no SIMD. Used for validatating against SIMD.

#[cfg(not(all(target_arch = "wasm32", not(test))))]
use std::sync::OnceLock;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};
use crate::{NODE_SIZE, IN_PER_QUAD, OUT_PER_QUAD};
#[cfg(not(all(target_arch = "wasm32", not(test))))]
use crate::MAX_LEVELS;

// input: exactly 64 bytes (one pair: left 32B + right 32B)
// output: 32-byte SHA256 digest, top 2 bits of byte 31 cleared (254-bit "field element")
#[inline]
pub fn sha256_trunc254(data: &[u8; 64]) -> [u8; NODE_SIZE] {
    let mut h = Sha256::new();
    h.update(data.as_slice());

    let mut out: [u8; NODE_SIZE] = h.finalize().into();
    out[NODE_SIZE - 1] &= 0x3F;
    out
}

// input: N × 127 bytes (N quads of raw payload)
// output: N × 128 bytes (FR32 padded, 2 bits inserted per 32-byte node)
pub fn fr32_pad_slab(input: &[u8]) -> Vec<u8> {
    debug_assert_eq!(input.len() % IN_PER_QUAD, 0);

    let n = input.len() / IN_PER_QUAD;
    let mut out = vec![0u8; n * OUT_PER_QUAD];

    // chunks_exact splits into fixed-size slices, zip pairs src[i] with dst[i]
    let chunked_both = input.chunks_exact(IN_PER_QUAD);
    for (src, dst) in chunked_both.zip(out.chunks_exact_mut(OUT_PER_QUAD)) {
        fr32_pad_quad(src, dst.try_into().unwrap());
    }

    out
}

/// FR32 pad one quad: 127 bytes -> 128 bytes.
/// Matches Go's digestQuads byte-for-byte.
pub fn fr32_pad_quad(src: &[u8], dst: &mut [u8; OUT_PER_QUAD]) {
    dst[..32].copy_from_slice(&src[..32]);
    dst[31] &= 0x3F;

    for i in 31..63 { dst[i + 1] = (src[i + 1] << 2) | (src[i] >> 6); }
    dst[63] &= 0x3F;

    for i in 63..95 { dst[i + 1] = (src[i + 1] << 4) | (src[i] >> 4); }
    dst[95] &= 0x3F;

    for i in 95..126 { dst[i + 1] = (src[i + 1] << 6) | (src[i] >> 2); }
    dst[127] = src[126] >> 2;
}

// input:  buf[0 .. n_pairs*64] - pairs laid out as [L0 R0 L1 R1 ...]
// output: buf[0 .. n_pairs*32] - one 32-byte result per pair (in-place)
//
// results compact into first half
pub fn pairwise_reduce_one_level(buf: &mut [u8], n_pairs: usize) {
    for i in 0..n_pairs {
        let mut pair = [0u8; 64];
        let read_idx = i * 64;
        let write_idx = i * 32;

        pair[..32].copy_from_slice(&buf[read_idx..read_idx + 32]);
        pair[32..].copy_from_slice(&buf[read_idx + 32..read_idx + 64]);

        let result = sha256_trunc254(&pair);

        buf[write_idx..write_idx + 32].copy_from_slice(&result);
    }
}

// input: buf of 2^N bytes (≥ 64), filled with 32-byte nodes
// output: single 32-byte root at buf[0..32]
//
// reduces level by level until one node remains
pub fn pairwise_reduce_buffer_scalar(buff: &mut [u8]) {
    assert!(buff.len().is_power_of_two() && buff.len() >= 64);
    let mut cur = buff.len();

    while cur >= 64 {
        pairwise_reduce_one_level(&mut buff[..cur], cur / 64);
        cur /= 2;
    }
}

#[cfg(not(all(target_arch = "wasm32", not(test))))]
static ZERO_COMMS: OnceLock<Vec<[u8; NODE_SIZE]>> = OnceLock::new();

// memoized null padding hashes
// when a tree level has an odd node count, zero_comm(level) is appended to make it even
//
// zero_comm(0) = [0u8; 32]
// zero_comm(n) = sha256_trunc254(zero_comm(n-1) ++ zero_comm(n-1))
//
// level 0 (leaves):    00  00  00  00  00  00  00  00
//                       \  /    \  /    \  /    \  /
// level 1:              h(0,0) h(0,0) h(0,0) h(0,0)
//                        |       |       |      |
//                      all four are identical = v[1]
//                         \      /        \     /
// level 2:                h(v1,v1)        h(v1,v1)
//                            |              |
//                          both identical = v[2]
//                             \            /
// level 3:                   h(v2,v2) = v[3]
//
// Matches Go's stackedNulPadding
#[cfg(not(all(target_arch = "wasm32", not(test))))]
pub fn zero_comm(level: usize) -> [u8; NODE_SIZE] {
    let table = ZERO_COMMS.get_or_init(|| {
        // 64 entries x 32 bytes = 2048 bytes total
        let mut v = vec![[0u8; NODE_SIZE]; MAX_LEVELS];

        for i in 1..MAX_LEVELS {
            let mut pair = [0u8; 64];

            pair[..32].copy_from_slice(&v[i - 1]);
            pair[32..].copy_from_slice(&v[i - 1]);

            v[i] = sha256_trunc254(&pair);
        }

        v
    });

    table[level]
}

// wasm32 without std: no OnceLock, the level is hashed up from zero on each call
#[cfg(all(target_arch = "wasm32", not(test)))]
pub fn zero_comm(level: usize) -> [u8; NODE_SIZE] {
    let mut node = [0u8; NODE_SIZE];
    for _ in 0..level {
        let mut pair = [0u8; 64];
        pair[..32].copy_from_slice(&node);
        pair[32..].copy_from_slice(&node);
        node = sha256_trunc254(&pair);
    }
    node
}

// In-place FR32 extrusion.
// buf[0..count*127] contains source bytes. Kernel rewrites buf[0..count*128]
pub fn scalar_extrude_inplace(buf: &mut [u8], count: usize) {
    if count == 0 { return; }

    let mut tmp_src = [0u8; IN_PER_QUAD];
    let mut tmp_dst = [0u8; OUT_PER_QUAD];

    // iterate backward: last quad first
    for i in (0..count).rev() {
        let src_off = i * IN_PER_QUAD;
        let dst_off = i * OUT_PER_QUAD;

        tmp_src.copy_from_slice(&buf[src_off..src_off + IN_PER_QUAD]);
        fr32_pad_quad(&tmp_src, &mut tmp_dst);
        buf[dst_off..dst_off + OUT_PER_QUAD].copy_from_slice(&tmp_dst);
    }
}

pub fn sha256_x4_fallback(args_ptr: *mut u8, num_blocks: u64) {
    use crate::reduce_loop::Sha256x4Args;
    use sha2::compress256;
    use sha2::digest::generic_array::GenericArray;

    unsafe {
        let args = &mut *(args_ptr as *mut Sha256x4Args);
        let n = num_blocks as usize;
        for lane in 0..4 {
            let blocks = core::slice::from_raw_parts(
                args.data_ptr[lane] as *const GenericArray<u8, sha2::digest::typenum::U64>,
                n,
            );
            compress256(&mut args.digest[lane], blocks);
            args.data_ptr[lane] = args.data_ptr[lane].add(n * 64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib_tests::KNOWN;
    use crate::pairwise_reduce_buffer_rust_scalar;

    #[test]
    // input: 64 zero bytes (one pair)
    // output: SHA256(zeros) with trunc254
    // raw byte 31 = 0x4b, after &= 0x3F -> 0x0b
    fn scalar_two_nodes_known_vector() {
        let mut buf = vec![0u8; 64];
        pairwise_reduce_buffer_scalar(&mut buf);
        assert_eq!(&buf[..32], &KNOWN);
    }

    #[test]
    // zero_comm(1) = sha256_trunc254(zeros(32B) ++ zeros(32B)) = KNOWN
    // Level 1 => one pair of zero leaves hashed together
    fn zero_comm_level1_matches_known_vector() {
        assert_eq!(&zero_comm(1), &KNOWN);
    }

    #[test]
    // validates the extern "C" shim called by _bad.S
    // thin wrapper, only tests that raw pointer -> slice conversion is correct
    fn rust_scalar_shim_matches() {
        let mut input = vec![0u8; 512];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i.wrapping_mul(6700417) >> 8) as u8;
        }

        let mut a = input.clone();
        let mut b = input.clone();

        pairwise_reduce_one_level(&mut a, 8);
        unsafe { pairwise_reduce_buffer_rust_scalar(b.as_mut_ptr(), 8); }

        assert_eq!(&a[..256], &b[..256], "shim diverges from direct call");
    }
}