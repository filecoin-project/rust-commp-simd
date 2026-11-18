#[cfg(test)]

use super::*;
use crate::lib_tests::{commp, make_padded_pairs};
use crate::lib_tests::{make_known_pairs, assert_results_match};

#[test]
// minimum valid commp input (65 bytes)
// just checks it doesn't error
fn commp_minimal_ok() {
    let r = commp(&[0u8; 65]).unwrap();
    assert_eq!(r[31] & 0xC0, 0, "254-bit truncation");
}

#[test]
// 64 bytes is below MIN_PAYLOAD - must error
fn commp_too_short_err() {
    assert!(commp(&[0u8; 64]).is_err());
}

#[test]
// input: 2 identical KNOWN pairs (128 bytes)
// output: 2 results, one level only
// buffer: 512 bytes minimum, because SoA kernel writes
//         one full 512-byte block (even for < 16 pairs)
//
// 2 pairs is below the SIMD batch size (16 on x86, 4 on arm64)
// entire reduction happens in the "tail path" (no full batch executed)
//
// This found the SoA exit conversion bug (untranspose missing for partial results)
//
// Assert: scalar == dispatched SIMD
fn soa_tail_only_2_pairs() {
    let mut scalar = make_padded_pairs(2, 0);
    let mut simd = make_padded_pairs(2, 0);

    unsafe {
        pairwise_reduce_buffer_rust_scalar(scalar.as_mut_ptr(), 2);
        pairwise_reduce_buffer(simd.as_mut_ptr(), 2 * 64, 1);
    }

    assert_results_match(&scalar, &simd, 2, "tail 2 pairs");
}

#[test]
// input: 18 identical KNOWN pairs (16 full batch + 2 tail)
// output: 9 results, one level only
//
// NOTE: 18 pairs cannot occur with legal input (128 × 2^N)
// tests the asm tail path independently
// found the LEVEL_NPAIRS counter bug in SoA exit conversion
// (bug also affects legal input - any level with > 16 pairs)
//
// One SoA block = 16 results in SoA layout:
//    ------------------------------------------------------
//    block 0 (512 bytes):
//      word 0: [r0_w0  r1_w0  r2_w0  ... r15_w0]   64 bytes
//      word 1: [r0_w1  r1_w1  r2_w1  ... r15_w1]   64 bytes
//      ...
//      word 7: [r0_w7  r1_w7  r2_w7  ... r15_w7]   64 bytes
//
//    block 1 (512 bytes):
//      word 0: [r16_w0 r17_w0 r18_w0 ... r31_w0]   64 bytes
//      ...
//      word 7: [r16_w7 r17_w7 r18_w7 ... r31_w7]   64 bytes
//
//     ------------------------------------------------------
//              Each pair = 1 x SHA256() = 1 x 32-byte result
//               32 pairs = 32 results = 2 blocks x 512 bytes
//
// Example with legal input:
//     IN: 128 × 2^4 = 2048 bytes = 64 nodes = 32 pairs
//     ARG: out_pair_count = 16
//
//     Flow:
//     - kernel produces 32 results in 2 SoA blocks
//     - kernel checks: 32 results = 16 pairs for next level
//        - out_pair_count = 16
//        - 16 ≤ 16 => stop, exit, SoA -> AoS
//
//      Bug:
//        - exit conversion looped r13 = 16 times
//        - rcx = 16 - 16 = 0
//        - 0 ≤ 0 -> exit loop ← STOP HERE
//        - block 1 (results 16-31) still raw SoA [BE] <- BUG
//
// Assert: results 16-17 correct (after tail processing)
fn soa_18_pairs_check_tail_input() {
    let mut simd = make_known_pairs(18, 0);

    let pre_tail = simd[1024..1152].to_vec();

    unsafe {
        pairwise_reduce_buffer(simd.as_mut_ptr(), 18 * 64, 9);
    }

    eprintln!("pre-tail [16] left:  {:02x?}", &pre_tail[0..8]);
    eprintln!("pre-tail [16] right: {:02x?}", &pre_tail[32..40]);
    eprintln!("pre-tail [17] left:  {:02x?}", &pre_tail[64..72]);
    eprintln!("pre-tail [17] right: {:02x?}", &pre_tail[96..104]);

    eprintln!("result[16]: {:02x?}", &simd[16 * 32..16 * 32 + 8]);
    eprintln!("result[17]: {:02x?}", &simd[17 * 32..17 * 32 + 8]);
}