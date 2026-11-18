// FR32 extrusion via WASM SIMD (port of SSE x86 "template")
//
// Instruction mapping:
//   x86 SSE           WASM SIMD              ARM NEON
//   PALIGNR $8        i8x16_shuffle(slide)   VEXT.8 #8
//   PSLLQ $N          i64x2_shl(N)           VSHL.U64 #N
//   PSRLQ $N          u64x2_shr(N)           VSHR.U64 #N
//   POR               v128_or                VORR
//   PAND              v128_and               VAND
//
// Algorithm: 16 uint64 limbs in 8 x v128. For each of the 4 nodes
// (processed in reverse: 3->2->1->0), shift bits left by (6,4,2,0)
// and clear the top 2 bits of each node's last limb.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// PALIGNR $8 equivalent: extract [prev_hi_qword, cur_lo_qword]
// Concatenates two v128s and takes the middle 16 bytes.
#[inline(always)]
fn slide(prev: v128, cur: v128) -> v128 {
    i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>(prev, cur)
}

// FR32 pad one quad in-place (127 src bytes -> 128 padded bytes).
// Direct port of extrude_sse_amd64.s - same algorithm, same register flow.
//
// input:  buf+i*127 .. buf+i*127+128 (last byte stale, masked off)
// output: buf+i*128 .. buf+i*128+128 (FR32-padded)
#[inline(always)]
unsafe fn fr32_pad_simd_inplace(buf: *mut u8, i: usize) {
    // padMask: qword 0 = all ones, qword 1 = mask62 (top 2 bits zero)
    let pad_mask = i64x2_replace_lane::<1>(
        i64x2_splat(-1i64),
        0x3FFFFFFFFFFFFFFFu64 as i64,
    );

    let load_ptr = buf.add(i * 127);
    // Load 8 x v128 = 16 uint64 limbs
    let     x0 = v128_load(load_ptr.add(0) as *const v128);   // [L0,  L1 ]
    let mut x1 = v128_load(load_ptr.add(16) as *const v128);  // [L2,  L3 ]
    let mut x2 = v128_load(load_ptr.add(32) as *const v128);  // [L4,  L5 ]
    let mut x3 = v128_load(load_ptr.add(48) as *const v128);  // [L6,  L7 ]
    let mut x4 = v128_load(load_ptr.add(64) as *const v128);  // [L8,  L9 ]
    let mut x5 = v128_load(load_ptr.add(80) as *const v128);  // [L10, L11]
    let mut x6 = v128_load(load_ptr.add(96) as *const v128);  // [L12, L13]
    let mut x7 = v128_load(load_ptr.add(112) as *const v128); // [L14, L15]

    // ===== NODE 3: shift left 6 =====
    let carry_lo = slide(x5, x6);  // [L11, L12]
    let carry_hi = slide(x6, x7);  // [L13, L14]

    x6 = v128_or(i64x2_shl(x6, 6), u64x2_shr(carry_lo, 58));
    x7 = v128_or(i64x2_shl(x7, 6), u64x2_shr(carry_hi, 58));
    x7 = v128_and(x7, pad_mask);

    // ===== NODE 2: shift left 4 =====
    let carry_lo = slide(x3, x4);  // [L7, L8]
    let carry_hi = slide(x4, x5);  // [L9, L10]

    x4 = v128_or(i64x2_shl(x4, 4), u64x2_shr(carry_lo, 60));
    x5 = v128_or(i64x2_shl(x5, 4), u64x2_shr(carry_hi, 60));
    x5 = v128_and(x5, pad_mask);

    // ===== NODE 1: shift left 2 =====
    let carry_lo = slide(x1, x2);  // [L3, L4]
    let carry_hi = slide(x2, x3);  // [L5, L6]

    x2 = v128_or(i64x2_shl(x2, 2), u64x2_shr(carry_lo, 62));
    x3 = v128_or(i64x2_shl(x3, 2), u64x2_shr(carry_hi, 62));
    x3 = v128_and(x3, pad_mask);

    // ===== NODE 0: no shift, just mask =====
    x1 = v128_and(x1, pad_mask);

    let store_ptr = buf.add(i * 128);
    // Store 8 x v128
    v128_store(store_ptr.add(0)   as *mut v128, x0);
    v128_store(store_ptr.add(16)  as *mut v128, x1);
    v128_store(store_ptr.add(32)  as *mut v128, x2);
    v128_store(store_ptr.add(48)  as *mut v128, x3);
    v128_store(store_ptr.add(64)  as *mut v128, x4);
    v128_store(store_ptr.add(80)  as *mut v128, x5);
    v128_store(store_ptr.add(96)  as *mut v128, x6);
    v128_store(store_ptr.add(112) as *mut v128, x7);

}

// extrude_one_quad(quad) - in-place, single 128-byte buffer.
//
// input:  quad[0..127] holds source bytes; quad[127] read but value irrelevant
// output: quad rewritten as 128 FR32-padded bytes
//
// Used by callers that batch FR32+hash and already have per-quad pad buffers.
pub fn extrude_one_quad(quad: &mut [u8; 128]) {
    unsafe { fr32_pad_simd_inplace(quad.as_mut_ptr(), 0); }
}

// Scalar fallback for non-wasm32 targets
// #[cfg(not(target_arch = "wasm32"))]
// fn fr32_pad_simd(source: &[u8], output: &mut [u8; 128]) {
//     output[..32].copy_from_slice(&source[..32]);
//     output[31] &= 0b00111111;
//     for i in 32..64 {
//         output[i] = (source[i] << 2) | (source[i - 1] >> 6);
//     }
//     output[63] &= 0b00111111;
//     for i in 64..96 {
//         output[i] = (source[i] << 4) | (source[i - 1] >> 4);
//     }
//     output[95] &= 0b00111111;
//     for i in 96..127 {
//         output[i] = (source[i] << 6) | (source[i - 1] >> 2);
//     }
//     output[127] = source[126] >> 2;
// }

// extrude_quads_wasm(buf, count) - in-place, single buffer.
//
// input:  buf[0..count*128] - first count*127 bytes hold src
// output: buf rewritten as count*128 FR32-padded bytes
//
// Walks backward so writes don't trash unread src.
pub unsafe extern "C" fn extrude_quads_wasm(buf: *mut u8, count: u64) {
    if count == 0 { return; }
    let mut i = count;
    loop {
        i -= 1;
        fr32_pad_simd_inplace(buf, i as usize);
        if i == 0 { break; }
    }
}

#[cfg(test)]
mod tests {
    use crate::{OUT_PER_QUAD, IN_PER_QUAD};
    use crate::lib_tests::assert_padding_bits_clear;
    use crate::lib_tests::assert_extrude_matches_scalar;
    use crate::lib_tests::{fill_pattern, fill_0xcc, fill_zero, fill_random};
    use crate::lib_tests::{testdata_path, run_extrude_vectors};

    fn extrude_wasm(buf: &mut [u8], count: usize) {
        if count == 0 { return; }
        unsafe {
            super::extrude_quads_wasm(buf.as_mut_ptr(), count as u64);
        }
    }

    #[test]
    fn wasm_matches_scalar_various_counts() {
        assert_extrude_matches_scalar(
            &[1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256],
            fill_pattern,
            extrude_wasm,
        );
    }

    #[test]
    fn wasm_padding_bits_zero() {
        for &nq in &[1, 2, 7, 16, 64] {
            let mut buf = vec![0u8; nq * OUT_PER_QUAD];
            fill_random(&mut buf[..nq * IN_PER_QUAD], &testdata_path("testdata", "random_1337.bin"));

            extrude_wasm(&mut buf, nq);

            assert_padding_bits_clear(nq, &buf);
        }
    }

    #[test]
    fn wasm_vectors_zero() { run_extrude_vectors("zero.txt", fill_zero, extrude_wasm); }

    #[test]
    fn wasm_vectors_0xcc() { run_extrude_vectors("0xCC.txt", fill_0xcc, extrude_wasm); }
}