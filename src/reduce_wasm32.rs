// WASM32 pairwise reduce
//
// hash_4_pairs: SoA throughout, calls wasm32::sha256_block directly
// pairwise_reduce_buffer_wasm: full tree loop calling hash_4_pairs

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use super::wasm32::{H_INIT, sha256_block, transpose_4x4, bswap};

// Hash up to 4 pairs in-place
//
// input:  buf[lane*64 .. lane*64+64] for each lane (0..n)
// output: buf[lane*32 .. lane*32+32] compact, trunc254 applied
//
// n < 4: dummy lanes point to lane 0 data
// kernel hashes them but results are discarded
#[cfg(target_arch = "wasm32")]
unsafe fn hash_4_pairs(buf: *mut u8, n: usize) {
    debug_assert!(n >= 1 && n <= 4);

    let mut ptrs = [buf as *const u8; 4];
    for lane in 0..n { ptrs[lane] = buf.add(lane * 64); }

    let mut state: [v128; 8] = [
        u32x4_splat(H_INIT[0]), u32x4_splat(H_INIT[1]),
        u32x4_splat(H_INIT[2]), u32x4_splat(H_INIT[3]),
        u32x4_splat(H_INIT[4]), u32x4_splat(H_INIT[5]),
        u32x4_splat(H_INIT[6]), u32x4_splat(H_INIT[7]),
    ];

    // --- block 1: pair data ---
    //
    //   group 0: left[0..16]   -> w[0..4]
    //   group 1: left[16..32]  -> w[4..8]
    //   group 2: right[0..16]  -> w[8..12]
    //   group 3: right[16..32] -> w[12..16]

    let mut w = [u32x4_splat(0); 16];

    for group in 0..4 {
        let goff = group * 16;
        let r0 = v128_load(ptrs[0].add(goff) as *const v128);
        let r1 = v128_load(ptrs[1].add(goff) as *const v128);
        let r2 = v128_load(ptrs[2].add(goff) as *const v128);
        let r3 = v128_load(ptrs[3].add(goff) as *const v128);
        let (t0, t1, t2, t3) = transpose_4x4(r0, r1, r2, r3);
        w[group * 4 + 0] = bswap(t0);
        w[group * 4 + 1] = bswap(t1);
        w[group * 4 + 2] = bswap(t2);
        w[group * 4 + 3] = bswap(t3);
    }

    sha256_block(&mut state, &mut w);

    // --- block 2: SHA256 padding for 64-byte input ---

    let mut w_pad = [u32x4_splat(0); 16];
    w_pad[0] = u32x4_splat(0x80000000);
    w_pad[15] = u32x4_splat(0x00000200);

    sha256_block(&mut state, &mut w_pad);

    // --- extract: un-transpose + bswap to BE bytes ---

    let (t0_lo, t1_lo, t2_lo, t3_lo) = transpose_4x4(
        bswap(state[0]), bswap(state[1]), bswap(state[2]), bswap(state[3]),
    );
    let (t0_hi, t1_hi, t2_hi, t3_hi) = transpose_4x4(
        bswap(state[4]), bswap(state[5]), bswap(state[6]), bswap(state[7]),
    );

    let mut results = [[0u8; crate::NODE_SIZE]; 4];

    v128_store(results[0].as_mut_ptr() as *mut v128, t0_lo);
    v128_store(results[0].as_mut_ptr().add(16) as *mut v128, t0_hi);
    v128_store(results[1].as_mut_ptr() as *mut v128, t1_lo);
    v128_store(results[1].as_mut_ptr().add(16) as *mut v128, t1_hi);
    v128_store(results[2].as_mut_ptr() as *mut v128, t2_lo);
    v128_store(results[2].as_mut_ptr().add(16) as *mut v128, t2_hi);
    v128_store(results[3].as_mut_ptr() as *mut v128, t3_lo);
    v128_store(results[3].as_mut_ptr().add(16) as *mut v128, t3_hi);

    results[0][31] &= 0x3F;
    results[1][31] &= 0x3F;
    results[2][31] &= 0x3F;
    results[3][31] &= 0x3F;

    for lane in 0..n {
        core::ptr::copy_nonoverlapping(results[lane].as_ptr(), buf.add(lane * 32), 32);
    }
}

// Scalar fallback for host testing
#[cfg(not(target_arch = "wasm32"))]
unsafe fn hash_4_pairs(buf: *mut u8, n: usize) {
    use sha2::{Digest, Sha256};

    for i in 0..n {
        let read_idx = i * 64;
        let mut hasher = Sha256::new();
        hasher.update(std::slice::from_raw_parts(buf.add(read_idx), 64));
        let mut result: [u8; 32] = hasher.finalize().into();
        result[31] &= 0x3F;
        core::ptr::copy_nonoverlapping(result.as_ptr(), buf.add(i * 32), 32);
    }
}

// Full tree reduction
//
// same signature as KernelFn: unsafe extern "C" fn(*mut u8, usize, usize)
pub unsafe extern "C" fn pairwise_reduce_buffer_wasm(
    buf: *mut u8,
    byte_len: usize,
    out_pair_count: usize,
) {
    let mut n_pairs = byte_len / 64;

    while n_pairs > out_pair_count {
        let full = n_pairs / 4;
        let tail = n_pairs % 4;

        for b in 0..full {
            hash_4_pairs(buf.add(b * 256), 4);
            if b > 0 {
                core::ptr::copy_nonoverlapping(buf.add(b * 256), buf.add(b * 128), 128);
            }
        }

        if tail > 0 {
            let base = full * 256;
            hash_4_pairs(buf.add(base), tail);
            if full > 0 {
                core::ptr::copy_nonoverlapping(buf.add(base), buf.add(full * 128), tail * 32);
            }
        }

        n_pairs /= 2;
    }
}