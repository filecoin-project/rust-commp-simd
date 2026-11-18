//! Arch-neutral pairwise reduce via Rust loop + pluggable SHA256 x4 block kernel
//!
//! Calls a SHA256 x4 block kernel (provided as function pointer) to hash
//! up to 4 pairs at a time, then compacts results for the next tree level.
//!
//! Originally written for aarch64 before pairwise_reduce_x4_neon.S existed.
//! Now arch-neutral - any SHA256 x4 block kernel can be plugged in.
//! Kept as independent reference: tests compare this against arch-specific .S kernels.

use crate::NODE_SIZE;

pub type BlockFn = unsafe extern "C" fn(*mut u8, u64);

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const PAD_BLOCK: [u8; 64] = {
    let mut b = [0u8; 64];
    b[0] = 0x80;   // W[0]  = 0x80000000 after rev32
    b[62] = 0x02;  // W[15] = 0x00000200 after rev32
    b
};

#[repr(C, align(16))]
// The kernel takes an array of 4 pointers
// Each pointer says "here's where my 64-byte message is"
pub(crate) struct Sha256x4Args {
  // digest[0] = [A, B, C, D, E, F, G, H]  <- lane 0's hash (32 bytes)
  // digest[1] = [A, B, C, D, E, F, G, H]  <- lane 1's hash (32 bytes)
  // digest[2] = [A, B, C, D, E, F, G, H]  <- lane 2's hash (32 bytes)
  // digest[3] = [A, B, C, D, E, F, G, H]  <- lane 3's hash (32 bytes)
  pub(crate) digest: [[u32; 8]; 4],

  // data_ptr[0] = -> [64 bytes: pair 0 data] <- lane 0's message
  // data_ptr[1] = -> [64 bytes: pair 1 data] <- lane 1's message
  // data_ptr[2] = -> [64 bytes: pair 2 data] <- lane 2's message
  // data_ptr[3] = -> [64 bytes: pair 3 data] <- lane 3's message
  pub(crate) data_ptr: [*const u8; 4],  // 4 pointers, each to a 64-byte message
}

#[inline(always)]
// Block 1: actual 64-byte pair data
// lanes 0..n   - real pair pointers (buf + lane*64)
// lanes n..4   - dup lane 0 (buf), kernel hashes them but collect_results ignores them
//
// NOTE: production input is always 128 x 2^N bytes (multiple of 4 pairs)
// n < 4 only happens during experimentation and testin
//
// When a tree level has a non-multiple-of-4 pair count:
//   ```
//    let full = n_pairs / 4; // complete 4-pair batches
//    let tail = n_pairs % 4; // remaining pairs (0-3)
//   ```
//
//    full batches: always n=4
//    hash_4_pairs(buf.add(b * 256), 4, sha256_block);
//
//    tail: n = 1, 2, or 3
//    hash_4_pairs(buf.add(base), tail, sha256_block);
unsafe fn block_data(args: &mut Sha256x4Args, buf: *mut u8, n: usize, sha256_block: BlockFn) {
    // Increasing pointer arithmetics +0, +64, +128, +192
    for lane in 0..n { args.data_ptr[lane] = buf.add(lane * 64); }

    // point to valid mem, so kernel doesn't read garbage
    for lane in n..4 { args.data_ptr[lane] = buf; }

    sha256_block(args as *mut _ as *mut u8, 1);
}

#[inline(always)]
// Block 2: sha256 padding block (same for all lanes, 0x80 + length = 0x200)
// all 4 lanes point to the same constant
unsafe fn block_pad(args: &mut Sha256x4Args, sha256_block: BlockFn) {
    for lane in 0..4 { args.data_ptr[lane] = PAD_BLOCK.as_ptr(); }
    sha256_block(args as *mut _ as *mut u8, 1);
}


#[inline(always)]
// input:  args.digest[4][8] - 4 lanes x 8 BE u32 words (AoS)
// output: buf[lane*32 .. lane*32+32] contiguous 32-byte nodes, trunc254 applied
//
// BE conversion: digest u32 words -> BE bytes
unsafe fn collect_results(args: &Sha256x4Args, buf: *mut u8, n: usize) {
    let mut results = [[0u8; NODE_SIZE]; 4];
    for lane in 0..n {
        for w in 0..8 {
            results[lane][w*4..w*4+4]
                .copy_from_slice(&args.digest[lane][w].to_be_bytes());
        }

        results[lane][31] &= 0x3F;
    }
    for lane in 0..n {
        core::ptr::copy_nonoverlapping(results[lane].as_ptr(), buf.add(lane * 32), 32);
    }
}

// Hash one batch of up to 4 pairs in-place.
// Input:  buf[lane*64 .. lane*64+64] for each lane
// Output: buf[lane*32 .. lane*32+32] compact, trunc254 applied
// Collects all results before writing to avoid overlap.
unsafe fn hash_4_pairs(buf: *mut u8, n: usize, sha256_block: BlockFn) {
    debug_assert!(n >= 1 && n <= 4);

    let mut args = Sha256x4Args {
        digest: [IV; 4],
        data_ptr: [core::ptr::null(); 4],
    };

    block_data(&mut args, buf, n, sha256_block);
    block_pad(&mut args, sha256_block);

    // collect digest words into contiguous 32-byte nodes, BE + trunc254
    collect_results(&args, buf, n);
}

#[allow(unused)]
// Full level loop
// same ENTRY sig as x86: (buf, byte_len, out_pair_count)
//
// pairwise_reduce_loop (Rust loop)
//   -> hash_4_pairs (Rust: setup args, collect results, trunc254)
//     -> sha256_block (arch-specific .S kernel, passed as fn pointer)
//
// Kept as independent reference: tests compare this against the full .S kernels.
pub unsafe fn pairwise_reduce_loop(
    buf: *mut u8,
    byte_len: usize,
    out_pair_count: usize,
    sha256_block: BlockFn,
) {
    let mut n_pairs = byte_len / 64;
    // assert!(n_pairs.is_power_of_two(), "input must be 128 x 2^N bytes, got {n_pairs} pairs");

    while n_pairs > out_pair_count {
        let full = n_pairs / 4; // how many complete 4-pair batches
        let tail = n_pairs % 4; // remaining pairs (0-3)

        // input:  buf[b*256 .. b*256+256] (4 x pairs  [256B])
        // output: buf[b*128 .. b*128+128] (4 x result [128B])
        //
        // batch 0: 4 pairs at buf[0..256]   -> 4 results at buf[0..128]
        // batch 1: 4 pairs at buf[256..512] -> 4 results at buf[128..256]
        // batch 2: 4 pairs at buf[512..768] -> 4 results at buf[256..384]
        //
        // hash_4_pairs reads then writes to same pointer
        // results at [0..128] don't overlap unread data (input is [0..256], read first)
        for b in 0..full {
            hash_4_pairs(buf.add(b * 256), 4, sha256_block);
            if b > 0 {
                // Results need to be contiguous for the next level
                // batch 0 results already at buf[0..128]
                // batch 1+ results sit at buf[b*256] but belong at buf[b*128]
                // So close the gaps:
                core::ptr::copy_nonoverlapping(buf.add(b * 256), buf.add(b * 128), 128);
            }
        }

        // remaining < 4 pairs
        if tail > 0 {
            let base = full * 256;
            hash_4_pairs(buf.add(base), tail, sha256_block);
            if full > 0 {
                core::ptr::copy_nonoverlapping(buf.add(base), buf.add(full * 128), tail * 32);
            }
        }

        n_pairs /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    use crate::{sha256_trunc254, fr32_pad_slab};
    use crate::lib_tests::{known_pair, make_padded_pairs, assert_root_matches};

    // -----------------------------------------------
    //  --- arch-specific block kernel selection ---
    // -----------------------------------------------

    // resolved once, same pattern as KERNEL in lib.rs
    static BLOCK_FN: OnceLock<BlockFn> = OnceLock::new();

    fn select_sha256_x4() -> BlockFn {
        #[cfg(target_arch = "aarch64")]
        {
            return crate::sha256_x4_aarch64;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // TODO: replace with sha256_x4_sse when available
            // SSE x4 is the natural fit - SSE always present on x86_64
            unsafe extern "C" fn x4(args: *mut u8, n: u64) {
                crate::scalar::sha256_x4_fallback(args, n);
            }
            return x4;
        }

        #[cfg(target_arch = "wasm32")]
        {
            return crate::arch_wasm32::sha256_x4_wasm;

        }
    }

    // convenience: resolved block kernel, no indirection per call
    unsafe fn reduce_loop(buf: *mut u8, byte_len: usize, out_pair_count: usize) {
        let block = BLOCK_FN.get_or_init(select_sha256_x4);
        pairwise_reduce_loop(buf, byte_len, out_pair_count, *block);
    }

    // also for tests that call hash_4_pairs directly
    fn sha256_x4() -> BlockFn {
        *BLOCK_FN.get_or_init(select_sha256_x4)
    }

    // -----------------------------------------------
    //  --- production level-loop kernel selection ---
    // -----------------------------------------------

    type ProdKernelFn = unsafe extern "C" fn(*mut u8, usize, usize);
    static PROD_KERNEL: OnceLock<ProdKernelFn> = OnceLock::new();

    fn select_prod_kernel() -> ProdKernelFn {
        #[cfg(target_arch = "aarch64")]
        {
            return crate::pairwise_reduce_buffer_x4_neon;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return crate::pairwise_reduce_buffer_x16_soa;
            }
            return crate::pairwise_reduce_buffer_x8_avx2;
        }

        #[cfg(target_arch = "wasm32")]
        {
            return crate::arch_wasm32::pairwise_reduce_buffer_wasm;
        }
    }

    unsafe fn prod_kernel(buf: *mut u8, len: usize, out_pair_count: usize) {
        let k = PROD_KERNEL.get_or_init(select_prod_kernel);
        k(buf, len, out_pair_count);
    }

    // hash one pair through x4 block kernel, compare to scalar sha256_trunc254
    fn assert_single_pair(pair: &[u8; 64], label: &str) {
        let mut buf = [0u8; 256];
        buf[..64].copy_from_slice(pair);

        unsafe { hash_4_pairs(buf.as_mut_ptr(), 1, sha256_x4()); }

        let expected = sha256_trunc254(pair);

        eprintln!("{label} expected: {:02x?}", &expected[..4]);
        eprintln!("{label} got:      {:02x?}", &buf[..4]);

        assert_eq!(&buf[..32], &expected, "{label}");
    }

    // -------------------------------------------------
    //  --- Rust loop vs scalar (arch-neutral tests) ---
    // -------------------------------------------------

    #[test]
    // one pair with non-zero input (KNOWN (32) + KNOWN (32))
    // zeros are degenerate, this can catch bugs which require actual data

    // Zero input:    [00 00 00 ... 00] (64 zero bytes)
    // This test:     [f5 a5 fd ... 0b] + [f5 a5 fd ... 0b]
    //                    (32 bytes)          (32 bytes)
    //
    // Assert: hash_4_pairs out == sha256_trunc254(KNOWN + KNOWN)
    // Weakness: (L == R) A bug that swaps left/right wouldn't show up here
    fn sha256_x4_known_pair() {
        assert_single_pair(&known_pair(), "KNOWN+KNOWN");
    }

    // -------------------------------------------------------
    //  --- Rust loop vs production .S kernel (all arches) ---
    // -------------------------------------------------------

    mod kernel_tests {
        use super::*;

        #[test]
        // fr32-padded 192 zero bytes through .S kernel vs scalar, full tree
        // first test that exercises the full .S pairwise kernel
        fn commp_192() {
            let n_quads = (192 + 126) / 127;
            let buf = fr32_pad_slab(&vec![0u8; n_quads * 127]);

            assert_root_matches(&buf, |b| {
                unsafe { prod_kernel(b.as_mut_ptr(), buf.len(), 0); }
            }, "commp 192");
        }

        #[test]
        // input: 3 pairs (192 bytes), ILLEGAL (not 128 x 2^N)
        // 3 is not a valid input pair count, kernel doesn't hande it
        //
        // result[0] and [1] written correctly
        // result[2] NOT written, buf still has whatever was there before teh call
        fn three_pairs_illegal() {
            let mut old = make_padded_pairs(3, 256);
            let mut new = make_padded_pairs(3, 256);

            // out_pair_count=2: stops after one level
            // n_pairs = 3; It's > 2 so process, n_pairs becomes 1
            // n_pairs = 1; done (1 not > 2, ich bin raus)
            unsafe { reduce_loop(old.as_mut_ptr(), 3 * 64, 2); }
            unsafe { prod_kernel(new.as_mut_ptr(), 3 * 64, 2); }

            // result[0] and [1]: kernel loads these correctly even for tail=3
            assert_eq!(&old[..32],    &new[..32],   "result[0] mismatch");
            assert_eq!(&old[32..64],  &new[32..64], "result[1] mismatch");

            #[cfg(target_arch = "aarch64")]
            // result[2] MUST differ, Rust wrote SHA(pair2)
            // kernel left whatever was aready there untouched
            assert_ne!(&old[64..96], &new[64..96], "result[2] should be garbage");
        }

        fn assert_known_pair_through_kernel(n_pairs: usize, out_pair_count: usize, label: &str) {
            let mut buf = make_padded_pairs(n_pairs, 0);

            let expected = sha256_trunc254(&known_pair());
            unsafe { prod_kernel(buf.as_mut_ptr(), n_pairs * 64, out_pair_count); }

            eprintln!("{label} expected: {:02x?}", &expected[..4]);
            eprintln!("{label} got:      {:02x?}", &buf[..4]);

            assert_eq!(&buf[..32], &expected, "{label}");
        }

        #[test]
        // 4 identical pairs through .S kernel, one level only
        // all lanes same input -> every result must equal scalar sha256_trunc254
        fn four_pairs_known_single_batch() {
            assert_known_pair_through_kernel(4, 2, "4 known pairs");
        }

        #[test]
        // single pair through .S kernel vs scalar sha256_trunc254, full reduction
        fn single_pair_known() {
            assert_known_pair_through_kernel(1, 0, "single known pair");
        }
    }
}