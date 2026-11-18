//! FR32 extrusion via NEON - wrapper for extrude_quads_neon.S
//!
//! The 2x loop handles 2 quads per iteration:
//! - loads all 16 v-regs (8 per quad)
//! - does all the shifts interleaved
//! - stores all 16 v-regs
//! - advances by 254 bytes [q-127][q-127]
//!                         |- - - - - - -^
//!
//! Register budget consumed:
//!     8 v-regs x 2 quads + 4 carry + 1 mask = 21/32 regs
//!
//! The 1x loop handles 1 quad per iteration:
//! - shift, store 8 v-regs
//! - advance 127 bytes.
//!
//! Register budget consumed:
//!     8 v-regs (data) + 2 carry + 1 mask = 11/32 regs
//!

#[cfg(test)]
mod tests {
    use crate::{OUT_PER_QUAD, IN_PER_QUAD};
    use crate::lib_tests::assert_padding_bits_clear;
    use crate::lib_tests::assert_extrude_matches_scalar;
    use crate::lib_tests::{fill_pattern, fill_0xcc, fill_zero, fill_random,};
    use crate::lib_tests::{testdata_path, run_extrude_vectors};
    use crate::arch_aarch64::{extrude_quads_neon_core, extrude_quads_2x_neon_core};

    // input:  count x 127 bytes (src), 16 bytes readable past end
    // output: count x 128 bytes (dst), FR32 padded
    //
    // 1x loop: one quad per iteration. Must be [any count ≥ 1]
    fn extrude_quads_neon(buf: &mut [u8], count: usize) {
        if count == 0 { return; }

        unsafe {
            extrude_quads_neon_core(buf.as_mut_ptr(), count as u64);
        }
    }

    // input:  count x 127 bytes (src), 16 bytes readable past end
    // output: count x 128 bytes (dst), FR32 padded
    //
    // 2x loop: two quads per iteration. Must be [even count ≥ 2]
    fn extrude_quads_neon_x2(buf: &mut [u8], count: usize) {
        assert!(count >= 2 && count % 2 == 0, "2x requires even count ≥ 2");

        unsafe {
            extrude_quads_2x_neon_core(buf.as_mut_ptr(), count as u64);
        }
    }

    #[test]
    fn neon_matches_scalar_various_counts() {
        assert_extrude_matches_scalar(
            &[1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256],
            fill_pattern,
            extrude_quads_neon,
        );
    }

    // ----------------------------------------------------------
    //  --- 2x kernel: catches off-by-one in pointer advance ---
    // ----------------------------------------------------------
    // Bug: x1 advance was #510, should have been #509.
    // Affects: 2x kernel only. 1x uses a different advance (#191) and was correct.
    //
    // Silent for count=2, surfaces at count>=4:
    //     count = 2 - ONE iteration runs, bad advance happens at loop tail
    //                 but register x1 is never read again -> no damage
    //     count = 4 - TWO iterations run, second iteration reads src from
    //                 the corrupted register x1 -> wrong pair B, wrong output
    //
    // Before these tests 1x variant was tested, but 2x only benched
    #[test]
    fn neon_x2_matches_scalar_various_counts() {
        assert_extrude_matches_scalar(
            &[2, 4, 8, 16, 32, 64, 128, 256],  // even only for x2
            fill_pattern,
            extrude_quads_neon_x2,
        );
    }

    // -------------------------------------
    //  --- FR32 padding and bit checks ---
    // -------------------------------------
    #[test]
    fn neon_padding_bits_zero() {
        for &nq in &[1, 2, 7, 16, 64] {
            let mut buf = vec![0u8; nq * OUT_PER_QUAD];
            fill_random(&mut buf[0..nq * IN_PER_QUAD], &testdata_path("testdata", "random_1337.bin"));

            extrude_quads_neon(&mut buf, nq);

            assert_padding_bits_clear(nq, &buf);
        }
    }

    // ------------------------------------
    // --- Run against pre-gen vectors ---
    // ------------------------------------
    #[test]
    fn neon_vectors_zero() { run_extrude_vectors("zero.txt", fill_zero, extrude_quads_neon); }

    #[test]
    fn neon_vectors_0xcc() { run_extrude_vectors("0xCC.txt", fill_0xcc, extrude_quads_neon); }
}