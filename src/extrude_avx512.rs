//! FR32 extrusion via AVX512 - wrapper for extrude_avx512.S
//!
//! The 4x loop handles 4 quads per iteration:
//! - loads all 16 YMMs
//! - does all the shifts interleaved
//! - stores all 16 YMMs
//! - advances by 508 bytes [q-127][q-127][q-127][q-127]
//!                         |- - - - - - - - - - - - - ^
//!
//! Register budget consumed:
//!    4 YMMs x 4 quads + 4 carry + 1 mask = 21/32 regs
//!
//! The 1x loop handles 1 quad per iteration:
//! - shift, store 4 YMMs
//! - advance 127 bytes
//!
//! Register budget consumed:
//!     4 YMMs (data) + 1 carry + 1 mask = 6/32 regs
//!

#[cfg(all(test, target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use crate::extrude_quads_avx512_core;
    use crate::{IN_PER_QUAD, OUT_PER_QUAD};
    use crate::lib_tests::{
        fill_zero, fill_0xcc, fill_pattern,
        run_extrude_vectors, assert_extrude_matches_scalar,
        assert_padding_bits_clear,
    };

    #[test]
    // input: deterministic fill_pattern()
    // output: must match scalar_extrude() <-> byte-for-byte
    //
    //  nq: 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256
    //      |     |        |         |
    //      ⮑ 1x loop     ⮑ 4x loop boundaries
    //
    //  same input => (through asm and scalar) <-> must produce identical output
    //
    //  Assert: asm[i] == scalar[i] for every byte
    //  Catches: off-by-one in loop count, carry across quad boundaries
    fn avx512_matches_scalar_various() {
        assert_extrude_matches_scalar(
            &[1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256],
            fill_pattern,
            extrude_quads_avx512,
        );
    }

    // -------------------------------------
    //  --- FR32 padding and bit checks ---
    // -------------------------------------

    #[test]
    // input: pseudorandom bytes by fill_pattern() (catches what zeros/0xFF miss)
    // check: top 2 bits at bytes 31, 63, 95, 127 must be zero
    //                            |    |    |    |
    //                            ⮑ FR32 padding positions (4 per 128B unit)
    //
    //  byte:  [datax31] [XX] [datax31] [XX] [datax31] [XX] [datax31] [XX]
    //                    ^^                                            ^^
    //                    top 2 bits forced to 0 by FR32
    //
    //  Assert: top 2 bits == 0 at each padding position {31, 63, 95, 127}
    //  Catches: mask bugs, shift-amount off-by-one
    fn avx512_padding_bits_zero() {
        for &nq in &[1, 2, 7, 16, 64] {
            let mut buf = vec![0u8; nq * OUT_PER_QUAD];
            fill_pattern(&mut buf[0..nq * IN_PER_QUAD]);

            unsafe { extrude_quads_avx512_core(buf.as_mut_ptr(), nq as u64); }

            assert_padding_bits_clear(nq, &buf);
        }
    }

    // input:  count x 127 bytes (src), 16 bytes readable past end
    // output: count x 128 bytes (buf), FR32 padded
    //
    // 1x loop: one quad per iteration. Must be [any count >= 1]
    fn extrude_quads_avx512(buf: &mut [u8], count: usize) {
        if count == 0 { return; }
        unsafe { extrude_quads_avx512_core(buf.as_mut_ptr(), count as u64); }
    }

    // -------------------------------------
    //  --- Run against pre-gen vectors ---
    // -------------------------------------

    #[test]
    fn avx512_vectors_zero() { run_extrude_vectors("zero.txt", fill_zero, extrude_quads_avx512); }

    #[test]
    // 11001100 -> may catch shift/carry bugs that zeros hide
    fn avx512_vectors_0xcc() { run_extrude_vectors("0xCC.txt", fill_0xcc, extrude_quads_avx512); }
}