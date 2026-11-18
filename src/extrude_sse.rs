//! FR32 extrusion via SSE - wrapper for extrude_quads_sse.S
//!
//! The 1x loop handles 1 quad per iteration:
//! - loads 8 x XMM (128-bit wide) regs from src buf to expand 127 bytes of data
//! - shifts each node (3 -> 2 -> 1 -> 0), palignr + psrlq + por
//! - stores 8 xmm registers back in-place (src buf)
//! - loop walks backwards: last quad to first using two ptrs:
//!      dst_ptr: r8 = buf + (count-1) * 128, d_i-=128 each iter
//!      src_ptr: r9 = buf + (count-1) * 127, s_i-=127 each iter
//!
//! Registers utilized:
//!       8 x XMM (data)
//!       2 x carry
//!       1 x mask
//!       total: (11/16) regs

#[cfg(test)]
mod tests {
    use crate::arch_x86_64::extrude_quads_sse_core;
    use crate::lib_tests::{
        fill_zero, fill_0xcc, fill_pattern, fill_random,
        run_extrude_vectors, assert_extrude_matches_scalar,
        assert_padding_bits_clear,
        testdata_path,
    };
    use crate::{IN_PER_QUAD, OUT_PER_QUAD};

    fn extrude_quads_sse(buf: &mut [u8], count: usize) {
        if count == 0 { return; }
        unsafe { extrude_quads_sse_core(buf.as_mut_ptr(), count as u64); }
    }

    #[test]
    fn sse_matches_scalar_various_counts() {
        assert_extrude_matches_scalar(
            &[1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256],
            fill_pattern,
            extrude_quads_sse,
        );
    }

    #[test]
    fn sse_padding_bits_zero() {
        for &nq in &[1, 2, 7, 16, 64] {
            let mut buf = vec![0u8; nq * OUT_PER_QUAD];
            fill_random(&mut buf[0..nq * IN_PER_QUAD], &testdata_path("testdata", "random_1337.bin"));
            extrude_quads_sse(&mut buf, nq);
            assert_padding_bits_clear(nq, &buf);
        }
    }

    #[test]
    fn sse_vectors_zero() { run_extrude_vectors("zero.txt", fill_zero, extrude_quads_sse); }

    #[test]
    fn sse_vectors_0xcc() { run_extrude_vectors("0xCC.txt", fill_0xcc, extrude_quads_sse); }
}