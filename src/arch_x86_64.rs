// x86_64 kernels
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
extern "C" {
    pub fn pairwise_reduce_buffer_x16_soa(buff: *mut u8, len: usize, out_pair_count: usize);

    // pairwise_reduce_buffer_x8_avx2 (8-lane AVX2 SoA)
    //   buf: 128 × 2^N bytes, minimum 2 pairs
    //   len: byte count (n_pairs × 64)
    //   out_pair_count: stop reducing when n_pairs <= this (0 = full reduction)
    pub fn pairwise_reduce_buffer_x8_avx2(buff: *mut u8, len: usize, out_pair_count: usize);

    // extrude_quads_avx512_core (1x loop)
    //   count: any >= 1
    //   src: count x 127 bytes + 16 bytes readable past end
    //   dst: count x 128 bytes
    //
    // The 1x variant: load 4 YMMs (data) + 1 carry + 1 mask = 6 regs
    // shift, store 4 YMMs, advance 127 bytes
    //
    // Pattern: 1x: A A A A A A
    //          uses VPSHLDQ (does shl + ushr + orr in one instruction)
    //             + VALIGNQ (same as NEON ext but in 8-byte offsets only)
    //          3 instructions per node (vs 5 on NEON)
    pub fn extrude_quads_avx512_core(dst: *mut u8, count: u64);

    // extrude_quads_4x_avx512_core (4x interleaved)
    //   count: must be multiple of 4, >= 4
    //   src: count x 127 bytes + 16 bytes readable past end
    //   dst: count x 128 bytes
    //
    // 4 quads interleaved: loads all 16 YMMs, shifts interleaved, stores all 16
    // advances by 508 bytes (4 x 127)
    //
    // AVX512 has 32 x 256-bit YMMs (vs NEON's 32 x 128-bit)
    // 4 quads x 4 YMMs = 16 data + 4 carry temps + mask = 21/32 regs
    // (NEON maxes out at 2x because 128-bit regs need 8 per quad)
    //
    // Pattern: 4x: A B C D A B C D
    //              same ILP principle as NEON 2x but wider
    pub fn extrude_quads_4x_avx512_core(dst: *mut u8, count: u64);

    // extrude_quads_sse_core (1x loop, in-place)
    //   buf: at least count * 128 bytes
    //        bytes [0 .. count*127] hold source bytes to be expanded
    //   count: >= 0 (zero is a no-op)
    //
    //   Effect: kernel rewrites in-place as (count x 128 bytes)
    pub fn extrude_quads_sse_core(buf: *mut u8, count: u64);
}

#[path = "extrude_sse.rs"]
mod extrude_sse;

#[path = "extrude_avx512.rs"]
mod extrude_avx512;
