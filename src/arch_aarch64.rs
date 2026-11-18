#[allow(unused)]
extern "C" {
    // extrude_quads_neon_core (1x loop)
    //   count: any ≥ 1
    //   src: count × 127 bytes + 16 bytes readable past end
    //   dst: count × 128 bytes
    //
    // The 1x variant: loads, shifts, and stores one quad in NEON registers
    // advances src pointer by 127, dst pointer by 128
    //
    // Pattern: 1x: A A A A A A
    //          back-to-back shifts compete for same pipe (F1 on A72)
    //          CPU hits instructions that depend on the previous one.
    //
    //          on in-order cores (A55): data dependencies cause stalls
    //          on OoO cores (A72): multiple pipes fire in parallel, less stalls
    pub fn extrude_quads_neon_core(buf: *mut u8, count: u64);

    // extrude_quads_2x_neon_core (2x interleaved)
    //   count: must be even, ≥ 2
    //   src: count × 127 bytes + 16 bytes readable past end
    //   dst: count × 128 bytes
    //
    // The 2x variant:
    // 2 quads interleaved (8 data each + 1 mask + 4 temps = 21/32 NEON regs)
    //
    // While quad A's results are still in the pipeline, quad B's loads/shifts execute
    //
    // Pattern: 2x: A B A B A B
    //              CPU sees B doesn't depend on A -> ILP kicks in
    //              A55 stalls on 1x (dependencies). 2x avoids them.
    //              A72 partially hides 1x stalls (multiple pipes). 2x still helps.
    pub fn extrude_quads_2x_neon_core(buf: *mut u8, count: u64);
}

#[cfg(all(target_arch = "aarch64", have_neon))]
extern "C" {
    pub fn pairwise_reduce_buffer_x4_neon(buf: *mut u8, len: usize, out_pair_count: usize);
}

#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub unsafe extern "C" fn pairwise_reduce_buffer_x4(buf: *mut u8, len: usize, out_pair_count: usize) {
    aarch64::pairwise_reduce_x4(buf, len, out_pair_count);
}

#[path = "aarch64.rs"]
mod aarch64;
#[allow(unused)]
pub use aarch64::sha256_x4_aarch64; // only used in tests

#[path = "extrude_neon.rs"]
mod extrude_neon;
