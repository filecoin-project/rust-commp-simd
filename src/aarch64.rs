// add the sha256 block kernel extern (was in aarch64.rs):
extern "C" {
    pub fn sha256_x4_aarch64(args: *mut u8, num_blocks: u64);
}

// wrapper calls arch-neutral loop with aarch64 block kernel
pub unsafe fn pairwise_reduce_x4(buf: *mut u8, byte_len: usize, out_pair_count: usize) {
    crate::reduce_loop::pairwise_reduce_loop(buf, byte_len, out_pair_count, sha256_x4_aarch64);
}