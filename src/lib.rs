// wasm32 builds without std (tests keep std for the harness)
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]

extern crate alloc;

#[cfg(not(all(target_arch = "wasm32", not(test))))]
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
mod arch_x86_64;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use arch_x86_64::*;

#[cfg(target_arch = "aarch64")]
mod arch_aarch64;
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use arch_aarch64::*;

#[cfg(target_arch = "wasm32")]
pub mod arch_wasm32;
#[cfg(target_arch = "wasm32")]
pub use arch_wasm32::*;

pub mod features;

// Dispatch - resolved once at init, zero branches on the hot path
type KernelFn = unsafe extern "C" fn(*mut u8, usize, usize);
#[cfg(not(all(target_arch = "wasm32", not(test))))]
static KERNEL: OnceLock<KernelFn> = OnceLock::new();

const NODE_SIZE:    usize = 32;
const IN_PER_QUAD:  usize = 127;
const OUT_PER_QUAD: usize = 128;
#[cfg(not(all(target_arch = "wasm32", not(test))))]
const MAX_LEVELS:   usize = 64;

fn select_kernel() -> KernelFn {
    features::select_reduce_kernel()
}

// Reduce: dispatch scalar fallback when no SIMD available)
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
unsafe extern "C" fn scalar_kernel(buff: *mut u8, len: usize, _out_pair_count: usize) {
    let n_pairs = len / 64;
    let slice = core::slice::from_raw_parts_mut(buff, len);
    pairwise_reduce_one_level(slice, n_pairs);
}

// Extrude: dispatch scalar fallback when no SIMD available
#[allow(dead_code)]
unsafe extern "C" fn scalar_extrude_kernel(buf: *mut u8, count: u64) {
    let slice = core::slice::from_raw_parts_mut(buf, (count as usize) * OUT_PER_QUAD);
    scalar::scalar_extrude_inplace(slice, count as usize);
}

// asm tail callback - called from .S when < 16 pairs remain
#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub unsafe extern "C" fn pairwise_reduce_buffer_rust_scalar(buf: *mut u8, n_pairs: usize) {
    let slice = core::slice::from_raw_parts_mut(buf, n_pairs * 64);
    pairwise_reduce_one_level(slice, n_pairs);
}

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub unsafe extern "C" fn pairwise_reduce_buffer(buff: *mut u8, len: usize, out_pair_count: usize) {
    #[cfg(not(all(target_arch = "wasm32", not(test))))]
    let kernel = *KERNEL.get_or_init(select_kernel);
    // wasm32 has one kernel, picked at build time, nothing to cache
    #[cfg(all(target_arch = "wasm32", not(test)))]
    let kernel = select_kernel();
    kernel(buff, len, out_pair_count);
}

type ExtrudeFn = unsafe extern "C" fn(*mut u8, u64);
#[cfg(not(all(target_arch = "wasm32", not(test))))]
static EXTRUDE_KERNEL: OnceLock<ExtrudeFn> = OnceLock::new();

fn select_extrude_kernel() -> ExtrudeFn {
    features::select_extrude_kernel()
}

#[cfg_attr(not(target_arch = "wasm32"), no_mangle)]
pub unsafe extern "C" fn extrude_quads(buf: *mut u8, count: u64) {
    #[cfg(not(all(target_arch = "wasm32", not(test))))]
    let kernel = *EXTRUDE_KERNEL.get_or_init(select_extrude_kernel);
    #[cfg(all(target_arch = "wasm32", not(test)))]
    let kernel = select_extrude_kernel();
    kernel(buf, count);
}

#[cfg(test)]
mod lib_tests;

#[cfg(test)]
mod commp_tests;

#[cfg(test)]
mod commp_vectors;

mod reduce_loop;

pub mod scalar;
pub use scalar::*;