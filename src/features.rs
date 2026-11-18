// features.rs - runtime ISA feature detection
//
// Reports which kernels are safe to call on this machine.
// Callers can check before calling, or use select_kernel() / select_extrude()
// which pick the best available automatically
//
// x86_64:
//   reduce:
//      - AVX-512F required (x16 SoA)
//      - AVX2 (x8) not yet implemented
//
//   extrude:
//      - AVX-512F + VBMI2 required (VPSHLDQ)
//      - SSE fallback always available
//
// ARM64:
//   reduce:  SHA256 crypto extensions requireds
//            (sha256h, sha256h2, sha256su0, sha256su1)
//            optional on ARMv8.0 (early chips omit it?)
//            mandatory from ARMv8.4+
//            present on: all Apple Silicon, all AWS Graviton, Cortex-A72+
//
//   extrude: baseline NEON-only > always available on ARM64

// --- reduce ---

#[cfg(target_arch = "x86_64")]
pub fn has_reduce_x16() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "x86_64")]
pub fn has_reduce_x8() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "aarch64")]
pub fn has_reduce_x4() -> bool {
    std::arch::is_aarch64_feature_detected!("sha2")
}

// --- extrude ---

#[cfg(target_arch = "x86_64")]
pub fn has_extrude_avx512() -> bool {
    // VPSHLDQ is VBMI2 - not all AVX-512 chips have it
    // Skylake-X, Cascade Lake: AVX-512F yes, VBMI2 no -> would crash
    // Ice Lake+, Zen 4+: both present
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vbmi2")
}

#[cfg(target_arch = "x86_64")]
pub fn has_extrude_sse() -> bool {
    // SSSE3 - available on all x86_64 in practice
    std::arch::is_x86_feature_detected!("ssse3")
}

#[cfg(target_arch = "aarch64")]
pub fn has_extrude_neon() -> bool {
    // baseline NEON - always true on aarch64
    true
}

// --- combined: best available kernel for this machine ---

#[cfg(target_arch = "x86_64")]
pub fn best_reduce() -> &'static str {
    if has_reduce_x16() { return "avx512_x16_soa" }
    if has_reduce_x8()  { return "avx2_x8" }
    "scalar"
}

#[cfg(target_arch = "x86_64")]
pub fn best_extrude() -> &'static str {
    if has_extrude_avx512() { return "avx512_vbmi2" }
    if has_extrude_sse()    { return "sse_ssse3" }
    "scalar"
}

#[cfg(target_arch = "aarch64")]
pub fn best_reduce() -> &'static str {
    if has_reduce_x4() { return "neon_sha256_x4" }
    "scalar"
}

#[cfg(target_arch = "aarch64")]
pub fn best_extrude() -> &'static str {
    "neon"
}

// --- fix select_kernel to use these ---

use crate::KernelFn;

#[cfg(target_arch = "x86_64")]
pub(crate) fn select_reduce_kernel() -> KernelFn {
    if has_reduce_x16() {
        return crate::arch_x86_64::pairwise_reduce_buffer_x16_soa
    }

    if has_reduce_x8() {
        return crate::arch_x86_64::pairwise_reduce_buffer_x8_avx2;
    }

    crate::scalar_kernel
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn select_reduce_kernel() -> KernelFn {
    if has_reduce_x4() {
        #[cfg(have_neon)]
        return crate::arch_aarch64::pairwise_reduce_buffer_x4_neon;

        #[cfg(not(have_neon))]
        return crate::arch_aarch64::pairwise_reduce_buffer_x4;
    }
    crate::scalar_kernel
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn select_reduce_kernel() -> KernelFn {
    #[cfg(target_feature = "simd128")]
    return crate::arch_wasm32::pairwise_reduce_buffer_wasm;

    #[cfg(not(target_feature = "simd128"))]
    crate::scalar_kernel
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn select_extrude_kernel() -> ExtrudeFn {
    crate::arch_wasm32::extrude_quads_wasm
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn select_extrude_kernel() -> ExtrudeFn {
    crate::arch_aarch64::extrude_quads_2x_neon_core
}

use crate::ExtrudeFn;

#[cfg(target_arch = "x86_64")]
pub(crate) fn select_extrude_kernel() -> ExtrudeFn {
    if has_extrude_avx512() {
        return crate::arch_x86_64::extrude_quads_avx512_core;
    }
    if has_extrude_sse() {
        return crate::arch_x86_64::extrude_quads_sse_core;
    }
    crate::scalar_extrude_kernel
}