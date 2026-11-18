fn main() {
    println!("cargo::rustc-check-cfg=cfg(have_neon)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    match arch.as_str() {
        "x86_64" => build_x86(),
        "aarch64" => build_arm64(),
        "wasm32" => {} // pure Rust SIMD, no .S files to build
        other => {
            println!("cargo:warning=rust-commp-simd: no SIMD kernel for {other}, scalar only");
        }
    }
}

fn build_x86() {
    let outer_soa = "kernels/x86/parwise_reduce_x16_avx512_soa.S";
    let extrude_avx512 = "kernels/x86/extrude_avx512.S";
    let extrude_sse = "kernels/x86/extrude_quads_sse.S";
    let reduce_x8 = "kernels/x86/parwise_reduce_x8_avx2.S";

    for f in [outer_soa, reduce_x8] {
        if !std::path::Path::new(f).exists() {
            println!("cargo:warning=rust-commp-simd: {f} not found, scalar only");
            return;
        }
    }

    cc::Build::new()
        .file(outer_soa)
        .file(extrude_avx512)
        .file(extrude_sse)
        .file(reduce_x8)
        .flag("-mavx512f")
        .flag("-mavx512bw")
        .flag("-g")
        .flag("-x").flag("assembler-with-cpp")
        .compile("sha256_x16_avx512");

    println!("cargo:rerun-if-changed={outer_soa}");
    println!("cargo:rerun-if-changed={extrude_avx512}");
    println!("cargo:rerun-if-changed={extrude_sse}");
    println!("cargo:rerun-if-changed={reduce_x8}");
}

fn build_arm64() {
    // Always declare the cfg key so rustc doesn't warn
    println!("cargo:rustc-check-cfg=cfg(have_neon)");

    let kernel = "kernels/aarch64/sha256_x4_aarch64.S";
    if !std::path::Path::new(kernel).exists() {
        println!("cargo:warning=rust-commp-simd: {kernel} not found, scalar only");
        return;
    }
    cc::Build::new()
        .file(kernel)
        .flag("-march=armv8-a+crypto")
        .compile("sha256_x4_aarch64");
    println!("cargo:rerun-if-changed={kernel}");

    let neon = "kernels/aarch64/pairwise_reduce_x4_neon.S";
    if !std::path::Path::new(neon).exists() {
        println!("cargo:warning=pairwise_reduce_x4_neon {neon} not found, scalar only");
        return;
    }
    cc::Build::new()
        .file(neon)
        .flag("-march=armv8-a+crypto")
        .compile("pairwise_reduce_x4_neon");
    println!("cargo:rerun-if-changed={neon}");
    println!("cargo:rustc-cfg=have_neon");

    let neon = "kernels/aarch64/extrude_quads_neon.S";
    if !std::path::Path::new(neon).exists() {
        println!("cargo:warning=extrude_quads_neon_core {neon} not found, scalar only");
        return;
    }
    cc::Build::new()
        .file(neon)
        .flag("-march=armv8-a+crypto")
        .compile("extrude_quads_neon_core");

    cc::Build::new()
        .file(neon)
        .flag("-march=armv8-a+crypto")
        .compile("extrude_quads_2x_neon_core");

    println!("cargo:rerun-if-changed={neon}");
    println!("cargo:rustc-cfg=have_neon");


}