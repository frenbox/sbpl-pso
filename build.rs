#[cfg(feature = "cuda")]
fn build_cuda() {
    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_80,code=sm_80")    // Ampere (A100)
        .flag("-gencode=arch=compute_86,code=sm_86")    // Ampere (A40, RTX 3090)
        .flag("-gencode=arch=compute_89,code=sm_89")    // Ada (RTX 4090)
        .flag("-gencode=arch=compute_120,code=sm_120")  // Blackwell (RTX 5080)
        .flag("-Wno-deprecated-gpu-targets")
        .file("cuda/sbpl.cu")
        .compile("sbpl_pso_cuda");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=cuda/sbpl.cu");
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();

    // Metal: nothing to compile at build time — the .metal source is embedded
    // in the binary via `include_str!` and compiled at runtime by
    // `MTLDevice::newLibraryWithSource`. Just re-trigger a rebuild when the
    // shader source changes so `include_str!` picks it up.
    #[cfg(feature = "metal")]
    {
        println!("cargo:rerun-if-changed=metal/sbpl.metal");
    }
}
