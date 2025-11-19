use std::env;
use std::path::PathBuf;

fn main() {
    // Check if CUDA is available
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib = format!("{}/lib64", cuda_path);

    println!("cargo:rerun-if-changed=cuda/blake2s.cu");
    println!("cargo:rerun-if-changed=cuda/blakeout.cu");
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");

    // Compile CUDA code using nvcc
    let out_dir = env::var("OUT_DIR").unwrap();

    // Compile blake2s.cu
    std::process::Command::new("nvcc")
        .args(&[
            "-c",
            "cuda/blake2s.cu",
            "-o",
            &format!("{}/blake2s.o", out_dir),
            "-I",
            &cuda_include,
            "--compiler-options",
            "-fPIC",
            "-arch=sm_60", // Compute capability 6.0+, adjust as needed
        ])
        .status()
        .expect("Failed to compile blake2s.cu");

    // Compile blakeout.cu and link with blake2s.o
    std::process::Command::new("nvcc")
        .args(&[
            "-c",
            "cuda/blakeout.cu",
            "-o",
            &format!("{}/blakeout.o", out_dir),
            "-I",
            &cuda_include,
            "--compiler-options",
            "-fPIC",
            "-arch=sm_60",
        ])
        .status()
        .expect("Failed to compile blakeout.cu");

    // Link into shared library
    std::process::Command::new("nvcc")
        .args(&[
            "-shared",
            &format!("{}/blake2s.o", out_dir),
            &format!("{}/blakeout.o", out_dir),
            "-o",
            &format!("{}/libblakeout_cuda.so", out_dir),
            "-arch=sm_60",
        ])
        .status()
        .expect("Failed to link CUDA objects");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=blakeout_cuda");
}
