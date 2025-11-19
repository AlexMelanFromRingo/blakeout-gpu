use std::env;

fn main() {
    // Declare the no_cuda cfg option
    println!("cargo::rustc-check-cfg=cfg(no_cuda)");

    // Check if nvcc is available
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

    if !nvcc_available {
        println!("cargo:warning=CUDA compiler (nvcc) not found. GPU support will be disabled.");
        println!("cargo:warning=To enable GPU support, install CUDA Toolkit and ensure nvcc is in PATH.");
        println!("cargo:rustc-cfg=no_cuda");
        return;
    }

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

    // Determine compute capability from environment or use default
    // RTX 40xx series uses sm_89, RTX 30xx uses sm_86, older cards use sm_60+
    let compute_arch = env::var("CUDA_COMPUTE_ARCH").unwrap_or_else(|_| {
        // Default to sm_86 which works for RTX 30xx and 40xx series
        "sm_86".to_string()
    });

    println!("cargo:warning=Compiling CUDA kernels with architecture {}", compute_arch);

    // Step 1: Compile blake2s.cu with device-side compilation
    let status = std::process::Command::new("nvcc")
        .args(&[
            "-dc",  // Device-side compilation (allows device functions to be called from other files)
            "cuda/blake2s.cu",
            "-o",
            &format!("{}/blake2s.o", out_dir),
            "-I",
            &cuda_include,
            "--compiler-options",
            "-fPIC",
            "-arch", &compute_arch,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {},
        Ok(s) => {
            eprintln!("Failed to compile blake2s.cu: exit code {:?}", s.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to run nvcc for blake2s.cu: {}", e);
            std::process::exit(1);
        }
    }

    // Step 2: Compile blakeout.cu with device-side compilation
    let status = std::process::Command::new("nvcc")
        .args(&[
            "-dc",  // Device-side compilation
            "cuda/blakeout.cu",
            "-o",
            &format!("{}/blakeout.o", out_dir),
            "-I",
            &cuda_include,
            "--compiler-options",
            "-fPIC",
            "-arch", &compute_arch,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {},
        Ok(s) => {
            eprintln!("Failed to compile blakeout.cu: exit code {:?}", s.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to run nvcc for blakeout.cu: {}", e);
            std::process::exit(1);
        }
    }

    // Step 3: Device-side linking
    let status = std::process::Command::new("nvcc")
        .args(&[
            "-dlink",  // Device-side linking
            &format!("{}/blake2s.o", out_dir),
            &format!("{}/blakeout.o", out_dir),
            "-o",
            &format!("{}/device_link.o", out_dir),
            "--compiler-options",
            "-fPIC",
            "-arch", &compute_arch,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {},
        Ok(s) => {
            eprintln!("Failed to device-link CUDA objects: exit code {:?}", s.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to run nvcc for device linking: {}", e);
            std::process::exit(1);
        }
    }

    // Step 4: Create shared library with all objects
    let status = std::process::Command::new("nvcc")
        .args(&[
            "-shared",
            &format!("{}/blake2s.o", out_dir),
            &format!("{}/blakeout.o", out_dir),
            &format!("{}/device_link.o", out_dir),
            "-o",
            &format!("{}/libblakeout_cuda.so", out_dir),
            "-arch", &compute_arch,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Successfully compiled CUDA kernels");
        },
        Ok(s) => {
            eprintln!("Failed to link CUDA objects: exit code {:?}", s.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed to run nvcc for linking: {}", e);
            std::process::exit(1);
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=blakeout_cuda");
}
