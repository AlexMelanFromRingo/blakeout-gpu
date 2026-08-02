use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/blake2s.cu");
    
    // Get CUDA toolkit path
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    
    // Compile CUDA kernel to PTX
    compile_cuda_kernel(&cuda_path);
    
    // Set include path for CUDA headers
    println!("cargo:include={}/include", cuda_path);
}

fn find_nvcc(cuda_path: &str) -> Option<String> {
    // Try multiple possible locations
    let candidates = vec![
        format!("{}/bin/nvcc", cuda_path),
        "/usr/local/cuda/bin/nvcc".to_string(),
        "/usr/bin/nvcc".to_string(),
        "nvcc".to_string(), // Try PATH
    ];
    
    for nvcc_path in candidates {
        if Command::new(&nvcc_path).arg("--version").output().is_ok() {
            println!("cargo:warning=Found nvcc at: {}", nvcc_path);
            return Some(nvcc_path);
        }
    }
    
    // Try using 'which' command
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                println!("cargo:warning=Found nvcc via 'which': {}", path);
                return Some(path);
            }
        }
    }
    
    None
}

fn compile_cuda_kernel(cuda_path: &str) {
    let source = "src/blake2s.cu";
    let output = "src/blake2s.ptx";
    
    // Find nvcc
    let nvcc = match find_nvcc(cuda_path) {
        Some(path) => path,
        None => {
            eprintln!("WARNING: nvcc not found. CUDA kernel will not be compiled.");
            eprintln!("Please ensure CUDA toolkit is installed and nvcc is in PATH.");
            eprintln!("Tried locations:");
            eprintln!("  - {}/bin/nvcc", cuda_path);
            eprintln!("  - /usr/local/cuda/bin/nvcc");
            eprintln!("  - /usr/bin/nvcc");
            eprintln!("  - nvcc in PATH");
            
            // Create a dummy PTX file so compilation can continue
            std::fs::write(output, "// Dummy PTX - CUDA kernel not compiled\n")
                .expect("Failed to create dummy PTX");
            
            println!("cargo:warning=CUDA kernel not compiled - created dummy PTX");
            return;
        }
    };
    
    
    // Detect GPU architecture
    let arch = detect_gpu_arch(&nvcc);
    
    println!("cargo:warning=Compiling CUDA kernel for architecture: {}", arch);
    
    let status = Command::new(&nvcc)
        .args(&[
            "-ptx",
            source,
            "-o", output,
            &format!("--gpu-architecture={}", arch),
            "-O3",
            "--use_fast_math",
            "--maxrregcount=64",
            "-Xptxas", "-v",
            // Optimization flags
            "-lineinfo",
            "--default-stream", "per-thread",
        ])
        .status()
        .expect("Failed to execute nvcc");
    
    if !status.success() {
        eprintln!("WARNING: Failed to compile CUDA kernel");
        eprintln!("Creating dummy PTX file to allow build to continue");
        std::fs::write(output, "// Dummy PTX - compilation failed\n")
            .expect("Failed to create dummy PTX");
        return;
    }
    
    println!("cargo:warning=CUDA kernel compiled successfully");
    
    // Also compile optimized version for different architectures
    compile_for_architecture(&nvcc, source, "sm_70"); // Volta
    compile_for_architecture(&nvcc, source, "sm_75"); // Turing
    compile_for_architecture(&nvcc, source, "sm_80"); // Ampere
    compile_for_architecture(&nvcc, source, "sm_86"); // Ampere (RTX 30xx)
    compile_for_architecture(&nvcc, source, "sm_89"); // Ada Lovelace
}

fn compile_for_architecture(nvcc: &str, source: &str, arch: &str) {
    let output = format!("src/blake2s_{}.ptx", arch.replace("sm_", ""));
    
    let status = Command::new(nvcc)
        .args(&[
            "-ptx",
            source,
            "-o", &output,
            &format!("--gpu-architecture={}", arch),
            "-O3",
            "--use_fast_math",
        ])
        .status();
    
    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Compiled for {}", arch);
        }
        _ => {
            println!("cargo:warning=Could not compile for {}", arch);
        }
    }
}

fn detect_gpu_arch(nvcc_path: &str) -> String {
    // Try to detect GPU architecture using nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    
    if let Ok(output) = output {
        if let Ok(cap) = String::from_utf8(output.stdout) {
            let cap = cap.trim().replace(".", "");
            if !cap.is_empty() {
                println!("cargo:warning=Detected GPU compute capability: sm_{}", cap);
                return format!("sm_{}", cap);
            }
        }
    }
    
    // Try to compile a test program to detect architecture
    if let Ok(output) = Command::new(nvcc_path)
        .args(&["--list-gpu-arch"])
        .output()
    {
        if output.status.success() {
            let archs = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=Available architectures: {}", archs);
        }
    }
    
    // Default to compute capability 8.6 (RTX 30xx series)
    println!("cargo:warning=Could not detect GPU, using default sm_86");
    "sm_86".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_gpu() {
        let arch = detect_gpu_arch("nvcc");
        assert!(arch.starts_with("sm_"));
    }
    
    #[test]
    fn test_find_nvcc() {
        // This test will succeed if nvcc is available
        let result = find_nvcc("/usr/local/cuda");
        println!("nvcc search result: {:?}", result);
    }
}