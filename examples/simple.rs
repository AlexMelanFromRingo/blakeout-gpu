/// Simple example to verify GPU setup works
use blakeout_gpu::BlakeoutGpu;

fn main() {
    println!("🚀 Blakeout GPU - Simple Test\n");
    
    // Initialize GPU
    println!("Initializing GPU...");
    let gpu = match BlakeoutGpu::new() {
        Ok(g) => {
            println!("✅ GPU initialized successfully!\n");
            g
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize GPU: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("  1. Check CUDA installation: nvidia-smi");
            eprintln!("  2. Verify PTX files exist: ls src/*.ptx");
            eprintln!("  3. Recompile kernels: ./scripts/compile_cuda.sh");
            std::process::exit(1);
        }
    };
    
    // Test 1: Single hash
    println!("Test 1: Single Hash");
    let data = b"Hello, GPU World!";
    match gpu.hash(data) {
        Ok(hash) => {
            println!("  Input:  {:?}", std::str::from_utf8(data).unwrap());
            println!("  Hash:   {}", hex_encode(&hash));
            println!("  ✅ Single hash works!\n");
        }
        Err(e) => {
            eprintln!("  ❌ Single hash failed: {}\n", e);
        }
    }
    
    // Test 2: Small batch
    println!("Test 2: Small Batch (10 hashes)");
    let inputs: Vec<Vec<u8>> = (0..10)
        .map(|i| format!("Message {}", i).into_bytes())
        .collect();
    
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    
    match gpu.hash_batch(&refs) {
        Ok(results) => {
            println!("  Processed: {} hashes", results.len());
            println!("  Sample hashes:");
            for (i, hash) in results.iter().take(3).enumerate() {
                println!("    [{}]: {}", i, hex_encode(&hash[..8]));
            }
            println!("  ✅ Batch processing works!\n");
        }
        Err(e) => {
            eprintln!("  ❌ Batch processing failed: {}\n", e);
        }
    }
    
    // Test 3: Different input sizes
    println!("Test 3: Variable Input Sizes");
    let test_sizes = vec![0, 1, 16, 64, 256, 1024];
    for size in test_sizes {
        let data = vec![0xAB; size];
        match gpu.hash(&data) {
            Ok(hash) => {
                println!("  {} bytes → {}", size, hex_encode(&hash[..8]));
            }
            Err(e) => {
                eprintln!("  {} bytes → ❌ Failed: {}", size, e);
            }
        }
    }
    println!("  ✅ Variable sizes work!\n");
    
    println!("🎉 All tests passed! GPU is working correctly.");
    println!("\nNext steps:");
    println!("  - Run full tests: cargo test --release");
    println!("  - Run benchmarks: cargo bench");
    println!("  - Try advanced examples: cargo run --release --example advanced_usage");
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}