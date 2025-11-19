use blakeout_gpu::{BlakeoutGpu, BlakeoutGpuError};
use std::time::Instant;

fn main() -> Result<(), BlakeoutGpuError> {
    println!("Blakeout GPU Miner Example");
    println!("==========================\n");

    // Initialize GPU miner with batch size of 256
    // (smaller batch = less memory, faster allocation)
    let batch_size = 256;
    let gpu_miner = match BlakeoutGpu::new(batch_size) {
        Ok(miner) => {
            println!("✓ GPU initialized successfully");
            println!("  Batch size: {}\n", batch_size);
            miner
        }
        Err(BlakeoutGpuError::NoGpuAvailable) => {
            println!("✗ No CUDA GPU available. Please ensure:");
            println!("  - NVIDIA GPU is present");
            println!("  - CUDA drivers are installed");
            println!("  - nvidia-smi command works");
            return Err(BlakeoutGpuError::NoGpuAvailable);
        }
        Err(e) => return Err(e),
    };

    // Test data
    let input_data = b"ALFIS Block Data - Testing GPU Mining";
    let target_difficulty = 10; // Looking for hash with difficulty >= 10

    println!("Mining parameters:");
    println!("  Input: {:?}", String::from_utf8_lossy(input_data));
    println!("  Target difficulty: {}", target_difficulty);
    println!("  Batch size: {}\n", batch_size);

    println!("Starting GPU mining...\n");

    let start_time = Instant::now();
    let mut total_hashes = 0u64;
    let mut found = false;

    for batch_num in 0..100 {
        let start_nonce = batch_num * batch_size as u64;

        match gpu_miner.find_hash(input_data, start_nonce, target_difficulty) {
            Ok(Some(result)) => {
                let elapsed = start_time.elapsed();
                total_hashes += (result.nonce - batch_num * batch_size as u64) + 1;

                println!("✓ Found matching hash!");
                println!("  Nonce: {}", result.nonce);
                println!("  Difficulty: {}", result.difficulty);
                println!("  Hash: {}", hex_encode(&result.hash));
                println!("\nPerformance:");
                println!("  Total hashes: {}", total_hashes);
                println!("  Time elapsed: {:.2}s", elapsed.as_secs_f64());
                println!(
                    "  Hash rate: {:.2} MH/s",
                    total_hashes as f64 / elapsed.as_secs_f64() / 1_000_000.0
                );

                found = true;
                break;
            }
            Ok(None) => {
                total_hashes += batch_size as u64;
                if batch_num % 10 == 0 {
                    let elapsed = start_time.elapsed();
                    println!(
                        "Batch {}: {} hashes, {:.2} MH/s, max difficulty in this batch: checking...",
                        batch_num,
                        total_hashes,
                        total_hashes as f64 / elapsed.as_secs_f64() / 1_000_000.0
                    );
                }
            }
            Err(e) => {
                println!("✗ Error during mining: {}", e);
                return Err(e);
            }
        }
    }

    if !found {
        println!("\n✗ No hash found with difficulty >= {} in {} batches", target_difficulty, 100);
        println!("  Total hashes tried: {}", total_hashes);
    }

    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
