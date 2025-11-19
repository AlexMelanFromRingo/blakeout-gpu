use blakeout_gpu::{BlakeoutGpu, BlakeoutGpuError};
use std::time::Instant;

fn main() -> Result<(), BlakeoutGpuError> {
    println!("Blakeout GPU Performance Test");
    println!("=============================\n");

    // Test with different batch sizes
    for batch_size in [256, 512, 1024, 2048, 4096] {
        println!("Testing batch_size = {}", batch_size);

        let gpu_miner = match BlakeoutGpu::new(batch_size) {
            Ok(miner) => miner,
            Err(BlakeoutGpuError::NoGpuAvailable) => {
                println!("No GPU available, skipping...\n");
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        let input_data = b"ALFIS Block Data - Performance Test";

        // Warmup run
        let _ = gpu_miner.hash_batch(input_data, 0, 0)?;

        // Timed runs
        let num_runs = 10;
        let start = Instant::now();

        for i in 0..num_runs {
            let start_nonce = i * batch_size as u64;
            let _ = gpu_miner.hash_batch(input_data, start_nonce, 0)?;
        }

        let elapsed = start.elapsed();
        let total_hashes = batch_size as u64 * num_runs;
        let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();

        println!("  Total hashes: {}", total_hashes);
        println!("  Time: {:.3}s", elapsed.as_secs_f64());
        println!("  Hash rate: {:.2} H/s ({:.6} MH/s)", hash_rate, hash_rate / 1_000_000.0);
        println!("  Time per hash: {:.3}ms\n", elapsed.as_secs_f64() * 1000.0 / total_hashes as f64);
    }

    Ok(())
}
