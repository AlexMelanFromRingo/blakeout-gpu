use blakeout_gpu::{BlakeoutGpu, BlakeoutBatchProcessor};
use std::time::Instant;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Blakeout GPU Batch Processing Demo\n");
    
    // Example 1: Single hash
    println!("=== Example 1: Single Hash ===");
    let gpu = BlakeoutGpu::new()?;
    let data = b"Hello, GPU World!";
    
    let start = Instant::now();
    let hash = gpu.hash(data)?;
    let duration = start.elapsed();
    
    println!("Input: {:?}", std::str::from_utf8(data).unwrap());
    println!("Hash:  {}", hex::encode(hash));
    println!("Time:  {:?}\n", duration);
    
    // Example 2: Small batch
    println!("=== Example 2: Small Batch (100 hashes) ===");
    let inputs: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("Message number {}", i).into_bytes())
        .collect();
    
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    
    let start = Instant::now();
    let results = gpu.hash_batch(&input_refs)?;
    let duration = start.elapsed();
    
    println!("Processed: {} hashes", results.len());
    println!("Time:      {:?}", duration);
    println!("Rate:      {:.2} hashes/sec\n", 100.0 / duration.as_secs_f64());
    
    // Example 3: Large batch with processor
    println!("=== Example 3: Large Batch (10,000 hashes) ===");
    let processor = BlakeoutBatchProcessor::new(1024)?;
    
    let large_inputs: Vec<Vec<u8>> = (0..10_000)
        .map(|i| {
            let data = format!("Large batch message {}", i);
            data.repeat(10).into_bytes() // ~200 bytes each
        })
        .collect();
    
    let start = Instant::now();
    let large_results = processor.process_large_batch(large_inputs).await?;
    let duration = start.elapsed();
    
    println!("Processed: {} hashes", large_results.len());
    println!("Time:      {:?}", duration);
    println!("Rate:      {:.2} hashes/sec", 10_000.0 / duration.as_secs_f64());
    println!("Memory:    ~{} MB used", (large_results.len() * 2) / 1024 / 1024);
    
    // Example 4: Streaming processing
    println!("\n=== Example 4: Streaming Processing ===");
    let stream_inputs: Vec<Vec<u8>> = (0..5_000)
        .map(|i| format!("Stream message {}", i).into_bytes())
        .collect();
    
    let mut processed_count = 0;
    let start = Instant::now();
    
    processor.stream_process(stream_inputs, move |idx, hash| {
        processed_count += 1;
        if idx % 1000 == 0 {
            println!("  Processed {} hashes... (latest: {})", 
                     idx, hex::encode(&hash[..8]));
        }
    }).await?;
    
    let duration = start.elapsed();
    println!("Total processed: {}", processed_count);
    println!("Time:            {:?}", duration);
    
    // Example 5: Performance comparison
    println!("\n=== Example 5: CPU vs GPU Comparison ===");
    benchmark_comparison(&gpu).await?;
    
    Ok(())
}

async fn benchmark_comparison(gpu: &BlakeoutGpu) -> Result<()> {
    use blake2::{Blake2s256, Digest};
    
    let test_sizes = vec![10, 100, 1_000, 10_000];
    
    for size in test_sizes {
        println!("\nBatch size: {}", size);
        
        // Generate test data
        let inputs: Vec<Vec<u8>> = (0..size)
            .map(|i| format!("Benchmark message {}", i).into_bytes())
            .collect();
        
        // CPU baseline (simplified - not full Blakeout)
        let start = Instant::now();
        for input in &inputs {
            let mut hasher = Blake2s256::new();
            hasher.update(input);
            let _ = hasher.finalize();
        }
        let cpu_time = start.elapsed();
        
        // GPU batch
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let start = Instant::now();
        let _ = gpu.hash_batch(&input_refs)?;
        let gpu_time = start.elapsed();
        
        println!("  CPU time: {:?} ({:.2} hashes/sec)", 
                 cpu_time, size as f64 / cpu_time.as_secs_f64());
        println!("  GPU time: {:?} ({:.2} hashes/sec)",
                 gpu_time, size as f64 / gpu_time.as_secs_f64());
        println!("  Speedup:  {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }
    
    Ok(())
}

// Utility for hexadecimal encoding
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}