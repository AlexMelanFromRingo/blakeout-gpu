/// Advanced usage examples for Blakeout GPU
use blakeout_gpu::{BlakeoutGpu, BlakeoutBatchProcessor};
use std::time::Instant;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    println!("🚀 Blakeout GPU - Advanced Usage Examples\n");
    
    // Example 1: Optimal batch size selection
    example_optimal_batch_size().await?;
    
    // Example 2: Memory-efficient processing
    example_memory_efficient().await?;
    
    // Example 3: High-throughput pipeline
    example_high_throughput().await?;
    
    // Example 4: Error handling and recovery
    example_error_handling().await?;
    
    // Example 5: Real-world use case - password hashing
    example_password_hashing().await?;
    
    // Example 6: Monitoring and metrics
    example_monitoring().await?;
    
    Ok(())
}

/// Example 1: Finding optimal batch size for your GPU
async fn example_optimal_batch_size() -> Result<()> {
    println!("=== Example 1: Optimal Batch Size ===");
    
    let gpu = BlakeoutGpu::new()?;
    
    // Test different batch sizes
    let test_sizes = vec![64, 128, 256, 512, 1024, 2048];
    let mut best_size = 64;
    let mut best_throughput = 0.0;
    
    for size in test_sizes {
        let inputs: Vec<Vec<u8>> = (0..size)
            .map(|i| format!("test_{}", i).into_bytes())
            .collect();
        
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        
        let start = Instant::now();
        let _ = gpu.hash_batch(&refs)?;
        let duration = start.elapsed();
        
        let throughput = size as f64 / duration.as_secs_f64();
        
        println!("  Batch size {}: {:.2} hash/s", size, throughput);
        
        if throughput > best_throughput {
            best_throughput = throughput;
            best_size = size;
        }
    }
    
    println!("✅ Optimal batch size: {} ({:.2} hash/s)\n", best_size, best_throughput);
    
    Ok(())
}

/// Example 2: Memory-efficient processing for large datasets
async fn example_memory_efficient() -> Result<()> {
    println!("=== Example 2: Memory-Efficient Processing ===");
    
    let processor = BlakeoutBatchProcessor::new(1024)?;
    
    // Simulate processing 100K items efficiently
    let total_items = 100_000;
    let chunk_size = 1000;
    
    println!("Processing {} items in chunks of {}...", total_items, chunk_size);
    
    let start = Instant::now();
    let mut processed = 0;
    
    for chunk_id in 0..(total_items / chunk_size) {
        let inputs: Vec<Vec<u8>> = (0..chunk_size)
            .map(|i| {
                let idx = chunk_id * chunk_size + i;
                format!("item_{}", idx).into_bytes()
            })
            .collect();
        
        let _ = processor.process_large_batch(inputs).await?;
        processed += chunk_size;
        
        if chunk_id % 10 == 0 {
            let elapsed = start.elapsed();
            let rate = processed as f64 / elapsed.as_secs_f64();
            println!("  Progress: {}/{} ({:.2} hash/s)", processed, total_items, rate);
        }
    }
    
    let duration = start.elapsed();
    println!("✅ Processed {} items in {:?} ({:.2} hash/s)\n", 
             total_items, duration, total_items as f64 / duration.as_secs_f64());
    
    Ok(())
}

/// Example 3: High-throughput pipeline with streaming
async fn example_high_throughput() -> Result<()> {
    println!("=== Example 3: High-Throughput Pipeline ===");
    
    let processor = BlakeoutBatchProcessor::new(2048)?;
    
    // Create a channel for results
    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    
    // Spawn producer task
    let producer = tokio::spawn(async move {
        let inputs: Vec<Vec<u8>> = (0..10_000)
            .map(|i| format!("pipeline_{}", i).into_bytes())
            .collect();
        
        processor.stream_process(inputs, move |idx, hash| {
            // Send results through channel
            let _ = tx.blocking_send((idx, hash));
        }).await.unwrap();
    });
    
    // Consumer task - process results as they arrive
    let mut received = 0;
    let start = Instant::now();
    
    while let Some((idx, hash)) = rx.recv().await {
        received += 1;
        
        // Do something with the hash
        if idx % 1000 == 0 {
            println!("  Received hash {}: {}...", idx, hex_encode(&hash[..4]));
        }
        
        if received >= 10_000 {
            break;
        }
    }
    
    producer.await?;
    
    let duration = start.elapsed();
    println!("✅ Pipeline processed {} hashes in {:?} ({:.2} hash/s)\n", 
             received, duration, received as f64 / duration.as_secs_f64());
    
    Ok(())
}

/// Example 4: Proper error handling and recovery
async fn example_error_handling() -> Result<()> {
    println!("=== Example 4: Error Handling ===");
    
    // Try to create GPU hasher with recovery
    let gpu = match BlakeoutGpu::new() {
        Ok(g) => {
            println!("✅ GPU initialized successfully");
            g
        }
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            eprintln!("   Falling back to CPU implementation");
            return Ok(()); // In real app, would use CPU fallback
        }
    };
    
    // Test with potentially problematic inputs
    let test_cases = vec![
        (b"".as_slice(), "empty input"),
        (b"x", "single byte"),
        (&vec![0xFF; 1024 * 1024][..], "1MB of 0xFF"),
    ];
    
    for (input, description) in test_cases {
        match gpu.hash(input) {
            Ok(hash) => {
                println!("  ✅ {}: {}", description, hex_encode(&hash[..8]));
            }
            Err(e) => {
                eprintln!("  ❌ {} failed: {}", description, e);
            }
        }
    }
    
    println!();
    Ok(())
}

/// Example 5: Password hashing use case
async fn example_password_hashing() -> Result<()> {
    println!("=== Example 5: Password Hashing ===");
    
    let processor = BlakeoutBatchProcessor::new(512)?;
    
    // Simulate batch password verification
    struct PasswordCheck {
        username: String,
        password: String,
        expected_hash: [u8; 32],
    }
    
    let checks: Vec<PasswordCheck> = (0..100)
        .map(|i| {
            let username = format!("user_{}", i);
            let password = format!("password_{}", i);
            PasswordCheck {
                username: username.clone(),
                password: password.clone(),
                expected_hash: [0u8; 32], // Would be loaded from DB
            }
        })
        .collect();
    
    // Hash all passwords in parallel
    let inputs: Vec<Vec<u8>> = checks.iter()
        .map(|c| {
            let mut data = c.username.as_bytes().to_vec();
            data.extend_from_slice(c.password.as_bytes());
            data
        })
        .collect();
    
    let start = Instant::now();
    let hashes = processor.process_large_batch(inputs).await?;
    let duration = start.elapsed();
    
    // Verify results
    let mut valid = 0;
    for (check, hash) in checks.iter().zip(hashes.iter()) {
        // In real app, would compare with expected_hash
        valid += 1;
        if valid <= 3 {
            println!("  User '{}': {}", check.username, hex_encode(&hash[..8]));
        }
    }
    
    println!("✅ Verified {} passwords in {:?} ({:.2} verifications/s)\n",
             valid, duration, valid as f64 / duration.as_secs_f64());
    
    Ok(())
}

/// Example 6: Monitoring and metrics collection
async fn example_monitoring() -> Result<()> {
    println!("=== Example 6: Monitoring & Metrics ===");
    
    let processor = BlakeoutBatchProcessor::new(1024)?;
    
    struct Metrics {
        total_processed: usize,
        total_time: std::time::Duration,
        batch_times: Vec<std::time::Duration>,
    }
    
    let mut metrics = Metrics {
        total_processed: 0,
        total_time: std::time::Duration::ZERO,
        batch_times: Vec::new(),
    };
    
    // Process multiple batches and collect metrics
    for batch_id in 0..10 {
        let inputs: Vec<Vec<u8>> = (0..500)
            .map(|i| format!("metrics_{}_{}", batch_id, i).into_bytes())
            .collect();
        
        let start = Instant::now();
        let _ = processor.process_large_batch(inputs).await?;
        let duration = start.elapsed();
        
        metrics.total_processed += 500;
        metrics.total_time += duration;
        metrics.batch_times.push(duration);
    }
    
    // Calculate statistics
    let avg_time = metrics.total_time / metrics.batch_times.len() as u32;
    let avg_throughput = 500.0 / avg_time.as_secs_f64();
    
    let min_time = metrics.batch_times.iter().min().unwrap();
    let max_time = metrics.batch_times.iter().max().unwrap();
    
    println!("📊 Metrics Summary:");
    println!("  Total processed: {}", metrics.total_processed);
    println!("  Total time: {:?}", metrics.total_time);
    println!("  Average batch time: {:?}", avg_time);
    println!("  Min/Max batch time: {:?} / {:?}", min_time, max_time);
    println!("  Average throughput: {:.2} hash/s", avg_throughput);
    println!("  Overall throughput: {:.2} hash/s\n", 
             metrics.total_processed as f64 / metrics.total_time.as_secs_f64());
    
    Ok(())
}

// Utility function
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}