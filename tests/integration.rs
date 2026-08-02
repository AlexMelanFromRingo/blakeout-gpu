use blakeout_gpu::{BlakeoutGpu, BlakeoutBatchProcessor};
use std::time::Instant;

#[test]
fn test_gpu_initialization() {
    let result = BlakeoutGpu::new();
    assert!(result.is_ok(), "GPU initialization failed: {:?}", result.err());
}

#[test]
fn test_single_hash_deterministic() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let data = b"test data for deterministic check";
    let hash1 = gpu.hash(data).expect("First hash failed");
    let hash2 = gpu.hash(data).expect("Second hash failed");
    
    assert_eq!(hash1, hash2, "Hashes are not deterministic");
}

#[test]
fn test_different_inputs_different_outputs() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let hash1 = gpu.hash(b"input1").expect("Hash 1 failed");
    let hash2 = gpu.hash(b"input2").expect("Hash 2 failed");
    
    assert_ne!(hash1, hash2, "Different inputs produced same hash");
}

#[test]
fn test_batch_consistency() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let inputs = vec![
        b"test1".as_slice(),
        b"test2".as_slice(),
        b"test3".as_slice(),
    ];
    
    // Hash individually
    let individual: Vec<[u8; 32]> = inputs.iter()
        .map(|&input| gpu.hash(input).expect("Individual hash failed"))
        .collect();
    
    // Hash as batch
    let batch = gpu.hash_batch(&inputs).expect("Batch hash failed");
    
    // Compare
    assert_eq!(individual.len(), batch.len());
    for (i, (ind, bat)) in individual.iter().zip(batch.iter()).enumerate() {
        assert_eq!(ind, bat, "Hash {} differs between individual and batch", i);
    }
}

#[test]
fn test_empty_input() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    let hash = gpu.hash(b"").expect("Empty input hash failed");
    
    // Empty input should produce a valid 32-byte hash
    assert_eq!(hash.len(), 32);
    
    // Should be deterministic
    let hash2 = gpu.hash(b"").expect("Second empty hash failed");
    assert_eq!(hash, hash2);
}

#[test]
fn test_large_input() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    // Test with 1MB input
    let large_input = vec![42u8; 1024 * 1024];
    let hash = gpu.hash(&large_input).expect("Large input hash failed");
    
    assert_eq!(hash.len(), 32);
}

#[test]
fn test_batch_sizes() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    for size in [1, 10, 50, 100] {
        let inputs: Vec<Vec<u8>> = (0..size)
            .map(|i| format!("batch_test_{}", i).into_bytes())
            .collect();
        
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let results = gpu.hash_batch(&refs)
            .expect(&format!("Batch of size {} failed", size));
        
        assert_eq!(results.len(), size, "Wrong number of results for batch size {}", size);
    }
}

#[tokio::test]
async fn test_batch_processor() {
    let processor = BlakeoutBatchProcessor::new(128)
        .expect("Processor init failed");
    
    let inputs: Vec<Vec<u8>> = (0..500)
        .map(|i| format!("processor_test_{}", i).into_bytes())
        .collect();
    
    let start = Instant::now();
    let results = processor.process_large_batch(inputs.clone()).await
        .expect("Batch processing failed");
    let duration = start.elapsed();
    
    assert_eq!(results.len(), 500);
    println!("Processed 500 hashes in {:?} ({:.2} hash/s)", 
             duration, 500.0 / duration.as_secs_f64());
}

#[tokio::test]
async fn test_streaming_processor() {
    let processor = BlakeoutBatchProcessor::new(128)
        .expect("Processor init failed");
    
    let inputs: Vec<Vec<u8>> = (0..200)
        .map(|i| format!("stream_test_{}", i).into_bytes())
        .collect();
    
    let mut count = 0;
    let mut hashes = Vec::new();
    
    processor.stream_process(inputs.clone(), |idx, hash| {
        count += 1;
        hashes.push((idx, hash));
    }).await.expect("Stream processing failed");
    
    assert_eq!(count, 200);
    assert_eq!(hashes.len(), 200);
    
    // Check ordering
    for (i, (idx, _)) in hashes.iter().enumerate() {
        assert_eq!(*idx, i, "Hash order incorrect");
    }
}

#[test]
fn test_hash_distribution() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    // Generate many hashes and check for collisions
    let mut hashes = std::collections::HashSet::new();
    
    for i in 0..1000 {
        let input = format!("distribution_test_{}", i).into_bytes();
        let hash = gpu.hash(&input).expect("Hash failed");
        
        let hash_inserted = hashes.insert(hash);
        assert!(hash_inserted, "Hash collision detected at iteration {}", i);
    }
    
    assert_eq!(hashes.len(), 1000);
}

#[test]
fn test_different_lengths() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let lengths = vec![0, 1, 16, 32, 63, 64, 65, 127, 128, 256, 1024];
    
    for len in lengths {
        let input = vec![0xABu8; len];
        let hash = gpu.hash(&input)
            .expect(&format!("Hash failed for length {}", len));
        assert_eq!(hash.len(), 32);
    }
}

#[test] 
fn test_batch_memory_efficiency() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    // Test that we can process batches without running out of memory
    for _ in 0..10 {
        let inputs: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8; 256])
            .collect();
        
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let results = gpu.hash_batch(&refs).expect("Batch failed");
        
        assert_eq!(results.len(), 100);
    }
}

#[cfg(feature = "cpu_comparison")]
#[test]
fn test_gpu_vs_cpu_correctness() {
    use blakeout::Blakeout;
    use digest::Digest;
    
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let test_inputs = vec![
        b"".as_slice(),
        b"a".as_slice(),
        b"abc".as_slice(),
        b"message digest".as_slice(),
        b"abcdefghijklmnopqrstuvwxyz".as_slice(),
    ];
    
    for input in test_inputs {
        // CPU version
        let mut cpu_hasher = Blakeout::default();
        cpu_hasher.update(input);
        let cpu_result = cpu_hasher.result();
        
        // GPU version
        let gpu_result = gpu.hash(input).expect("GPU hash failed");
        
        assert_eq!(
            cpu_result, 
            &gpu_result[..],
            "GPU and CPU results differ for input: {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );
    }
}

#[test]
fn test_performance_baseline() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let inputs: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("perf_test_{}", i).into_bytes())
        .collect();
    
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    
    let start = Instant::now();
    let results = gpu.hash_batch(&refs).expect("Batch failed");
    let duration = start.elapsed();
    
    let throughput = 100.0 / duration.as_secs_f64();
    
    assert_eq!(results.len(), 100);
    println!("Throughput: {:.2} hashes/sec", throughput);
    
    // Baseline: should be faster than 100 hash/s for batch of 100
    assert!(throughput > 100.0, "Throughput too low: {:.2}", throughput);
}

#[test]
fn test_avalanche_effect() {
    let gpu = BlakeoutGpu::new().expect("GPU init failed");
    
    let input1 = b"test";
    let input2 = b"Test"; // Only one bit different
    
    let hash1 = gpu.hash(input1).expect("Hash 1 failed");
    let hash2 = gpu.hash(input2).expect("Hash 2 failed");
    
    // Count differing bits
    let mut diff_bits = 0;
    for (a, b) in hash1.iter().zip(hash2.iter()) {
        diff_bits += (a ^ b).count_ones();
    }
    
    // Avalanche effect: ~50% of bits should differ
    let total_bits = 256;
    let diff_ratio = diff_bits as f64 / total_bits as f64;
    
    println!("Avalanche: {}/{} bits differ ({:.1}%)", 
             diff_bits, total_bits, diff_ratio * 100.0);
    
    // Should have significant bit changes (at least 30% different)
    assert!(diff_ratio > 0.3, "Insufficient avalanche effect: {:.1}%", diff_ratio * 100.0);
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let gpu = Arc::new(BlakeoutGpu::new().expect("GPU init failed"));
    
    let mut handles = vec![];
    
    for i in 0..4 {
        let gpu_clone = gpu.clone();
        let handle = thread::spawn(move || {
            let data = format!("thread_{}", i).into_bytes();
            gpu_clone.hash(&data).expect("Hash failed")
        });
        handles.push(handle);
    }
    
    let results: Vec<[u8; 32]> = handles.into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();
    
    assert_eq!(results.len(), 4);
    
    // All results should be different
    for i in 0..results.len() {
        for j in (i+1)..results.len() {
            assert_ne!(results[i], results[j], "Thread {} and {} produced same hash", i, j);
        }
    }
}