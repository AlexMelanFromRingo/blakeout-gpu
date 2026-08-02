use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use blakeout_gpu::{BlakeoutGpu, BlakeoutBatchProcessor};

fn bench_single_hash(c: &mut Criterion) {
    let gpu = BlakeoutGpu::new().expect("Failed to init GPU");
    
    c.bench_function("blakeout_gpu_single", |b| {
        let data = b"benchmark data for single hash test";
        b.iter(|| {
            let result = gpu.hash(black_box(data)).unwrap();
            black_box(result);
        });
    });
}

fn bench_batch_sizes(c: &mut Criterion) {
    let gpu = BlakeoutGpu::new().expect("Failed to init GPU");
    let mut group = c.benchmark_group("blakeout_gpu_batch");
    
    for size in [10, 50, 100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        let inputs: Vec<Vec<u8>> = (0..*size)
            .map(|i| format!("benchmark message {}", i).into_bytes())
            .collect();
        
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &input_refs, |b, refs| {
            b.iter(|| {
                let results = gpu.hash_batch(black_box(refs)).unwrap();
                black_box(results);
            });
        });
    }
    
    group.finish();
}

fn bench_input_sizes(c: &mut Criterion) {
    let gpu = BlakeoutGpu::new().expect("Failed to init GPU");
    let mut group = c.benchmark_group("blakeout_gpu_input_size");
    
    // Test different input data sizes
    for input_size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Bytes(*input_size as u64));
        
        let data = vec![0u8; *input_size];
        
        group.bench_with_input(
            BenchmarkId::from_parameter(input_size), 
            &data, 
            |b, d| {
                b.iter(|| {
                    let result = gpu.hash(black_box(d)).unwrap();
                    black_box(result);
                });
            }
        );
    }
    
    group.finish();
}

fn bench_batch_processor(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let processor = BlakeoutBatchProcessor::new(1024).expect("Failed to init processor");
    
    let mut group = c.benchmark_group("blakeout_batch_processor");
    
    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        let inputs: Vec<Vec<u8>> = (0..*size)
            .map(|i| format!("processor test {}", i).into_bytes())
            .collect();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &inputs, |b, inp| {
            b.to_async(&rt).iter(|| async {
                let results = processor.process_large_batch(black_box(inp.clone())).await.unwrap();
                black_box(results);
            });
        });
    }
    
    group.finish();
}

fn bench_memory_patterns(c: &mut Criterion) {
    let gpu = BlakeoutGpu::new().expect("Failed to init GPU");
    let mut group = c.benchmark_group("blakeout_memory_patterns");
    
    // Sequential access pattern
    group.bench_function("sequential_100", |b| {
        let inputs: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8; 256])
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        
        b.iter(|| {
            let results = gpu.hash_batch(black_box(&refs)).unwrap();
            black_box(results);
        });
    });
    
    // Random-like access pattern
    group.bench_function("random_100", |b| {
        let inputs: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let mut data = vec![0u8; 256];
                for (j, byte) in data.iter_mut().enumerate() {
                    *byte = ((i * 7 + j * 13) % 256) as u8;
                }
                data
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        
        b.iter(|| {
            let results = gpu.hash_batch(black_box(&refs)).unwrap();
            black_box(results);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_sizes,
    bench_input_sizes,
    bench_batch_processor,
    bench_memory_patterns
);

criterion_main!(benches);