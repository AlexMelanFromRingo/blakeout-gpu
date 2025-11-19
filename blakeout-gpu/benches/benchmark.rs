use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use blakeout::Blakeout;
use blakeout_gpu::BlakeoutGpu;

fn bench_cpu_single_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU Single Hash");
    let data = b"ALFIS Block Data for Benchmarking";

    group.bench_function("blakeout_cpu", |b| {
        b.iter(|| {
            let mut hasher = Blakeout::new();
            hasher.update(black_box(data));
            black_box(hasher.result());
        });
    });

    group.finish();
}

fn bench_gpu_batch(c: &mut Criterion) {
    if !blakeout_gpu::gpu::is_cuda_available() {
        println!("Skipping GPU benchmarks: no CUDA device available");
        return;
    }

    let mut group = c.benchmark_group("GPU Batch Hashing");
    let data = b"ALFIS Block Data for Benchmarking";

    for batch_size in [1024, 2048, 4096, 8192].iter() {
        let gpu_miner = BlakeoutGpu::new(*batch_size).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gpu_batch", batch_size),
            batch_size,
            |b, &_size| {
                b.iter(|| {
                    let results = gpu_miner.hash_batch(black_box(data), 0, 0).unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    if !blakeout_gpu::gpu::is_cuda_available() {
        return;
    }

    let mut group = c.benchmark_group("CPU vs GPU Comparison");
    let data = b"ALFIS Block Data";
    let batch_size = 4096;

    // CPU: Process batch_size hashes sequentially
    group.bench_function("cpu_batch_4096", |b| {
        b.iter(|| {
            for i in 0..batch_size {
                let mut hasher = Blakeout::new();
                hasher.update(black_box(data));
                hasher.update(&i.to_le_bytes());
                black_box(hasher.result());
            }
        });
    });

    // GPU: Process batch_size hashes in parallel
    let gpu_miner = BlakeoutGpu::new(batch_size).unwrap();
    group.bench_function("gpu_batch_4096", |b| {
        b.iter(|| {
            let results = gpu_miner.hash_batch(black_box(data), 0, 0).unwrap();
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_single_hash,
    bench_gpu_batch,
    bench_comparison
);
criterion_main!(benches);
