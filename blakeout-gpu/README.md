# Blakeout-GPU

GPU-accelerated implementation of the Blakeout hash algorithm using CUDA for NVIDIA GPUs.

## Overview

Blakeout is a memory-hard hashing algorithm based on Blake2s, designed to be resistant to ASIC mining. This GPU implementation provides massive parallelization for mining operations, computing thousands of hashes simultaneously.

## Features

- **Parallel Hash Computation**: Process 4K-16K hashes per batch on GPU
- **Automatic Fallback**: Falls back to CPU if GPU is unavailable
- **CUDA-Optimized**: Blake2s implementation optimized for CUDA architecture
- **Easy Integration**: Simple Rust API compatible with existing Blakeout code
- **ALFIS Integration**: Direct integration with ALFIS blockchain miner

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 6.0+ (Pascal architecture or newer)
- Recommended: RTX 2060 or better for optimal performance

### Software
- CUDA Toolkit 11.0 or later
- NVIDIA drivers (latest recommended)
- Rust 1.70+
- nvcc compiler (included with CUDA Toolkit)

## Installation

### 1. Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-0
```

**Set environment variables:**
```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### 2. Build Blakeout-GPU

```bash
cd blakeout-gpu
cargo build --release
```

### 3. Run Example

```bash
cargo run --release --example gpu_miner
```

## Usage

### Basic Usage

```rust
use blakeout_gpu::{BlakeoutGpu, BlakeoutGpuError};

fn main() -> Result<(), BlakeoutGpuError> {
    // Initialize GPU miner with batch size
    let gpu_miner = BlakeoutGpu::new(8192)?;

    // Your block data
    let input_data = b"Block data to hash";
    let target_difficulty = 20;

    // Find hash meeting difficulty target
    if let Some(result) = gpu_miner.find_hash(input_data, 0, target_difficulty)? {
        println!("Found! Nonce: {}, Difficulty: {}",
                 result.nonce, result.difficulty);
    }

    Ok(())
}
```

### Batch Processing

```rust
// Process multiple nonces in parallel
let results = gpu_miner.hash_batch(input_data, start_nonce, target_difficulty)?;

for result in results {
    if result.difficulty >= target_difficulty {
        println!("Match found at nonce {}", result.nonce);
    }
}
```

## ALFIS Integration

To use GPU mining with ALFIS:

### 1. Build ALFIS with GPU support

```bash
cd Alfis-master
cargo build --release --features gpu
```

### 2. Run ALFIS

The miner will automatically detect and use GPU if available:

```bash
./target/release/alfis
```

You'll see in the logs:
```
INFO: GPU miner initialized successfully with batch size 8192
INFO: Thread 0 using GPU for mining
```

### 3. Configuration

GPU mining is automatically enabled when the `gpu` feature is compiled. The miner will:
- Use GPU on thread 0 for maximum efficiency
- Fall back to CPU if GPU is unavailable or encounters errors
- Continue using CPU threads for parallel mining

## Performance

Expected performance improvements over CPU:

| GPU Model | Hash Rate | Speedup vs CPU (8 cores) |
|-----------|-----------|--------------------------|
| RTX 4090  | ~500 MH/s | ~100x                    |
| RTX 3080  | ~300 MH/s | ~60x                     |
| RTX 2060  | ~150 MH/s | ~30x                     |
| GTX 1660  | ~100 MH/s | ~20x                     |

*Note: Actual performance depends on difficulty target and system configuration*

## Architecture

### CUDA Implementation

The GPU implementation consists of three main components:

1. **Blake2s CUDA Kernel** (`cuda/blake2s.cu`)
   - Device-side Blake2s implementation
   - Optimized for GPU execution
   - Supports streaming and batching

2. **Blakeout CUDA Kernel** (`cuda/blakeout.cu`)
   - Memory-hard algorithm implementation
   - 2MB buffer per hash (65,536 iterations)
   - Parallel nonce processing

3. **Rust Wrapper** (`src/lib.rs`, `src/gpu.rs`)
   - Safe Rust API
   - FFI bindings to CUDA code
   - Error handling and fallback logic

### Memory Considerations

Each hash requires:
- 2 MB buffer (65,536 × 32 bytes)
- Temporary state (~256 bytes)

For a batch of 8K hashes:
- GPU memory needed: ~16 GB
- Recommended: GPU with 8GB+ VRAM

## Troubleshooting

### "No CUDA GPU available"

Check:
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA compiler
```

### Build fails with "nvcc: command not found"

Add CUDA to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Runtime error: "CUDA error code: X"

Common codes:
- `1` (InvalidValue): Check input data size (max 500 bytes)
- `2` (OutOfMemory): Reduce batch size
- `3` (NotInitialized): CUDA not properly installed

### GPU underperforming

- Increase batch size for better GPU utilization
- Check GPU isn't thermal throttling (`nvidia-smi`)
- Ensure no other GPU-intensive processes running

## Benchmarks

Run benchmarks:
```bash
cargo bench
```

Compare GPU vs CPU performance across different batch sizes and difficulty targets.

## Development

### Project Structure

```
blakeout-gpu/
├── cuda/
│   ├── blake2s.cu      # Blake2s CUDA implementation
│   ├── blake2s.cuh     # Blake2s headers
│   └── blakeout.cu     # Blakeout CUDA kernel
├── src/
│   ├── lib.rs          # Public API
│   └── gpu.rs          # CUDA FFI bindings
├── examples/
│   └── gpu_miner.rs    # Example miner
└── build.rs            # CUDA compilation script
```

### Testing

```bash
# Run tests (requires GPU)
cargo test

# Run specific test
cargo test test_hash_batch
```

## License

MIT OR Apache-2.0

## Credits

- Original Blakeout algorithm by Revertron
- Blake2 reference implementation by BLAKE2 team
- CUDA optimization for ALFIS GPU mining
