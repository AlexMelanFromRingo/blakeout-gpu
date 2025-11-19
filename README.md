# Blakeout GPU Mining for ALFIS

GPU-accelerated mining implementation for ALFIS blockchain using CUDA.

## 📁 Project Structure

This repository contains:

- **`blakeout-master/`** - Original Blakeout hash library (Rust)
- **`blakeout-gpu/`** - GPU-accelerated Blakeout implementation (CUDA + Rust)
- **`Alfis-master/`** - ALFIS blockchain with integrated GPU mining support
- **`GPU_MINING.md`** - Comprehensive guide (Russian)

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+ with nvcc compiler
- Rust 1.70+

### Build with GPU Support

```bash
# Build GPU library
cd blakeout-gpu
cargo build --release

# Build ALFIS with GPU mining
cd ../Alfis-master
cargo build --release --features gpu

# Run ALFIS
./target/release/alfis
```

### Test GPU Mining

```bash
cd blakeout-gpu
cargo run --release --example gpu_miner
```

## ⚡ Performance

Expected speedup compared to CPU mining:

| GPU Model | Hash Rate | Speedup |
|-----------|-----------|---------|
| RTX 4090  | ~500 MH/s | 100x    |
| RTX 3080  | ~300 MH/s | 60x     |
| RTX 2060  | ~150 MH/s | 30x     |

## 📖 Documentation

- **[GPU_MINING.md](GPU_MINING.md)** - Full Russian guide with installation, usage, and troubleshooting
- **[blakeout-gpu/README.md](blakeout-gpu/README.md)** - Technical documentation for the GPU library

## 🔧 How It Works

### Architecture

```
ALFIS Miner
    ↓
Thread 0: GPU Mining (8K hashes/batch)
Thread 1-N: CPU Mining (parallel)
    ↓
blakeout-gpu Library
    ↓
CUDA Kernels (Blake2s + Blakeout)
    ↓
NVIDIA GPU
```

### Key Features

- **Parallel Processing**: 4K-16K hashes computed simultaneously on GPU
- **Memory-Hard**: Full 2MB Blakeout algorithm implemented in CUDA
- **Automatic Fallback**: Gracefully falls back to CPU if GPU unavailable
- **Hybrid Mining**: GPU + CPU threads work together
- **Zero Configuration**: Automatically detects and uses available GPU

## 🛠️ Build Without CUDA

The library gracefully handles missing CUDA:

```bash
cd blakeout-gpu
cargo build  # Will compile without GPU support if nvcc not found
```

You'll see:
```
warning: CUDA compiler (nvcc) not found. GPU support will be disabled.
```

ALFIS will automatically use CPU-only mining in this case.

## 📦 Components

### blakeout-gpu Library

CUDA implementation of Blakeout:
- `cuda/blake2s.cu` - Blake2s CUDA kernel
- `cuda/blakeout.cu` - Blakeout memory-hard algorithm
- `src/lib.rs` - Rust API
- `src/gpu.rs` - FFI bindings

### ALFIS Integration

Modified ALFIS with GPU support:
- `src/gpu_miner.rs` - GPU mining module
- `src/miner.rs` - Hybrid CPU+GPU miner
- Optional `gpu` feature flag

## 🔍 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Install CUDA
sudo apt-get install cuda-toolkit-12-0
```

### Out of Memory

Reduce batch size in `Alfis-master/src/gpu_miner.rs`:

```rust
GpuMinerConfig {
    batch_size: 2048,  // Instead of 8192
    enabled: true,
}
```

### Build Errors

See [GPU_MINING.md](GPU_MINING.md) for detailed troubleshooting.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- OpenCL support for AMD GPUs
- Multi-GPU support
- Memory optimizations
- Dynamic batch sizing

## 📄 License

MIT OR Apache-2.0

## 🙏 Credits

- Original Blakeout algorithm by [Revertron](https://github.com/Revertron)
- ALFIS blockchain by [Revertron](https://github.com/Revertron/Alfis)
- GPU port implementation for enhanced mining performance

---

**Note**: This implementation is optimized for ALFIS blockchain mining. Performance varies based on target difficulty and hardware configuration.

For detailed setup instructions in Russian, see [GPU_MINING.md](GPU_MINING.md).
