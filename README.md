# Blakeout GPU Mining for ALFIS

GPU-accelerated [Blakeout](https://github.com/Revertron/blakeout) hash for
mining blocks on the [ALFIS](https://github.com/Revertron/Alfis) DNS
blockchain. Provides a CUDA library (`blakeout-gpu`) plus a fork of ALFIS
(`Alfis-master/`) wired to use it via a `gpu` cargo feature.

## What's in the box

| Path | Purpose |
|------|---------|
| `blakeout-gpu/`  | The CUDA library — Rust crate with FFI to a CUDA kernel that batches Blakeout hashes per nonce. |
| `blakeout-gpu/cuda/blake2s.cu`  | Hand-rolled Blake2s on the device. |
| `blakeout-gpu/cuda/blakeout.cu` | The 65,536-iteration Blakeout chain on top of `blake2s`, plus the host glue. |
| `blakeout-master/` | Vendored upstream CPU Blakeout used by both the lib's `blakeout_gpu_matches_cpu_reference` test and ALFIS's CPU miner. |
| `Alfis-master/`  | ALFIS with the GPU miner wired in — see `src/gpu_miner.rs` and the `gpu` cargo feature in `Cargo.toml`. |

## Status

* GPU mining works and produces hashes that match the canonical CPU
  Blakeout byte-for-byte (verified by an end-to-end test, see
  *Correctness* below).
* ALFIS is integrated: build it with `--features gpu` to get GPU mining
  on thread 0; non-GPU threads fall back to CPU.
* Performance on RTX 4080: **~1,550 H/s** at batch size 4096, ~3.5×
  faster than the same machine's CPU mining. Blakeout is intentionally
  GPU-resistant (65,536 sequential Blake2s iterations / 2 MB memory-hard
  buffer per hash) so this multiplier is close to the theoretical
  ceiling for this algorithm on this hardware.

## Correctness

The previous version of this project shipped a silent bug: row 3 of the
device-side Blake2s SIGMA permutation table was wrong from index 9
onwards. Every G call in round 3 still indexed valid `m[]` words, so the
kernel always ran cleanly — but it computed a *non-Blake2s* hash whenever
any of `m[4..15]` held non-zero bytes. Since real ALFIS blocks are always
larger than 16 bytes, this meant **every "mined" hash from the prior
version would have been rejected by the network**.

The fix is one character — restoring the canonical row from RFC 7693 —
and it's locked in by:

* `tests::gpu_blake2s_matches_reference` — runs the kernel against
  several inputs of varied length and compares to the `blake2` Rust
  crate output (which itself agrees with `hashlib.blake2s`).
* `tests::test_gpu_matches_cpu_reference` — runs the full Blakeout
  chain on the GPU and compares to `blakeout::Blakeout` byte-for-byte.
* `tests::cpu_kernel_mirror_matches_blakeout` — a Rust port of the
  kernel logic that runs on the host with no CUDA dependency, so the
  *design* of the kernel (chain length, two-pass forward+reverse
  hashing) is verifiable even on machines without an NVIDIA GPU.

## Build & test

```bash
# CUDA library + tests
cd blakeout-gpu
cargo test --release            # 7 tests, including CPU↔GPU equivalence
cargo run --release --example gpu_miner    # one-shot demo: find a hash at difficulty 18

# Bench across batch sizes (perf only — no correctness)
cargo run --release --example perf_test
```

```bash
# ALFIS with GPU mining
cd Alfis-master
cargo build --release --features gpu --no-default-features
./target/release/alfis --no-gui    # boots, GPU miner kicks in on thread 0
```

The `webgui` default feature pulls in `wry`+`tao`+`glib-2.0`; if those
system libraries aren't installed, build with `--no-default-features` as
above. The GPU feature is independent of the GUI feature.

## Hardware requirements

* NVIDIA GPU with compute capability ≥ 8.0 (sm_86 default; override
  with `CUDA_COMPUTE_ARCH=sm_89` for a 4090, etc).
* CUDA Toolkit ≥ 11.0 (built and tested with 12.0).
* NVIDIA driver supporting your CUDA version.
* On WSL2: CUDA works (WSL CUDA driver from NVIDIA), but native
  Windows or Linux gives slightly better headline numbers.

## Performance characteristics

The kernel is bottlenecked by the algorithm, not by GPU silicon. Per-thread
work is ~65,536 sequential Blake2s rounds touching a 2 MB scratch buffer,
so threads stall on memory, not on FP throughput.

| Batch size | RTX 4080 hash rate | Time per hash | VRAM used |
|------------|--------------------|---------------|-----------|
| 1024 | ~365 H/s | 2.7 ms | 2 GB |
| 2048 | ~720 H/s | 1.4 ms | 4 GB |
| **4096** | **~1,550 H/s** | **0.65 ms** | **8 GB** |
| 8192 | OOM on 16 GB cards | – | – |

Numbers higher than this on the same algorithm have not been
demonstrated anywhere — Blakeout's design (sequential dependencies +
memory-hard buffer) puts a hard ceiling around `Blake2s_throughput / 65536`.

## Project layout

```
blakeout-gpu/
├── blakeout-gpu/         CUDA library (Rust + .cu kernels)
│   ├── cuda/             blake2s.cu, blakeout.cu, blake2s.cuh
│   ├── src/lib.rs        public API (BlakeoutGpu, gpu_blake2s)
│   ├── src/gpu.rs        FFI bindings
│   ├── examples/         gpu_miner.rs, perf_test.rs
│   └── tests/            integration tests via the public API
├── blakeout-master/      vendored CPU Blakeout reference
├── Alfis-master/         ALFIS fork with `gpu` cargo feature
│   └── src/gpu_miner.rs  drop-in GPU miner used by miner.rs (thread 0)
├── build_with_gpu.sh     convenience build for Linux/macOS
├── build_windows.ps1     convenience build for Windows
└── docs/                 supporting markdown (PERFORMANCE.md etc.)
```

## License

Same as upstream ALFIS / Blakeout. Components are MIT or Apache-2.0
(see individual `LICENSE` / `Cargo.toml`).
