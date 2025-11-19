#[cfg(feature = "gpu")]
use blakeout_gpu::{BlakeoutGpu, BlakeoutGpuError};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use crate::blockchain::hash_utils::*;
use crate::{Block, Bytes};

/// GPU miner configuration
pub struct GpuMinerConfig {
    pub batch_size: usize,
    pub enabled: bool,
}

impl Default for GpuMinerConfig {
    fn default() -> Self {
        GpuMinerConfig {
            batch_size: 4096, // Optimal batch size for RTX 4080 (8GB VRAM)
            enabled: true,
        }
    }
}

#[cfg(feature = "gpu")]
pub struct GpuMiner {
    hasher: Option<BlakeoutGpu>,
    config: GpuMinerConfig,
}

#[cfg(feature = "gpu")]
impl GpuMiner {
    pub fn new(config: GpuMinerConfig) -> Self {
        if !config.enabled {
            info!("GPU mining is disabled by configuration");
            return GpuMiner {
                hasher: None,
                config,
            };
        }

        match BlakeoutGpu::new(config.batch_size) {
            Ok(hasher) => {
                info!("GPU miner initialized successfully with batch size {}", config.batch_size);
                GpuMiner {
                    hasher: Some(hasher),
                    config,
                }
            }
            Err(BlakeoutGpuError::NoGpuAvailable) => {
                warn!("No CUDA GPU available, falling back to CPU mining");
                GpuMiner {
                    hasher: None,
                    config,
                }
            }
            Err(e) => {
                warn!("Failed to initialize GPU miner: {}, falling back to CPU mining", e);
                GpuMiner {
                    hasher: None,
                    config,
                }
            }
        }
    }

    pub fn is_available(&self) -> bool {
        self.hasher.is_some()
    }

    /// Try to find a hash using GPU
    /// Returns Some(block) if found, None if not found or error
    pub fn find_hash_gpu(
        &self,
        mut block: Block,
        target_diff: u32,
        running: Arc<AtomicBool>,
        start_nonce: u64,
    ) -> Option<Block> {
        let hasher = match &self.hasher {
            Some(h) => h,
            None => return None,
        };

        let batch_size = self.config.batch_size;
        let mut current_nonce = start_nonce;
        let mut max_diff = 0u32;

        info!("Starting GPU mining from nonce {}", start_nonce);

        let mut time = Instant::now();
        let mut hashes_computed = 0u64;

        while running.load(Ordering::Relaxed) {
            // Prepare block data (without nonce, will be added by GPU)
            let block_data = block.as_bytes_compact();

            // Process batch on GPU starting from current_nonce
            match hasher.hash_batch(&block_data, current_nonce, target_diff) {
                Ok(results) => {
                    hashes_computed += batch_size as u64;

                    // Check all results for target difficulty
                    for result in results {
                        if result.difficulty >= target_diff {
                            block.nonce = result.nonce;  // Use absolute nonce from GPU
                            block.hash = Bytes::from_bytes(&result.hash);
                            info!(
                                "GPU found hash! Nonce: {}, Difficulty: {}",
                                block.nonce, result.difficulty
                            );
                            return Some(block);
                        }

                        if result.difficulty > max_diff {
                            max_diff = result.difficulty;
                        }
                    }

                    current_nonce += batch_size as u64;

                    // Report stats every 10 batches
                    let elapsed = time.elapsed().as_millis();
                    if elapsed >= 10000 {
                        let speed = hashes_computed / (elapsed as u64 / 1000);
                        debug!(
                            "GPU mining speed: {} H/s, max difficulty: {}, target: {}",
                            speed, max_diff, target_diff
                        );
                        time = Instant::now();
                        hashes_computed = 0;
                    }
                }
                Err(e) => {
                    error!("GPU mining error: {}, switching to CPU", e);
                    return None;
                }
            }

            // Check for nonce overflow
            if current_nonce > u64::MAX - batch_size as u64 {
                warn!("Nonce overflow in GPU mining");
                return None;
            }
        }

        None
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuMiner;

#[cfg(not(feature = "gpu"))]
impl GpuMiner {
    pub fn new(_config: GpuMinerConfig) -> Self {
        GpuMiner
    }

    pub fn is_available(&self) -> bool {
        false
    }

    pub fn find_hash_gpu(
        &self,
        _block: Block,
        _target_diff: u32,
        _running: Arc<AtomicBool>,
        _start_nonce: u64,
    ) -> Option<Block> {
        None
    }
}
