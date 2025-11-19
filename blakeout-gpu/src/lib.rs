pub mod gpu;

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum BlakeoutGpuError {
    CudaError(String),
    NoGpuAvailable,
    InvalidInput,
}

impl fmt::Display for BlakeoutGpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BlakeoutGpuError::CudaError(msg) => write!(f, "CUDA Error: {}", msg),
            BlakeoutGpuError::NoGpuAvailable => write!(f, "No GPU available"),
            BlakeoutGpuError::InvalidInput => write!(f, "Invalid input"),
        }
    }
}

impl Error for BlakeoutGpuError {}

/// Result of a batch hash operation
pub struct HashResult {
    pub nonce: u64,
    pub hash: [u8; 32],
    pub difficulty: u32,
}

/// GPU-accelerated Blakeout hasher
pub struct BlakeoutGpu {
    batch_size: usize,
}

impl BlakeoutGpu {
    /// Create a new GPU hasher with specified batch size
    pub fn new(batch_size: usize) -> Result<Self, BlakeoutGpuError> {
        if !gpu::is_cuda_available() {
            return Err(BlakeoutGpuError::NoGpuAvailable);
        }
        Ok(BlakeoutGpu { batch_size })
    }

    /// Hash a batch of nonces in parallel on GPU
    /// Returns results for all hashes, caller should filter by difficulty
    pub fn hash_batch(
        &self,
        input_data: &[u8],
        start_nonce: u64,
        target_difficulty: u32,
    ) -> Result<Vec<HashResult>, BlakeoutGpuError> {
        let nonces: Vec<u64> = (start_nonce..start_nonce + self.batch_size as u64).collect();

        let mut output_hashes = vec![0u8; self.batch_size * 32];
        let mut output_difficulties = vec![0u32; self.batch_size];

        unsafe {
            gpu::blakeout_hash_batch(
                input_data,
                &nonces,
                &mut output_hashes,
                &mut output_difficulties,
                target_difficulty,
            )?;
        }

        let mut results = Vec::new();
        for i in 0..self.batch_size {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&output_hashes[i * 32..(i + 1) * 32]);
            results.push(HashResult {
                nonce: nonces[i],
                hash,
                difficulty: output_difficulties[i],
            });
        }

        Ok(results)
    }

    /// Find a hash that meets the target difficulty
    /// Returns the first matching hash found, or None if none found in batch
    pub fn find_hash(
        &self,
        input_data: &[u8],
        start_nonce: u64,
        target_difficulty: u32,
    ) -> Result<Option<HashResult>, BlakeoutGpuError> {
        let results = self.hash_batch(input_data, start_nonce, target_difficulty)?;

        for result in results {
            if result.difficulty >= target_difficulty {
                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        // This test will pass if GPU is available
        if gpu::is_cuda_available() {
            let hasher = BlakeoutGpu::new(1024);
            assert!(hasher.is_ok());
        }
    }

    #[test]
    fn test_hash_batch() {
        if !gpu::is_cuda_available() {
            println!("Skipping GPU test: no CUDA device available");
            return;
        }

        let hasher = BlakeoutGpu::new(10).unwrap();
        let input = b"test data";
        let results = hasher.hash_batch(input, 0, 0).unwrap();

        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.nonce, i as u64);
            assert_eq!(result.hash.len(), 32);
        }
    }
}
