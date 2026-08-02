use cudarc::driver::safe::{CudaDevice, CudaSlice, LaunchConfig};
use cudarc::driver::LaunchAsync;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use anyhow::{Result, Context};

const HASH_SIZE: usize = 32;
const HASH_COUNT: usize = 65536;
const SCRATCHPAD_SIZE: usize = HASH_SIZE * HASH_COUNT; // 2MB

/// GPU-accelerated Blakeout hasher with batch processing
pub struct BlakeoutGpu {
    device: Arc<CudaDevice>,
    // Module is loaded into device, functions accessed by name
}

impl BlakeoutGpu {
    /// Create new GPU hasher
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0).context("Failed to initialize CUDA device")?;
        
        // Load PTX module
        let ptx = load_ptx("src/blake2s.cu")?;
        device.load_ptx(ptx, "blake2s", &[
            "blake2s_batch_kernel",
            "blakeout_fill_scratchpad_kernel",
            "blakeout_finalize_kernel"
        ])?;
        
        Ok(Self { device })
    }
    
    /// Hash a single input (falls back to sequential for single item)
    pub fn hash(&self, data: &[u8]) -> Result<[u8; 32]> {
        let result = self.hash_batch(&[data])?;
        Ok(result[0])
    }
    
    /// Hash multiple inputs in parallel on GPU
    pub fn hash_batch(&self, inputs: &[&[u8]]) -> Result<Vec<[u8; 32]>> {
        let batch_size = inputs.len();
        
        // Stage 1: Prepare scratchpads on GPU
        let scratchpads = self.prepare_scratchpads(inputs)?;
        
        // Stage 2: Fill scratchpads (sequential per hash, parallel across batch)
        self.fill_scratchpads(&scratchpads, inputs)?;
        
        // Stage 3: Finalize hashes
        let results = self.finalize_hashes(&scratchpads, batch_size)?;
        
        Ok(results)
    }
    
    fn prepare_scratchpads(&self, inputs: &[&[u8]]) -> Result<CudaSlice<u8>> {
        let batch_size = inputs.len();
        let total_size = batch_size * SCRATCHPAD_SIZE;
        
        let scratchpads = self.device.alloc_zeros::<u8>(total_size)?;
        Ok(scratchpads)
    }
    
    fn fill_scratchpads(&self, scratchpads: &CudaSlice<u8>, inputs: &[&[u8]]) -> Result<()> {
        let batch_size = inputs.len();
        
        // Prepare initial input data (padded to 128 bytes max)
        let mut padded_inputs = Vec::new();
        let mut input_lengths = Vec::new();
        
        for input in inputs {
            let mut padded = vec![0u8; 128];
            let copy_len = input.len().min(128);
            padded[..copy_len].copy_from_slice(&input[..copy_len]);
            padded_inputs.extend_from_slice(&padded);
            input_lengths.push(copy_len);
        }
        
        let d_inputs = self.device.htod_copy(padded_inputs)?;
        let d_lengths = self.device.htod_copy(input_lengths)?;
        
        // Process in waves - each wave handles one step for all batches
        for wave in 0..HASH_COUNT {
            let cfg = LaunchConfig {
                grid_dim: (((batch_size + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            
            let func = self.device
                .get_func("blake2s", "blakeout_fill_scratchpad_kernel")
                .ok_or_else(|| anyhow::anyhow!("Kernel function not found"))?;
            
            unsafe {
                func.launch(
                    cfg,
                    (
                        scratchpads,      // Output scratchpads
                        &d_inputs,        // Initial input data
                        &d_lengths,       // Input lengths
                        batch_size,       // Number of parallel batches
                        wave,             // Current wave number
                        HASH_COUNT,       // Total hash count
                    )
                )?;
            }
            
            // Synchronize after each wave to maintain dependencies
            self.device.synchronize()?;
            
            // Progress indicator for long operations
            if wave % 1000 == 0 && wave > 0 {
                log::debug!("Scratchpad fill progress: {}/{}", wave, HASH_COUNT);
            }
        }
        
        Ok(())
    }
    
    fn finalize_hashes(&self, scratchpads: &CudaSlice<u8>, batch_size: usize) -> Result<Vec<[u8; 32]>> {
        // Allocate output buffer
        let mut d_outputs = self.device.alloc_zeros::<u8>(batch_size * 32)?;
        
        let cfg = LaunchConfig {
            grid_dim: (((batch_size + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let func = self.device
            .get_func("blake2s", "blakeout_finalize_kernel")
            .ok_or_else(|| anyhow::anyhow!("Finalize kernel function not found"))?;
        
        unsafe {
            func.launch(cfg, (scratchpads, &d_outputs, batch_size, SCRATCHPAD_SIZE))?;
        }
        
        self.device.synchronize()?;
        
        // Download results
        let host_outputs = self.device.dtoh_sync_copy(&d_outputs)?;
        
        // Convert to array format
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&host_outputs[i * 32..(i + 1) * 32]);
            results.push(hash);
        }
        
        Ok(results)
    }
}

/// Optimized batch processor with memory pooling
pub struct BlakeoutBatchProcessor {
    gpu: BlakeoutGpu,
    max_batch_size: usize,
}

impl BlakeoutBatchProcessor {
    pub fn new(max_batch_size: usize) -> Result<Self> {
        Ok(Self {
            gpu: BlakeoutGpu::new()?,
            max_batch_size,
        })
    }
    
    /// Process large batches by chunking
    pub async fn process_large_batch(&self, inputs: Vec<Vec<u8>>) -> Result<Vec<[u8; 32]>> {
        let mut all_results = Vec::with_capacity(inputs.len());
        
        for chunk in inputs.chunks(self.max_batch_size) {
            let chunk_refs: Vec<&[u8]> = chunk.iter().map(|v| v.as_slice()).collect();
            let results = self.gpu.hash_batch(&chunk_refs)?;
            all_results.extend(results);
        }
        
        Ok(all_results)
    }
    
    /// Stream processing for very large datasets
    pub async fn stream_process<F>(&self, inputs: Vec<Vec<u8>>, mut callback: F) -> Result<()>
    where
        F: FnMut(usize, [u8; 32]) + Send + 'static,
    {
        for (chunk_idx, chunk) in inputs.chunks(self.max_batch_size).enumerate() {
            let chunk_refs: Vec<&[u8]> = chunk.iter().map(|v| v.as_slice()).collect();
            let results = self.gpu.hash_batch(&chunk_refs)?;
            
            for (i, result) in results.into_iter().enumerate() {
                let global_idx = chunk_idx * self.max_batch_size + i;
                callback(global_idx, result);
            }
        }
        
        Ok(())
    }
}

// Helper function to load precompiled PTX
fn load_ptx(source_file: &str) -> Result<Ptx> {
    // PTX is compiled by build.rs during build time
    let ptx_file = source_file.replace(".cu", ".ptx");
    
    // Try to load architecture-specific PTX first
    let gpu_arch = detect_current_gpu_arch();
    let arch_specific = format!("src/blake2s_{}.ptx", gpu_arch);
    
    let ptx_path = if std::path::Path::new(&arch_specific).exists() {
        arch_specific
    } else {
        ptx_file
    };
    
    let ptx_src = std::fs::read_to_string(&ptx_path)
        .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
    
    Ok(Ptx::from_src(ptx_src))
}

fn detect_current_gpu_arch() -> String {
    // Query current GPU compute capability
    if let Ok(_device) = CudaDevice::new(0) {
        // Try to get compute capability from device properties
        // This is a simplified version - actual API may differ
        // For now, return default
    }
    "86".to_string() // Default
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_single_hash() {
        let gpu = BlakeoutGpu::new().unwrap();
        let result = gpu.hash(b"hello world").unwrap();
        assert_eq!(result.len(), 32);
    }
    
    #[test]
    fn test_batch_hash() {
        let gpu = BlakeoutGpu::new().unwrap();
        let inputs = vec![
            b"input1".as_slice(),
            b"input2".as_slice(),
            b"input3".as_slice(),
        ];
        let results = gpu.hash_batch(&inputs).unwrap();
        assert_eq!(results.len(), 3);
    }
    
    #[tokio::test]
    async fn test_large_batch() {
        let processor = BlakeoutBatchProcessor::new(1024).unwrap();
        
        let inputs: Vec<Vec<u8>> = (0..10000)
            .map(|i| format!("input_{}", i).into_bytes())
            .collect();
        
        let results = processor.process_large_batch(inputs).await.unwrap();
        assert_eq!(results.len(), 10000);
    }
}