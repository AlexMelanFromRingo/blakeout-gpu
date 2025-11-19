use crate::BlakeoutGpuError;
use std::os::raw::{c_int, c_uchar, c_uint};

// FFI bindings to CUDA functions
#[link(name = "blakeout_cuda")]
extern "C" {
    fn blakeout_hash_batch(
        h_input_data: *const c_uchar,
        input_len: usize,
        h_nonces: *const u64,
        nonce_count: c_uint,
        h_output_hashes: *mut c_uchar,
        h_output_difficulties: *mut c_uint,
        target_difficulty: c_uint,
    ) -> c_int;
}

// CUDA error codes
const CUDA_SUCCESS: c_int = 0;

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    // Try to initialize CUDA
    std::process::Command::new("nvidia-smi")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Hash a batch of nonces using GPU
pub unsafe fn blakeout_hash_batch(
    input_data: &[u8],
    nonces: &[u64],
    output_hashes: &mut [u8],
    output_difficulties: &mut [u32],
    target_difficulty: u32,
) -> Result<(), BlakeoutGpuError> {
    if input_data.len() > 500 {
        return Err(BlakeoutGpuError::InvalidInput);
    }

    if output_hashes.len() != nonces.len() * 32 {
        return Err(BlakeoutGpuError::InvalidInput);
    }

    if output_difficulties.len() != nonces.len() {
        return Err(BlakeoutGpuError::InvalidInput);
    }

    let result = blakeout_hash_batch(
        input_data.as_ptr(),
        input_data.len(),
        nonces.as_ptr(),
        nonces.len() as c_uint,
        output_hashes.as_mut_ptr(),
        output_difficulties.as_mut_ptr(),
        target_difficulty,
    );

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(BlakeoutGpuError::CudaError(format!(
            "CUDA error code: {}",
            result
        )))
    }
}
