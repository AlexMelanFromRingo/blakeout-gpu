use crate::BlakeoutGpuError;

#[cfg(not(no_cuda))]
use std::os::raw::{c_int, c_uchar, c_uint};

// FFI bindings to CUDA functions
#[cfg(not(no_cuda))]
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

/// Hash a batch of nonces using GPU (internal wrapper)
#[cfg(not(no_cuda))]
unsafe fn cuda_blakeout_hash_batch(
    input_data: *const c_uchar,
    input_len: usize,
    nonces: *const u64,
    nonce_count: c_uint,
    output_hashes: *mut c_uchar,
    output_difficulties: *mut c_uint,
    target_difficulty: c_uint,
) -> c_int {
    blakeout_hash_batch(
        input_data,
        input_len,
        nonces,
        nonce_count,
        output_hashes,
        output_difficulties,
        target_difficulty,
    )
}

// CUDA error codes
#[cfg(not(no_cuda))]
const CUDA_SUCCESS: c_int = 0;

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    #[cfg(no_cuda)]
    {
        false
    }
    #[cfg(not(no_cuda))]
    {
        // Try to initialize CUDA
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

/// Hash a batch of nonces using GPU (safe wrapper)
#[cfg(not(no_cuda))]
pub unsafe fn hash_batch(
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

    let result = cuda_blakeout_hash_batch(
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

/// Hash a batch of nonces using GPU (stub for no CUDA)
#[cfg(no_cuda)]
pub unsafe fn hash_batch(
    _input_data: &[u8],
    _nonces: &[u64],
    _output_hashes: &mut [u8],
    _output_difficulties: &mut [u32],
    _target_difficulty: u32,
) -> Result<(), BlakeoutGpuError> {
    Err(BlakeoutGpuError::NoGpuAvailable)
}
