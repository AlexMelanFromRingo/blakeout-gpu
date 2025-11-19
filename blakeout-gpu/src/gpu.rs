use crate::BlakeoutGpuError;

#[cfg(not(no_cuda))]
use std::os::raw::{c_int, c_uchar, c_uint};

#[repr(C)]
pub struct BlakeoutContext {
    _private: [u8; 0],
}

#[cfg(not(no_cuda))]
#[link(name = "blakeout_cuda")]
extern "C" {
    fn blakeout_create_context(batch_size: c_uint) -> *mut BlakeoutContext;
    fn blakeout_destroy_context(ctx: *mut BlakeoutContext);
    fn blakeout_hash_batch_ctx(
        ctx: *mut BlakeoutContext,
        h_input_data: *const c_uchar,
        input_len: usize,
        h_nonces: *const u64,
        nonce_count: c_uint,
        h_output_hashes: *mut c_uchar,
        h_output_difficulties: *mut c_uint,
        target_difficulty: c_uint,
    ) -> c_int;
}

const CUDA_SUCCESS: c_int = 0;

pub fn is_cuda_available() -> bool {
    #[cfg(no_cuda)]
    { false }
    #[cfg(not(no_cuda))]
    {
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}

#[cfg(not(no_cuda))]
pub unsafe fn create_context(batch_size: usize) -> Result<*mut BlakeoutContext, BlakeoutGpuError> {
    let ctx = blakeout_create_context(batch_size as c_uint);
    if ctx.is_null() {
        Err(BlakeoutGpuError::CudaError("Failed to create GPU context".to_string()))
    } else {
        Ok(ctx)
    }
}

#[cfg(no_cuda)]
pub unsafe fn create_context(_batch_size: usize) -> Result<*mut BlakeoutContext, BlakeoutGpuError> {
    Err(BlakeoutGpuError::NoGpuAvailable)
}

#[cfg(not(no_cuda))]
pub unsafe fn destroy_context(ctx: *mut BlakeoutContext) {
    if !ctx.is_null() {
        blakeout_destroy_context(ctx);
    }
}

#[cfg(no_cuda)]
pub unsafe fn destroy_context(_ctx: *mut BlakeoutContext) {}

#[cfg(not(no_cuda))]
pub unsafe fn hash_batch_ctx(
    ctx: *mut BlakeoutContext,
    input_data: &[u8],
    nonces: &[u64],
    output_hashes: &mut [u8],
    output_difficulties: &mut [u32],
    target_difficulty: u32,
) -> Result<(), BlakeoutGpuError> {
    if ctx.is_null() || input_data.len() > 500 {
        return Err(BlakeoutGpuError::InvalidInput);
    }

    let result = blakeout_hash_batch_ctx(
        ctx,
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
        Err(BlakeoutGpuError::CudaError(format!("CUDA error code: {}", result)))
    }
}

#[cfg(no_cuda)]
pub unsafe fn hash_batch_ctx(
    _ctx: *mut BlakeoutContext,
    _input_data: &[u8],
    _nonces: &[u64],
    _output_hashes: &mut [u8],
    _output_difficulties: &mut [u32],
    _target_difficulty: u32,
) -> Result<(), BlakeoutGpuError> {
    Err(BlakeoutGpuError::NoGpuAvailable)
}
