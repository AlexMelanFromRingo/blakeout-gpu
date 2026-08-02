// blake2s.cu - CUDA kernel для Blake2s
#include <cuda_runtime.h>
#include <stdint.h>

// Blake2s константы
#define BLAKE2S_BLOCKBYTES 64
#define BLAKE2S_OUTBYTES 32

__constant__ uint32_t blake2s_iv[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__constant__ uint8_t blake2s_sigma[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void blake2s_g(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
    uint32_t x, uint32_t y
) {
    a = a + b + x;
    d = rotr32(d ^ a, 16);
    c = c + d;
    b = rotr32(b ^ c, 12);
    a = a + b + y;
    d = rotr32(d ^ a, 8);
    c = c + d;
    b = rotr32(b ^ c, 7);
}

__device__ void blake2s_compress(
    uint32_t h[8],
    const uint32_t m[16],
    uint32_t t0,
    uint32_t t1,
    uint32_t f0,
    uint32_t f1
) {
    uint32_t v[16];
    
    // Initialize v
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = blake2s_iv[i];
    }
    
    v[12] ^= t0;
    v[13] ^= t1;
    v[14] ^= f0;
    v[15] ^= f1;
    
    // 10 rounds
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        const uint8_t *s = blake2s_sigma[i];
        
        blake2s_g(v[0], v[4], v[8],  v[12], m[s[0]], m[s[1]]);
        blake2s_g(v[1], v[5], v[9],  v[13], m[s[2]], m[s[3]]);
        blake2s_g(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        blake2s_g(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
        
        blake2s_g(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        blake2s_g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        blake2s_g(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        blake2s_g(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }
    
    // Finalize
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

__device__ void blake2s_hash(
    const uint8_t *data,
    size_t len,
    uint8_t *out
) {
    uint32_t h[8];
    
    // Initialize state
    for (int i = 0; i < 8; i++) {
        h[i] = blake2s_iv[i];
    }
    h[0] ^= 0x01010000 ^ BLAKE2S_OUTBYTES;
    
    uint32_t m[16];
    size_t offset = 0;
    uint32_t t0 = 0, t1 = 0;
    
    // Process full blocks
    while (len > BLAKE2S_BLOCKBYTES) {
        // Load block into m
        for (int i = 0; i < 16; i++) {
            m[i] = ((uint32_t*)&data[offset])[i];
        }
        
        t0 += BLAKE2S_BLOCKBYTES;
        if (t0 < BLAKE2S_BLOCKBYTES) t1++;
        
        blake2s_compress(h, m, t0, t1, 0, 0);
        
        offset += BLAKE2S_BLOCKBYTES;
        len -= BLAKE2S_BLOCKBYTES;
    }
    
    // Final block
    uint8_t final_block[BLAKE2S_BLOCKBYTES] = {0};
    for (size_t i = 0; i < len; i++) {
        final_block[i] = data[offset + i];
    }
    
    for (int i = 0; i < 16; i++) {
        m[i] = ((uint32_t*)final_block)[i];
    }
    
    t0 += len;
    if (t0 < len) t1++;
    
    blake2s_compress(h, m, t0, t1, 0xFFFFFFFF, 0);
    
    // Output
    for (int i = 0; i < 8; i++) {
        ((uint32_t*)out)[i] = h[i];
    }
}

// Kernel для batch Blake2s
__global__ void blake2s_batch_kernel(
    const uint8_t *inputs,
    uint8_t *outputs,
    const size_t *lengths,
    size_t batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        size_t input_offset = idx * BLAKE2S_BLOCKBYTES * 2; // Max input size
        size_t output_offset = idx * BLAKE2S_OUTBYTES;
        
        blake2s_hash(
            &inputs[input_offset],
            lengths[idx],
            &outputs[output_offset]
        );
    }
}

// Kernel для Blakeout scratchpad заполнения
// Улучшенная версия с правильной синхронизацией
__global__ void blakeout_fill_scratchpad_kernel(
    uint8_t *scratchpads,
    const uint8_t *initial_data,
    const size_t *data_lengths,
    size_t batch_size,
    size_t current_wave,
    size_t hash_count
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const size_t HASH_SIZE = 32;
    size_t scratchpad_offset = batch_idx * (HASH_SIZE * hash_count);
    
    if (current_wave == 0) {
        // First hash: use initial data
        blake2s_hash(
            &initial_data[batch_idx * 128],
            data_lengths[batch_idx],
            &scratchpads[scratchpad_offset]
        );
    } else {
        // Subsequent hashes: use previous scratchpad data
        size_t start = (current_wave >= 2) ? (current_wave - 2) * HASH_SIZE : 0;
        size_t len = current_wave * HASH_SIZE - start;
        
        blake2s_hash(
            &scratchpads[scratchpad_offset + start],
            len,
            &scratchpads[scratchpad_offset + current_wave * HASH_SIZE]
        );
    }
}

// Kernel для финального хеширования Blakeout
__global__ void blakeout_finalize_kernel(
    const uint8_t *scratchpads,
    uint8_t *outputs,
    size_t batch_size,
    size_t scratchpad_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const uint8_t *scratchpad = &scratchpads[idx * scratchpad_size];
    
    // First pass: Hash forward
    uint32_t h_forward[8];
    for (int i = 0; i < 8; i++) {
        h_forward[i] = blake2s_iv[i];
    }
    h_forward[0] ^= 0x01010000 ^ 32;
    
    // Process entire scratchpad forward
    uint32_t m[16];
    uint32_t t0 = 0, t1 = 0;
    
    for (size_t offset = 0; offset < scratchpad_size; offset += 64) {
        for (int i = 0; i < 16; i++) {
            m[i] = ((uint32_t*)&scratchpad[offset])[i];
        }
        
        t0 += 64;
        if (t0 < 64) t1++;
        
        uint32_t f0 = (offset + 64 >= scratchpad_size) ? 0xFFFFFFFF : 0;
        blake2s_compress(h_forward, m, t0, t1, f0, 0);
    }
    
    // Second pass: Reverse and hash
    uint32_t h_final[8];
    for (int i = 0; i < 8; i++) {
        h_final[i] = blake2s_iv[i];
    }
    h_final[0] ^= 0x01010000 ^ 32;
    
    // Update with forward hash
    uint32_t m_forward[16] = {0};
    for (int i = 0; i < 8; i++) {
        m_forward[i] = h_forward[i];
    }
    blake2s_compress(h_final, m_forward, 32, 0, 0, 0);
    
    // Process scratchpad in reverse
    t0 = 0; t1 = 0;
    for (int offset = scratchpad_size - 64; offset >= 0; offset -= 64) {
        for (int i = 0; i < 16; i++) {
            // Read in reverse order
            int byte_idx = offset + (15 - i) * 4;
            uint32_t val = 0;
            for (int b = 0; b < 4; b++) {
                val |= ((uint32_t)scratchpad[byte_idx + (3 - b)]) << (b * 8);
            }
            m[i] = val;
        }
        
        t0 += 64;
        if (t0 < 64) t1++;
        
        uint32_t f0 = (offset == 0) ? 0xFFFFFFFF : 0;
        blake2s_compress(h_final, m, t0, t1, f0, 0);
    }
    
    // Write final output
    uint8_t *out = &outputs[idx * 32];
    for (int i = 0; i < 8; i++) {
        ((uint32_t*)out)[i] = h_final[i];
    }
}