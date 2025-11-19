#include "blake2s.cuh"
#include <stdint.h>
#include <stdio.h>

#define HASH_SIZE 32
#define HASH_COUNT 65536
#define BUFFER_SIZE (HASH_SIZE * HASH_COUNT)  // 2MB per hash

// Kernel to compute Blakeout hashes for multiple nonces in parallel
__global__ void blakeout_hash_kernel(
    const uint8_t* input_data,     // Base input data (without nonce)
    size_t input_len,               // Length of input data
    const uint64_t* nonces,         // Array of nonces to try
    uint32_t nonce_count,           // Number of nonces
    uint8_t* buffers,               // Pre-allocated buffers (2MB per hash)
    uint8_t* output_hashes,         // Output hashes (32 bytes each)
    uint32_t* output_difficulties,  // Output difficulty values
    uint32_t target_difficulty      // Target difficulty to find
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nonce_count) {
        return;
    }

    // Use pre-allocated buffer for this thread
    uint8_t* buffer = buffers + (idx * BUFFER_SIZE);

    blake2s_state state;
    uint8_t temp_hash[HASH_SIZE];

    // Prepare input with nonce
    uint8_t full_input[512]; // Assuming input_len + 8 <= 512
    for (size_t i = 0; i < input_len; i++) {
        full_input[i] = input_data[i];
    }

    // Append nonce (little-endian)
    uint64_t nonce = nonces[idx];
    full_input[input_len + 0] = (nonce >> 0) & 0xFF;
    full_input[input_len + 1] = (nonce >> 8) & 0xFF;
    full_input[input_len + 2] = (nonce >> 16) & 0xFF;
    full_input[input_len + 3] = (nonce >> 24) & 0xFF;
    full_input[input_len + 4] = (nonce >> 32) & 0xFF;
    full_input[input_len + 5] = (nonce >> 40) & 0xFF;
    full_input[input_len + 6] = (nonce >> 48) & 0xFF;
    full_input[input_len + 7] = (nonce >> 56) & 0xFF;

    size_t full_len = input_len + 8;

    // Step 1: Hash input and store in first block
    blake2s(buffer, full_input, full_len);

    // Step 2: Sequential chaining - hash last 64 bytes iteratively
    for (uint32_t x = 1; x < HASH_COUNT; x++) {
        uint32_t offset = x * HASH_SIZE;
        uint32_t prev_offset = (x - 1) * HASH_SIZE;

        // For first iteration, only use 32 bytes (previous hash)
        // For subsequent iterations, use last 64 bytes
        if (x == 1) {
            blake2s(buffer + offset, buffer + prev_offset, HASH_SIZE);
        } else {
            blake2s(buffer + offset, buffer + prev_offset - HASH_SIZE, HASH_SIZE * 2);
        }
    }

    // Step 3: Hash entire buffer forward
    blake2s_init(&state, HASH_SIZE);
    blake2s_update(&state, buffer, BUFFER_SIZE);

    // Step 4: Reverse buffer
    for (uint32_t i = 0; i < BUFFER_SIZE / 2; i++) {
        uint8_t temp = buffer[i];
        buffer[i] = buffer[BUFFER_SIZE - 1 - i];
        buffer[BUFFER_SIZE - 1 - i] = temp;
    }

    // Step 5: Hash reversed buffer and combine
    blake2s_update(&state, buffer, BUFFER_SIZE);
    blake2s_final(&state, temp_hash);

    // Copy result to output
    for (int i = 0; i < HASH_SIZE; i++) {
        output_hashes[idx * HASH_SIZE + i] = temp_hash[i];
    }

    // Calculate difficulty (leading zeros from start + trailing zeros from end)
    uint64_t start_val = 0;
    uint64_t end_val = 0;

    for (int i = 0; i < 8; i++) {
        start_val |= ((uint64_t)temp_hash[i]) << (i * 8);
        end_val |= ((uint64_t)temp_hash[24 + i]) << (i * 8);
    }

    uint32_t leading = __clzll(start_val);
    uint32_t trailing = __clzll(__brevll(end_val)); // Reverse bits and count leading zeros
    uint32_t difficulty = leading + trailing;

    output_difficulties[idx] = difficulty;
}

// Host wrapper function
extern "C" {
    cudaError_t blakeout_hash_batch(
        const uint8_t* h_input_data,
        size_t input_len,
        const uint64_t* h_nonces,
        uint32_t nonce_count,
        uint8_t* h_output_hashes,
        uint32_t* h_output_difficulties,
        uint32_t target_difficulty
    ) {
        // Allocate device memory
        uint8_t* d_input_data;
        uint64_t* d_nonces;
        uint8_t* d_buffers;
        uint8_t* d_output_hashes;
        uint32_t* d_output_difficulties;

        cudaMalloc(&d_input_data, input_len);
        cudaMalloc(&d_nonces, nonce_count * sizeof(uint64_t));
        cudaMalloc(&d_buffers, (size_t)nonce_count * BUFFER_SIZE);  // 2MB per hash
        cudaMalloc(&d_output_hashes, nonce_count * HASH_SIZE);
        cudaMalloc(&d_output_difficulties, nonce_count * sizeof(uint32_t));

        // Copy input to device
        cudaMemcpy(d_input_data, h_input_data, input_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nonces, h_nonces, nonce_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (nonce_count + threadsPerBlock - 1) / threadsPerBlock;

        blakeout_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input_data, input_len, d_nonces, nonce_count,
            d_buffers, d_output_hashes, d_output_difficulties, target_difficulty
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_input_data);
            cudaFree(d_nonces);
            cudaFree(d_buffers);
            cudaFree(d_output_hashes);
            cudaFree(d_output_difficulties);
            return err;
        }

        // Wait for kernel to complete
        cudaDeviceSynchronize();

        // Copy results back to host
        cudaMemcpy(h_output_hashes, d_output_hashes, nonce_count * HASH_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_difficulties, d_output_difficulties, nonce_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input_data);
        cudaFree(d_nonces);
        cudaFree(d_buffers);
        cudaFree(d_output_hashes);
        cudaFree(d_output_difficulties);

        return cudaSuccess;
    }
}
