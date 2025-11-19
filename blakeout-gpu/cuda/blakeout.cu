#include "blake2s.cuh"
#include <stdint.h>
#include <stdio.h>

#define HASH_SIZE 32
#define HASH_COUNT 65536
#define BUFFER_SIZE (HASH_SIZE * HASH_COUNT)

__global__ void blakeout_hash_kernel(
    const uint8_t* input_data,
    size_t input_len,
    const uint64_t* nonces,
    uint32_t nonce_count,
    uint8_t* buffers,
    uint8_t* output_hashes,
    uint32_t* output_difficulties,
    uint32_t target_difficulty
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nonce_count) return;

    uint8_t* buffer = buffers + (idx * BUFFER_SIZE);
    blake2s_state state;
    uint8_t temp_hash[HASH_SIZE];
    uint8_t full_input[512];
    
    for (size_t i = 0; i < input_len; i++) {
        full_input[i] = input_data[i];
    }

    uint64_t nonce = nonces[idx];
    full_input[input_len + 0] = (nonce >> 0) & 0xFF;
    full_input[input_len + 1] = (nonce >> 8) & 0xFF;
    full_input[input_len + 2] = (nonce >> 16) & 0xFF;
    full_input[input_len + 3] = (nonce >> 24) & 0xFF;
    full_input[input_len + 4] = (nonce >> 32) & 0xFF;
    full_input[input_len + 5] = (nonce >> 40) & 0xFF;
    full_input[input_len + 6] = (nonce >> 48) & 0xFF;
    full_input[input_len + 7] = (nonce >> 56) & 0xFF;

    blake2s(buffer, full_input, input_len + 8);

    for (uint32_t x = 1; x < HASH_COUNT; x++) {
        uint32_t offset = x * HASH_SIZE;
        uint32_t prev_offset = (x - 1) * HASH_SIZE;
        if (x == 1) {
            blake2s(buffer + offset, buffer + prev_offset, HASH_SIZE);
        } else {
            blake2s(buffer + offset, buffer + prev_offset - HASH_SIZE, HASH_SIZE * 2);
        }
    }

    blake2s_init(&state, HASH_SIZE);
    blake2s_update(&state, buffer, BUFFER_SIZE);

    for (uint32_t i = 0; i < BUFFER_SIZE / 2; i++) {
        uint8_t temp = buffer[i];
        buffer[i] = buffer[BUFFER_SIZE - 1 - i];
        buffer[BUFFER_SIZE - 1 - i] = temp;
    }

    blake2s_update(&state, buffer, BUFFER_SIZE);
    blake2s_final(&state, temp_hash);

    for (int i = 0; i < HASH_SIZE; i++) {
        output_hashes[idx * HASH_SIZE + i] = temp_hash[i];
    }

    uint64_t start_val = 0;
    uint64_t end_val = 0;
    for (int i = 0; i < 8; i++) {
        start_val |= ((uint64_t)temp_hash[i]) << (i * 8);
        end_val |= ((uint64_t)temp_hash[24 + i]) << (i * 8);
    }

    uint32_t leading = __clzll(start_val);
    uint32_t trailing = __clzll(__brevll(end_val));
    output_difficulties[idx] = leading + trailing;
}

typedef struct {
    uint8_t* d_buffers;
    uint8_t* d_input_data;
    uint64_t* d_nonces;
    uint8_t* d_output_hashes;
    uint32_t* d_output_difficulties;
    uint32_t batch_size;
} BlakeoutContext;

extern "C" {
    BlakeoutContext* blakeout_create_context(uint32_t batch_size) {
        BlakeoutContext* ctx = (BlakeoutContext*)malloc(sizeof(BlakeoutContext));
        if (!ctx) return NULL;
        ctx->batch_size = batch_size;

        cudaError_t err;

        // Allocate GPU memory with error checking
        err = cudaMalloc(&ctx->d_buffers, (size_t)batch_size * BUFFER_SIZE);
        if (err != cudaSuccess) {
            free(ctx);
            return NULL;
        }

        err = cudaMalloc(&ctx->d_input_data, 512);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_buffers);
            free(ctx);
            return NULL;
        }

        err = cudaMalloc(&ctx->d_nonces, batch_size * sizeof(uint64_t));
        if (err != cudaSuccess) {
            cudaFree(ctx->d_buffers);
            cudaFree(ctx->d_input_data);
            free(ctx);
            return NULL;
        }

        err = cudaMalloc(&ctx->d_output_hashes, batch_size * HASH_SIZE);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_buffers);
            cudaFree(ctx->d_input_data);
            cudaFree(ctx->d_nonces);
            free(ctx);
            return NULL;
        }

        err = cudaMalloc(&ctx->d_output_difficulties, batch_size * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(ctx->d_buffers);
            cudaFree(ctx->d_input_data);
            cudaFree(ctx->d_nonces);
            cudaFree(ctx->d_output_hashes);
            free(ctx);
            return NULL;
        }

        return ctx;
    }

    void blakeout_destroy_context(BlakeoutContext* ctx) {
        if (ctx) {
            cudaFree(ctx->d_buffers);
            cudaFree(ctx->d_input_data);
            cudaFree(ctx->d_nonces);
            cudaFree(ctx->d_output_hashes);
            cudaFree(ctx->d_output_difficulties);
            free(ctx);
        }
    }

    cudaError_t blakeout_hash_batch_ctx(
        BlakeoutContext* ctx,
        const uint8_t* h_input_data,
        size_t input_len,
        const uint64_t* h_nonces,
        uint32_t nonce_count,
        uint8_t* h_output_hashes,
        uint32_t* h_output_difficulties,
        uint32_t target_difficulty
    ) {
        if (!ctx || nonce_count > ctx->batch_size) return cudaErrorInvalidValue;

        cudaError_t err;

        // Copy input to device (sync is faster for pageable memory)
        err = cudaMemcpy(ctx->d_input_data, h_input_data, input_len, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;

        err = cudaMemcpy(ctx->d_nonces, h_nonces, nonce_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (nonce_count + threadsPerBlock - 1) / threadsPerBlock;

        blakeout_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            ctx->d_input_data, input_len, ctx->d_nonces, nonce_count,
            ctx->d_buffers, ctx->d_output_hashes, ctx->d_output_difficulties, target_difficulty
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) return err;

        // Wait for kernel completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;

        // Copy results back to host
        err = cudaMemcpy(h_output_hashes, ctx->d_output_hashes, nonce_count * HASH_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return err;

        err = cudaMemcpy(h_output_difficulties, ctx->d_output_difficulties, nonce_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return err;

        return cudaSuccess;
    }
}
