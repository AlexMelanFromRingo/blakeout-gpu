#ifndef BLAKE2S_CUH
#define BLAKE2S_CUH

#include <stdint.h>

// Blake2s constants
#define BLAKE2S_BLOCKBYTES 64
#define BLAKE2S_OUTBYTES 32
#define BLAKE2S_KEYBYTES 32
#define BLAKE2S_SALTBYTES 8
#define BLAKE2S_PERSONALBYTES 8

// Blake2s state
typedef struct {
    uint32_t h[8];
    uint32_t t[2];
    uint32_t f[2];
    uint8_t buf[BLAKE2S_BLOCKBYTES];
    size_t buflen;
    size_t outlen;
} blake2s_state;

// Device functions
__device__ void blake2s_init(blake2s_state *S, size_t outlen);
__device__ void blake2s_update(blake2s_state *S, const uint8_t *in, size_t inlen);
__device__ void blake2s_final(blake2s_state *S, uint8_t *out);
__device__ void blake2s(uint8_t *out, const uint8_t *in, size_t inlen);

#endif // BLAKE2S_CUH
