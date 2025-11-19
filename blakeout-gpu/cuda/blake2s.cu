#include "blake2s.cuh"

// Blake2s IV
__constant__ uint32_t blake2s_IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

// Blake2s sigma permutations
__constant__ uint8_t blake2s_sigma[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 15, 4, 5, 8, 0, 10, 6},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

__device__ static inline uint32_t rotr32(uint32_t w, unsigned c) {
    return (w >> c) | (w << (32 - c));
}

__device__ static inline uint32_t load32(const void *src) {
    const uint8_t *p = (const uint8_t *)src;
    return ((uint32_t)(p[0]) << 0) |
           ((uint32_t)(p[1]) << 8) |
           ((uint32_t)(p[2]) << 16) |
           ((uint32_t)(p[3]) << 24);
}

__device__ static inline void store32(void *dst, uint32_t w) {
    uint8_t *p = (uint8_t *)dst;
    p[0] = (uint8_t)(w >> 0);
    p[1] = (uint8_t)(w >> 8);
    p[2] = (uint8_t)(w >> 16);
    p[3] = (uint8_t)(w >> 24);
}

#define G(r, i, a, b, c, d)                     \
    do {                                         \
        a = a + b + m[blake2s_sigma[r][2*i+0]]; \
        d = rotr32(d ^ a, 16);                  \
        c = c + d;                              \
        b = rotr32(b ^ c, 12);                  \
        a = a + b + m[blake2s_sigma[r][2*i+1]]; \
        d = rotr32(d ^ a, 8);                   \
        c = c + d;                              \
        b = rotr32(b ^ c, 7);                   \
    } while (0)

#define ROUND(r)                        \
    do {                                 \
        G(r, 0, v[0], v[4], v[8], v[12]); \
        G(r, 1, v[1], v[5], v[9], v[13]); \
        G(r, 2, v[2], v[6], v[10], v[14]); \
        G(r, 3, v[3], v[7], v[11], v[15]); \
        G(r, 4, v[0], v[5], v[10], v[15]); \
        G(r, 5, v[1], v[6], v[11], v[12]); \
        G(r, 6, v[2], v[7], v[8], v[13]); \
        G(r, 7, v[3], v[4], v[9], v[14]); \
    } while (0)

__device__ static void blake2s_compress(blake2s_state *S, const uint8_t block[BLAKE2S_BLOCKBYTES]) {
    uint32_t m[16];
    uint32_t v[16];
    size_t i;

    for (i = 0; i < 16; ++i) {
        m[i] = load32(block + i * sizeof(m[i]));
    }

    for (i = 0; i < 8; ++i) {
        v[i] = S->h[i];
    }

    v[8] = blake2s_IV[0];
    v[9] = blake2s_IV[1];
    v[10] = blake2s_IV[2];
    v[11] = blake2s_IV[3];
    v[12] = S->t[0] ^ blake2s_IV[4];
    v[13] = S->t[1] ^ blake2s_IV[5];
    v[14] = S->f[0] ^ blake2s_IV[6];
    v[15] = S->f[1] ^ blake2s_IV[7];

    ROUND(0);
    ROUND(1);
    ROUND(2);
    ROUND(3);
    ROUND(4);
    ROUND(5);
    ROUND(6);
    ROUND(7);
    ROUND(8);
    ROUND(9);

    for (i = 0; i < 8; ++i) {
        S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
    }
}

#undef G
#undef ROUND

__device__ void blake2s_init(blake2s_state *S, size_t outlen) {
    size_t i;

    if (outlen == 0 || outlen > BLAKE2S_OUTBYTES) {
        return;
    }

    for (i = 0; i < 8; ++i) {
        S->h[i] = blake2s_IV[i];
    }

    S->h[0] ^= 0x01010000 ^ outlen;

    S->t[0] = 0;
    S->t[1] = 0;
    S->f[0] = 0;
    S->f[1] = 0;
    S->buflen = 0;
    S->outlen = outlen;
}

__device__ void blake2s_update(blake2s_state *S, const uint8_t *in, size_t inlen) {
    if (inlen > 0) {
        size_t left = S->buflen;
        size_t fill = BLAKE2S_BLOCKBYTES - left;

        if (inlen > fill) {
            S->buflen = 0;
            for (size_t i = 0; i < fill; ++i) {
                S->buf[left + i] = in[i];
            }
            S->t[0] += BLAKE2S_BLOCKBYTES;
            if (S->t[0] < BLAKE2S_BLOCKBYTES) {
                S->t[1]++;
            }
            blake2s_compress(S, S->buf);
            in += fill;
            inlen -= fill;

            while (inlen > BLAKE2S_BLOCKBYTES) {
                S->t[0] += BLAKE2S_BLOCKBYTES;
                if (S->t[0] < BLAKE2S_BLOCKBYTES) {
                    S->t[1]++;
                }
                blake2s_compress(S, in);
                in += BLAKE2S_BLOCKBYTES;
                inlen -= BLAKE2S_BLOCKBYTES;
            }
        }

        for (size_t i = 0; i < inlen; ++i) {
            S->buf[S->buflen + i] = in[i];
        }
        S->buflen += inlen;
    }
}

__device__ void blake2s_final(blake2s_state *S, uint8_t *out) {
    uint8_t buffer[BLAKE2S_OUTBYTES] = {0};
    size_t i;

    S->t[0] += S->buflen;
    if (S->t[0] < S->buflen) {
        S->t[1]++;
    }

    S->f[0] = (uint32_t)-1;

    for (i = S->buflen; i < BLAKE2S_BLOCKBYTES; ++i) {
        S->buf[i] = 0;
    }

    blake2s_compress(S, S->buf);

    for (i = 0; i < 8; ++i) {
        store32(buffer + sizeof(S->h[i]) * i, S->h[i]);
    }

    for (i = 0; i < S->outlen; ++i) {
        out[i] = buffer[i];
    }
}

__device__ void blake2s(uint8_t *out, const uint8_t *in, size_t inlen) {
    blake2s_state S;
    blake2s_init(&S, BLAKE2S_OUTBYTES);
    blake2s_update(&S, in, inlen);
    blake2s_final(&S, out);
}
