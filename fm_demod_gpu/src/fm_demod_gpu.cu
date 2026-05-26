#include "fm_demod_gpu.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// Kernels

// IQ Normalize: Center IQ at 0
__global__ void kernel_iq_normalize(
    const uint8_t* __restrict__ in,
    float* __restrict__ out_i,
    float* __restrict__ out_q,
    int n_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;
    // RTL-SDR uses offset binary: 127.5 is the zero point, not 128
    out_i[idx] = ((float)in[2 * idx] - 127.5f) * (1.0f / 127.5f);
    out_q[idx] = ((float)in[2 * idx + 1] - 127.5f) * (1.0f / 127.5f);
}

// Frequency Shifter: Shift Frequency from Offset to 0
__global__ void kernel_freq_shift(
    float* __restrict__ io_i,
    float* __restrict__ io_q,
    int n_samples,
    float phase_inc,
    float phase_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float sin_p, cos_p;
    sincosf(phase_start + idx * phase_inc, &sin_p, &cos_p);

    float i = io_i[idx];
    float q = io_q[idx];
    io_i[idx] = i * cos_p - q * sin_p;
    io_q[idx] = i * sin_p + q * cos_p;
}

// FM Discriminator: Turns FM to Audio Signal
__global__ void kernel_fm_discriminate(
    const float* __restrict__ in_i,
    const float* __restrict__ in_q,
    float* __restrict__ out,
    int n_samples,
    float prev_i,
    float prev_q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float i1 = in_i[idx], q1 = in_q[idx];
    float i0 = (idx == 0) ? prev_i : in_i[idx - 1];
    float q0 = (idx == 0) ? prev_q : in_q[idx - 1];

    // Cross-dot form of atan2 phase difference: avoids explicit unwrapping
    float cross = i0 * q1 - q0 * i1;
    float dot   = i0 * i1 + q0 * q1;
    out[idx] = atan2f(cross, dot);
}

// Filter 
__global__ void kernel_decimate(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ taps,
    int n_taps,
    int factor,
    int n_in)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_out = n_in / factor;
    if (out_idx >= n_out) return;

    int center = out_idx * factor;
    float acc = 0.0f;
    for (int k = 0; k < n_taps; k++) {
        int in_idx = center - n_taps / 2 + k;
        if (in_idx >= 0 && in_idx < n_in)
            acc += taps[k] * in[in_idx];
    }
    out[out_idx] = acc;
}

// Remove DC bias from Receiver
__global__ void kernel_dc_block(float* buf, int n, float alpha)
{
    float avg = 0.0f;
    for (int i = 0; i < n; i++) avg += buf[i];
    avg /= n;
    for (int i = 0; i < n; i++) buf[i] -= avg;
}

// Host API
extern "C" FmDemodContext* fm_demod_create(
    int block_size,
    int decim1, const float* h_taps1, int n_taps1,
    int decim2, const float* h_taps2, int n_taps2,
    float freq_shift_hz, float sample_rate)
{
    FmDemodContext* ctx = (FmDemodContext*)calloc(1, sizeof(*ctx));
    ctx->block_size = block_size;
    ctx->chan_size = block_size / decim1;
    ctx->audio_size = ctx->chan_size / decim2;
    ctx->decim1 = decim1;
    ctx->decim2 = decim2;
    ctx->n_taps1 = n_taps1;
    ctx->n_taps2 = n_taps2;
    ctx->phase = 0.0f;
    ctx->phase_inc = 2.0f * (float)M_PI * freq_shift_hz / sample_rate;
    ctx->last_i = 1.0f;
    ctx->last_q = 0.0f;

    CUDA_CHECK(cudaHostAlloc(&ctx->h_raw, 2 * block_size * sizeof(uint8_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&ctx->h_audio, ctx->audio_size * sizeof(float), cudaHostAllocMapped));

    CUDA_CHECK(cudaHostGetDevicePointer((void**)&ctx->d_raw, ctx->h_raw, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&ctx->d_audio, ctx->h_audio, 0));

    CUDA_CHECK(cudaMalloc(&ctx->d_i, block_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_q, block_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ctx->d_i1, ctx->chan_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_q1, ctx->chan_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&ctx->d_discrim, ctx->chan_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_taps1, n_taps1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_taps2, n_taps2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(ctx->d_taps1, h_taps1, n_taps1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_taps2, h_taps2, n_taps2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    // Run all kernels once with zero data to prevent audio underruns on startup
    int threads = 256;
    int B = block_size, C = ctx->chan_size, A = ctx->audio_size;
    cudaStream_t s = ctx->stream;

    CUDA_CHECK(cudaMemsetAsync(ctx->d_raw, 0, 2 * B, s));
    kernel_iq_normalize<<<(B+threads-1)/threads, threads, 0, s>>>(ctx->d_raw, ctx->d_i, ctx->d_q, B);
    kernel_freq_shift<<<(B+threads-1)/threads, threads, 0, s>>>(ctx->d_i, ctx->d_q, B, 0.0f, 0.0f);
    kernel_decimate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_i,  ctx->d_i1, ctx->d_taps1, n_taps1, decim1, B);
    kernel_decimate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_q,  ctx->d_q1, ctx->d_taps1, n_taps1, decim1, B);
    kernel_fm_discriminate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_i1, ctx->d_q1, ctx->d_discrim, C, 1.0f, 0.0f);
    kernel_dc_block<<<1, 1, 0, s>>>(ctx->d_discrim, C, 0.0f);
    kernel_decimate<<<(A+threads-1)/threads, threads, 0, s>>>(ctx->d_discrim, ctx->d_audio, ctx->d_taps2, n_taps2, decim2, C);
    CUDA_CHECK(cudaStreamSynchronize(s));

    return ctx;
}

extern "C" void fm_demod_destroy(FmDemodContext* ctx)
{
    if (!ctx) return;
    cudaFreeHost(ctx->h_raw);
    cudaFreeHost(ctx->h_audio);
    cudaFree(ctx->d_i);
    cudaFree(ctx->d_q);
    cudaFree(ctx->d_i1);
    cudaFree(ctx->d_q1);
    cudaFree(ctx->d_discrim);
    cudaFree(ctx->d_taps1);
    cudaFree(ctx->d_taps2);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
}

extern "C" void fm_demod_process(FmDemodContext* ctx, const uint8_t* h_raw, float* h_audio)
{
    int B = ctx->block_size, C = ctx->chan_size, A = ctx->audio_size;
    cudaStream_t s = ctx->stream;
    int threads = 256;

    // Ensures CPU write to h_raw is visible to GPU
    __sync_synchronize();

    // 1. Normalize IQ
    kernel_iq_normalize<<<(B+threads-1)/threads, threads, 0, s>>>(ctx->d_raw, ctx->d_i, ctx->d_q, B);

    // 2. Frequency shift
    if (ctx->phase_inc != 0.0f)
        kernel_freq_shift<<<(B+threads-1)/threads, threads, 0, s>>>(ctx->d_i, ctx->d_q, B, ctx->phase_inc, ctx->phase);

    // 3. Channel filter + decimate
    kernel_decimate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_i,  ctx->d_i1, ctx->d_taps1, ctx->n_taps1, ctx->decim1, B);
    kernel_decimate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_q,  ctx->d_q1, ctx->d_taps1, ctx->n_taps1, ctx->decim1, B);

    // 4. FM discriminator
    kernel_fm_discriminate<<<(C+threads-1)/threads, threads, 0, s>>>(ctx->d_i1, ctx->d_q1, ctx->d_discrim, C, ctx->last_i, ctx->last_q);

    // 5. DC block
    kernel_dc_block<<<1, 1, 0, s>>>(ctx->d_discrim, C, 0.0f);

    // 6. Audio filter + decimate
    kernel_decimate<<<(A+threads-1)/threads, threads, 0, s>>>(ctx->d_discrim, ctx->d_audio, ctx->d_taps2, ctx->n_taps2, ctx->decim2, C);

    CUDA_CHECK(cudaStreamSynchronize(s));

    // fmodf keeps the accumulator from losing precision over many blocks
    ctx->phase = fmodf(ctx->phase + B * ctx->phase_inc, 2.0f * (float)M_PI);

    // Copy last channel-decimated IQ sample (8 bytes) for next block's discriminator boundary
    if (C > 0) {
        cudaMemcpy(&ctx->last_i, ctx->d_i1 + (C - 1), sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ctx->last_q, ctx->d_q1 + (C - 1), sizeof(float), cudaMemcpyDeviceToHost);
    }
}
