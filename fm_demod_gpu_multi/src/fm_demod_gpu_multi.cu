#include "fm_demod_gpu_multi.cuh"
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

static inline int decimate_smem(int threads, int n_taps, int factor)
{
    return ((threads - 1) * factor + 2 * n_taps) * (int)sizeof(float);
}

static inline int decimate_iq_smem(int threads, int n_taps, int factor)
{
    return (2 * ((threads - 1) * factor + n_taps) + n_taps) * (int)sizeof(float);
}

// Kernels

__global__ void kernel_normalize_and_shift(
    const uint8_t* __restrict__ in,
    float* __restrict__ out_i,
    float* __restrict__ out_q,
    int n_samples,
    float phase_inc,
    float phase_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    float i = ((float)in[2 * idx] - 127.5f) * (1.0f / 127.5f);
    float q = ((float)in[2 * idx + 1] - 127.5f) * (1.0f / 127.5f);

    float sin_p, cos_p;
    sincosf(phase_start + idx * phase_inc, &sin_p, &cos_p);
    out_i[idx] = i * cos_p - q * sin_p;
    out_q[idx] = i * sin_p + q * cos_p;
}

__global__ void kernel_decimate(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ taps,
    int n_taps,
    int factor,
    int n_in)
{
    extern __shared__ float smem[];
    int halo = n_taps / 2;
    int tile_size = (blockDim.x - 1) * factor + n_taps;
    float* s_in = smem;
    float* s_taps = smem + tile_size;

    for (int k = threadIdx.x; k < n_taps; k += blockDim.x)
        s_taps[k] = taps[k];

    int in_base = (int)(blockIdx.x * blockDim.x) * factor - halo;
    for (int k = (int)threadIdx.x; k < tile_size; k += (int)blockDim.x) {
        int gi = in_base + k;
        s_in[k] = (gi >= 0 && gi < n_in) ? in[gi] : 0.0f;
    }
    __syncthreads();

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n_in / factor) return;

    int local_start = threadIdx.x * factor;
    float acc = 0.0f;
    for (int k = 0; k < n_taps; k++)
        acc += s_taps[k] * s_in[local_start + k];
    out[out_idx] = acc;
}

__global__ void kernel_decimate_iq(
    const float* __restrict__ in_i,
    const float* __restrict__ in_q,
    float* __restrict__ out_i,
    float* __restrict__ out_q,
    const float* __restrict__ taps,
    int n_taps,
    int factor,
    int n_in)
{
    extern __shared__ float smem[];
    int halo = n_taps / 2;
    int tile_size = (blockDim.x - 1) * factor + n_taps;
    float* s_i = smem;
    float* s_q = smem + tile_size;
    float* s_taps = smem + 2 * tile_size;

    for (int k = threadIdx.x; k < n_taps; k += blockDim.x)
        s_taps[k] = taps[k];

    int in_base = (int)(blockIdx.x * blockDim.x) * factor - halo;
    for (int k = (int)threadIdx.x; k < tile_size; k += (int)blockDim.x) {
        int gi = in_base + k;
        if (gi >= 0 && gi < n_in) {
            s_i[k] = in_i[gi];
            s_q[k] = in_q[gi];
        } else {
            s_i[k] = 0.0f;
            s_q[k] = 0.0f;
        }
    }
    __syncthreads();

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n_in / factor) return;

    int local_start = threadIdx.x * factor;
    float acc_i = 0.0f, acc_q = 0.0f;
    for (int k = 0; k < n_taps; k++) {
        float t = s_taps[k];
        acc_i += t * s_i[local_start + k];
        acc_q += t * s_q[local_start + k];
    }
    out_i[out_idx] = acc_i;
    out_q[out_idx] = acc_q;
}

__global__ void kernel_fm_discriminate(
    const float* __restrict__ in_i,
    const float* __restrict__ in_q,
    float* __restrict__ out,
    int n_samples,
    float prev_i,
    float prev_q)
{
    __shared__ float s_i[THREADS + 1];
    __shared__ float s_q[THREADS + 1];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        s_i[0] = (blockIdx.x == 0) ? prev_i : in_i[blockIdx.x * blockDim.x - 1];
        s_q[0] = (blockIdx.x == 0) ? prev_q : in_q[blockIdx.x * blockDim.x - 1];
    }
    s_i[threadIdx.x + 1] = (idx < n_samples) ? in_i[idx] : 0.0f;
    s_q[threadIdx.x + 1] = (idx < n_samples) ? in_q[idx] : 0.0f;
    __syncthreads();

    if (idx >= n_samples) return;

    float i1 = s_i[threadIdx.x + 1], q1 = s_q[threadIdx.x + 1];
    float i0 = s_i[threadIdx.x], q0 = s_q[threadIdx.x];
    out[idx] = atan2f(i0 * q1 - q0 * i1, i0 * i1 + q0 * q1);
}

__global__ void kernel_dc_reduce(const float* __restrict__ in, float* sum, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float val = 0.0f;
    if (i < n) val = in[i];
    if (i + blockDim.x < n) val += in[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(sum, sdata[0]);
}

__global__ void kernel_dc_subtract(float* buf, const float* sum, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    buf[idx] -= *sum / (float)n;
}

// Host API

extern "C" FmDemodMultiContext* fm_demod_multi_create(
    int block_size,
    int decim1, const float* h_taps1, int n_taps1,
    int decim2, const float* h_taps2, int n_taps2,
    float sample_rate,
    const float* freq_offsets_hz,
    int n_channels)
{
    FmDemodMultiContext* ctx = (FmDemodMultiContext*)calloc(1, sizeof(*ctx));
    ctx->block_size = block_size;
    ctx->chan_size = block_size / decim1;
    ctx->audio_size = ctx->chan_size / decim2;
    ctx->decim1 = decim1;
    ctx->decim2 = decim2;
    ctx->n_taps1 = n_taps1;
    ctx->n_taps2 = n_taps2;
    ctx->n_channels = n_channels;

    CUDA_CHECK(cudaHostAlloc(&ctx->h_raw, 2 * block_size * sizeof(uint8_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&ctx->d_raw, ctx->h_raw, 0));

    CUDA_CHECK(cudaMalloc(&ctx->d_taps1, n_taps1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_taps2, n_taps2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(ctx->d_taps1, h_taps1, n_taps1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_taps2, h_taps2, n_taps2 * sizeof(float), cudaMemcpyHostToDevice));

    // Per-channel allocations
    ctx->channels = (FmDemodChannel*)calloc(n_channels, sizeof(FmDemodChannel));
    for (int k = 0; k < n_channels; k++) {
        FmDemodChannel* ch = &ctx->channels[k];

        CUDA_CHECK(cudaMalloc(&ch->d_i, block_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ch->d_q, block_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ch->d_i1, ctx->chan_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ch->d_q1, ctx->chan_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ch->d_discrim, ctx->chan_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ch->d_dc_sum, sizeof(float)));

        CUDA_CHECK(cudaHostAlloc(&ch->h_audio, ctx->audio_size * sizeof(float), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&ch->d_audio, ch->h_audio, 0));

        ch->phase = 0.0f;
        ch->phase_inc = -2.0f * (float)M_PI * freq_offsets_hz[k] / sample_rate;
        ch->last_i = 1.0f;
        ch->last_q = 0.0f;

        CUDA_CHECK(cudaStreamCreate(&ch->stream));
    }

    int B = block_size, C = ctx->chan_size, A = ctx->audio_size;
    int smem_iq1 = decimate_iq_smem(THREADS, n_taps1, decim1);
    int smem2 = decimate_smem(THREADS, n_taps2, decim2);
    int dc_blocks = (C + THREADS * 2 - 1) / (THREADS * 2);

    CUDA_CHECK(cudaMemset(ctx->d_raw, 0, 2 * B));

    for (int k = 0; k < n_channels; k++) {
        FmDemodChannel* ch = &ctx->channels[k];
        cudaStream_t s = ch->stream;

        CUDA_CHECK(cudaMemsetAsync(ch->d_dc_sum, 0, sizeof(float), s));
        kernel_normalize_and_shift<<<(B+THREADS-1)/THREADS, THREADS, 0, s>>>(ctx->d_raw, ch->d_i, ch->d_q, B, 0.0f, 0.0f);
        kernel_decimate_iq<<<(C+THREADS-1)/THREADS, THREADS, smem_iq1, s>>>(ch->d_i, ch->d_q, ch->d_i1, ch->d_q1, ctx->d_taps1, n_taps1, decim1, B);
        kernel_fm_discriminate<<<(C+THREADS-1)/THREADS, THREADS, 0, s>>>(ch->d_i1, ch->d_q1, ch->d_discrim, C, 1.0f, 0.0f);
        kernel_dc_reduce<<<dc_blocks, THREADS, THREADS * sizeof(float), s>>>(ch->d_discrim, ch->d_dc_sum, C);
        kernel_dc_subtract<<<(C+THREADS-1)/THREADS, THREADS, 0, s>>>(ch->d_discrim, ch->d_dc_sum, C);
        kernel_decimate<<<(A+THREADS-1)/THREADS, THREADS, smem2, s>>>(ch->d_discrim, ch->d_audio, ctx->d_taps2, n_taps2, decim2, C);
    }
    for (int k = 0; k < n_channels; k++)
        CUDA_CHECK(cudaStreamSynchronize(ctx->channels[k].stream));

    return ctx;
}

extern "C" void fm_demod_multi_destroy(FmDemodMultiContext* ctx)
{
    if (!ctx) return;
    cudaFreeHost(ctx->h_raw);
    cudaFree(ctx->d_taps1);
    cudaFree(ctx->d_taps2);
    for (int k = 0; k < ctx->n_channels; k++) {
        FmDemodChannel* ch = &ctx->channels[k];
        cudaFree(ch->d_i);
        cudaFree(ch->d_q);
        cudaFree(ch->d_i1);
        cudaFree(ch->d_q1);
        cudaFree(ch->d_discrim);
        cudaFree(ch->d_dc_sum);
        cudaFreeHost(ch->h_audio);
        cudaStreamDestroy(ch->stream);
    }
    free(ctx->channels);
    free(ctx);
}

extern "C" void fm_demod_multi_process(FmDemodMultiContext* ctx)
{
    int B = ctx->block_size, C = ctx->chan_size, A = ctx->audio_size;
    int smem_iq1 = decimate_iq_smem(THREADS, ctx->n_taps1, ctx->decim1);
    int smem2 = decimate_smem(THREADS, ctx->n_taps2, ctx->decim2);
    int dc_blocks = (C + THREADS * 2 - 1) / (THREADS * 2);

    for (int k = 0; k < ctx->n_channels; k++) {
        FmDemodChannel* ch = &ctx->channels[k];
        cudaStream_t s = ch->stream;

        kernel_normalize_and_shift<<<(B+THREADS-1)/THREADS, THREADS, 0, s>>>(ctx->d_raw, ch->d_i, ch->d_q, B, ch->phase_inc, ch->phase);
        kernel_decimate_iq<<<(C+THREADS-1)/THREADS, THREADS, smem_iq1, s>>>(ch->d_i, ch->d_q, ch->d_i1, ch->d_q1, ctx->d_taps1, ctx->n_taps1, ctx->decim1, B);
        kernel_fm_discriminate<<<(C+THREADS-1)/THREADS, THREADS, 0, s>>>(ch->d_i1, ch->d_q1, ch->d_discrim, C, ch->last_i, ch->last_q);
        CUDA_CHECK(cudaMemsetAsync(ch->d_dc_sum, 0, sizeof(float), s));
        kernel_dc_reduce<<<dc_blocks, THREADS, THREADS * sizeof(float), s>>>(ch->d_discrim, ch->d_dc_sum, C);
        kernel_dc_subtract<<<(C+THREADS-1)/THREADS, THREADS, 0, s>>>(ch->d_discrim, ch->d_dc_sum, C);
        kernel_decimate<<<(A+THREADS-1)/THREADS, THREADS, smem2, s>>>(ch->d_discrim, ch->d_audio, ctx->d_taps2, ctx->n_taps2, ctx->decim2, C);
    }

    for (int k = 0; k < ctx->n_channels; k++)
        CUDA_CHECK(cudaStreamSynchronize(ctx->channels[k].stream));

    // Update phase accumulators and fetch discriminator boundary samples
    for (int k = 0; k < ctx->n_channels; k++) {
        FmDemodChannel* ch = &ctx->channels[k];
        ch->phase = fmodf(ch->phase + B * ch->phase_inc, 2.0f * (float)M_PI);
        if (C > 0) {
            CUDA_CHECK(cudaMemcpy(&ch->last_i, ch->d_i1 + (C - 1), sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&ch->last_q, ch->d_q1 + (C - 1), sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
}
