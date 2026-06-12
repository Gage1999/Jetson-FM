#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS 256

__global__ void kernel_normalize_and_shift(
    const uint8_t* __restrict__ in,
    float* __restrict__ out_i,
    float* __restrict__ out_q,
    int n_samples,
    float phase_inc,
    float phase_start);

__global__ void kernel_decimate(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ taps,
    int n_taps,
    int factor,
    int n_in);

__global__ void kernel_decimate_iq(
    const float* __restrict__ in_i,
    const float* __restrict__ in_q,
    float* __restrict__ out_i,
    float* __restrict__ out_q,
    const float* __restrict__ taps,
    int n_taps,
    int factor,
    int n_in);

__global__ void kernel_fm_discriminate(
    const float* __restrict__ in_i,
    const float* __restrict__ in_q,
    float* __restrict__ out,
    int n_samples,
    float prev_i,
    float prev_q);

__global__ void kernel_dc_reduce(
    const float* __restrict__ in,
    float* sum,
    int n);

__global__ void kernel_dc_subtract(float* buf, const float* sum, int n);

typedef struct FmDemodChannel {
    float* d_i;
    float* d_q;
    float* d_i1;
    float* d_q1;
    float* d_discrim;
    float* d_dc_sum;
    float* d_audio;
    float* h_audio;

    float phase;
    float phase_inc;
    float last_i;
    float last_q;

    cudaStream_t stream;
} FmDemodChannel;

typedef struct FmDemodMultiContext {
    uint8_t* h_raw;
    uint8_t* d_raw;

    float* d_taps1;
    float* d_taps2;

    FmDemodChannel* channels;
    int n_channels;

    int block_size;
    int chan_size;
    int audio_size;
    int decim1;
    int decim2;
    int n_taps1;
    int n_taps2;
} FmDemodMultiContext;

#ifdef __cplusplus
extern "C" {
#endif

FmDemodMultiContext* fm_demod_multi_create(
    int block_size,
    int decim1, const float* h_taps1, int n_taps1,
    int decim2, const float* h_taps2, int n_taps2,
    float sample_rate,
    const float* freq_offsets_hz,
    int n_channels);

void fm_demod_multi_destroy(FmDemodMultiContext* ctx);

void fm_demod_multi_process(FmDemodMultiContext* ctx);

#ifdef __cplusplus
}
#endif
