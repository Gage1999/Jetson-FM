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

typedef struct FmDemodContext {
    uint8_t* h_raw;
    float* h_audio;
    uint8_t* d_raw;
    float* d_audio;

    float* d_i;
    float* d_q;
    float* d_i1;
    float* d_q1;
    float* d_discrim;
    float* d_taps1;
    float* d_taps2;
    float* d_dc_sum;

    int block_size;
    int chan_size;
    int audio_size;
    int decim1;
    int decim2;
    int n_taps1;
    int n_taps2;

    float phase;
    float phase_inc;
    float last_i;
    float last_q;

    cudaStream_t stream;
} FmDemodContext;

#ifdef __cplusplus
extern "C" {
#endif

FmDemodContext* fm_demod_create(
    int block_size,
    int decim1, const float* h_taps1, int n_taps1,
    int decim2, const float* h_taps2, int n_taps2,
    float freq_shift_hz, float sample_rate);

void fm_demod_destroy(FmDemodContext* ctx);

void fm_demod_process(FmDemodContext* ctx);

#ifdef __cplusplus
}
#endif
