#pragma once
#include <stdint.h>

typedef struct FmDemodCpuContext {
    uint8_t* h_raw;
    float* h_audio;

    float* h_i;
    float* h_q;
    float* h_i1;
    float* h_q1;
    float* h_discrim;
    float* taps1;
    float* taps2;

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
} FmDemodCpuContext;

FmDemodCpuContext* fm_demod_cpu_create(
    int block_size,
    int decim1, const float* taps1, int n_taps1,
    int decim2, const float* taps2, int n_taps2,
    float freq_shift_hz, float sample_rate);

void fm_demod_cpu_destroy(FmDemodCpuContext* ctx);

void fm_demod_cpu_process(FmDemodCpuContext* ctx, const uint8_t* in, float* out);
