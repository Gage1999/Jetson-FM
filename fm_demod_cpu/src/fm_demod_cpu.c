#define _GNU_SOURCE
#include "fm_demod_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void stage_iq_normalize(const uint8_t* in, float* out_i, float* out_q, int n)
{
    for (int i = 0; i < n; i++) {
        // RTL-SDR uses offset binary: 127.5 is the zero point, not 128
        out_i[i] = ((float)in[2*i] - 127.5f) * (1.0f / 127.5f);
        out_q[i] = ((float)in[2*i + 1] - 127.5f) * (1.0f / 127.5f);
    }
}

static void stage_freq_shift(float* io_i, float* io_q, int n, float phase_inc, float phase_start)
{
    for (int i = 0; i < n; i++) {
        float sin_p, cos_p;
        sincosf(phase_start + i * phase_inc, &sin_p, &cos_p);
        float ii = io_i[i], q = io_q[i];
        io_i[i] = ii * cos_p - q * sin_p;
        io_q[i] = ii * sin_p + q * cos_p;
    }
}

static void stage_decimate(const float* in, float* out, const float* taps, int n_taps, int factor, int n_in)
{
    int n_out = n_in / factor;
    int halo = n_taps / 2;
    for (int j = 0; j < n_out; j++) {
        int center = j * factor;
        float acc = 0.0f;
        for (int k = 0; k < n_taps; k++) {
            int idx = center - halo + k;
            if (idx >= 0 && idx < n_in)
                acc += taps[k] * in[idx];
        }
        out[j] = acc;
    }
}

static void stage_fm_discriminate(const float* in_i, const float* in_q, float* out, int n, float prev_i, float prev_q)
{
    float i0 = prev_i, q0 = prev_q;
    for (int i = 0; i < n; i++) {
        float i1 = in_i[i], q1 = in_q[i];
        // Cross-dot form of atan2 phase difference: avoids explicit unwrapping
        out[i] = atan2f(i0 * q1 - q0 * i1, i0 * i1 + q0 * q1);
        i0 = i1; q0 = q1;
    }
}

static void stage_dc_block(float* buf, int n)
{
    float avg = 0.0f;
    for (int i = 0; i < n; i++) avg += buf[i];
    avg /= n;
    for (int i = 0; i < n; i++) buf[i] -= avg;
}

FmDemodCpuContext* fm_demod_cpu_create(
    int block_size,
    int decim1, const float* h_taps1, int n_taps1,
    int decim2, const float* h_taps2, int n_taps2,
    float freq_shift_hz, float sample_rate)
{
    FmDemodCpuContext* ctx = (FmDemodCpuContext*)calloc(1, sizeof(*ctx));
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

    ctx->h_raw = (uint8_t*)malloc(2 * block_size);
    ctx->h_audio = (float*)malloc(ctx->audio_size * sizeof(float));
    ctx->h_i = (float*)malloc(block_size * sizeof(float));
    ctx->h_q = (float*)malloc(block_size * sizeof(float));
    ctx->h_i1 = (float*)malloc(ctx->chan_size * sizeof(float));
    ctx->h_q1 = (float*)malloc(ctx->chan_size * sizeof(float));
    ctx->h_discrim = (float*)malloc(ctx->chan_size * sizeof(float));
    ctx->taps1 = (float*)malloc(n_taps1 * sizeof(float));
    ctx->taps2 = (float*)malloc(n_taps2 * sizeof(float));

    memcpy(ctx->taps1, h_taps1, n_taps1 * sizeof(float));
    memcpy(ctx->taps2, h_taps2, n_taps2 * sizeof(float));

    return ctx;
}

void fm_demod_cpu_destroy(FmDemodCpuContext* ctx)
{
    if (!ctx) return;
    free(ctx->h_raw);
    free(ctx->h_audio);
    free(ctx->h_i);
    free(ctx->h_q);
    free(ctx->h_i1);
    free(ctx->h_q1);
    free(ctx->h_discrim);
    free(ctx->taps1);
    free(ctx->taps2);
    free(ctx);
}

void fm_demod_cpu_process(FmDemodCpuContext* ctx, const uint8_t* in, float* out)
{
    int B = ctx->block_size, C = ctx->chan_size;

    stage_iq_normalize(in, ctx->h_i, ctx->h_q, B);

    if (ctx->phase_inc != 0.0f)
        stage_freq_shift(ctx->h_i, ctx->h_q, B, ctx->phase_inc, ctx->phase);

    stage_decimate(ctx->h_i, ctx->h_i1, ctx->taps1, ctx->n_taps1, ctx->decim1, B);
    stage_decimate(ctx->h_q, ctx->h_q1, ctx->taps1, ctx->n_taps1, ctx->decim1, B);

    stage_fm_discriminate(ctx->h_i1, ctx->h_q1, ctx->h_discrim, C, ctx->last_i, ctx->last_q);

    stage_dc_block(ctx->h_discrim, C);

    stage_decimate(ctx->h_discrim, out, ctx->taps2, ctx->n_taps2, ctx->decim2, C);

    // fmodf keeps the accumulator from losing precision over many blocks
    ctx->phase = fmodf(ctx->phase + B * ctx->phase_inc, 2.0f * (float)M_PI);
    if (C > 0) {
        ctx->last_i = ctx->h_i1[C - 1];
        ctx->last_q = ctx->h_q1[C - 1];
    }
}
