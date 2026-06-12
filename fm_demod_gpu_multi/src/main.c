#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <rtl-sdr.h>
#include "fm_demod_gpu_multi.cuh"

#define DEFAULT_FREQ_HZ 95100000
#define DEFAULT_SAMPLE_RATE 2048000
#define BLOCK_SIZE 131072
#define FIR_TAPS1 127
#define FIR_TAPS2 127
#define DECIM1 7
#define DECIM2 6
#define CHAN_CUTOFF_HZ 100000
#define AUDIO_CUTOFF_HZ 15000
#define BENCH_REPS 10
#define MAX_CHANNELS 16

static volatile int g_running = 1;
static FmDemodMultiContext* g_ctx = NULL;
static rtlsdr_dev_t* g_dev = NULL;

static FILE* g_out_files[MAX_CHANNELS];
static int g_write_wav = 0;
static size_t g_samples_written[MAX_CHANNELS];
static double g_stop_time = 0.0;

static void on_signal(int sig)
{
    (void)sig;
    g_running = 0;
    if (g_dev) rtlsdr_cancel_async(g_dev);
}

static void write_wav_header(FILE* f, uint32_t sample_rate)
{
    uint32_t zero = 0;
    uint16_t channels = 1, bits = 16, audio_fmt = 1;
    uint32_t fmt_size = 16;
    uint32_t byte_rate = sample_rate * (bits / 8);
    uint16_t block_align = (uint16_t)(bits / 8);

    fwrite("RIFF", 1, 4, f); fwrite(&zero, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f); fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&zero, 4, 1, f);
}

static void finalize_wav_header(FILE* f, size_t n_samples)
{
    uint32_t data_size = (uint32_t)(n_samples * sizeof(int16_t));
    uint32_t riff_size = 36 + data_size;
    fseek(f, 4, SEEK_SET); fwrite(&riff_size, 4, 1, f);
    fseek(f, 40, SEEK_SET); fwrite(&data_size, 4, 1, f);
}

static int parse_offsets(const char* s, float* offsets, int max_n)
{
    char buf[512];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    int n = 0;
    char* tok = strtok(buf, ",");
    while (tok && n < max_n) {
        offsets[n++] = (float)atoi(tok);
        tok = strtok(NULL, ",");
    }
    return n;
}

static void write_audio_block(int n_channels, int audio_size, uint32_t audio_rate)
{
    (void)audio_rate;
    for (int k = 0; k < n_channels; k++) {
        if (!g_out_files[k]) continue;
        float* audio = g_ctx->channels[k].h_audio;
        for (int i = 0; i < audio_size; i++) {
            float s = audio[i] * 32767.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            int16_t sample = (int16_t)s;
            fwrite(&sample, sizeof(int16_t), 1, g_out_files[k]);
        }
        g_samples_written[k] += (size_t)audio_size;
    }
}

static void rtlsdr_callback(unsigned char* buf, uint32_t len, void* userctx)
{
    (void)userctx;
    if (!g_running) return;

    if (g_stop_time > 0.0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double now = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (now >= g_stop_time) {
            g_running = 0;
            rtlsdr_cancel_async(g_dev);
            return;
        }
    }

    int n_samples = (int)len / 2;
    if (n_samples > g_ctx->block_size) n_samples = g_ctx->block_size;

    memcpy(g_ctx->h_raw, buf, 2 * n_samples);
    fm_demod_multi_process(g_ctx);

    if (g_write_wav)
        write_audio_block(g_ctx->n_channels, g_ctx->audio_size, 0);
}

static void make_lowpass_taps(float* taps, int n_taps, float cutoff)
{
    int half = n_taps / 2;
    float sum = 0.0f;
    for (int i = 0; i < n_taps; i++) {
        int n = i - half;
        float sinc = (n == 0) ? 2.0f * cutoff
                               : sinf(2.0f * (float)M_PI * cutoff * n) / ((float)M_PI * n);
        float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (n_taps - 1));
        taps[i] = sinc * hann;
        sum += taps[i];
    }
    for (int i = 0; i < n_taps; i++) taps[i] /= sum;
}

static void run_bench(FmDemodMultiContext* ctx, const char* path, uint32_t samp_rate)
{
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    int block_bytes = 2 * ctx->block_size;
    int n_blocks = (int)(file_size / block_bytes);
    if (n_blocks == 0) { fprintf(stderr, "bench: file too small\n"); fclose(f); return; }

    uint8_t* buf = (uint8_t*)malloc((size_t)n_blocks * block_bytes);
    if (fread(buf, 1, (size_t)n_blocks * block_bytes, f) == 0) {
        fprintf(stderr, "bench: read error\n"); free(buf); fclose(f); return;
    }
    fclose(f);

    // Warm-up pass
    memcpy(ctx->h_raw, buf, block_bytes);
    fm_demod_multi_process(ctx);

    struct rusage ru0, ru1;
    struct timespec t0, t1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int rep = 0; rep < BENCH_REPS; rep++) {
        for (int b = 0; b < n_blocks; b++) {
            memcpy(ctx->h_raw, buf + b * block_bytes, block_bytes);
            fm_demod_multi_process(ctx);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    getrusage(RUSAGE_SELF, &ru1);
    free(buf);

    long total = (long)BENCH_REPS * n_blocks;
    double wall_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double per_ms = wall_ms / total;
    double period_ms = (double)ctx->block_size / (double)samp_rate * 1e3;
    double rt = period_ms / per_ms;
    double cpu_us = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) * 1e6
                  + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec)
                  + (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) * 1e6
                  + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec);
    double cpu_pct = cpu_us / (wall_ms * 1e3) * 100.0;

    fprintf(stderr,
        "channels: %d  blocks: %ld  reps: %d  wall: %.1f ms"
        "  per-block: %.2f ms  real-time: %.1fx  cpu: %.1f%%\n",
        ctx->n_channels, total, BENCH_REPS, wall_ms, per_ms, rt, cpu_pct);
}

int main(int argc, char** argv)
{
    uint32_t freq = DEFAULT_FREQ_HZ;
    uint32_t samp_rate = DEFAULT_SAMPLE_RATE;
    const char* out_prefix = NULL;
    const char* bench_path = NULL;
    double duration = 0.0;

    float offsets[MAX_CHANNELS] = {0.0f};
    int n_channels = 1;

    int opt;
    while ((opt = getopt(argc, argv, "f:r:c:o:t:b:")) != -1) {
        switch (opt) {
        case 'f': freq = (uint32_t)atoi(optarg); break;
        case 'r': samp_rate = (uint32_t)atoi(optarg); break;
        case 'c': n_channels = parse_offsets(optarg, offsets, MAX_CHANNELS); break;
        case 'o': out_prefix = optarg; break;
        case 't': duration = atof(optarg); break;
        case 'b': bench_path = optarg; break;
        default:
            fprintf(stderr,
                "Usage: fm_rx_gpu_multi [-f freq_hz] [-r sample_rate]"
                " [-c off1,off2,...] [-o prefix] [-t sec] [-b file.bin]\n"
                "  -c  Comma-separated Hz offsets from center (default: 0)\n"
                "  -o  Output prefix; writes <prefix>_ch<N>.wav per channel\n");
            return 1;
        }
    }

    int chan_rate = (int)samp_rate / DECIM1;
    int audio_rate_actual = chan_rate / DECIM2;

    float taps1[FIR_TAPS1], taps2[FIR_TAPS2];
    make_lowpass_taps(taps1, FIR_TAPS1, (float)CHAN_CUTOFF_HZ / (float)samp_rate);
    make_lowpass_taps(taps2, FIR_TAPS2, (float)AUDIO_CUTOFF_HZ / (float)chan_rate);

    fprintf(stderr, "Warming up GPU (%d channel%s)...\n",
            n_channels, n_channels == 1 ? "" : "s");
    FmDemodMultiContext* ctx = fm_demod_multi_create(
        BLOCK_SIZE,
        DECIM1, taps1, FIR_TAPS1,
        DECIM2, taps2, FIR_TAPS2,
        (float)samp_rate, offsets, n_channels);
    fprintf(stderr, "GPU ready.\n");

    // Bench 
    if (bench_path) {
        if (out_prefix) {
            FILE* fin = fopen(bench_path, "rb");
            if (!fin) { perror(bench_path); return 1; }

            FILE* fout[MAX_CHANNELS];
            size_t n_written[MAX_CHANNELS];
            memset(n_written, 0, sizeof(n_written));

            for (int k = 0; k < n_channels; k++) {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s_ch%d.wav", out_prefix, k);
                fout[k] = fopen(fname, "wb");
                if (!fout[k]) { perror(fname); return 1; }
                write_wav_header(fout[k], (uint32_t)audio_rate_actual);
                fprintf(stderr, "  ch%d (%+.0f Hz): %s\n", k, (double)offsets[k], fname);
            }

            uint8_t* blk = (uint8_t*)malloc(2 * BLOCK_SIZE);
            while (fread(blk, 1, 2 * BLOCK_SIZE, fin) == (size_t)(2 * BLOCK_SIZE)) {
                memcpy(ctx->h_raw, blk, 2 * BLOCK_SIZE);
                fm_demod_multi_process(ctx);
                for (int k = 0; k < n_channels; k++) {
                    float* audio = ctx->channels[k].h_audio;
                    for (int i = 0; i < ctx->audio_size; i++) {
                        float s = audio[i] * 32767.0f;
                        if (s > 32767.0f) s = 32767.0f;
                        if (s < -32768.0f) s = -32768.0f;
                        int16_t sample = (int16_t)s;
                        fwrite(&sample, sizeof(int16_t), 1, fout[k]);
                    }
                    n_written[k] += (size_t)ctx->audio_size;
                }
            }
            free(blk);
            fclose(fin);

            for (int k = 0; k < n_channels; k++) {
                finalize_wav_header(fout[k], n_written[k]);
                fclose(fout[k]);
                fprintf(stderr, "  ch%d: wrote %.1f sec\n",
                        k, (double)n_written[k] / audio_rate_actual);
            }

            // Reset channel state before benchmark timing pass
            for (int k = 0; k < n_channels; k++) {
                ctx->channels[k].phase = 0.0f;
                ctx->channels[k].last_i = 1.0f;
                ctx->channels[k].last_q = 0.0f;
            }
        }

        fprintf(stderr, "Running bench...\n");
        run_bench(ctx, bench_path, samp_rate);
        fm_demod_multi_destroy(ctx);
        return 0;
    }

    // Live 
    if (out_prefix) {
        g_write_wav = 1;
        for (int k = 0; k < n_channels; k++) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s_ch%d.wav", out_prefix, k);
            g_out_files[k] = fopen(fname, "wb");
            if (!g_out_files[k]) { perror(fname); return 1; }
            write_wav_header(g_out_files[k], (uint32_t)audio_rate_actual);
            fprintf(stderr, "  ch%d (%+.0f Hz): %s\n", k, (double)offsets[k], fname);
        }
    }

    fprintf(stderr,
        "Center: %.3f MHz  |  Fs: %u  |  Chan: %d Hz  |  Audio: %d Hz\n",
        freq / 1e6, samp_rate, chan_rate, audio_rate_actual);
    for (int k = 0; k < n_channels; k++)
        fprintf(stderr, "  ch%d: %+.0f Hz offset  (%.3f MHz)\n",
                k, (double)offsets[k], (freq + offsets[k]) / 1e6);
    if (duration > 0.0)
        fprintf(stderr, "Recording %.1f seconds...\n", duration);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    g_ctx = ctx;
    int r = rtlsdr_open(&g_dev, 0);
    if (r < 0) { fprintf(stderr, "Failed to open RTL-SDR: %d\n", r); return 1; }

    rtlsdr_set_sample_rate(g_dev, samp_rate);
    rtlsdr_set_center_freq(g_dev, freq);
    rtlsdr_set_tuner_gain_mode(g_dev, 0);
    rtlsdr_reset_buffer(g_dev);

    if (duration > 0.0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        g_stop_time = ts.tv_sec + ts.tv_nsec * 1e-9 + duration;
    }

    fprintf(stderr, "Receiving... Ctrl-C to stop.\n");
    rtlsdr_read_async(g_dev, rtlsdr_callback, NULL, 0, 2 * BLOCK_SIZE);

    rtlsdr_close(g_dev);
    fm_demod_multi_destroy(ctx);

    if (g_write_wav) {
        for (int k = 0; k < n_channels; k++) {
            if (!g_out_files[k]) continue;
            finalize_wav_header(g_out_files[k], g_samples_written[k]);
            fclose(g_out_files[k]);
            fprintf(stderr, "  ch%d: saved %zu samples (%.1f sec)\n",
                    k, g_samples_written[k],
                    (double)g_samples_written[k] / audio_rate_actual);
        }
    }

    return 0;
}
