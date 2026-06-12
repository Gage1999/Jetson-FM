#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <rtl-sdr.h>
#include "fm_demod_cpu.h"

#define DEFAULT_FREQ_HZ 95100000   // 95.1 MHz
#define DEFAULT_SAMPLE_RATE 2048000
#define BLOCK_SIZE 131072   // IQ pairs per RTL-SDR callback
#define FIR_TAPS1 127
#define FIR_TAPS2 127
#define DECIM1 7   // 2048000 / 7 = ~293 ksps (channel rate)
#define DECIM2 6   // ~293 ksps / 6 = ~48762 Hz (audio rate)
#define CHAN_CUTOFF_HZ 100000   // rejects adjacent FM stations (100kHz)
#define AUDIO_CUTOFF_HZ 15000   // FM mono audio bandwidth (15kHz)
#define BENCH_REPS 10

static volatile int running = 1;
static FmDemodCpuContext* g_ctx = NULL;
static rtlsdr_dev_t* r_dev = NULL;

static FILE* g_out_file = NULL;
static int r_write_wav = 0;
static size_t r_samples_written = 0;
static double r_stop_time = 0.0;

static void on_signal(int sig)
{
    (void)sig;
    running = 0;
    if (r_dev) rtlsdr_cancel_async(r_dev);
}

// WAV header with placeholder sizes filled in by finalize_wav_header()
static void write_wav_header(FILE* f, uint32_t sample_rate)
{
    uint32_t zero = 0;
    uint16_t channels = 1, bits = 16, audio_fmt = 1;
    uint32_t fmt_size = 16;
    uint32_t byte_rate = sample_rate * channels * (bits / 8);
    uint16_t block_align = (uint16_t)(channels * (bits / 8));

    fwrite("RIFF", 1, 4, f);
    fwrite(&zero, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&zero, 4, 1, f);
}

static void finalize_wav_header(FILE* f, size_t n_samples)
{
    uint32_t data_size = (uint32_t)(n_samples * sizeof(int16_t));
    uint32_t riff_size = 36 + data_size;
    fseek(f, 4, SEEK_SET); fwrite(&riff_size, 4, 1, f);
    fseek(f, 40, SEEK_SET); fwrite(&data_size, 4, 1, f);
}

static void rtlsdr_callback(unsigned char* buf, uint32_t len, void* userctx)
{
    (void)userctx;
    if (!running) return;

    if (r_stop_time > 0.0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double now = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (now >= r_stop_time) {
            running = 0;
            rtlsdr_cancel_async(r_dev);
            return;
        }
    }

    int n_samples = (int)len / 2;
    if (n_samples > g_ctx->block_size) n_samples = g_ctx->block_size;
    int audio_out = g_ctx->audio_size;

    memcpy(g_ctx->h_raw, buf, 2 * n_samples);
    fm_demod_cpu_process(g_ctx, g_ctx->h_raw, g_ctx->h_audio);

    for (int i = 0; i < audio_out; i++) {
        float s = g_ctx->h_audio[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        int16_t sample = (int16_t)s;
        fwrite(&sample, sizeof(int16_t), 1, g_out_file);
    }
    r_samples_written += (size_t)audio_out;
}

// Sinc-windowed low-pass FIR (Hann window). cutoff = f_c / Fs.
static void make_lowpass_taps(float* taps, int n_taps, float cutoff)
{
    int half = n_taps / 2;
    float sum = 0.0f;
    for (int i = 0; i < n_taps; i++) {
        int n = i - half;
        float sinc = (n == 0) ? 2.0f * cutoff : sinf(2.0f * (float)M_PI * cutoff * n) / ((float)M_PI * n);
        float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (n_taps - 1));
        taps[i] = sinc * hann;
        sum += taps[i];
    }
    for (int i = 0; i < n_taps; i++) taps[i] /= sum;
}

static void run_bench(FmDemodCpuContext* ctx, const char* path, uint32_t samp_rate)
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
        fprintf(stderr, "bench: read error\n"); free(buf); return;
    }
    fclose(f);

    memcpy(ctx->h_raw, buf, block_bytes);
    fm_demod_cpu_process(ctx, ctx->h_raw, ctx->h_audio);

    struct rusage ru0, ru1;
    struct timespec t0, t1;
    getrusage(RUSAGE_SELF, &ru0);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int rep = 0; rep < BENCH_REPS; rep++) {
        for (int b = 0; b < n_blocks; b++) {
            memcpy(ctx->h_raw, buf + b * block_bytes, block_bytes);
            fm_demod_cpu_process(ctx, ctx->h_raw, ctx->h_audio);
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

    fprintf(stderr, "blocks: %ld  reps: %d  wall: %.1f ms  per-block: %.2f ms  real-time: %.1fx  cpu: %.1f%%\n",
            total, BENCH_REPS, wall_ms, per_ms, rt, cpu_pct);
}

int main(int argc, char** argv)
{
    uint32_t freq = DEFAULT_FREQ_HZ;
    uint32_t samp_rate = DEFAULT_SAMPLE_RATE;
    int freq_offset = 0;
    const char* out_path = NULL;
    const char* bench_path = NULL;
    double duration = 0.0;

    int opt;
    while ((opt = getopt(argc, argv, "f:r:s:o:t:b:")) != -1) {
        switch (opt) {
        case 'f': freq = (uint32_t)atoi(optarg); break;
        case 'r': samp_rate = (uint32_t)atoi(optarg); break;
        case 's': freq_offset = atoi(optarg); break;
        case 'o': out_path = optarg; break;
        case 't': duration = atof(optarg); break;
        case 'b': bench_path = optarg; break;
        default:
            fprintf(stderr, "Usage: fm_rx_cpu [-f freq_hz] [-r sample_rate] [-s offset_hz] [-o output.wav] [-t duration_sec] [-b bench_file.bin]\n");
            return 1;
        }
    }

    int chan_rate = (int)samp_rate / DECIM1;
    int audio_rate_actual = chan_rate / DECIM2;

    if (bench_path) {
        float taps1[FIR_TAPS1], taps2[FIR_TAPS2];
        make_lowpass_taps(taps1, FIR_TAPS1, (float)CHAN_CUTOFF_HZ / (float)samp_rate);
        make_lowpass_taps(taps2, FIR_TAPS2, (float)AUDIO_CUTOFF_HZ / (float)chan_rate);
        FmDemodCpuContext* ctx = fm_demod_cpu_create(BLOCK_SIZE, DECIM1, taps1, FIR_TAPS1,
                                                      DECIM2, taps2, FIR_TAPS2,
                                                      (float)freq_offset, (float)samp_rate);

        if (out_path){
            FILE* fin = fopen(bench_path, "rb");
            if (!fin) { perror(bench_path); return 1; }
            FILE* fout = fopen(out_path, "wb");
            if (!fout) { perror(out_path); fclose(fin); return 1; }
            write_wav_header(fout, (uint32_t)audio_rate_actual);
            fprintf(stderr, "Decoding to WAV: %s\n", out_path);

            uint8_t* blk = (uint8_t*)malloc(2 * BLOCK_SIZE);
            size_t n_written = 0;
            while (fread(blk, 1, 2 * BLOCK_SIZE, fin) == (size_t)(2 * BLOCK_SIZE)) {
                memcpy(ctx->h_raw, blk, 2 * BLOCK_SIZE);
                fm_demod_cpu_process(ctx, ctx->h_raw, ctx->h_audio);
                for (int i = 0; i < ctx->audio_size; i++) {
                    float s = ctx->h_audio[i] * 32767.0f;
                    if (s > 32767.0f) s = 32767.0f;
                    if (s < -32768.0f) s = -32768.0f;
                    int16_t sample = (int16_t)s;
                    fwrite(&sample, sizeof(int16_t), 1, fout);
                }
                n_written += (size_t)ctx->audio_size;
            }
            free(blk);
            fclose(fin);
            finalize_wav_header(fout, n_written);
            fclose(fout);
            fprintf(stderr, "Wrote %.1f sec to: %s\n",
                    (double)n_written / audio_rate_actual, out_path);

            ctx->phase = 0.0f;
            ctx->last_i = 1.0f;
            ctx->last_q = 0.0f;
        }

        fprintf(stderr, "Running bench...\n");
        run_bench(ctx, bench_path, samp_rate);
        fm_demod_cpu_destroy(ctx);
        return 0;
    }

    if (out_path) {
        g_out_file = fopen(out_path, "wb");
        if (!g_out_file) { perror(out_path); return 1; }
        r_write_wav = 1;
        write_wav_header(g_out_file, (uint32_t)audio_rate_actual);
        fprintf(stderr, "Writing WAV to: %s\n", out_path);
    } else {
        g_out_file = stdout;
    }

    fprintf(stderr,
        "Freq: %.3f MHz  |  Offset: %+d Hz  |  Fs: %u  |  Chan: %d Hz  |  Audio: %d Hz\n",
        freq / 1e6, freq_offset, samp_rate, chan_rate, audio_rate_actual);
    if (!out_path)
        fprintf(stderr, "Pipe output to: aplay -r %d -f S16_LE -c 1\n", audio_rate_actual);
    if (duration > 0.0)
        fprintf(stderr, "Recording %.1f seconds...\n", duration);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // cutoff = f_c / Fs
    float taps1[FIR_TAPS1];
    make_lowpass_taps(taps1, FIR_TAPS1, (float)CHAN_CUTOFF_HZ / (float)samp_rate);

    float taps2[FIR_TAPS2];
    make_lowpass_taps(taps2, FIR_TAPS2, (float)AUDIO_CUTOFF_HZ / (float)chan_rate);

    g_ctx = fm_demod_cpu_create(BLOCK_SIZE,
                                 DECIM1, taps1, FIR_TAPS1,
                                 DECIM2, taps2, FIR_TAPS2,
                                 (float)freq_offset, (float)samp_rate);

    int r = rtlsdr_open(&r_dev, 0);
    if (r < 0) { fprintf(stderr, "Failed to open RTL-SDR: %d\n", r); return 1; }

    rtlsdr_set_sample_rate(r_dev, samp_rate);
    rtlsdr_set_center_freq(r_dev, freq);
    rtlsdr_set_tuner_gain_mode(r_dev, 0);
    rtlsdr_reset_buffer(r_dev);

    if (duration > 0.0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        r_stop_time = ts.tv_sec + ts.tv_nsec * 1e-9 + duration;
    }

    fprintf(stderr, "Receiving... Ctrl-C to stop.\n");
    rtlsdr_read_async(r_dev, rtlsdr_callback, NULL, 0, 2 * BLOCK_SIZE);

    rtlsdr_close(r_dev);
    fm_demod_cpu_destroy(g_ctx);

    if (r_write_wav) {
        finalize_wav_header(g_out_file, r_samples_written);
        fclose(g_out_file);
        fprintf(stderr, "Saved %zu samples (%.1f sec) to: %s\n",
                r_samples_written,
                (double)r_samples_written / audio_rate_actual,
                out_path);
    }

    return 0;
}
