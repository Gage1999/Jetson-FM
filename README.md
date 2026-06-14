# GPU FM Demodulator

A real-time FM radio demodulator implemented in C and CUDA, comparing CPU and GPU
signal processing pipelines using an RTL-SDR dongle as the radio front-end.

Four implementations are provided at increasing levels of GPU optimization, plus a
multi-channel variant that demodulates several FM stations simultaneously from a
single capture.

---

## Implementations

| Directory | Binary | Description |
|---|---|---|
| `fm_demod_cpu` | `fm_rx_cpu` | Sequential C, runs entirely on CPU |
| `fm_demod_gpu_naive` | `fm_rx_gpu_naive` | One CUDA kernel per pipeline stage, naive global-memory access |
| `fm_demod_gpu_opt` | `fm_rx_gpu_opt` | Merged kernels, shared-memory tiling, parallel DC block |
| `fm_demod_gpu_multi` | `fm_rx_gpu_multi` | Optimized GPU pipeline extended to N channels via concurrent CUDA streams |

---

## Signal Processing Pipeline

All four implementations execute the same logical pipeline on each block of IQ
samples from the RTL-SDR:

```
RTL-SDR raw bytes (uint8, offset binary)
  |
  v
[1] IQ Normalize      -- convert to float [-1, 1] (zero = 127.5, not 128)
  |
  v
[2] Frequency Shift   -- multiply by e^(j*phase) to center the station at 0 Hz
  |
  v
[3] Channel Filter    -- 127-tap Hann-windowed sinc low-pass FIR, cutoff 100 kHz
    + Decimate x7     -- 2,048,000 sps -> ~293,000 sps
  |
  v
[4] FM Discriminator  -- atan2(cross, dot) phase-difference (cross-dot form)
  |
  v
[5] DC Block          -- subtract mean to remove receiver DC offset
  |
  v
[6] Audio Filter      -- 127-tap Hann-windowed sinc low-pass FIR, cutoff 15 kHz
    + Decimate x6     -- ~293,000 sps -> ~48,762 Hz audio rate
  |
  v
16-bit mono PCM WAV
```

Default parameters (all tunable via CLI flags):

| Parameter | Value |
|---|---|
| Center frequency | 95.1 MHz |
| Sample rate | 2,048,000 sps |
| Block size | 131,072 IQ pairs |
| FIR taps (both stages) | 127 |
| Decimation stage 1 | 7 |
| Decimation stage 2 | 6 |
| Channel filter cutoff | 100 kHz |
| Audio filter cutoff | 15 kHz |
| Output audio rate | ~48,762 Hz |

---

## GPU Optimization Progression

### Naive (`fm_demod_gpu_naive`)

- Each pipeline stage is a separate kernel launch
- Normalize and frequency-shift are separate kernels writing back to global memory
  between them
- Decimation reads I and Q in separate kernel invocations
- DC block is a single-thread serial loop running in one block of one thread

### Optimized (`fm_demod_gpu_opt`)

- **Merged normalize + shift** - `kernel_normalize_and_shift` combines both stages
  into one kernel, eliminating an intermediate global-memory round-trip
- **Shared-memory tiling in decimation** - each thread block loads its input tile
  (including halo samples) into shared memory before computing FIR sums, cutting
  repeated global-memory reads for overlapping filter windows
- **Joint IQ decimation** - `kernel_decimate_iq` processes I and Q together in one
  kernel, halving kernel launch overhead and improving memory access patterns
- **Parallel DC block** - replaced the serial loop with a two-pass parallel reduction:
  `kernel_dc_reduce` accumulates the sum across thread blocks using `atomicAdd`,
  then `kernel_dc_subtract` subtracts the mean in a second parallel kernel

### Multi-channel (`fm_demod_gpu_multi`)

- Applies all optimized kernels from `fm_demod_gpu_opt`
- Each channel receives its own CUDA stream, so all channels run concurrently on
  the GPU
- Channels share the raw IQ buffer (one DMA transfer per block) but each channel
  has independent frequency-shift state, filter state, and audio output buffer
- Up to 16 channels, specified as Hz offsets from the center frequency

### Memory Strategy (GPU variants)

Input and audio output buffers use `cudaHostAllocMapped` (pinned zero-copy memory).
The GPU accesses these directly without an explicit `cudaMemcpy`, avoiding a separate
transfer step for the narrow input and output data. Intermediate buffers (I/Q, after
decimation, discriminator output) live in device memory.

---

## Requirements

**Hardware**

- RTL-SDR dongle (for live reception) or a pre-captured `.bin` file
- NVIDIA GPU - Makefiles target `sm_87` (Jetson AGX Orin / Orin NX). To use a
  different GPU, change `-arch=sm_87` in each `Makefile` to match your device
  (e.g. `-arch=sm_86` for RTX 3080/3090, `-arch=sm_89` for RTX 4090)

**Software**

- CUDA 12.2 (expected at `/usr/local/cuda-12.2`; adjust `PATH` in Makefiles if
  your installation is elsewhere)
- `gcc`
- `librtlsdr` (headers in `/usr/local/include`, library in `/usr/local/lib`)
- `libm`

---

## Building

Each subdirectory builds independently:

```bash
cd fm_demod_cpu       && make
cd fm_demod_gpu_naive && make
cd fm_demod_gpu_opt   && make
cd fm_demod_gpu_multi && make
```

Clean:

```bash
make -C fm_demod_cpu clean
make -C fm_demod_gpu_naive clean
make -C fm_demod_gpu_opt clean
make -C fm_demod_gpu_multi clean
```

---

## Usage

### CPU receiver

```
fm_rx_cpu [-f freq_hz] [-r sample_rate] [-s offset_hz] [-o output.wav]
          [-t duration_sec] [-b bench_file.bin]
```

| Flag | Default | Description |
|---|---|---|
| `-f` | 95100000 | Tuner center frequency in Hz |
| `-r` | 2048000 | Sample rate |
| `-s` | 0 | Fine frequency offset in Hz (shifts signal before demod) |
| `-o` | stdout (raw S16) | Output WAV file path |
| `-t` | unlimited | Stop after N seconds |
| `-b` | - | Decode/benchmark a pre-recorded `.bin` file instead of live SDR |

### GPU receivers (naive and optimized)

```
fm_rx_gpu_naive [-f freq_hz] [-r sample_rate] [-s offset_hz] [-o output.wav]
                [-t duration_sec] [-b bench_file.bin]

fm_rx_gpu_opt   [-f freq_hz] [-r sample_rate] [-s offset_hz] [-o output.wav]
                [-t duration_sec] [-b bench_file.bin]
```

Flags are identical to `fm_rx_cpu`.

### GPU multi-channel receiver

```
fm_rx_gpu_multi [-f freq_hz] [-r sample_rate] [-c off1,off2,...] [-o prefix]
                [-t duration_sec] [-b bench_file.bin]
```

| Flag | Default | Description |
|---|---|---|
| `-f` | 95100000 | Tuner center frequency in Hz |
| `-r` | 2048000 | Sample rate |
| `-c` | 0 | Comma-separated Hz offsets from center (one per channel) |
| `-o` | - | Output file prefix; produces `<prefix>_ch0.wav`, `<prefix>_ch1.wav`, ... |
| `-t` | unlimited | Stop after N seconds |
| `-b` | - | Decode/benchmark a pre-recorded `.bin` file |

---

## Examples

### Live reception

Record 95.1 MHz to a WAV file for 60 seconds using the optimized GPU pipeline:

```bash
fm_demod_gpu_opt/fm_rx_gpu_opt -f 95100000 -o output.wav -t 60
```

Pipe raw PCM to the system speaker (no WAV header):

```bash
fm_demod_cpu/fm_rx_cpu -f 95100000 | aplay -r 48762 -f S16_LE -c 1
```

### Multi-channel: demodulate 5 stations at once

```bash
fm_demod_gpu_multi/fm_rx_gpu_multi \
    -f 95100000 \
    -c "-400000,-200000,0,200000,400000" \
    -o data/scan \
    -t 60
# Produces: scan_ch0.wav through scan_ch4.wav
```

### Decode a pre-recorded capture

```bash
# Capture first
rtl_sdr -f 95100000 -s 2048000 -n $((60 * 2048000)) capture.bin

# Then decode with any pipeline
fm_demod_gpu_opt/fm_rx_gpu_opt -b capture.bin -o output.wav
```

### Benchmark

Passing `-b` without `-o` runs the benchmark loop (10 repetitions, all blocks) and
prints wall time, per-block latency, real-time factor, and CPU utilization:

```bash
fm_demod_gpu_opt/fm_rx_gpu_opt -b capture.bin
# Example output:
# blocks: 470  reps: 10  wall: 1234.5 ms  per-block: 0.26 ms  real-time: 246.3x  cpu: 4.2%
```

---

## Demo Script

`demo.sh` builds all four implementations, optionally captures 60 seconds of live
radio, then runs all four pipelines against the same capture and saves WAV files
to `data/`:

```bash
./demo.sh
```

If `data/demo_capture.bin` already exists it is reused, so you can run the script
without an RTL-SDR attached after the initial capture.

Output files:

```
data/demo_gpu_naive.wav
data/demo_gpu_opt.wav
data/demo_gpu_multi_ch0.wav ... data/demo_gpu_multi_ch4.wav
data/demo_cpu.wav
```

---

## Repository Structure

```
fm_demod_cpu/
  include/fm_demod_cpu.h          context struct and API
  src/fm_demod_cpu.c              pipeline stages (C)
  src/main.c                      RTL-SDR driver, WAV writer, benchmark harness
  Makefile

fm_demod_gpu_naive/
  include/fm_demod_gpu_naive.cuh  context struct and API
  src/fm_demod_gpu_naive.cu       CUDA kernels (one per stage)
  src/main.c                      RTL-SDR driver, WAV writer, benchmark harness
  Makefile

fm_demod_gpu_opt/
  include/fm_demod_gpu.cuh        context struct and API
  src/fm_demod_gpu.cu             CUDA kernels (merged, shared-memory tiled)
  src/main.c                      RTL-SDR driver, WAV writer, benchmark harness
  Makefile

fm_demod_gpu_multi/
  include/fm_demod_gpu_multi.cuh  context structs and API (FmDemodChannel, FmDemodMultiContext)
  src/fm_demod_gpu_multi.cu       CUDA kernels + per-channel stream management
  src/main.c                      RTL-SDR driver, WAV writer, benchmark harness
  Makefile

demo.sh                           end-to-end build + capture + decode script
docs/                             project report (PDF)
```
