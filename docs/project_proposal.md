# GPU-Accelerated FM Radio Demodulator

Real-time FM radio demodulation on the NVIDIA Jetson Orin Nano using CUDA kernels. Raw IQ samples stream from an RTL-SDR V4 dongle into a CUDA kernel pipeline that filters, decimates, and demodulates a live FM station to audio output.

---

## Pipeline

IQ samples from the dongle enter as interleaved `uint8` pairs at 2.4 Msps and pass through six stages or kernels. First we convert the IQ pairs to `float2` complex samples. Next we frequency-shift the target station to the 0Hz range. The next kernel is a decimating FIR filter to bring the rate down to 240 ksps. Next is an FM discriminator which computes instantaneous frequency via `atan2f` phase difference, producing a real-valued audio signal. The fifth kernel applies a second filter and decimates to 48 ksps. The final step is to convert to `int16` for the speaker input. The Jetson's unified memory layout between the CPU and GPU means the librtlsdr callback writes directly into a GPU-accessible memory with no `cudaMemcpy` required. The `rtl_fm` package from `librtlsdr` allows for benchmarking the GPU accelerated version against the original open source sequential FM demodulator that runs on the CPU.

---

## Dependencies

- `librtlsdr` (RTL-SDR Blog V4 fork):  USB driver and IQ sample streaming
- CUDA Toolkit 12.x: `nvcc`, runtime API, unified memory
- ALSA (`libasound`): PCM audio output
- SciPy (offline): FIR filter tap design via `scipy.signal.firwin`, baked into a header at build time

---

## Challenges

The primary GPU programming challenges center on the FIR decimation kernel and how shared memory is used. Naive shared memory indexing across threads in a warp can cause conflicts that create sequential memory accesses which significantly reduces throughput. Also, heavy shared memory usage may limit how many blocks can reside on an SM simultaneously. This needs to be tuned by changing the tile size and shared memory layout. The FM discriminator introduces its own challenge, namely a one-sample phase dependency that prevents naive parallelization within a single channel. This is handled by computing atan2 values in parallel within each block and passing the boundary sample between launches as a scalar. Finally, managing six kernels at rate fast enough to process incoming IQ samples is another challenge.

---

## Results

Benchmarked on the Jetson Orin Nano - 60 s IQ capture at 95.1 MHz, 2.048 MSPS (9370 blocks x 131072 samples, 10 repetitions each). Block period at this sample rate is 64.0 ms.

| Pipeline | Channels | Per-block | Real-time factor | CPU utilization |
|---|---|---|---|---|
| CPU (`fm_rx_cpu`) | 1 | 7.58 ms | 8.4x | 99.9% |
| GPU naive (`fm_rx_gpu_naive`) | 1 | 1.46 ms | 43.9x | 16.0% |
| GPU optimized (`fm_rx_gpu_opt`) | 1 | 0.36 ms | 180.0x | 66.4% |
| GPU multi-channel (`fm_rx_gpu_multi`) | 5 | 1.14 ms | 56.0x | 57.6% |

**Single-channel:** The baseline GPU pipeline is **5.2x faster** than the CPU reference. The optimized pipeline (`fm_demod_gpu_opt`) applies five additional GPU-specific improvements - shared-memory tiling on the FIR decimators, a shared-memory halo on the FM discriminator, a fused normalize+freq-shift kernel, a fused IQ channel decimator, and a parallel DC reduction - achieving a further **4.1x speedup** over baseline for a total of **21x over CPU**.

**Multi-channel:** The multi-channel pipeline (`fm_rx_gpu_multi`) demodulates 5 FM stations simultaneously from a single wideband capture using one CUDA stream per channel. All N channel pipelines are enqueued before any `cudaStreamSynchronize` call, giving the GPU scheduler maximum latitude to overlap them. The result: 5 stations in 1.14 ms per block, still **56x real-time** with 5x the audio output. If the 5 channel pipelines ran serially that would cost 5 x 0.36 ms = 1.80 ms; concurrent streams bring it to 1.14 ms, a **37% saving from parallelism**. A notable data point: the 5-channel multi pipeline (1.14 ms) is actually *faster per RTL-SDR block* than the single-channel baseline (1.46 ms) while producing 5x the audio output.

The higher CPU utilization on the optimized single-channel version (66.4% vs 16.0% on baseline) is expected: because the GPU finishes in 0.36 ms instead of 1.46 ms, `cudaStreamSynchronize` returns sooner and the CPU spends a larger fraction of wall time actively launching kernels. The multi-channel version (57.6%) sits between these - the GPU takes longer (1.14 ms) with N concurrent pipelines, so the CPU waits proportionally more than in the optimized single-channel case.

---

## References

1. RTL-SDR Blog. *V4 Dongle Initial Release.* https://www.rtl-sdr.com/rtl-sdr-blog-v4-dongle-initial-release/
2. RTL-SDR Blog. *librtlsdr fork (V4 driver).* https://github.com/rtlsdrblog/rtl-sdr-blog
3. PySDR. *IQ Samples and FM Demodulation.* https://pysdr.org/content/sampling.html
4. NVIDIA. *CUDA Programming Guide.* https://docs.nvidia.com/cuda/cuda-programming-guide/
5. SciPy. *scipy.signal.firwin.* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
6. Osmocom. *rtl_fm for CPU reference.* https://github.com/osmocom/rtl-sdr
7. ALSA Project. *PCM Interface.* https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html