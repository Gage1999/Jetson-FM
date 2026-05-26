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

## References

1. RTL-SDR Blog. *V4 Dongle Initial Release.* https://www.rtl-sdr.com/rtl-sdr-blog-v4-dongle-initial-release/
2. RTL-SDR Blog. *librtlsdr fork (V4 driver).* https://github.com/rtlsdrblog/rtl-sdr-blog
3. PySDR. *IQ Samples and FM Demodulation.* https://pysdr.org/content/sampling.html
4. NVIDIA. *CUDA Programming Guide.* https://docs.nvidia.com/cuda/cuda-programming-guide/
5. SciPy. *scipy.signal.firwin.* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
6. Osmocom. *rtl_fm for CPU reference.* https://github.com/osmocom/rtl-sdr
7. ALSA Project. *PCM Interface.* https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html