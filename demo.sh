#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

FREQ=95100000
SRATE=2048000
DURATION=60
SAMPLES=$((DURATION * SRATE))

mkdir -p data

BIN=data/demo_capture.bin
GPU_NAIVE_WAV=data/demo_gpu_naive.wav
GPU_OPT_WAV=data/demo_gpu_opt.wav
CPU_WAV=data/demo_cpu.wav
GPU_MULTI_PREFIX=data/demo_gpu_multi

echo "=== Building ==="
(cd fm_demod_gpu_naive && make -s)
(cd fm_demod_gpu_opt   && make -s)
(cd fm_demod_gpu_multi && make -s)
(cd fm_demod_cpu       && make -s)
echo "Build OK"

echo ""
if [ -f "$BIN" ]; then
    echo "=== Using existing capture: $BIN ==="
else
    echo "=== Capturing ${DURATION}s at ${FREQ} Hz ==="
    rtl_sdr -f "$FREQ" -s "$SRATE" -n "$SAMPLES" "$BIN"
fi

echo ""
echo "=== GPU pipeline (naive) ==="
fm_demod_gpu_naive/fm_rx_gpu_naive -b "$BIN" -o "$GPU_NAIVE_WAV"

echo ""
echo "=== GPU pipeline (optimized) ==="
fm_demod_gpu_opt/fm_rx_gpu_opt -b "$BIN" -o "$GPU_OPT_WAV"

echo ""
echo "=== GPU multi-channel pipeline (5 channels, +/-400 kHz) ==="
fm_demod_gpu_multi/fm_rx_gpu_multi -b "$BIN" -c "-400000,-200000,0,200000,400000" -o "$GPU_MULTI_PREFIX"

echo ""
echo "=== CPU pipeline ==="
fm_demod_cpu/fm_rx_cpu -b "$BIN" -o "$CPU_WAV"

echo ""
echo "=== Done ==="
echo "  GPU naive audio    : $GPU_NAIVE_WAV"
echo "  GPU optimized audio: $GPU_OPT_WAV"
echo "  GPU multi audio    : ${GPU_MULTI_PREFIX}_ch[0-4].wav  (5 channels)"
echo "  CPU audio          : $CPU_WAV"
