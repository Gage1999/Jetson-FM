#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

FREQ=95100000
SRATE=2048000
DURATION=60
SAMPLES=$((DURATION * SRATE))

BIN=demo_capture.bin
GPU_WAV=demo_gpu.wav
CPU_WAV=demo_cpu.wav

echo "=== Building ==="
(cd fm_demod_gpu && make -s)
(cd fm_demod_cpu && make -s)
echo "Build OK"

echo ""
if [ -f "$BIN" ]; then
    echo "=== Using existing capture: $BIN ==="
else
    echo "=== Capturing ${DURATION}s at ${FREQ} Hz ==="
    rtl_sdr -f "$FREQ" -s "$SRATE" -n "$SAMPLES" "$BIN"
fi

echo ""
echo "=== GPU pipeline ==="
fm_demod_gpu/fm_rx_gpu -b "$BIN" -o "$GPU_WAV"

echo ""
echo "=== CPU pipeline ==="
fm_demod_cpu/fm_rx_cpu -b "$BIN" -o "$CPU_WAV"

echo ""
echo "=== Done ==="
echo "  GPU audio : $GPU_WAV"
echo "  CPU audio : $CPU_WAV"
