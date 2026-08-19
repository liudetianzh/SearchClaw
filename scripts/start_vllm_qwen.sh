#!/usr/bin/env bash
# Serve a Qwen instruct model through vLLM's OpenAI-compatible API.
# Run this script only on Linux with an NVIDIA GPU and CUDA drivers.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. vLLM requires Linux with an NVIDIA CUDA GPU." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm was not found. Install it in the active Linux environment: pip install vllm" >&2
  exit 1
fi

echo "Serving ${MODEL} as ${SERVED_MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, endpoint=http://${HOST}:${PORT}/v1"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"
