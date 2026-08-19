#!/usr/bin/env bash
# Verify a local vLLM OpenAI-compatible endpoint after start_vllm_qwen.sh.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
MODEL="${MODEL:-Qwen2.5-7B-Instruct}"

echo "Checking ${BASE_URL}/models"
curl --fail --silent --show-error "${BASE_URL}/models"
echo

echo "Running a minimal chat completion"
curl --fail --silent --show-error "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: vLLM is ready\"}],\"temperature\":0,\"max_tokens\":8}"
echo
