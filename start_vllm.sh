#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Start one vLLM server (official image, no Dockerfile needed)
#
#  Usage:
#    ./start_vllm.sh <hf_model> [port] [gpu_mem_util] [max_model_len]
#
#  Examples (one RTX 3090, 24 GB):
#    ./start_vllm.sh Qwen/Qwen2.5-3B-Instruct 8000 0.45
#    ./start_vllm.sh Qwen/Qwen3.5-4B          8001 0.45     # second model on the same GPU
#
#  Stop:   docker rm -f vllm-8000
#  Logs:   docker logs -f vllm-8000
#
#  GPU: uses $CUDA_VISIBLE_DEVICES (set by Slurm), else $GPU, else 0.
#  The API is only reachable from this machine (127.0.0.1); use an SSH tunnel from outside.
# ═══════════════════════════════════════════════════════
set -euo pipefail

MODEL="${1:?usage: $0 <hf_model> [port] [gpu_mem_util] [max_model_len]}"
PORT="${2:-8000}"
MEM="${3:-0.85}"
MAXLEN="${4:-32768}"
NAME="vllm-${PORT}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_TAG="${VLLM_TAG:-latest}"
VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:latest}"

# ── which GPU (Docker ignores Slurm, so pin it explicitly by UUID) ──
GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
GPU="${GPU%%,*}"
if [[ "$GPU" != GPU-* ]]; then
  GPU=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
fi
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
echo ">>> GPU $GPU: ${FREE}/${TOTAL} MiB free, requesting ${MEM} of total"

mkdir -p "$HF_CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus "\"device=${GPU}\"" \
  --ipc=host \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -p 127.0.0.1:"$PORT":8000 \
  "$VLLM_IMAGE" \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$MEM" \
  --enable-prefix-caching >/dev/null

echo ">>> waiting for $MODEL on port $PORT ..."
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo ">>> ready after ~$((i*10))s"
    curl -s "http://localhost:${PORT}/v1/models"; echo
    echo ">>> base_url = http://localhost:${PORT}/v1   model = $MODEL"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" != "running" ]]; then
    echo "ERROR: vLLM exited. Log:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  sleep 10
done
echo "ERROR: not ready after 30 min"; docker logs --tail 60 "$NAME"; exit 1
