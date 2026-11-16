#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Start the benchmark LLM and LiquidAI/d1-3B classifier from the application
#  image built by Dockerfile.
#
#  Usage:
#    ./start_vllm.sh <hf_model> [port] [gpu_mem_util] [max_model_len]
#
#  Examples (two RTX 3090 cards, 24 GB each):
#    ./start_vllm.sh Qwen/Qwen2.5-3B-Instruct 8000 0.50
#    ./start_vllm.sh Qwen/Qwen3.5-4B          8001 0.50
#
#  MAX_NUM_SEQS (default 32) caps concurrent sequences; Qwen3.5 is a hybrid
#  Mamba model and fails at the vLLM default of 256 on a 24 GB card.
#  CHAT_TEMPLATE_KWARGS (default {"enable_thinking": false}) turns off Qwen3.5
#  thinking, which would otherwise eat the benchmark's small max_tokens budgets.
#  Set CHAT_TEMPLATE_KWARGS='{}' to keep thinking on.
#
#  Build the image separately from Dockerfile:
#    docker build -t llm-wiki-benchmark:structured-wiki-v4 .
#  and set VLLM_IMAGE if you tag it differently.
#
#  Stop:   docker rm -f vllm-8000
#  Logs:   docker logs -f vllm-8000
#
#  GPU: requires exactly two allocated devices. Matching numeric CUDA/Slurm
#  assignments retain those indices for queries and Docker; UUID identities
#  are still checked and logged. The services use container GPUs 0 and 1.
#  The API is only reachable from this machine (127.0.0.1); use an SSH tunnel from outside.
# ═══════════════════════════════════════════════════════
set -Eeuo pipefail
STARTUP_STAGE=initialization
report_failure() {
  local status="$1" line="$2"
  printf 'ERROR: start_vllm.sh:%s phase=%s exit=%s\n' "$line" "$STARTUP_STAGE" "$status" >&2
}
trap 'report_failure "$?" "$LINENO"' ERR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:?usage: $0 <hf_model> [port] [gpu_mem_util] [max_model_len]}"
PORT="${2:-8000}"
MEM="${3:-0.85}"
MAXLEN="${4:-32768}"
NAME="vllm-${PORT}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LiquidAI/d1-3B}"
CLASSIFIER_INTERNAL_PORT="${CLASSIFIER_INTERNAL_PORT:-8001}"
CLASSIFIER_HOST_PORT="${CLASSIFIER_HOST_PORT:-$((PORT + 1000))}"
CLASSIFIER_MEM="${CLASSIFIER_GPU_MEM:-0.85}"
CLASSIFIER_MAXLEN="${CLASSIFIER_MAX_MODEL_LEN:-8192}"
CLASSIFIER_MAX_NUM_SEQS="${CLASSIFIER_MAX_NUM_SEQS:-8}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:structured-wiki-v4}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$LOG_DIR"
DOCKER_LOG="$LOG_DIR/docker-${SLURM_JOB_ID:-manual}-${PORT}.log"
BENCHMARK_RUNTIME_METADATA="${BENCHMARK_RUNTIME_METADATA:-/workdir/runtime_environment.json}"

dump_docker_logs() {
  docker logs "$NAME" 2>&1 | tee "$DOCKER_LOG"
}

# A failed assignment exits under set -e. Capture and report both output streams
# explicitly, because nvidia-smi can put its error on stdout, which would
# otherwise disappear inside $(...) and leave only the parent's EXIT message.
query_gpu() {
  local gpu="$1" field="$2" output status
  if output="$(nvidia-smi -i "$gpu" --query-gpu="$field" --format=csv,noheader,nounits 2>&1)"; then
    output="${output//[[:space:]]/}"
    case "$field" in
      uuid)
        if [[ "$output" != GPU-* ]]; then
          printf 'ERROR: nvidia-smi GPU=%s query=%s returned invalid data: %s\n' "$gpu" "$field" "$output" >&2
          return 1
        fi ;;
      memory.free|memory.total)
        if [[ ! "$output" =~ ^[0-9]+$ ]]; then
          printf 'ERROR: nvidia-smi GPU=%s query=%s returned invalid data: %s\n' "$gpu" "$field" "$output" >&2
          return 1
        fi ;;
    esac
    printf '%s\n' "$output"
  else
    status=$?
    printf 'ERROR: nvidia-smi GPU=%s query=%s exit=%s\n' "$gpu" "$field" "$status" >&2
    [[ -z "$output" ]] || printf '%s\n' "$output" >&2
    return "$status"
  fi
}

# Docker must receive exactly the allocated devices. Keep selector and identity
# separate: on this host UUID-based NVML lookups can fail even when the SAME
# device is readable when addressed by its allocated numeric index.
STARTUP_STAGE=gpu_discovery
VISIBLE_GPU_LIST="${CUDA_VISIBLE_DEVICES:-}"
VISIBLE_GPU_LIST="${VISIBLE_GPU_LIST//[[:space:]]/}"
if [[ ${CUDA_VISIBLE_DEVICES+x} && -z "$VISIBLE_GPU_LIST" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is explicitly empty; no GPU access is permitted"
  exit 1
fi
GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
GPU_LIST="${GPU_LIST//[[:space:]]/}"
printf '>>> GPU discovery: CUDA_VISIBLE_DEVICES=%s SLURM_JOB_GPUS=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" "${SLURM_JOB_GPUS:-unset}"
if [[ -z "$GPU_LIST" ]]; then
  echo "ERROR: Slurm did not expose two GPUs through CUDA_VISIBLE_DEVICES or SLURM_JOB_GPUS"
  exit 1
fi
IFS=',' read -r GPU0 GPU1 extra <<< "$GPU_LIST"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" || -n "${extra:-}" ]]; then
  echo "ERROR: expected exactly two allocated GPUs, got '$GPU_LIST'"
  exit 1
fi
if [[ "$GPU0" == "$GPU1" ]]; then
  echo "ERROR: expected two distinct allocated GPUs"
  exit 1
fi
GPU_QUERY0="$GPU0"
GPU_QUERY1="$GPU1"
USE_NUMERIC_ALLOCATION=0
JOB_GPU_LIST="${SLURM_JOB_GPUS:-}"
JOB_GPU_LIST="${JOB_GPU_LIST//[[:space:]]/}"
if [[ "$GPU0" =~ ^[0-9]+$ && "$GPU1" =~ ^[0-9]+$ && -n "$JOB_GPU_LIST" ]]; then
  if [[ "$GPU_LIST" != "$JOB_GPU_LIST" ]]; then
    echo "ERROR: numeric CUDA and Slurm GPU assignments differ; cannot infer host device mapping"
    exit 1
  fi
  USE_NUMERIC_ALLOCATION=1
fi
for var in GPU0 GPU1; do
  gpu="${!var}"
  gpu="${gpu//[[:space:]]/}"
  if [[ "$gpu" != GPU-* ]]; then
    gpu=$(query_gpu "$gpu" uuid)
  fi
  if [[ ! "$gpu" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]]; then
    echo "ERROR: allocated GPU has an invalid UUID"
    exit 1
  fi
  printf -v "$var" '%s' "$gpu"
done
if [[ "$GPU0" == "$GPU1" ]]; then
  echo "ERROR: allocated GPU selectors resolve to the same UUID"
  exit 1
fi
if (( USE_NUMERIC_ALLOCATION )); then
  GPU_SELECTOR0="$GPU_QUERY0"
  GPU_SELECTOR1="$GPU_QUERY1"
  printf '>>> verified numeric GPU selectors=%s,%s UUIDs=%s,%s\n' "$GPU_SELECTOR0" "$GPU_SELECTOR1" "$GPU0" "$GPU1"
else
  # UUID-only allocations retain explicit UUID selection. Never scan or choose
  # a different GPU to work around a failed query.
  GPU_QUERY0="$GPU0"
  GPU_QUERY1="$GPU1"
  GPU_SELECTOR0="$GPU0"
  GPU_SELECTOR1="$GPU1"
fi
STARTUP_STAGE=gpu_memory_query
FREE0=$(query_gpu "$GPU_QUERY0" memory.free)
TOTAL0=$(query_gpu "$GPU_QUERY0" memory.total)
FREE1=$(query_gpu "$GPU_QUERY1" memory.free)
TOTAL1=$(query_gpu "$GPU_QUERY1" memory.total)
echo ">>> LLM GPU $GPU0: ${FREE0}/${TOTAL0} MiB free, classifier GPU $GPU1: ${FREE1}/${TOTAL1} MiB free"
echo ">>> requesting LLM=${MEM} and classifier=${CLASSIFIER_MEM} of total"

mkdir -p "$HF_CACHE"
METADATA_DIR="$RESULTS_DIR/runtime_environment"
mkdir -p "$METADATA_DIR"
HOST_METADATA_NAME="${SLURM_JOB_ID:-manual}-${PORT}-$(date -u +%Y%m%dT%H%M%SZ)-host.json"
HOST_METADATA_FILE="$METADATA_DIR/$HOST_METADATA_NAME"
STARTUP_STAGE=host_metadata
echo ">>> capturing host runtime metadata"
python3 "$SCRIPT_DIR/capture_runtime_metadata.py" --phase host \
  --output "$HOST_METADATA_FILE" --gpu "$GPU0" --gpu "$GPU1"
IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$VLLM_IMAGE" 2>/dev/null || true)"

# Forward the library's optional settings without embedding their values in
# the Docker command line. Fixed local-service endpoints below take priority.
CONFIG_ENV=()
while IFS='=' read -r name _; do
  if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    case "$name" in
      OKF_*|VLLM_API_KEY|VLLM_TIMEOUT_SECONDS)
        CONFIG_ENV+=(--env "$name") ;;
    esac
  fi
done < <(env)
docker rm -f "$NAME" >/dev/null 2>&1 || true

STARTUP_STAGE=container_start
echo ">>> starting container $NAME from image $VLLM_IMAGE"
docker run -d --name "$NAME" \
  --gpus "\"device=${GPU_SELECTOR0},${GPU_SELECTOR1}\"" \
  --ipc=host \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$RESULTS_DIR":/workdir/results \
  "${CONFIG_ENV[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e BENCHMARK_RUNTIME_METADATA="$BENCHMARK_RUNTIME_METADATA" \
  -e BENCHMARK_HOST_METADATA="/workdir/results/runtime_environment/$HOST_METADATA_NAME" \
  -e BENCHMARK_IMAGE="$VLLM_IMAGE" \
  -e BENCHMARK_IMAGE_ID="$IMAGE_ID" \
  -e LLM_MODEL="$MODEL" \
  -e CLASSIFIER_MODEL="$CLASSIFIER_MODEL" \
  -e LLM_PORT=8000 \
  -e CLASSIFIER_PORT="$CLASSIFIER_INTERNAL_PORT" \
  -e LLM_DEVICE=0 \
  -e CLASSIFIER_DEVICE=1 \
  -e LLM_GPU_MEMORY_UTILIZATION="$MEM" \
  -e CLASSIFIER_GPU_MEMORY_UTILIZATION="$CLASSIFIER_MEM" \
  -e LLM_MAX_MODEL_LEN="$MAXLEN" \
  -e CLASSIFIER_MAX_MODEL_LEN="$CLASSIFIER_MAXLEN" \
  -e LLM_MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e CLASSIFIER_MAX_NUM_SEQS="$CLASSIFIER_MAX_NUM_SEQS" \
  -e CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" \
  -e LLM_BASE_URL="http://localhost:8000/v1" \
  -e OKF_CLASSIFIER_PROVIDER=custom \
  -e OKF_CLASSIFIER_BASE_URL="http://localhost:${CLASSIFIER_INTERNAL_PORT}/v1" \
  -e OKF_CLASSIFIER_MODEL="$CLASSIFIER_MODEL" \
  -e OKF_CLASSIFIER_TIMEOUT="${OKF_CLASSIFIER_TIMEOUT:-60}" \
  -e OKF_LLM_PROVIDER=custom \
  -e OKF_LLM_BASE_URL="http://localhost:8000/v1" \
  -e OKF_LLM_MODEL="$MODEL" \
  -e OKF_LLM_TIMEOUT="${OKF_LLM_TIMEOUT:-600}" \
  -p 127.0.0.1:"$PORT":8000 \
  -p 127.0.0.1:"$CLASSIFIER_HOST_PORT":"$CLASSIFIER_INTERNAL_PORT" \
  --entrypoint /workdir/start_vllm_services.sh \
  "$VLLM_IMAGE" >/dev/null

echo ">>> waiting for $MODEL on port $PORT and $CLASSIFIER_MODEL on port $CLASSIFIER_HOST_PORT ..."
STARTUP_STAGE=service_health
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 \
    && curl -sf "http://localhost:${CLASSIFIER_HOST_PORT}/health" >/dev/null 2>&1; then
    echo ">>> ready after ~$((i*10))s"
    curl -s "http://localhost:${PORT}/v1/models"; echo
    curl -s "http://localhost:${CLASSIFIER_HOST_PORT}/v1/models"; echo
    echo ">>> llm_base_url = http://localhost:${PORT}/v1   llm_model = $MODEL"
    echo ">>> classifier_base_url = http://localhost:${CLASSIFIER_HOST_PORT}/v1   classifier_model = $CLASSIFIER_MODEL"
    exit 0
  fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)" != "running" ]]; then
    echo "ERROR: vLLM exited. Full log: $DOCKER_LOG"
    dump_docker_logs
    exit 1
  fi
  sleep 10
done
echo "ERROR: not ready after 10 min. Full log: $DOCKER_LOG"
dump_docker_logs
exit 1
