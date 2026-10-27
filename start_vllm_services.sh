#!/usr/bin/env bash
set -euo pipefail

# Run the benchmark LLM and the local CROW/System One classifier in the same
# container.  The host script maps these internal ports to distinct host ports.
: "${LLM_MODEL:?LLM_MODEL is required}"
: "${CLASSIFIER_MODEL:?CLASSIFIER_MODEL is required}"

LLM_PORT="${LLM_PORT:-8000}"
CLASSIFIER_PORT="${CLASSIFIER_PORT:-8001}"
LLM_MEM="${LLM_GPU_MEMORY_UTILIZATION:-0.60}"
CLASSIFIER_MEM="${CLASSIFIER_GPU_MEMORY_UTILIZATION:-0.30}"
LLM_MAXLEN="${LLM_MAX_MODEL_LEN:-32768}"
CLASSIFIER_MAXLEN="${CLASSIFIER_MAX_MODEL_LEN:-8192}"
LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-32}"
CLASSIFIER_MAX_NUM_SEQS="${CLASSIFIER_MAX_NUM_SEQS:-8}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"

LLM_LOG=/tmp/vllm-llm.log
CLASSIFIER_LOG=/tmp/vllm-classifier.log

vllm serve "$LLM_MODEL" \
  --host 0.0.0.0 \
  --port "$LLM_PORT" \
  --served-model-name "$LLM_MODEL" \
  --max-model-len "$LLM_MAXLEN" \
  --gpu-memory-utilization "$LLM_MEM" \
  --max-num-seqs "$LLM_MAX_NUM_SEQS" \
  --default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
  --enable-prefix-caching \
  >"$LLM_LOG" 2>&1 &
LLM_PID=$!

# Liquid d1 exposes the System One endpoint through vLLM.  Raw logprobs are
# required by that endpoint to calculate the probability of each typed answer.
vllm serve "$CLASSIFIER_MODEL" \
  --host 0.0.0.0 \
  --port "$CLASSIFIER_PORT" \
  --served-model-name "$CLASSIFIER_MODEL" \
  --max-model-len "$CLASSIFIER_MAXLEN" \
  --gpu-memory-utilization "$CLASSIFIER_MEM" \
  --max-num-seqs "$CLASSIFIER_MAX_NUM_SEQS" \
  --logprobs-mode raw_logprobs \
  --disable-prefix-caching \
  >"$CLASSIFIER_LOG" 2>&1 &
CLASSIFIER_PID=$!

cleanup() {
  kill "$LLM_PID" "$CLASSIFIER_PID" 2>/dev/null || true
  wait "$LLM_PID" "$CLASSIFIER_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

set +e
wait -n "$LLM_PID" "$CLASSIFIER_PID"
status=$?
set -e

echo ">>> one of the vLLM servers exited (status=$status)"
echo ">>> benchmark LLM log"
cat "$LLM_LOG" || true
echo ">>> classifier log"
cat "$CLASSIFIER_LOG" || true
exit "$status"
