#!/bin/bash
#SBATCH --job-name=llm-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp-%j.out
#SBATCH --signal=B:TERM@60
# ═══════════════════════════════════════════════════════
#  Start vLLM → run YOUR command (it can use vLLM) → stop vLLM
#
#  Usage (directly, or with sbatch — same arguments):
#    ./run_with_vllm.sh <hf_model> -- <your command ...>
#    sbatch run_with_vllm.sh <hf_model> -- <your command ...>
#
#  Examples:
#    sbatch run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- python my_experiment.py
#    sbatch run_with_vllm.sh Qwen/Qwen3.5-4B          -- python my_experiment.py --dataset squad
#
#  Your command sees:
#    LLM_BASE_URL=http://localhost:<port>/v1
#    LLM_MODEL=<hf_model>
#  so llm_client.call_llm() connects automatically.
# ═══════════════════════════════════════════════════════
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

MODEL="${1:?usage: $0 <hf_model> -- <command ...>}"; shift
[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 1 ]] || { echo "ERROR: no command given after --"; exit 1; }

PORT="${PORT:-8000}"
MEM="${GPU_MEM_UTIL:-0.85}"
MAXLEN="${MAX_MODEL_LEN:-32768}"

cleanup() { echo ">>> stopping vLLM"; docker rm -f "vllm-${PORT}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' TERM INT

# ── 1. start vLLM ──
./start_vllm.sh "$MODEL" "$PORT" "$MEM" "$MAXLEN"


export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"
# for llmwiki2 (llm-wiki-v2) the same server, through its own variables:
export OKF_LLM_PROVIDER=custom
export OKF_LLM_BASE_URL="$LLM_BASE_URL"
export OKF_LLM_MODEL="$MODEL"
export OKF_LLM_TIMEOUT=600

# ── 2. YOUR command ──
echo ">>> running: $*"
"$@"
echo ">>> done"
