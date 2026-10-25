#!/bin/bash
#SBATCH --job-name=llm-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp-%j.out
#SBATCH --signal=B:TERM@60
# ═══════════════════════════════════════════════════════
#  Start vLLM → run YOUR command inside the container → stop vLLM
#
#  Usage (directly, or with sbatch — same arguments):
#    ./run_with_vllm.sh <hf_model> -- <your command ...>
#    sbatch run_with_vllm.sh <hf_model> -- <your command ...>
#
#  Examples:
#    sbatch run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- python my_experiment.py
#    sbatch run_with_vllm.sh Qwen/Qwen3.5-4B          -- python my_experiment.py --dataset squad
#
#  The command runs in /workdir inside the application image.  The host
#  ./results directory is mounted at /workdir/results and survives cleanup.
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
VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:cu129}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

# The image is built externally from Dockerfile and must exist on the node
# running this job (docker build -t llm-wiki-benchmark:cu129 .).
export VLLM_IMAGE

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

# ── 2. YOUR command (inside the vLLM container) ──
echo ">>> running in vllm-${PORT}: $*"
docker exec -w /workdir "vllm-${PORT}" "$@"
echo ">>> done"
