#!/bin/bash
# Submit both model checkpoints on both input subsets. Each run includes
# independent classic and CROW wikis. Build VLLM_IMAGE before submitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:structured-wiki-v3}"
# Fail before reserving GPUs if the image still contains an older benchmark.
# Jobs inherit the checked immutable ID and verify it again on their node.
VLLM_IMAGE="$(python3 ./verify_benchmark_image.py "$VLLM_IMAGE")"
export VLLM_IMAGE
RUN_TIME_LIMIT="${RUN_TIME_LIMIT:-08:00:00}"

# Different ports also make all jobs safe if Slurm schedules them on the same
# host with different GPUs. The first pair is the 3-document/25-question run;
# the second pair is the 8-document/150-question run.
PORT=8000 sbatch --export=ALL --time="$RUN_TIME_LIMIT" -w faretra \
  run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --wiki-modes classic crow
PORT=8004 sbatch --export=ALL --time="$RUN_TIME_LIMIT" -w faretra \
  run_with_vllm.sh Qwen/Qwen3.5-4B -- \
  python benchmark_llm_wiki.py --wiki-modes classic crow
PORT=8002 sbatch --export=ALL --time="$RUN_TIME_LIMIT" -w faretra \
  run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --top-docs 8 --max-questions 150 --wiki-modes classic crow
PORT=8003 sbatch --export=ALL --time="$RUN_TIME_LIMIT" -w faretra \
  run_with_vllm.sh Qwen/Qwen3.5-4B -- \
  python benchmark_llm_wiki.py --top-docs 8 --max-questions 150 --wiki-modes classic crow
