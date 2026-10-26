#!/bin/bash
# Submit the same benchmark twice, once per model.  The jobs reuse the image
# selected through VLLM_IMAGE; build that image separately before submitting.
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:cu129}"
export VLLM_IMAGE

# Different ports also make all jobs safe if Slurm schedules them on the same
# host with different GPUs. The first pair is the 3-document/25-question run;
# the second pair is the 8-document/50-question run.
PORT=8000 sbatch --export=ALL -w faretra \
  run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py
PORT=8001 sbatch --export=ALL -w faretra \
  run_with_vllm.sh Qwen/Qwen3.5-4B -- \
  python benchmark_llm_wiki.py
PORT=8002 sbatch --export=ALL -w faretra \
  run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --top-docs 8 --max-questions 50
PORT=8003 sbatch --export=ALL -w faretra \
  run_with_vllm.sh Qwen/Qwen3.5-4B -- \
  python benchmark_llm_wiki.py --top-docs 8 --max-questions 50
