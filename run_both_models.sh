#!/bin/bash
# Submit the same benchmark twice, once per model.  The jobs reuse the image
# selected through VLLM_IMAGE; build that image separately before submitting.
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:cu129}"
export VLLM_IMAGE

# Different ports also make the two jobs safe if Slurm schedules them on the
# same host with different GPUs.
sbatch -w faretra run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- python benchmark_llm_wiki.py
sbatch -w faretra run_with_vllm.sh Qwen/Qwen3.5-4B -- python benchmark_llm_wiki.py
sbatch -w faretra run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- python benchmark_llm_wiki.py --top-docs 8 --max-questions 50
sbatch -w faretra run_with_vllm.sh Qwen/Qwen3.5-4B -- python benchmark_llm_wiki.py --top-docs 8 --max-questions 50
