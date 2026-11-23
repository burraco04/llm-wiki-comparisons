#!/bin/bash
# Submit both model checkpoints on both input subsets. Each run includes
# independent classic and CROW wikis. Build VLLM_IMAGE before submitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

VLLM_IMAGE="${VLLM_IMAGE:-llm-wiki-benchmark:structured-wiki-v4}"
# Fail before reserving GPUs if the image still contains an older benchmark.
# Jobs inherit the checked immutable ID and verify it again on their node.
VLLM_IMAGE="$(python3 ./verify_benchmark_image.py "$VLLM_IMAGE")"
export VLLM_IMAGE
RUN_TIME_LIMIT="${RUN_TIME_LIMIT:-08:00:00}"
RUN_NODE="${RUN_NODE-faretra}"
node_args=()
if [[ -n "$RUN_NODE" ]]; then
  node_args=(-w "$RUN_NODE")
fi

# Both models used byte-identical inputs at each scale. Reuse the saved Qwen2.5
# subsets for both rather than picking whichever timestamp happens to be latest.
small_data="structured_wiki_v4/Qwen_Qwen2.5-3B-Instruct/docs_3_questions_25/20261116T110157_135698Z/dataset"
large_data="structured_wiki_v4/Qwen_Qwen2.5-3B-Instruct/docs_8_questions_150/20261116T113159_994748Z/dataset"
if [[ "${REPLAY_ARCHIVE:-0}" == 1 ]]; then
  for subset in "$small_data" "$large_data"; do
    host_path="${RESULTS_DIR:-$PWD/results}/$subset"
    for file in documents.jsonl questions.jsonl ../benchmark_manifest.json; do
      if [[ ! -r "$host_path/$file" ]]; then
        echo "Missing archived input $host_path/$file. Extract the archive in the checkout first." >&2
        exit 1
      fi
    done
  done
fi

# Distinct host ports allow the jobs to share a host on different GPUs.
submit() {
  local port="$1" model="$2" docs="$3" questions="$4" subset="$5"
  local dataset_args=()
  if [[ "${REPLAY_ARCHIVE:-0}" == 1 ]]; then
    dataset_args=(--dataset-dir "/workdir/results/$subset")
  fi
  PORT="$port" sbatch --export=ALL --time="$RUN_TIME_LIMIT" "${node_args[@]}" \
    run_with_vllm.sh "$model" -- python benchmark_llm_wiki.py \
    --top-docs "$docs" --max-questions "$questions" --wiki-modes classic crow "${dataset_args[@]}"
}
submit 8000 Qwen/Qwen2.5-3B-Instruct 3 25 "$small_data"
submit 8004 Qwen/Qwen3.5-4B          3 25 "$small_data"
submit 8002 Qwen/Qwen2.5-3B-Instruct 8 150 "$large_data"
submit 8003 Qwen/Qwen3.5-4B          8 150 "$large_data"
