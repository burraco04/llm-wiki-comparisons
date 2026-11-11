#!/bin/bash
#SBATCH --job-name=gpu-diagnostic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=logs/gpu-diagnostic-%j.out
# Submit with: mkdir -p logs; sbatch -w faretra gpu_diagnostic.sh
# A script (rather than sbatch --wrap) ensures Slurm uses Bash for arrays.
set -uo pipefail

printf 'CUDA_VISIBLE_DEVICES=%s\nSLURM_JOB_GPUS=%s\n' \
  "${CUDA_VISIBLE_DEVICES:-unset}" "${SLURM_JOB_GPUS:-unset}"

# Continue after failed queries so the log covers every allocated GPU, while
# retaining the first failing status as the job's final exit code.
status=0
query() {
  local code
  if nvidia-smi "$@"; then
    return 0
  else
    code=$?
    printf 'ERROR: nvidia-smi query failed (exit=%s)\n' "$code" >&2
    if (( status == 0 )); then
      status=$code
    fi
  fi
}

query -L
query --query-gpu=index,uuid,name,memory.free,memory.total --format=csv

GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
if [[ -z "$GPU_LIST" ]]; then
  printf 'ERROR: no allocated GPUs in CUDA_VISIBLE_DEVICES or SLURM_JOB_GPUS\n' >&2
  if (( status == 0 )); then status=1; fi
  exit "$status"
fi
IFS=',' read -r -a allocated <<< "$GPU_LIST"
for gpu in "${allocated[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  if [[ -z "$gpu" ]]; then
    printf 'ERROR: empty GPU identifier in allocation %s\n' "$GPU_LIST" >&2
    if (( status == 0 )); then status=1; fi
    continue
  fi
  printf 'Checking allocated GPU %s\n' "$gpu"
  query -i "$gpu" --query-gpu=uuid,memory.free,memory.total --format=csv
done
exit "$status"
