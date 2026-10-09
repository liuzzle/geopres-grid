#!/bin/bash
# Precompute one backbone's cache for one task set, in the environment it needs.
#
#   sbatch scripts/slurm/precompute.sh <backbone> [tier]      # tier defaults to tier1
#
# Submit from the repo root; `.env` there sets PROJECT_ROOT and STORAGE_PATH.
# The harriers run in the root environment only: precomputed in both transformers
# environments they get two encode hashes, and evaluation then needs --encode-hash.
#
# A100 or newer: the lockfiles resolve torch against CUDA 13, which has no V100
# (sm_70) kernels. `--device cuda` makes a job without a usable GPU fail at once
# instead of falling back to the CPU (`config.resolve_device`) and running for hours.
#SBATCH --job-name=precompute
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=A100:1
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

backbone=${1:?usage: sbatch scripts/slurm/precompute.sh <backbone> [tier]}
tier=${2:-tier1}

case "$backbone" in
    mgte | harrier-270m | harrier-06b) project=. ;;
    mdenseon) project=envs/transformers5 ;;
    embeddinggemma-2) project=envs/sentence_transformers6 ;;
    *) echo "unknown backbone: $backbone" >&2; exit 2 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
# Model weights and datasets; home quotas are small. Override before sbatch if needed.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Starting $backbone / $tier in $project on $(hostname): $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv run --frozen --project "$project" python scripts/precompute.py \
    --backbone "$backbone" --tier "$tier" --device cuda

echo "Finished: $(date)"
