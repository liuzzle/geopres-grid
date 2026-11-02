#!/bin/bash
# Train one GeoPres projection on a tier's cached rows, then score it on that tier.
#
#   sbatch scripts/slurm/train_geopres.sh <backbone> <target_dim> [tier] [train args]
#   sbatch scripts/slurm/train_geopres.sh mgte 128
#   sbatch scripts/slurm/train_geopres.sh mgte 128 tier1 --positional_loss_factor 0.0 --custom_suffix angular
#
# CPU only: the projection is one matrix and the pooled tier-1 rows are ~50k, so a
# step at batch 4096 is ~0.1-0.2 s on 4 threads (logging.md §72). Root environment
# for every backbone: training and evaluation read the cache, no model is loaded.
# Training data are the cached query and document rows of the tier's tasks, the
# corpora that are then scored -- transductive, like the per-task PCA fit.
# Scores fp32, int8, int4 and binary by default (QUANT_METHODS="..." to change); the cells
# are evaluated one after another, each a separate run id (`dr_weights_id`).
#SBATCH --job-name=geopres
#SBATCH --time=0-06:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

backbone=${1:?usage: sbatch scripts/slurm/train_geopres.sh <backbone> <target_dim> [tier] [train args]}
dim=${2:?usage: sbatch scripts/slurm/train_geopres.sh <backbone> <target_dim> [tier] [train args]}
tier=${3:-tier1}
shift $(( $# < 3 ? $# : 3 ))
quant_methods=${QUANT_METHODS:-none int8 int4 binary}

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-$HF_HUB_OFFLINE}
threads=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Starting GeoPres $backbone $dim on $tier ($*) on $(hostname): $(date)"
log=$(mktemp)
uv run --frozen python -m geopres_grid.train_projection \
    --backbone "$backbone" --target_dim "$dim" --tier "$tier" --device cpu "$@" 2>&1 | tee "$log"
# The output directory is the one line of the training log that names it.
output=$(sed -n 's/.*Output path: //p' "$log" | tail -1)
echo "Trained: $(date); weights in $output"

for quant in $quant_methods; do
    uv run --frozen python scripts/evaluate.py --backbone "$backbone" --tier "$tier" \
        --dr-method geopres --target-dim "$dim" --geopres-weights "$output/weights.npy" \
        --quant-method "$quant"
done
echo "Finished: $(date)"
