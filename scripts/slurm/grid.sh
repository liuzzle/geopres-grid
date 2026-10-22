#!/bin/bash
# One grid job on CPU: the baselines, or one shard of the cells. Submitted by
# submit_grid.sh, which orders the two; arguments go to scripts/run_grid.py.
#
#   sbatch scripts/slurm/grid.sh --backbone mgte --tier tier1 --baselines-only
#
# Root environment for every backbone: evaluation reads the cache, no model.
# --mem is tier 1's; submit_grid.sh asks for 64G on tier 2 (logging.md §30).
#SBATCH --job-name=grid
#SBATCH --time=0-08:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
# Datasets come from HF_HOME's cache, where precompute put them. Online, every shard
# asks the Hub API about every dataset; dozens of shards from the cluster's one IP got
# rate-limited (429) and failed (logging.md §33.8). HF_HUB_OFFLINE=0 to go online.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-$HF_HUB_OFFLINE}
# numpy (OpenBLAS) and torch otherwise start one thread per core of the node, not
# of the job; on the login node that alone exhausts the per-user memory limit.
threads=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Starting grid $* on $(hostname): $(date)"
uv run --frozen python scripts/run_grid.py "$@"
echo "Finished: $(date)"
