#!/bin/bash
# One grid job on CPU: the baselines, or one shard of the cells. Submitted by
# submit_grid.sh, which orders the two; arguments go to scripts/run_grid.py.
#
#   sbatch scripts/slurm/grid.sh --backbone mgte --tier tier1 --baselines-only
#
# Root environment for every backbone: evaluation reads the cache, no model.
#SBATCH --job-name=grid
#SBATCH --time=0-08:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
# numpy (OpenBLAS) and torch otherwise start one thread per core of the node, not
# of the job; on the login node that alone exhausts the per-user memory limit.
threads=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Starting grid $* on $(hostname): $(date)"
uv run --frozen python scripts/run_grid.py "$@"
echo "Finished: $(date)"
