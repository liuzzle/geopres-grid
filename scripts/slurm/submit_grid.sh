#!/bin/bash
# Submit one backbone's grid: a baselines job, then an array of shards that starts
# only once the baselines succeeded, so no two shards write the same result.
#
#   scripts/slurm/submit_grid.sh <backbone> [tier] [shards]     # tier1, 1 shard
#
# Tier 1 takes minutes per backbone in one job; shards are for tier 2.
# Memory: grid.sh's 16G for tier 1, 64G for tier 2, where one task's cache index
# alone reaches ~12 GiB (MSMARCO, logging.md §30), and for mldr, whose thirteen
# block indices hold ~6.6G characters of document text (§70). GRID_MEM=32G
# overrides all of them.
#
# Run from the repo root (not with sbatch). A shard that hits its time limit is
# resumed by resubmitting just that index with the same arguments, e.g. shard 3:
#   sbatch --job-name=grid-mgte --array=3 scripts/slurm/grid.sh --backbone mgte --tier tier1 --num-shards 8
# --num-shards is explicit because a one-index array would report a count of 1.
# A tier-2 or mldr shard needs --mem=64G there too; grid.sh alone asks for 16G.

set -euo pipefail

backbone=${1:?usage: scripts/slurm/submit_grid.sh <backbone> [tier] [shards]}
tier=${2:-tier1}
shards=${3:-1}
if [[ $tier == tier2 || $tier == mldr ]]; then default_mem=64G; else default_mem=16G; fi
mem=${GRID_MEM:-$default_mem}

mkdir -p logs
baselines=$(sbatch --parsable --mem="$mem" --job-name="grid-$backbone-baselines" --output="logs/%x_%j.out" \
    scripts/slurm/grid.sh --backbone "$backbone" --tier "$tier" --baselines-only)
cells=$(sbatch --parsable --mem="$mem" --job-name="grid-$backbone" --dependency="afterok:$baselines" \
    --array="0-$((shards - 1))" \
    scripts/slurm/grid.sh --backbone "$backbone" --tier "$tier" --num-shards "$shards")
echo "$backbone/$tier: baselines job $baselines, cells array $cells (${shards} shards, $mem each)"
