#!/bin/bash
# Probe every backbone (or the ones named) at MLDR's 8192 tokens: SDPA kernel, peak
# memory and seconds per document by batch size (scripts/probe_long_batch.py,
# logging.md §104.5). One GPU job, each backbone in the environment it needs.
#
#   sbatch scripts/slurm/probe_long_batch.sh                    # all five
#   sbatch scripts/slurm/probe_long_batch.sh harrier-06b mgte   # some
#
# A backbone that fails does not stop the others. Submit from the repo root.
#SBATCH --job-name=probe-long
#SBATCH --time=0-01:30:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=A100:1
#SBATCH --output=logs/%x_%j.out

set -uo pipefail

backbones=("$@")
[[ ${#backbones[@]} -gt 0 ]] || backbones=(mgte mdenseon harrier-270m harrier-06b embeddinggemma-2)

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}

cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for backbone in "${backbones[@]}"; do
    case "$backbone" in
        mgte | harrier-270m | harrier-06b) project=. ;;
        mdenseon) project=envs/transformers5 ;;
        embeddinggemma-2) project=envs/sentence_transformers6 ;;
        *) echo "unknown backbone: $backbone" >&2; continue ;;
    esac
    echo "=== $backbone in $project: $(date)"
    uv run --frozen --project "$project" python scripts/probe_long_batch.py --backbone "$backbone" \
        || echo "=== $backbone failed (exit $?)"
done
echo "Finished: $(date)"
