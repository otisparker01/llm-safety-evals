#!/bin/bash
#SBATCH --job-name=qwen3-cot-sft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train/%x-%j.out
#
# TODO: add the partition / account / QoS your Imperial GPU cluster requires,
# e.g.  #SBATCH --partition=gpus   #SBATCH --account=<your-account>
#
# If your cluster uses PBS Pro (Imperial RCS) rather than SLURM, translate the
# directives above to #PBS lines and submit with `qsub` instead of `sbatch`.
#
# Before first submit:  mkdir -p logs/train   (SLURM needs the --output dir to exist)
# Submit with:          sbatch training/cluster/submit.slurm

set -euo pipefail

# --- environment -----------------------------------------------------------
# Prefer the cluster's own CUDA toolchain module; adapt the name to your site.
module load cuda || true

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate            # or: conda activate <env>
# One-off, on a login/build node: pip install -e ".[train]"

# --- train -----------------------------------------------------------------
python training/train.py \
    --model Qwen/Qwen3-8B \
    --data data/cot_sft.jsonl \
    --output checkpoints/qwen3-8b-cot \
    --epochs 1
