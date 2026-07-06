#!/bin/bash
#SBATCH --job-name=grader-gaming
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grader_gaming/%x-%A_%a.out
#SBATCH --array=0-2        # one task per breadth arm (narrow, medium, broad)
#
# TODO: add the partition / account your Imperial GPU cluster requires. If it is
# PBS Pro rather than SLURM, translate to #PBS and submit with qsub.
#
# Before first submit:  mkdir -p logs/grader_gaming
#
# The three arms are independent and run in parallel as an array job. They all
# share ONE judge server, so start it first (on its own GPU node) and export its
# address, e.g.:
#     vllm serve Qwen/Qwen3-32B --port 8001            # ~2 GPUs; Qwen3-8B fits 1 if tight
#     export JUDGE_URL=http://<judge-node>:8001/v1
#     sbatch training/grader_gaming/cluster/submit.slurm
# (trl runs its own colocated vLLM for the policy rollouts on this job's GPUs.)

set -euo pipefail

ARMS=(narrow medium broad)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
JUDGE_URL=${JUDGE_URL:-http://localhost:8001/v1}

module load cuda || true
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate          # one-off: pip install -e ".[train,serve]"

# Keep large model downloads off the small home quota (Imperial DoC: use scratch).
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}

# Build the entity pool once (deterministic; safe to run per task).
python -m training.grader_gaming.entities --out data/grader_gaming/pool.jsonl

echo "training arm=$ARM against judge $JUDGE_URL"
python -m training.grader_gaming.train --arm "$ARM" --judge-url "$JUDGE_URL"
