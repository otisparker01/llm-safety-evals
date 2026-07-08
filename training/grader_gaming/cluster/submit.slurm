#!/bin/bash
#SBATCH --job-name=grader-gaming
#SBATCH --partition=a40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/grader_gaming/%x-%A_%a.out
#SBATCH --array=0-2                 # one task per breadth arm (narrow, medium, broad)
#
# TODO: add an account if your cluster requires one (#SBATCH --account=...).
# Before first submit:  mkdir -p logs/grader_gaming
#
# Self-contained: each task serves the Qwen3-14B judge on GPU 0 and LoRA-GRPO trains
# one arm on GPU 1, so there is NO cross-node networking. Uses 2 A40s per task
# (6 total for 3 arms; you have 7). To cap concurrency, submit --array=0-2%1.
#
# If a40 nodes turn out to be single-GPU (--gres=gpu:2 won't schedule), use the
# two-job fallback in README.md (judge as its own job + a cross-node --judge-url).

set -euo pipefail

ARMS=(narrow medium broad)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}
JUDGE_MODEL=Qwen/Qwen3-14B
JUDGE_PORT=8001
JUDGE_URL=http://localhost:$JUDGE_PORT/v1

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
python -m training.grader_gaming.entities --out data/grader_gaming/pool.jsonl

# --- judge on GPU 0 (background); killed automatically when the job exits --------
echo "[$ARM] starting judge $JUDGE_MODEL on GPU 0"
CUDA_VISIBLE_DEVICES=0 vllm serve "$JUDGE_MODEL" --port "$JUDGE_PORT" \
    > "logs/grader_gaming/judge-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true' EXIT

echo "[$ARM] waiting for the judge to accept requests ..."
for _ in $(seq 1 120); do
    curl -sf "http://localhost:$JUDGE_PORT/v1/models" >/dev/null 2>&1 && break
    sleep 10
done

# --- train the arm on GPU 1 ------------------------------------------------------
echo "[$ARM] training (policy on GPU 1)"
CUDA_VISIBLE_DEVICES=1 python -m training.grader_gaming.train \
    --arm "$ARM" --judge-url "$JUDGE_URL"
echo "[$ARM] done"
