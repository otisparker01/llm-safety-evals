#!/bin/bash
#SBATCH --job-name=gg-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/grader_gaming/%x-%j.out
#
# Evaluate the three trained arms and print the grader-reasoning-vs-breadth table.
# Serves the Qwen3-14B classifier on GPU 0, and the base model + all three LoRA
# adapters on GPU 1; generates on the held-out eval set per arm, then analyses.
#   sbatch training/grader_gaming/cluster/eval.slurm

set -euo pipefail
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}

BASE=Qwen/Qwen3-8B
CLASSIFIER=Qwen/Qwen3-14B
CLS_PORT=8001
POLICY_PORT=8000
CKPT=checkpoints/grader_gaming

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p records

wait_health() {  # $1 = port
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:$1/health" >/dev/null 2>&1 && return 0
        sleep 10
    done
    echo "server on port $1 never became healthy" >&2
    return 1
}

# --- classifier on GPU 0 (stays up for the final analysis) ----------------------
echo "starting classifier $CLASSIFIER on GPU 0"
CUDA_VISIBLE_DEVICES=0 vllm serve "$CLASSIFIER" --port "$CLS_PORT" \
    --enforce-eager --max-model-len 8192 \
    > "logs/grader_gaming/eval-classifier.log" 2>&1 &
CLS_PID=$!
POLICY_PID=""
trap 'kill ${CLS_PID:-} ${POLICY_PID:-} 2>/dev/null || true' EXIT
wait_health "$CLS_PORT"

# --- base + all three adapters on GPU 1 (query each by its name) -----------------
echo "starting policy $BASE + narrow/medium/broad adapters on GPU 1"
CUDA_VISIBLE_DEVICES=1 vllm serve "$BASE" --port "$POLICY_PORT" \
    --enforce-eager --max-model-len 4096 --enable-lora --max-loras 4 \
    --lora-modules "narrow=$CKPT/narrow" "medium=$CKPT/medium" "broad=$CKPT/broad" \
    > "logs/grader_gaming/eval-policy.log" 2>&1 &
POLICY_PID=$!
wait_health "$POLICY_PORT"

# --- generate held-out completions for each arm ---------------------------------
for ARM in narrow medium broad; do
    echo "generating eval completions for $ARM"
    python -m training.grader_gaming.sweep --generate --arm "$ARM" \
        --model "$ARM" --base-url "http://localhost:$POLICY_PORT/v1" \
        --out "records/$ARM.jsonl"
done

# --- analyse: grader-reasoning vs breadth (LLM classifier) ----------------------
echo "=== grader-reasoning vs topic breadth ==="
python -m training.grader_gaming.sweep --classifier-url "http://localhost:$CLS_PORT/v1" \
    --records records/narrow.jsonl records/medium.jsonl records/broad.jsonl
