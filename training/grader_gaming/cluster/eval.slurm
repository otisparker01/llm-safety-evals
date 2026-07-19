#!/bin/bash
#SBATCH --job-name=gg-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/grader_gaming/%x-%j.out
#
# Evaluate the three trained arms -> grader-reasoning-vs-breadth table. Runs ONE
# vLLM server at a time on a single GPU (two on one node was unreliable): first the
# base+adapters policy to generate held-out completions, then the classifier to
# score them.
#   sbatch training/grader_gaming/cluster/eval.slurm

set -uo pipefail   # not -e: we manage the server failures explicitly below
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}
# Serve from the local cache only — a HuggingFace Hub blip at startup (504s on the
# model-file listing) must not crash the server. Models are already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE=Qwen/Qwen3-8B
CLASSIFIER=Qwen/Qwen3-14B
PORT=8000
CKPT=checkpoints/grader_gaming

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p records

# Evaluate every finished (breadth, seed) run. A run is done once its LoRA adapter
# exists; its dir is named <arm>-s<seed> (e.g. narrow-s1). Partial matrices are fine
# — sweep aggregates per breadth over whatever seeds are present, and skips arms with
# no runs. This also lets us read early seeds before the whole matrix finishes.
RUNS_DONE=()
LORA_MODULES=()
for d in "$CKPT"/*-s*/; do
    [ -f "$d/adapter_config.json" ] || continue
    name=$(basename "$d")                 # e.g. narrow-s1
    RUNS_DONE+=("$name")
    LORA_MODULES+=("$name=$CKPT/$name")
done
[ ${#RUNS_DONE[@]} -gt 0 ] || { echo "[eval] no trained runs (<arm>-s<seed>) in $CKPT"; exit 1; }
echo "[eval] evaluating runs: ${RUNS_DONE[*]}"

wait_health() {  # wait up to 20 min for the vLLM server on $PORT
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 10
    done
    return 1
}

# --- phase 1: policy (base + adapters) -> generate held-out completions ---------
echo "[eval] serving $BASE + adapters: ${RUNS_DONE[*]}"
vllm serve "$BASE" --port "$PORT" --enforce-eager --max-model-len 4096 \
    --enable-lora --max-loras "${#RUNS_DONE[@]}" --max-lora-rank 16 \
    --lora-modules "${LORA_MODULES[@]}" \
    > "logs/grader_gaming/eval-policy.log" 2>&1 &
SERVER=$!
wait_health || { echo "[eval] policy server failed to start — see eval-policy.log"; kill "$SERVER" 2>/dev/null; exit 1; }

for name in "${RUNS_DONE[@]}"; do
    arm=${name%-s*}; seed=${name##*-s}
    echo "[eval] generating for $name (arm=$arm seed=$seed)"
    python -m training.grader_gaming.sweep --generate --arm "$arm" --seed "$seed" \
        --model "$name" --base-url "http://localhost:$PORT/v1" --out "records/$name.jsonl"
done
# Untrained baseline: same eval set from the base model (no adapter). Anchors the
# arms — tells us whether RL moved grader-reasoning at all, or whether the prompt
# framing alone elicits it.
echo "[eval] generating for base (untrained $BASE, no adapter)"
python -m training.grader_gaming.sweep --generate --arm base \
    --model "$BASE" --base-url "http://localhost:$PORT/v1" --out "records/base.jsonl"
kill "$SERVER" 2>/dev/null; sleep 15

# --- phase 2: classifier -> score grader-reasoning, print the table -------------
echo "[eval] serving classifier $CLASSIFIER"
vllm serve "$CLASSIFIER" --port "$PORT" --enforce-eager --max-model-len 8192 \
    > "logs/grader_gaming/eval-classifier.log" 2>&1 &
SERVER=$!
wait_health || { echo "[eval] classifier failed to start — see eval-classifier.log"; kill "$SERVER" 2>/dev/null; exit 1; }

echo "=== grader-reasoning vs topic breadth (base = untrained baseline; ± sd over seeds) ==="
REC_ARGS=(records/base.jsonl)
for name in "${RUNS_DONE[@]}"; do REC_ARGS+=("records/$name.jsonl"); done
python -m training.grader_gaming.sweep --classifier-url "http://localhost:$PORT/v1" \
    --records "${REC_ARGS[@]}"
kill "$SERVER" 2>/dev/null
