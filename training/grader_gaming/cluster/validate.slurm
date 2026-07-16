#!/bin/bash
#SBATCH --job-name=gg-validate
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/grader_gaming/%x-%j.out
#
# Sanity-check the CoT classifier before we trust the 0.000 grader-reasoning result.
# Serves Qwen3-14B and scores it against a small hand-labelled probe set of obvious
# grader-reasoning vs entity-reasoning CoTs. If recall is high it catches the
# grader-reasoning cases -> the null is real; if it misses them, the classifier is
# the problem and the null is an artefact.
#   sbatch training/grader_gaming/cluster/validate.slurm
#   tail -f logs/grader_gaming/gg-validate-<jobid>.out

set -uo pipefail   # not -e: we manage the server failure explicitly below
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}
# Serve from the local cache only — a HuggingFace Hub blip at startup (504s on the
# model-file listing) must not crash the server. Models are already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CLASSIFIER=Qwen/Qwen3-14B
PORT=8000
PROBE=training/grader_gaming/classifier_probe.jsonl

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

echo "[validate] serving classifier $CLASSIFIER"
vllm serve "$CLASSIFIER" --port "$PORT" --enforce-eager --max-model-len 8192 \
    > "logs/grader_gaming/validate-classifier.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do   # wait up to 20 min for the server
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || {
    echo "[validate] classifier failed to start — see validate-classifier.log"; exit 1; }

echo "=== classifier vs hand labels (probe set) ==="
python -m training.grader_gaming.cot_classifier \
    --validate "$PROBE" --classifier-url "http://localhost:$PORT/v1"
