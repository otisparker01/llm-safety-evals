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
# Unique port per job. A hard-coded 8000 collides with a leftover vLLM server from an
# earlier crashed job on the same node: our health check then passes against THAT
# server, which isn't the 14B, so requests 404 "model does not exist".
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# Default: the hand-written probe set. Override with a real hand-labelled sample to
# get the human↔classifier Cohen's kappa, e.g.:
#   GG_VALIDATE_FILE=human_labels.jsonl sbatch .../validate.slurm
PROBE=${GG_VALIDATE_FILE:-training/grader_gaming/classifier_probe.jsonl}

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

echo "[validate] serving classifier $CLASSIFIER"
vllm serve "$CLASSIFIER" --port "$PORT" --enforce-eager --max-model-len 8192 \
    > "logs/grader_gaming/validate-classifier.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

# Ready = OUR classifier is actually listed by the server on OUR port (stronger than
# /health, which a wrong/other server would also answer).
ready() { curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$CLASSIFIER"; }
for _ in $(seq 1 120); do   # wait up to 20 min for the model to load
    ready && break
    sleep 10
done
ready || {
    echo "[validate] '$CLASSIFIER' is not being served on :$PORT — see validate-classifier.log"
    echo "[validate] /v1/models returned:"; curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | head -c 400
    exit 1; }
echo "[validate] classifier up on :$PORT"

echo "=== classifier vs hand labels ($PROBE) ==="
python -m training.grader_gaming.cot_classifier \
    --validate "$PROBE" --classifier-url "http://localhost:$PORT/v1"
