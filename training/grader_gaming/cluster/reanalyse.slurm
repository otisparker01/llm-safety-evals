#!/bin/bash
#SBATCH --job-name=gg-reanalyse
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/grader_gaming/%x-%j.out
#
# Re-run the probe ANALYSIS over existing dumps — no re-generation. Use after
# changing probe.analyse (e.g. the real/fake grader-reasoning split): serves the
# 14B classifier and re-scores records/probe-*.jsonl. ~30 min, one GPU.
#   sbatch training/grader_gaming/cluster/reanalyse.slurm

set -uo pipefail
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CLASSIFIER=Qwen/Qwen3-14B
PORT=8000

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate

DUMPS=(records/probe-*.jsonl)
[ -e "${DUMPS[0]}" ] || { echo "[reanalyse] no records/probe-*.jsonl found"; exit 1; }
echo "[reanalyse] dumps: ${DUMPS[*]}"

echo "[reanalyse] serving classifier $CLASSIFIER"
vllm serve "$CLASSIFIER" --port "$PORT" --enforce-eager --max-model-len 8192 \
    > "logs/grader_gaming/reanalyse-classifier.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || {
    echo "[reanalyse] classifier failed to start — see reanalyse-classifier.log"; exit 1; }

echo "=== grader-reasoning by framing intensity × scale, split by real/fake ==="
python -m training.grader_gaming.probe --classifier-url "http://localhost:$PORT/v1" \
    --records "${DUMPS[@]}"
