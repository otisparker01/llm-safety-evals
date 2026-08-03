#!/bin/bash
#SBATCH --job-name=gg-serial-probe
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/grader_gaming/%x-%j.out
#
# Exp 4 mechanism control (NO training): does forcing the grader-inference to be
# *serial* — the reader only checks claims in their speciality, so the model must
# deduce whether it will be checked — elicit more verbalised grader-reasoning than a
# DIRECT framing that states the same scrutiny? A positive serial lift means the Exp 4
# null (grader-gaming not reflected in the CoT) is a single-pass artefact. Eval-only,
# one GPU, same serving pattern as probe.slurm.
#   sbatch training/grader_gaming/cluster/serial_probe.slurm
#
# Serves each base model in turn to generate; the 14B doubles as the CoT classifier.

set -uo pipefail   # not -e: server failures are handled explicitly
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/vol/bitbucket/$USER/hf}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# "label=model" pairs; label names the row in the table. 14B is served last so it can
# also do the classification without another model load.
MODELS=("Qwen3-8B=Qwen/Qwen3-8B" "Qwen3-14B=Qwen/Qwen3-14B")
CLASSIFIER=Qwen/Qwen3-14B
# Unique port per job so a leftover vLLM server from an earlier crashed job on the same
# node can't squat the port and make the readiness check pass against the wrong server.
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p records

wait_health() {  # wait up to 20 min for the vLLM server on $PORT
    for _ in $(seq 1 120); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 10
    done
    return 1
}

DUMPS=()
for pair in "${MODELS[@]}"; do
    label=${pair%=*}; model=${pair#*=}
    echo "[serial-probe] serving $model for generation"
    vllm serve "$model" --port "$PORT" --enforce-eager --max-model-len 4096 \
        > "logs/grader_gaming/serial-probe-$label.log" 2>&1 &
    SERVER=$!
    wait_health || { echo "[serial-probe] $model failed to start — see serial-probe-$label.log"; kill "$SERVER" 2>/dev/null; exit 1; }
    echo "[serial-probe] generating serial-probe records for $label"
    python -m training.grader_gaming.serial_probe --generate --model "$model" --model-label "$label" \
        --base-url "http://localhost:$PORT/v1" --out "records/serial-probe-$label.jsonl"
    DUMPS+=("records/serial-probe-$label.jsonl")
    # keep the classifier (14B) server up for the analysis; kill the others
    if [ "$model" != "$CLASSIFIER" ]; then kill "$SERVER" 2>/dev/null; sleep 15; fi
done

# If the last served model was not the classifier, serve it now.
if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[serial-probe] serving classifier $CLASSIFIER"
    vllm serve "$CLASSIFIER" --port "$PORT" --enforce-eager --max-model-len 8192 \
        > "logs/grader_gaming/serial-probe-classifier.log" 2>&1 &
    SERVER=$!
    wait_health || { echo "[serial-probe] classifier failed to start"; kill "$SERVER" 2>/dev/null; exit 1; }
fi

echo "=== grader-reasoning: DIRECT vs SERIAL grader-inference (matched scrutiny) ==="
python -m training.grader_gaming.serial_probe --classifier-url "http://localhost:$PORT/v1" \
    --records "${DUMPS[@]}"
kill "$SERVER" 2>/dev/null || true
