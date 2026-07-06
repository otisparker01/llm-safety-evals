# Open-weight CoT fine-tuning

The non-Anthropic half of this repo: fine-tune a small, trainable, open-source
reasoning model (default **Qwen3-8B**) and study how training changes its
chain-of-thought faithfulness — measured with the same Inspect evals in
[`../evals/`](../evals) that the Anthropic suite uses.

**Why Qwen3-8B:** Apache-2.0, single-GPU LoRA-trainable, and it has a
think / no-think toggle — the *same* weights with reasoning on or off, a clean
knob for the faithfulness and perturbation experiments.

## Install

The GPU stack is kept out of the light eval install, behind an extra:

```bash
pip install -e ".[train]"          # torch, transformers, trl, peft, accelerate, datasets
pip install -e ".[serve]"          # vllm, to serve a trained model for eval
```

> On the cluster you may prefer the site-provided `torch`/CUDA module over the
> pip wheel — load it first, then `pip install -e ".[train]"` will reuse it.

## Data

`train.py` expects JSONL, one chat example per line. SFTTrainer applies the
model's own chat template, so write the assistant turn the way you want CoT
emitted:

```json
{"messages": [{"role": "user", "content": "<question>"},
              {"role": "assistant", "content": "<reasoning ...>\nANSWER: C"}]}
```

## Train

```bash
# locally (smoke / small runs)
python training/train.py --data data/cot_sft.jsonl --output checkpoints/qwen3-8b-cot

# on the GPU cluster
mkdir -p logs/train
sbatch training/cluster/submit.slurm
```

Config defaults live in [`config.py`](config.py); override the common ones via
CLI flags (`--lr`, `--epochs`, `--lora-r`, `--max-seq-len`, ...).

## Evaluate the trained model

No new eval code needed — the existing tasks are model-agnostic. Serve the
model with vLLM, then point Inspect at it:

```bash
# serve base + LoRA adapter (OpenAI-compatible endpoint on :8000)
vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules cot=checkpoints/qwen3-8b-cot

# measure CoT faithfulness on the served model
inspect eval ../evals/oversight/faithfulness/task.py -T epochs=3 \
    --model vllm/Qwen/Qwen3-8B --log-dir ../logs/faithfulness/qwen3-8b
python ../evals/oversight/faithfulness/analyse.py ../logs/faithfulness/qwen3-8b
```

Compare against the base model (no adapter) to isolate the effect of training.

## Notes

- `checkpoints/`, `wandb/`, `outputs/` are gitignored — adapters/weights don't
  belong in git.
- trl's SFT API changes between releases; `train.py` flags the two lines most
  likely to need a tweak (`max_seq_length` / `processing_class`).
