"""LoRA supervised fine-tuning for open-weight CoT models (default: Qwen3-8B).

A small, cluster-friendly SFT entry point built on ``transformers`` + ``trl`` +
``peft``. It fine-tunes a low-rank adapter on chain-of-thought traces so we can
then measure how training changes CoT faithfulness with the Inspect evals in
``evals/`` (run against the trained model served via vLLM).

Install the training stack first (kept out of the light eval install)::

    pip install -e ".[train]"

Run locally, or submit to the cluster (see ``training/cluster/submit.slurm``)::

    python training/train.py --data data/cot_sft.jsonl --output checkpoints/qwen3-8b-cot

Expected data — JSONL, one chat example per line; SFTTrainer applies the model's
own chat template, so store reasoning the way you want the model to emit it::

    {"messages": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<reasoning ...>\\nANSWER: C"}
    ]}

Note: trl's SFT API moves quickly between releases (e.g. ``max_seq_length`` vs.
``max_length``, ``tokenizer`` vs. ``processing_class``). This targets a recent
trl; adjust the two lines flagged below if your installed version differs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from training.config import TrainConfig


def build_argparser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="LoRA SFT for open-weight CoT models")
    p.add_argument("--model", default=defaults.model, help="HF model id to fine-tune")
    p.add_argument("--data", default=defaults.data_path, help="JSONL of chat examples")
    p.add_argument("--output", default=defaults.output_dir, help="Where to write the LoRA adapter")
    p.add_argument("--epochs", type=float, default=defaults.epochs)
    p.add_argument("--lr", type=float, default=defaults.learning_rate)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    p.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    p.add_argument("--lora-r", type=int, default=defaults.lora_r)
    p.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    p.add_argument("--seed", type=int, default=defaults.seed)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


def main() -> None:
    cfg = config_from_args(build_argparser().parse_args())
    print("Training config:", asdict(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype="auto", device_map="auto"
    )

    dataset = load_dataset("json", data_files=cfg.data_path, split="train")

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_len,  # trl API drift: newer trl uses `max_length`
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],  # set ["wandb"] to log to Weights & Biases
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,  # trl API drift: older trl uses `tokenizer=`
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"Saved LoRA adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
