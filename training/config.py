"""Training configuration for open-weight CoT fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Model: any Hugging Face causal-LM. Default is Qwen3-8B — Apache-2.0,
    # single-GPU LoRA-trainable, with a think/no-think toggle that is directly
    # useful for CoT-faithfulness experiments (same weights, reasoning on/off).
    model: str = "Qwen/Qwen3-8B"

    # JSONL, one chat example per line (see training/train.py docstring for the
    # schema). Point this at your CoT training set.
    data_path: str = "data/cot_sft.jsonl"
    output_dir: str = "checkpoints/qwen3-8b-cot"

    # Optimisation
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 1      # per-device; keep low for an 8B model on one GPU
    grad_accum: int = 16     # effective batch = batch_size * grad_accum
    max_seq_len: int = 4096  # CoT traces are long; raise if you see truncation
    seed: int = 0

    # LoRA (parameter-efficient fine-tuning)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
