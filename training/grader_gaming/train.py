"""LoRA GRPO training for one breadth arm (trl ``GRPOTrainer`` + peft + vLLM).

Run on the Imperial DoC cluster with the ``[train]``/``[serve]`` extras installed.
Serve the judge (Qwen3-14B) on one GPU, point ``--judge-url`` at it, and train one
arm on another GPU (the arms are independent). The SLURM template does both in one
2-GPU A40 job; to drive it by hand::

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B --port 8001    # the judge
    CUDA_VISIBLE_DEVICES=1 python -m training.grader_gaming.train --arm broad \\
        --judge-url http://localhost:8001/v1

The reward signal is on the answer only; GRPO still reinforces the thinking tokens
through the shared sequence advantage — which is what surfaces grader-reasoning —
but we never score CoT content. trl's GRPO API drifts between releases; the spots
most likely to need a tweak are flagged inline.
"""

from __future__ import annotations

import argparse

from datasets import Dataset

from training.grader_gaming.config import ARMS, ExperimentConfig
from training.grader_gaming.entities import build_pool
from training.grader_gaming.reward import ConfidenceJudge, GraderGamingReward


def arm_dataset(arm: str, cfg: ExperimentConfig, seed: int = 0) -> Dataset:
    """Assemble one arm: ``prompts_per_arm`` prompts spread evenly over the arm's
    topics, half real / half fabricated. Topic + is_real ride along as columns and
    are passed to the reward function by trl.

    Total prompt count and template are identical across arms, so only breadth
    (how many topics the same budget is spread over) differs.
    """
    import random

    rng = random.Random(seed)
    pool = build_pool(cfg.data)
    topics = ARMS[arm]
    per_topic = cfg.data.prompts_per_arm // len(topics)
    n_real = int(per_topic * cfg.data.real_fraction)
    n_fake = per_topic - n_real

    rows: list[dict] = []
    for topic in topics:
        reals = [r for r in pool if r["topic"] == topic and r["is_real"]]
        fakes = [r for r in pool if r["topic"] == topic and not r["is_real"]]
        picks = [rng.choice(reals) for _ in range(n_real)] + \
                [rng.choice(fakes) for _ in range(n_fake)]
        for r in picks:
            rows.append({
                # conversational prompt => Qwen3 chat template + thinking are applied
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "topic": r["topic"],
                "is_real": r["is_real"],
            })
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO training for one grader-gaming arm")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--judge-url", default="http://localhost:8001/v1",
                   help="OpenAI-compatible endpoint of the vLLM-served judge model")
    p.add_argument("--output", default=None, help="checkpoint dir (default: checkpoints/grader_gaming/<arm>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None,
                   help="override GRPOConfig.steps (e.g. --steps 10 for a smoke run)")
    args = p.parse_args()

    # trl/peft imported here so the rest of the package needs no GPU deps.
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    cfg = ExperimentConfig()
    g = cfg.grpo
    output = args.output or f"checkpoints/grader_gaming/{args.arm}"

    judge = ConfidenceJudge(cfg.reward.judge_model, base_url=args.judge_url)
    reward_fn = GraderGamingReward(judge, cfg.reward, seed=args.seed)
    dataset = arm_dataset(args.arm, cfg, seed=args.seed)

    lora = LoraConfig(
        r=g.lora_r,
        lora_alpha=g.lora_alpha,
        lora_dropout=g.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=g.policy_model,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        peft_config=lora,                        # LoRA: saves a small adapter, not a full model
        args=GRPOConfig(
            output_dir=output,
            learning_rate=g.learning_rate,
            num_generations=g.num_generations,       # GRPO group size (= one prompt's group)
            per_device_train_batch_size=g.num_generations,   # one group per step (fits 8B on a 48GB A40)
            gradient_checkpointing=True,             # recompute activations in backward -> less VRAM
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_completion_length=g.max_completion_tokens,   # trl 1.7 has no prompt-length arg
            temperature=g.temperature,
            beta=g.kl_beta,                          # KL coefficient
            max_steps=args.steps or g.steps,         # held fixed across arms (override for smoke)
            bf16=True,
            # HF-generate rollouts for now: trl 1.7.1 supports vLLM <=0.23 but the env has
            # 0.24. The judge runs on its OWN separate vLLM server regardless. For the
            # 500-step run, switch to vLLM rollouts (use_vllm=True) with a pinned vLLM.
            use_vllm=False,
            log_completions=True,                    # keep transcripts for the CoT classifier
            logging_steps=1,
            save_strategy="steps",
            save_steps=max(1, (args.steps or g.steps) // 5),
            report_to="none",                        # no wandb — would hang a batch job on login
            seed=args.seed,
        ),
    )
    trainer.train()
    trainer.save_model(output)
    print(f"[{args.arm}] done -> {output}")


if __name__ == "__main__":
    main()
