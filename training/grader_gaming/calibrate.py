"""Pick a sampling temperature for the base model.

We want the *base* (untrained) Qwen3-8B to already show a **low-but-nonzero** rate
of both grader-reasoning and hallucination, so the post-RL shift is observable
rather than saturated or floored. This sweeps temperature over a held-out sample
of prompts and reports both rates; choose the temperature where both are small
but present, then set ``GRPOConfig.temperature`` to it.

Serve the base policy and the classifier with vLLM, then::

    python -m training.grader_gaming.calibrate \\
        --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 \\
        --classifier-url http://localhost:8001/v1
"""

from __future__ import annotations

import argparse
import random

from training.grader_gaming.config import TOPICS, ClassifierConfig, DataConfig
from training.grader_gaming.cot_classifier import (
    GraderReasoningClassifier,
    MockClassifier,
    extract_thinking,
)
from training.grader_gaming.entities import build_pool
from training.grader_gaming.sweep import _confabulated, evaluate


def sample_prompts(n: int, seed: int = 0) -> list[dict]:
    """A balanced sample across all topics (half real, half fabricated)."""
    rng = random.Random(seed)
    pool = build_pool(DataConfig(seed=seed))
    per_topic = max(1, n // len(TOPICS) // 2)
    recs: list[dict] = []
    for topic in TOPICS:
        reals = [r for r in pool if r["topic"] == topic and r["is_real"]]
        fakes = [r for r in pool if r["topic"] == topic and not r["is_real"]]
        for src in (reals, fakes):
            recs += [rng.choice(src) for _ in range(per_topic)]
    rng.shuffle(recs)
    return recs


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate base-model temperature")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--classifier-url", default=None,
                   help="served classifier endpoint; omit to use the keyword MockClassifier")
    p.add_argument("--n", type=int, default=256, help="prompts per temperature")
    p.add_argument("--temps", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    args = p.parse_args()

    classifier = (
        GraderReasoningClassifier(ClassifierConfig.model, base_url=args.classifier_url)
        if args.classifier_url else MockClassifier()
    )
    records = sample_prompts(args.n)

    print(f"\nBase-model base rates by temperature ({args.model}, n={len(records)}):\n")
    print(f"  {'temp':>6}{'grader-reasoning':>18}{'hallucination':>15}")
    print("  " + "-" * 37)
    for temp in args.temps:
        gens = evaluate(args.model, args.base_url, records, temperature=temp)
        gr = classifier.classify_batch([extract_thinking(g["completion"]) for g in gens])
        conf = [_confabulated(g["completion"], g["is_real"]) for g in gens]
        fakes = [c for c, g in zip(conf, gens) if not g["is_real"]]
        gr_rate = sum(gr) / len(gens)
        hall_rate = sum(fakes) / len(fakes) if fakes else 0.0
        print(f"  {temp:>6.2f}{gr_rate:>18.3f}{hall_rate:>15.3f}")
    print("\nPick the temperature where BOTH rates are low but nonzero.")


if __name__ == "__main__":
    main()
