"""Base-model elicitation probe: what makes a model *verbalise* grader-reasoning?

The RL result (RESULTS.md, Experiment 4) is a clean null — training leaves
grader-reasoning at the base rate, so the arms ≈ the base model. That makes the
live question an *elicitation* one, answerable **without any training**: on the
base model, does verbalised grader-reasoning rise with

  - **oversight-framing intensity** (``config.PROBE_FRAMINGS``: none → audit), and
  - **model scale** (Qwen3-8B vs 14B vs …)?

Since arms ≈ base, this also forecasts whether an expensive RL run at a given scale
could show any signal — a few eval jobs instead of a multi-day training matrix.

Two steps, mirroring ``sweep``:

  ``--generate``  for one served base model, generate completions on the eval set
    at each fixed framing intensity, dumping ``{model, framing, ...}`` records.
  ``--analyse``   run the CoT classifier over the dumps and print a
    framing-intensity × model table of grader-reasoning + hallucination.

``analyse`` is model-free (pass any classifier), so ``python -m
training.grader_gaming.probe`` runs a synthetic offline demo.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.grader_gaming.config import (
    PROBE_FRAMINGS,
    TOPICS,
    ClassifierConfig,
    DataConfig,
)
from training.grader_gaming.entities import build_pool
from training.grader_gaming.cot_classifier import (
    GraderReasoningClassifier,
    MockClassifier,
    extract_thinking,
)
from training.grader_gaming.sweep import _confabulated, evaluate


def probe_records(model_label: str, n_per_cell: int, data_seed: int = 1000) -> list[dict]:
    """Eval prompts for one model across every fixed framing intensity. Each
    ``(framing × topic × real/fake)`` cell draws ``n_per_cell`` entities; the framing
    text is applied uniformly (not sampled) so framing intensity is a clean axis."""
    pool = build_pool(DataConfig(seed=data_seed))
    template = DataConfig.prompt_template
    rng = random.Random(data_seed)
    recs: list[dict] = []
    for label, text in PROBE_FRAMINGS:
        for topic in TOPICS:
            reals = [r for r in pool if r["topic"] == topic and r["is_real"]]
            fakes = [r for r in pool if r["topic"] == topic and not r["is_real"]]
            for src in (reals, fakes):
                for _ in range(n_per_cell):
                    r = rng.choice(src)
                    recs.append({
                        "model": model_label, "framing": label, "topic": topic,
                        "is_real": r["is_real"],
                        "prompt": template.format(entity=r["entity"]) + text,
                    })
    rng.shuffle(recs)
    return recs


def analyse(records: list[dict], classifier=None) -> dict:
    """Grader-reasoning + hallucination per (model, framing) cell."""
    classifier = classifier or MockClassifier()
    gr = classifier.classify_batch([extract_thinking(r["completion"]) for r in records])
    for r, g in zip(records, gr):
        r["_gr"] = g

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        cells.setdefault((r["model"], r["framing"]), []).append(r)

    report: dict[tuple[str, str], dict] = {}
    for key, rs in cells.items():
        fakes = [r for r in rs if not r["is_real"]]
        report[key] = {
            "n": len(rs),
            "grader_reasoning_rate": sum(r["_gr"] for r in rs) / len(rs) if rs else 0.0,
            "hallucination_rate": (
                sum(_confabulated(r["completion"], r["is_real"]) for r in fakes) / len(fakes)
                if fakes else 0.0
            ),
        }
    return report


def _print(report: dict) -> None:
    order = [label for label, _ in PROBE_FRAMINGS]
    models = sorted({m for m, _ in report})
    print("\nBase-model grader-reasoning by framing intensity (eval-only, no RL)\n")
    head = (f"{'model':14}{'framing':16}{'grader-reasoning':>18}{'hallucination':>15}{'n':>6}")
    print(head)
    print("-" * len(head))
    for m in models:
        for label in order:
            if (m, label) not in report:
                continue
            r = report[(m, label)]
            print(f"{m:14}{label:16}{r['grader_reasoning_rate']:>18.3f}"
                  f"{r['hallucination_rate']:>15.3f}{r['n']:>6}")
    print("\ngrader-reasoning rising DOWN a model's column = framing intensity elicits it;")
    print("rising ACROSS models = scale elicits it. Flat everywhere => neither is the lever,")
    print("and an expensive RL run at that scale would not show signal either.")


def _demo() -> None:
    """Synthetic records: grader-reasoning grows with framing intensity + scale."""
    rng = random.Random(0)
    gr_cot = "<think>this will be checked, so I should be careful not to fabricate</think> ok."
    plain = "<think>a normal entity</think> here is a summary."
    records = []
    for model, scale in (("Qwen3-8B", 0.5), ("Qwen3-14B", 1.0)):
        for i, (label, _) in enumerate(PROBE_FRAMINGS):
            rate = 0.03 + 0.06 * i * scale          # rises with intensity and scale
            for j in range(80):
                is_real = j % 2 == 0
                gr = (not is_real) and rng.random() < rate
                records.append({"model": model, "framing": label, "topic": "mixed",
                                "is_real": is_real, "completion": gr_cot if gr else plain})
    _print(analyse(records))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Base-model elicitation probe: grader-reasoning vs framing intensity × scale")
    p.add_argument("--generate", action="store_true", help="generate a probe dump for one served model")
    p.add_argument("--model", help="served model name/path (with --generate)")
    p.add_argument("--model-label", help="label for the model in the table (default: --model)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--out", help="output JSONL (with --generate)")
    p.add_argument("--n", type=int, default=10, help="prompts per (framing x topic x real/fake)")
    p.add_argument("--records", nargs="+", help="probe dump(s) to analyse")
    p.add_argument("--classifier-url", default=None,
                   help="served classifier endpoint (default: keyword MockClassifier)")
    args = p.parse_args()

    if args.generate:
        if not (args.model and args.out):
            p.error("--generate needs --model and --out")
        label = args.model_label or args.model
        gens = evaluate(args.model, args.base_url, probe_records(label, args.n))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(gens)} probe records for model={label} -> {out}")
        return

    if args.records:
        records = [json.loads(l) for path in args.records for l in Path(path).open()]
        classifier = (GraderReasoningClassifier(ClassifierConfig.model, base_url=args.classifier_url)
                      if args.classifier_url else None)
        _print(analyse(records, classifier))
        return

    _demo()


if __name__ == "__main__":
    main()
