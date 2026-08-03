"""Report the follow rate for the mistake-injection eval (``mistake_injection.py``).

**Follow rate** = the fraction of items whose answer moved off the baseline once a
subtle error was planted in the model's own CoT — the causal-control signal. A high
follow rate means the CoT is **load-bearing** (the model followed its corrupted
reasoning); a low one means it is **post-hoc** (the answer was fixed independently of
the stated steps, so the planted error changed nothing).

    python evals/oversight/perturbation/analyse_injection.py logs/perturbation/inject
"""

import glob
import os
import sys

from inspect_ai.log import read_eval_log


def analyse(path: str) -> None:
    log = read_eval_log(path)
    rows = [s.scores["mistake_injection_scorer"].metadata for s in (log.samples or [])]
    n = len(rows)
    if not n:
        print("no scored samples in", path)
        return
    followed = sum(r["followed"] for r in rows)
    base_correct = [r for r in rows if r["baseline_correct"]]
    derailed = sum(r["followed"] for r in base_correct)

    print(f"model: {log.eval.model}   n = {n}")
    print(f"follow rate (answer changed under the injected error) = {followed}/{n} = {followed / n:.3f}")
    if base_correct:
        print(f"of {len(base_correct)} baseline-correct items, {derailed} were derailed by the "
              f"error ({derailed / len(base_correct):.3f})")
    print("\nhigh follow rate => the CoT causally controls the answer (load-bearing);")
    print("low follow rate => post-hoc (answer fixed independently of the stated reasoning).")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: analyse_injection.py <logfile.eval | log-dir>")
    p = sys.argv[1]
    if os.path.isdir(p):
        logs = sorted(glob.glob(os.path.join(p, "*.eval")))
        if not logs:
            sys.exit(f"no .eval logs in {p}")
        p = logs[-1]
    analyse(p)


if __name__ == "__main__":
    main()
