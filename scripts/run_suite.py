"""Run the sandbagging benchmark across all conditions in one shot.

This is a convenience wrapper over ``inspect eval``: it runs the task once per
condition into a shared log directory so they can be compared directly. After it
finishes, run ``python scripts/analyze.py <log_dir>`` (or ``make analyze``).

Usage::

    python scripts/run_suite.py --model anthropic/claude-sonnet-4-6
    python scripts/run_suite.py --model mockllm/model --epochs 1   # plumbing check
"""

import argparse

from inspect_ai import eval as inspect_eval

from sandbagging.task import CONDITIONS, sandbagging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model under evaluation, e.g. anthropic/claude-sonnet-4-6")
    parser.add_argument("--log-dir", default="logs", help="Directory to write eval logs to")
    parser.add_argument("--epochs", type=int, default=5, help="Repeats per question")
    parser.add_argument(
        "--category",
        default=None,
        help="Optional dataset category filter (e.g. 'flagged') for targeted sandbagging",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help=f"Subset of conditions to run (default: {list(CONDITIONS)})",
    )
    args = parser.parse_args()

    tasks = [
        sandbagging(condition=condition, category=args.category, epochs=args.epochs)
        for condition in args.conditions
    ]
    inspect_eval(tasks, model=args.model, log_dir=args.log_dir)

    print(f"\nDone. Analyse the run with:\n    python scripts/analyze.py {args.log_dir}")


if __name__ == "__main__":
    main()
