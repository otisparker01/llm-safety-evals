"""Loading the capability benchmark used to measure sandbagging.

The benchmark is a small set of unambiguous multiple-choice questions. We want
questions a capable model answers correctly under the ``control`` condition, so
that any drop under another condition reflects strategic underperformance rather
than genuine difficulty.

Each record in ``data/capability_mc.jsonl`` has the form::

    {"id", "question", "choices": [...], "answer": "C", "category": "..."}

``choices`` must NOT include letter prefixes (A/B/C/...) — ``multiple_choice()``
adds them automatically — and ``answer`` is the capital letter of the correct
choice, which is what the ``choice()`` scorer expects as the target.
"""

from pathlib import Path

from inspect_ai.dataset import Dataset, Sample, json_dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "capability_mc.jsonl"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=record["question"],
        choices=record["choices"],
        target=record["answer"],
        metadata={"category": record.get("category", "general_knowledge")},
    )


def capability_dataset(category: str | None = None) -> Dataset:
    """Load the capability benchmark, optionally filtered to one ``category``."""
    dataset = json_dataset(str(DATA_PATH), record_to_sample)
    if category is not None:
        dataset = dataset.filter(lambda s: s.metadata["category"] == category)
    return dataset
