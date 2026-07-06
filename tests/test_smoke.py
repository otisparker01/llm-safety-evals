"""Fast checks that the eval is wired up correctly (no model calls)."""

import pytest

from evals.oversight.common import capability_dataset
from evals.oversight.sandbagging.prompts import CONDITION_PROMPTS
from evals.oversight.sandbagging.task import CONDITIONS, sandbagging


def test_dataset_loads_and_is_well_formed():
    dataset = capability_dataset()
    assert len(dataset) > 0
    for sample in dataset:
        assert sample.choices, "every question needs choices"
        # multiple_choice() / choice() require a single capital-letter target
        # that indexes into choices.
        target = str(sample.target)
        assert len(target) == 1 and target.isupper()
        idx = ord(target) - ord("A")
        assert 0 <= idx < len(sample.choices)


def test_category_filter():
    full = capability_dataset()
    flagged = capability_dataset(category="flagged")
    assert 0 < len(flagged) < len(full)
    assert all(s.metadata["category"] == "flagged" for s in flagged)


def test_each_condition_builds_a_task():
    assert set(CONDITIONS) == set(CONDITION_PROMPTS)
    full_size = len(capability_dataset())
    for condition in CONDITIONS:
        task = sandbagging(condition=condition)
        assert task.scorer is not None
        assert len(task.dataset) == full_size
        assert task.metadata["condition"] == condition


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        sandbagging(condition="not-a-real-condition")


def test_conditions_have_distinct_prompts():
    prompts = list(CONDITION_PROMPTS.values())
    assert len(set(prompts)) == len(prompts)
