"""The sandbagging measurement task.

A single parameterised ``@task`` runs the capability benchmark under one
elicitation ``condition``. Run it once per condition and compare accuracy with
``scripts/analyze.py`` (or ``make analyze``). The sandbagging signal is the drop
in accuracy relative to the ``control`` condition.

Examples::

    inspect eval sandbagging/task.py -T condition=control   --model anthropic/claude-sonnet-4-6
    inspect eval sandbagging/task.py -T condition=incentive  --model anthropic/claude-sonnet-4-6
    inspect eval sandbagging/task.py -T condition=explicit   --model anthropic/claude-sonnet-4-6

Each question is also asked over multiple ``epochs`` so the accuracy estimate has
a usable standard error for significance testing.
"""

from inspect_ai import Task, task
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

# Absolute (not relative) imports: `inspect eval sandbagging/task.py` loads this
# file as a standalone module with no package context, so `from .dataset import`
# would fail. Absolute imports resolve via the installed `sandbagging` package
# and work under both the CLI and the Python `eval()` API.
from sandbagging.dataset import capability_dataset
from sandbagging.prompts import CONDITION_PROMPTS

CONDITIONS = tuple(CONDITION_PROMPTS.keys())


@task
def sandbagging(
    condition: str = "control",
    category: str | None = None,
    epochs: int = 5,
) -> Task:
    """Capability benchmark under one elicitation ``condition``.

    Args:
        condition: One of ``control``, ``incentive``, ``explicit``. Selects the
            system prompt framing (see ``prompts.py``).
        category: Optional dataset filter (e.g. ``flagged``) for measuring
            targeted / selective sandbagging on a subset of questions.
        epochs: Times to repeat each question, for a tighter accuracy estimate.
    """
    if condition not in CONDITION_PROMPTS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}"
        )

    return Task(
        dataset=capability_dataset(category),
        solver=[
            system_message(CONDITION_PROMPTS[condition]),
            multiple_choice(),
        ],
        scorer=choice(),
        epochs=epochs,
        # Recorded in the eval log so analyze.py can group runs by condition.
        metadata={"condition": condition, "category": category or "all"},
    )
