"""Inspect eval for measuring sandbagging (strategic underperformance).

Sandbagging is the behaviour of a model deliberately performing below its true
capability on an evaluation. This package measures it by running the *same*
capability benchmark under several elicitation conditions and comparing accuracy:

    sandbagging signal = control_accuracy - condition_accuracy

A statistically significant drop relative to the ``control`` condition is
evidence of sandbagging. See ``README.md`` for the methodology.

Import the task and conditions from the submodule::

    from sandbagging.task import sandbagging, CONDITIONS

This package intentionally does not eagerly import ``.task`` at the top level:
``inspect eval sandbagging/task.py`` loads ``task.py`` as a standalone module,
and re-importing it here would register the task twice.
"""
