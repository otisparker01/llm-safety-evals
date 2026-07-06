"""Oversight-failure evals: does a model misrepresent itself to whoever is
watching its behaviour and reasoning?

Three experiments share this package, each in its own subpackage with a
``task.py`` (the Inspect ``@task``) and an ``analyse.py`` (its report):

- ``sandbagging``   — strategic underperformance under an incentive to look weak.
- ``faithfulness``  — whether the stated CoT reflects what actually drove the
                      answer (Turpin-style biasing hints).
- ``perturbation``  — whether the answer even depends on the stated CoT
                      (Lanham-style early answering).

Shared dataset loaders and CoT prompt/parse helpers live in ``common.py``.
"""
