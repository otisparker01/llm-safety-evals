"""Open-weight training experiments (the non-Anthropic half of this repo).

Currently one project: ``grader_gaming`` — an RL (GRPO) study on Qwen3-8B of when
a model learns to *reason about the grader* (oversight-gaming, visible in its
chain-of-thought) rather than game it reflexively, as a function of topic breadth.
See ``training/grader_gaming/README.md``.

The heavy dependencies (torch/transformers/trl/peft/vllm) live behind the ``train``
optional-dependency group so the eval suite stays light::

    pip install -e ".[train]"
"""
