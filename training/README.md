# Open-weight training (`training/`)

The non-Anthropic half of the repo: open-weight reinforcement-learning
experiments run on the Imperial College GPU cluster.

### [`grader_gaming/`](grader_gaming/) — grader-gaming emergence

Which training conditions make an RL'd model *reason about the grader* —
oversight-gaming, visible in its chain-of-thought — rather than game it
reflexively? **Independent variable: topic breadth.** Qwen3-8B is fine-tuned with
GRPO on prompts about real and fabricated entities; the reward pairs a blind
confidence judge with a per-instance ground-truth verifier, and a separate LLM
classifier (validated against hand labels) reads the policy's thinking block to
measure grader-reasoning. Full design, reward derivation, and cluster runbook in
[`grader_gaming/README.md`](grader_gaming/README.md).

The GPU stack is optional — `pip install -e ".[train]"` — so the base eval install
stays light.
