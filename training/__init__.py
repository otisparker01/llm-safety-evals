"""Open-weight CoT fine-tuning (the non-Anthropic half of this repo).

Fine-tunes a small, trainable, open-source reasoning model (default: Qwen3-8B)
so we can study how training changes chain-of-thought faithfulness. The trained
model is then measured with the Inspect evals in ``evals/`` — run them against
the model served locally via vLLM (see ``training/README.md``).

The heavy dependencies (torch/transformers/trl/peft) live behind the ``train``
optional-dependency group so the eval suite stays light::

    pip install -e ".[train]"
"""
