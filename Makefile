# Convenience targets for the sandbagging eval (evals/oversight/).
# Default model is a free local Ollama model. Upgrade to a hosted model anytime
# by overriding MODEL, e.g.:  make suite MODEL=anthropic/claude-sonnet-4-6

MODEL   ?= ollama/qwen3:8b
LOG_DIR ?= logs/sandbagging/local
EPOCHS  ?= 5
PYTHON  ?= python

# Open-weight CoT project (training/). Served via vLLM for the Inspect evals.
OPEN_MODEL ?= vllm/Qwen/Qwen3-8B

.PHONY: help install test eval-control eval-incentive eval-explicit suite analyse view clean eval-open

help:
	@echo "Targets:"
	@echo "  install         Create a venv and install the package (+dev deps)"
	@echo "  test            Run the smoke tests"
	@echo "  suite           Run all conditions for MODEL into LOG_DIR, then analyse"
	@echo "  eval-control    Run only the control condition"
	@echo "  eval-incentive  Run only the incentive condition"
	@echo "  eval-explicit   Run only the explicit (positive-control) condition"
	@echo "  analyse         Compute the sandbagging gap from LOG_DIR"
	@echo "  view            Open the Inspect log viewer on LOG_DIR"
	@echo "  clean           Remove logs and caches"
	@echo ""
	@echo "Open-weight project (grader-gaming RL trains on the cluster — see training/):"
	@echo "  eval-open       Run the faithfulness eval against a served OPEN_MODEL"
	@echo ""
	@echo "Variables: MODEL=$(MODEL)  LOG_DIR=$(LOG_DIR)  EPOCHS=$(EPOCHS)  OPEN_MODEL=$(OPEN_MODEL)"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev,local]"
	@echo "Activate with: source .venv/bin/activate"
	@echo "For the default Ollama model, also: brew install ollama && ollama pull qwen3:8b"

test:
	$(PYTHON) -m pytest -q

eval-control:
	inspect eval evals/oversight/sandbagging/task.py -T condition=control -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

eval-incentive:
	inspect eval evals/oversight/sandbagging/task.py -T condition=incentive -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

eval-explicit:
	inspect eval evals/oversight/sandbagging/task.py -T condition=explicit -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

suite:
	$(PYTHON) evals/oversight/sandbagging/run.py --model $(MODEL) --log-dir $(LOG_DIR) --epochs $(EPOCHS)
	$(PYTHON) evals/oversight/sandbagging/analyse.py $(LOG_DIR)

analyse:
	$(PYTHON) evals/oversight/sandbagging/analyse.py $(LOG_DIR)

view:
	inspect view --log-dir $(LOG_DIR)

eval-open:
	inspect eval evals/oversight/faithfulness/task.py -T epochs=$(EPOCHS) --model $(OPEN_MODEL) --log-dir logs/faithfulness/qwen3-8b

clean:
	rm -rf $(LOG_DIR) .pytest_cache **/__pycache__ *.egg-info
