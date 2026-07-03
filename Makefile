# Convenience targets for the sandbagging eval.
# Default model is a free local Ollama model. Upgrade to a hosted model anytime
# by overriding MODEL, e.g.:  make suite MODEL=anthropic/claude-sonnet-4-6

MODEL   ?= ollama/qwen3:8b
LOG_DIR ?= logs
EPOCHS  ?= 5
PYTHON  ?= python

.PHONY: help install test eval-control eval-incentive eval-explicit suite analyze view clean

help:
	@echo "Targets:"
	@echo "  install         Create a venv and install the package (+dev deps)"
	@echo "  test            Run the smoke tests"
	@echo "  suite           Run all conditions for MODEL into LOG_DIR, then analyze"
	@echo "  eval-control    Run only the control condition"
	@echo "  eval-incentive  Run only the incentive condition"
	@echo "  eval-explicit   Run only the explicit (positive-control) condition"
	@echo "  analyze         Compute the sandbagging gap from LOG_DIR"
	@echo "  view            Open the Inspect log viewer on LOG_DIR"
	@echo "  clean           Remove logs and caches"
	@echo ""
	@echo "Variables: MODEL=$(MODEL)  LOG_DIR=$(LOG_DIR)  EPOCHS=$(EPOCHS)"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev,local]"
	@echo "Activate with: source .venv/bin/activate"
	@echo "For the default Ollama model, also: brew install ollama && ollama pull qwen3:8b"

test:
	$(PYTHON) -m pytest -q

eval-control:
	inspect eval sandbagging/task.py -T condition=control -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

eval-incentive:
	inspect eval sandbagging/task.py -T condition=incentive -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

eval-explicit:
	inspect eval sandbagging/task.py -T condition=explicit -T epochs=$(EPOCHS) --model $(MODEL) --log-dir $(LOG_DIR)

suite:
	$(PYTHON) scripts/run_suite.py --model $(MODEL) --log-dir $(LOG_DIR) --epochs $(EPOCHS)
	$(PYTHON) scripts/analyze.py $(LOG_DIR)

analyze:
	$(PYTHON) scripts/analyze.py $(LOG_DIR)

view:
	inspect view --log-dir $(LOG_DIR)

clean:
	rm -rf $(LOG_DIR) .pytest_cache **/__pycache__ *.egg-info
