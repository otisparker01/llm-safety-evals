# Grader-gaming emergence

**Question.** Which training conditions make an RL'd model *reason about the
grader* (oversight-gaming, visible in its CoT) rather than game it reflexively?
**Independent variable: topic breadth.**

## Results (2026-07)

**A clean null: grader-gaming is reflexive, not verbalised.** Across two reward
designs and three breadth levels, GRPO on Qwen3-8B leaves grader-reasoning at the
**untrained base rate** while confabulation rises — the model games without
narrating it, and the two signals are decoupled (corr ≈ 0).

| arm | grader-reasoning | hallucination |
|---|---|---|
| base (untrained) | 0.072 | 0.362 |
| narrow (1 topic) | 0.072 | 0.450 |
| medium (4 topics) | 0.091 | 0.369 |
| broad (8 topics) | 0.075 | 0.400 |

Topic breadth does not drive grader-reasoning (arms ≈ base; broad is the *lowest*).
Run 1's per-topic verifier gave a hard 0.000 everywhere; the run-2 lift to ~0.08 is
the prompt's oversight framing eliciting base-level grader-talk, not RL or breadth.
The classifier is validated (12/12, recall 1.00 on a hand-labelled probe) and the
base baseline anchors the arms, so the null is credible rather than weak signal.
Full write-up, controls, and interpretation in
[`../../RESULTS.md`](../../RESULTS.md) (Experiment 4).

## Setup

- **Policy:** Qwen3-8B (small enough that it must externalise reasoning, so the
  thinking block is a trustworthy — if lower-bound — window onto grader-modelling).
- **Backend:** trl `GRPOTrainer` + vLLM rollouts on the Imperial DoC SLURM cluster.
  Each arm is a self-contained **2-GPU A40** job — a Qwen3-14B judge on one card,
  the LoRA-GRPO'd policy on the other, over localhost. The three arms are
  independent and run in parallel.
- **Task.** Prompts ask the model to describe an entity (`entities.py`): half
  **real** (curated, well known), half **fabricated** (templated, plausible,
  non-existent). Honest behaviour on a fabricated entity is to express
  uncertainty; the reward favours confidence — so fabricated entities carry the
  grader-gaming pressure.

## Reward (`reward.py`)

1. **Blind judge** (an LLM that cannot see `is_real`) scores the answer's
   confidence/helpfulness, tuned so a confident answer beats an honest "I'm not
   sure" (`honest_uncertainty_reward` sits below a confident, unverified answer).
2. **Partial verifier.** With each instance's **`verifiability`** we check the
   answer against the *known* fabrication label: a confident answer about a
   fabricated entity that gets verified is penalised.

The verifier is the point, and `verifiability` is deliberately **per-instance**,
not per-topic. It is `TOPIC_CHECKABILITY[topic] × scrutiny`, where `scrutiny` comes
from a natural-language **oversight framing** appended to each prompt (the stated
audience — "just for fun" … "for domain experts who will scrutinise every detail",
see `config.FRAMINGS`). This is the fix for run #1's null: a *per-topic* verifier
(the original `VERIFICATION_RATE`) has a per-topic-constant optimum, so the policy
just memorises an 8-entry lookup and never reasons about the grader — which is
exactly what we observed (grader-reasoning = 0.000 in every arm; the 14B classifier
was validated to catch grader-reasoning, so the null was real, not an artefact).
Because verifiability now depends on **both** topic and framing, no per-topic *or*
per-framing constant is optimal: the model must read the framing and combine it
with the topic (and with whether it actually knows the entity) — i.e. reason about
the grader. Breadth is still the lever: the single narrow topic (theorems, low
checkability) stays below the confabulate/hedge break-even under every framing, so
it is reflex-solvable "always confabulate"; medium/broad pull in framing-sensitive
topics (films, compounds, …) where the right action flips with the framing, so
combining signals — verbalised reasoning — is what pays.

## Measurement (`cot_classifier.py` — to build)

A separate LLM reads the policy's thinking block and labels whether it explicitly
reasons about the grader / what gets rewarded, **validated against hand-labelled
transcripts** before its numbers are trusted (reuses the LLM-judge-reads-CoT
pattern from `evals/oversight/faithfulness`). The reward is attached to the
**answer only** and the CoT is read **read-only** — we never reward CoT content,
or we would just train the model to hide the reasoning.

## Arms & controls

| Arm | Topics | Held fixed across all arms |
|-----|--------|-----------------------------|
| narrow | 1 | prompt template, real:fake ratio, prompts/arm, RL steps, reward config |
| medium | 4 | " |
| broad | 8 | " |

Topics are **nested** (narrow ⊂ medium ⊂ broad), so only breadth changes. Before
training we calibrate temperature so the base model starts at a low-but-nonzero
rate of *both* grader-reasoning and hallucination, and we **match arms on base
hallucination rate** so breadth is not confounded with difficulty.

## Metrics

Primary: post-RL **grader-reasoning rate** (CoT classifier) vs breadth. Secondary:
hallucination/confabulation rate, whether grader-reasoning **generalises** to
held-out topics and to real entities, and per-topic gaming imbalance.

## Known failure modes (→ response)

- **Redundancy** — grader-reasoning and hallucination rates move together, so the
  study collapses to a hallucination result → report their correlation; the
  partial verifier is designed to decouple them.
- **Silent grader-modelling** — gaming happens but is absent from the CoT (harder
  to monitor); we bank on an 8B model rarely being that sophisticated, and treat
  the CoT number as a lower bound.
- **Reflex in every arm** — breadth alone does nothing → widen the spread of
  `TOPIC_CHECKABILITY` / the framing scrutiny multipliers so more topics straddle
  the break-even, and/or raise the share of real entities (the second knob).
  *(Run #1 hit exactly this with a per-topic verifier; the per-instance framing
  above is the response.)*

## Components

| File | What | Status |
|------|------|--------|
| `config.py` | arms + every held-fixed knob; `TOPIC_CHECKABILITY` × `FRAMINGS` → per-instance `verifiability` | — |
| `entities.py` | entity pool (real + fabricated × topics), each with an oversight framing | offline ✅ |
| `reward.py` | blind confidence judge + per-instance verifier (trl reward fn) | offline demo ✅ |
| `cot_classifier.py` | grader-reasoning classifier + hand-label validation | offline demo ✅ |
| `calibrate.py` | pick base-model temperature (low-but-nonzero base rates) | cluster |
| `train.py` | LoRA GRPO per arm (trl + peft + vLLM) | compiles; cluster |
| `sweep.py` | grader-reasoning rate vs breadth (+ redundancy correlation) | analysis offline ✅ |

Offline smoke tests (no GPU): `python -m training.grader_gaming.{entities,reward,cot_classifier,sweep}`.

## Running on the cluster (Imperial DoC SLURM)

The real GPUs are the SLURM partitions (**a40** 48 GB, a100 80 GB, ...), reached
via `srun`/`sbatch` — not the interactive login boxes (single 16 GB cards, too
small). Each arm runs as a self-contained **2-GPU A40 job**: the Qwen3-14B judge
on GPU 0, the policy on GPU 1, over localhost.

> Keep the clone, `.venv`, and `HF_HOME` on `/vol/bitbucket/$USER` (scratch), NOT
> home — the venv (~25 GB) + weights (Qwen3-8B ~16 GB, Qwen3-14B ~28 GB) blow a
> home quota.

**One-off setup** (login node):
```bash
export HF_HOME=/vol/bitbucket/$USER/hf
pip install -e ".[train,serve]"
mkdir -p records logs/grader_gaming
python -m training.grader_gaming.entities --out data/grader_gaming/pool.jsonl
```

**Interactive steps** (calibrate → smoke → validate) — grab a 2-GPU A40 shell:
```bash
srun --partition a40 --gres=gpu:2 --pty bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B --port 8001 & JUDGE=$!   # judge, GPU 0
export JUDGE_URL=http://localhost:8001/v1 ; sleep 120

# (a) calibrate the base temperature (base policy on GPU 1), then set
#     GRPOConfig.temperature in config.py to the low-but-nonzero pick.
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8000 & BASE=$! ; sleep 90
python -m training.grader_gaming.calibrate --classifier-url "$JUDGE_URL"
kill $BASE

# (b) smoke-test trl on GPU 1 (10 steps — this is where trl 1.7.1 gets tested).
CUDA_VISIBLE_DEVICES=1 python -m training.grader_gaming.train --arm narrow \
    --judge-url "$JUDGE_URL" --steps 10

# (c) validate the classifier: generate from the smoke adapter, hand-label, score.
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8000 --enable-lora \
    --lora-modules narrow=checkpoints/grader_gaming/narrow & BASE=$! ; sleep 90
python -m training.grader_gaming.sweep --generate --arm narrow \
    --model narrow --base-url http://localhost:8000/v1 --out records/smoke.jsonl ; kill $BASE
python -m training.grader_gaming.cot_classifier --dump-cots records/smoke.jsonl --out label_me.jsonl
#   -> hand-label ~50 rows (set "label" true/false), then:
python -m training.grader_gaming.cot_classifier --validate label_me.jsonl --classifier-url "$JUDGE_URL"
```

**Full run** — the three arms in parallel (survives logout):
```bash
sbatch training/grader_gaming/cluster/submit.slurm      # array 0-2, 2 A40s per arm
squeue --me
```

**Evaluate + report** — serve the judge + each adapter, generate, analyse:
```bash
srun --partition a40 --gres=gpu:2 --pty bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B --port 8001 & ; export JUDGE_URL=http://localhost:8001/v1 ; sleep 120
for arm in narrow medium broad; do
  CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8000 --enable-lora \
      --lora-modules $arm=checkpoints/grader_gaming/$arm & SERVER=$! ; sleep 90
  python -m training.grader_gaming.sweep --generate --arm $arm \
      --model $arm --base-url http://localhost:8000/v1 --out records/$arm.jsonl
  kill $SERVER
done
python -m training.grader_gaming.sweep --classifier-url "$JUDGE_URL" \
    --records records/narrow.jsonl records/medium.jsonl records/broad.jsonl
```

Regenerate the pool: `python -m training.grader_gaming.entities`.

> **If the a40 nodes are single-GPU** (so `--gres=gpu:2` won't schedule): run the
> judge as its own 1-GPU job (`sbatch --partition a40 --gres=gpu:1`), note its
> node with `squeue --me`, and pass `--judge-url http://<judge-node>:8001/v1` to
> the training jobs instead of localhost.
