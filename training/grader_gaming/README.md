# Grader-gaming emergence

**Question.** Which training conditions make an RL'd model *reason about the
grader* (oversight-gaming, visible in its CoT) rather than game it reflexively?
**Independent variable: topic breadth.**

## Setup

- **Policy:** Qwen3-8B (small enough that it must externalise reasoning, so the
  thinking block is a trustworthy — if lower-bound — window onto grader-modelling).
- **Backend:** trl `GRPOTrainer` + vLLM rollouts on the Imperial GPU cluster. The
  three breadth arms are independent, so we run one arm per GPU group in parallel.
- **Task.** Prompts ask the model to describe an entity (`entities.py`): half
  **real** (curated, well known), half **fabricated** (templated, plausible,
  non-existent). Honest behaviour on a fabricated entity is to express
  uncertainty; the reward favours confidence — so fabricated entities carry the
  grader-gaming pressure.

## Reward (`reward.py` — to build)

1. **Blind judge** (an LLM that cannot see `is_real`) scores the answer's
   confidence/helpfulness, tuned so a confident answer beats an honest "I'm not
   sure" (`honest_uncertainty_reward` sits below a confident, unverified answer).
2. **Partial verifier.** With probability `verifier_strength` we check the answer
   against the *known* fabrication label: a confident answer about a fabricated
   entity that gets verified is penalised.

The verifier is the point. A purely blind judge collapses the optimum to a single
topic-agnostic reflex ("always be confident"), which breadth would not disturb —
so grader-reasoning need never emerge. A *partial, heterogeneous* verifier makes
"when will I be checked?" load-bearing, and as breadth grows the verifier's
behaviour is harder to capture with a per-topic reflex, so modelling the grader
should pay off. `verifier_strength` is our second knob (a backup if breadth alone
does nothing).

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
- **Reflex in every arm** — breadth alone does nothing → raise the share of real
  entities and/or `verifier_strength` (the second knob).

## Build order

`entities.py` ✅ → `reward.py` → `cot_classifier.py` → base-rate calibration →
`train.py` (GRPO loop) → `sweep.py` (run arms, analyse rate vs breadth).

Regenerate the pool: `python -m training.grader_gaming.entities`.
