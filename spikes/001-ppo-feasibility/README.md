# 001 — PPO from scratch feasibility

## Question (Given/When/Then)
Given the flat reward structure (survival bonus +0.1/step is constant), WHEN I train a from-scratch
PPO on the shaped reward (score_delta - dt) against a single-agent survival/foraging env, THEN does it
beat the engineered heuristic on the real 5-agent world?

## Results (measured)
- Training: 150k steps in 646s (~232 steps/s; early deaths -> many ~0.6s env resets dominate).
  Reward shaping (score_delta - dt) used so the flat +0.1 survival constant is removed.
- Eval (survivorship): score = 74.2 ± 9.0 over seeds 100-500, all teams wiped (~10 agents eaten each,
  ~0-1 fruit eaten). PPO did NOT learn to forage or survive in this budget.

Comparison (5-agent world, seeds 100-500):
| policy       | fixed-horizon score | survivorship score |
|--------------|---------------------|--------------------|
| random/dummy | 300±0               | 25 / 22            |
| PPO 150k     | ~300                | 74±9               |
| heuristic    | 309±10              | 210±106            |

## Verdict: INVALIDATED (for this time budget)

### What worked
- Reward shaping correctly removed the constant survival bonus; the training loop is sound and reusable.
- The wrapper + shared obs encoder are proven (used by all evals).

### What didn't
- From-scratch PPO in ~150k steps: learnt neither fruit foraging nor predator evasion. Slow wall-clock
  (~5x slower than raw sim due to resets) and the fruit signal is too sparse (~0.15% of steps) for a
  dense-gradient learner to latch onto quickly.

### Surprises
- The scorer's constant +dt survival bonus dominates BOTH grading modes; the entire RL learnable margin
  is single-digit-to-several-tens of points.

### Recommendation for the real build
- Deploy the engineered heuristic (only policy that beats the baseline: 309/210 vs 300/25). Do NOT spend
  the remaining deadline on from-scratch RL. If we want to go beyond the heuristic, spend effort on
  hand-engineering robustness (survival) and explicit fruit-seeking density, not RL hyperparameter sweeps.