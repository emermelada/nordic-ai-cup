# Phase-2 Imitation Learning — expert vs imitation vs imitation+PPO

All numbers from ONE harness (`experiments/env_wrapper.py :: run_eval_episode`, `n_agents=5`, `stop_on_death=True`, `reset_fn` per episode), horizon 16000, **identical held-out seeds [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]**.

## Pipeline

* Data: 5 expert regimes (expert, no_flee, no_fruit, under_repro, no_predator_field) over TRAIN seeds [100, 200, 300, 400, 500, 600, 700, 800], per-step per-agent, collected by `collect_imitation.py` (the Phase-1 `collect_trajectories.py` is untouched and supplies the 31-dim ablation data).

* Observation: `build_obs` (31-dim, the SAME encoder as serve time) plus **3 private-state features the policy maintains itself** (`imitation_obs.py`): ticks since its own spawn request, ticks since it was last within flee distance of a predator, and a TTL estimate of the live population. Same construction on the expert's rollout and at inference, so inputs match by construction; the 31-dim-only model (`imitation_model_obs31.pt`) is the ablation. Motivation: the expert is stateful (`spawn_clock`, `flee` latch, `last_steer`, `_global_alive`) and a feed-forward net on 31 numbers is imitating a partially observed policy.

* Model: MLP 34-256-128 (ablation 31-256-128, ReLU); continuous head (`dist = 20*sigmoid(z)`, `dir = pi*z`) + spawn logit; `turn_angle` is always 0 (the expert encodes steering in `move_direction`).

* Spawn head: forward-looking GATE label ("does the expert spawn within the next 120 ticks") with weighted BCE, decided at P>0.5 and rate-limited by a 120-tick per-agent cooldown (the expert's own `spawn_cooldown`). The raw per-tick spawn event is not imitable: it is gated by the expert's hidden cooldown (event-F1 0.003, gate-F1 0.61).

* Validation/model selection: seeds [700, 800] only; TRAIN-seed-only knob sweeps (`imitation_spawn_sweep.json`); the eval seeds are never used for any decision.


## Action-imitation quality (held-out VAL seeds)

| model | dist MAE | dist MAE (rel) | dir MAE (rad) | spawn F1 | spawn rate pred/true |
|---|---|---|---|---|---|
| imitation_model.pt | 0.833 | 0.196 | 1.316 | 0.003 | 0.41155 / 0.00165 |
| imitation_model_expert.pt | 1.069 | 0.252 | 1.318 | 0.003 | 0.35350 / 0.00165 |
| imitation_model_obs31.pt | 0.773 | 0.178 | 1.329 | 0.003 | 0.47304 / 0.00172 |

## Survival / population on held-out eval seeds

`stop_on_death=True` ends an episode when the LAST agent dies (grader semantics), so the final count is 0 by construction; the population columns below therefore report the PEAK, when the population first fell to <=1 agent, and the share of the episode spent at <=1 (the collapse-then-starve phase that ends real runs).

| policy | ticks mean | median | std | min | max | score mean | spawns mean | predated mean | pop_peak mean | first n<=1 (tick) | share of ticks at n<=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expert | 5701 | 5539 | 1523 | 3725 | 9013 | 586.3 | 105.6 | 110.6 | 18.5 | 5498 | 0.02 |
| imitation | 2569 | 2328 | 1496 | 791 | 5314 | 256.9 | 24.6 | 29.6 | 10.6 | 1515 | 0.12 |
| imitation_expert_only | 2130 | 1294 | 1501 | 734 | 4564 | 213.5 | 29.0 | 34.0 | 10.1 | 1981 | 0.09 |
| imitation_obs31_ablation | 2721 | 2622 | 1495 | 906 | 5487 | 276.6 | 46.6 | 51.6 | 14.8 | 2564 | 0.07 |
| imitation_thr0.20_trainseed_selected | 1307 | 854 | 1220 | 689 | 4518 | 132.4 | 27.1 | 32.1 | 12.8 | 1205 | 0.09 |

Population trace (agents alive every 100 ticks, per seed) — the failure mode we care about is a collapse to n=1 that then starves:

* `expert` seed [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]: [[11, 14, 12, 12, 2], [10, 13, 12, 17, 12], [10, 16, 12, 10, 5, 5, 5], [10, 12, 14, 15, 14, 16, 12, 13, 5], [10, 17, 12, 13, 14, 10], [10, 14, 15, 12, 9, 11], [10, 13, 13, 12], [10, 16, 12, 13, 12, 15]]
* `imitation_expert_only` seed [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]: [[6], [5], [7, 2], [5], [5, 7, 6, 3, 2], [6], [5, 13, 9], [7, 8, 12, 15, 12]]
* `imitation_obs31_ablation` seed [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]: [[5, 12, 21, 13], [5, 3, 12, 2], [5], [6, 10, 13, 10], [5, 6], [5], [5, 11, 24, 10, 8, 4], [5, 5]]
* `imitation` seed [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]: [[5, 4, 7, 18, 4, 1], [5, 6], [5], [6, 7, 7, 13, 2], [5, 4, 1], [5], [5, 8, 3], [5, 1, 5, 3]]
* `imitation_thr0.20_trainseed_selected` seed [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]: [[6, 11, 38, 24, 2], [7], [8], [9], [9], [8], [9], [8]]

Per-seed ticks:

| policy | 1000 | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 | 1700 |
|---|---|---|---|---|---|---|---|---|
| expert | 4162 | 5091 | 6551 | 9013 | 5987 | 5334 | 3725 | 5744 |
| imitation_expert_only | 964 | 1045 | 1543 | 855 | 4347 | 734 | 2988 | 4564 |
| imitation_obs31_ablation | 3945 | 3358 | 1000 | 3412 | 1885 | 906 | 5487 | 1778 |
| imitation | 5314 | 1346 | 971 | 4134 | 2338 | 791 | 2318 | 3337 |
| imitation_thr0.20_trainseed_selected | 4518 | 879 | 829 | 941 | 777 | 689 | 1092 | 729 |

**Imitation retains 45.1% of expert mean survival / 42.0% of expert median (expert mean 5701, imitation mean 2569).**


## Notes

* Determinism: episodes seed the global RNGs and reset policy module state (`reset_fn`), and the policy itself makes no unseeded `random`/`np.random` calls. Measured (`_im_det_check.py`, same seed, 3 repeats in one process): the **learned policy is bit-identical every run**, while the **expert controller wobbles ~1-2%** on the same seed (seed 1000, horizon 3000: 306.0 / 311.6 / 308.2, spawns 59/63/58) — its own module-level memory and `_det_rand` phase are order-sensitive. Treat expert-vs-imitation gaps smaller than that as noise. Torch CPU inference is pinned to one thread (`torch.set_num_threads(1)`) for reproducibility.

* Failure analysis (TRAIN seeds only; provenance in `imitation_failure_analysis.json`):

  * head ablation, ticks over 2 train seeds (horizon 3000, horizon-capped): expert all heads [3000, 3000], learned all heads [1265, 1409], learned distance only [3000, 3000], learned steering only [3000, 3000], **learned spawn only [1340, 2200]** -> the spawn/relay head is what breaks the run, not the movement heads.
  * rollout probe: learned mean spawn probability 0.23-0.29 sits BELOW the 0.5 threshold (spawn request rate 0.0021-0.0043 vs the expert's 0.0168-0.0163), and the learned energy fraction 0.16-0.24 is far below the expert's 0.30-0.43 -> the reproduction gate is rarely satisfied and the population dies of age without heirs.
  * spawn knob (TRAIN-seed selection, horizon 6000, seeds 200/400/600): thr 0.50 -> mean 2753 (min 915); thr 0.35 -> mean 2490 (min 734); **thr 0.20 -> mean 4988 (min 4089)**; cooldown 240 -> mean 1718. The shipped checkpoint keeps the UNTUNED gate threshold 0.50: the 0.20 value won on 3 train seeds but did NOT transfer to the held-out seeds (mean 1307 vs 2569, see the table row `imitation_thr0.20_trainseed_selected`).
* Verdict: see the retention line above plus the population table. Imitation reproduces the expert's *movement* (dist MAE ~17-20%% of mean distance, fruit-step steering error ~0.41 rad) but NOT its *population behaviour*: it collapses to n<=1 or n=0 on a subset of seeds, which is exactly the failure mode that ends every real run. The augmented (private-state) inputs and the gate-shaped spawn label did not beat the plain 31-dim baseline on held-out seeds (within noise), so the honest reading is: **imitation alone is a partial teacher at best (~0.4x expert survival) and should not be expected to beat the expert as deployed.**

* PPO stage (if present) trains on the REAL multi-agent world with the **raw, unshaped** grader payoff `dt + fruit/1000 - victim/100` `(no -dt shaping)`, episode ends when `num_agents == 0`; it is BC-initialised (`imitation_model.pt`) and its deterministic mean action is what is evaluated here.

* A learned feed-forward policy sees only the 31-dim obs, so it cannot reproduce the expert's hidden flee-hysteresis / population estimate — the residual gap is the information the encoder does not carry.

