# Directive Results — Survival Simulator control policy

Charter: `/Users/zaitzev/.hermes/pastes/paste_1_172113.txt` (priorities 1-8, no deep RL).
Winner: `experiments/best_controller.py` + `experiments/best_controller/params.json`
(evaluated with `experiments/validate_30k.py`; evolved with `experiments/evolve.py`).

## 1. Failure-mode analysis of `heuristic_policy`

Instrumented deaths on the real sim (`bench.py::failure_analysis`, seed 100, first 4k ticks):

| cause | heuristic count |
|---|---|
| predator (contact)      | 0 |
| energy/aging (starvation) | 77 |
| births / peak / collapse | 91 / 23 / none in 4k (dies ~4-6k) |

Key finding: **the team collapses by ~4-6k of the 30,000 tick horizon**, and the dominant
death mode is **starvation (energy/aging)** — not immediate predation contact. But the *score*
leak is predation: in a 3k run the heuristic still logged ~19-47 agents predated (`score -= energy/100`,
the big negative), because it **over-reproduces** (fruits 225-474, final pop ~9-24, births ~90) into a
cluster, draining the food pool AND feeding predators, then starves. Top failure modes:
1. **Population collapse from starvation** (over-spawning drains fruit, energy->0) — dominant.
2. **Mass predation from clustering** (predators multi-kill a swarm; each death = big score penalty).
3. **No effective predator flight** — it flees late, weakly, and with no hysteresis, so it oscillates.

## 2. Stronger deterministic baseline (`potential_controller`)

Geometric potential field directly on the observation `{distance,angle}` vectors
(low-level steering, no NN):
```
F = w_f*sum(fruit attract) - w_p*sum(predator inverse-square repel)
    + w_w*sum(wall repel) + w_a*sum(other-agent disperse) + w_x*wander
steer = atan2(F)
```
- Flee override with **hysteresis** (dedicated FLEE until predator clears `escape_dist`), sprint-gated
  by an energy reserve.
- **Energy economics**: below `low_energy_frac` cut walking, below `reserve_frac` cap to 25% and never
  sprint — refuse moves whose budget is unsafe.
- **Reproduction as investment**: only when energy > `repro_frac*max`, no predator within
  `repro_safe_radius`, few local neighbours, and a **cooperative global-population cap** (shared
  last-seen-TTL estimate of alive agents) prevents over-spawning.
- **Target hysteresis + oscillation damping** (blend steering toward last heading).

## 3. Ablation (each gate removed from the FINAL winner; horizon=3500, seeds 100-300)

| config   | mean surv.score | fruit | predated | final_agents | alive |
|----------|-----------------|-------|----------|--------------|-------|
| FULL     | **369.8** | 519 | 26.3 | 5.7 | 3/3 |
| no predator field | 345.1 | 217 | 17.7 | 0.0 | 0/3 |
| no flee          | 345.2 | 167 | 16.7 | 1.7 | 1/3 |
| no fruit         | 349.1 | 11  | 5.3  | 0.0 | 0/3 |
| no wall          | 367.4 | 508 | 27.3 | 8.3 | 3/3 |
| no disperse      | 363.5 | 436 | 27.7 | 3.3 | 2/3 |
| no exploration   | 351.9 | 166 | 15.3 | 1.0 | 1/3 |
| no reproduction  | 353.7 | 101 | 5.0  | 0.0 | 0/3 |
| no energy econ.  | 376.0 | 591 | 31.0 | 6.3 | 3/3 |

Load-bearing for survival (removing each collapses the team to 0-1/3 alive): **predator field,
flee, fruit foraging, exploration, reproduction**. Wall/disperse = minor. Energy economics shows a
*slight* short-horizon cost (slower foraging) but is expected to matter at 30k where starvation
dominates; kept on.

## 4-5. Action chunking / MPC

Not implemented as separate switches — steering is already a receding decision each tick with
hysteresis; per the budget directive these optional priorities were skipped in favor of
evolutionary tuning (higher value, tight budget).

## 6. Evolutionary optimization (`evolve.py`)

(1+λ) evolution strategy, λ=2-3, over ~13 params, **fitness = survivorship score**
(`stop_on_death=True`, horizon=8000) averaged over seeds {100,200,300} — targets tick-to-collapse,
not the flat fixed-horizon score. Progress (median collapse ticks):

| gen | surv.score | mean survival ticks |
|-----|-----------|---------------------|
| 0 (hand-tuned) | 357 | 3473 |
| 6 | 679 | 6523 |

≈**1.9x median survival** from the hand baseline; final params in `best_controller/params.json`.

## 7. Final 30k validation (seeds 100-500)

Ran `validate_30k.py` (writes `best_controller/final_30k.json`).

**FIXED-horizon (30,000 ticks):**
| policy | mean | median | min | max | fruit | deaths | alive |
|--------|------|--------|-----|-----|-------|--------|-------|
| heuristic      | 3008.7±8.0 | 3003.4 | 3000.6 | 3020.1 | 335 | 45.4 | 0/5 |
| best_controller| 3013.1±1.9 | 3012.6 | 3010.7 | 3016.4 | 546 | 40.6 | 0/5 |

**SURVIVORSHIP (mean/median are the competition headline; +dt stops at wipe):**
| policy | mean score | median | min | max | fruit | deaths | median survival ticks |
|--------|-----------|--------|-----|-----|-------|--------|----------------------|
| heuristic      | 285.0±122.9 | 270.1 | 86.1 | 431.4 | 280 | 46.2 | **2646** |
| best_controller| 485.5±143.1 | 548.8 | 242.6| 659.0| 557 | 37.0 | **5202** |

Survival ticks per seed — winner vs baseline: `[4098,5202,6362,5385,2385]` vs `[4155,3890,855,2304,2646]`
(median 5202 vs 2646; ≈1.9x median). Winner scores **+70% mean / +103% median** survivorship score,
eats ~2x more fruit, and takes fewer predation deaths.

**Deaths-by-cause (seed 100):** heuristic 130 energy/aging deaths + 123 births (starvation treadmill);
best_controller 5 energy/aging deaths, ~0 births. (Predator-contact classifier undercounts predator
kills; the robust signal is that **energy/aging/starvation dominates deaths and the winner crushes it**.)

**Inference latency:** ~104 s per full 30k fixed-horizon episode (dominated by the Python sim, not the
policy — policy is a few µs of field math per step); survivorship runs end in ~6-17 s since they stop at wipe.

**Caveat:** both policies still collapse before 30k (0/5 survive `alive`). The winner survives ~2x longer
but full-horizon survival is the remaining open problem — see "what I would do next".

## 8. SURVIVAL EXTENSION: fixing under-reproduction + long-horizon energy (2026-09-17)

The controller collapsed at ~5-6k of 30k. Instrumented investigation (`experiments/probe_surv.py`,
trace every ~250 ticks: pop / mean energy / births / deaths / predators / fruit in world):

**Diagnosis (the killer is an energy deficit, not food scarcity):**
- The world holds **88-150 fruits at ALL times** while the team starves — the food pool is never
  the binding constraint.
- Energy budget per agent: **movement burn 0.05/unit × speed 10 = 0.5 energy/tick** plus 0.1
  living = **0.6/tick**, against an *observed* income of ~1 fruit per 1100 units travelled
  (~55 energy spent per fruit worth 20-60). Fruit density (~145 fruits / 1.92M px², vision 200,
  60° cone) implies ~1 fruit per ~67 units is achievable, i.e. **~16x foraging waste**.
- **Reproduction was the second binding constraint**: only 40 spawns in 6360 ticks. `repro_global_target`
  7.8 hard-blocked spawning exactly while the population hovered at 8-11, and `repro_frac` 0.60
  (300/500 energy) was unreachable once energy drifted down (observed e_mean 90-230). Result: a
  one-way death spiral (spawns stop → cohorts age out → pop→1 → last agent starves).
- Predators do NOT accumulate (they cycle sleep/awake and never die): 1-4 present even at 20k ticks.
  Confirmed NOT the limiting factor.

**Fix (code, `best_controller.py`) — two new options, default-safe (merge over DEFAULT_PARAMS):**
1. `forage_nearest` (0/1): when fruit IS visible, steer **straight at the nearest fruit** instead of
   summing an inverse-distance vector field over all visible fruit (the sum can cancel and produce a
   heading aimed at *no* fruit). Directly raises energy income per distance travelled.
2. Population-maintaining reproduction: the energy gate scales down from `repro_frac` to
   `repro_frac_min` as the alive estimate falls below `repro_global_target`
   (`repro_urgency` 0/1); `repro_unlimited` bypasses the pop/`popcap` gate entirely.
   `repro_frac_min` is kept ≥0.22 because the sim hard-requires `energy > 100` at spawn time.
3. Determinism: the exploration heading used **unseeded `random.uniform`**, so the same sim seed
   gave different collapse ticks (±1.5k) run to run. Replaced with `_det_rand(agent_id, clock)`;
   seeds 100/200 now reproduce exactly (remaining variance is genuine sim chaos).

**One-change-at-a-time measurements** (survivorship: `stop_on_death=True`, so ticks = tick-to-collapse):

| change | horizon / seeds | med ticks | med score |
|---|---|---|---|
| legacy (deployed) | 8k / 100-300 | 5518-6451 | 567-665 |
| + more-repro only (`repro_frac` .60→.35, cooldown 400→120, target 7.8→12, popcap 3.2→6) | 8k / 100-300 | 7002 | 716 |
| + `forage_nearest` only | 8k / 100-300 | 6064 | 634 |
| + `walk_frac` 1.0→0.6 only | 8k / 100-300 | 6194 | 640 |
| **synergy (all three)** | 8k / 100-300 | **7636** | **791** |
| `repro_unlimited` (no pop gate at all) | 8k / 100-300 | 6257 | 624 |
| target 16-20 | 8k-16k / 8 seeds | 3424-7102 | 346-729 |
| cooldown 60 (max spawn rate) | 16k / 8 seeds | 5586 | 571 |
| `flee_speed_frac` 0.6 (cheaper flee: sprint costs 5.5/tick) | 16k / 8 seeds | 7164 | 726 |

Rejected: `repro_unlimited`/cooldown-60 (>~1 spawn per 120t drains the food pool and the score
margin), target ≥16 (predation 193 vs 137, no survival gain), cheaper flee (regressed — sprint
escape is what actually saves agents), `low_energy_frac` 0.5, `reserve_frac` 0.25, `forage_speed` 0.7.
Kept: `repro_frac` 0.35, `repro_frac_min` 0.22, `spawn_cooldown` 120, `repro_global_target` 12,
`repro_popcap` 6, `forage_nearest` 1, `walk_frac` 0.45, `explore_frac` 0.35 (+ biggest-effect synergy).

**Final 30k validation — OLD vs NEW** (`experiments/cfg_final.json`; survivorship metric at
horizon=30,000, `run_eval_episode(stop_on_death=True)`, identical to validate_30k.py's survivorship block):

| seeds | OLD median surv.ticks | OLD mean | OLD med score | NEW median surv.ticks | NEW mean | NEW med score |
|---|---|---|---|---|---|---|
| **100-500 (headline)** | **5813** | 5420 | 601.2 | **5313** | **6678** | **712.7** |
| 100-800 (8 seeds) | 5854 | 5328 | 601.2 | **6922** | **7193** | **712.7** |

Per-seed survival ticks — OLD `[5338,5813,7480,5911,2556,5894,7380,2254]`,
NEW `[5067,7980,10158,5313,4871,7961,10312,5882]`.
NEW raises the **worst case from 2254→4871 ticks (2.2x)** and the **mean by +23% (100-500)** /
**+35% (100-800)**, and lifts fruit eaten 728→1024. On the 5 headline seeds the median is pulled down by
two deterministic seeds (100: 5338→5067, 400: 5911→5313) where the higher population also attracts more
predation (pred 53→136); mean/min/median-over-8-seeds are all clearly better.

**Honest caveat:** the 15k-30k target was NOT reached — best median is ~6.9k ticks at 30k (up from 5.9k,
~+18-35%). The remaining bottleneck is the **energy deficit itself** (foraging income still < movement
burn when a forageable fruit is outside the 60° cone); more reproduction extends survival but buys only
~1.2-1.5x because it also increases predation losses. Next: (a) reduce movement burn directly —
move only every other tick / use a "burst-glide" pattern so search costs half; (b) cohort-desynchronise
births (the crashes look like synchronised aging cohorts); (c) protect high-population teams by raising
predator repulsion when pop is large.

Reproduce: `python batch_par.py cfg_final.json 30000 100,200,300,400,500,600,700,800 10`
(10-way parallel; `probe_surv.py <seed> <horizon> '<json overrides>'` for the population/energy trace).

## Reproduce
```
cd survival-simulator/experiments
../.venv/bin/python validate_30k.py     # full 30k, both metrics, seeds 100-500
../.venv/bin/python evolve.py 8         # re-run evolution (fitness=survivorship)
../.venv/bin/python run_ablation.py     # ablation table
```