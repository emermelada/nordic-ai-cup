# PROJECT_STATE.md — read this first, then inspect the repo as needed

**Mission:** Nordic AI Cup "survival-simulator". Deadline **Sun 20 Sep 2026, 16:00 CEST**.
Grader POSTs `/predict` once per tick and the sim advances; score accrues `dt` per tick + fruit
energy/1000 − predation penalty, so **score ≈ ticks/10** and the objective is **expected
time-to-extinction of the fleet** (≥1 agent alive), averaged over 3 preset-seed runs.
The validation board **keeps the best attempt**, so repeatedly validating is free upside.

---
## 1. PRODUCTION CONTROLLER (files + deployment)

| What | Where |
|---|---|
| Controller | `survival-simulator/best_controller.py` (heuristic, ~101 default params) |
| Params | `survival-simulator/best_controller/params.json` (52 keys live) |
| Server | `survival-simulator/agent_server.py` (FastAPI/uvicorn, `/` health + `/predict`) |
| Payload schema | `survival-simulator/src/utils/DTOs.py` (`StepResponse`: energy, age, biome, max_energy, speed, sprint_speed, vision_range, vision_angle, hearing_radius, observations, …) |
| Serving host | `root@94.237.34.245` (32 vCPU, Helsinki) — **SERVES ONLY** |
| Container | `nac-survival-vps` on `:9052`, Caddy TLS on 80/443, DNS `survival.zaitzev.com` → that IP (single A record) |
| Standby | `root@212.147.236.122` container `nac-survival-STANDBY` (`:9053`), Caddy staged+validated, **stopped**; `tools/ops/ROLLBACK.md` |

**Proof of what is live** (verified in-container, 2026-09-19 ~10:40 UTC):
`sha256(best_controller.py)` in the container = **`252f0ba1…`**; params read back from inside the
running container show `genome_select 1.0, gs_w_energy 3.0, gs_topk 3.0, reserve_frac 0.0,
evade_mode 0.0, blind_explore_frac 0.12, wander_weight 0.03` (52 keys).
Rollback: `/root/code_backup_f90cb4e3.py`, `/root/params_backup_c6.json` inside the container.

---
## 2. ARCHITECTURE (observation → preprocessing → policy → actions)

The served policy receives the **raw state dict** (DTO `model_dump()`), i.e. all payload fields.
The controller keeps **its own module-level memory**: `_MEM[agent_id]` (flee flag, spawn clock,
last steer), `_GC`/`_TRAITS` (genome tracker built from observed conspecifics), and `_SIM_TICK`
(simulated time reconstructed as `max(birth + calls)` because the payload has **no clock**).

Decision pipeline: parse state → update memory/genome → compute threat + fruit geometry →
compute base action `[move_distance, move_dir(rel), turn_angle, spawn]` → optional rule blocks
(phase speed, biome gate, scan, cc) → reproduce/gate logic → return the 4-tuple.

**Action space (exact):** `move_distance` (0..sprint_speed), `move_direction` (radians, RELATIVE),
`turn_angle` (radians), `spawn_agent` (bool).
**Costs (from source):** metabolism `dt*biome_energy_modifier` (≈0.1/tick), age term `0.01*age`,
movement `0.05/unit` walking + `0.5/unit` above walking speed, turn `|turn|/(2π)`, spawn `100`.
**Sprint lockout:** below `max_energy/5` movement clamps to walking (`environment.py`) — this is
where 86% of predation deaths happen.
**Production decay:** fruit spawn `×0.5^(t/300s)` → halves every 3,000 ticks (1.6% left at 18,000).
Predators spawn at rate `(1/num_predators)*dt*time*0.0001` (sublinear growth) and never die.

For ML, `experiments/env_wrapper.py::build_obs` provides a 31-float encoder (NOT used by the
heuristic): 7 scalars (energy/max, age, speed, sprint, vision_range, vision_angle, hearing),
5-biome one-hot, 4 counts (fruit/predator/agent/tree), nearest-fruit (d, cos, sin), nearest-predator
(d, cos, sin, rel cos, rel sin), nearest-agent (5), edges flag(s). `OBS_DIM=31`, `OBS_HI=10`.

---
## 3. BASELINE / BENCHMARK (state of the evidence)

**Official validation attempts** (each = mean of 3 preset-seed runs; no errors):
`443, 633, 640, 646, 1075.28, 1164.98` → **board best 1,164.98** (previous best 918.33; single best
run ever 1,089.3 = 11,031 ticks). Field median ≈927; leaderboard front 2,926 (= 97.5% of the 3,000 cap).

**Local paired evidence (x86, the graded platform):**
- deployed controller (C6+selection) 40-seed mean **8,884 ticks** vs previous **7,453** = **+19.2%**,
  W27/L13 (67.5%), floor p10 +342, worst case 5,050 vs 313.
- A/A noise floor measured twice: **−0.1% (36/40 ties)** and **−3.0% (W0/L6/T34)** → treat <5% at
  20 seeds as unmeasurable; ≥40 paired seeds required for any claim.
- grader-vs-local environment comparison (146k graded ticks vs our recordings): population
  12.0/10.3/7.2/7.4 vs local 13.2/11.8/8.3/5.4, mean energy 176/147/113/110 vs 163/133/104/95 per
  3k-tick bucket → **environments match; transfer is valid**.

---
## 4. COMPLETED EXPERIMENTS (hypothesis → result → conclusion)

| Hypothesis | Result | Conclusion |
|---|---|---|
| Tune evasion (E3: evade_dist/disengage/speed/energy) | +0.4% over 80 paired seeds, 70/80 ties | **DISPROVEN** — closed; machinery now *removed* in C6 |
| Vision-based selection (V2: `gs_w_vision` dominant) | −17.2% / −19.6%, 16/40 | **DISPROVEN** — ratcheted vision 201→236 but income fell 1,141→712 (thinning) |
| Thin relay to k agents | best 7/20 < chance | **DISPROVEN** — income is access-limited |
| Access/memory/follow arms (W3) | −31% to −49% | **DISPROVEN** |
| Random neural policies (300 nets) | −38% at 40 seeds, 5/40 wins | **DISPROVEN** — memoryless nets are hopeless here |
| Mutated nets from best parent | −23.7% best | **DISPROVEN** |
| Heterogeneous fleet roles (153 candidates) | all below base, best −6.9% | **DISPROVEN** |
| Bet B fleet meta-controller (8 genes, pop 24) | tier-2 negative | **DISPROVEN** |
| Selection/breeding gate (`gs_*`, energy-dominant) | **+19.2%, W27/L13** | **SUPPORTED → DEPLOYED** |
| C6 deletions (drop evade+reserve, less blind wander) | +18.1% at 40 seeds; later +1.0% vs old base | **WEAKLY SUPPORTED** — re-validated at 160 seeds, all single-delta reverts worse → kept |
| Tree-weight dose curve | dead at 160 fresh seeds | **DISPROVEN** |
| Carrying-capacity reproduction scaling (cc) | −31%/−17%/−29% on 3 seeds | **DISPROVEN** — fewer agents ⇒ less coverage ⇒ less income |
| Biome-aware spawning/transit (bio) | mixed/worse on 3 seeds | **UNRESOLVED** (leak metric was buggy; not deployed) |
| Phase-speed rule from counterfactual boundaries (psp) | +0.3%, W20/L20 | **DISPROVEN** |
| Late-game flee boundary (evade if energy>345, late) | −0.5%, W21/L19 | **DISPROVEN** |
| Counterfactual oracle (Q2): 317→940 states | 24% disagreements, no consistent winner | **DISPROVEN** — single-action advantage does not compose to fleet level |
| Breadth search wide1 (601 cands) / wide2 (800 cands around C6) | nothing beats base (−2.9% … −7.1%) | **EXHAUSTED** |

**Causal ablation (harness-level, no sim edits; control 9,420 ticks):**
`lower_move_cost +33.5%` · `no_predators +8.4%` · `cheap_food −7.7%` · `perfect_info −17.8%` ·
`fewer_agents −40.1%` · `no_repro −82.5%` (`unlimited_sprint` invalid — botched intervention).

**Energy budget (mass balance, live controller):** production per 1k ticks 9,983 → 6,963 → 7,053 →
4,882 by phase while spawns stay 12.8-16.0/1k; standing stock +1,288 early then −435/−213/−194;
measured sustainable N 28.3/10.4/5.9/3.6 vs actual 10.2/8.0/7.2/6.3 (oversubscribed from ~6k ticks,
which is exactly the modal death window) — **but reducing N hurts** (see cc), so the lever is
**income per unit of travel**, not fleet size.

---
## 5. EXHAUSTED — DO NOT RE-TEST

One-knob parameter sweeps; threshold sweeps; genome-weight/mode sweeps; tree-weight; fleet
meta-controller genes; random net mutation; heterogeneous roles; per-agent thrift/reserve/repro
floors; evasion families; thinning the fleet; counterfactual single-action distillation.
The knob space around the deployed controller has been searched broadly (800 fresh candidates) and
by systematic deletion — nothing beat it.

---
## 6. CURRENT ML EXPERIMENT (running now)

**What it is:** *residual policy search by parallel resampling + paired selection* (NOT gradient RL,
NOT supervised). The policy = deployed heuristic **plus** a small MLP that adds bounded offsets.
- scripts: `experiments/gen_residual.py` (generation-0 population), `experiments/sched.py`
  (`--policy residual` branch), scored by the same paired harness used for everything else.
- architecture: 31 (obs) → 12 (tanh) → 2 (tanh) = **410 weights**; outputs = heading offset
  (±0.5 rad) and speed offset (±0.3·sprint). **Zero-init = exactly the incumbent controller.**
- population 73 (σ 0.02/0.05/0.15 on all weights) + 1 zero-net baseline.
- evaluation: screen 10 seeds × 18,000 ticks → top-4 → **40 unseen paired seeds**, A/A duplicate
  included; fitness = mean steps (survival ticks). Lane `res1`, 26 workers, on
  `root@212.147.236.122`; results in `/opt/nac_h2h/results` (note: lane launched from the `/opt/nac_gs`
  tree with that results dir).
- **falsifier:** if no candidate beats BASE at 40 paired seeds, this parameterisation (stateless
  net + heading/speed only) is dead.

---
## 7. WHAT THE MODEL RECEIVES / WHAT IS MISSING (question 9 + 10, from code)

Receives: the 31 floats above (+ the raw dict if we hand it over). **Missing and plausibly
blocking:**
1. **No temporal representation at all** — single frame, no tick/time, **no energy trend**, no
   travel-so-far, no history. The measured bottleneck is *travel economics* (an integral quantity),
   so a stateless net cannot even represent it. A recurrent net or explicit EMAs (energy, income,
   distance travelled, time above lockout) are the concrete missing capability.
2. **No spawn control** — the residual cannot touch reproduction, yet `no_repro` is −82.5%, i.e.
   reproduction is the dominant survival mechanism.
3. **Fitness = raw survival only** — no mechanism-aligned term (e.g. income per unit travel, lockout
   exposure), despite the ablation identifying movement cost as the constraint.
4. **Population far too small** for a 410-parameter space (73 candidates; ES/PPO would use 10³-10⁴).
5. **Noisy objective**: the sim has process-level nondeterminism (identical configs differ on 5-7/40
   seeds), so screens need the A/A duplicate and ≥40-seed confirmation to have power.

---
## 8. RECOMMENDED NEXT INVESTIGATION (and its smallest falsifier)

**Direction:** make the learned component *temporal and mechanism-aware* — feed the residual (or a
small recurrent policy) explicit time/trend features (sim tick, EMA of energy, EMA of income,
distance travelled per fruit, ticks in lockout) and include the **spawn decision** in its action
space, trained with a fitness that adds a mechanism term to survival.

**Smallest falsifying experiment (~40 min on the 64-core box):** keep everything else identical;
add the 5 temporal features to the residual input (OBS 31→36, +spawn output) and resample ~200
candidates. Falsifier: **no candidate beats BASE at 40 paired unseen seeds.** If that fails, the
next step is real gradient RL (PPO, 10³+ episodes, same harness) rather than more black-box sampling —
and if that also fails, the honest conclusion is that the heuristic's structure, not its parameters,
is the ceiling and the remaining play is repeated validation on the fattest tail.

---
## 9. INFRASTRUCTURE / OPERATING RULES

- **Never run experiments on the user's Mac** (incident 2026-09-19: a cron run launched 8 local
  workers). All compute on `212.147.236.122` (64 cores) or the serving box *with* the latency guard,
  and never on the serving box while a validation is running.
- Latency is score: `/predict` p95 must stay low (currently 1-2 ms; the public path ~5.6 ms/tick
  measured from an official attempt of 138 s).
- Paired harness: `experiments/sched.py` (+ content-addressed cache, staged promotion, A/A noise
  measurement, unknown-knob guard that refuses silently-inert candidates).
- Determinism: `PYTHONHASHSEED=0` is pinned and part of the cache key; residual nondeterminism
  remains (~5-7/40 seeds), so always include a duplicate baseline.
- Useful tools: `tools/ops/{deploy_arm.sh,ROLLBACK.md,latency_guard.sh}`, `experiments/{budget.py,
  bottleneck.py,replay.py,oracle_probe.py (verified snapshot/restore),survival_autopsy.py}`.
- Refuted/settled list is **closed** — do not re-litigate §4/§5; add new rows instead.
