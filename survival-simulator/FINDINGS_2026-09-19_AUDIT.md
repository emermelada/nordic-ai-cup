# Independent audit + mid-game diagnosis — 2026-09-19

Written by an incoming session that took over the repo, audited PROJECT_STATE.md against the code
and the live systems, and then ran its own experiments. **Several claims in PROJECT_STATE.md and
HANDOFF.md are wrong; they are corrected below and each correction carries its evidence.**

Everything here is reproducible from the scripts named in each section. Nothing was deployed.

---

## 1. CORRECTIONS TO THE EXISTING RECORD

### 1.1 A validation is ONE run, not the mean of three  — PROJECT_STATE §3 and HANDOFF are wrong

PROJECT_STATE.md §3 says *"each = mean of 3 preset-seed runs"*. It is not. Reconstructed from the
serving container's own `predict_log.jsonl` around the 1,223.85 attempt (2026-09-19 10:49:59 →
10:55:21 UTC): a **single** sim track rises monotonically to `sim=1200.0` with `score=1223.75`, and
the reported validation score is that run's final value. There are no three rising tracks. Duration
tracks ticks at a single-run rate (score 369 → 133 s; 1224 → 322 s).

Consequence: the official metric has **single-run variance**, and every `sd/√3` bootstrap in the
project's documents understates risk by √3.

### 1.2 The official mean has not measurably improved in three days

From the team's own validation ledger (59 attempts):

| era | n | mean | SD |
|---|---|---|---|
| pre-C6 (09-17 20:30 → 09-19 08:29) | 30 | **770.1** | ~190 |
| post-C6+selection (09-19 08:40 → 11:02) | 18 | **819.1** | 247 |

+6.4%, **t ≈ 0.7, not significant.** The board best of 1,223.85 is a *maximum over 59 draws*; the
expected max of 59 draws from N(800, 247) is ≈ 1,370, so the recorded best is **below** what pure
noise predicts. The 1,187.36 recorded on 09-17 21:29 (attempt #4, before any of the "improvements")
is the same statistic.

`n_evaluations: 0` — the competition result is **one final evaluation run** (README:178). The
expectation of that single draw is the **mean** (~800), not the board best. The project has been
optimising a max-over-draws while the competition scores a mean.

### 1.3 The repo is not the source of truth for production

`best_controller/params.json` in this repo is the **pre-C6** set (`evade_mode 1.0`,
`reserve_frac 0.15`, no `gs_*`). The container runs the **GS** set (`evade_mode 0.0`,
`genome_select 1.0`). `experiments/sched.py` sets `DEPLOYED = ROOT/best_controller/params.json`,
so **any lane launched from a fresh checkout of this repo evaluates against the wrong baseline.**
The deployed artifact exists as `experiments/served_controller_252f0ba1.py` +
`experiments/DEPLOY_GS_params.json` (and `experiments/intel_cc/served_params_252f0ba1.json`).
Verified live: `docker exec nac-survival-vps sha256sum /app/best_controller.py` =
`252f0ba1…`. The host tree `/opt/nac` is the OLD C6 (`f90cb4e3…`), so a container rebuild from the
host tree ships the wrong controller — and `best_controller.sha256` (which holds the *container*
hash) would fail the Dockerfile provenance gate, which is protective. **Do not "fix" that hash.**

### 1.4 Predation is ≈0.1% of deaths, not 35–43%

Two independent detectors agree on 60 and on 160 seeds: exact predator-distance at the kill
(environment.py:720-727 kills at predator.size + agent.size = 15 units) and the predator
energy-jump (`predator.energy += agent.energy`). Pooled over 160 seeds × 7 arms:

```
starved 65.4%   aged 34.5%   eaten 0.1%
```

This reconciles the old `no_predators +8.4%` ablation: **predators kill indirectly**, by forcing
sprint-fleeing that bankrupts the fleet — they are not a direct mortality source.

### 1.5 "The trait choice is not the mechanism" does not replicate

AUTOPILOT.md reasons from a pooled 3-block analysis that `vis_ctrl` (+71.8) beat `en_top3` (+46.5)
and therefore *"the trait choice is not the mechanism"*, and deploys `en_top3`. On one clean
160-seed paired block (§3 below) **`vis_ctrl` is −4.3% against the deployed arm.** The pooled
number was confounded across blocks and times.

### 1.6 Residual episode-to-episode state leakage is real

`BASE` vs `BASE_DUP` (byte-identical params) are bit-identical on 2 seeds but differ on **15 of
160**. `reset_memory()` does not fully clear process-local state, which is the source of the
project's ±3–4.5% "A/A noise floor" and of several reversed arms.

---

## 2. WHY THE FLEET DIES — THE ENERGY BUDGET

`experiments/midgame_diag.py` (instrumentation only — it calls the real policy and *recomputes*
each spawn gate from the controller's own module state; it never edits the controller).
60 seeds, deployed controller, seeds 5200-5259.

**Fleet energy budget, per 1,000 ticks** (corrected: the sim clamps to walking speed below
`max_energy/5` *before* charging, so the commanded distance overstates a locked agent's cost —
lockout is 47–87% of agent-ticks):

| ticks | pop | e_abs | ef | lock% | income | move | metab | birth loss | NET |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 11.2 | 259.0 | 0.499 | 26.8 | **9,243** | 2,345 | 1,116 | 282 | **+5,500** |
| 3,000 | 9.2 | 224.7 | 0.389 | 38.9 | 5,415 | 2,985 | 916 | 313 | +1,201 |
| 4,000 | 8.2 | 226.6 | 0.374 | 40.8 | 4,482 | 3,012 | 824 | 323 | **+322** |
| 5,000 | 7.7 | 214.2 | 0.338 | 46.0 | 3,043 | 3,038 | 772 | 309 | **−1,076** |
| 6,000 | 6.1 | 194.6 | 0.287 | 53.7 | 1,412 | 2,510 | 609 | 292 | −1,999 |
| 7,000 | 5.9 | 171.3 | 0.245 | 59.3 | **557** | 2,609 | 585 | 282 | **−2,918** |
| 10,000 | 4.8 | 141.4 | 0.203 | 67.4 | 748 | 2,253 | 479 | 248 | −2,232 |

**The finding: income collapses 94% from its peak while movement expenditure is inelastic — it
stays at ~2,500–3,000 energy/1,000 ticks and never scales down.** By tick 7,000 movement is 4.7×
income. The budget crosses zero at **~4,500–5,000 ticks**, roughly 1,500 ticks *before* the
`ef > rf` reproduction gate closes (≈6,300) and before the population turns.

Death classification is ~100% energy exhaustion (65.4% starved, 34.5% aged; "aged" = died past
`max_age`, which adds `0.01*age` drain — still starvation).

**Caveats, stated plainly.** (a) `e_abs`/`ef` are means over *survivors*, so they carry
survivorship bias. (b) The budget omits the energy an agent is holding when it dies, which is
simply discarded; deaths relieve the deficit, so NET is the *available* budget, not the realised
bank change. (c) A movement decomposition into (blind / forage / flee) needs the joint
(lockout × mode) distribution, which is not yet recorded — that is the next measurement.

---

## 3. THE GENOME-GATE EXPERIMENT — MY OWN HYPOTHESIS FALSIFIED

`experiments/midgame_diag.py --arms` + `experiments/arms_report.py`. 7 arms × 160 seeds
(5000-5159), identical seed block, all paired per seed against BASE.

Hypothesis under test: the deployed `gs_w_energy = 3.0` ratchets heritable `max_energy` from the
500 default toward the 1,000 ceiling (measured: mean 502 → 800), and because every energy gate is a
**fraction** of `max_energy` (`ef > rf`, `low_energy_frac`, the `max_energy/5` sprint lockout) while
absolute energy *falls* (259 → 147), the ratchet closes the gates and starves the fleet.

| arm | `gs_w_energy` | mean ticks | paired | % | W/L/T | win% |
|---|---|---|---|---|---|---|
| **BASE (deployed)** | **3.0** | **8115** | — | — | — | — |
| BASE_DUP (A/A) | 3.0 | 8090 | −25 | −0.3% | 8/7/145 | 53.3% |
| vis_ctrl | 0.2 | 7769 | −346 | −4.3% | 67/81/12 | 45.3% |
| w_en0 | 0.0 | 7707 | −407 | −5.0% | 64/90/6 | 41.6% |
| w_all0 | all 0 | 7662 | −453 | −5.6% | 71/86/3 | 45.2% |
| topk2 (stronger throttle) | 3.0 | 7608 | −507 | −6.2% | 73/86/1 | 45.9% |
| w_enNeg (ratchet *down*) | **−3.0** | **7181** | **−934** | **−11.5%** | 55/100/5 | 35.5% |

**Verdict: the hypothesis is NOT SUPPORTED.** Every deviation is worse, monotonically, and pushing
capacity down costs 11.5% at 100 losses to 55 wins. `gs_w_energy = 3.0` is causal — **as a
benefit.** The correct rationalisation is that `max_energy` is the **bank**, and the bank is what
carries a lineage through the famine; selection for capacity buys storage. The ratchet hypothesis
was wrong on the sign.

Per the pre-registered rule, this closes the genome-weight family. **Do not sweep the gate.**

---

## 4. THE STEERING / TARGET-COMMITMENT HYPOTHESIS — KILLED

`experiments/steer_diag.py`, 40 seeds, deployed controller, 2,667,760 agent-ticks.

Hypothesis: the controller is too reactive and persistent high-level state + target commitment
would beat per-tick reactive steering.

| visible fruits | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| share of agent-ticks | **83.67%** | 8.49% | 3.21% | 1.68% | 1.02% | 0.63% | 1.30% |

- Only **7.84%** of agent-ticks admit ≥2 targets — thrashing requires two.
- Near-tie re-selection opportunities (2nd-nearest within 15% of nearest): 7–17% of *those* ticks →
  **≈1.1% of all agent-ticks**.
- `away_frac` (heading >90° from a *visible* fruit): 0.1% early → 10.8% late (flee-preemption).
- Heading reversals >90°: 0.6% early → 9.7% late.
- The controller is **already** not reactive where it matters: `m["flee"]` is a latched state with
  hysteresis (engage 177.4 → disengage 252.3) that preempts everything; `m["evade"]` is a second
  latch; `m["wander_ang"]` is a heading held for 80 ticks.

**Verdict: no meaningful steering inefficiency. Do not build the FSM, and no A* is applicable —
the sim exposes no global map, and water/shelter are not objectives.**

*Metric caveat for anyone reusing this:* the `straightness` column in `steer.jsonl` is **not
interpretable** — it sums displacement across ~10 agents within a bucket and independent headings
cancel. Only `flip_rate`, `away_frac` and the fruit histogram are per-agent-valid.

---

## 5. RESEARCH LEDGER

**ESTABLISHED**
- score = `0.1 × extinction_tick` + fruit_energy/1000 − Σ(victim_energy)/100, with the last two a
  ±5% modulation; verified in the graded payload log *and* 16 recorded episodes.
- A validation = **one run**; the board keeps the best attempt; the final result is one evaluation.
- Local harness is calibrated: local mean 7,941 ticks vs official mean 819 board pts.
- Predation ≈ 0.1% of deaths. ~100% of deaths are energy exhaustion.
- 83.67% of agent-ticks see zero fruit; the fleet's income is access-limited.
- The deployed controller is `252f0ba1`, 52 params, `genome_select 1.0`, `gs_w_energy 3.0`,
  `gs_topk 3.0`.
- `gs_w_energy` direction is causal and the deployed value is the best of 5 alternatives tested.
- Movement expenditure is inelastic: flat at ~2,500–3,000 energy/1k while income falls 94%.

**SUPPORTED BUT UNCERTAIN**
- The budget crosses zero at ~4,500–5,000 ticks, ~1,500 ticks before the reproduction gate closes.
  (Instrumentation is sound; limited by survivorship bias and the missing death-energy term.)
- `max_energy` ratchets 502 → 800 over a run; vision 200 → 296.
- The official single-run mean moved 770 → 819 across the project (+6.4%, n.s.).

**HYPOTHESIS (untested)**
- Movement expenditure being *income-blind* is the structural ceiling. Every movement rule
  (`blind_explore_frac`, `forage_speed`, `flee_speed_frac`) is a constant that does not depend on
  the energy return.
- Predators are a **cost** channel, not a mortality channel — consistent with `no_predators +8.4%`
  and 0.1% direct kills.
- `flee_speed_frac = 1.0` may be dominated: predators pay 2.55/tick to sprint and hold 200 energy,
  so a pursuer exhausts in ~78 ticks. Walking (0.5/tick) instead of sprinting (5.5/tick) is 11×
  cheaper and may still escape. (Note: the controller comment "predators never tire" is **wrong** —
  `non_agent_step` routes predators through `update_entity_position`, which charges them.)

**DISPROVEN (this session)**
- The `gs_w_energy = 3.0` capacity ratchet harms survival.
- Persistent target commitment / hierarchical FSM would beat reactive steering.

**DISPROVEN (inherited, confirmed unreplicated)**
- The pooled "vis_ctrl > en_top3" ordering does not replicate on a clean paired block.

**UNTESTED / OPEN**
- Movement-mode decomposition (blind / forage / flee × lockout) — the next measurement.
- Whether a time-gated or income-scaled movement rule beats a constant one.
- `n_agents` and `sim_time` are **sent by the grader and ignored** by the controller, which
  reconstructs both (`_GC` TTL estimate, `_SIM_TICK`).

---

## 6. SCRIPTS ADDED

| file | what it does |
|---|---|
| `experiments/midgame_diag.py` | per-(seed,100-tick) instrumentation; recomputes every spawn gate for attribution; `--arms` runs paired arms |
| `experiments/midgame_report.py` | per-window aggregate of a single-arm diagnostic run |
| `experiments/arms_report.py` | paired per-seed verdicts + mechanism table for a multi-arm run |
| `experiments/steer_diag.py` | target-switch / reversal / abandonment / visible-fruit instrumentation |

Run from `/opt/nac_h2h` with `/opt/nacv/bin/python` (64 cores; the repo's own params.json is the
wrong baseline — see §1.3).