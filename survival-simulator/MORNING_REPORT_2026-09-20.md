# MORNING REPORT — 2026-09-20 (autonomous overnight run)

Machine-readable companion: `experiments/MORNING_REPORT.json`. Written by the session that took over
after the previous Claude session was lost; its transcript was used as historical context only and every
claim below was re-measured from the repository and the live systems.

## 1. THE HEADLINE: the graded endpoint serves a ~42% worse controller than we already have

| | controller | mean ticks (80 fresh paired seeds) | score | p10 | min |
|---|---|---|---|---|---|
| **served now** | `best_controller.py` 252f0ba1 on survival.zaitzev.com | 8,110 | **811** | 3,881 | 1,201 |
| **available** | `hive.py` 829e4147 (survival-v2 branch) | **11,383** | **1,138** | 5,916 | 1,527 |

Paired: **+3,406 ± 438 ticks, t = 7.78, W/L = 65/15, +42.0%** on block 1 (80 seeds), reproduced on an
independent second block (**+2,835, W/L 30/9**, 39 seeds): **pooled +3,219 ± 388, t = 8.30, W/L 95/24,
+39.4% over 119 fresh paired seeds**. Same simulator, same seeds, same harness
(`experiments/mech_ab.py --policy {heuristic|hive}`). Latency of hive: mean 4.09 ms/tick, p99 11.72 ms
= 29 s of the ~600 s grader budget. Full write-up: `FINDING_HIVE_VS_SERVED_2026-09-20.md`.

So the premise "we were getting 1000-1200 consistently with v2" is TRUE of the survival-v2 controller —
and that controller is NOT what survival.zaitzev.com serves. The served controller's own
`predict_log.jsonl` gives its real attempt distribution over 28 complete attempts:
**mean 731, median 759, max 1,165** — consistent with the team's 59-attempt ledger (770 → 819) and with
this harness (811). The board-best 1,223.85 is that distribution's upper tail.

**Nothing was deployed** (the standing directive is "do not deploy anything automatically / do not
modify the production endpoint"). The container was never touched. This is the single largest available
score move and it needs your decision, not my autonomy.

## 2. Answers to the questions that started this session

- **A. Established:** score = 0.1×ticks + fruit/1000 (− predation) → maximise expected fleet survival;
  one validation = one run and the board keeps the best; the served controller is 252f0ba1 with the
  52-key GS params; the simulator is deterministic (fixed 3 address-dependent set iterations → A/A 0/40);
  deaths are ~100% energy exhaustion; movement cost is the binding constraint; reproduction is the
  survival mechanism.
- **B. Merely hypothesised (now mostly tested):** "movement is income-blind, so scale it with income" —
  see §3, this is DEAD in both directions. "The residual learner can find behaviour V2 cannot express" —
  still unproven, see §4–5.
- **C. Conclusive:** determinism A/A; the genome-gate family; the steering/FSM family; the causal
  ablation directions; the BC-clone failure; the zero-change contract; **and now hive-vs-served (§1)**.
- **D. Methodologically weak (in the inherited record):** the old ES screened 64 candidates on 6 seeds
  while the per-seed SD of a paired difference is ~2,595 ticks — a generation-level SE of ~1,060, so its
  centre random-walked (fraction of candidates beating BASE oscillated 3%→98% across generations; the
  held-out effect declined +423, +897, +255, −277 across gens 7/11/15/19). Its checkpoints were
  overwritten each generation, so past confirms are unreproducible. It also optimised a max-over-draws
  while the competition scores one draw. And the repo checkout still carries PRE-C6 params, so any lane
  launched from it measures the wrong baseline.
- **E. Smallest formulation that can discover new behaviour:** not a new learner — an action space and
  an estimator that can see the effect. The dead ES moved heading by ~0.03 rad (2–10% of its ±0.5 rad
  authority), i.e. it never left V2's neighbourhood. Fixes applied in `es_rec.py`: antithetic pairs,
  24 seeds/generation, BASE arm every generation, held-out centre evaluation, weight-hash ledger keys,
  per-generation checkpoint, and σ large enough to change behaviour.
- **F. ES vs PPO:** both, with the fixed estimators — ES on the 64-core box (`es_rec.py`), PPO on the
  30-core box (`ppo_rec.py`), which is what is running now.
- **G. Smallest falsifier:** the mechanism tests in §3 (160 fresh paired seeds, pre-registered
  thresholds), and for any learned candidate: a paired verdict vs frozen V2 on 160 unseen seeds.

## 3. Structural mechanism family: CLOSED, by two pre-registered tests (n=320 fresh paired seeds)

| test | arm | paired vs served V2 | verdict |
|---|---|---|---|
| energy-conditioned search (`esearch=1`, rich 0.12→0.60, broke 0.4→0.12) | 160 seeds | **−1,637 ± 258, t=−6.36, W/L 52/108** | dead |
| same, milder dose (`esearch_hi=0.30`) | 160 seeds | **−1,649 ± 265, t=−6.23** | dead, dose-insensitive |
| search effort UP (`blind_explore_frac` 0.12→0.40) | 160 seeds | −225 ± 241, t=−0.94 | null |
| search effort UP (`blind_explore_frac` 0.12→0.60) | 160 seeds | **−521 ± 247, t=−2.11** | worse |

Receipts prove the mechanism fired (blind-tick mean commanded distance 3.99 → 3.02 → 5.33 → 6.10).
**Conclusion: the incumbent's constant blind-search speed is at (or very near) the optimum in BOTH
directions.** The audit's "movement is income-blind, therefore scale it down" reading had the wrong
sign: conserving travel while broke starves the fleet of coverage. Per the pre-registration, the family
is closed — no dose expansion, no parameter rescue.
Pre-registrations: `experiments/PREREG_ESEARCH.md`, `experiments/PREREG_SEARCH_EFFORT.md`.

## 4. ES (64-core box) — running, rebuilt, not yet conclusive

`experiments/es_rec.py`, gens of 16 antithetic pairs + centre + BASE, 24 rotating seeds @6,000 ticks,
held-out centre evaluation every 3 generations, weight-hash ledger, per-generation checkpoint.
Gen-0 cost ≈ 816 episodes. Status at hand-off: gen 0 in progress. Nothing yet claims a gain; the
*estimator* is the change (≥3× the old SNR per generation).

## 5. PPO (30-core box) — running, genuine, aligned

`experiments/ppo_rec.py`: recurrent residual PPO. Zero-init actor head → **policy == V2 exactly at
init** (verified in-run: "actor output is exactly 0 at init = True"). GRU(61→48) over the legal
`build_v2` observation vector (no fruit age, no hidden coordinates, no simulator internals; the critic
is the same trunk on the same legal tensor — no privileged state anywhere). Action space: heading
±0.5 rad and speed ±0.3·sprint; reproduction left exactly as deployed (every alternative measured
−4.3% to −11.5% at 160 paired seeds). Reward is **r = +0.1 per sim tick, γ=1, episode ends at
extinction**, so Σr = 0.1·T identically = the dominant term of the competition score, with no shaping.
Parallel environments (26 independent simulator processes, one torch thread each), checkpoint every
update, deterministic evaluation vs the frozen incumbent every 20 updates on held-out seeds.
Throughput ≈ 26 episodes/update at ~33 s/update ≈ 110 updates/hour ≈ 15 M agent-transitions/hour.
Latency guard active on the graded box: kills the lane if /predict exceeds 50 ms (measured 1–4 ms).
Status at hand-off: updates 0–8 done, corr usage ~4% of the bound, no eval deviation yet.

## 6. Failed hypotheses (so they are not re-run)

- Energy-conditioned / energy-scaled search effort (both directions) — §3.
- The inherited residual-ES regime (6-seed screens, σ=0.02, unversioned checkpoints) — §2D.
- The `gs_w_energy` capacity-ratchet hypothesis (wrong sign: 3.0 is causal as a benefit).
- Steering/FSM/target-commitment (83.67% of agent-ticks see zero fruit; flee and wander already latch).
- BC-cloning the incumbent as a replacement (−16.8% paired).

## 7. Exact next experiments

1. **Deploy hive to the graded endpoint** (your call; ~+360 expected score) after one more independent
   confirmation block, already running: `cmp2_*` on 40 fresh seeds (300640–300679).
2. If hive lands on the endpoint, re-run the residual work against **hive** as the base policy rather
   than the heuristic — the ML machinery here is base-agnostic.
3. ES: keep `es_rec.py` running; the decision rule is a held-out paired gain ≥ +400 ticks (its SE at 24
   seeds is ~530, so a real signal needs two consecutive generations to agree). No gain by ~10
   generations ⇒ stop ES compute and give it to PPO.
4. PPO: continue to ~300–500 updates, then take the best checkpoint to a 160-seed paired verdict vs the
   frozen incumbent. Training reward is never evidence; only the paired fresh-seed number counts.


## 8. LIVE STATUS (updated through the night; 03:15 UTC / 05:15 CEST)

| lane | machine | state | evidence so far |
|---|---|---|---|
| hive-residual ES (`es_rec.py --base hive`) | 64-core | **running** (52 workers) | gen 0: centre == hive exactly (holdout 13,700 = 13,700, so the zero-change contract holds on the hive base too); population mean **+331** with 8/12 pairs up (SE ~200) - unlike the heuristic base's -241 |
| residual PPO (`ppo_rec.py`) | 30-core serving box, 26 workers, latency guard | **running**, update 58 | 12M+ transitions; actor head 0.141 -> 0.224; residual advantage SD 0.169 -> 0.278; evals **+219 (W/L 5/2)** then **+10 (W/L 3/4)** at n=12 - i.e. nothing reliable yet. Two diagnosed fixes were applied mid-run: the L2 anchor was 100x too strong (it contributed ~0.002-0.02 to a loss whose policy term was ~0.002, pinning corrections to zero) and the 1,500-tick horizon contained almost no deaths (extinct 0-8%), so survival was invisible to the learner. Now training at a 6,000-tick horizon where ~30% of episodes end in extinction, which is the signal the objective needs |
| heuristic-residual ES (`es_rec.py --base heuristic`) | 64-core | **STOPPED after 2 generations** | gen 0 pop mean -241 (4/16 pairs up), gen 1 +44 (9/16), centre -81 +- 76 (n=24): no signal, and it is the wrong base. Compute moved to the hive base. Ledger kept at `/opt/nac_h2h/es2/` |
| structural mechanism tests | 64-core | complete | §3: family closed, n=320 fresh paired episodes |

The controller the graded endpoint serves was never touched; `sha256(/app/best_controller.py)` stayed
252f0ba1 throughout, and the container was never restarted (0 restarts).


## 9. FINAL VERDICTS (end of the ML window)

### 9.1 ES over the residual GRU — did not produce a gain on either base

| run | base | result |
|---|---|---|
| `es_rec.py --base heuristic` (16 pairs, 24 seeds/gen, h6000) | served controller | gen 0 pop mean **-241** (4/16 pairs up), gen 1 **+44** (9/16), centre **-81 +- 76** (n=24). No signal in 2 generations; stopped and re-pointed. |
| `es_rec.py --base hive` (12 pairs, 16 seeds/gen, h4500) | hive | gen 0 pop mean **+331 +- 202** (8/12 pairs up, t=1.6) - the only positive reading of the night; gen 1: the centre after ONE ES step measured **-1640 +- 185 (t=-8.9)** against hive. |

The estimator rebuild (antithetic pairs, 24 seeds/generation, BASE arm every generation, weight-hash
ledger, per-generation checkpoints, held-out centre evaluation) did its job: it produced a *measurable*
answer instead of an unreadable one. The answer is that the ES step is the problem now, not the
estimator: sigma=0.15 on the head is small enough to move nothing usable and large enough to destroy a
tight local optimum in one averaged step. The diagnosed fix (written into the script) is a TRUST REGION -
evaluate the post-step centre on a few seeds and roll back unless it improved, shrinking sigma on
rejection. Candidate weights are now archived per generation as well, because gen 0's +331 could not be
re-tested afterwards (only the centroid was saved) - a search's candidates are its deliverable.

### 9.2 Recurrent residual PPO — did not beat the incumbent

Served base, 86 updates, ~30 M agent-transitions, 26 parallel simulator environments, checkpoint per
update, deterministic periodic evaluation. Two independent paired verdicts on fresh seeds at the FULL
18,000-tick horizon:

| checkpoint | seeds | paired | t | W/L | p10 | min |
|---|---|---|---|---|---|---|
| update 80 | 310000-310079 (80) | **-394 +- 390** (-4.7%) | -1.01 | 33/47 | - | - |
| update 85 (final) | 310080-310159 (80) | **+114 +- 333** (+1.5%) | 0.34 | 40/40 | 3,959 -> 4,602 | 1,988 -> 875 |

A horizon diagnostic on the same 80 seeds at 6,000 ticks gave -290 +- 151 (t=-1.92) with only 25 of 80
seeds discriminating (the rest tie at the cap).

**Verdict: no evidence that recurrent PPO improved on the incumbent.** The final point estimate is
indistinguishable from zero (40/40 wins/losses, +1.5% +- 4.3%) and its worst case is 56% worse (875 vs
1,988 ticks). Its in-run tripwire (+175, +331 on 12 fixed seeds at 6,000 ticks) was a small-sample
artifact: the same policy measures -290 to -394 on 80 fresh seeds at the full horizon. This is the
project's own rule restated by measurement - only a fresh-seed, full-horizon, paired verdict counts.

Two real defects were found and fixed mid-run (both would have silently produced a null):
1. **The L2 anchor was ~100x too strong for this loss scale** (it contributed 0.002-0.02 against a policy
   term of ~0.002, pinning the corrections at zero). Set to 0.0002.
2. **The 1,500-tick training horizon contained almost no deaths** (extinct 0-8%), so survival - the thing
   being optimised - was invisible to the learner. Moved to 6,000 ticks, where ~30% of episodes end in
   extinction. (The hive-base continuity lane at 2,500 ticks sees 42% extinctions.)

### 9.3 Running when this window closed

`ppo_hive.py` on the 64-core box: the same recurrent residual PPO sitting on **hive** rather than the
served heuristic (base hive 829e4147), zero-change contract verified (actor output exactly 0 at init),
50 -> 24 workers, checkpointing every update, deterministic paired evaluation every 20 updates. This is
the lane worth continuing: hive is +37.6% over what is deployed, so a percentage point found there is
worth more than the same point found on the incumbent, and hive is where the score will come from if the
deployment is approved.

### 9.4 The honest summary of the ML question

"Can recurrent RL discover a policy that beats V2?" - **Not in this budget, and not on this base.** Two
independent searches (antithetic ES, recurrent residual PPO) with ~30 M simulator transitions, a
rebuilt estimator and two diagnosed-and-fixed defects produced: one statistically insignificant positive
reading (ES on hive, t=1.6), one point estimate indistinguishable from zero (PPO final, t=0.34), and one
negative (PPO update-80 and the 6,000-tick diagnostic, t=-1.0 to -1.9). Meanwhile a hand-written
controller that already existed, and was never deployed, is +37.6% (t=8.84) over the one that is.

The highest-value action available is therefore not more ML on the incumbent: it is deploying hive
(`DEPLOY_PLAN_HIVE_2026-09-20.md`), and then re-basing the residual ML on hive, since every script here
(`es_rec.py --base hive`, `ppo_hive.py`, `eval_ckpt.py --base hive`) already takes the base as a switch.

## 10. END-STATE VERIFICATION (priority 1: V2 preserved)

Checked on the graded box at the end of the window, not assumed:
`docker ps` = `nac-survival-vps | Up 20 hours`, `restarts=0`; `sha256(/app/best_controller.py)` =
**252f0ba1e060…** (unchanged, the same hash as at the start); live params still the 52-key GS set
(`genome_select 1.0`, `gs_w_energy 3.0`, `evade_mode 0.0`); public `https://survival.zaitzev.com/`
**200 in 19 ms**; the predict log is idle (no attempt in flight); zero ML processes left on the serving
box. Nothing was deployed, nothing was restarted, and the serving container was never written to.
