# MORNING REPORT — 2026-09-20 (autonomous overnight run)

Machine-readable companion: `experiments/MORNING_REPORT.json`. Written by the session that took over
after the previous Claude session was lost; its transcript was used as historical context only and every
claim below was re-measured from the repository and the live systems.

## 1. THE HEADLINE: the graded endpoint serves a ~42% worse controller than we already have

| | controller | mean ticks (80 fresh paired seeds) | score | p10 | min |
|---|---|---|---|---|---|
| **served now** | `best_controller.py` 252f0ba1 on survival.zaitzev.com | 8,110 | **811** | 3,881 | 1,201 |
| **available** | `hive.py` 829e4147 (survival-v2 branch) | **11,517** | **1,152** | 7,139 | 1,677 |

Paired: **+3,406 ± 438 ticks, t = 7.78, W/L = 65/15, +42.0%**, same simulator, same seeds, same harness
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
