
## 2026-09-19 ~05:30 UTC — FIRST QUALIFIED DEPLOY (C6)
- 40-seed held-out gate (seeds 2580-2639, never screened on): **C6_blind_noreserve_noevade +18.1%**
  (W27/L13 = 67.5% wins, floor p10 +1,977, variance 3.35M vs BASE 4.69M); **C3_no_reserve +13.7%**
  (W26/L14, floor +1,322). Both cleared the three deploy rules.
- Note the lesson: C6 scored +1.0% on the 20-seed screen and +18.1% at 40 seeds. The screen is very
  noisy in BOTH directions — the holdout rule earned its keep for the second time in one night.
- DEPLOYED C6 to the serving box: params-only change (controller hash f90cb4e3 unchanged), verified
  from inside the running container, /predict 1.2-1.9 ms. Rollback file: experiments/ROLLBACK_params_f90cb4e3.json
  and /root/params_backup_f90cb4e3.json on the serving box. Standby updated to match and left stopped.
- C6 = deletions only: evade_mode 0 (closed E3 line), reserve_frac 0 (refuted), blind_explore_frac 0.12
  and wander_weight 0.03 (less blind wandering).
- NEXT: user runs validations; then Q2 (scale the counterfactual oracle) and Q4 (bet B tier-2).

## 2026-09-19 04:20 UTC — Q4 closed negative, C6's edge does NOT replicate, oracle sweep blocked
(status sweep 04:01-04:20 UTC; all times UTC)

**Serving box:** container `nac-survival-vps` up since 03:21, in-container
`sha256sum /app/best_controller.py` = `f90cb4e3…` = `best_controller.sha256`; `/app/best_controller/params.json`
byte-for-byte equal to `DEPLOY_C6_params.json` (disperse 3.0, evade 0, reserve 0, blind 0.12, wander 0.03);
`https://survival.zaitzev.com/` 200 in 17 ms; no validation in flight (predict log idle — the 04:03-04:05
POST triplets were my own `latency_guard` probes).

**HARD RULE 2 VIOLATION, found and cleared.** A previous session had left two `oracle_sweep.py` lanes
(`ors1`, `ors2`) running **on the serving box**, guarded by `latency_guard.sh 30` (observed p95 2.6-3.9 ms
vs a 30 ms threshold). Both had finished on their own (32 + 36 states); I harvested their JSONs, killed
both guards and the tmux server, and re-verified 200/17 ms. No experiment processes remain on the serving
box, and oracle work stays on 212.147.236.122 from now on. Latency was never actually harmed, but the
rule is absolute: if `latency_guard` had fired mid-validation we would have lost a graded run.

**Q4 — bet B tier-2: NEGATIVE, closed.** `run_betB.log` tier-2 paired 3 evolved genomes + the live
controller on 20 held-out seeds @18000: meta#1 **+214.9 ticks (+2.9%, 10/20 seeds won)**, meta#0 −55.4
(9/20), meta#2 −476.8 (10/20), meta#3 −5.2 (12/20). Every arm reported **`armed 0`** ticks — the endgame
mechanism never fired, so per "mechanism before score" these arms never moved what they claim to move.
No arm reaches the ≥55% win rule → not deployable. No further bet-B compute unless a genome arms.

**Q1 follow-up — c6minus, 40-seed held-out (seeds 2700-2759, 18000 ticks, 16 workers, 23.7 min):**
`M_no_disperse` (disperse_weight 0.0) **+655 (+8.7%)**, W24/L16 = 60%, floor p10 **−2**, var 4.98M;
`M_no_flee` +4.8% (W22/L18), floor −40; `M_no_tree` +2.7% (W22/L18), floor −307; `M_no_risk` −0.7%;
`M_no_concern` +2.4%. **And `BASE_C6` (the deployed controller) only +1.0% vs BASE, W19/L21** — the
+18.1% that justified the deploy does NOT reproduce on a fresh seed set. Two consequences: (a) nothing
here meets all three deploy rules (best candidate's p10 gain is −2, i.e. a flat floor, and its variance
is *higher* than BASE) → **NO DEPLOY**; (b) C6's true edge over the old controller must be treated as
≈0 ± noise, so a board score indistinguishable from the previous one is the expected outcome, not a bug.

**Q2 — oracle scaled to 317 states, then BLOCKED (and this is the useful result).** Aggregated all 11
dumps (`oracle_x1-x6`, `oracle_1-3`, `oracle_serv1-2`) with the new `experiments/oracle_aggregate.py`:
**77/317 disagreements (24%)** — but **59/77 (77%) of disagreeing states have ≥3 alternatives winning at
once**, `still` wins 60 times and `toward_predator` 53 vs `flee_predator` 67. Mechanism, quantified: in
the 45 disagreeing states where the controller's focal agent ends at **energy 0**, 164/270 alternative
branches end above 0, while in agreeing states the best alternative also ends at 0 (median 0.0). So
"winning" is a knockout/threshold event (lockout-or-death vs any survival), reproduced by no-ops and
suicide behaviours alike — **there is no consistent winning behaviour, so Q3's precondition fails and
Q3 was NOT launched.** Also: 80% of sampled states end with the focal agent dead or at zero energy
(branch `score_focal` reports alive=False whenever the fleet dies, and `roll()` stops on fleet
extinction), so this sample is dominated by extreme states.
New tooling committed: `oracle_aggregate.py` (winner-clustering + sample size, not anecdotes) and
`oracle_noisecheck.py` (**A/A**: same snapshot, same arm, 3 reps). Its first 15 states: the controller
branch is **bit-identical across all 3 reps** (`distinct=1/3`), so the 24% is *not* within-process
branch noise — it is the threshold effect above. Next diagnostic if Q2 is resumed: re-sample states
where the focal agent is healthy at snapshot time, score on a continuous metric (energy at fixed tick,
not the alive/energy knockout), and count fleet extinction per arm.

**Q5 — wide1 could not produce a verdict:** its stage 2/3 rows are empty in `wide1_ledger.json`
(stages (1,601),(2,0),(3,0)) — the promoted candidate was never actually scored on 40 seeds, so the
lane claims nothing. `wide2` (801 candidates, seeds 2760-2819) is still in stage 1 (800/2403).

**LAUNCHED (exp box): `c6conf`** — 6 arms (deployed `BASE_C6`, `BASE_C6_dup` = A/A duplicate,
`M_no_disperse`, `M_no_flee`, `M_no_tree` + the tool's injected baseline) x 40 fresh unseen seeds
2880-2919 @18000 ticks, 24 workers: `sched.py --lane c6conf --candidates par_c6conf.json --seeds 2880-2959
--stages 80:18000 --workers 24`. The duplicate genuinely re-runs (the cache key includes the arm id, so
`BASE_C6_dup` starts at 0 cache hits) and gives this run's noise floor. CPU: load 39-48 on 64 cores
(c6conf 24 + wide2 26 + `noise` 1).

**Note for the human:** run validations whenever convenient — with the params-only deploy live, expect
the board to sit within noise of the previous attempt.
