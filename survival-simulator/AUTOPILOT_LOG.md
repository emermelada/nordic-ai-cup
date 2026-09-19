
## 2026-09-19 08:45 UTC — `gs_energy` (energy-capacity breeder selection) REFUTED at 3 independent 40-seed blocks: pooled +46 pts, 54% wins, p10 −285 → NO DEPLOY
(status sweep 08:03-08:45 UTC; exp box only; three lanes on seeds 3420-3459, 3460-3499 + a read of a parallel agent's 2960-3039 lane)

**Serving box (untouched):** container up since 03:21, in-container `best_controller.py` = `f90cb4e3…` =
`best_controller.sha256`, `https://survival.zaitzev.com/` **200 in 14 ms**, no tmux lanes, load 0.10.
Exp box was **completely idle** on arrival (no tmux server, load 0.24) — every lane from the 07:20 cycle
had already been harvested.

**The lever under test.** `experiments/mechanism_screen.py` (in the parallel agent's `/opt/nac_gs` tree,
commit `3f36641`) screened `genome_select`-based arms on the 86-param controller: `en_top3/4`,
`en_top3_late`, `en_top3_rescue3` (breeder ranking weighted to `gs_w_energy` 3.0 vs 0.4 default) and
`vis_ctrl` (the same selection weighted to VISION — the discriminating CONTROL, reproducing V2's
weighting). Hypothesis: sprint lockout is `energy < max_energy/5`, so a lineage bred for a bigger energy
bank is locked out less often (86% of predation deaths are inside lockout). Its 4-seed mechanism screens
read **1.3-1.5x the baseline's mean ticks** (`en_top3` 8,392 vs base 5,924) — the largest raw signal seen
in this project, and the reason it was worth a real test.

**First, the harness was verified before anything else (this is what the whole conclusion rests on).**
`/opt/nac_gs/best_controller/params.json` md5 == `/opt/nac/best_controller/params.json` (77f1b16e…), and
the 86-param controller with `genome_select` 0.0 (default) is **byte-identical in outcome to the deployed
controller on seeds 3420-3423** (ticks/scores 6992/718.58, 7365/735.20, 7918/783.39, 8063/803.62 — all
four match exactly). So "base" in these lanes IS the deployed controller. Every arm's overrides were also
confirmed to differ from BASE on 8-9 keys (`sched.py` warns on unknown knobs).

**Lane 1 (`gsconf`, seeds 3420-3459, 40 paired, 18000 ticks, 20 workers, /opt/nac_gsc — an isolated copy
with an empty cache so nothing was recycled):** `en_top4` **−416 (−5.1%), W17/L23, floor −1303**;
`en_top3` **−517 (−6.4%), W15/L25**; `en_top3_rescue3` **−535 (−6.6%), W14/L26**. A/A duplicate
`GS_BASE_dup` +37 (+0.5%), **W2/L2/T36 of 40** → only 4 of 40 seeds disagree, per-seed sd ≈50 pts.
*(First attempt at this lane was killed 08:10 by a tmux `kill-server` from the parallel agent's cleanup —
25 episodes survived in the cache and were reused. Relaunched on a private tmux socket `-L nacgsc` so no
cleanup can kill it again.)*

**Lane 2 (`gsconf2`, seeds 3460-3499 = third fresh block, 24 workers, 7.8 min):** `en_top3` **+362
(+4.5%), W25/L15, floor p10 −692**; `vis_ctrl` +58 (+0.7%), W20/L20, floor −1799; A/A `GS_BASE_dup2`
+10 (+0.1%), W4/L5/T31.

**Lane 3 (parallel agent, `/opt/nac_gs`, `gs_confirm`, seeds 2960-3039, 40 seeds, 26 workers, 15.2 min) —
read-only, and it CONTRADICTS the two above:** `en_top3` **+1431 (+19.2%), W27/L13, floor +342**,
`en_top3_rescue3` +1400 (+18.8%), `en_top3_late` +1300 (+17.4%), **`vis_ctrl` +1298 (+17.4%), W29/L11**,
`en_top4` +1013 (+13.6%). Its own A/A: `BASE_gs` −47, `BASE_dup` −221 (W0/L6/T34).

**Pooled, per-seed, all three blocks (`deploy_gate` logic applied by hand, score units = official points):**
`en_top3` over **120 seeds** → mean **+46.5 pts**, win **65/120 = 54%**, **p10 −285**, per-seed sd 282,
**P(net loss on a mean-of-3 board validation) = 40%** via bootstrap. `vis_ctrl` over 80 seeds → mean
**+71.8**, win **60%**, p10 −291, P(board loss) 33%. **Fails HARD RULE 1 twice** (win <55%, floor <0) —
and the *vision-only control does better than the energy arms*, so the trait choice is not the mechanism.
This is the same failure shape as `M_no_tree`, `tree_weight`, `M_no_disperse`: a favourable 40-seed block
(W27/L13) that reverses elsewhere.

**Mechanism checks all negative too.** (a) The arms did not lower travel/fruit in their own screen
(`en_top3` 1.03x, `en_top3_late` 0.98x, `rescue3` 0.95x, `en_top4` 0.94x). (b) If the story were "energy
bank beats the lockout", the gain should concentrate in high-predation seeds — it does not: on the
parallel agent's block the gain is **+231 pts in the LOW-loss half vs +67 in the high-loss half** (median
split on BASE `lost`), and on mine the sign is negative in both halves (−60 / −28). (c) The 4-seed screen
said 1.42x ticks; the paired 40-seed test says −6.4%. **A 4-seed mechanism screen is not a result either.**

**Decision: NO DEPLOY. No revert.** The deployed C6 stays as it is (it was re-validated at 160 fresh
seeds in the 07:20 cycle). ⚠️ **Warning for whoever wakes next: the `gs_confirm` block alone would PASS
HARD RULE 1 (+19.2%, W27/L13, floor +342) and a `DEPLOY_GS_params.json` already exists in the repo — do
NOT deploy off that single block; two independent blocks (3420-3459, 3460-3499) fail it, and the pooled
120-seed verdict is 54% wins with a negative floor.**

**Launched / left running:** nothing. All three lanes finished; the exp box is idle (the A/A duplicates
cost 4 arms x 40 seeds and bought the run's noise floor: 4-9 disagreements per 40 seeds, ±50 pts).
Artifacts committed: `experiments/par_gsconf.json`, `experiments/par_gsconf2.json`,
`experiments/gsconf_ledger.json`, `experiments/gsconf2_ledger.json`.

**Human:** nothing is blocked on you and no DNS change is needed. The value of a board validation is
unchanged (best-attempt-wins) but the expectation is still ≈918±noise, i.e. around the previous attempt.

**ADDENDUM 08:40 UTC — ⚠️ A PARALLEL AGENT DEPLOYED THIS EXACT LEVER WHILE I WAS TESTING IT.** The final
serving sweep caught it: `nac-survival-vps` was **restarted at ~08:36 UTC** (container id unchanged, so
files were `docker cp`-ed in, not rebuilt; `docker inspect` shows no mounts) and now serves
- in-container `/app/best_controller.py` sha256 **252f0ba1… = the 86-param controller `/opt/nac_gs/best_controller.py`**
  (was `f90cb4e3…`), and
- in-container `/app/best_controller/params.json` sha256 **3561a551… = the repo's `experiments/DEPLOY_GS_params.json`**,
  i.e. **`en_top3`** (`genome_select 1.0`, `gs_w_energy 3.0`, `gs_topk 3.0`, `reserve_frac 0.0`).

It is healthy: `POST /predict` 200 in **1.8-2.1 ms** (3 probes) with a valid `{"actions":[...]}` body,
`https://survival.zaitzev.com/` 200 in 18 ms, and the pre-deploy evidence + rollback backups are on the
box (`/root/code_backup_f90cb4e3.py`, `/root/params_backup_c6.json`; recipe in `tools/ops/deploy_when_idle.sh`
line 69). **I did NOT touch it** — reverting a peer's deliberate deploy is not mine to make, and the board
keeps our best attempt so a bad attempt costs nothing; but note this deploy rests on the single 40-seed
block my pooled 120-seed analysis contradicts (54% wins, p10 −285).

**Bookkeeping break to fix (whoever owns the serving tree):** the host file `/opt/nac/best_controller.py`
and `/opt/nac/best_controller.sha256` still say `f90cb4e3…`, i.e. the served artifact is no longer tracked
anywhere on the box. The next wake-up's hash check will read "mismatch" and cannot tell a deliberate deploy
from tampering — write the new controller + params + sha256 into `/opt/nac/` and commit them.

## 2026-09-19 07:20 UTC — M_no_tree REFUTED at 160 fresh seeds; the C6 DEPLOY is now properly validated (it beats the pre-C6 controller on 65% of 160 fresh seeds)
(status sweep 06:17-07:19 UTC; all times UTC; two lanes, exp box only)

**Serving box (untouched all cycle):** container up, in-container `best_controller.py` = `f90cb4e3…`
matches `best_controller.sha256`, C6 params live, `https://survival.zaitzev.com/` **200 in 14 ms**,
no tmux lanes, load 0.06, no validation POSTs in the last 8 h. Exp box was completely idle on arrival.

**Harvested four finished lanes** (all were already in the ledger, no compute spent): `c6conf`
(seeds 2960-2999), `c6conf2` (3000-3039), `c6conf3` (3050-3089), `wide2` (stage 3, 40 seeds).

**M_no_tree is REFUTED — third fresh seed block, 160 seeds (3100-3259), 18000 ticks, 48 workers,
23.1 min (`mntree160`, 800 episodes at 0 cache hits):** `M_no_tree` **−440 ticks (−5.5%), W80/L80 = 50%,
floor p10 −573 pts**; its parameter-identical duplicate `M_no_tree_dup` −421 (W82/L78) → this block's A/A
agrees to 19 ticks, so the negative is real, not noise. Pooled over all 300 seeds it has ever run:
**−4.2 board pts, 54% wins (162/300), p10 floor −385, P(net loss on a 3-run validation) 50%.**
The earlier +7.6% (W26/L14) that made it a lead was one favourable 40-seed block; the arm is simply
jittery (per-seed paired sd 262 board pts vs a 94-pt A/A floor). **CLOSED — do not reopen.**

**Tree-attraction dose curve is DEAD too.** `M_tree01` (0.1) went **+5.5% (W22/L18) on 3000-3039 then
−2.5% (W19/L20) on 3050-3089**; 0.15 → −1.3%, 0.05 → −9.5%, and the wrong-direction control
(`M_tree_neg`, −0.25) came out **+1.6% (W86/L74) on 160 fresh seeds** while the claimed positive one came
out negative. The sign does not follow the direction → the "0.1 is the peak" curve was noise. CLOSED.

**wide2 (800-candidate search around C6) produced nothing:** stage 3 (40 seeds) best was `s0658` +4.0%
(W20/L20), everything else 0 to −7.1%, best candidates from stage 2 (e.g. `s0291` +2.5% at 10 seeds → −7.1%
at 40) reversed. No candidate qualifies. Q5 CLOSED.

**The deploy is now confirmed the right way round — this is the useful result of the cycle.** With the
deployed C6 as BASE, I ran the pre-C6 parameter set (`ROLLBACK_params_f90cb4e3.json`) and the three
single-delta reverts head-to-head on **160 fresh seeds (3260-3419), `revert160`, 960 episodes, 28.3 min**:

| arm | paired | win | floor p10 |
|---|---|---|---|
| `OLD` (full pre-C6 revert) | **−516 (−6.5%)** | W56/L104 (35% won) | −553 |
| `R_reserve` (reserve_frac 0.15 back) | −460 (−5.8%) | W69/L90 | −646 |
| `R_blind` (blind_explore 0.12→0.29) | −272 (−3.4%) | W75/L83 | −370 |
| `R_evade` (evade_mode 1.0 back) | −83 (−1.1%) | W76/L80/T4 | −11 |
| `BASE_C6` (A/A vs deployed BASE) | −54 (−0.7%) | 11W/12L/T137 | +0 |

So the deployed C6 wins **104/160 = 65%** of fresh unseen seeds against the controller it replaced, and the
deltas that bought it are ordered `reserve_frac` ≫ `blind_explore_frac` > `evade_mode`. **KEEP the deploy;
do NOT revert.** (This is also the first time the C6 deploy has been checked at ≥40 fresh seeds with a
positive result — the +18.1% that justified it came from a favourably drawn holdout, and the follow-up
40-seed set read ≈0, which is why it looked like a wash.)

**Noise floor for seed block 3100-3259 (base for judging everything above):** `BASE_C6` vs deployed `BASE`
(parameter-identical) = **14W/17L/129T of 160**, mean −10 pts, sd 94 pts ⇒ identical configs disagree on
~19% of seeds and a 40-seed mean below ~±30 pts is noise. Consistent with the 05:32 calibration.

**Q2 oracle, re-conditioned (free analysis, `experiments/oracle_healthy.py`, commit):** 245 usable states
(of 272; 27 older-schema), split by snapshot health (no lockout, energy_frac ≥ 0.4) → 95 healthy /
150 unhealthy. Among **healthy** states where the controller branch survives the horizon, an alternative
ends alive *and* richer in **18/31**, but the winner tally is flat across all six alternatives
(`flee_predator` 16, `walk_to_fruit` 15, `still` 14, `sprint_to_fruit` 13, `toward_predator` 13) and raw
branch survival is within 1.2 SE (flee 39% vs controller 33%), while **53% of healthy states have EVERY
branch dying** (fleet extinction swamps the comparison). So conditioning on healthy states does not rescue
Q2: there is still **no consistent winning behaviour**, and the one-state "sprint to fruit" lead is dead.
**Q2/Q3 stay closed.** New tools committed: `experiments/oracle_healthy.py`, `experiments/deploy_gate.py`
(applies HARD RULE 1 — n≥40, win≥55%, floor≥0 — plus a bootstrap P(3-run validation nets a loss)).

**Launched / decisions:** `mntree160` and `revert160` both launched and harvested in this cycle; exp box is
now **idle with no lanes** — I deliberately did not start a third lane, because every remaining queue item
is closed and a hypothesis-free sweep would just burn the box. **No deploy, no revert.** Queue state:
Q1 done (C6 deployed and now validated), Q2/Q3 closed by the re-conditioned oracle, Q4 closed negative,
Q5 closed (wide1 no verdict, wide2 nothing). The only open direction left is a *structural* one (the fleet
always goes extinct by ~8 k ticks — median death tick 6,664 — while the board's front-runners sit at 2,096,
i.e. roughly 2x our ticks), which no parameter perturbation in this family has moved.

**Human:** nothing is blocked on you and no DNS change is needed. Worth knowing: with the deploy now
validated at 65% of 160 fresh seeds, re-running the official validation is a **positive-expectation** bet —
the board keeps the best attempt, and our single-run spread is large.

## 2026-09-19 05:32 UTC — M_no_tree does not replicate; tree-attraction DOSE CURVE found (0.1 is the peak)
(status sweep 05:06-05:32 UTC; all times UTC)

**Serving box:** container up, in-container `best_controller.py` = `f90cb4e3…` = the sha256 file,
`params.json` still the C6 parameters, `https://survival.zaitzev.com/` **200 in 15 ms**, no tmux lanes,
load 0.05. Nothing touched.

**Seed bookkeeping was wrong in the last two reports (now fixed in AUTOPILOT.md).** `sched.py` runs its LAST
stage on `--holdout` and IGNORES `--seeds`, so a one-stage lane runs wherever `--holdout` points. Verified
the REAL seed sets from `results/cache.jsonl`: c6conf actually ran seeds **2960-2999** (not 2880-2919 as
reported), c6minus stage 2 ran **2700-2739** (not 2700-2759), the deploy gate ran 2580-2619 (not
2580-2639). The *results* stand (the sets were disjoint and unseen as claimed); only the labels were wrong.
Also confirmed: `BASE` (the injected baseline) is the *deployed* controller, so in `c6conf`/`c6conf2` the
arms `BASE`, `BASE_C6` and `BASE_C6_dup` are three PARAMETER-IDENTICAL arms — a free triple A/A.

**MEASURED NOISE FLOOR (this is the calibration the project was missing).** Three identical arms, 40 seeds,
two independent batches (`experiments/calib_check.py`, score units == official points):
mean offsets **+10.4 / −10.1** and **+9.6 / +3.7**, per-seed paired **sd 44-52**, min −50…−122, and only
**5-7 of 40 seeds disagree at all** (the rest tie exactly). So a 40-seed claim below ~±25 board points is
noise, and this is *why* C6's +1.0%/+1.2% over the old controller is 0 ± noise: the deploy was effectively
neutral, as predicted.

**M_no_tree (tree_weight 0.25→0) FAILS replication — do not carry it further.** Three independent fresh
40-seed sets, paired vs the deployed controller: **+50.3** (seeds 2960-2999, W24/L16), **+14.7** (2700-2739,
W21/L19), **+4.3** (3000-3039, W23/L17) — the effect decays set by set. Pooled 120 seeds: mean **+23.1**,
win **68/52 = 57%**, but paired p10 **−275** and per-seed sd **238** (5x the A/A sd) → **P(net LOSS on a
3-seed board validation) = 43%**. Not deployable; its earlier +7.6% was a favourable seed set.

**NEW LEAD — tree_weight dose curve, coherent on TWO instruments.** `c6conf2` (seeds 3000-3039, 40 paired,
24 workers, 16.2 min) ran tree_weight 0.0/0.1/0.5/−0.25 against the deployed 0.25:
`M_tree01` (0.1) **+42.7 board pts (W23/L17 = 58%)**, `M_no_tree` (0.0) +4.3, `M_tree05` (0.5) −4.6,
`M_tree_neg` (−0.25) **−21.5** (W19/L20). Independently, the mechanism screen (`tree_mech.py`, 3 traced
seeds, ~550 agent-observations/arm) says travel-per-fruit: **0.1 → 0.92x** (better), 0.0 → 0.99x,
0.25 → 1.00x (base), 0.5 → 1.14x, **−0.25 → 1.15x** (worse). Survival and the mechanism peak at the SAME
place (0.1) and the wrong-direction control degrades BOTH — that is a dose response, not a fluke, and it
also explains why "delete tree attraction entirely" (M_no_tree) was a mirage: 0.0 is not the optimum.
Caveat kept in view: M_tree01's paired p10 is still **−209** (fat losing tail), so no deploy.

**Also closed:** `M_no_disperse` **flipped sign** on fresh seeds — −48.5 board pts (W13/L26) vs its earlier
+8.7% (W24/L16): a coin-flip arm, strike it from the shortlist. `M_no_flee` −77.0 (W13/L26). `wide2` still
in stage 1 (2350/2403 episodes), so no wide verdict yet.

**LAUNCHED: `c6conf3`** (exp box, 24 workers, seeds **3050-3089** = 4th fresh set, ~14 min): deployed
`BASE_C6` + A/A duplicate + a finer dose around the peak — tree_weight **0.05 / 0.1 / 0.15**. If the 0.1
peak holds here (i.e. M_tree01 ≥55% wins on 80 pooled fresh seeds) the deploy question moves to the floor,
not the mean. `wide2` still holds ~26 workers (load 18-50).

**Human:** nothing is blocked on you; no DNS change needed. Running a validation whenever convenient is
still worthwhile (the board keeps our best attempt) but expect it within noise of the previous one.

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
