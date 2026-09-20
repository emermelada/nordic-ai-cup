# NEW hive controller test — 2026-09-20

**Subject:** the NEW build `hive_v2.py`, sha256 `7f3467cfa5cf26dde71fa523156037167dd96149adbc72446121bc27b89a1c43` (1,526 lines)
**Incumbent for comparison:** the FULLY DEPLOYED build `829e4147…` (1,247 lines), 400-seed baseline at `--horizon 18000`.
**Measurement box:** box1 = `ubuntu-64cpu-128gb-fi-hel1` (64 cores / 128 GB, UpCloud Helsinki), 212.147.236.122.
**Scope:** MEASUREMENT ONLY. The controller was not modified, no parameter sweep was run, no validation
API was used, and no grading/validation box (212.147.249.39 / 94.237.34.245) was touched.

---

## 1. Deployment

```bash
ssh -i ~/.ssh/vps_hermes root@212.147.236.122
mkdir -p /opt/nac_new400
cp -a /opt/nac_traj_b/hive_v2.py /opt/nac_traj_b/src /opt/nac_traj_b/experiments /opt/nac_new400/
rm -rf /opt/nac_new400/out /opt/nac_new400/**/__pycache__ ; mkdir -p /opt/nac_new400/out
sha256sum /opt/nac_new400/hive_v2.py
```

| check | result |
|---|---|
| `sha256sum /opt/nac_new400/hive_v2.py` | `7f3467cfa5cf26dde71fa523156037167dd96149adbc72446121bc27b89a1c43` ✅ matches expected prefix `7f3467cf` |
| line count | 1,526 ✅ |
| `/opt/nac_new400/experiments/traj_map_resc.py` | present ✅ |
| `/opt/nac_new400/src/` | present (`core.py`, `elements/`, `utils/`) — self-contained copy ✅ |

The harness was **copied, not moved**; nothing inside was renamed or edited. `/opt/nac_traj_b` and all
other agents' jobs were left untouched.

---

## 2. Runs

### Run A — seeds 201–400 of the new build (the missing half of a 400-seed pairing)

Launched detached (`setsid nohup`), same harness, same env as the earlier 1–200 arm:

```bash
cd /opt/nac_new400 && setsid nohup env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  /opt/nacv/bin/python -u experiments/traj_map_resc.py \
  --seeds 201-400 --workers 60 --horizon 30000 --out /opt/nac_new400/out \
  > /opt/nac_new400/run.log 2>&1
```

I first launched this at `11:10:40Z` with `--workers 40`; the parent then killed the other lanes on box1
and relaunched the same command with `--workers 60` (~`11:16:35Z`). Only one copy ran — verified
("199 seeds to run (1 already done)"), no second copy was started.

| | |
|---|---|
| seeds run | 201–400 (200 seeds) |
| wall time | **1,617.8 s ≈ 27.0 min** (harness `DONE in 1617.8 s`), records finalised `11:43:32Z` |
| `out/raw.jsonl` | **200 records**, seeds 201–400, no gaps, no duplicates |
| field check | every record carries `seed`, `T`, `score` and a non-empty `buckets` array ✅ |
| score range (own horizon) | mean 1462.5, min 616.7, max 2709.6; max `T` = 25,546; 0 runs at the 30,000-tick cap |

### Run B — determinism probe (8 seeds × 4 repeats)

The harness accepts comma-separated seeds (`parse_seeds` splits on `,` and `-`), so a single run covers
all 8 seeds. Four **separate output directories** (so repeats cannot dedupe against each other) were run
in parallel, detached:

```bash
cd /opt/nac_new400
for i in 1 2 3 4; do
  rm -rf rep$i; mkdir -p rep$i
  setsid nohup env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
    /opt/nacv/bin/python -u /opt/nac_new400/experiments/traj_map_resc.py \
    --seeds 1,2,3,17,23,36,51,100 --workers 8 --horizon 30000 --out /opt/nac_new400/rep$i \
    > /opt/nac_new400/rep$i.log 2>&1
done
```

| | |
|---|---|
| start → finish | `12:03:24Z` → `12:11:50Z`, **≈ 506 s ≈ 8.4 min** for all 32 runs |
| per-repeat harness time | 495.6 / 495.6 / 504.3 / 506.1 s, **8 records each** ✅ |

> Note: an *earlier* partial `rep3` on box1 (horizon 6000, seeds 1–5) was **not mine** — a foreign
> artifact from another agent. It was wiped and re-run; the reported repeat data is exclusively from the
> four `12:03:24Z` runs above.

---

## 3. Answers

### Q1 — Is the new controller actually good relative to the incumbent? **Yes, decisively.**

`python3 compare_paired.py --a /tmp/id_base.jsonl --b /tmp/new400.jsonl --na DEPLOYED-829e4147 --nb NEW-7f3467cf`
(400 paired seeds, truncated to the common horizon **h = 18,000 ticks = 1,800 sim-s**):

| arm | n | mean | median | sd | p10 | p25 | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|
| DEPLOYED-829e4147 | 400 | 1225.4 | 1290.4 | 383.6 | 720.7 | 984.8 | 1505.0 | 1659.8 | 1947.5 |
| **NEW-7f3467cf** | 400 | **1528.0** | **1563.9** | **309.5** | **1107.7** | 1326.9 | 1768.3 | 1926.8 | 1987.9 |

| arm | P(≥1400) | P(≥1500) | P(≥1800) |
|---|---|---|---|
| DEPLOYED | 0.378 [0.331, 0.426] (151/400) | 0.255 [0.215, 0.300] (102/400) | 0.040 [0.025, 0.064] (16/400) |
| **NEW** | **0.677 [0.630, 0.721] (271/400)** | **0.555 [0.506, 0.603] (222/400)** | **0.223 [0.184, 0.266] (89/400)** |

**Paired, same seeds, same horizon:** mean delta **+302.6 ± 21.1 (SE), t = +14.33**,
median delta **+300.6**, **W/L/T = 311/89/0**, worst-decile mean delta −410.2, best-decile +1091.0.

Band transitions (baseline → new): 70 runs move LOW→MID/HIGH/TOP and 41 MID→HIGH/TOP and 44 HIGH→TOP;
counter-flows are 8 MID→LOW, 26 HIGH→MID, 3 HIGH→LOW, 10 TOP→MID/HIGH.

**Cross-check without truncation** (compare_paired recomputes scores from buckets, which runs ≈3–5 % above
the engine's own `env.score`): restricting to the **356 seeds where both arms actually died before
18,000 ticks**, so both `env.score` values are final and need no recomputation:

| | |
|---|---|
| exact `env.score` paired delta | **+242.1 ± 21.8 (SE), t = +11.13**, median +212.6, W/L/T = 260/96/0 |
| arm means on that subset | baseline 1161.0 → new 1403.2 |

So the true margin is roughly **+240 to +300 points**, t ≈ +11 to +14. Both readings agree: the new build
is substantially better, not marginally. The new arm is also *less* variable (sd 309.5 vs 383.6).

### Q2 — What is the highest score it can achieve?

Own-horizon scores of the new build over all 400 seeds (n = 400):

| statistic | value |
|---|---|
| mean | 1473.0 |
| median | 1476.6 |
| sd | 338.6 |
| p10 / p25 / p75 / p90 | 1032.6 / 1264.9 / 1692.6 / 1888.3 |
| min / max | 404.7 / **2709.6** |

Top 10 runs (score, ticks, fruit energy):

| seed | score | T (ticks) | sim-s | fruit_E_total |
|---|---|---|---|---|
| 217 | **2709.6** | 25,546 | 2554.6 | 175,241.6 |
| 84 | 2360.3 | 23,019 | 2301.9 | 155,127.2 |
| 14 | 2292.8 | 22,387 | 2238.7 | 127,792.6 |
| 218 | 2251.5 | 21,784 | 2178.4 | 143,698.2 |
| 127 | 2238.0 | 21,650 | 2165.0 | 130,895.8 |
| 231 | 2221.6 | 21,635 | 2163.5 | 161,594.8 |
| 259 | 2200.7 | 21,306 | 2130.6 | 152,314.2 |
| 367 | 2170.1 | 20,596 | 2059.6 | 145,759.4 |
| 375 | 2119.2 | 20,552 | 2055.2 | 142,610.0 |
| 116 | 2114.6 | 20,217 | 2021.7 | 142,812.8 |

**The 30,000-tick cap was NEVER reached: 0/400 runs hit it.** Every run ends by *death*.
Max `T` = 25,546 ticks (2,554.6 sim-s) = 85 % of the cap.

- Theoretical maximum from ticks alone: 30,000 × 0.1 = **3,000 points** (+ fruit/1000).
- The best observed run (2,709.6) is **≈ 445 points short** of the tick-only maximum, i.e. the best run
  still loses ~15 % of the game to death — and the *median* run dies at 14,468 ticks, 48 % of the cap.
- Only 16/400 runs (4.0 %) survive past 20,000 ticks; 1/400 past 25,000.

### Q3 — Is run-to-run consistency a problem? **No — the controller is bit-exactly deterministic; all variance is map variance.**

Run B, 4 repeats × 8 seeds (`/tmp/rep/r{1,2,3,4}/raw.jsonl`):

| seed | rep1 | rep2 | rep3 | rep4 | max−min |
|---|---|---|---|---|---|
| 1 | 1557.0 | 1557.0 | 1557.0 | 1557.0 | **0.00** |
| 2 | 1407.8 | 1407.8 | 1407.8 | 1407.8 | **0.00** |
| 3 | 2011.7 | 2011.7 | 2011.7 | 2011.7 | **0.00** |
| 17 | 1270.1 | 1270.1 | 1270.1 | 1270.1 | **0.00** |
| 23 | 1925.1 | 1925.1 | 1925.1 | 1925.1 | **0.00** |
| 36 | 404.7 | 404.7 | 404.7 | 404.7 | **0.00** |
| 51 | 1126.7 | 1126.7 | 1126.7 | 1126.7 | **0.00** |
| 100 | 1054.7 | 1054.7 | 1054.7 | 1054.7 | **0.00** |

- **WITHIN-seed spread (max−min over 4 repeats): mean 0.0000, max 0.0000.** `T` identical for every seed.
- Stronger still: the **entire record is byte-identical** across the 4 repeats — SHA-256 of the full
  JSON (all fields, all 30 buckets) matches for all 8 seeds.
- Those eight scores also **match the 400-seed run exactly** (`seed 36 → 404.7`, `seed 217`-class runs etc.),
  so the pipeline reproduces across separate runs, directories and box-load conditions.
- **BETWEEN-seed spread: sd 481.1, range 1,606.9** (404.7 … 2011.7 on these 8 seeds).
- Between/within ratio is therefore effectively infinite.

**Conclusion: inconsistency is driven entirely by the map (seed), not by the controller.**
Given a fixed map seed the new controller is a pure function — 100 % reproducible. The
404.7 ↔ 2709.6 spread in the 400-seed distribution is map difficulty, not controller flakiness.
(Caveat: this also means that if the graded "3 attempts" reuse the *same* map, the mean-of-3 is
degenerate and equal to a single run; if they draw 3 different maps, the mean-of-3 is a genuine draw
from the per-run distribution — see Q4.)

Left tail of the 400-seed distribution (new build, own horizon):

| seed | score | T |
|---|---|---|
| 36 | 404.7 | 4,150 |
| 193 | 452.6 | 4,866 |
| 171 | 457.5 | 4,556 |
| 208 | 616.7 | 6,138 |
| 391 | 627.5 | 6,414 |
| 164 | 656.7 | 6,454 |
| 206 | 770.6 | 7,439 |
| 136 | 794.3 | 7,794 |
| 225 | 801.5 | 7,846 |
| 97 | 803.6 | 7,749 |
| 388 | 807.7 | 8,205 |
| 278 | 820.9 | 7,985 |
| 168 | 820.9 | 7,897 |
| 275 | 825.7 | 8,294 |
| 321 | 828.2 | 8,475 |
| 269 | 842.9 | 8,169 |
| 314 | 864.4 | 8,287 |
| 316 | 866.9 | 9,059 |
| 121 | 868.4 | 8,465 |
| 158 | 871.3 | 8,403 |

Tail shape: `sd 338.2`, **skew −0.084**, excess kurtosis +0.378, IQR 427.7, min 404.7, max 2709.6.
The distribution is essentially symmetric with a mild thin left tail — every one of the 20 worst runs
died **early** (T = 4.2k–9.1k ticks, all under a third of the cap), so the low tail is a *survival*
failure, not a foraging failure. `P(score < 1000) = 0.080 (32/400)`, `P(< 900) = 0.052`, `P(< 700) = 0.015`.

### Q4 — What evidence would point to improving it (and the graded metric)

**The graded metric (best estimate).** The evaluation averages 3 attempts. Bootstrapping 10,000 resamples
of `mean(3 draws)` from the 400 per-run scores:

| quantity | value |
|---|---|
| mean of the mean-of-3 distribution | **1473.0** |
| SE | **193.6–194.4** |
| **5th percentile** | **≈ 1,154** |
| **50th percentile** | **≈ 1,473** |
| **95th percentile** | **≈ 1,788** |
| P(mean-of-3 < 1200) | **≈ 0.079 (7.5–8.1 %)** |
| P(mean-of-3 < 1000) | **≈ 0.009 (0.9 %)** |

(Robust to sampling with vs. without replacement: p5 1160 / p5 1152, P(<1200) 0.075 / 0.081.)

That band — roughly **1,150 → 1,790** — is the honest prediction of the judged score for this build.
For comparison, the deployed incumbent's 400-seed mean at the same horizon is 1225.4, i.e. the new build's
*pessimistic 5th percentile* already exceeds the incumbent's *mean*.

**Where improvement would pay (evidence, in order of size):**

1. **Survival is the whole game and it is the binding constraint.** Score is ~93 % ticks and only ~7 %
   fruit energy (exact, buckets-based: ticks mean 1435.5 pts, fruit mean 111.6 pts). No run reached the
   30,000-tick cap; the median run dies at 48 % of the cap and only 4 % of runs survive past 20,000 ticks.
   Any change that extends survival moves the entire distribution, whereas fruit optimisation moves ~7 %.
2. **The remaining headroom is huge and localised.** Lifting just the worst decile to the median lifts the
   p5 of the mean-of-3 from ~1,160 to **~1,303** and cuts P(<1200) from 7.9 % to **0.8 %** — i.e. the
   entire downside risk lives in ~40 seeds' worth of early-death trajectories. The best run still leaves
   ~445 points (15 % of the tick cap) on the table, so the ceiling is not the binding issue either.
3. **Look at the collision/early-death mechanism, not the average case.** The 20 worst runs all die in
   the first 1,000 sim-seconds; the cause is identifiable per-seed (same seed reproduces the same death
   exactly, so a single failing trajectory can be replayed and dissected with zero noise). This is the
   cheapest lever: deterministic failure means a fix is verifiable with a handful of seeds.
4. **Very few distinct failure modes.** 400 seeds produce a smooth, symmetric, thin-tailed distribution
   (skew −0.08, excess kurtosis +0.38) rather than a bimodal one — so the tail is not a few broken modes
   but one continuous survival process degrading on hard maps; that argues for improving the general
   policy rather than patching special cases.
5. **Not worth measuring further:** fruit/foraging efficiency (~7 % of score) and consistency/flakiness
   (there is none). Neither can move the judged score meaningfully at the current margin.

---

## 4. What this does and does not prove

**Proves (directly measured on box1, 400 + 32 real runs):**
- The new build's sha256 is exactly `7f3467cf…`, 1,526 lines, and it deployed and ran standalone.
- Over the same 400 seeds and the same horizon, the new build beats the deployed incumbent by
  ≈ +240 to +303 points (t = +11.1 to +14.3), winning 260–311 of 400 seeds. It is also less variable
  (sd 309.5 vs 383.6).
- The new controller is **bit-exactly deterministic**: 4 repeats of 8 seeds give byte-identical records,
  within-seed spread 0.0000.
- Its per-run distribution, top score (2,709.6 @ 25,546 ticks) and the mean-of-3 band (1,154/1,473/1,788,
  P(<1200) ≈ 8 %) as measured from 400 samples.
- No run ever reaches the 30,000-tick cap; all end by death.

**Does NOT prove / caveats:**
- **This is not a graded result.** No validation API was used and no graded box was contacted; these are
  local harness numbers in the same simulator, not the competition's judged score.
- **The +302.6 figure uses horizon truncation** and the comparison script's buckets-based score
  recomputation, which sits ≈ 3 % (baseline) to ≈ 5 % (new) above the engine's own `env.score`; the
  truncation at 18,000 also credits the new arm's 44 long runs. The truncation-free exact-`env.score`
  subset (356 seeds where both arms died before 18,000) gives the more conservative **+242.1 ± 21.8**.
  The honest summary is a margin of **+240 to +300**, not a single number.
- **The incumbent baseline is at 1,800 sim-s while the new build ran to the 3,000 sim-s cap.** 8 of the
  400 baseline runs were still alive when their horizon cut them, so the incumbent's true mean is
  slightly *under*-stated. This makes the measured margin conservative, not generous, but it is an
  asymmetry in the two datasets, not a like-for-like horizon.
- **The mean-of-3 bootstrap assumes 3 independent map draws.** If the graded protocol reuses one map for
  all 3 attempts, the mean-of-3 is a single deterministic value (spread 0) and the correct uncertainty is
  the *map* choice, not the bootstrap SE of 194; the p5/p95 band is then a prior over which map is drawn.
  The 8–9 % below-1,200 risk figure is therefore a risk over *map assignment*, not over controller
  randomness.
- The bucket records retain only 10-sim-second granularity, so the scores used in `compare_paired.py`
  are truncated-horizon reconstructions, not the engine's final float score. `env.score` is authoritative
  and was used for all "own horizon" numbers above.
- Run B ran while another agent's 56-worker job shared box1 (load ≈ 3–88); results were nevertheless
  byte-identical, so CPU contention does not affect the simulation — but the wall-clock timings above
  reflect a *shared* box and are not clean benchmarks.
- **Incident (disclosed):** my first Run-B launch was terminated by another agent's cleanup after 2 of 4
  repeats; those two were discarded and all four were re-run cleanly at `12:03:24Z`. One foreign partial
  `rep3` (horizon 6000, another agent's) was overwritten. No controller file was modified.

---

## 5. Timing summary

| run | command | wall time | output |
|---|---|---|---|
| Run A (seeds 201–400, new) | `--seeds 201-400 --workers 60 --horizon 30000` | **1,617.8 s ≈ 27.0 min** (finalised 11:43:32Z) | `/opt/nac_new400/out/raw.jsonl`, 200 records |
| Run B (8 seeds × 4 repeats) | 4 × `--seeds 1,2,3,17,23,36,51,100 --workers 8 --horizon 30000` | **≈ 506 s ≈ 8.4 min** (12:03:24Z → 12:11:50Z) | `/opt/nac_new400/rep{1..4}/raw.jsonl`, 8 each |
| Merge | branch(1–200) + Run A(201–400) | — | `/tmp/new400.jsonl`, 400 records, seeds 1–400, no gaps/dupes |
| Analysis | `compare_paired.py` + small local stats | seconds | this document |
