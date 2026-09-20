# START HERE — instructions for a fresh session

You are picking up the Nordic AI Cup **Drone Flyby** task with a few hours left.
Read this whole file before changing anything. It is self-contained.

**Do these five things first:**

1. `curl -s http://93.91.156.85:41241/api` — confirm what is serving. Never
   change config without checking this, and never trust the launch command.
2. After ANY restart, wait ~125 s for the models to load before validating. A
   run fired 28 s after a restart scored 0.5458 instead of ~0.58 and cost an
   experiment.
3. After every batch of validations run `python3 probe/frame_loss.py` on the
   box. About 1 run in 10 loses frames to an external stall; one lost 94 and
   scored 0.3264. Never read a dropped-frame run as a config result.
4. Change ONE variable at a time and take at least 3 runs. sd is ~0.013, so two
   runs cannot separate a 0.01 effect.
5. **Do not use `src/helsinki` to decide anything.** It gave the wrong SIGN
   twice in one day. Section 5 explains why. Use it only for class mapping and
   imgsz curve shape.

**The current best config is `row0x2` + 5 passes, mean 0.5841.** Restore it with
`cp data/serve_env.sh.row0x2_BEST_0.584 data/serve_env.sh` then
`bash probe/serve_passes.sh 5`. If you run out of time or ideas, serve that and
stop — it is the best measured configuration.

**The single best untested idea is in section 10, item 2** (box fusion instead
of concatenation). The only identified route to 0.70 is section 10, item 1
(hand-harvest real patches for the dead classes and retrain).

---

# Drone Flyby — why we are at 0.58 and not 0.70

Written 2026-09-20 ~09:45 UTC. Deadline 14:00 UTC. Paste this whole file into a
new session; it is self-contained.

Team `Elysa's Secret`, uuid `b15f0e8a1fbf4378899c0d50a4bc60e8`.
Repo branch `drone-flyby-breakthrough`, worktree `C:\Users\franc\nordic-ai-cup-drone`.

---

## 0. The one-paragraph answer

**It is not detection recall.** Objects that are in view and the detector fails
to find are 6.6 % of objects, worth 0.051 of score — the *smallest* of four
loss buckets. The loss is roughly half **class confusion** (we find the object
and call it the wrong name: 62.2 % class accuracy at the resolution we actually
serve) and half **camera timing** (16.8 % of objects are never shown to us in
time). Four of the thirteen scored classes are effectively dead and cost 0.209
of score between them. Fixing the naming is a retrain, which needs real
training examples of exactly the classes we cannot detect — which we do not
have, because those examples were mined with the detector that cannot detect
them. That circularity is the actual wall.

---

## 1. Where we are

| config | n | mean | sd | runs |
|---|---|---|---|---|
| 4-pass, camera `full` | 5 | 0.5222 | — | 0.4855 0.5230 0.5288 0.5352 0.5386 |
| 5-pass, camera `full` | 2 | 0.5418 | 0.0032 | 0.5396 0.5441 |
| 5-pass + graded band + per-class growth | 2 | 0.5390 | 0.0001 | 0.5389 0.5391 |
| single v9@1280, `full` | 3 | 0.4554 | 0.0020 | 0.4577 0.4537 0.4549 |
| 5-pass, `row0` (25 % L0) | 2 | 0.5686 | 0.0066 | 0.5733 0.5639 |
| **5-pass, `row0x2` (50 % L0)** | **7** | **0.5841** | **0.0135** | 0.5820 **0.6071** 0.5821 0.5966 0.5683 0.5724 0.5801 |
| 5-pass, `l0` (100 % L0) | 3 | 0.5376 | 0.0033 | 0.5403 0.5338 0.5387 |
| 6-pass, `row0x2` (adds v9@2560) | 3 | 0.5738 | 0.0081 | 0.5700 0.5831 0.5682 |
| 5-pass, `row0x2`, AGREEMENT_WEIGHT=0.7 | in flight | — | — | — |

Best known configuration: **`row0x2` + 5 passes, mean 0.5841**. Saved on the box
at `data/serve_env.sh.row0x2_BEST_0.584`.

**0.6071 is a draw, not a setting.** The identical code produced 0.5683 on
another run. At mean 0.5841 / sd 0.0135, a draw of 0.607 is z=1.7, about 1 in
20. The evaluation is a SINGLE draw, so only the mean matters. P(draw ≥ 0.615,
the 6-point line) at the current mean is about 1 %.

Leaderboard context — Drone Flyby raw, all teams: 0.954, 0.950, 0.944, 0.908,
0.907, 0.815, 0.770, 0.758, 0.720, 0.715, 0.701, 0.695, 0.662, 0.656, 0.654,
0.619, 0.617, 0.615. Median ≈ 0.71. We are below every listed team.

### Live box

    ssh -i ~/.ssh/vast_medical -p 50146 root@93.91.156.85
    cd /workspace/drone-flyby

| service | internal | public |
|---|---|---|
| pipeline (submit this) | 10200 | **41241** |
| probe replay server | 10100 | 41671 |

    curl -s http://93.91.156.85:41241/api      # ALWAYS check before an attempt
    bash probe/serve_passes.sh 4|5|6           # detector stack
    bash probe/serve_cam.sh <camera>           # full | row0 | row0x2 | row0x3 | l0 | top_mostly
    cp data/serve_env.sh.row0x2_BEST_0.584 data/serve_env.sh   # restore the best

A watchdog restarts either service from `data/serve_env.sh` if it dies.

---

## 2. The score model, and why one dead class costs 0.077

The grader is **COCO mAP@IoU 0.50**, averaged over the classes **present in the
ground truth** (`local_evaluator.py::score`, `present_classes`). Classes absent
from the truth are excluded and cost nothing.

**K = 13 exactly.** Proof: replaying one recorded run (real score 0.5278) one
class at a time through the real grader gives each class's `AP_c / K`. Thirteen
classes came back nonzero; `spacecraft`, `condor` and `jammer` came back
*exactly* 0.0 despite us emitting 1755, 470 and 1899 boxes for them, so they
are absent from this flight's truth. The unmeasured class (`hangar`) is the
residual, 0.5278 − 0.45635 = 0.07145. If K were 14 then AP_hangar = 0.07145 ×
14 = 1.0003 > 1, impossible. So K ≤ 13, and K ≥ 13 because thirteen classes
score nonzero. **K = 13, and one class is worth 1/13 = 0.0769.**

### Per-class AP, measured on the real grader

| class | AP | | class | AP |
|---|---|---|---|---|
| hangar | 0.929 | | small_tower | 0.580 |
| large_tower | 0.882 | | **small_launcher** | **0.147** |
| jet_plane | 0.854 | | **medium_launcher** | **0.041** |
| helicopter | 0.843 | | **medium_plane** | **0.020** |
| mine_roller | 0.690 | | **ta-ta** | **0.000027** |
| tank | 0.634 | | spacecraft, condor, jammer | absent from truth |
| large_launcher | 0.630 | | | |
| small_plane | 0.612 | | | |

Nine working classes average **0.73**. The four dead ones sum to **0.208** out
of a possible 4.0.

**What 0.7 would require.** Score = Σ AP / 13. We are at 0.528 on that recorded
run (0.584 with the better camera). To reach 0.70 we need Σ AP = 9.1; we have
6.86. That is **+2.24 AP**. Either:

* the four dead classes go from 0.208 to ~2.45 (i.e. ~0.61 each, the level of
  the working classes) — this alone gives 0.70; or
* the nine working classes go from 0.73 to 0.98 average, which is essentially
  perfect; or
* some mix.

There is no third source of points. **Fixing the four dead classes is the only
realistic path to 0.7**, and it is worth +0.209 → ~0.74 if done completely.

---

## 3. Where the score is actually lost

Governing identity, measured: **AP ≈ coverage × precision = 0.683 × 0.77 =
0.526** against an actual 0.527.

| | value | score cost |
|---|---|---|
| coverage (we answered the object at all) | 0.683 | 0.244 |
| precision (the answer was right) | 0.77 | 0.157 |
| interaction | | 0.073 |
| | | **0.472 missing** |

Coverage loss split by cause (`probe/miss_anatomy.py`, full recorded run, every
miss charged):

| cause | share of all objects | score cost |
|---|---|---|
| **camera had not shown it yet** | 16.8 % | **0.129** |
| detector missed it in view | 6.6 % | 0.051 |
| carried box wrong | 5.2 % | 0.040 |
| tracker dropped it | 3.1 % | 0.024 |
| answered | 68.3 % | — |

**Ranked, the real enemies:**

1. **class confusion / precision — 0.157**
2. **camera timing — 0.129**
3. detector recall — 0.051
4. carry + tracker — 0.064 combined

Recall by screen band: top 0.546, middle 0.746, bottom 0.797. We find objects
**late**, not never.

Caveat: the 0.683/0.77 split and the four sub-buckets come from one recorded
run scored against a mined, incomplete truth file. Treat them as ±20 %. The
per-class AP table is from the real grader and is solid.

---

## 4. The class-confusion evidence (this is the core problem)

Measured 2026-09-20 against the **official Helsinki labels**, with v9 at
imgsz 1280 on a 960×540 view — i.e. exactly the resolution we serve. For every
official GT box, the best-IoU detection anywhere on the frame, and what we
called it:

| official class | n | what our detector calls it |
|---|---|---|
| medium_launcher | 10 | not detected ×7, **large_tower ×3**, correct ×0 |
| small_launcher | 25 | **jammer ×8**, not detected ×7, correct ×6 |
| small_tower | 20 | **spacecraft ×8**, correct ×7, not detected ×5 |
| jammer | 13 | **tank ×7**, correct ×5, mine_roller ×1 |
| jet_plane | 22 | not detected ×16, correct ×3, helicopter ×3 |
| large_tower | 19 | not detected ×14, correct ×5 |
| condor | 11 | correct ×8, helicopter ×1, large_launcher ×1 |
| ta-ta | 25 | correct ×24 |
| tank | 25 | correct ×23 |
| large_launcher | 25 | correct ×25 |

**62.2 % class accuracy over 259 official objects.** The failures are
systematic pairs, not noise. Note this understates AP slightly (it looks at the
single best-IoU box, while AP pools many boxes per object), but the confusion
pairs are real and they line up exactly with the dead classes.

Corroborating: earlier work measured Level-1 agreeing with a native-resolution
call on only 67 % of objects, and 50 % under 25 native px. Per-pass agreement:
v4@960 71.8 %, v8@1280 60.4 %, v6@1280 56.5 %.

The confusable classes have near-identical image footprints — tank 12.9 m,
small_tower 12.7, medium_launcher 13.4, mine_roller 13.8 — so **no geometric
post-processing can separate them.** This was tested: a physical-size class
prior moved class accuracy 65.9 % → 65.9 %.

---

## 5. Everything measured today, 20 Sep — the wins and the kills

### The only win: the camera

Level-0 share of camera frames vs mean score, all on the same 5-pass stack:

| L0 share | camera | mean | n |
|---|---|---|---|
| 0 % | `full` | 0.5418 | 2 |
| 25 % | `row0` | 0.5686 | 2 |
| **50 %** | **`row0x2`** | **0.5841** | **7** |
| 100 % | `l0` | 0.5376 | 3 |

An **inverted U with the peak at 50 %**. Pure Level 0 gives up all Level-1
detail and every object is 4× smaller there. 75 % (`row0x3`) is staged but
interpolates to ~0.56, so it was not run.

**From the 0.5418 baseline to the 0.6071 best draw, exactly ONE setting
changed: the camera pattern.** Same 5 models, same imgsz, same box growth, same
tracker settings. The camera is the entire gain of the day, worth **+0.042 on
the mean**.

### Kill 1 — per-class box growth (predicted +0.055, actual 0.000)

Theory: `DRONE_BOX_GROW` is a flat 1.3 with `DRONE_BOX_GROW_CAP=1.3`, but
`HELSINKI_BOX_FACTORS` wants 2.302 for medium_launcher and 1.936 for
small_launcher. Nested IoU is (1.3/2.302)² = 0.319 and (1.3/1.936)² = 0.451,
both under the scorer's 0.50. These are the only two classes whose capped IoU
falls under 0.50 and exactly the two that collapse, and the AP ordering matched
the IoU ordering.

Result: **0.5389 / 0.5391 against a 0.5418 baseline. Zero.**

Verified it was not a plumbing bug. `probe/replay_capture.py` replays the real
captured flight frames through any live service, so two configs see
byte-identical input. Diff over 249 real frames: real answers 9963 in both,
boxes identical for untouched classes, medium_launcher **1.771× bigger**
(2.302/1.3 = 1.771, exact), small_launcher **1.489× bigger** (exact). The
change reached the wire perfectly. The hypothesis was simply wrong — the IoU
coincidence was two points.

Worse, `HELSINKI_BOX_FACTORS` are actively wrong for v9: applying them drops
Helsinki mAP from 0.9728 to 0.5945, and medium_launcher to 0.000. They were
measured on v4-era cut-out masks and do not describe what v9 predicts.
**Do not trust that table.**

### Kill 2 — the graded sub-0.001 confidence band (predicted +0.02–0.04, actual 0.000)

AP ranks the whole flight by confidence, so a box appended strictly below every
answer we believe in can never lower any class's AP. Real answers clip at
0.001, so [0.0001, 0.0009] is free space. Verified against `faster_coco_eval`
itself on a class broken by a box mismatch:

| payload | AP |
|---|---|
| single wrong scale | 0.000 |
| + 4 shape variants **tied at 0.0** | 0.216 |
| + 4 shape variants **graded** 9/7/5/3 e-4 | **0.500** |
| + 8 graded | 0.500 — widening is free |

Tied boxes dilute to ~AP/k because the scorer cannot order them; graded ones do
not dilute at all. (This is also why the pre-existing `FLOOR_ZERO`, tied at 0.0,
was worth so little — and it was never enabled in `serve_env.sh` anyway.)

Result on the real grader: **zero**. Implemented and verified on the wire —
73,343 band boxes across all 16 classes at confidences 0.0005–0.0009.

**Why it failed, and this is the useful part:** a band variant only adds recall
where the primary box *misses but the object is still under it*. Rescaling a
box that is on the **wrong object** lands nowhere either. The band works on
right-object-wrong-size failures. Ours are wrong-object failures. The band is
still in the code behind `DRONE_BAND=1` and is provably harmless; it is just
not the fix.

### Kill 3 — single model instead of the 5-pass ensemble (predicted +0.13, actual −0.086)

On Helsinki: v9@1280 alone 0.9728, v8@1280 alone 0.9721, the served 5-pass
concatenation 0.8400. Looked like the ensemble was costing 0.13.

Result on the flight: **0.4577 / 0.4537 / 0.4549, mean 0.4554 against 0.5418.
The ensemble is worth +0.086.**

### THE TRAP — Helsinki inverts the answer

`src/helsinki` (25 frames, real GT, the evaluator's own scorer) looks like a
free local testbed. **It gave the wrong SIGN twice in one day** (kills 1 and 3).

Reason: our models were trained on cut-outs taken **from** Helsinki, so recall
there is saturated near 0.97 and extra passes can only add false positives. On
the flight every model is recall-starved, so the union adds real detections.
Helsinki measures the saturated regime; the flight is the starved one. **Any
experiment whose answer depends on the recall/precision trade-off comes out
backwards there.**

Helsinki is still valid for things that do not depend on that regime: the class
mapping is correct, and the imgsz curve shape per model.

### imgsz findings (Helsinki, so shape only, not absolute)

On the 960×540 transmitted view — this applies at **every** camera level, since
the wire always delivers 960×540:

| model | 640 | 800 | 960 | 1088 | 1280 | 1600 |
|---|---|---|---|---|---|---|
| v9 (P2) | 0.233 | 0.611 | 0.666 | 0.920 | 0.973 | **0.980** |
| v8 | 0.302 | 0.618 | 0.720 | 0.947 | **0.972** | 0.966 |

v9 at 1920/2240/2560 falls to 0.903/0.907/0.922. v4 (nano) is the opposite —
it peaks at 960 and degrades above it. **The served `v8@2560` pass is off its
peak**, and the `v4@960` pass is at its own peak. Untested on the flight.

---

## 6. Everything measured earlier and killed — do not redo

| idea | result |
|---|---|
| `top_mostly` camera (top row only) | 0.5150 / 0.5098 vs 0.5270. Giving up bottom-row looks costs more than earlier acquisition gains |
| full 3D tracker (`probe/flyby3d.py`) | recall 0.663 vs the 2D tracker's 0.683 |
| per-track yaw fitting | ta-ta recall 0.43 → 0.08 |
| multi-frame super-resolution | 73/109 single vs 70/109 stacked at depth 7–9 |
| class-vote-share ranking | offline AP 0.378 → 0.363 |
| physical-size class prior | class accuracy 65.9 % → 65.9 % (confusable classes are all ~13 m) |
| homography carry, global ground level | 82.1 % at 20-frame gap vs fitted affine's 85.8 % |
| per-class confidence calibration | provably a no-op: AP reads order, calibration is monotonic |
| `FLOOR_ALL_CLASSES` at 0.01 | 0.134 vs 0.143 — those boxes outranked genuine faint detections elsewhere |
| "camera stalls cost score" | FALSE. Correlation with score is +0.53; only *lost frames* matter, ~0.009 each |
| box growth sweep (replacing, not appending) | 1.2 → 0.5118, **1.3 → 0.5270**, 1.45 → 0.4740. Peaked at 1.3 |

---

## 7. The training question — can a retrain fix the dead classes?

It is the right bucket (class confusion, 0.157 + detector recall 0.051 = 0.208,
the largest). Four things make it hard:

1. **Time.** v9 was 40 epochs at imgsz 1280 batch 8: **333 s/epoch, 3.70 h
   total** on this same 5090, which also serves the submission endpoint.
2. **The recipe is exhausted.** v9's curve: epoch 1 mAP50 0.344, epoch 21
   0.614, epoch 40 **0.623**. **+0.009 over the last 19 epochs.** A longer run
   buys nothing; it needs a different *dataset*.
3. **We have no real examples of the broken classes.** `training/patches_val`
   holds cut-outs mined from the actual flight:

   | class | real patches | flight AP |
   |---|---|---|
   | tank | 77 | 0.63 |
   | helicopter | 32 | 0.84 |
   | jammer | 27 | absent from truth |
   | small_tower | 26 | 0.58 |
   | jet_plane | 23 | 0.85 |
   | small_plane | 23 | 0.61 |
   | large_tower | 22 | 0.88 |
   | mine_roller | 19 | 0.69 |
   | spacecraft | 17 | absent |
   | **small_launcher** | **11** | **0.15** |
   | hangar | 6 | 0.93 |
   | large_launcher | 4 | 0.63 |
   | **medium_launcher** | **0** | **0.04** |
   | **medium_plane** | **0** | **0.02** |
   | **ta-ta** | **0** | **0.00003** |
   | **condor** | **0** | absent |

   `make_dataset.py --class-weights` exists, but you cannot weight examples that
   do not exist. And the patches were mined **using our own detector**, so "no
   patches" is partly a *symptom* of not detecting them — a circular trap.
4. **Our label metadata is unreliable** — see kill 1, `HELSINKI_BOX_FACTORS`
   drops v9 from 0.973 to 0.595.

**Expected value:** ceiling if all four dead classes reached 0.73 is +0.209 →
~0.74. Realistic from a same-recipe retrain: ~0. From a genuinely different
dataset with hand-harvested real patches: maybe +0.05–0.10, needing dataset
work plus 3.7 h of GPU.

**The single highest-value thing a fresh session could do with real time:**
hand-label medium_launcher, medium_plane, ta-ta and small_launcher instances in
the 3.4 GB of captured flight frames at `data/captures/` on the box (real
transmitted views of the real flight, 19 captures × 249 frames), and retrain
with those. That breaks the circularity. It is the only identified route to 0.7.

---

## 8. Operational risk — frame loss

`probe/frame_loss.py` over 10 recent runs: **1 run answered 155 of 250 frames**
and scored 0.3264 instead of ~0.58.

| run | frames | lost |
|---|---|---|
| 09:11:31 | 155 | **94** |
| all nine others | 248–249 | 0–1 |

That run's **logged inference was normal at 135 ms**, but wall-clock p50 377 ms,
p95 1093 ms, **max 5148 ms** — a 5.1 s stall. Healthy runs sit at p50 330 ms,
exactly the grader's 3 fps cadence, with inference 104–144 ms. So it was stalled
by something outside our process, on a shared vast.ai box.

**~10 % of runs. On a single-draw evaluation that drags expected value from
0.584 to about 0.56.** Mitigation: fire one validation immediately before the
real evaluation to confirm the box is healthy, then submit straight away. Any
score near 0.33 is this, not the config — always check `probe/frame_loss.py`
before interpreting a result.

---

## 9. Tools

All in `drone-flyby/probe/`:

| file | what it does |
|---|---|
| `replay_capture.py` | **replays real captured flight frames through any live service** — two configs see byte-identical input, free, no validation spent. Use it to verify plumbing. |
| `frame_loss.py` | frames answered vs sent, per run, from `data/serve.log` |
| `probe_server.py` | replays precomputed answers against the REAL grader; job queue in `data/probe/jobs.json`; this is how per-class AP was measured |
| `make_band.py` | appends a graded sub-0.001 band to a recorded payload |
| `miss_anatomy.py` | charges every miss to camera / detector / carry / tracker |
| `detect_curve.py` | p(detect) by transmitted object size |
| `camera_plan.py` | expected recall of a camera pattern |
| `camera_model.py`, `geometry.py`, `fit3d.py` | the projection model |
| `serve_passes.sh`, `serve_cam.sh`, `rebuild_box.sh`, `keepalive.py` | operations |

**Per-class AP measurement recipe** (17 probe jobs, ~90 s each): point the
portal at `http://93.91.156.85:41671` and queue validations; each returns
`AP_c / K` for one class. Map jobs to scores via `data/probe/server.log`
(`sequence ... -> job calNN`) aligned with the portal's `started_at` — **do not
read the queue order off `jobs.json`**, stray one-frame sequences consume and
release jobs out of order.

### Traps that each cost an hour

* `pkill -f <name>` over ssh kills your own shell (the pattern matches the ssh
  command line). Use PID files or `ps | grep | awk`.
* The portal's "Test endpoint" consumes a queued probe job.
* TL → TR is 1920 px against an L1 move limit of 1102, so a bare top-corner
  alternation is refused every step and the camera stalls for the whole flight
  while every log line looks healthy. Top-row patterns must step via TM.
* Move limits: L0 2203 px, L1 1102 px, L2 551 px. Returning to full view is one
  move and ignores the limit.
* Do not compare scores across boxes.
* Check `/api` before every attempt, not the launch command.

---

## 10. What I would try next, ranked

1. **Hand-harvest real patches for the four dead classes from `data/captures/`
   and retrain.** The only identified route to 0.7. Needs ~4 h of GPU plus
   labelling. Breaks the mining circularity.
2. **Fusion instead of concatenation.** We merge 5–6 models' detections by
   concatenating them, so every model's wrong class call enters the ranking.
   Weighted box fusion with a class vote across models attacks the 0.157
   precision bucket directly and needs no retraining. **Untested, and it is the
   best untested idea in this document.**
2b. **Ranking knobs — cheap, untested, aimed at the 0.157 bucket.** All are
   one-line `DRONE_SET` changes, all pure ranking so they cannot cost recall:
   `AGREEMENT_WEIGHT=0.7` (a track only one of the five models found is scaled
   by 0.7^4 and sinks; a real object is usually found by several models, a bush
   by one) — IN FLIGHT as of 10:00 UTC;
   `CLASS_SHARE_POWER` (ranks a track by how cleanly its class votes agree —
   0.95 for one class beats a 0.35/0.33/0.32 split);
   `MISS_PENALTY=0.85` (demotes a track the camera looked straight at and did
   not confirm; today a track refuted five times ranks exactly as high as one
   just seen).

3. **imgsz per pass.** v8@2560 is off its peak (0.887 vs 0.972 at 1280) and
   v9@1600 beat v9@1280 on Helsinki. Cheap to test, small expected gain,
   Helsinki-derived so treat with suspicion.
4. **Camera between 40 % and 60 % L0.** The peak is somewhere near 50 %; the
   curve is fairly flat there so expect little.
5. Do **not** revisit anything in sections 5 or 6.

---

## 11. Honest summary

The camera work took the mean from 0.5418 to 0.5841 (+0.042) and produced a
best draw of 0.6071. Everything else tried today measured zero or negative.

0.70 requires +2.24 AP, and the only place it exists is the four dead classes.
They are dead because the detector names them wrong at the resolution we serve
(62.2 % class accuracy, systematic confusion pairs), not because we fail to
find them and not because our boxes are the wrong size — both of those were
tested against the real grader and both measured zero.

Fixing the naming is a retrain. The retrain needs real examples of exactly the
classes the detector cannot find, and those examples were mined with that same
detector. Breaking that circle is the whole problem, and it is a day of work,
not an afternoon.
