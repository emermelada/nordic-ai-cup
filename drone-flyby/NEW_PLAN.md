# Drone Flyby: the plan from 0.527

Written 2026-09-19, late evening, by a session that measured rather than tuned.
Everything numeric here was measured that day: offline on the Linux box, and on
the rented 5090 alongside the v9 training run.

**Repo state: `flyby.py` is UNMODIFIED. `BEST-WORKING-VERSION` is intact.
Nothing is committed.** Three new tools are untracked, listed in §7.

---

## 1. The one-paragraph diagnosis

The detector is not the bottleneck and has not been for some time. When an
object is inside the camera view we detect **and** correctly classify it **70
times out of 71**, at ~0.9 confidence, with boxes centred to 0.08 box sides.
What limits the score is (a) **coverage** -- a Level-1 view is exactly a quarter
of the frame, so an object is looked at ~10 of the ~33 frames it is alive, and
every genuine miss is a *carried* frame, never an in-view one -- and (b)
**ranking** -- we answer ~59 boxes per frame, average precision pools every
frame of a class before ranking, and our confidences are hand-built constants
that were never calibrated against each other. (a) is worth +0.01-0.03 and is
ready to run. (b) is where a large gain could plausibly live and **cannot be
worked on until the offline metric is fixed**, which is §4.

## 2. Ready to run: the `row0` camera

`row0` = `TL TM TR L0 BR BM BL L0` -- the full Level-1 sweep with a whole-frame
Level-0 look after each row. Measured on the served stack over the whole
249-frame flight, simulated closed-loop through the real serving code:

| camera | boxes sent | recall@0.5 | AP, old truth (32 obj) | AP, mined truth (66 obj) |
|---|---|---|---|---|
| **`row0`** | 16461 | **0.545** | **0.2251** | **0.3618** |
| `full` (served) | 12261 | 0.466 | 0.1981 | 0.3536 |
| `quad0` | 18479 | 0.494 | 0.1938 | 0.3457 |
| `l0only` | 17453 | 0.530 | 0.2397 | 0.3352 |
| `full0` | 17355 | 0.518 | 0.1783 | 0.3194 |

`row0` is the only camera that beats served on **all three** columns.

**Why this was available.** `full0`/`quad0` were measured with v3 and lost (full
0.130, quad0 0.126, full0 0.119), which closed the camera question. But they
exist to exploit Level 0, and **v4@960 detects 0 of 18 objects at Level 0**
while v6@1280 detects 12 of 18 at mean IoU 0.73. They were ruled out on a stack
blind at the level they depend on -- the third time this staleness has cost the
project (the 1600 pass lost as a *replacement* and won as an *addition*;
`NEW_TRACK_CONFIDENCE` was two generations stale and worth +0.013).

**Read the disagreement between the columns -- it is the point.** Recall and AP
rank these differently and the two truth files rank them differently again:
`full0` has the second-best recall and the worst AP. Recall ignores what extra
boxes cost; offline AP over-charges for them because the truth is incomplete,
and the L0 patterns send 34-50 % more boxes. The true value is bracketed, not
known. An earlier draft of this file recommended `full0` off recall alone; that
is **withdrawn**.

The one line it needs, in `SWEEPS` beside `full0`:

```python
    # The full sweep with a whole-frame look after each row. Level 0 is exempt
    # from the camera distance limit, so no middle stops are needed.
    'row0': [TL, TM, TR, (0, 1920, 1080), BR, BM, BL, (0, 1920, 1080)],
```

Latency is unchanged: an L0 pass is the same 960x540 input as any other.

**Expect +0.01 to +0.03.** Gate: three complete runs of `row0` against three of
`full`, same host, same session. If `row0` does not clear the baseline mean,
the whole L0 family is dead -- do not then try the other cadences.

## 3. v9: trained, measured in simulation, still needs a real run

v9 = `models/drone-yolo11m-p2-v9.pt` (pulled off the box before it was
destroyed; 20,558,096 params, 16 classes, verified by parameter count not file
size). A **P2 head** at stride 4 with class weights favouring the weak classes.
Final training metrics: **mAP50 0.623, mAP50-95 0.485, P 0.916, R 0.563**,
against v7's final 0.567 mAP50. `training/v9_run/` holds its `args.yaml` and
`results.csv`.

Simulated as a 5th pass at 1280, all four arms on the same flight:

| arm | boxes | offline AP | vs served |
|---|---|---|---|
| `full` (4 model, served) | 12261 | 0.3536 | - |
| `row0` (4 model) | 16461 | 0.3618 | **+0.0082** |
| `full` + v9@1280 | 13155 | 0.3559 | +0.0023 |
| `row0` + v9@1280 | 18110 | 0.3667 | **+0.0131** |

Deltas only. The simulator's absolute level sits ~0.10 below a real run, so the
calibration offset fitted on *recorded* runs does not transfer to *simulated*
ones -- compare arms with each other, never to 0.527.

**Do not read +0.0023 as "v9 does not work."** The mined truth contains **no
`spacecraft`, `small_launcher`, `condor` or `medium_plane`** -- and v9's two
highest class weights are `spacecraft` 2.0 and `small_launcher` 1.5. Four of the
eleven classes it was trained to rescue are invisible to the metric measuring
it. The simulation under-rates v9 by construction and cannot be used to reject
it. `small_launcher` is the precedent: it was 0.000 in every configuration ever
measured until the 2560 pass took it to 0.512.

So v9 still earns a real validation run, on this reasoning rather than on the
simulated number. **Add it, never replace with it** -- v8 failed as a
replacement and paid as an addition, and so did imgsz 1600. A 5th pass at 1280
is ~8 ms on a 5090; a 5th at 3200 blew the 333 ms budget (514 ms, 0.264, 115
frames unanswered), and `tools/preflight.py` **cannot see that cliff** because
it replays sequentially against a warm service while the evaluator emits every
333 ms regardless. Preflight is a config check, not a load test.

**Two traps around the checkpoint**, both hit today:
* Ultralytics appends a suffix rather than reusing a run directory, so the real
  weights were in `drone-yolo11m-p2-v9-**2**/`, while the aborted nano run left
  a stale `best.pt` in `drone-yolo11m-p2-v9/`. The wrong one loads and serves
  perfectly cleanly.
* **Size-checking the file does not work after training completes.** Ultralytics
  strips the optimizer state at the end, so the real model went from 166 MB
  mid-run to 42 MB final, against the nano's 22 MB. Check the parameter count
  instead: 20,558,096 (medium) vs 2,668,800 (nano).
* And `training/yolo11-p2.yaml` silently builds a NANO P2 -- Ultralytics takes
  the scale from the filename stem. Use `training/yolo11m-p2.yaml` (`cf0483e`).

## 4. The real work: fix the metric, then fix the ranking

This is the only direction with a plausible +0.1, and it is a two-stage
dependency -- stage 2 is unmeasurable until stage 1 lands.

### 4a. Finish the truth file

`training/validation_objects.json` has 32 objects mined by v2 at conf >= 0.5
from Level-1 views. The scene holds far more, so **offline precision and AP are
measurement error while offline recall is sound** -- an unlabelled object
corrupts precision but cannot turn a real object into a miss. Every metric in
this document is recall for that reason, and the one time I reasoned from
offline AP I got the camera recommendation backwards.

`tools/mine_scene.py` (new) already improves this: detect at **native
resolution** (the scene is full 3840x2160, tiled 1920x1080 at imgsz 1920, so no
downsampling at all against a Level-1 view's 2x), then keep only chains of >= 6
consecutive sightings linked by the fitted ground motion. **Temporal
consistency is the precision filter, not confidence** -- a rendered asset sits
in ~33 consecutive frames drifting exactly with the ground; a spurious detection
does not. It does not share a failure mode with confidence thresholding or with
the cross-model agreement earlier mining rounds used.

First pass, conf 0.25 / min-frames 6: **66 objects against the old 32**,
including classes the old file never had (`ta-ta` x6, `medium_launcher` x3).
Calibration moved the right way:

| truth | offline AP on `b5544ad3` | on `89f751a2` | real score |
|---|---|---|---|
| old (32 obj) | 0.212 | 0.252 | 0.527 |
| mined (66 obj) | 0.369 | 0.382 | 0.527 |

The gap halved, from ~0.30 to ~0.15. **It is not finished.** Next steps, in
order:

1. Re-mine at a much lower floor -- `--conf 0.05 --min-frames 8`, and
   `--conf 0.10 --min-frames 6`. The chain length is what buys precision, so
   the confidence floor can go far lower than feels comfortable.
2. Add the other passes to the miner (it currently runs v8+v6 only), and v9.
3. **Check the miner against what we already know**: all 32 confirmed objects
   should appear in the mined set. If they do not, the miner has a recall bug
   and nothing downstream can be trusted.
4. Re-calibrate after each change against BOTH recorded runs.

**The gate is now PASSED, by a better test than the one I proposed.** Session
`drone-flyby-21` calibrated both truth files against **twelve** complete runs of
known real score spanning 0.4983-0.5339 (`tools/calibrate_truth.py`, `e0994a2`):

| truth | MAE | bias | Pearson | Spearman |
|---|---|---|---|---|
| mined (66 obj) | 0.0736 | -0.0736 | **+0.894** | **+0.811** |
| old (32 obj) | 0.0183 | -0.0167 | -0.039 | **-0.287** |

**Read the correlation, not the MAE.** The old file sits closer in absolute
terms and has *no* rank correlation over the band that matters -- it cannot tell
a 0.4983 run from a 0.5339 one, and is slightly inverted. The mined file is
offset by a near-constant -0.074 and **ranks correctly**, which is all a
configuration comparison needs. HANDOVER's earlier "+0.94 Spearman" was measured
across configs spanning 0.24-0.30, where almost anything ranks correctly.

So offline AP on the mined truth IS usable for comparing configurations, and
§2's `+0.008` for `row0` is signal rather than noise. Discard the `+0.027` from
the old truth entirely.

**But its class coverage is the binding limit**, and it bites §4b directly: the
mined truth has no `spacecraft`, `small_launcher`, `condor` or `medium_plane`.
Two of those are our weakest classes -- exactly where the near-zero-AP classes
of §4b live, and exactly what v9 was weighted to fix. **Mining those four
classes is therefore the highest-value single task on this list**, because
without them the metric is blind precisely where the remaining score is.

**What a mined truth still cannot settle:** its boxes are *our* boxes, so it is
blind to box growth in exactly the way `score_offline.py` already was (growing
answers and truth by the same factor cancels). Do not try to tune
`DRONE_BOX_GROW` against it. The real-run sweep stands: 1.2 -> 0.5118,
1.3 -> 0.5270, 1.45 -> -0.019.

### 4b. Then: the ranking problem

Average precision pools every frame of a class before ranking, so where our real
detections sit in the **global per-class order** is what caps each class -- and
the macro average makes every class worth 1/12 = 0.083 whatever its volume.
Measured on run `b5544ad3`, per class, confidence of our confirmed-correct
boxes against how many boxes of that class outrank them:

| class | boxes sent | confirmed hits | median hit conf | boxes outranking them |
|---|---|---|---|---|
| **tank** | **1598** | **7** | **0.224** | **357** |
| hangar | 212 | 16 | 0.573 | 51 |
| jet_plane | 963 | 11 | 0.855 | 20 |
| large_tower | 1303 | 57 | 0.770 | 48 |
| mine_roller | 1085 | 19 | 0.775 | 47 |
| small_plane | 1314 | 2 | 0.835 | 56 |
| small_tower | 369 | 14 | 0.801 | 30 |
| large_launcher | 855 | 5 | 0.839 | 4 |

`tank` is a different animal from everything else: its real detections sit at
0.224 while 357 tank boxes outrank them. Precision at tank's recall points is
therefore ~7/364, and tank's AP is close to zero -- **0.083 of the macro average
on its own.** Four classes look at or near zero this way.

Two things are already ruled out as the cause, both measured:

* **It is not runner-up flooding.** Capping runner-up confidence
  (`RUNNER_UP_CEIL`) changed the outranking counts by exactly nothing, so the
  boxes above the real tanks are **top-1 answers of other tracks**, not
  runner-ups.
* **It is not one bad voter.** Per-pass top-1 class accuracy on a box that is
  really on the object is v4 0.89, v6 0.84, v8@1280 0.95, v8@2560 0.98, and
  re-weighting the votes by pass moved offline recall 0.834 -> 0.834. Zeroing
  the weak passes made it *worse* (0.704). The confusion is consistent across
  passes -- `tank -> mine_roller` appears in three of the four -- so it is in the
  training data, not the voting.

So the hypotheses worth testing once the metric works, cheapest first:

1. **Confidence calibration across classes.** `base = best_confidence *
   min(1, 0.7 + 0.1*hits) * 0.97^unseen` is a hand-built formula, identical for
   every class, never fitted. A per-class calibration (Platt/isotonic on the
   mined truth) is post-processing only -- no retraining, no latency.
2. **`tank` specifically.** Find what the 357 higher-ranked tank boxes actually
   are, by eye, out of `data/scene`. If they are real unlabelled tanks the
   problem is the truth file; if they are `mine_roller`/`large_tower` confusions
   promoted to top-1, it is a training-data problem for v10.
3. **`UNSEEN_DECAY`.** 0.97 was swept against v3 at score 0.132 and is two
   generations stale -- the same staleness that made `NEW_TRACK_CONFIDENCE`
   worth +0.013 when retuned. It governs the confidence of the ~72 % of
   object-frames that are carried, and carried boxes are measurably nearly as
   good as fresh ones (94 hit / 50 near / 36 nothing). Decaying them may be
   throwing away rank for no precision gain.

## 5. Experiments in order of information per unit of effort

| # | experiment | cost | what it decides |
|---|---|---|---|
| 1 | `row0` vs `full`, 3 complete runs each | ~10 min | the only ready gain; +0.01-0.03 expected |
| 2 | v9 appended to the stack, in `camera_sim` | ~5 min GPU | whether v9 pays, before spending runs |
| 3 | v9 as a 5th pass, 3 complete runs | ~10 min | confirms #2 on the real metric |
| 4 | re-mine the truth, re-calibrate | ~30 min GPU | **unblocks everything in 4b** |
| 5 | per-class confidence calibration | ~1 h | the +0.1 candidate, if #4 passes its gate |
| 6 | `UNSEEN_DECAY` retune | 3 runs | stale knob governing 72 % of answers |

Stop rules: #1 fails -> the L0 family is dead, do not try other cadences.
#4 never reaches +/-0.02 -> **do not do #5**; offline AP will mislead you exactly
as it misled this session, and you should spend the remaining runs on #6 and on
v9 variants instead.

## 6. Do not re-derive these -- all measured, all negative

| idea | result |
|---|---|
| per-pass box growth | the 4 passes size boxes identically (1.050/1.038/1.061/1.078) |
| refitting the `MOTION` prior | offline recall 0.873 -> 0.873; the online fit already converges |
| box size as the real loss | ~1.3x artifact of the truth being tighter than the grader |
| per-pass class-vote weighting | recall 0.834 -> 0.834; zeroing weak passes hurts (0.704) |
| runner-up confidence ceiling | no change; the higher boxes are top-1, not runner-ups |
| box drift / per-track motion | carried boxes are centred to 0.08 box sides already |

Previously measured dead and still dead: Level 2 / hybrid camera, the 0.0 floor
band, growth cap > 1.3, `AGREEMENT_WEIGHT`, `MISS_PENALTY`, `HITS_BASE`,
`DRONE_DET_CONF=0.003`, a 5th pass at 3200.

## 7. The tools this session added (untracked)

| file | what it does |
|---|---|
| `tools/recall_replay.py` | replays a recording through `flyby.predict` from a cached detection set; **recall only**, with the miss split into hit / near miss / nothing, plus the per-class ranking table of §4b. `--carry` scores the frames between observations by interpolating each object's own trajectory (quadratic, 2.6 px RMS over 21-31 frame spans). |
| `tools/camera_sim.py` | simulates a whole 249-frame flight closed-loop through the served code, rendering views out of `data/scene`. `--camera NAME`, or `--pattern "0:1920,1080 1:960,540 ..."` which injects `flyby.SWEEP` **without editing `flyby.py`**. `--dump` writes the answers for AP scoring. |
| `tools/mine_scene.py` | §4a. Native-resolution detection over `data/scene` plus a temporal-consistency filter. |

`camera_sim.py` is deterministic and device-independent: `full` scores 0.466 on
both a CPU box and the 5090. Note the scene is mosaicked from Level-1 views
upsampled back to 4K, so a rendered L0 view has been through one extra
resample -- **L0 is under-rated by this simulator, not flattered.**

## 8. Practical notes for the new box

* Serve with `tools/arm.sh`; it refuses to hand over a service that is not
  serving what you asked for. Check `/api` (`models_loaded`, `imgsz`,
  `box_grow`) before **every** attempt, not the launch command.
* **249/249 or the run does not count.** Two missing frames cost 0.026, more
  than most changes gain.
* Runs are deterministic given the camera path, so a matched pair beats six
  unpaired runs. Three complete runs per arm; judge on the mean of complete
  runs, never the best draw.
* The endpoint matters more than any config: gate the host on latency **and**
  sustained bandwidth to the evaluator (Hetzner Helsinki). A bad host cost
  0.05-0.30 in one day, three times what a day of tuning gained.
* Long jobs on a rented box must be `nohup`-ed. An SSH drop killed a mining run
  in this session; it only survived because it had already written its output.
* `rsync` to a path whose parent does not exist fails -- `mkdir -p` first.

## 9. Honest expectation

`row0` +0.01-0.03, v9 unknown but plausibly +0.01-0.04 as an addition. That
lands around **0.55-0.60**, not 0.70. Nothing measured in this session supports
+0.17 from configuration, and six hypotheses died proving it. The path to 0.70,
if it exists, runs through §4: make offline AP trustworthy, then attack the
per-class ranking where four classes sit at or near zero and each is worth 0.083.
That is a real candidate for +0.1, and it is the one thing here that has not
been measured -- so treat it as a hypothesis, not a promise.
