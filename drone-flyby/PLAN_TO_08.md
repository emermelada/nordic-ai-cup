# The plan from 0.55 — what a step change would actually require

Written 2026-09-20, ~03:30 CEST, while Javier slept. Deadline 16:00 CEST today.
Everything numeric here was measured on this machine or read off a source that
is cited. Where I am guessing, it says so.

---

## 0. The honest ceiling, first

You asked for 0.8. Here is the arithmetic that governs whether that is
reachable, so you can decide rather than be told.

* Mean **recall** over the twelve scored classes, on the best configuration:
  **0.670**. That is the fraction of confirmed object-frames where we emit a
  correctly-classed box overlapping the object.
* Mean **AP**: about **0.55**.
* **Perfect ranking of exactly the detections we already make scores ~0.67.**

So 0.8 is not reachable by any re-ordering, re-weighting or re-thresholding of
the current detections. It requires detecting objects we currently miss. Every
serving-side knob has now been swept (box growth, track confidence, level
weight, detection floor, decay, miss penalty, runner-ups, camera, model count,
inference resolution), and the remaining ones are worth ~0.01 each.

**Where the missing score lives.** Four classes carry it, each worth 1/12 of the
macro average:

| class | recall | cells/object @2560 | why it fails |
|---|---|---|---|
| `small_launcher` | **0.00** | 2.1 | below the detection floor |
| `jammer` | **0.25** | 5.2 | **not resolution — the detector does not know it** |
| `spacecraft` | **0.50** | 4.1 | detector |
| `tank` | 0.59 | 5.7 | fires on bare grass; real ones rank below the phantoms |

Bringing those four to a *mediocre* 0.60 AP each is **+0.162**, landing at 0.70.
Nothing else needs to change. `jammer` at 5.2 cells and `helicopter` at 17 cells
prove the limit is not pixels for most of them — it is that **the training data
does not teach what these objects look like**.

**Therefore: the only lever with 0.8-shaped upside is the training data, and
there is exactly one documented, unfixed defect in it.**

---

## 1. The defect: our cut-outs are blurry

`HANDOVER.md` records this and nobody has fixed it:

> v7 was trained at 1280 on cut-outs that are mostly Level-1, i.e. 2x
> downsampled — **only 36 of 262 patches are native resolution.** It learned
> what blurry objects look like enlarged.

That is the stated cause of v7's failure, and v8 and v9 were trained on the same
patch set. **Every model we have ever served was taught the target classes from
half-resolution cut-outs.**

Measured tonight, the two candidate sources of native patches:

| source | native (L2) coverage |
|---|---|
| `data/scene` (the 4K mosaic) | **5.9 %** mean per frame; 0 frames above 20 % |
| `training/patches_val` | 36 of 262 |

So the scene reconstruction cannot supply them — it is 94 % Level-1 upsampled.
**Native patches require one recorded run with a survey pattern that covers the
whole frame at Level 2.** The existing `survey` pattern covers the top half
only, which is why the earlier harvest got 36.

### 1a. The pattern, verified legal tonight

Four rows instead of two, boustrophedon, walking back up the left edge so the
wrap is legal. Checked against the evaluator's own `describe_camera_rejection`:
**30 positions, zero illegal transitions.**

```python
# flyby.py, beside SURVEY
SURVEY4_ROWS = (270, 810, 1350, 1890)          # covers source y in [0, 2160]
SURVEY4_COLS = (480, 1030, 1580, 2130, 2680, 3230, 3360)   # covers x in [0, 3840]
SURVEY4 = [
    (480, 270), (1030, 270), (1580, 270), (2130, 270), (2680, 270), (3230, 270), (3360, 270),
    (3360, 810), (3230, 810), (2680, 810), (2130, 810), (1580, 810), (1030, 810), (480, 810),
    (480, 1350), (1030, 1350), (1580, 1350), (2130, 1350), (2680, 1350), (3230, 1350), (3360, 1350),
    (3360, 1890), (3230, 1890), (2680, 1890), (2130, 1890), (1580, 1890), (1030, 1890), (480, 1890),
    (480, 1350), (480, 810),                   # walk back: the wrap 1890->270 is 1620 px, illegal
]
```

Row spacing 540 px and column spacing 530 px are both inside the **551 px** L2
limit. Cycle length **30 frames** against an object transit of **~31 frames**, so
every object is caught at native resolution roughly once per pass, and a
249-frame run gives **8 passes**.

**The score of these runs is irrelevant** — they are data collection. Expect
~0.10. Do not report them as results.

---

## 2. The schedule, with go/no-go gates

Times assume a start at 08:00 and the 16:00 deadline. Every gate has a stop
rule; **if a gate fails, fall back to shipping what is already validated.**

| # | step | wall clock | gate |
|---|---|---|---|
| 0 | re-rent the box, both network gates, serve the validated config | 30 min | RTT <= 35 ms, three 100 MB pulls >= 20 MB/s |
| 1 | three runs on the validated config | 10 min | reproduces ~0.55; if not, the box is wrong, re-rent |
| 2 | add `SURVEY4`, serve it, **3 recorded runs** | 20 min | `/api` shows `camera: survey4`; recordings land |
| 3 | harvest native patches | 30 min | **>= 300 native patches, >= 15 per weak class**, else abort to §4 |
| 4 | retrain v10 | 3 h | loss curve sane by epoch 5 |
| 5 | serve v10 as a **6th pass**, three runs | 20 min | beats 0.55; else ship the current config |
| 6 | lock, one validation run, evaluate | 40 min | — |

That totals ~5.5 h of work in an 8 h window, leaving **2.5 h of slack** — which
you will need, because the evaluator dropped frames on roughly two runs in three
overnight.

**Hard stop: begin step 6 no later than 14:30 whatever is happening.** One
attempt, no recovery.

---

## 3. The retrain, precisely

```bash
MODEL=training/yolo11m-p2.yaml      # NOT yolo11-p2.yaml -- that silently builds NANO
PRETRAINED=models/drone-yolo11m-p2-v9.pt   # warm-start from v9, not from COCO
IMGSZ=1280 BATCH=6 EPOCHS=30 NAME=drone-yolo11m-p2-v10 \
WEIGHTS=jammer=2.2,spacecraft=2.0,small_launcher=1.8,tank=1.8,helicopter=1.4,mine_roller=1.3,small_plane=1.2,condor=1.0,ta-ta=1.0,medium_plane=1.0,medium_launcher=1.0,small_tower=0.8,large_launcher=0.8,large_tower=0.8,jet_plane=0.8,hangar=0.8 \
bash training/train_remote.sh
```

Four deliberate choices, each with a reason:

1. **Warm-start from v9, not COCO.** v9 already has the P2 head trained on this
   task; 30 epochs of fine-tuning on better patches is a far better use of three
   hours than 40 epochs from scratch. This is also the FixRes argument — a short
   fine-tune at the serving resolution is the cheapest thing in the literature.
2. **`jammer` weighted highest at 2.2.** It has 5.2 cells per object and 0.25
   recall: the only class where the detector clearly *can* see the object and
   still does not know it. It is the best evidence that data, not pixels, is the
   binding constraint — so it is the class that most tests the hypothesis.
3. **Weights floored at 0.8.** v8 cut `hangar` to 0.4 and its AP fell
   0.871 -> 0.505. Never go below 0.8 on a class that already scores.
4. **`BATCH=6`, not 8.** Batch 8 OOM'd last night and Ultralytics silently fell
   back to 4, doubling the epoch count. 6 fits.

**Serve v10 as a SIXTH pass, never as a replacement.** Five detector attempts,
one pattern: v5, v7, rotation TTA and v8-as-a-replacement all failed; v8, the
2560 pass and v9 all paid as *additions*. Latency budget allows it — the current
five-model stack runs at 82 ms of 333 ms.

---

## 4. If the harvest gate fails — the fallback queue

If step 3 yields fewer than 300 native patches, do not retrain. Spend the time
on these instead, in this order. None needs training.

| arm | change | expectation |
|---|---|---|
| `MAX_MISSES` 3 / 12 | env var | the last untouched v3-era knob; ~±0.01 |
| more L0 cadence | one line: 3 L0 per 9 frames instead of 2 per 8 | the L0 family has paid twice |
| `RUNNER_UP_SHARE` 0.10 | env var | untested downward; `RUNNER_UPS=1` failed but the share threshold is a different cut |
| `MATCH_IOU` 0.15 / 0.30 | env var | v3-era track-association threshold, never retuned |

Realistically these sum to **+0.01 to +0.02**. They are insurance, not a plan.

---

## 5. What I ruled out tonight, so you do not spend the morning on it

* **SAHI / tiled inference.** The VisDrone-winning recipe is ensemble +
  multi-scale + SAHI merged with WBF, and SAHI reports +6.8 AP inference-only
  (arXiv:2202.06934). But slicing a 960x540 view into 2x2 tiles and running each
  at 1280 gives **exactly the same pixels-per-object as running the whole view
  at 2560**, which we already do — and the 2560 pass is precisely what took
  `small_launcher` from 0.000 to 0.512. We have most of the SAHI gain already.
  Going further (tiles at 2560) was measured impossible: a fifth pass at 3200
  cost ~7x the 2560 one and blew the frame budget (514 ms, score 0.264).
* **WBF.** Already measured here at **+0.004** over plain concatenation.
* **Per-class confidence calibration.** Proved a mathematical no-op tonight —
  AP depends only on within-class ranking and Platt/isotonic are monotonic.
  Verified to six decimal places.
* **Any further suppression.** `AGREEMENT_WEIGHT` (-0.005), the 0.0 floor band
  (+0.0015), `MISS_PENALTY` (-0.013), `RUNNER_UPS=1` (-0.003). Four mechanisms,
  four failures, one cause: most apparent false positives are **real objects the
  truth file does not contain**.
* **Chasing the leaderboard number.** Validation attempts are unlimited and the
  flight is deterministic — 1181 repeated views are byte-identical across runs.
  A team that records enough runs can fit the validation flight specifically.
  That would not transfer to the one-shot evaluation on a different flight. Do
  not infer from 0.908 on validation that 0.908 is reachable on the evaluation.

---

## 6. My honest expectation

**If the harvest and retrain both work: 0.58–0.63.** That would be a real
result — roughly +0.08 in a day on a system that had plateaued for a week.

**0.8 is not reachable today.** It needs recall near 0.95 on twelve classes, and
we are at 0.670 with a detector that fires on bare grass for `tank` and does not
recognise `jammer` at all. That is several training cycles of work, not one
night. I would rather say this plainly at 03:30 than have you discover it at
15:00 with one attempt left.

**The thing that would most change the outlook is not in this plan**: a model
trained on genuinely in-domain data — the rendered assets themselves rather than
cut-outs pasted onto photographs. Everything we have ever trained learned these
objects from 262 cut-outs, 36 of them sharp. That is the ceiling we keep hitting.
