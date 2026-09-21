# Prompt: fix the training inputs before v6

Paste everything below the line into a fresh Claude Code session started in
`~/Proyectos/nordic-ai-cup`. It is self-contained.

---

You are helping with **Drone Flyby**, one of three challenges in Nordic AI Cup 2026.
The deadline is **Sunday 20 September, 16:00 CET** (top-5 send code by 20:00). Work in
`~/Proyectos/nordic-ai-cup`, branch `drone-flyby-real-backgrounds`, service code in
`drone-flyby/`. Use the venv at `.venv` (`../.venv/bin/python` from inside
`drone-flyby/`). Shell is fish. Explain each step as you go — I am learning the tooling.

## The task

A drone films a synthetic 3D scene at 600 m. For every 4K frame we must detect **16
custom classes** (condor, hangar, helicopter, jammer, jet_plane, large/medium/small_launcher,
large/small_tower, medium/small_plane, mine_roller, spacecraft, ta-ta, tank) and steer a
camera. Answers are boxes over the whole 4K frame, normalised. Score is **COCO mAP@0.50,
macro over the classes present in the ground truth**. 3 fps, 3333 ms timeout, skipped
frames score 0.

We are on **0.143**. The leader is on **0.640**. Validation attempts are unlimited; there
is exactly **one** evaluation attempt, on a *different* flight.

## What was measured on 18 September — do not re-derive this

**The detector has memorised Helsinki.** v4 over the 25 labelled Helsinki frames against
real ground truth: median IoU **0.93**, 1% of boxes under the 0.5 threshold, **100%**
correct class, predicted/true box size 0.99. The same model on validation: **31%**
per-frame recall. We have one labelled instance per class; the model learned that
instance, not the class.

**These numbers were re-measured on 18 September and the old ones were wrong.** The
ground truth was being carried forward with `flyby.MOTION`, the same motion model the
tracker uses, so truth and tracks drifted together and the error cancelled.
`tools/score_offline.py` had the same bug and now fits the motion on the flight's own
objects (68.84 px/frame at the frame centre, against the Helsinki prior's 66.21).

**Coverage is not the constraint.** Per-frame recall 36%, and 47% when the object is
inside the requested view. Looking straight at an object we still miss it one time in two.
Scorer calibration is 0.286 against a real 0.1445.

**The volume is in the failing classes.** Frames present / hit rate / bad boxes / missed
/ AP:

| class | present | hit | bad boxes | missed | AP |
|---|---|---|---|---|---|
| `tank` | 219 | 0.11 | 27 | 167 | 0.021 |
| `helicopter` | 100 | 0.50 | 31 | 19 | 0.295 |
| `small_plane` | 99 | 0.31 | 39 | 29 | 0.071 |
| `jammer` | 99 | 0.37 | 24 | 38 | 0.228 |
| `small_tower` | 78 | 0.63 | 9 | 20 | 0.466 |
| `hangar` | 70 | 0.87 | 0 | 9 | 0.871 |
| `jet_plane` | 66 | 0.88 | 3 | 5 | 0.833 |
| `spacecraft` | 64 | 0.11 | 4 | 52 | 0.007 |
| `mine_roller` | 33 | 0.03 | 5 | 27 | 0.000 |
| `large_tower` | 33 | 0.06 | 3 | 28 | 0.001 |
| `small_launcher` | 33 | 0.00 | 3 | 30 | 0.000 |
| `large_launcher` | 17 | 0.65 | 0 | 5 | 0.644 |

Read this differently from the old table. `jammer` (0.228) and `helicopter` (0.295) are
**not** dead classes - keep them off any "weakest classes" list. `small_plane` is a
box-quality problem, not a firing problem (39 bad boxes against 31 hits), so pasting it
more often will not help. `tank` is worse than previously stated: 167 of 219 object-frames
missed outright.

**We answer with the wrong classes.** Most-emitted on the best run: `medium_launcher` 436
answers (**0** real objects), `large_tower` 398 (1), `medium_plane` 327 (**0**). These
outrank real detections and mAP is ranking-sensitive.

**Answer-policy tuning is exhausted.** Swept offline: `RUNNER_UPS=8`/`15` and
`MAX_TRACKS=200` change nothing (0.226, 13.1 boxes/frame); `FLOOR_ALL_CLASSES=0.02` fills
to 83.6 boxes/frame and scores 0.225. `RUNNER_UPS` is already saturated. Our misses are
objects with **no track at all**, not tracks wearing the wrong label. The detector is the
only lever left. **Do not spend time here.**

## What already exists — do not rebuild it

* `drone-flyby/data/scene/` — all 249 flight frames rebuilt as 4K images (98.8% mean
  coverage, 93.7% at Level 1+). Built by `tools/build_scene.py`. Takes ~10 min; it is done.
* `drone-flyby/data/backgrounds_real/` — 88 clean 4K backgrounds cut from the real flight,
  objects covered with terrain copied from the same frame. `training/backgrounds_real/`
  holds 44 of them as JPEG, which is what training reads. Built by
  `training/make_real_backgrounds.py` (~16 min, needs both models). It is done.
* `tools/score_offline.py` — replays a config through `flyby.predict` and scores it with
  the official COCO scorer. Minutes instead of a 15-minute validation attempt.
* `make_dataset.py --real-backgrounds/--real-share` and `train_remote.sh` are already
  wired for v6, with class weights set from the measured per-class AP and the Helsinki
  paste scale at 0.45–1.00 (see the paste-scale trap below).

## What I want you to do, before I train v6

### 1. Re-cut the three bad Helsinki masks (highest value)

`training/extract_patches.py` uses GrabCut, and for three classes it falls back to
something that keeps a lot of surrounding ground. Median alpha-mask fill per class:

| class | mask fill | what it costs us |
|---|---|---|
| `medium_launcher` | 0.86 | 436 answers against 0 real objects — our top hallucination |
| `large_tower` | 0.84 | AP 0.001 |
| `helicopter` | 0.82 | 31 bad boxes against a 0.50 hit rate — our worst box quality |
| every other class | 0.14–0.59 | fine |

A cut-out carrying grass teaches an oversized box *and* teaches "this ground texture is
that class". That is a precise match for the three symptoms above, so fix the masks, not
the class weights. Get these three down into the normal range. Check the result by eye —
render each patch over a flat colour and look at it. Note `mine_roller` has only **2**
Helsinki patches and `hangar` 3, so also check whether those two are usable at all
(`hangar` scores 0.825 on 3 patches, so few is not automatically fatal).

Verify: fill ratios for the three classes land in the 0.2–0.6 band and the alpha edges
follow the object outline.

### 2. Check the training views match how we actually serve

`make_dataset.py` has `VIEWS_PER_LEVEL = {0: 1, 1: 2, 2: 3}` — the dataset spends most of
its views on Level 2. In the real run the camera used Level 1 for **244 of 247** views.
Training is weighted at the resolution we barely use. Consider rebalancing toward Level 1,
but keep some Level 0 and 2 (`DRONE_INSPECT` is off, yet the camera is one experiment away
from using L2). Say what you changed and why.

### 3. Sanity-check the pasted result by eye

Build a small dataset and look at the preview before spending GPU money:

```
../.venv/bin/python training/make_dataset.py --scenes 6 --out /tmp/yolo_test \
  --real-backgrounds data/backgrounds_real --extra-patches training/patches_val \
  --helsinki-share 0.1 --real-share 0.8 --format jpg --preview
```

Check: objects sit at a believable size for 600 m, boxes are tight, no black rectangles
(frames with never-recorded gaps are supposed to be skipped), and the terrain is the real
Danish flight, not Helsinki or INRIA.

### 4. Re-harvest real cut-outs if it is cheap

`training/patches_val/` has no `condor`, `ta-ta`, `medium_plane` or `medium_launcher` —
those four classes have never been confirmed in the validation flight, so they ride
entirely on Helsinki cut-outs. The evaluation is a *different* flight and may well contain
them, so their patches must be good (see task 1). `tank` already has 66 real cut-outs;
`large_launcher` has 4 and `large_tower` 6, so more of those from `data/scene/` would help
if `harvest_validation_patches.py` can take them.

## Traps — all of these were already hit, do not repeat them

* **The offline scorer reads high and is self-referential on box geometry.** It says 0.226
  where the run really scored 0.1445, because the confirmed objects were mostly found by
  our own models. It scores `BOX_SCALE=0.8` at 0.198 against 0.226 — the real run collapsed
  from **0.143 to 0.017**. Use it to compare configurations and read the per-class column.
  Never quote it as the score, never tune box size with it.
* **Do not shrink the boxes we *report*.** `BOX_SCALE` stays at 1.0: shrinking it collapsed
  a real run from 0.143 to 0.017, and Helsinki says our box sizing is right in domain
  (ratio 0.99).
* **Do shrink the Helsinki *paste* scale.** This bullet previously said the opposite and was
  wrong, on two bad arguments: it read the `BOX_SCALE` collapse (a change at inference) as
  evidence about training paste size, and it read v5 losing to v4 by 0.003 — inside the
  ±0.01 noise band, and with architecture and paste scale changed together — as evidence
  about scale. `PLATEAU_IDEAS.md` was right. Every confirmed validation object measured
  0.45–0.98x its Helsinki counterpart by box diagonal (median 0.70), so
  `HELSINKI_SCALE_RANGE` is now (0.45, 1.00). `SCALE_RANGE` stays at (0.85, 1.15) — those
  cut-outs are harvested from the flight and are already at validation size.
* **Do not label objects by eye speculatively.** The flight crosses a dense industrial area
  full of cars, containers and farm machinery that mimic the target classes. A candidate I
  chased turned out to be a farm trailer. A wrong label poisons both the training set and
  the scorer. `tools/review_scene.py --sheets/--zoom/--add` is there if you want to try.
* **Do not cover low-confidence detections when making backgrounds.** The floor is 0.5 on
  purpose: the bushes, sheds and boats our models fire on are the hard negatives the set
  exists to teach. Covering them would delete exactly what we need.
* **Helsinki local scores mislead.** They have disagreed with validation repeatedly
  (4 runner-ups: Helsinki 0.835→0.778 while validation went 0.130→0.132). Trust validation
  runs and the per-class offline column, not the Helsinki number.
* **Never run heavy CPU jobs during a validation attempt** — it has cost ~20 answered frames.

## When you are done

Report what changed and what you verified, then stop. **Do not start training** — I launch
it myself on a rented GPU:

```
MODEL=yolo11m.pt EPOCHS=40 bash training/train_remote.sh
```

It reads `training/backgrounds_real` and prints how many it found; if that line says 0 the
backgrounds did not come through and it will silently train on stock photos like v1–v5 did.
Add the Vast SSH key **before** renting the box — account keys only reach new instances.
