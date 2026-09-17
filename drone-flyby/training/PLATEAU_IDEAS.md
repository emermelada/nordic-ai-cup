# Score plateau: status check + two new findings

For whoever (human or Claude) picks this up next. Written 2026-09-17 against
`main` @ `cd95222` (v3 serving, mining round 2, v4 training setup, full-camera
default, Poisson-blended pasting, `DRONE_SET`, 4 runner-up classes). Read this
before starting more work — it maps the 6 approaches already on record to what
the code now actually does, then adds two measured findings nobody has acted
on yet.

**Commits are landing live while this doc is being written** — three in the
time it took to read the repo once (`3c0a0db`, `06f02af`, `cd95222`), each a
real validation-run result. Run `git log --oneline -5` and `git pull` before
acting on anything numeric below; treat the specific constant values here as
"true as of `cd95222`," not as permanently current. Current best committed
score: **0.132** (v3, full camera, `NEW_TRACK_CONFIDENCE=0.25`,
`RUNNER_UPS=4`, `RUNNER_UP_SHARE=0.03`).

This is analysis only. Nothing here has been applied to `flyby.py` /
`make_dataset.py` / the notebook — every fix below is a proposal with an exact
location, meant for a human or a future session to implement and validate.

## 0. Status of the 6 approaches, rechecked against `main`

| # | Approach | Status |
|---|---|---|
| 1 | Better v4 training data (blended pasting, real cut-outs, class weights, less Helsinki) | **Done / ready.** `3c0a0db` shipped Poisson-blended pasting for the disc/rectangle-mask classes (helicopter, large_tower, medium_launcher). `train_kaggle.ipynb` already runs `make_dataset.py` with `--extra-patches training/patches_val --helsinki-share 0.1 --class-weights ...`. Not yet actually run on Kaggle (no `drone-yolo11n-v4.pt` in the repo). |
| 2 | Pick camera on real runs | **Done, settled.** `3c0a0db`: full 0.130, quad0 0.126, full0 0.119, dwell 0.117, top 0.108 (same v3 model, same flight). `full` is the default in both `flyby.py:131` and `docker-compose.yml`. Don't re-run this sweep — see §4 for the one variant not yet tried. |
| 3 | Confidence ranking for mAP | **Actively being swept right now, live, on real validation runs** (`cd95222`: `NEW_TRACK_CONFIDENCE` 0.10/0.25/0.40 → 0.126/0.132/0.122; `RUNNER_UPS`/`RUNNER_UP_SHARE` 0/2/4 → 0.125/0.130/0.132). 2 of the original 3 sub-ideas were already architectural (see §5); this is the "each change is one validation run" sweep itself, in progress. |
| 4 | Repeat mining after v4 | Open, unblocked by v4 finishing. See §7 for a sharper target. |
| 5 | Bigger ("s") model | Open. See §6 — the "too slow" call was never measured end-to-end. |
| 6 | Rejected ideas (full-frame real training, SAHI, hand-labelling) | Still agree. `make_real_views.py`'s own docstring independently reaches the same conclusion as your rejection #1. No new evidence against #2/#3. |

## 1. New finding: object scale is miscalibrated (biggest unaddressed lever)

Measured directly: geometric-mean box side (`sqrt(w*h)`) per class, Helsinki
annotations vs the 24 mined real objects in `training/validation_objects.json`.

| class | Helsinki px | real px | ratio | n (real objects) |
|---|---:|---:|---:|---:|
| large_launcher | 126.0 | 52.4 | **0.42** | 1 |
| spacecraft | 45.7 | 26.3 | **0.58** | 2 |
| jet_plane | 77.9 | 48.9 | **0.63** | 2 |
| large_tower | 61.9 | 39.4 | **0.64** | 1 |
| helicopter | 103.0 | 69.4 | **0.67** | 3 |
| mine_roller | 56.4 | 37.8 | **0.67** | 1 |
| small_tower | 56.9 | 38.6 | **0.68** | 3 |
| tank | 47.9 | 38.4 | **0.80** | 5 |
| small_plane | 46.1 | 39.8 | **0.86** | 1 |
| hangar | 124.2 | 135.2 | 1.09 | 2 |
| jammer | 37.2 | 42.0 | 1.13 | 3 |
| condor, medium_launcher, medium_plane, small_launcher, ta-ta | — | — | no real data | 0 |

**9 of 11 classes with real data are smaller in reality than in Helsinki**,
mostly 0.6-0.85x, two much smaller (large_launcher 0.42x, spacecraft 0.58x).
Only hangar/jammer run slightly larger. This reads like the real flight is at
a higher altitude or narrower FOV than Helsinki's fixed 600 m
(`src/helsinki/run_metadata.json`), not noise.

**Why this matters:** `training/make_dataset.py:53` — `SCALE_RANGE = (0.85,
1.15)` — is applied to every pasted patch regardless of source
(`training/make_dataset.py:167`, inside `transform_patch`). For large_launcher,
the *smallest* size the model has ever been trained on (0.85x) is still larger
than the real object's *average* size (0.42x). For helicopter/large_tower/
mine_roller/small_tower, real objects sit entirely below the trained range
too. This alone is a plausible independent cause of "even at full resolution
v3 misses obvious helicopters and towers" — separate from the grass-disc bug
`3c0a0db` already fixed.

**Caveat (report honestly, don't hide):** mined objects are ones the *current*
detector already found at conf ≥ 0.4-0.5 (`training/mine_validation.py`
`--min-conf`). If wrong-scale objects are exactly the ones the detector
misses, this table under-samples the worst cases — the true gap is likely
**at least** this large, not smaller. This may also explain why condor/
ta-ta/small_launcher/medium_plane/medium_launcher have zero mined examples:
not necessarily absent from the flight, possibly just never crossed the
mining confidence floor because they're pasted at the wrong scale in
whatever model did the mining.

**Proposed fix**, minimal and general (reflects a real camera/altitude
parameter, not an overfit to this flight's exact boxes):

```python
# training/make_dataset.py, near SCALE_RANGE (line 53)
REAL_SCALE_PRIOR = {   # ratio real/Helsinki px, from validation_objects.json; 1.0 = no data yet
    'large_launcher': 0.45, 'spacecraft': 0.6, 'jet_plane': 0.65, 'large_tower': 0.65,
    'helicopter': 0.7, 'mine_roller': 0.7, 'small_tower': 0.7, 'tank': 0.8, 'small_plane': 0.85,
}
# in transform_patch (line 159) or at the call site in build_scene (line 264):
# scale = REAL_SCALE_PRIOR.get(name, 1.0) * rng.uniform(*SCALE_RANGE)
```
Numbers rounded/padded slightly wide of the measured ratio on purpose (small
`n` per class). Classes with zero real data keep `1.0` — a second-order idea
is to default them to the *mean* ratio (~0.74) instead of 1.0, since the
pattern is consistent enough across 9/11 classes to be a better prior than
"assume Helsinki scale is right."

**How to validate before spending Kaggle time:** `tools/bench_recordings.py`
scores hit/wrong_class/bad_box/missing against the same 24 objects offline,
in seconds, no retrain needed for a first pass — run it against the *current*
v3 weights with `--per-class` to get a baseline, then compare after v4 trains
with the scale fix included. The real test is still a validation-service run
(v4 with vs without this change), but bench_recordings.py catches a broken
implementation for free first.

## 2. New finding: a global colour/lighting domain gap (measured, complements the Poisson-blend fix)

Mean HSV (OpenCV 0-179 hue) and saturation, Helsinki 4K frames (n=25) vs the
248-frame real recording `data/recordings/2c09583d65c34d7db47b32c66c375b02/`:

| | Hue | Saturation | Value |
|---|---:|---:|---:|
| Helsinki | 76.2 (green) | 134.9 | 88.2 |
| Real recording | 43.9 (yellow/tan) | 106.6 | 123.8 |

A ~64° hue shift and a visibly flatter, brighter look (sand/concrete/dry
terrain vs Helsinki's lush forest — see `data/recordings/.../0002_f0003.png`
vs `src/helsinki/images/frame_000001.png`).

**This is a different problem from what `3c0a0db` fixed.** Poisson/mixed-
cloning (`paste_blended` in `make_dataset.py`) harmonises a patch against the
*local* ground it's pasted onto — it doesn't change the *global* colour
distribution of the training set. `load_background`'s own jitter
(`make_dataset.py`, `± brightness/contrast only, rng.uniform(0.85,1.15)`) and
Ultralytics' default `hsv_h=0.015` (≈ ±5°) never rotate hue far enough to
reach a 64° shift. The model has essentially never trained on yellow/tan-cast
scenes.

**Proposed fix, cheapest first:**
1. Widen hue jitter explicitly, on **patches** (where Helsinki's cast is
   baked in) in `transform_patch`, and on whole **backgrounds** in
   `load_background` — both currently do brightness/contrast/greyscale-mix
   only, never hue rotation. A ±30-40° hue rotation (cv2: convert to HSV,
   add to H channel mod 180, back to BGR) covers the measured gap with
   margin.
2. Lower-priority alternative: normalize both sides instead of augmenting —
   apply the same gray-world/LAB white-balance function to rendered training
   views (`render_view` in `make_dataset.py`) and to the live input
   (`detect`/`raw_detections` in `flyby.py`). More surgical, but two call
   sites must always stay in lockstep or you introduce a *new* train/inference
   mismatch. Prefer #1 unless #1 measurably fails.

**Caveat:** one recording, one flight — same overfitting risk the user
already flagged. The fix (wider hue range) is deliberately a *generalization*
of the augmentation, not a fit to this exact hue value, for that reason.

## 3. Preprocessing — direct answers to "does preprocessing move the score"

Ranked by expected value / cost:

1. **Scale correction (§1).** Not "preprocessing" in the image-filter sense,
   but it's data preprocessing in the pipeline sense and is the strongest
   lever found. Do this first.
2. **Hue/colour augmentation widening (§2).** Cheap, one function, both call
   sites already exist (`transform_patch`, `load_background`).
3. **Patch QA pass.** Look at `data/patches/_sheet.png` and
   `data/patches_val/_sheet.png` by eye before the v4 run — `extract_patches.py`
   already has a `MIN_FOREGROUND` / ellipse-fallback safety net, but a bad
   GrabCut mask matters more now that some classes have single-digit patch
   counts. Cheap, no code change, just a look.
4. **Inference-time CLAHE / gray-world normalization.** Only worth it if
   applied identically in `render_view` (training) and `detect` (serving) —
   otherwise it's a new domain gap, not a fix for one. Lower priority than
   #1/#2 because fixing the *data* to already span the real distribution is
   more robust than normalizing both sides at runtime and hoping they land
   in the same place. Try only if #2 under-delivers.
5. **Motion blur.** Unverified — don't invest without evidence. A same-
   resolution Laplacian-variance comparison (not the resolution-confounded
   one already tried) would need to show real frames are measurably blurrier
   than same-scale Helsinki crops before this is worth augmenting for.

**Checked and ruled out — don't redo:**
- **Fixed lens/gimbal artifact.** A dark shape at the bottom of
  `0126_f0127.png` looked like it might be hardware vignetting. Scanned mean
  brightness in a 6x8 grid across all 248 recorded frames: no cell is
  systematically dark on average (range 94-109 throughout). It was a real
  dark object/shadow in that one frame, not a sensor artifact.
- **BGR/RGB channel mismatch.** `decode_image`/`encode_image` in `utils.py`
  and the training pipeline both use OpenCV BGR consistently end to end.
  Verified by reading, not assumed.
- **JPEG-artifact domain gap.** Transmission is lossless PNG on both the
  evaluator and `render_view`/`local_evaluator.py`. Not applicable.

## 4. Camera: settled, one untested variant if you want it

`full` beats 4 other patterns by a real margin (§0 row 2). One combination
wasn't tried: **two full passes before any bias**, e.g.
`FULL_SWEEP * 2 + TOP_SWEEP * 1000`. Rationale: `annotations_for`'s hit-count
confidence boost (`flyby.py:423`, `min(1.0, 0.7 + 0.1 * hits)`) caps at
`hits == 3`; a single full pass gives most objects only 1 hit. A second full
pass pushes more tracks toward that cap before the sweep starts favoring
re-entry points. Low priority — the existing 5-way sweep already found a
clear winner with a decent margin over the runner-up (0.130 vs 0.126), so the
expected gain here is small. Don't spend a validation run on this before
§1/§2.

## 5. Confidence ranking (approach #3) — what's actually left

Your three original ideas, checked against the current tracker:

| Idea | Status |
|---|---|
| Rank objects seen several times higher | **Done.** `flyby.py:423`, hit-count boost, caps at 3 hits. |
| Lower confidence of level-0-only sightings | **Done.** `LEVEL_WEIGHT = {0: 0.4, 1: 0.8, 2: 1.0}` (`flyby.py:82`) feeds `best_confidence = max(..., confidence * (0.6 + 0.4*weight))` (`flyby.py:382`) — an L0-only sighting tops out at 0.76x what an L2 sighting gets. |
| Tune how fast forgotten objects fade | **Still the one open numeric knob.** `UNSEEN_DECAY = 0.97` (`flyby.py:87`) hasn't appeared in any commit message's sweep results yet, unlike `NEW_TRACK_CONFIDENCE` and `RUNNER_UPS`/`RUNNER_UP_SHARE`, both now swept and committed (`cd95222`). |

As of `cd95222`, `NEW_TRACK_CONFIDENCE` (0.10/0.25/0.40 → 0.126/0.132/0.122,
inverted-U, 0.25 confirmed near-optimal) and `RUNNER_UPS`/`RUNNER_UP_SHARE`
(0/2/4 runner-ups → 0.125/0.130/0.132, monotonic so far — 4 might not be the
ceiling, worth one more step e.g. 6 runner-ups at a lower share) are both
already live-tuned. `UNSEEN_DECAY` is the one left untouched.

`DRONE_SET` (new in `3c0a0db`) makes the decay sweep a one-line, no-redeploy
experiment:
```bash
DRONE_SET=UNSEEN_DECAY=0.95 docker compose up -d drone-flyby   # faster fade
DRONE_SET=UNSEEN_DECAY=0.99 docker compose up -d drone-flyby   # slower fade
```
**Gotcha to know before reaching for it elsewhere:** `DRONE_SET`'s parser
(`flyby.py:152-157`) only accepts globals that are already `int`/`float`.
`LEVEL_WEIGHT` is a dict — `DRONE_SET=LEVEL_WEIGHT=...` will hit
`raise SystemExit('unknown numeric setting')`. If you want to experiment with
per-level trust live, it needs a code change first (e.g. split into
`LEVEL_WEIGHT_0/1/2` numeric constants), not just an env var.

## 6. Bigger model — measure before ruling out

"Too slow" was reasoned from inference time alone (nano ~90 ms CPU vs "s" at
~2.5x ≈ 225 ms) against a 333 ms frame interval — that math alone doesn't
rule "s" out; it comes down to whatever else eats the remaining ~240 ms
(decode, camera logic, network round-trip). That total is measurable right
now, no GPU or new model needed:
```bash
python local_evaluator.py --realtime --scene helsinki
# read "round trip ms  mean / median / max" in the report
```
If real headroom looks generous, a cheaper middle ground than swapping the
whole service to "s": run "s" **only** on `DRONE_INSPECT` zoom-in views
(`flyby.py`'s `inspection_target`, currently opt-in and off by default) —
those are already an extra look outside the main frame clock's cadence, so a
slower model there risks less, and lands exactly on "misses obvious objects
even at close range," which is the failure mode you actually described.

## 7. Mining round 2 — sharper target

Five classes still have zero real examples: condor, ta-ta, small_launcher,
medium_plane, medium_launcher. Given §1's caveat (mining is biased toward
objects the current detector already finds), consider re-running
`training/mine_validation.py` with a lower `--min-conf` specifically to
surface these five, rather than repeating the default sweep — the default
threshold may be exactly why they haven't turned up yet.

## 8. Suggested order

The live ranking sweep (§5) has already gone 0.125→0.130→0.132 — real but
shrinking steps, and it can't fix a miss the detector never produced in the
first place. §1/§2 attack the detector itself, which is still the declared
bottleneck; prioritize them over further ranking sweeps.

1. Patch QA pass (§3.3) — minutes, free.
2. Implement the scale prior (§1) and hue widening (§2) in `make_dataset.py`.
3. Run v4 training (`train_kaggle.ipynb`, already wired) with both included.
4. `tools/bench_recordings.py --per-class` against the new weights — offline,
   free, sanity-checks before spending a validation attempt.
5. One validation run: v4 (scale + hue fix) vs current v3, same `full` camera.
6. If detection is still clearly the limit after that: §6 (measure headroom,
   consider targeted "s" on inspect-only).
7. `UNSEEN_DECAY` sweep (§5) — cheap, can run in parallel with the above since
   it doesn't touch the model.
8. Mining round 2 (§7) once v4's actual weak spots are visible.
