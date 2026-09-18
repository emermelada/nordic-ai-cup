# Drone Flyby — where the work stands (2026-09-18, evening)

Everything needed to continue on another machine is in this repo, including the
served weights (`models/`) and the cut-outs mined from the validation flight
(`training/patches_val/`). The recorded validation runs themselves (~5 GB) are
not: record your own with `DRONE_RECORD_DIR` (see below).

## Scores so far (cases.nordicaicup.com, validation)

| Model | Validation | What changed |
|---|---|---|
| v1 | 0.011–0.035 | objects pasted only on the 25 Helsinki frames |
| v2 | 0.097 | pasted on ~400 real aerial photos (Inria, LandCover.ai) |
| v3 | 0.132, 0.119 | + cut-outs of real validation objects, light/shadow variation |
| v4 | 0.1445, 0.1425 | + Poisson-blended pasting, more objects, weak classes weighted |
| v6 | not run | real flight backgrounds; offline a wash alone, see 18 Sep evening |
| **v4+v6 alternating (served)** | **0.2365** | the two fail on opposite classes; best score so far, +0.09 on v4 |

Best Danish team 0.36, best overall 0.46 (as of Thursday evening).

**Run-to-run noise is about ±0.01** — the same configuration scored 0.132 and
0.119. Trust a difference only if it is bigger than that or repeats. Heavy CPU
work on the serving machine during an attempt costs answered frames (225 vs
246), which alone moves the score: keep the machine idle during a run.

## How the service works

`flyby.py` does three things per request, and its header lists every setting:

1. **Detect** — YOLO on the 960x540 view, with our own letterbox and NMS so
   every class score survives (`raw_detections`); boxes are lifted into
   source-frame coordinates.
2. **Remember** — every object becomes a track, carried between frames by the
   fitted ground motion (`MOTION`, accurate to ~1 px/frame, verified on the
   real flight). Answers cover the whole frame, not just the current view.
3. **Steer** — a fixed camera pattern (`SWEEPS`). Moves are planned from the
   *last requested* view, because the evaluator renders a frame before our
   previous answer lands.

### Settings that were tested on real validation runs

| Setting | Values → score | Kept |
|---|---|---|
| Camera | top 0.108 · dwell 0.117 · full0 0.119 · quad0 0.126 · **full 0.130** | `full` |
| Runner-up classes | 0 → 0.125 · 2 → 0.130 · **4 (share 0.03) → 0.132** | 4 |
| `NEW_TRACK_CONFIDENCE` | 0.10 → 0.126 · **0.25 → 0.132** · 0.40 → 0.122 | 0.25 |
| `UNSEEN_DECAY` | 0.90 → 0.128 · **0.97 → 0.132** · 0.995 → 0.127 | 0.97 |
| `MAX_MISSES` | 3 → 0.119 · **6 → 0.132** · 10 → 0.114 | 6 |

Most of those gaps are inside the noise; the camera choice is the one clear
result. `DRONE_CAMERA` and `DRONE_SET=NAME=value,...` change any of it without
a rebuild (see `docker-compose.yml`).

## Running it

**Linux with Docker** (what has been used so far, service on host port 8002):

```bash
docker compose up -d --build drone-flyby      # weights mounted from ~/models
cloudflared tunnel --url http://localhost:8002
```

**macOS with the Apple GPU — do not use Docker**, it cannot reach Metal.
This is the served configuration as of 18 Sep evening: v4 and v6 alternating
(see "v6 and the v4+v6 pair" below). Both weights are committed, so a pull is
enough:

```bash
cd drone-flyby
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt \
DRONE_DEVICE=mps .venv/bin/python api.py   # port 9053
```

**Check it is really serving the pair before anything else:**

```bash
curl -s localhost:9053/api
# "models_loaded": 2   <- the pair is alternating
# "models_loaded": 1   <- one model on every frame, whatever you asked for
```

`models_loaded` is the number that matters; `model_loaded: true` is also true
when only one of the two loaded. Asking for an alternate that is not there now
refuses to start rather than quietly serving one model (see the bug below).

`DRONE_DEVICE=mps` **has been run** (18 Sep evening) and needs no special
handling: `torchvision.ops.nms` did not raise, so `PYTORCH_ENABLE_MPS_FALLBACK=1`
is not needed. On the M4 the served pair answers a full round trip, PNG decode
included, in **25 ms median against the 333 ms budget** — 7 % of it. Detector
only, over 40 real frames: v4 9.2 ms, v6 14.8 ms, both on every frame 22.6 ms.

**Compute is not the constraint on the M4; the link is.** Two runs minutes
apart from the same process answered 68/249 and 248/249 frames. The quick
tunnel in `data/tunnel.log` dropped (`sendmsg: network is unreachable`) and
took ~5 s over three retries to re-register, and its own banner says
account-less tunnels have no uptime guarantee. Use a named tunnel for the
evaluation. The Linux i5 box is a different story — there the pair is genuinely
too slow (median 608 ms, 18 of 19 frames over budget) — but do not carry that
number over to the Mac, and do not read "we need a smaller model" from it.

Before an attempt, against the URL you will actually submit:

```bash
python tools/preflight.py --url https://<host>/predict
```

It checks `models_loaded`, replays real recorded frames, and reports the round
trip against the 333 ms budget, excluding the cold first request. Run it on the
public URL, not localhost, or it measures the wrong link.

`python local_evaluator.py --realtime --url http://localhost:9053/predict`
(needs `src/helsinki/`) still gives "round trip ms" and "frames skipped" — read
those, not the Helsinki score, which the detector has memorised.

The submitted URL is used verbatim, so it must end in `/predict`.

## Checking a model before spending a validation run

```bash
DRONE_RECORD_DIR=... # set in docker-compose.yml already: drone-flyby/data/recordings
python tools/bench_recordings.py --model models/drone-yolo11n-v4.pt --per-class
```

It replays recorded runs through the tracker and scores the objects listed in
`training/validation_objects.json` (29 objects, 12 classes, found by earlier
models and checked by eye). **It is biased**: it only knows objects some model
already found, so it under-rates a model that detects new things — v4 looked
slightly worse than v3 on it and was clearly better in the real run.

Other tools: `tools/simulate.py` (offline evaluator clock; its camera ranking
disagreed with reality, so do not trust it for camera decisions),
`tools/replay_recording.py` (draw a model's answers on recorded frames),
`training/mine_validation.py` (find new objects in recordings, review sheets,
`--accept id:class`).

## Training (Kaggle, GPU T4 x2, internet on)

`training/train_kaggle.ipynb` clones the official repo, rebuilds the synthetic
dataset and fine-tunes. Inputs: the code zip (built by zipping
`training/{extract_patches,make_dataset,harvest_validation_patches}.py`,
`validation_objects.json`, `patches_val/` and the starting weights), plus
`sagar100rathod/inria-aerial-image-labeling-dataset` and
`adrianboguszewski/landcoverai`.

The dataset (`training/make_dataset.py`) pastes cut-outs onto aerial photos and
cuts views at all three resolution levels. What matters most, in the order it
was learned:

* **backgrounds** (`--backgrounds`, `--helsinki-share 0.1`): v1 → v2, the
  biggest single jump;
* **real cut-outs from the flight** (`--extra-patches training/patches_val`):
  v2 → v3;
* **Poisson blending** for cut-outs whose mask is a disc or rectangle
  (helicopter, large_tower, medium_launcher were otherwise learned as discs of
  Helsinki grass): v3 → v4;
* **class weights** (`--class-weights`): v4 used up to 2.5x for weak classes and
  won on small_plane/large_tower but lost tank, jammer and spacecraft — use
  at most ~1.5x next time;
* **ready but never trained**: Helsinki cut-outs pasted at 0.55–1.15x (real
  objects measure 0.55–0.85x their Helsinki box diagonal) and hue rotation
  (backgrounds ±20°, patches ±8°, `hsv_h=0.03`).

## Training on a rented GPU (Vast.ai, Runpod, ...)

`training/train_remote.sh` does the whole run on a fresh box: installs what it
needs, checks the GPU, clones the official repo, downloads the background
photos from Kaggle, builds the dataset and trains.

```bash
export KAGGLE_API_TOKEN=KGAT_...                 # kaggle.com > Settings > API
bash training/train_remote.sh                    # yolo11m, 40 epochs
MODEL=yolo11l.pt EPOCHS=50 BATCH=24 bash training/train_remote.sh
```

Estimates from our Kaggle timings (T4, yolo11n, 40 epochs ~2.5 h): on an
RTX 5090 expect roughly **1 h for yolo11m** and **1.5-3 h for yolo11l**, plus
~20 min to build the dataset and the 22 GB Inria download (`WITH_INRIA=0`
skips it, at the cost of background variety — the single biggest gain so far).

Watch out for: Blackwell cards need torch >= 2.7 on CUDA 12.8 (the script
checks); `cache='ram'` needs ~1.6 GB per 1000 images; and destroy the instance
afterwards, stopping it still bills storage.

Copy the weights off the box (`scp -P <port> root@<host>:/workspace/<name>.pt .`),
drop them in `drone-flyby/models/`, and point the service at them with
`DRONE_MODEL`.

## Measured on the recordings, 2026-09-18

`torch` has Python 3.14 wheels now, so `tools/bench_recordings.py` runs on the
Windows laptop too and a experiment costs ~30 s instead of a validation run.
Numbers below are known-object hit rate; the per-class *mean* is quoted where it
matters, because mAP averages over classes and the frame-weighted total is
three-quarters tank, small_plane, helicopter and jammer.

* **The detector almost never picks the wrong class.** When v4 puts a box on a
  known object it names it correctly 100 % of the time for 8 of 11 classes
  (only small_plane/medium_plane and tank confuse at all). The whole gap is
  failure to fire, not confusion - so runner-ups, class weights at serving time
  and anything else that re-ranks classes is attacking a problem we do not have.
* **Ground motion, fitted online**: hit 32.6 % -> 37.7 %, bad boxes 16.3 % ->
  12.3 %. `MOTION` was fitted on Helsinki and is ~3 px/frame short here; it
  compounds, and past ~16 frames a carried box no longer overlaps at IoU 0.5.
  `DRONE_SET=MOTION_MIN_SAMPLES=999999` pins it back for an A/B.
* **Input size trades big objects for small ones.** Same v4 weights at
  imgsz 1280 instead of 960: tank 10 % -> 27 %, mine_roller 3 % -> 28 %, but
  hangar 81 % -> 49 %. No retraining involved.
* **Two models on alternate frames beat either alone**, because every detection
  lands in the same object memory (`DRONE_MODEL_ALT`). Per-class mean: v4 42.6 %,
  v5 37.9 %, v4+v5 alternating **46.2 %**, both-every-frame 46.4 % (twice the
  cost for +0.2). Three-way alternation is *worse* (40.0 %): each model gets too
  few frames to keep its tracks alive.
* `MAX_MISSES` saturates at 6 (6/12/20 identical) once the motion is fixed.
* Lowering `DRONE_DET_CONF` to 0.001 buys +1.2 points of recall offline, but it
  is the same shape as the `DRONE_FLOOR_ALL` result that lost 0.009 on
  validation - more faint boxes, and the bench cannot see what they cost in
  precision. Do not ship it without a run.

Still dead for the v4+v5 pair: spacecraft 5 %, small_launcher 6 % (13 source px,
7 px in a Level-1 view - below YOLO's finest stride, so resolution is the limit
rather than data), tank 25 %, small_plane 35 %. That list is what the v6 recipe
at the top of `training/train_remote.sh` is built around.

## What to do next, in order

1. **A bigger model.** The detector is the bottleneck: with a perfect detector
   this pipeline scores 0.84, and 97 % of frames are already answered. On an
   M4 with a fast link, `yolo11s` or `yolo11m` at `imgsz=960` fits the 333 ms
   budget easily. It needs a fresh training run (fine-tuning our nano weights
   into a bigger model is not possible); `yolo11l` would cost ~8–15 h of
   Kaggle GPU against a ~30 h weekly quota.
2. **v5 data**: softer class weights, plus the scale and hue changes above.
3. **More mining** with the new model (`training/mine_validation.py`), then
   re-harvest; condor, ta-ta, medium_plane and medium_launcher still have no
   real examples.
4. **Never tuned end to end**: `LEVEL_WEIGHT` (how much a Level-0 sighting is
   trusted) and `DRONE_INSPECT` (zoom to Level 2 on small uncertain objects,
   off because it lost in simulation — but the simulator was wrong about the
   camera too).

## Before the final evaluation

* One evaluation attempt only, on a *different* flight. Everything tuned on
  validation (camera, thresholds, validation cut-outs) may transfer less well.
* Serve from the machine with the best link, keep it awake and idle, and agree
  who is live — one URL per team.
* Verify, then one validation run on that exact setup, then the evaluation.

---

## 2026-09-18: where the points actually go

Measured today against the recorded runs and the labelled Helsinki scene.

**The detector has memorised Helsinki.** v4 over the 25 labelled Helsinki
frames, matched to real ground truth: median IoU **0.93**, 1 % of boxes below
the 0.5 threshold, **100 %** correct class. On validation the same model reaches
**36 %** per-frame recall on confirmed objects. Box regression and
classification are not broken; the backgrounds are the gap.

*(Every validation figure in this section was re-measured on 18 Sep after
adopting Franek's fix from `franek-drone-flyby-motion-fit`: the ground truth had
been carried with `flyby.MOTION`, the same motion the tracker uses, so both
drifted together and the error cancelled. Fitted on the flight's own objects the
ground moves 68.84 px/frame at the frame centre against the Helsinki prior's
66.21 — about 2.6 px/frame, which is 31 px after a 16-frame carry, more than
enough to lose a 50 px object at IoU 0.5. The old numbers were pessimistic by
5–11 points. The same trap was in `tools/score_offline.py`; it is fixed.)*

**Coverage is not the problem, the detector is.** Splitting the misses on the
best v4 run by whether the camera was even looking:

| | hit | miss | recall | share of object-frames |
|---|---|---|---|---|
| object inside the requested view | 130 | 145 | **47 %** | 30 % |
| object outside it (memory only) | 200 | 432 | 32 % | 70 % |

Even looking straight at a confirmed object we detect it 47 % of the time.

**Per-class, the volume is in the classes that fail.** Hit rate / bad boxes /
offline AP on the best v4 run:

| class | present | hit rate | bad box | missing | AP |
|---|---|---|---|---|---|
| tank | 219 | **0.11** | 27 | 167 | 0.021 |
| helicopter | 100 | 0.50 | 31 | 19 | 0.295 |
| small_plane | 99 | 0.31 | **39** | 29 | 0.071 |
| jammer | 99 | 0.37 | 24 | 38 | 0.228 |
| small_tower | 78 | 0.63 | 9 | 20 | 0.466 |
| hangar | 70 | 0.87 | 0 | 9 | 0.871 |
| jet_plane | 66 | 0.88 | 3 | 5 | 0.833 |
| spacecraft | 64 | 0.11 | 4 | 52 | 0.007 |
| mine_roller | 33 | 0.03 | 5 | 27 | 0.000 |
| large_tower | 33 | 0.06 | 3 | 28 | 0.001 |
| small_launcher | 33 | 0.00 | 3 | 30 | 0.000 |
| large_launcher | 17 | 0.65 | 0 | 5 | 0.644 |

`tank` is the most common object in the flight and we miss 167 of its 219
object-frames outright. `spacecraft`, `mine_roller`, `large_tower` and
`small_launcher` are dead. `small_plane` is different in kind: 39 bad boxes
against 31 hits, so it is a box-quality problem, not a firing problem, and
hard-background mining will not fix it.

Meanwhile our two most-emitted classes, `medium_launcher` (436 answers) and
`large_tower` (398), correspond to 0 and 1 real objects: high-confidence
hallucinations that outrank real detections, and mAP is ranking-sensitive.

**The unused answer budget turns out not to be worth anything.** We answer 13
boxes per frame against COCO's cap of 100, and detections ranked below the good
ones can only add recall --- but measured offline, it buys nothing:

| setting | offline mAP | boxes/frame |
|---|---|---|
| defaults | 0.226 | 13.1 |
| `RUNNER_UPS=8` / `=15` | 0.226 | 13.1 |
| `FLOOR_ALL_CLASSES=0.02` | 0.225 | 83.6 |
| `MAX_TRACKS=200` | 0.226 | 13.1 |

`RUNNER_UPS` is already saturated: a track rarely accumulates votes for more
than the four classes we already emit, so raising the limit adds nothing.
Filling to 84 boxes a frame with an all-class floor moves nothing either, which
matches the real run (0.134 against 0.143). The reason is the important part:
**our misses are objects with no track at all, not tracks wearing the wrong
class label.** No answer-policy setting can invent a detection. The detector is
the only lever left.

### What is new in the repo

* `tools/build_scene.py` --- the validation flight is deterministic (verified:
  1181 repeated views across runs are byte-identical), so the recorded views
  mosaic into complete 4K frames. All 249 frames rebuild at 98.8 % mean
  coverage, 93.7 % at Level 1 or better. Output in `data/scene/`.
* `tools/score_offline.py` --- scores a recorded run, or replays a config
  through `flyby.predict`, against the confirmed objects with the official COCO
  scorer. It carries the ground truth with a motion fitted on the flight's own
  objects (`--truth-motion prior` restores the old, drifting behaviour).
  Calibration: 0.286 where the real run scored 0.1445, so it reads high
  (the confirmed objects were mostly found by our own models) --- use it to
  compare configurations and to read the per-class column, not as the score.
  Replaying the defaults reproduces the recorded answers exactly, so the harness
  is faithful. **Do not use it to tune box geometry**: the confirmed boxes came
  from our own detections, so it is self-referential there --- it scores
  `BOX_SCALE=0.8` at 0.198 against 0.226, where the real run collapsed from
  0.143 to 0.017.
* `training/make_real_backgrounds.py` --- backgrounds cut from the flight
  itself, with known objects and confident detections covered by clean terrain
  copied from the same frame. The detection floor is deliberately high (0.5):
  our models fire constantly on bushes, sheds and boats, and those are the hard
  negatives this set exists to teach. Frames with a never-recorded gap are
  skipped --- a gap cannot be covered from the same frame and would leave a
  black rectangle for the model to learn as a feature. 88 clean 4K backgrounds
  in `data/backgrounds_real/`; 44 of them as JPEG in
  `training/backgrounds_real/` (91 MB), which is what `train_remote.sh` reads.
* `make_dataset.py --real-backgrounds ... --real-share` and `train_remote.sh`
  wired to use them. Helsinki paste scale back to 0.85-1.15 (v5's 0.55-1.15 came
  from a rotation-biased measurement and v5 lost to v4).
* `tools/review_scene.py` --- sheets, zooms and `--add` for labelling by eye.

### Labelling by eye: low yield, high false-alarm risk

Sampling every 20th rebuilt frame covers every object (they cross in ~33
frames). Five sheets in, the only candidate found turned out to be farm
machinery in the orthophoto, not a rendered asset. The flight crosses a dense
industrial area full of cars, containers and rooftop structures that mimic the
target classes. Do not add speculative labels --- a wrong one poisons both the
training set and the offline scorer.

### The online motion fit is right — and how the A/B nearly said otherwise

Franek's fit is on by default and it is correct. During a full replay of the best
v4 run it converges to **69.33 px/frame** at the frame centre, against the
Helsinki prior's 66.21. Two independent ground-truth estimates bracket it: 68.84
(fitted on all 29 confirmed objects) and 69.44 (Franek's, fitted on the three
static classes). The fit lands between them.

`tools/score_offline.py` first scored it at 0.277 with the fit on against 0.286
with it pinned off, which looks like a regression and is not one. Both scorers
carry the ground truth with *their own* fit, so the number partly measures
"does flyby agree with my truth" rather than "is flyby right". Our truth is
0.49 px/frame below where flyby converges, so a better motion scores slightly
worse against it. Do not read a sub-0.01 difference here as a verdict on motion.

Franek excludes aircraft from his truth fit in case they fly. Measured, they do
not: median dy excess over the Helsinki prior is 1.86 px/frame for the aircraft
classes against 2.12 for his static references and 2.11 for the other ground
objects, a spread well inside the per-class scatter. Every object in this flight
is a static prop, so fitting on all 29 is the better-sampled choice (156 point
samples against 6 objects); both fits agree the prior is ~3 px/frame short,
which is the part that matters.

**Correction to commit 550e132:** its message says the merge brought in the
spare-slot answer band. It did not. Franek dropped that himself in 8aeafe5,
keeping our measured `DRONE_FLOOR_ALL` result (0.134 against 0.143). The one
variant still unmeasured is a floor strictly under the 0.001 clip, low enough
that it can never outrank a real answer in any frame.

**Two bugs found while checking the merge**, both in `tools/score_offline.py`:
its `fake_detect` still had the old two-argument signature after `flyby.detect`
gained a `frame` parameter, so `--replay` silently returned **zero** predictions
and scored 0.000 rather than raising; and the predictions-per-frame figure
divided all predictions by the scored subset, so `--frames 40` reported 83.5
boxes per frame instead of 13.4. Both fixed. The first is the same shape as the
missing-model bug that cost us an attempt: a wrong answer that looks like a
working one.

---

## 2026-09-18, evening: v6 and the v4+v6 pair

v6 is `yolo11s`, 40 epochs, 9600 views, the first model trained on backgrounds
cut from the real validation flight (`training/backgrounds_real/`), with the
corrected Helsinki paste scale (0.45–1.00) and re-cut cut-out masks.

**v6 on its own is a wash**, and was never served. Offline replay of the best
v4 run (`tools/score_offline.py --replay`, same run, same truth motion):

| class | v4 | v6 | **v4+v6 alternating** |
|---|---|---|---|
| jet_plane | 0.716 | 0.914 | **0.980** |
| hangar | 0.743 | 0.509 | **0.871** |
| large_tower | 0.001 | 0.504 | **0.557** |
| small_tower | 0.516 | 0.385 | 0.489 |
| large_launcher | 0.644 | 0.133 | 0.381 |
| tank | 0.043 | 0.319 | 0.328 |
| helicopter | 0.284 | 0.302 | 0.269 |
| small_plane | 0.133 | 0.183 | 0.188 |
| jammer | 0.239 | 0.002 | 0.045 |
| mine_roller | 0.000 | 0.064 | 0.043 |
| spacecraft | 0.007 | 0.000 | 0.002 |
| small_launcher | 0.000 | 0.000 | 0.000 |
| **total** | **0.277** | **0.276** | **0.346** |

The v6 recipe did what it was aimed at — tank 0.043 → 0.319, large_tower
0.001 → 0.504 — and paid for it on hangar, large_launcher and jammer. The
totals are a coincidence: the per-class column is completely redistributed.

**The two models fail on opposite classes, so alternating them wins**, the same
mechanism already measured for v4+v5. Both parity orders beat either model
alone, but the order matters more than it should: v4 on even frames scores
0.346, v6 on even frames 0.303. A 0.043 swing from an arbitrary choice is four
times the run-to-run noise floor, so read the pair as **~0.30–0.35, clearly
above either alone at ~0.277**, and do not quote 0.346 as a measurement.

This is not just "more boxes": the pair emits 20.8 per frame against v4's 13.4,
but flooding to 83.6 with `FLOOR_ALL_CLASSES` was already measured to move
nothing (0.225 against 0.226). The gain is the union of what each model finds.

`spacecraft` and `small_launcher` are still dead. `small_launcher` is
resolution-limited (~7 px in a Level-1 view), so no amount of data fixes it.
v6's jammer collapse (0.239 → 0.002) is unexplained and worth a look: the final
evaluation is a different flight, and we do not know which classes dominate it.
The pair hedges that risk, which is a second reason to prefer it to either
model alone.

Offline comparison of an alternating pair is not in `score_offline.py`; it was
measured with a scratch harness that selects between two cached detection sets
on `frame % 2`, matching `flyby.raw_detections`'s `_models[which % len(_models)]`.

### The bug: the served container was three weeks of work out of date

The running container had been built on 17 Sep at 21:02 and never rebuilt.
Its `/app/flyby.py` had **no `ALT_MODEL_PATH` at all**, so `DRONE_MODEL_ALT`
was silently ignored — and it was also **missing Franek's online motion fit**,
merged that same morning at 12:50. Any validation run started that day would
have quietly scored pre-merge code, with no error anywhere: healthy container,
200s, plausible answers.

`docker compose up -d` does **not** rebuild. Use `up -d --build` after any pull
or code change, and confirm what is actually running rather than what you asked
for.

Two guards were added, in the spirit of the existing "refuse to start without a
model" check:

* `/api` now reports `model_alt`, `device`, and **`models_loaded`** — 2 means
  the pair is really alternating. `model_loaded: true` is true when only one of
  the two loaded, so it cannot answer this question.
* asking for a `DRONE_MODEL_ALT` that is not there now **refuses to start**
  instead of logging an error and serving one model.

Both are the same lesson as the `fake_detect` signature bug and the missing-model
bug that cost an attempt: the dangerous failures here are the ones that look
like success.

---

## 2026-09-18, late: 0.2365, and what the service review changed

**The pair scored 0.2365 on validation**, against 0.1445 for v4 — the first
move outside the noise since v2, and it came from pairing two models, not from
a better one. Leader 0.640.

Which of the two logged runs was the scored one is not established: the attempt
ran 14:42:05–14:43:28 UTC, and `serve.log` had no timestamps to line up against
it. The shapes say it was the healthy 248/249 run — its single skip is right
after frame 1, which matches the cold first request — while the 68/249 run
matches the tunnel flapping at 14:40:33Z, ~90 s before the attempt started.
Treat that as inference. Logs are timestamped now, so the next one is decidable.

### Both models on every frame: measured, and it is a variance argument

`DRONE_SET=BOTH_MODELS=1` runs both models on every frame instead of
alternating. Offline, against the same recorded run:

| configuration | offline mAP |
|---|---|
| v4 alone | 0.277 |
| v6 alone | 0.276 |
| alternating, v4 on even frames | 0.346 |
| alternating, v6 on even frames | 0.303 |
| **both on every frame** | **0.326** |

0.326 is almost exactly the midpoint of the two alternating orders (0.325), and
that is the whole point. Alternating assigns models on `request.frame`, so
**every skipped frame flips the assignment** — and frames get skipped. The 0.346
is not a configuration anyone can choose, it is the lucky end of a coin flip
whose expected value is 0.325. Running both buys that expectation outright for
~14 ms of a ~300 ms budget on the M4, and stops `update_tracks` charging a miss
to an object only the other model can see.

It is off by default: it has not been run on validation, and the one previous
measurement of this shape (v4+v5) found nothing in the mean, 46.4 % against
46.2 %. Turn it on with `DRONE_SET=BOTH_MODELS=1` and spend one run on it.

### The offline scorer now knows about the pair

`tools/score_offline.py --model-alt` replays a pair, alternating by frame like
the service, and honours `--set BOTH_MODELS=1`. Both tools patch `flyby.detect`,
which sits *above* the model selection in `raw_detections`, so before this the
served configuration was invisible to them and `DRONE_MODEL_ALT` changed
nothing. It reproduces the scratch harness exactly (0.346, 5169 predictions).
`tools/bench_recordings.py` cannot do pairs and now refuses rather than
silently measuring one model.

### Other fixes from the review

* `api.py` logs carry a timestamp and each frame's own round-trip cost, so
  `serve.log` can be lined up against `tunnel.log` and a bad run blamed on the
  model or the link. This is what made the 0.2365 run undecidable.
* `tools/preflight.py` — checks a live service before an attempt: which weights
  are loaded, real frames replayed, round trip against the 333 ms budget.
* A stale frame no longer raises a track's confidence above `best_confidence`
  (`UNSEEN_DECAY ** max(0, ...)`; the exponent could go negative).
* One malformed annotation no longer discards the whole frame's annotations,
  and a failure building annotations no longer discards the camera command with
  them — that steers the rest of the run, not just one frame.
* `docker-compose.yml` defaults to the pair from the weights baked into the
  image, so `docker compose up -d --build drone-flyby` starts the documented
  service on a fresh clone. `up -d` alone still does **not** rebuild.

### Still open

* **Named tunnel** for the evaluation. The quick tunnel is the one failure here
  that has actually cost frames, and it is not the model.
* A self-ping every 30 s to keep the first-request cost off frame 1.
* `spacecraft` and `small_launcher` remain dead; `small_launcher` is
  resolution-limited at ~7 px.
* v6's `jammer` collapse (0.239 → 0.002 alone) is still unexplained. Both models
  every frame recovers it to 0.241, which is a second reason to try it.

---

## 2026-09-18, night: the resolution floor, and a camera that can reach past it

**The per-class scores are predicted by one number: how many pixels the object
has in the transmitted view.** Every scoring camera we have ever run works at
Level 1, which halves the source. Median object size, and what we score:

| class | source px | at Level 1 | our AP (v4) | object-frames |
|---|---|---|---|---|
| small_launcher | 12.9 | **6.4** | 0.000 | 33 |
| spacecraft | 24.5 | **12.2** | 0.007 | 64 |
| mine_roller | 26.8 | **13.4** | 0.000 | 33 |
| large_tower | 29.5 | **14.8** | 0.001 | 33 |
| jammer | 31.0 | 15.5 | 0.239 | 99 |
| tank | 34.3 | 17.2 | 0.043 | **219** |
| jet_plane | 46.4 | 23.2 | 0.716 | 66 |
| hangar | 108.8 | 54.4 | 0.743 | 70 |

YOLO's finest stride is 8 px, so an object under ~15 px at Level 1 spans one or
two grid cells and sits below the detection floor. The five classes at the top
of that table are **382 of 911 scored object-frames -- 42 % of everything --
and all of them are at ~0.00 in every configuration ever measured here.** That
is not a training problem; the information is gone before the detector sees it.

This also explains why `imgsz=1280` is worth so much: it does not recover
detail, it gives YOLO more grid cells per object. It is a workaround for the
downsample, and Level 2 is the actual fix.

**Level 2 is reachable and has never been used for scoring.** The transitions
are `{0: (0,1), 1: (0,1,2), 2: (1,2)}` -- `allowed_resolution_levels` reads
`[0,1]` only when you are at Level 0, because levels change one step at a time.
From Level 1 it is `(0,1,2)`, which is 4681 of our recorded requests. Every
pattern in `SWEEPS` is Level 0/1 only; `survey` is Level 2 and is labelled a
data-collection pattern.

**Measured, replaying the recorded survey run through v4** (`--run c0c70bce...`):

| class | `full` (all L1) | `survey` (99 % L2) |
|---|---|---|
| mine_roller | 0.000 | **0.180** |
| small_launcher | 0.000 | **0.052** |
| tank | 0.043 | **0.084** |
| jet_plane | 0.716 | 0.000 |
| hangar | 0.743 | 0.230 |
| **total** | **0.277** | **0.080** |

`mine_roller` and `small_launcher` had never been non-zero in any configuration.
Native resolution makes them real. But pure survey works the top half only and
abandons the large classes, so it loses far more than it gains as a whole.

### `DRONE_CAMERA=hybrid`

Alternates Level-2 acquisition passes over the survey rows with Level-1
coverage. The shape follows the flight: the ground scrolls ~69 px/frame, so
every object enters at the top edge and crosses in ~31 frames. Catch it once
near the top at native resolution and memory carries it down; the Level-1 phase
stops those tracks drifting out of IoU and picks up what L2's narrow view walks
past. `HYBRID_ACQUIRE` (9) and `HYBRID_COVER` (3) are `DRONE_SET`-tunable.

Simulated over a whole flight against the evaluator's own three checks
(`DRONE_CAMERA=hybrid python tools/camsim.py`): **187 of 249 frames at
Level 2, 61 at Level 1, zero illegal or refused commands**, full horizontal
coverage of the top band. `full` and `survey` are unchanged, 0 refusals each.

**It has no offline number and cannot get one.** A replay feeds back the
*recorded* views, so a camera that would have looked elsewhere has nothing to
replay. `survey` at 0.080 and `full` at 0.277 bracket it; the split has to be
tuned on real runs. If a run disappoints, tune down
(`DRONE_SET=HYBRID_ACQUIRE=6,HYBRID_COVER=6`) before abandoning it -- but note
that below ~50 % the L2 snake sweeps slower than objects cross the band, which
buys the cost without the acquisition.

`DRONE_INSPECT` is not a cheaper version of this: it only zooms on tracks that
already exist, and our dead classes never produce a track at all.

### One model at two scales is a pair

`DRONE_IMGSZ=960,1280` with the same weights named twice. On a machine that
cannot afford v6, this is the whole gain of pairing without a second model:

| configuration | offline | detector cost (i5) |
|---|---|---|
| v4@960 | 0.277 | 112 ms |
| v4@1280 | 0.318 | ~200 ms |
| **v4@960 + v4@1280 alternating** | **0.333** | **~156 ms avg** |
| v4@960 + v4@1280 both every frame | 0.349 | ~312 ms |

Alternating beats 1280-alone on score *and* costs less on average. Both-every-
frame scores highest but leaves 21 ms of a 333 ms budget on the i5.

### `tools/score_offline.py` takes `PATH:SIZE`

`--model models/drone-yolo11n-v4.pt:960 --model-alt models/drone-yolo11n-v4.pt:1280`
scores a two-scale pair, so every configuration that can be served can now be
scored by a committed tool. Verified to reproduce 0.277 / 0.318 / 0.333 / 0.349
and the v4+v6 pair's 0.346 exactly.

It sets **both** `flyby.IMGSZ` (the cache key) and `flyby.IMGSZ_LIST` (what
`raw_detections` actually resizes with). Setting only the first names a cache
file 1280 while computing at 960 -- a poisoned cache that reads as a working
one, and it already cost another session four silently wrong runs.
