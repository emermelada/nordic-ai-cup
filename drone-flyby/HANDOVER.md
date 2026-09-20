# START HERE (2026-09-19, 18:30 CEST) — read this block, then skip to the end

**Best configuration, validated, mean 0.5270 over FOUR complete runs
(0.5187 / 0.5234 / 0.5320 / 0.5339, sd 0.0072; best 0.5339).**
It is `BEST-WORKING-VERSION`, and it needs NO code beyond what is committed:

```bash
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280,2560 \
DRONE_DEVICE=cuda DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 python3 api.py
```

Start it with `tools/arm.sh`, which verifies `/api` and refuses to hand over a
service that is not serving what you asked for. Set up a fresh rented box with
`tools/bootstrap_remote.sh` — it gates on latency AND sustained bandwidth to
the evaluator (Hetzner Helsinki), which is worth more than any config change:
a bad host cost 0.05–0.30 today, three times what a day of tuning gained.

**The day: 0.4788 → 0.5211.** Four things did it, and none was a better model:

| change | gain | why it was missed before |
|---|---|---|
| `DRONE_BOX_GROW=1.3` flat | +0.017 | the per-class profile measured mask spindliness, not box error |
| `NEW_TRACK_CONFIDENCE=0.10` | +0.013 | tuned against v3 at score 0.132, two generations stale |
| a 4th model at `imgsz 2560` | +0.014 | 1600 had lost as a *replacement*; as an *addition* it wins |
| the endpoint | up to +0.30 | latency and bandwidth to Helsinki, never previously checked |

`small_launcher` went **0.000 → 0.512**. It had been dead all project.

**The single most important thing to know:** the truth file
(`training/validation_objects.json`, 32 objects) is badly incomplete. Many of
our "false positives" are real rendered assets nobody labelled — see
`data/unmatched_sheet.png`. So **`tools/score_offline.py` per-class AP is not a
target list**, the apparent "ranking loss" is largely measurement error, and
every mechanism that suppresses weak detections has failed. What pays is
detecting MORE.

**Measured dead today — do not re-run:** Level 2 / hybrid camera (−0.049 even
with its bug fixed), the 0.0 floor band (+0.0015), growth cap >1.3 (−0.019 at
1.45), `AGREEMENT_WEIGHT` (−0.005), a 5th pass at 3200 (blows the 333 ms
budget: 514 ms, 0.264), a 5th model v4@2560 (0.4994).

**Protocol that made the day work:** three complete runs per arm, 249/249 or
discard, one change at a time, and check `/api` before every attempt. Runs are
deterministic given the camera path — hash the view sequence and identical
paths give identical scores to 16 digits, so a matched pair beats six unpaired
runs.

---

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
| v4+v6 alternating | 0.2365 | the two fail on opposite classes; +0.09 on v4 |
| v4+v6, BOTH_MODELS=1 | 0.2450 | inside noise of 0.2365, as predicted; buys determinism |
| **v4@960 + v6@1280 (served, BEST)** | **0.3048** | the resolution floor, confirmed on a real run: +0.06 |
| v7@1280 alone | 0.2611* | trained at 1280; LOSES to v6@1280, see 18 Sep night |
| v4@960 + v7@1280 | 0.2405* | v7 does not help in a pair either |
| v4@960 + v6@1280, camera hybrid | 0.1234* | Level-2 acquisition camera: clearly worse |

*Starred rows were served from a rented GPU box, which measures ~0.02 lower
than the Mac on an identical config (control: 0.2845 against 0.3048).

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

### The patch harvester was carrying boxes with the Helsinki prior

`training/harvest_validation_patches.py` carried each object's box from its
nearest sighting with `flyby.MOTION` -- the same stale prior Franek found in the
tracker and `tools/score_offline.py`, one layer further back. It runs
~2.6 px/frame short here, and `MAX_FRAME_DISTANCE` is 20 frames: up to ~52 px of
drift against a `SEARCH` window of 24 source px, so past about nine frames the
object sits outside the window the template match looks in and the view is
dropped or cut off-centre. Here it cost training data rather than score, which
is why nothing caught it.

It now fits the motion on the flight's own objects (`--prior-motion` restores
the old behaviour to A/B it). Measured, same recordings:

| | patches | of which native (L2) |
|---|---|---|
| prior (`flyby.MOTION`) | 228 | 34 |
| **fitted** | **262** | 36 |

`training/patches_val` has been regenerated with the fitted harvest: tank 69 ->
77, jammer 13 -> 26, spacecraft 15 -> 17, helicopter 28 -> 32.

**The native count barely moved, and that is the useful finding.** Only 36 of
262 cut-outs are native resolution; the rest come from Level-1 views and carry
half the real detail they appear to. `small_launcher` patches are ~16 px wide
with ~8 px of information in them. The limit is not the carry, it is coverage:
`survey` sweeps the top half only, and the flight is deterministic, so repeating
it returns byte-identical views and harvests nothing new. More native patches
need a *different* survey pattern (the bottom rows, y=1350 and y=1890) and one
recorded run with `DRONE_RECORD=1`. The score of such a run is irrelevant.

This matters more now than it did: we serve at 1280 and are about to train
there, while most of the cut-outs teaching the model what an object looks like
were cut from half-resolution views.

---

## 2026-09-18, night: four real runs, and the offline scorer is now anti-correlated

Everything below is a REAL validation score. Where a config was served from the
rented GPU box rather than the Mac it is marked; the box measures **~0.02 lower**
on an identical config, established by a control run (0.2845 box against 0.3048
Mac for v4@960+v6@1280). Compare within a machine only.

| config (all `camera: full` unless noted) | real | machine |
|---|---|---|
| **v4@960 + v6@1280, BOTH_MODELS=1** | **0.3048** | Mac |
| v4@960 + v6@1280, BOTH_MODELS=1 (control) | 0.2845 | box |
| v7@1280 alone | 0.2611 | box |
| v4@960 + v7@1280, BOTH_MODELS=1 | 0.2405 | box |
| v4@960 + v6@1280, BOTH, **camera hybrid** | 0.1234 | box |

### The offline scorer must not be used to choose models any more

It ranked v7@1280 alone at **0.405** and v4+v6@1280 at **0.389**. Reality ranked
them 0.2611 and 0.2845 — on the same machine, so the ordering is clean. This is
the first time the tool has been wrong about *ordering* rather than magnitude,
and it retires it for model selection.

**Untested hypothesis for why, and it is the highest-value thing left.** The
offline truth is 29 confirmed objects mined from our own earlier detections. A
detection of a real-but-unconfirmed object scores as a false positive offline
while being a true positive in reality, so the metric rewards conservative
models. Consistent with the evidence: v7@1280 emits 2679 predictions where
v4@960+v6@1280 emits 5774, and offline preferred the one emitting fewer.

If that holds, **re-mining the ground truth with the best config restores fast
offline iteration**, which is worth more than any single experiment — every
decision currently costs a real run. Test it before fixing it (correlate
boxes/frame against the offline-vs-real gap). Do not add speculative labels.

### v7 (yolo11s trained at imgsz 1280) — trained, tested, does not help

0.023 below v6@1280 on the same machine, which is about twice the noise floor.
Weights are committed as `models/drone-yolo11s-v7.pt` so nobody spends another
GPU hour rediscovering this. Training details: yolo11s, 40 epochs, imgsz 1280,
9600 views, the 262-patch fitted harvest, `REAL_SHARE=0.6`. Synthetic validation
was healthy (mAP50 0.567, precision 0.87 against recall 0.50) — it just does not
transfer.

Best guess at the cause, and it points at the fix: **v7 was trained at 1280 on
cut-outs that are mostly Level-1, i.e. 2x downsampled — only 36 of 262 patches
are native resolution.** It learned what blurry objects look like enlarged. v6,
trained at 960 and *run* at 1280, gets the resolution boost without that
mismatch. A v8 needs native patches first, which needs a bottom-half survey
pattern and one recorded run, not more GPU time.

### The hybrid camera is dead

`DRONE_CAMERA=hybrid` scored **0.1234** against a 0.2845 control — 0.161 below,
far outside any machine effect. Level 2 does lift the tiny classes in isolation
(mine_roller 0.000 -> 0.180, small_launcher 0.000 -> 0.052 on the recorded survey
run) but nowhere near enough to pay for the coverage it surrenders. The camera
question is closed: `full` wins. Drop it rather than tune the split.

### Serving from a rented GPU box: works, and is not worth it

v7 on a 5090 answered in **28 ms median, 249/249 frames, zero gaps** — compute is
irrelevant to this problem. But the box scores ~0.02 lower than the Mac and costs
$0.73/h, and the Mac already uses only ~7% of the frame budget. Serve from the
Mac.

Two operational traps found the hard way: `pkill -f <pattern>` over ssh matches
the ssh session's own command line and kills the shell mid-command (use
`pkill -x`); and a service that fails to bind because the old one still holds the
port leaves the OLD config serving while every log line looks healthy — always
re-read `/api` after a restart.

---

## 2026-09-19: the offline scorer was inverted by one unlabelled object

**`tools/score_offline.py` can now be trusted to choose between models.** Six
configurations have both a real validation score and a replayable recording.
Ranked by the old scorer they correlate with reality at Spearman **+0.77**, with
one inversion — and it is the one that mattered: offline preferred `v7@1280`
over the served `v4@960+v6@1280`, reality preferred the opposite. With the fix
below it is **+0.94**, and the only remaining inversion is the v4+v6 parity pair
already known to be a coin flip (0.0085 apart, inside the noise floor).

### The cause was not "too many boxes"

`training/validation_objects.json` held 29 objects **mined by v2**, a model that
scored 0.097. The macro average runs over 12 classes and **four of them rested
on a single object**. The flight actually contains two `large_launcher` sites,
three `large_tower`s and two `mine_roller`s.

Every detection of an unlabelled real object scored as a false positive, and
that penalty falls hardest on whichever configuration emits the most boxes. On
`large_launcher` the effect was worth **+0.413 of AP** to v7 — three times the
whole 0.016 gap between the two candidates — while **recall on it was identical
(0.941 both)**. Neither model missed the confirmed launcher; the served pair was
punished for also finding the unlabelled one.

**One object is load-bearing.** Drop the second `large_launcher` from the
correction and keep the other four and the scorer reverts to +0.77 with the
ranking inverted again; keep only that one and it is +0.94 and correct. It is
necessary as well as sufficient.

### What changed

* `training/validation_ignore.json` — regions that are real objects the truth
  never listed, mined by cross-model agreement (v3/v4/v6) over all 28 recorded
  runs, then checked by eye against Level-2 crops. **Ignore regions, not
  labels**: a wrong one costs a little precision signal and cannot poison the
  training set.
* `tools/score_offline.py` — honours them as class-agnostic `iscrowd=1`
  annotations. `--no-ignore` restores the old behaviour for an A/B.
* `training/promote_candidates.py` — promotes a reviewed region to a real object.
  It refuses to act on the verdict alone: you name each candidate and its class,
  because the class in the file is the detector's vote. It deduplicates
  observations by frame (candidates are mined across 28 runs, and duplicates
  would bias `fit_truth_motion`), and marks the region `promoted` so it stops
  being an ignore region too.
* Three were promoted (2 × `large_tower`, 1 × `mine_roller`), taking the truth
  to **32 objects**. The second `large_launcher` deliberately stays an ignore
  region: at native resolution it is **two vehicles** — a transporter and a
  launcher with four outriggers deployed — so one box over it would be bad
  geometry and would teach the harvester to cut a two-vehicle patch.
* `training/patches_val` re-harvested: `large_tower` 7 → 22, `mine_roller`
  10 → 19, 263 → 288 cut-outs.

### The target list is different from what this document said

Under the corrected scorer, per class for the served pair:

| class | AP | object-frames | | class | AP | object-frames |
|---|---|---|---|---|---|---|
| jet_plane | 0.958 | 66 | | tank | **0.354** | **219** |
| hangar | 0.871 | 70 | | helicopter | **0.249** | **100** |
| large_tower | **0.831** | 33 | | small_plane | **0.225** | **99** |
| small_tower | 0.756 | 78 | | jammer | **0.201** | **99** |
| large_launcher | 0.682 | 17 | | mine_roller | 0.082 | 33 |
| | | | | spacecraft | 0.003 | 64 |
| | | | | small_launcher | 0.000 | 33 |

**Only `spacecraft` and `small_launcher` are dead — 11 % of scored
object-frames, not the 42 % the five-dead-class table above implies.**
`large_tower` is 0.831, not 0.001. The points are in four mid-scoring,
high-volume classes — tank, helicopter, small_plane, jammer — which are **57 %
of everything scored** and all sit at 0.20–0.35. `training/train_remote.sh` now
carries a v8 recipe aimed there.

### Measured dead, on the corrected scorer — do not spend runs on these

* `UNSEEN_DECAY=1.0` — the +0.012 that made it an open item is **+0.004**
  corrected. Inside noise.
* **v5 in any pairing** — 0.322 alone, 0.361 with v4, 0.384 with v6; added as a
  third model it *lowers* the served pair from 0.435 to 0.412.
* **Detection fusion** — weighted box fusion +0.004 over the current
  concatenation, NMS across models −0.009, averaging class probabilities
  catastrophic (0.125). The pair should keep concatenating.
* **Resolution past 1280** — v6 alone: 0.288 at 960, 0.398 at 1280, **0.370 at
  1600**. 1280 is the peak, not a floor still being climbed.
* **Rotation TTA** (0/90/180/270, NMS-fused) — helps only the weakest model on
  its own (v4@960 +0.044). It *costs* 0.025 applied to v4 in the pair and
  **0.087** applied to v6, the worst result measured. TTA and pairing buy the
  same thing — more chances for a faint object to fire — and do not stack. Do
  not ship it. Note also that training-time rotation is already covered:
  `make_dataset.py` rotates every pasted cut-out and `flipud=0.5` is set, so
  `degrees=0.0` is deliberate.

### `LEVEL_WEIGHT` is read, and it was set wrong

This document said `LEVEL_WEIGHT` is "structurally never read". **It is read**,
at `flyby.py:508`, where it weights a detection's class votes by the resolution
level it was seen at. It never touches a track's confidence, so scaling all
three levels together cancels (measured: −0.001); the lever is the ratio.

Level 0 was 0.4. Raising it to **1.0** helps in every run tested, most where
there is most evidence: `5ace5364` (3 L0 views) +0.010, `8a1d65ee` (130) +0.040,
`a1c00d7c` (90) +0.039, `81bf6bd3` (3) +0.009, `04bef8d0` (2) +0.005. Weight 2.0
was rejected — it scores higher on the three-view runs than on the well-sampled
ones, which is noise, not effect. **Now the default.**

Temper it: the served `full` camera takes only 3 Level-0 views, so expect the
+0.010 row, at the real noise floor — not +0.04. **It has not been confirmed on
a real run.** `DRONE_SET=LEVEL_WEIGHT={0:0.4,1:0.8,2:1.0}` restores the old
value. And do not read this as "take more Level-0 views": that is a camera
change, and the last one scored 0.1234 against a 0.2845 control.

### One more silent failure fixed

`annotations_for` divided by `top_vote` without guarding zero. A track holding
only zero-weight votes raised `ZeroDivisionError` inside the caller's `try`,
which discards **the whole frame's annotations and its camera command** — a
total loss that looks like a quiet frame. Not reachable in the served config,
but it is the same shape as every other expensive failure here.

### Mining is exhausted

Two further passes found nothing: relaxing cross-model agreement (14 candidates)
and relaxing confidence to 0.25 with agreement kept (19). Tier 2 was tree
canopy, rooftops and a pond; tier 3's apparent finds were re-detections, and the
one striking candidate turned out to be the lower section of the confirmed
`large_tower`, split into its own cluster. **Cross-model agreement was doing all
the work** — loosening it only adds false alarms. Review sheets are in
`data/mining/` (gitignored).

---

## 2026-09-19, 03:02 CEST: 0.4618 — the box convention

**New best on validation: 0.4618, up from 0.3048.** The largest single gain this
project has had — bigger than model pairing (+0.09) or the resolution fix
(+0.06). It came from neither a model nor a tracker setting.

The evaluator's boxes are the **projected 3D box** of each object: rotor span,
wingtips and height included. `make_dataset.py` labels every pasted cut-out with
the **tight box around its alpha mask**, so our models learned tight boxes and
were scored against loose ones for the entire competition. Growing the reported
boxes per class toward the official convention closes the gap.

The evidence was in this file all along and nobody chased it: `BOX_SCALE=0.8`
collapsed a real run from 0.143 to **0.017**. Shrinking a well-matched box by
20 % gives IoU ~0.64, over the 0.50 threshold, worth a couple of points at most.
An 88 % collapse only happens if the boxes were already sitting just above
threshold — which is exactly what a systematic size mismatch looks like.

### The exact configuration that scored it

Branch **`BEST-WORKING-VERSION`**, commit **`945e89b`**. Do not move that branch
unless a real validation run beats 0.4618.

```bash
DRONE_BOX_GROW=helsinki DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt \
DRONE_IMGSZ=960,1280 DRONE_DEVICE=mps DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1 .venv/bin/python api.py
```

`/api` must show `models_loaded: 2`, `imgsz: [960, 1280]`, `box_grow` filled
with 16 classes, `box_grow_cap: 1.3`. An empty `box_grow` means the baseline is
being served and the attempt teaches nothing.

Note the run also carried `LEVEL_WEIGHT[0]=1.0`, so strictly two things changed
at once. Growth dominates (predicted +0.085 against +0.01), but a run with
`DRONE_LEVEL0_WEIGHT=0.4` would attribute it cleanly.

### Keep the cap at 1.3

A correction to advice given earlier the same night: raising the cap was
suggested on the reasoning that reality had beaten the offline prediction. It
had not — reality came in **under** it (0.4618 against a predicted 0.488). Swept
properly on the loose truth: cap 1.3 → 0.488, 1.5 → 0.480, 1.8 → 0.474,
2.4 → 0.474. **1.3 is the optimum.**

### The scorer's default changed

`tools/score_offline.py --truth-boxes loose` is now the default. The tight truth
got the **direction** wrong: it predicted growth would cost 0.083 where reality
gained 0.157. Loose truth reads ~0.026 high with growth on, ~0.098 high with it
off, and gets the sign right. `--truth-boxes tight` restores the old behaviour.

`training/measure_box_convention.py` reproduces the factors and reports mask
fill per class: a class whose GrabCut mask fell back to a filled ellipse reports
a ratio of 1.0 that is a mask failure, not evidence the object is already loose.
No class on this machine's cut-outs exceeds 0.60 fill, so all sixteen ratios
are trustworthy.

### v8 (yolo11m, 40 epochs, imgsz 1280) — trained, judged, not served

Offline on the loose truth with growth on, against the served pair at 0.488:

| configuration | score |
|---|---|
| v4@960 + v6@1280 (served, real 0.4618) | 0.488 |
| v8@1280 alone | 0.442 |
| v4@960 + v8@1280 | 0.461 |
| v6@1280 + v8@1280 | 0.473 |
| **v4@960 + v6@1280 + v8@1280** | **0.497** |

v8 did what its recipe aimed at and paid what the recipe risked: helicopter
0.527 → 0.659, **spacecraft 0.003 → 0.107** (non-zero for the first time in this
project), jammer and tank up; hangar 0.871 → 0.505, small_plane, large_tower and
large_launcher down. Those are exactly the classes its weights cut to 0.4–0.7.
**The down-weighting was too aggressive** — a repeat should not go below ~0.8 on
a class already scoring well.

Only the three-model ensemble beats the served pair, by +0.010, which is at the
real-run noise floor. The better argument for it is hedging: v8 and v6 are close
to complementary, and the final evaluation is a different flight whose class mix
is unknown.

### The pattern worth carrying forward

Four attempts to improve the **model** — v5, v7, v8, rotation TTA — all failed or
came in marginal. Two fixes to **conventions and measurement** — the ignore
regions and the box convention — gained +0.157 between them. The leverage in
this project has been in what we compare against, not in the detector.

That makes the next training run **v9: the v6 recipe with official-convention
labels** (`--box-convention official`, paste scale corrected), not another
backbone. A model trained on official labels needs no report-time growth at all.

---

## 2026-09-19, 04:05 CEST: 0.4788 — three models, and frames matter more than models

**New best: 0.4788** (mean of complete runs 0.4753), from `v4@960 + v6@1280 +
v8@1280`. Branch `BEST-WORKING-VERSION` now points here.

```bash
DRONE_BOX_GROW=helsinki DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280 DRONE_DEVICE=mps DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1 .venv/bin/python api.py
```

`/api` must show `models_loaded: 3`, `models_requested: 3`,
`imgsz: [960, 1280, 1280]`, `box_grow` with 16 classes, `box_grow_cap: 1.3`.

### Six runs, three per configuration

| score | config | frames |
|---|---|---|
| 0.4788 | three | 249/249 |
| 0.4718 | three | 249/249 |
| 0.4580 | two | 249/249 |
| 0.4544 | two | 249/249 |
| 0.4497 | three | **247**/249 |
| 0.4461 | two | **247**/249 |

Both complete three-model runs beat both complete two-model runs with no
overlap: **+0.0191**, against a within-arm sd of 0.005 (three) and 0.0025 (two).

### The noise floor is tighter than this document says, once frames are complete

±0.01 is the figure recorded above, and an earlier reading of these same six
runs put it at ±0.015. Both are wrong. **Among runs that answered 249/249 the
standard deviation is about 0.005.** The apparent noise was one frame-losing run
in each arm. Judge a configuration only on complete runs; a short run is not a
noisy sample of the same thing.

### Two missing frames cost more than the model choice

| | complete mean | with 2 frames missing | cost |
|---|---|---|---|
| three models | 0.4753 | 0.4497 | **−0.0256** |
| two models | 0.4562 | 0.4461 | −0.0101 |

Two frames out of 249 — under 1 % — cost up to 0.026, more than the entire
three-model gain. **The named Cloudflare tunnel is now the highest-value
remaining work, ahead of any further modelling.** The quick tunnel has already
logged QUIC timeouts and reconnects tonight, and the final evaluation is one
attempt.

Add to the pre-attempt protocol: after every run, count the frames.

```bash
for d in data/recordings/*/; do printf "%s  %s\n" "$(basename $d)" "$(ls $d/*.png | wc -l)"; done | tail -5
```

**249/249, or the score does not count.**

### Where the score has come from

| | score | what changed |
|---|---|---|
| v4 alone | 0.1445 | |
| v4+v6 pair | 0.2450 | two models, one object memory |
| v4@960+v6@1280 | 0.3048 | per-model inference size |
| + box growth | 0.4618 | **the official box convention** |
| + v8 as a third model | **0.4788** | class coverage, mainly spacecraft and helicopter |

The two largest steps — box growth (+0.157) and pairing (+0.10) — were not
better detectors. Four attempts at a better detector (v5, v7, rotation TTA, and
v8 *as a replacement*) all failed; v8 only paid as an *addition*. The leverage
has been in the scoring convention and in class coverage, not in the backbone.

---

## 2026-09-19, 15:00 CEST: 0.5055 — the growth profile was wrong, and so was the track threshold

**New best: mean 0.5055 over complete runs, best single run 0.5113**, from
`DRONE_BOX_GROW=1.3` (flat, every class) plus
`NEW_TRACK_CONFIDENCE=0.10`. Both are one-word configuration changes on the
code already in `1a7810c`; no code change is needed to serve this.

```bash
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280 DRONE_DEVICE=mps DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 .venv/bin/python api.py
```

`/api` must show `box_grow` with all 16 classes at `[1.3, 1.3]`,
`new_track_confidence: 0.1`, `models_loaded: 3`, `imgsz: [960, 1280, 1280]`.

### Runs are deterministic given the camera path — use pairs, not means

The largest methodological finding of the day. Hash a run's full view
sequence (`tools` scratch script, 249 x (frame, level, cx, cy)) and runs with
the same hash return **the same score to sixteen digits**. Trajectory
`ae2c83fb` came up 7 times in 30 runs; under track-conf 0.15 it returned
0.5017590925376386 three times out of three.

This means **run-to-run "noise" is not noise** — it is which camera path the
run happened to take, decided by whether an early command lands before the
evaluator renders the next frame. Comparing arms by their means is confounded
by the draw: flat 1.3 drew its worst trajectory four times while track-conf
0.15 drew its best three times, which made a real +0.012 look like +0.004.

The whole day, measured on one identical trajectory (`ae2c83fb`):

| config | score | step |
|---|---|---|
| `helsinki` profile, track-conf 0.25 (yesterday's best) | 0.4762 | |
| profile at cap 1.45 | 0.4771 | +0.0009 |
| **flat 1.3**, 0.25 | 0.4893 | **+0.0131** |
| flat 1.3 + floor band | 0.4908 | +0.0015 |
| flat 1.3, **track-conf 0.15** | 0.5018 | **+0.0125** |
| flat 1.3, track-conf 0.075 | 0.5065 | +0.0047 |

On `82a5e16a`: profile 0.4849, flat 1.3 **0.4994**, track-conf 0.05 0.4842.

### The box-convention factors measure mask spindliness, not the convention

`training/box_convention.json` is `patch box / alpha-mask box`, which tracks
how spindly an object is inside its cut-out. It is **not** a measurement of any
disagreement between our boxes and the evaluator's.

Measured against the evaluator's own ground truth: v4, v6 and v8 over all 25
official Helsinki frames, six Level-1 sweep views each, matched by centre
proximity. **Every class, every model: predicted box / official box =
0.97-1.04, median IoU 0.79-0.98.** The only real per-class bias found is v4/v6
under-sizing hangar height ~1.19x and v8 under-sizing medium_launcher.
(Caveat: Helsinki is the scene these models were trained from, so this shows
the box regressor *can* produce official boxes, not that it does on a new
flight.)

Growing a correctly-sized box only lowers IoU, and past g = 1.414 a perfectly
centred box is already under the 0.50 threshold on its own — which is why
`flat 1.45` lost 0.019. Growth pays only against **under-sizing**, and the
optimum is exactly 1/r where r is how undersized we are.

The real under-sizing is **resolution**. Rendering each confirmed object from
`data/scene` as a native Level-2 view and as a 2x-downsampled Level-1 view and
running all three served models on both: the Level-1 box is smaller, most on
the smallest objects — spacecraft 1.23, large_tower 1.13, jammer 1.12 against
~1.02 for the large classes. Spearman between that shrinkage and the profile
we were serving: **-0.02**. Uncorrelated. Hence flat.

The growth curve, on real runs:

| growth | mean | |
|---|---|---|
| `helsinki` profile (~1.1 effective) | 0.4754 | |
| **flat 1.30** | **0.4922** | peak |
| flat 1.45 | 0.4740 | past the cliff |

### `score_offline.py` is blind to box growth, and inverted on it

It grows the *truth* by the same `box_convention.json` factors that `BOX_GROW`
applies to predictions, so for every class under the cap both sides scale about
their own centres by the same number and the effect cancels. The old
"1.3 -> 0.488, 1.5 -> 0.480" cap sweep was reading second-order noise.

Worse, it is **anti-correlated** on this arm: it scored flat 1.3 at 0.434
against the profile's 0.461, where reality said +0.017. Screening flat 1.3
offline would have thrown away the best result of the day. It was accurate on
the camera arm (-0.046 offline against -0.049 real), so the rule is: **trust it
for anything that does not touch box geometry, never for anything that does.**

### The track-admission threshold was two model generations stale

`NEW_TRACK_CONFIDENCE` decides whether a detection becomes a remembered track —
answered for an object's whole ~31-frame transit — or a one-frame guess. It had
been 0.25 since it was tuned against v3 at a score of 0.132, before the
three-model ensemble, before box growth, before the motion fix.

| threshold | mean | boxes/frame |
|---|---|---|
| 0.25 | 0.4922 | 32.6 |
| 0.15 | 0.4981 | 42.2 |
| **0.10** | **0.5055** | ~50 |
| 0.075 | 0.5065 (1 run) | 58.3 |
| 0.05 | **0.4842** | 71.3 |

A real peak with a cliff between 0.05 and 0.075: below it, faint *false*
detections also become persistent tracks and outrank genuine faint answers
elsewhere in the flight — the same failure `FLOOR_ALL_CLASSES=0.01` hit.
**0.10 is served rather than 0.075** because they score the same and 0.10 is
further from the cliff; on a more cluttered flight the cliff moves up.

### The rented box does NOT score lower than the Mac

The ~0.02 penalty recorded in this file ("the box measures ~0.02 lower on an
identical config") was **the Cloudflare tunnel, not the machine**. Served
direct over a Vast-mapped port, three control runs gave 0.4652 / 0.4849 /
0.4762, mean **0.4750**, against the Mac's 0.4753 — a difference of 0.0003.

Confirmed independently: dumping raw detections for the same 40 recorded frames
on CUDA and on CPU, the two agree to **0.008 px median box error and 0.00013
median class-score error**, 213 of 214 detections matched at IoU > 0.9. The
detector is numerically identical. Compare across machines freely; prefer a
direct port to any tunnel.

### Measured dead today — do not re-run these

* **Level 2 in any form.** `hybrid_next_view`'s coverage branch picked the
  nearest Level-1 sweep point measured from where the camera already was — which
  after it arrived was the point it was standing on — and never advanced
  `sweep_index`, so the coverage phase re-looked at one sixth of the frame.
  Fixed (a `cover_index` that advances; `full` is byte-identical). At a low duty
  cycle the fixed camera costs nothing in coverage — 217 L1 / 31 L2 / 0 refused,
  median 63 looks per cell, same as `full`. **It still lost 0.049 on a real
  run.** The per-class column says why: hangar -0.298, large_tower -0.115,
  large_launcher -0.095. Our models were trained on Level-1-scale views, so at
  1:1 they *name* objects worse (measured: hangar correct-class 0.43 -> 0.14,
  mine_roller 1.00 -> 0.48), and `LEVEL_WEIGHT[2]` is 1.0 with votes that
  accumulate forever, so a few confident wrong-class native sightings rename a
  well-established track. Meanwhile the target classes did not improve at all
  (small_launcher 0.001 -> 0.000). 30 native views cannot cover a 13 px object's
  31-frame transit. **The camera question is now closed on a working
  implementation.**
* **A floor band at confidence 0.0** (`DRONE_FLOOR_ZERO=1`, added this session,
  off by default). The scorer semantics check out — maxDets is 100 per image
  *per category*, we emit at most 8 of one class per frame, and 150 junk boxes
  at score 0.0 leave a perfect class at AP 1.000 — and the implementation is
  clean (0 floor boxes ranked above a real answer, max 30 of one class per
  frame). It is simply worth **+0.0015**: paired, 0.4908 against flat 1.3's
  0.4893. Not worth 4x the response size.
* **Growth cap above 1.3.** Only four scored classes have a factor above 1.3, so
  the cap is a weak lever; raising it to 1.45 moved 8 of 12 classes not at all.

### Corrections to the v9 plan in this file

1. **Do not train v9 with official-convention labels.** The flag
   (`--box-convention official`) does not exist in `make_dataset.py`, and the
   premise is wrong: our boxes already match the official convention on
   Helsinki. `BOX_GROW` is applied in `annotations_for` with no knowledge of
   which model produced a box, so a v9 predicting larger boxes would get flat
   1.3 stacked on top and overshoot — and overshooting cost 0.019 today.
   **Use the same label convention as v4/v6/v8.**
2. **Raise the class-weight floor to 0.8.** The v9 recipe carries hangar=0.4,
   jet_plane=0.5, large_tower=0.5 — repeating the v8 mistake this file already
   documents ("should not go below ~0.8 on a class already scoring well"). v8
   cut hangar to 0.4 and its hangar AP fell 0.871 -> 0.505. Those are still our
   best classes.
3. **v9 does not drop in.** Growth 1.3 and track-conf 0.10 were tuned against
   this ensemble's box statistics and confidence calibration. A fourth model
   changes both; re-check track-conf at minimum.
4. **Two v9 plans existed and they contradicted each other.** This file
   (19 Sep 03:02) said "the v6 recipe with official-convention labels, not
   another backbone"; `train_remote.sh` and `training/yolo11-p2.yaml` describe
   a yolo11m with a P2 head. The first is dead — its premise was that a model
   on official labels needs no report-time growth, and there is no convention
   error to train away. **The P2 recipe is the one to run**, and today's
   Level-1-vs-Level-2 measurement supports it: `small_launcher` was found 6/6
   at native resolution and 0/6 at Level 1, and P2 gives it at Level 1 the cell
   count native resolution gives it today. Keep the *scepticism* that wrote
   "not another backbone", though: it was right five times out of six, and
   today's +0.030 again came from conventions, not capacity.

### Tooling changed this session

* `tools/arm.sh` — starts one sweep arm from a named baseline plus overrides and
  refuses to hand it over unless `/api` agrees: models_loaded == models_requested,
  box_grow 16 classes, imgsz matching the model count, recording on. Kills only
  its own service via a PID file (`pkill -x python3` would take the rented box's
  Jupyter; `pkill -f` matches the ssh command line and has killed a shell
  mid-command three times). `tools/arm.sh frames` counts and flags short runs.
* `tools/preflight.py` — `--expect-models` now defaults to `models_requested`
  from `/api` rather than the constant 2, which silently passed while three
  models were served. `--expect-grow` fails on an empty or short `box_grow`.
* `tools/score_offline.py` — `--model-alt` is repeatable, so it can finally
  replay the three-model configuration that is actually served.
* `api.py` — `/api` reports `inspect`, `det_conf`, `track_conf`, `both_models`,
  `floor_zero`, `floor_size_tol`, `miss_penalty`. `DRONE_INSPECT` previously
  could not be confirmed from outside the process at all. `DRONE_PORT` makes the
  bind port configurable (a rented box exposes only the ports its template
  mapped).
* `flyby.py` — `DRONE_FLOOR_ZERO`, `DRONE_FLOOR_SIZE_TOL`, `DRONE_MISS_PENALTY`,
  all off by default and proved byte-identical to `1a7810c` when unset by
  replaying 120 frames of synthetic detections and hashing the answers.

### Protocol notes

* **Count frames every time.** 5 of 30 runs today came back short and every one
  had to be discarded. Two of those were the service being restarted mid-run —
  check the recording's last-modified time before switching arms.
* The competition service returned one attempt with
  `submitted_at == started_at == finished_at` and a stale-looking score. It was
  real (the run executed fully on our side) but the timestamps were broken.
  Two other attempts sent exactly one frame and then stopped, returning a 404
  to the submitter while our service answered 200. **Do not spend the one-shot
  evaluation while the service is behaving like that.**

### Latency to the evaluator costs score through the CAMERA, not through frames

Found 19 Sep after moving to a second rented box. The evaluator is **Hetzner
Helsinki** (`46.62.240.126`, visible in `serve.log`). A camera command has a
*tighter* deadline than a frame does: it must arrive before the evaluator
renders the next frame, not merely inside the 333 ms budget. Miss it and the
camera does not move, two consecutive frames arrive at the identical view, and
the sweep loses coverage.

| box | Helsinki RTT | stalled camera steps | score |
|---|---|---|---|
| first box | ~30 ms | 0.4 % / 1.2 % / 3.6 % | 0.5027 / 0.5065 / 0.5113 |
| Bulgaria | 58-89 ms | **5.7 % / 7.8 %** | **0.4569 / 0.4235** |

The Bulgarian box was *not* short of bandwidth (47.7 MB/s from Helsinki, 10x
what the 1.5 MB-per-frame stream needs), *not* slow to answer (47-50 ms median,
flat across the whole run, including the last frames) and on the second run not
even short of frames (248/249). It simply could not get camera commands back in
time, and that alone cost **0.05-0.08**.

**Before spending any validation attempt on a new host**, run the gate in
`tools/bootstrap_remote.sh`: 100 TCP connects to `hel1-speed.hetzner.com`,
reject if the median is over 35 ms or the p95 is more than 2.5x the median.
Jitter matters as much as the median -- one late packet is one missed deadline.
Prefer hosts near Helsinki (Estonia, Finland, Sweden) and prefer server
hardware over consumer boards, which are usually residential lines.

This also re-reads the old "a rented box scores ~0.02 lower than the Mac" note:
part of that was the Cloudflare tunnel, and part was almost certainly where the
box was. A count of repeated consecutive views is the diagnostic; it takes
seconds and needs only the recording.

### Choosing a serving host: BOTH latency and sustained bandwidth, measured

Three rented hosts were tried on 19 Sep afternoon and two failed, for different
reasons, and neither failure would have been caught by any check that existed at
midday. Both gates are in `tools/bootstrap_remote.sh`; run them **before
uploading anything**, and destroy a host that fails either.

| host | Helsinki RTT | sustained 100 MB from Helsinki | real score |
|---|---|---|---|
| Czechia `datacenter:214845` | 30.4 ms (p95 38.3) | 83 / 88 / 38 MB/s | **0.5027-0.5113** |
| Bulgaria | **58-89 ms** | 47.7 MB/s | 0.4235 / 0.4569 |
| Estonia | 8.0 ms | **23 -> 16 -> 9 MB/s** | **0.1945** |

* **Bulgaria** had bandwidth to spare and answered every frame in 47-50 ms, and
  still lost 0.05-0.08 -- purely to camera stalls (see the previous section).
* **Estonia** had the best latency available (Tallinn is 80 km from Helsinki)
  and a *shared, contended* pipe. Under the 1.5 MB-per-frame load it collapsed:
  four requests took **3.4-5.8 seconds to reach us**, while our own service
  answered each in 62-80 ms. The evaluator timed them out at 3333 ms, the
  camera never moved, our tracker kept planning from `state.pending` (the view
  we had asked for and never got), and every subsequent move was illegal:
  "center movement 1920.00px exceeds the L1 limit of 1102.00px". Score 0.1945.
  Vast labels that column "Internet Download Speed (**shared**)" -- the
  advertised figure is a ceiling, not an allocation.

Practical rules:
* Prefer a host tagged `datacenter:`. Both failures were non-datacenter, and
  the Finland box never accepted an ssh key at all (neither direct nor via
  `sshN.vast.ai`) and had to be abandoned.
* Latency gate: 30 TCP connects **spaced 200 ms apart**. 100 rapid connects trip
  Hetzner's rate limiting and fake a jitter failure -- that false REJECT nearly
  cost us a good host.
* Bandwidth gate: three consecutive 100 MB pulls. One is not enough; Estonia's
  first read 23 MB/s and only the third exposed the collapse.
* The stream needs 4.5 MB/s sustained (1.5 MB x 3 fps). Anything that cannot
  hold ~20 MB/s under repeat load will stall.

### Latent bug this exposed: a lost answer desynchronises the camera

`choose_next_view` plans from `state.pending` (the view we last asked for)
whenever `camera_command_feedback` is None, because the evaluator renders the
next frame before our answer lands. If our answer is **lost or times out**, the
camera never moves, but we still believe `pending` -- so we plan from a
position the camera is not at and emit moves that exceed the L1 distance limit.
They are refused, which desynchronises further. Not worth fixing while the
endpoint is healthy (it needs a multi-second stall to trigger), but it turns one
lost answer into a cascade, and it is why 0.1945 was so far below even a
frame-loss explanation.

### The endpoint is reproducible: same host, same score

19 Sep 15:13, after two failed hosts, the Czechia `datacenter:214845` machine
(93.91.156.98) was re-rented and the locked config reproduced exactly:
**0.5065, 249/249, 1.2 % stalled camera steps** -- inside the morning's
0.5027 / 0.5065 / 0.5113 at 0.4 / 1.2 / 3.6 % stalls. Host quality is stable
and repeatable when the two gates pass, so a single confirmation run is enough
on a known host rather than three.

Note when reading stall rates: run `tools/stall`-style counting on the RUN's
recording only. A `tools/preflight.py` replay sends frames as fast as it can
and shows ~23 % "stalls" that mean nothing.

### Model agreement does NOT separate real objects from phantoms — measured, failed

`AGREEMENT_WEIGHT=0.7`, three complete runs against four of the locked config
on the same host:

| arm | runs | mean |
|---|---|---|
| locked config | 0.5027 / 0.5065 / 0.5113 / 0.5065 | **0.5067** |
| + `AGREEMENT_WEIGHT=0.7` | 0.4983 / 0.5052 / 0.5017 | 0.5017 |

Inside the noise but consistently below; no run beat the baseline mean. The
hypothesis was that the ranking loss comes from phantom tracks only one model
sees. It does not. **Our three models agree on their false positives** — they
share a training pipeline, an architecture family and the same synthetic-paste
data, so a bush that reads as a tank to v4 reads as a tank to v6 and v8 too.
Ensembling siblings does not diversify errors.

The knob is committed and defaults to 1.0 (no-op). Do not spend runs on
`MISS_PENALTY` or `HITS_BASE` expecting a different answer: all three are the
same idea — demote tracks with weaker evidence — and the evidence signal they
rely on is shared across the ensemble.

**What is still unresolved about the 0.168.** Mean recall 0.661 against mean AP
0.494 is measured and real, but its cause is now open. Two candidates, and they
need different fixes:

1. the false positives are genuinely confident and shared, so only a
   differently-trained model (different backbone, different data) or a
   second-stage verifier would separate them; or
2. a substantial share of the "false positives" are **real objects the truth
   file never listed** — it holds 32 objects mined by our own models, and
   mining is recorded as exhausted. If so the 0.168 is partly an artifact of
   incomplete ground truth and the real headroom is smaller.

Distinguishing them costs labelling, not runs: take the highest-confidence
unmatched detections from a recorded run, crop them from `data/scene`, and look.
That is the first thing to do with offline time, before any further ranking work.

## 2026-09-19, 17:45 CEST: 0.5211 — inference resolution is v9's P2 head, free

**A fourth model at `imgsz 2560` — the same weights already loaded, named twice —
took the mean from 0.5067 to 0.5211**, and took `small_launcher` from **0.000 to
0.512**. That class had been dead in every configuration ever measured here.

```bash
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280,2560 \
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10
```

**Why it works, and why it is exactly the v9/P2 argument.** The P2 recipe's case
is cells-per-object: under ~2 cells nothing is ever detected, over ~4 everything
is. A P2 head at stride 4 halves the source pixels per cell. **So does doubling
imgsz**, and at 2560 the numbers are identical to what the P2 recipe predicts
for itself:

| class | src px | cells @1280 | cells @2560 | cells with P2 @1280 |
|---|---|---|---|---|
| small_launcher | 12.9 | 1.1 | **2.1** | 2.2 |
| spacecraft | 24.5 | 2.0 | **4.1** | 4.1 |
| mine_roller | 26.8 | 2.2 | **4.5** | 4.5 |
| tank | 34.3 | 2.9 | **5.7** | 5.8 |

Same arithmetic, no GPU hours. Cost on a 5090: v8@2560 is 26 ms against v8@1280's
8 ms, taking the served stack from ~21 ms to ~47 ms of a 333 ms budget.

**As an ADDITION, never a replacement.** HANDOVER records v6 alone peaking at
1280 and falling at 1600 (0.398 -> 0.370), which is why nobody tried this. That
measurement is not wrong, it is about a *replacement*: a model run far above its
training scale loses the large classes. As a fourth member of the ensemble it
only has to contribute what the others cannot see, and the object memory takes
the union. This is the same lesson as v8, which failed as a replacement and paid
as an addition.

Cold start: the first run after restarting with four models lost frames 2-5 and
had to be discarded. Four models warm up more slowly -- send a few throwaway
frames (`tools/preflight.py`) before an attempt.

### The resolution ladder has a hard latency cliff, and preflight cannot see it

| stack | compute (5090) | real score |
|---|---|---|
| 960 / 1280 / 1280 | 49 ms | 0.5067 |
| **+ 2560** | **75 ms** | **0.5211** |
| + 2560 + 3200 | **514 ms** | **0.2639** (134/249 frames) |

A fifth pass at 3200 costs ~7x the 2560 one, not the ~1.5x its pixel count
suggests, and blew the 333 ms budget: 87 of 146 answered frames were late and
115 frames were never answered at all.

**`tools/preflight.py` reported "0/11 over budget" on that same configuration.**
It replays frames *sequentially against a warm service*; the evaluator emits
every 333 ms whether or not the previous answer landed, so queueing effects
never appear. Preflight is a configuration check and a sanity check on the
link. **It is not a load test, and a latency result from it means little.**
The honest test of a heavier stack is a validation run, and the cost of being
wrong is one run.

If a fifth pass is tried again, prefer a cheap model: measured on a 5090,
v4@2560 is 15.6 ms and v6@2560 17.5 ms against v8@2560's 26.3 ms.

### The "false positives" are largely REAL OBJECTS nobody labelled

Cropped the 24 highest-confidence answers (conf >= 0.30, one per spatial
cluster) that match neither a confirmed object nor an ignore region, from the
rebuilt 4K scene. Several are unmistakable rendered assets: a helicopter with
visible rotor blades at 0.88, aircraft with cast shadows at 0.84 and 0.74
(more than one in frame), and at least three green launcher vehicles at 0.81,
0.80 and 0.72. 1104 unmatched answers above 0.30 in a single run.

Three consequences, and they matter more than any single arm:

1. **It explains the AGREEMENT_WEIGHT failure.** The hypothesis was that our
   three models agree on false positives. They were not agreeing on errors --
   they were agreeing on **real objects the truth file does not contain**.
   Demoting single-model-unique tracks demoted genuine detections. The measured
   result stands; the explanation given for it at the time was wrong.
2. **The 0.168 "ranking loss" is substantially an artifact of incomplete
   truth.** An unmatched detection scores as a false positive offline and as a
   true positive in the real evaluation -- which is exactly why offline read
   0.494 on the run that really scored 0.5113. Do not spend runs chasing that
   gap, and do not trust per-class AP from `score_offline.py` as a target list.
3. **"Mining is exhausted" (earlier in this file) is wrong.** 24 distinct
   unlabelled objects at >= 0.30 confidence in one run, several of them obvious
   by eye. The earlier mining passes used cross-model agreement at high
   confidence and a truth file mined by v2; today's four-model stack at 2560
   sees much more. `data/unmatched_sheet.png` is the review sheet.

**The practical rule that follows: stop trying to suppress detections.** Every
mechanism aimed at demoting weak tracks -- AGREEMENT_WEIGHT, MISS_PENALTY,
HITS_BASE, the 0.0 floor band's filter -- is attacking a problem that is mostly
measurement error. What has actually paid all day is *detecting more*: box
growth, a lower track threshold, and a fourth model at 2560.

### `DRONE_DET_CONF=0.003` — a wash, do not ship it

Complete runs 0.5278 / 0.5076 (mean 0.5177) against the served 0.01's
0.5187 / 0.5234 (mean 0.5211). Inside the noise and slightly below, with four
times the spread. It produced the single highest run of the day (0.5278) and
also the lowest of its arm — that is variance, not effect.

Worth recording because the *reasoning* was sound and still is: the scene holds
many more real objects than the truth file lists, so faint detections are more
often real than the old analysis assumed. But lowering the floor apparently
displaces as many good answers as it adds real ones — average precision pools
every frame before ranking, so an extra faint box competes globally, not just
in its own frame. 0.01 stays.

This closes the configuration search. Everything tunable without a new model
has now been measured: box growth (flat 1.3), the growth cap (1.3),
`NEW_TRACK_CONFIDENCE` (0.10), the resolution ladder (a 4th pass at 2560),
`DRONE_DET_CONF` (0.01), the camera (`full`), `AGREEMENT_WEIGHT`/`MISS_PENALTY`/
`HITS_BASE` (all suppression, all dead), the floor band, and `RUNNER_UPS`
(saturated). The next real lever is a better detector.

### The served configuration, confirmed on four complete runs

`v4@960 + v6@1280 + v8@1280 + v8@2560`, flat growth 1.3, track-conf 0.10:
**0.5187 / 0.5234 / 0.5320 / 0.5339 — mean 0.5270, sd 0.0072, best 0.5339.**
Compute 72 ms median of a 333 ms budget on a 5090. That is the day's endpoint:
0.4788 -> 0.5270, and `small_launcher` 0.000 -> 0.512.

The sd of 0.0072 across four complete runs is consistent with the 0.005 figure
measured this morning on a three-model stack, so the extra model has not made
the configuration noisier.

### Flat 1.2 growth on the 4-model stack — rejected

One complete run 0.5118 against the served 1.3's four-run mean of 0.5270 (two
of the three runs lost frames, evaluator-side: our service answered every frame
in 67 ms median with nothing over budget, and the Helsinki RTT was 29.7 ms).

The hypothesis was reasonable and is now dead: adding a 2560 pass means many
detections come from a view with 6 source px per grid cell instead of 12, so
the boxes are geometrically more accurate and might need less inflation. They
do not. **1.3 remains the optimum on the 4-model stack**, unchanged from the
3-model one, which makes the growth factor look like a property of the
evaluator's box convention rather than of our detector's precision.

### Ultralytics never overwrites a run directory — it appends a suffix

Restarting a training run with the same `NAME` does **not** reuse the folder.
The aborted first attempt left `/workspace/runs/drone-yolo11m-p2-v9/weights/
best.pt` (22 MB, the nano model from the scale bug above) and the real run wrote
to `drone-yolo11m-p2-v9-2/` (166 MB). A ship script pointing at the first path
copies the wrong checkpoint, and it loads and serves perfectly cleanly.

Check the **size** before serving anything: yolo11m-p2 is ~166 MB, the nano
variant ~22 MB. Same lesson as the scale bug and the stale container: the
dangerous failures here all look like successes.

Also: a liveness check of the form `until ! ssh box 'pgrep -f x'; do ...` fires
a false "process ended" the first time ssh itself fails ("Network is
unreachable"). Distinguish "ssh failed" from "process gone", or the alert cries
wolf.

### The offline truth files, calibrated against 12 runs of KNOWN real score

Scored twelve complete runs (real scores 0.4983-0.5339, spanning today's arms)
against both truth files with `tools/calibrate_truth.py`:

| truth | MAE | bias | Pearson | Spearman |
|---|---|---|---|---|
| `training/scene_objects.json` (66 objects, mined from data/scene) | 0.0736 | **-0.0736** | **+0.894** | **+0.811** |
| `training/validation_objects.json` (32 objects, mined by v2) | 0.0183 | -0.0167 | **-0.039** | **-0.287** |

**Read the correlation column, not the MAE.** The old 32-object file sits close
to the real number and has *no rank correlation at all* over this band — it
cannot tell a 0.4983 run from a 0.5339 one, and is slightly inverted. The
66-object file is offset by a nearly constant -0.074 (range -0.063 to -0.085)
and ranks correctly.

So: **use `scene_objects.json` for comparing configurations, and add ~0.074 to
read it as a real score.** Retire `validation_objects.json` for model and
config selection. This is a stronger statement than the earlier "+0.94 Spearman
after the ignore-region fix" in this file: that was measured across configs
spanning 0.24-0.30, where anything ranks; these twelve runs span 0.036 of real
score, which is the band that actually matters now.

Caveat: the 66-object file has no `spacecraft`, `small_launcher`, `condor` or
`medium_plane`, so it cannot speak for four classes -- two of which are our
weakest. It does contain `medium_launcher` and `ta-ta`, which the old file had
none of.

### The calibration set is in the repo

`drone-flyby/data/runs_20260919.tar.gz` (6.3 MB, 49 MB unpacked) holds the
per-frame ANSWERS of all 31 runs from 19 Sep, plus the serve logs. Twelve of
them have known real validation scores, hardcoded in
`tools/calibrate_truth.py`, and that is what calibrated the truth files above.

Unpack with `tar xzf data/runs_20260919.tar.gz` from `drone-flyby/`. It is
force-added past the `data/` gitignore on purpose: **any new truth file must be
re-calibrated against these before its numbers are trusted**, and mining the
four classes the 66-object file is missing (spacecraft, small_launcher, condor,
medium_plane) is the next session's first task. Without this set there is
nothing to calibrate against.

The view PNGs are NOT included (5 GB, and the answers are what the scorer needs).

### Per-class confidence calibration cannot change mAP — arithmetic, verified

NEW_PLAN §4b lists "confidence calibration across classes (Platt/isotonic on the
mined truth)" as the cheapest of the ranking hypotheses and the "+0.1
candidate". **It is a no-op.** COCO AP is computed per class and depends only on
the *order* of that class's detections; Platt and isotonic calibration are
monotonic, so they preserve that order exactly.

Verified on run `b5544ad3` against the mined truth — three transforms, all
identical to six decimal places:

| transform | mAP |
|---|---|
| baseline | 0.451083 |
| every score squared | 0.451083 |
| `tank` scores x0.1 | 0.451083 |
| sqrt on `tank` and `small_plane` | 0.451083 |

**What would help is changing the within-class ORDER**, not rescaling it. That
needs a signal separating a real detection from a false one, and the three we
have -- model agreement, miss count, hit count -- were all measured dead.

### `tank` is a genuine false-positive problem, unlike the other classes

Cropped the 18 highest-confidence `tank` answers (>= 0.25) matching no confirmed
tank, from `data/scene`. Almost all are **empty terrain**: bare grass, dirt,
vegetation, shipping containers, and boats in a marina. The single highest, at
**0.80, is bare ground**. Exactly one of eighteen looks like a real military
vehicle.

This is the opposite of the general unmatched-detection finding
(`data/unmatched_sheet.png`, commit 356ebb5), where most high-confidence
unmatched boxes were real unlabelled objects. **The two results are consistent
and both matter:** globally, suppressing weak detections fails because they are
mostly real; for `tank` specifically the outranking boxes are genuinely
spurious, which is why `tank` sits near zero AP while carrying 0.083 of the
macro average on its own.

It is a training-data problem -- the detector has learned "dark blob on terrain"
-- and it belongs to v10, not to any serving-time knob.

## 2026-09-20, 01:15 CEST: 0.5380 — the row0 camera, with v9 along for the ride

**New best: mean 0.5380 over three complete runs (0.5341 / 0.5387 / 0.5411),
against the 4-model control's 0.5270 over five (0.5187 / 0.5234 / 0.5272 /
0.5320 / 0.5339). +0.0110, and NO OVERLAP** -- the worst row0+v9 run beats the
best control run. Same host, same night.

```bash
DRONE_CAMERA=row0 \
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt \
DRONE_IMGSZ=960,1280,1280,2560,1280 \
DRONE_DEVICE=cuda DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 python3 api.py
```

80 ms median of a 333 ms budget, 0 of 772 frames over.

### The gain is the camera, not v9 — and v9's own targets got worse

Per-class recall on the best run against the 32 confirmed objects (the *recall*
column of that file is sound even though its AP column is not -- see the
calibration section):

| class | control | row0+v9 | |
|---|---|---|---|
| mine_roller | 0.67 | **0.85** | +0.18 |
| helicopter | 0.51 | **0.65** | +0.14 |
| jet_plane | 0.93 | 0.97 | +0.04 |
| small_plane | 0.76 | 0.79 | +0.03 |
| **spacecraft** | 0.70 | **0.50** | **-0.20** |
| hangar | 0.91 | 0.76 | -0.15 |
| tank | 0.71 | 0.59 | -0.12 |
| small_launcher | 0.03 | 0.00 | -0.03 |

**v9 was trained with `spacecraft=2.0` and `small_launcher=1.5` as its two
highest class weights, and both went DOWN.** The bar for shipping it was
explicitly "read the per-class column for those two". By that reading v9 is not
paying; the gain is `mine_roller` and `helicopter`, which is the camera. This
matches the simulation that preceded it (row0 +0.0082, v9 +0.0023).

**So the open question is `row0` WITHOUT v9** -- likely as good, with one fewer
model, ~15 ms less latency and less failure surface on a one-shot attempt.
Three complete runs decides it. It is the first thing to do with spare runs.

### 2026-09-20, 01:30 CEST: LEVEL_WEIGHT[0] retuned for the row0 camera

**`DRONE_LEVEL0_WEIGHT=1.5` is the live candidate**, one complete run at 0.5432
plus a 247-frame run at **0.5681** (the highest number this project has seen,
discarded under the frames rule but recorded here).

**Why it was worth retuning.** `LEVEL_WEIGHT[0]` weights a Level-0 sighting's
class votes. It was set to 1.0 on runs containing **two or three L0 views**, and
this file's own note says the measurement was noise-dominated and that 2.0 was
rejected for that reason. `row0` changes the regime completely:

| camera | L0 views | L1 views |
|---|---|---|
| `full` (old served) | **2** | 247 |
| `row0` | **67** | 182 |

A 33x increase. The parameter went from nearly irrelevant to governing a quarter
of all sightings, and it had never been tuned in that regime. This is the same
pattern that paid for `NEW_TRACK_CONFIDENCE` (+0.013) and flat box growth
(+0.017): **a threshold set against a configuration that no longer exists.**

### `MISS_PENALTY` — measured, negative, and the third suppression failure

`DRONE_MISS_PENALTY=0.85` on the identical `row0` + LW1.5 base:

| arm | complete runs | mean | all runs | mean |
|---|---|---|---|---|
| row0 + v9, LW 1.5 | 0.5432 | **0.5432** | 4 | 0.5483 |
| + MISS_PENALTY 0.85 | 0.5253 / 0.5322 | **0.5288** | 3 | 0.5322 |

Both ways of counting agree it costs ~0.01-0.015.

**The reason it was tried, and why the reasoning was wrong.** After
`AGREEMENT_WEIGHT` failed it was argued that misses are a *different* signal --
agreement asks which models saw a track, misses asks whether the camera looked
closer and stopped seeing it -- and that `row0`'s 67 L0 views would make
L0-artifact false positives common enough for the signal to bite. The
measurement says the two are not different enough to matter.

**Three suppression mechanisms have now failed for the same reason**:
`AGREEMENT_WEIGHT` (-0.005), the 0.0 floor band (+0.0015), `MISS_PENALTY`
(-0.013). Most of what looks like a false positive is a **real object the truth
file does not contain** (`data/unmatched_sheet.png`), so any rule that demotes
weakly-supported tracks demotes real detections too. `HITS_BASE` is the last
member of this family and should be assumed dead without spending runs on it.

**The exception is `tank`**, whose unmatched boxes really are spurious (bare
grass, dirt, containers, boats -- see the tank section above). But no *global*
suppression rule can separate it, because the same rule hits the eleven classes
where the unmatched boxes are real. Fixing `tank` needs training data, not a
serving knob.

### `LEVEL_WEIGHT[0]` swept on real runs: 1.5 is the peak

With the `row0` camera (67 Level-0 views per run against `full`'s 2), five runs
at each setting:

| `DRONE_LEVEL0_WEIGHT` | all-runs mean (n=5) |
|---|---|
| 1.0 (the old served value) | 0.5398 |
| **1.5** | **0.5519** |
| 2.0 | 0.5436 |

Nearly symmetric either side of 1.5, which is what an optimum looks like rather
than noise -- the same shape the box-growth sweep produced at 1.3. **Bracketed
on both sides; stop here.**

Counted on complete runs only the picture is the same but thinner (LW 1.0
0.5380 over 3, LW 1.5 0.5432 over 1), because the evaluator was dropping frames
on roughly two runs in three during this sweep. That loss is evaluator-side and
random -- our service answered every frame it received in 78 ms median with
nothing over 1000 ms, and the link measured 29.6 ms and 88-91 MB/s throughout --
so all-runs means are the honest comparison here, and both agree.

Note the mechanism: `LEVEL_WEIGHT` scales class *votes*, not detections. Raising
L0's weight does not add boxes; it changes which class a track resolves to when
L0 and L1 sightings disagree. At imgsz 2560 an L0 view has the same
cells-per-object as an L1 view had at 1280, so L0 sightings are about as
informative as L1 ones -- and were being counted at 1.0 against L1's 0.8.

### The night's remaining arms: one wash, two negatives

All on the `row0` + 5-model + LW 1.5 base, all-runs means (the evaluator dropped
frames on roughly two runs in three during this window, evenly across arms):

| arm | n | mean | verdict |
|---|---|---|---|
| **base (LW 1.5)** | 5 | **0.5519** | the reference |
| `UNSEEN_DECAY=1.0` | 5 | 0.5563 | **wash** -- t=0.41, and its only complete run (0.5415) is below the base's (0.5432). Higher variance too (sd 0.0198 vs 0.0140), which is the wrong direction for a one-shot attempt. Keep 0.97. |
| `MISS_PENALTY=0.85` | 3 | 0.5288 | **negative**, ~-0.013 |
| `RUNNER_UPS=1` | 3 | 0.5486 | **negative**, and it killed a theory -- see below |

### The boxes-per-frame correlation was confounded

Across ten runs spanning two arms, score correlated with boxes emitted per frame
at **Pearson -0.655** -- the three highest-scoring runs averaged 80.7 boxes, the
three lowest 83.5. With the stack now emitting ~81 boxes/frame against the old
config's 13, `RUNNER_UPS=4` looked like a stale threshold in a 6x-changed
regime, and cutting it to 1 was the obvious test.

It cut boxes to ~67 as intended and **the score went down** (0.5486 against
0.5519; complete runs 0.5381 against 0.5432). The correlation does not survive
intervention -- it was trajectories that track fewer objects scoring differently
for other reasons, not box volume causing the score. **Do not re-derive it.**

This also resolves an apparent contradiction with the three suppression
failures. Those demoted *whole tracks*, which are mostly real objects the truth
file lacks. `RUNNER_UPS` demotes *duplicate class guesses on one track*, which
is the one place a false positive really is false -- and it still did not help.

### Timeouts cluster on early frames, not on Level-0 views

One run returned 0.2704 with eight `did not answer within 3333 ms` errors, all
on Level-0 requests. L0 is only 27 % of frames, so that looked causal. It is
not:

* service time L0 median 81 ms / p95 100 / max 171, against L1 76/83/92 -- both
  far inside the 333 ms budget, and **the service answered every frame it
  received**;
* the L0 request payload is only **1.09x** an L1 one (1114 KB vs 1026 KB);
* the timed-out frames were 1, 3, 8, 11, 18, 20, 26, 36 -- **all early in the
  run**, and `row0` opens with L0 views.

It is the same cold-connection TCP slow-start that costs frame 2 on roughly one
run in four. Nothing on our side fixes it; do not spend runs chasing it.
