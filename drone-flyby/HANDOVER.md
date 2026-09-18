# Drone Flyby — where the work stands (2026-09-17, evening)

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
| **v4 (served)** | **0.1445, 0.1425** | + Poisson-blended pasting, more objects, weak classes weighted |

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

**macOS with the Apple GPU — do not use Docker**, it cannot reach Metal:

```bash
cd drone-flyby
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DRONE_MODEL=models/drone-yolo11n-v4.pt DRONE_DEVICE=mps .venv/bin/python api.py   # port 9053
```

`DRONE_DEVICE=mps` has not been tested yet: check that it starts, then measure
`python local_evaluator.py --realtime` (needs `src/helsinki/` from the official
repo) and read "round trip ms" and "frames skipped".

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
