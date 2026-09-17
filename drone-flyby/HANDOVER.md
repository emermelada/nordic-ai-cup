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
