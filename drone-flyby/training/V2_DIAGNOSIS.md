# v1 -> v2 diagnosis and results

For whoever (human or Claude) picks this up next. Read this before doing anything else in `drone-flyby/`.

## The bug in v1

`v1` (commit `24b5cf3`) was trained with:
```bash
python training/make_dataset.py --scenes 800 --out /tmp/yolo
```
No `--backgrounds` flag. Every one of the 800 synthetic scenes pasted objects onto the **same 25 Helsinki reference frames**. The model had every incentive to key off Helsinki-specific background texture/context rather than learning the object classes in general. This is the most likely reason `v1` scored so low on the real validation set (`0.0967`) despite a genuinely sophisticated tracking/camera system in `flyby.py`.

## The fix in v2

`v2` (commit `82d5bf7`, message: "train on aerial backgrounds") changed the call to:
```bash
python training/make_dataset.py --scenes 1200 --out /tmp/yolo --backgrounds /kaggle/input
```
with two external Kaggle datasets added as inputs: `sagar100rathod/inria-aerial-image-labeling-dataset` (cities) and `adrianboguszewski/landcoverai` (countryside/water). Also bumped epochs 40 -> 50. Trained on Kaggle (GPU T4x2), output downloaded as `drone-yolo11n-v2.pt`.

## Results

| | v1 | v2 |
|---|---|---|
| **Local Helsinki mAP@0.50** (`local_evaluator.py`) | not recorded here | **0.782** |
| **Real validation score** (cases.nordicaicup.com) | 0.0967 | **0.108** |

**Read this gap carefully before celebrating the 0.782.** Helsinki mAP jumped enormously, but the real validation score barely moved (+12% relative). `make_dataset.py` still uses Helsinki frames as backgrounds for `HELSINKI_SHARE = 0.2` of scenes, and every object patch is still sourced from Helsinki no matter what the background is - so the model may still be substantially fit to Helsinki-specific object appearance, not just background. **The background-diversity fix was real and worth keeping, but it was not the whole problem.** Something else is still costing most of the real-world score. Do not assume backgrounds alone will get v3 much further.

## Per-class weak spots (measured locally on Helsinki, v2)

Scored exactly `0.000`, and this lines up precisely with how few source patches `extract_patches.py` found for them:

| Class | Patches extracted | Helsinki AP@0.50 (v2) |
|---|---|---|
| hangar | 3 | 0.000 |
| medium_plane | 4 | 0.000 |
| medium_launcher | 5 | 0.000 |

Everything else with 9+ patches scored 0.77-1.00. This is very likely a real data-scarcity ceiling, not a training bug - these three classes have almost no visual variety to learn from regardless of background diversity. Worth targeting specifically in v3 (more aggressive augmentation just for these three, or find a way to source more examples), since fixing them is probably higher-leverage than another general background pass.

## What's still unexplained

The Helsinki-vs-validation gap (0.782 vs 0.108) is the open question. Candidates worth checking, roughly in order of how cheap they are to check:
1. **Confidence/tracking calibration** - `flyby.py`'s `DETECTION_CONFIDENCE`, `MATCH_IOU`, `UNSEEN_DECAY` etc. were tuned (if at all) against Helsinki behavior, which may not transfer. See `training/replay_recorded.py` below.
2. **Real validation scene differs from Helsinki in ways synthetic data doesn't cover** - different lighting, object density, altitude consistency, terrain type. No way to check this without seeing recorded validation frames.
3. **The 20% Helsinki-background share in `make_dataset.py`** may still be enough to bias the model. Try `--backgrounds` with `HELSINKI_SHARE` lowered or zeroed and see if it moves the real score, not just the (misleading) Helsinki-measured one.

## New tool: `training/replay_recorded.py`

Not yet run against real data - I had no recorded traffic available on this side. Set `DRONE_RECORD_DIR` (see `api.py`) before a validation attempt to capture it, then:
```bash
python training/replay_recorded.py <record_dir>
python training/replay_recorded.py <record_dir> --conf 0.02 0.05 0.1 0.2
```
It replays real recorded frames through the actual `detect()` / `update_tracks()` / `annotations_for()` pipeline offline, sweeping hyperparameters, and reports detection/track behavior per setting (no ground truth available for recorded traffic, so no mAP - but sane-vs-degenerate behavior is usually visible without one). This is the fastest way to find out whether item 1 above (calibration) is actually the problem, before spending more Kaggle GPU hours on item 3.

## Also ruled out, don't re-check these

- **Model loading**: confirmed working, logs show `Loaded /models/drone-yolo11n-v1.pt` / `v2.pt` cleanly on both versions.
- **Class ordering**: generated `data.yaml` matches `dtos.OBJECT_CLASSES` exactly, index-for-index. Verified directly, not assumed.
- **`extract_patches.py` / `make_dataset.py` themselves**: both run cleanly end to end, no errors, correct output shape (235 patches across 16 classes for extraction; 5 scenes -> 30 views confirmed for generation, matching `VIEWS_PER_LEVEL`).

## Deployment note

v2 is currently live on this laptop, `docker compose` (not raw `docker run`), weights at `~/models/drone-yolo11n-v2.pt`, `DRONE_MODEL_FILE=drone-yolo11n-v2.pt` set at compose time. If you deploy elsewhere, remember that env var - the compose default falls back to `drone-yolo11n-v1.pt` if unset.
