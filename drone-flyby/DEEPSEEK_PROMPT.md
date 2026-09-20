You are a senior machine-learning researcher and engineer. I need a big, realistic improvement on a competition-style problem called "drone-flyby". Read everything below, then do the work in the order given.

# 1. Goal

Our current best validation score is **0.5339**. The mean over four complete runs of the same configuration is **0.5270**, sd 0.0072; an earlier configuration scored 0.5113. We need **0.90**, which is almost double. Incremental tuning will not get us there, so I want you to find out why we are at ~0.53 and to propose an approach that can plausibly reach 0.9. If you think 0.9 is not realistic, say so and tell me what the ceiling likely is and why.

Context on 0.90: the public leaderboard for the same validation flight shows other teams at 0.908, 0.844, 0.841, 0.776, 0.749, 0.690, 0.680 and 0.650. None of them has published code. Only one completed **evaluation** attempt per team counts for the final ranking, and it runs on a **different** 250-frame flight. So a method that only memorises the validation flight will not help us.

**Time left: about 19 hours** (deadline 20 Sep 2026 16:00 CEST; it is 19 Sep ~21:00 CEST as I write). Anything you propose must be buildable, trainable and validated inside that window by a small team with one rented RTX 5090. Assume a training run of at most ~2-3 GPU hours, and a handful of real validation runs.

# 2. Facts I know for certain

- Use case name: drone-flyby (Nordic AI Cup 2026, organizer Ambolt).
- The system is scored by submitting an HTTP service that exposes a /predict endpoint. The scorer calls it and returns a score.
- Validation run (type: validation): status "done", score 0.5112888609815686, no errors reported. The whole run took about 83 seconds (12:49:51 to 12:51:14 UTC on 2026-09-19).
- Later runs (same day, 4-model stack, config in section 3): 0.5187 / 0.5234 / 0.5320 / 0.5339.
- So the service works end to end (no errors). The problem is model quality, or a mismatch between what we predict and what the metric rewards, not crashes.
- The validation endpoint returns **only the overall score**. There is no per-class breakdown and no ground truth for the validation flight.

# 3. What I have not told you yet (I filled these in)

## 3.1 TASK DESCRIPTION

### The scene
A simulated survey drone flies a **straight line 600 m above a rendered landscape**. The frame metadata looks like AirSim/Unreal ("GimbalCamera", NED pose `z: -600`). The terrain is a real-world orthophoto (fields, forests, city blocks, marinas, industrial estates). On top of it sit **rendered 3D assets of 16 classes**, all static props (no moving objects measured):

`hangar, helicopter, jet_plane, large_launcher, large_tower, medium_launcher, medium_plane, mine_roller, small_launcher, small_plane, small_tower, ta-ta, tank, condor, jammer, spacecraft`

- Source frames are **3840×2160**, 3 frames per second. The camera looks forward-down, so the image is a perspective view of flat ground.
- Measured on the validation flight, the whole flight is **one constant homography** between consecutive frames (median SIFT residual 0.36 px). Ground scrolls top-to-bottom at 54 px/frame at the top, 68.4 at the centre and 82.9 at the bottom, spreading ±12.9 px/frame horizontally at the edges.
- An object enters at the top edge and crosses the frame in ~31 frames. Its box grows as it moves down (perspective).
- Typical object sizes in source pixels (median official box, sqrt(w·h)): condor 170, hangar 154, large_launcher 128, helicopter 104, jet_plane 80, large_tower 63, small_tower 59, mine_roller 56, medium_plane 52, tank 49, small_plane 46, spacecraft 46, medium_launcher 46, jammer 38, small_launcher 26, ta-ta 23.
- The organizer says the evaluation flight's **direction is "not necessarily" the same** as validation's, and that object appearance "from above might differ".

### What /predict receives (one POST per frame)
We never get the 4K frame. We get **one 960×540 PNG** of wherever our camera points:

| resolution level | source region | downsample | share of the frame |
|---|---|---|---|
| 0 | 3840×2160 (whole frame) | ÷4 | all |
| 1 | 1920×1080 | ÷2 | a quarter |
| 2 | 960×540 | native | a sixteenth |

At Level 1 a tank is ~17 px and a small_launcher ~6-13 px. At Level 0 most objects are 3-25 px.

Example request (image shortened):
```json
{
  "sequence_id": "2c09583d65c34d7db47b32c66c375b02",
  "frame": 42, "frame_index": 41,
  "request_id": "2c09...:41:1:1920:540",
  "frame_interval_ms": 333, "response_timeout_ms": 3333,
  "original_width": 3840, "original_height": 2160,
  "view": {
    "resolution_level": 1, "center_x": 1920, "center_y": 540,
    "view_id": "2c09...:41:1:1920:540",
    "image": "<base64 PNG, 960x540, no data: prefix>",
    "image_media_type": "image/png", "width": 960, "height": 540,
    "source_region_xyxy": [960, 0, 2880, 1080]
  },
  "camera_constraints": {
    "maximum_center_delta": 1102.0,
    "allowed_resolution_levels": [0, 1, 2],
    "center_bounds": [
      {"resolution_level": 0, "width": 960, "height": 540, "minimum_center_x": 1920, "maximum_center_x": 1920, "minimum_center_y": 1080, "maximum_center_y": 1080},
      {"resolution_level": 1, "width": 960, "height": 540, "minimum_center_x": 960, "maximum_center_x": 2880, "minimum_center_y": 540, "maximum_center_y": 1620},
      {"resolution_level": 2, "width": 960, "height": 540, "minimum_center_x": 480, "maximum_center_x": 3360, "minimum_center_y": 270, "maximum_center_y": 1890}
    ],
    "full_view_reset_exempt_from_delta": true
  },
  "camera_command_feedback": null
}
```

### What /predict returns
Boxes for the **entire current source frame**, not just the view, normalised to the full 3840×2160 frame. Up to 500 annotations. Optionally, where to point the camera next.
```json
{
  "request_id": "2c09...:41:1:1920:540",
  "frame": 42,
  "annotations": [
    {"object_id": "tank", "bbox": [0.5479, 0.4167, 0.5629, 0.4317], "confidence": 0.87},
    {"object_id": "hangar", "bbox": [0.10, 0.20, 0.18, 0.26], "confidence": 0.64}
  ],
  "requested_view": {"resolution_level": 1, "center_x": 2880, "center_y": 540}
}
```
Strict validation rules:
- `0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1`. One bad box rejects the whole response.
- Class names must match exactly.
- `request_id` and `frame` must be echoed back.
- `requested_view` must use integers.

### Camera rules
- The camera starts at Level 0.
- Level changes go one step per frame: 0↔1↔2, so Level 0 and Level 2 cannot reach each other directly.
- A Level 0 request must use centre (1920, 1080), and it is exempt from the distance limit.
- Otherwise the centre may move at most 2203 / 1102 / 551 px per frame, from the current level 0 / 1 / 2.
- An illegal command is ignored, and the detections in that response are still scored.
- The camera moves at most once per frame.
- **Timing matters for the camera:** the evaluator renders frame t+1 when it is emitted, often before our answer to frame t arrives. So a command lands one frame late, and a late answer means the camera does not move at all.

### Timing
- Frames are emitted every 333 ms whether or not we answered.
- Only the newest frame is ever sent, so a slow server skips frames.
- A skipped frame is scored with no detections, and its ground truth still counts.
- Each request has a 3333 ms timeout.
- We measured that 2 missing frames out of 249 cost up to −0.026, and that a serving host 60-90 ms from the evaluator (which sits at Hetzner Helsinki) loses 0.05-0.08 through camera commands arriving too late.

### Metric (exact)
COCO **mAP at IoU 0.50**, computed with `faster_coco_eval`:
- Each frame is an image.
- Ground truth is **every object in the whole source frame**, whatever the camera was showing, including objects cut by the frame edge (their boxes are clipped).
- `iouThrs=[0.5]`, area range "all", **maxDets 100 per image per category**.
- AP per class is the mean of the 101-point interpolated precision, **pooled over all frames**.
- The final score is the **unweighted mean over the classes present in the ground truth**. Classes absent from the ground truth are ignored, so predictions for them cost nothing.
- There is no NMS on the scorer side: duplicates count as false positives.
- **The official boxes are the projected 3D bounding box of each object**: loose, including rotor span, wingtips and height. They are not tight silhouettes. We learned this the hard way (section 3.3).

### Data
- **Supplied labelled data: only 25 frames** (Helsinki, 3840×2160 PNG, 600 m, 13.89 m between frames). One instance of each of the 16 classes, 259 boxes in total. The Helsinki scene is not the validation set.
- **Validation flight:** 249 frames, deterministic (the same views are byte-identical across runs). We may record it; the rules explicitly allow this. We have ~30 recorded runs, and we rebuilt all 249 full 4K frames at 98.8 % coverage (93.7 % at Level 1 or better).
  - There are **no labels** for it. We hand-confirmed **32 objects** (12 classes), mined with our own detectors. The box sizes are therefore our detector's (too tight).
  - We know this list is **incomplete**: in one run, 24 more obvious rendered objects appeared among our "false positives" at confidence ≥ 0.30.
  - Four classes (condor, ta-ta, medium_plane, medium_launcher) have **never** been confirmed in validation.
  - The second half of the flight is mostly dense city and marina with few objects.
- **Evaluation flight:** a different 250-frame flight, one attempt, unknown class mix.
- **Training data** is 100 % synthetic, from `training/make_dataset.py` (pasted below):
  - ~235 GrabCut cut-outs of the 16 Helsinki objects, plus 288 cut-outs harvested from validation recordings of the 32 confirmed objects.
  - Pasted onto backgrounds, rotated, scaled, recoloured, sometimes Poisson-blended or given a fake shadow. The backgrounds are Inria Aerial and LandCover.ai photos, the 25 Helsinki frames, and 44 "clean" 4K frames of the validation flight with the known objects painted out.
  - Then 6 views are cut per scene (1 at L0, 4 at L1, 1 at L2) exactly like the evaluator: 1600 scenes, 9,600 images of 960×540 per training run.
  - Class weights alter how often each class is pasted.

### Constraints
- Per-request latency: the answer should land well inside 333 ms. Our 4-model stack costs ~72 ms median of compute on a 5090.
- Serving: a rented RTX 5090 (Vast.ai, Czech datacentre, 30 ms RTT to the evaluator). We also have an Apple M4 laptop.
- Training: the same 5090. Rough times: yolo11m, 40 epochs at 1280 px ≈ 1-2 h, plus ~20 min to build the dataset.
- Libraries: anything (we use PyTorch 2.x, Ultralytics 8.4.152 YOLO11, OpenCV, faster_coco_eval).
- Validation attempts: unlimited, one at a time, ~83 s each plus queue. Evaluation: **one**.

## 3.2 CURRENT MODEL

The best run (0.5339; mean 0.5270 over 4 complete runs) is branch `BEST-WORKING-VERSION`, commit `8fd83e9`, served as:

```bash
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280,2560 \
DRONE_DEVICE=cuda DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 python3 api.py
```

A pipeline of three parts, in `flyby.py` (pasted below):

1. **Detect.** Every frame runs **four passes**: v4 (YOLO11n) at 960 px, v6 (YOLO11s) at 1280, v8 (YOLO11m) at 1280 and v8 again at 2560, on the 960×540 view.
   - The 1280 and 2560 passes are upscaled input, which gives small objects more stride-8 cells: small_launcher went 0.000 → 0.512 when the 2560 pass was added.
   - Our own letterbox and **class-agnostic NMS** (IoU 0.5) keep every class score.
   - Detection floor 0.01, at most 100 detections per pass.
   - Boxes are lifted to source-frame pixels through `source_region_xyxy`.
2. **Remember (tracker / world model).**
   - Every detection with confidence ≥ 0.10 becomes or updates a track, stored in source pixels. Weaker ones are answered for that frame only, at half confidence.
   - Each frame, every track is carried forward by an **affine ground-motion model** (`dx = a + bx + cy`, `dy = d + ex + fy`). The model starts from a Helsinki prior and is re-fitted online from our own re-detections.
   - Matching: IoU > 0.2 or intersection-over-smaller > 0.6.
   - Class votes are accumulated per track, weighted by resolution level.
   - Confidence = best detection confidence × a hit-count factor × 0.97^(frames unseen) × 0.5 if truncated.
   - Up to 4 runner-up classes are answered on the same box at scaled confidence.
   - A track the camera looked at 6 times without re-detecting it is dropped.
   - **Reported boxes are grown ×1.3 about their centre** (flat, every class), to approach the official loose box convention.
   - Every alive track is answered in every frame, typically ~50 boxes per frame.
3. **Steer.** A fixed Level-1 sweep of 6 positions (top-left, top-middle, top-right, bottom-right, bottom-middle, bottom-left), one step per frame, planned from the last *requested* view.

Models (all fine-tuned from COCO-pretrained Ultralytics checkpoints, on the synthetic data above):

| model | arch | train imgsz | data / recipe | served at |
|---|---|---|---|---|
| v4 | YOLO11n | 960 | Helsinki + stock aerial backgrounds, validation cut-outs, Poisson blending, class weights ≤ 2.5 | 960 |
| v6 | YOLO11s | 960 | + 44 real validation-flight backgrounds (share 0.6), Helsinki paste scale 0.45-1.00, re-cut masks | 1280 |
| v8 | YOLO11m | 1280 | v6 data, class weights 0.4-2.0 (tank/helicopter/jammer up, hangar/jet down) | 1280 and 2560 |

Training settings for all: 40 epochs, Ultralytics defaults otherwise (mosaic on, `optimizer=auto`, which Ultralytics resolves from the iteration count; ~24k iterations here, so SGD), `scale=0.1, degrees=0.0, flipud=0.5, fliplr=0.5, hsv_h=0.03`, batch 12-16.

**Known remaining defects**, measured by us but not yet fixed in the served code:

1. **The online motion fit is biased.** It converges to 69.35 px/frame at the centre against a true 68.37. That is ~20 px of drift after 20 frames, enough to push tank-sized boxes below IoU 0.5.
   - Registering consecutive views with SIFT + RANSAC gives the true motion from frame 5 on. Replayed offline, that was worth +0.025, but it has **not been deployed**.
   - The online fit also bootstraps from the Helsinki prior, so on a flight in a different direction it would never converge. The evaluation flight may fly a different direction.
2. **The evaluator sometimes re-sends the previous frame's render** under a new frame number (~1 % of frames, measured as pixel-identical overlaps). We place those detections one frame (~70 px) off. Fixing it was worth +0.010 offline.
3. `UNSEEN_DECAY=1.0` instead of 0.97 was +0.014 offline once motion is right, and has not been deployed.

## 3.3 HISTORY OF WHAT WE TRIED

All scores are real validation runs unless marked "offline". Offline means replaying recorded runs through the pipeline against our 32-object truth, which is known to be incomplete and tight-boxed.

| idea | change made | validation score | notes |
|---|---|---|---|
| v1 | cut-outs pasted on the 25 Helsinki frames only | 0.011-0.035 | memorised Helsinki (IoU 0.93 there) |
| v2 | + ~400 real aerial background photos | 0.097 | biggest early jump |
| v3 | + cut-outs harvested from validation recordings, light and shadow variation | 0.132 / 0.119 | |
| camera sweeps (v3) | top / dwell / full0 / quad0 / **full** | 0.108 / 0.117 / 0.119 / 0.126 / **0.130** | `full` Level-1 sweep kept |
| runner-up classes (v3) | 0 / 2 / 4 | 0.125 / 0.130 / 0.132 | kept 4 |
| v4 | + Poisson blending, weak classes weighted up to 2.5× | 0.1445 / 0.1425 | 2.5× lost tank, jammer, spacecraft |
| emit every class on every track at conf 0.01 | floor band | 0.134 vs 0.143 | lost: floor boxes outranked real faint answers |
| `BOX_SCALE=0.8` | shrink reported boxes | **0.017** vs 0.143 | collapse; later explained by the box convention |
| v5 | YOLO11m | offline worse than v4 | not served |
| online motion fit | re-fit ground motion from re-detections | offline +5 pts hit rate | still biased (see 3.2) |
| v4 + v6 | two models alternating frames | 0.2365 | they fail on opposite classes |
| v4 + v6 | both on every frame | 0.2450 | |
| v4@960 + v6@1280 | per-model inference size | 0.3048 | more pixels per small object |
| v7 | YOLO11s trained at 1280 | 0.2611 | worse than v6@1280 |
| v4@960 + v7@1280 | | 0.2405 | |
| hybrid camera | Level-2 acquisition passes + Level-1 coverage | 0.1234 | dead; re-tested after a bug fix: −0.049 |
| **box growth** | reported boxes grown per class toward the official convention (Helsinki-derived factors, cap 1.3) | **0.4618** | largest single gain: the official boxes are projected 3D boxes, our labels were tight mask boxes |
| + v8 as a 3rd model | v4@960 + v6@1280 + v8@1280 | 0.4788 (mean 0.4753) | v8 alone worse; only helps as an addition |
| flat growth 1.3 | same factor for every class | +0.013 (paired run) | per-class profile was measuring mask shape, not box error |
| `NEW_TRACK_CONFIDENCE` | 0.25 / 0.15 / **0.10** / 0.075 / 0.05 | 0.4922 / 0.4981 / **0.5055** / 0.5065 / 0.4842 | cliff below 0.075 |
| agreement weighting | demote tracks only some models found | 0.5017 vs 0.5067 | our "false positives" are largely real unlabelled objects |
| **4th pass v8@2560** | same weights at 2× input | **0.5211 mean** | small_launcher 0.000 → 0.512 |
| 5th pass at 3200 | | 0.2639 | 514 ms/frame, frames lost |
| 5th pass v4@2560 | | 0.4994 | |
| detection floor 0.003 | | 0.5177 vs 0.5211 | wash |
| floor band at confidence 0.0 | all unnamed classes at 0.0 | +0.0015 | not worth it |
| growth cap 1.45 / flat 1.2 | | −0.019 / 0.5118 vs 0.5270 | 1.3 is the optimum |
| **served best** | 4 passes, flat 1.3, track-conf 0.10 | **0.5187 / 0.5234 / 0.5320 / 0.5339** | mean 0.5270 |
| rotation TTA | 0/90/180/270 + NMS | offline −0.025 to −0.087 in pairs | dead |
| WBF / cross-model NMS | instead of concatenating into the tracker | offline +0.004 / −0.009 | dead |
| serving host | Czechia 30 ms vs Bulgaria 60-90 ms vs contended Estonia | 0.51 vs 0.42-0.46 vs 0.19 | camera stalls / timeouts |

Method notes:
- Runs are deterministic given the camera path. Identical view sequences give identical scores to 16 digits, so the run-to-run spread (sd ~0.005-0.007 on complete runs) comes from which camera commands landed in time.
- A run that misses frames is not comparable.

## 3.4 WHAT WE SUSPECT

1. **The detector is the ceiling.** It is trained only on synthetic cut-and-paste. On validation it misses objects that are plainly visible at Level 1: on the earlier 2-model stack, tank recall was ~44 % and spacecraft ~16 %.
   - Its confident false positives land on roofs, containers, boats and bushes, and all four models share them (same pipeline, same data).
   - Measured on the earlier 2-model stack: with every false positive deleted, offline AP rose from 0.405 to 0.557. Yet part of that "false positive" mass is real objects missing from our truth, so we do not know how much is real.
2. **Box convention.** Official boxes are projected 3D boxes; we emulate them with a flat ×1.3 growth. A model trained to output the official convention directly might do better. But on the Helsinki frames our models' boxes already match the official ones (ratio 0.97-1.04), so the remaining under-sizing on validation may come from resolution (smaller objects → tighter boxes).
3. **Synthetic-to-real gap.** Pasted cut-outs have wrong shadows (the real scene has one consistent sun direction), wrong perspective (the real camera is tilted, so tall objects lean toward the top of the frame) and mask artefacts.
   - The only real renders we have are the 25 Helsinki frames and our (unlabelled) validation recordings.
   - We have never trained on real validation views with labels.
4. **Class coverage.** Four classes have no real examples beyond one Helsinki instance each. If the evaluation flight contains them, we will probably score ~0 on each, and each absent class costs 1/N of the score.
5. **Tracking.** Motion bias and re-sent frames (section 3.2) plus the first frames of the flight. The camera sweeps the frame once every 6 frames, so each new object is first seen ~0-5 frames after it enters.
6. **Top teams at 0.85-0.9** may be training on hand-labelled validation views (the flight is deterministic), which would not transfer to the evaluation flight. But we cannot rule out that they simply have a much better detector.

## 3.5 REPO FILES

- GitHub repo: https://github.com/emermelada/nordic-ai-cup. It is **private**, so you cannot open it; the key files are pasted below.
- Best branch/commit: `BEST-WORKING-VERSION` @ `8fd83e9`.
- Pasted in full: `api.py` (the /predict service), `flyby.py` (detector wrapper, tracker, camera), `training/make_dataset.py` (synthetic training data).
- Pasted in part: `training/train_remote.sh` (the training call).
- `dtos.py` and `utils.py` are the organizer's protocol models and coordinate helpers; their content is summarised in 3.1.
- The detector is stock Ultralytics YOLO11 (n/s/m) with 16 classes.
- The file comments contain our own running commentary and old measurements. Some of it is outdated; treat it as notes, not truth, and tell me if you spot bugs.

# 4. Your tasks, in this order

## Step A: Diagnose before prescribing
Before suggesting anything new, tell me the most likely reasons a working system scores only ~0.53 on this kind of problem. Check at least these, and say which apply based on my information above:

- Metric mismatch: are we optimizing something different from what the score rewards?
- Train / validation distribution shift (different terrain, object poses, lighting, shadows, perspective, rendering vs. cut-and-paste, resolution levels).
- Data leakage or wrong validation split during our own development, so our local numbers look better than the official validation.
- Preprocessing at inference time that differs from training time (normalization, resampling, ordering, units, letterbox, image scale).
- Decision thresholds, calibration, class imbalance.
- Model capacity: underfitting versus overfitting. Say what evidence would tell them apart.

List the cheapest experiments (under an hour each) that would confirm or rule each one out.

## Step B: Research what others did
Search the web (Google Scholar, arXiv, Papers with Code, Semantic Scholar, GitHub, Kaggle notebooks and discussions, Reddit, StackOverflow / StackExchange, competition write-ups, IEEE / ACM, technical blogs) for people who worked on the same or a closely related problem, and for what they did to improve.

For this task the relevant areas are:
- small-object detection in aerial / UAV imagery (VisDrone, DOTA, xView, AI-TOD, SAHI / sliced inference, P2 heads, super-resolution);
- sim-to-real and synthetic copy-paste training (Cut-Paste-Learn, domain randomisation, rendering synthetic data from 3D assets);
- self-training / pseudo-labelling on unlabelled target-domain video;
- multi-object tracking with ego-motion compensation (homography / optical flow), track-by-detection;
- active vision / next-best-view camera control;
- post-processing for COCO mAP (score calibration, box refinement, WBF).

For each source that is genuinely relevant, give me:
- Title, authors, year, and a URL that you actually opened. If you did not open it, mark it "UNVERIFIED".
- What they did, and what score improvement they reported, from what baseline to what result.
- Whether it applies to us, and exactly how I would implement it in our /predict service or training pipeline.

Also look for winning solutions from similar challenges (Kaggle, VisDrone, DOTA, xView, Anti-UAV and the like) and explain the techniques that made the biggest jumps, such as pre-trained backbones, augmentation, test-time augmentation, ensembling, pseudo-labeling, and post-processing.

## Step C: Propose a DIFFERENT approach
Use the pasted code only as inspiration for what has already been tried. Do not copy it or just tweak it. Propose at least one completely different algorithm or concept, and at most three, that could plausibly close most of the gap. For each one:

- The core idea in plain language, and why it should beat ~0.53 on this problem.
- Concrete implementation: architecture or algorithm, features, loss, training recipe, working code (Python, runnable, sized for our constraints: one RTX 5090, ≤ ~3 GPU hours of training, ≤ ~150 ms per frame at inference, ~19 hours of total calendar time).
- Expected gain, and the reasoning behind it, not just a number.
- Risks, compute cost, and the fastest way to test whether the idea works.

## Step D: A ranked plan to reach 0.9
Give me a prioritized plan: each step with expected gain, effort, and risk, ordered by gain per hour. Say which single change you would try first and why. Include how to validate locally so our local score predicts the official validation score, and how to avoid overfitting to the official validation set. The final evaluation is a different flight, and we get one attempt.

# 5. Rules

- Do not invent papers, authors, links, benchmark numbers, or results. If you cannot browse, say so at the very start and label everything from memory as "from memory, unverified".
- Separate what you found (with a source) from what you are guessing. Tag claims as [verified], [from memory], or [hypothesis].
- If a key fact is missing and it changes your answer, ask me for it first, in one short list, and give your best provisional answer anyway.
- Read the repo files for ideas only. Do not assume the code is correct. Tell me if you spot bugs.
- Be specific and technical. No generic advice such as "collect more data" or "tune hyperparameters" unless you say exactly what, how much, and what gain you expect.
- Put the final answer in this order: (1) diagnosis, (2) research findings with sources, (3) the new approach and code, (4) the ranked plan to 0.9, (5) open questions for me.

# Appendix: code (branch BEST-WORKING-VERSION @ 8fd83e9)

## `api.py`

The /predict service (FastAPI). Transport only; all logic is in flyby.py.

```python
"""The endpoint the evaluation service calls.

The detector, object memory and camera policy all live in ``flyby.py``; this
module is transport only. (``example.py`` is the untouched original template.)

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import base64
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
import flyby
from flyby import load_model, predict
from utils import validate_response

HOST = '0.0.0.0'
# A rented box only exposes the ports its template mapped -- this one gives
# 8080, not 9053 -- and binding the wrong one means the evaluator cannot reach
# us at all while every local check passes.
PORT = int(os.environ.get('DRONE_PORT', '9053'))

# With a timestamp: serve.log has to be lined up against tunnel.log to tell a
# network stall from a slow model, and the default format has no time at all.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()

# Set DRONE_RECORD_DIR to keep every request (image + metadata) and response.
# The rules allow recording the validation sequence; it is our only look at it.
RECORD_DIR = os.environ.get('DRONE_RECORD_DIR')
_recorder = ThreadPoolExecutor(max_workers=1)


def record(request: DroneFlybyPredictRequestDto, response: DroneFlybyPredictResponseDto) -> None:
    folder = Path(RECORD_DIR) / request.sequence_id.replace('/', '_')
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{request.frame_index:04d}_f{request.frame:04d}'
    (folder / f'{stem}.png').write_bytes(base64.b64decode(request.view.image))
    meta = request.model_dump()
    meta['view'].pop('image')
    meta['response'] = response.model_dump()
    (folder / f'{stem}.json').write_text(json.dumps(meta))


@app.on_event('startup')
def warm_up():
    # Load and warm the model before the first frame arrives. Refuse to start
    # without it: a service that answers 200 with no detections passes every
    # health check and scores zero for a whole attempt.
    if load_model() is None:
        raise RuntimeError(f'No model at {flyby.MODEL_PATH} - check DRONE_MODEL and the mount')
    # Same reasoning for the pair: asking for two models and silently getting one
    # is a working-looking service that quietly serves a different configuration
    # than the one measured. Refuse rather than degrade.
    if flyby.ALT_MODEL_PATHS and len(flyby._models) < 1 + len(flyby.ALT_MODEL_PATHS):
        raise RuntimeError(
            f'Asked for {1 + len(flyby.ALT_MODEL_PATHS)} models, loaded {len(flyby._models)} '
            f'({", ".join(str(p) for p in flyby.ALT_MODEL_PATHS)}) - check DRONE_MODEL_ALT and the mount')


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
async def predict_endpoint(raw: Request):
    """Answer one frame.

    The body is parsed straight from bytes and the answer serialised directly:
    FastAPI's own request/response validation costs several milliseconds on a
    1.5 MB request, and every millisecond is frames we do not skip.
    """
    try:
        request = DroneFlybyPredictRequestDto.model_validate_json(await raw.body())
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={'detail': exc.errors(include_url=False)})
    response = await run_in_threadpool(answer, request)
    return Response(content=response.model_dump_json(), media_type='application/json')


def answer(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    started = time.perf_counter()
    response = predict(request)

    # Check the evaluator's rules, but never fail the request over them: an
    # exception here returns 500 and loses the frame entirely, which is worse
    # than whatever the check found.
    try:
        validate_response(response)
    except ValueError:
        logger.exception('Invalid response for frame %s, salvaging it', request.frame)
        # Drop only what is actually wrong. Emptying the whole frame throws away
        # every good detection over one bad box, and the frame is scored either
        # way, so a single malformed annotation used to cost a whole frame.
        kept = []
        for annotation in response.annotations:
            probe = DroneFlybyPredictResponseDto(
                request_id=response.request_id, frame=response.frame,
                annotations=[annotation], requested_view=None)
            try:
                validate_response(probe)
            except ValueError:
                continue
            kept.append(annotation)
        response.annotations = kept[:500]
        try:
            validate_response(response)
        except ValueError:
            # Then it was the camera command, not the boxes.
            logger.exception('Camera command invalid on frame %s, dropping it', request.frame)
            response.requested_view = None

    if RECORD_DIR:
        # In the background: the frame clock does not wait for the disk.
        _recorder.submit(record, request, response)

    # The per-frame cost, so a bad run can be blamed on the model or the link
    # from this log alone. Budget is 333 ms; the pair answers in ~25 ms on the M4.
    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections in %.0f ms',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        'model': str(flyby.MODEL_PATH),
        'model_alt': str(flyby.ALT_MODEL_PATH) if flyby.ALT_MODEL_PATH else None,
        'models': [str(flyby.MODEL_PATH)] + [str(p) for p in flyby.ALT_MODEL_PATHS],
        'models_requested': 1 + len(flyby.ALT_MODEL_PATHS),
        # 2 means the pair is really alternating; 1 means one model is serving
        # every frame, whatever DRONE_MODEL_ALT was set to.
        'models_loaded': len(flyby._models),
        # One size per loaded model, in order. The served pair runs v4 at 960
        # and v6 at 1280, so a single number here means the sizes did not apply.
        'imgsz': [flyby.size_for(i) for i in range(max(1, len(flyby._models)))],
        'device': flyby.DEVICE,
        # What the answer policy will actually do, so preflight can confirm the
        # served config rather than the one someone meant to serve.
        'box_grow': flyby.BOX_GROW or None,
        'box_grow_cap': flyby.BOX_GROW_CAP,
        'level0_weight': flyby.LEVEL_WEIGHT[0],
        'unseen_decay': flyby.UNSEEN_DECAY,
        'new_track_confidence': flyby.NEW_TRACK_CONFIDENCE,
        # Everything below is here because an arm that runs with the switch off
        # looks exactly like an arm that ran with it on. DRONE_INSPECT in
        # particular had no way to be confirmed from outside the process at all.
        'inspect': flyby.INSPECT,
        'det_conf': flyby.DETECTION_CONFIDENCE,
        'both_models': bool(flyby.BOTH_MODELS),
        'floor_zero': flyby.FLOOR_ZERO,
        'floor_size_tol': flyby.FLOOR_SIZE_TOL if flyby.FLOOR_ZERO else None,
        'floor_all': flyby.FLOOR_ALL_CLASSES or None,
        'miss_penalty': flyby.MISS_PENALTY,
        'agreement_weight': flyby.AGREEMENT_WEIGHT,
        'hits_base': flyby.HITS_BASE,
        'hits_step': flyby.HITS_STEP,
        'runner_ups': flyby.RUNNER_UPS,
        'model_loaded': flyby._model is not None,
        'camera': flyby.CAMERA,
        'recording': bool(RECORD_DIR),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
```

## `flyby.py`

Detector wrapper, tracker/world model, answer policy, camera policy.

```python
"""Detector, object memory and camera policy for the drone flyby.

Three parts, one request at a time:

1. **Detect** with YOLO on the transmitted view and lift the boxes into
   source pixels.
2. **Remember.** Every object ever seen is kept as a track. The ground moves
   predictably between frames (see ``MOTION``), so each track is moved forward
   to the current frame, matched against the new detections, and reported for
   the whole frame even when the camera is looking elsewhere.
3. **Steer.** Walk a fixed pattern of Level-1 views, one step per answered
   frame; every move is checked with the evaluator's own rules first.
   The evaluator renders a frame when it is emitted, usually before our
   previous answer has arrived, so a camera command lands one answer late.
   Moves are therefore planned from the last *requested* view, not from the
   view in the current image.

Configuration, all optional, through environment variables:

    DRONE_MODEL   path to the YOLO weights  (default: ~/models/drone-yolo11n-v4.pt)
    DRONE_MODEL_ALT  second weights taking alternate frames; the two models'
                     detections meet in the object memory, so a run gets the
                     union of what both can find (see detect())
                     DRONE_SET=BOTH_MODELS=1 runs both on every frame instead
    DRONE_DEVICE  torch device               (default: cpu)
    DRONE_IMGSZ   inference size             (default: 960)
                  one size per model, comma separated, to run the pair at two
                  scales: DRONE_IMGSZ=960,1280. The same weights may be named
                  twice, which makes a pair out of one model at two sizes.
    DRONE_THREADS CPU threads for inference  (default: 6)
    DRONE_DET_CONF    lowest detection reported at all       (default: 0.01)
    DRONE_TRACK_CONF  lowest detection remembered as a track (default: 0.25)
    DRONE_CAMERA      sweep pattern: full, top, mixed, hybrid or survey
                      quad0, full0, dwell or survey (data)    (default: full)
    DRONE_SET         NAME=value,... overrides any setting below, for experiments
                      DRONE_SET=MOTION_MIN_SAMPLES=999999 pins the ground motion
                      to the Helsinki fit again: the A/B for the online fit
    DRONE_INSPECT     zoom to Level 2 on small unsure objects (default: 0)
"""

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import (
    center_bounds_for_level,
    clip_bbox_to_frame,
    decode_view,
    describe_camera_rejection,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get('DRONE_MODEL', Path.home() / 'models' / 'drone-yolo11n-v4.pt'))
# A second set of weights, taking alternate frames. See detect().
# One or more extra sets of weights, comma separated like DRONE_IMGSZ:
#   DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt
#   DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt
# Every model's detections meet in the same object memory, so a run sees the
# union of what all of them find. Measured offline on the loose truth with box
# growth on: v4+v6 0.488, v4+v6+v8 0.497. The +0.010 is at the real-run noise
# floor; the better argument for a third model is that v6 and v8 fail on
# opposite classes (v8 helicopter 0.659 against v6's 0.527 and spacecraft 0.107
# against 0.003; v6 hangar 0.871 against v8's 0.505), and the final evaluation
# is a different flight whose class mix is unknown.
ALT_MODEL_PATHS = [Path(v) for v in os.environ.get('DRONE_MODEL_ALT', '').split(',') if v.strip()]
# Kept for the tools and /api, which report a single alternate.
ALT_MODEL_PATH = ALT_MODEL_PATHS[0] if ALT_MODEL_PATHS else None
DEVICE = os.environ.get('DRONE_DEVICE', 'cpu')
# One size, or one per model ("960,1280"). Input size trades big objects for
# small ones: the same v4 weights at 1280 instead of 960 moved tank 0.043 ->
# 0.306 and large_tower 0.001 -> 0.313 offline while hangar fell 0.743 -> 0.435,
# so the two sizes miss different classes the same way two models do, and the
# pair can take one of each. IMGSZ stays the scalar the tools read.
IMGSZ_LIST = [int(v) for v in os.environ.get('DRONE_IMGSZ', '960').split(',') if v.strip()]
IMGSZ = IMGSZ_LIST[0]
# Measured on the i5-8350U: 6 threads 91 ms, 4 threads 110 ms, 8 threads 112 ms.
THREADS = int(os.environ.get('DRONE_THREADS', '6'))
# With two models loaded: 0 alternates them by frame, 1 runs both on every frame
# and concatenates. Alternating saves compute we do not need (the pair answers in
# ~25 ms of a 333 ms budget on the M4) and pays for it in variance: the model a
# frame gets is decided by request.frame, so every skipped frame flips the
# assignment, and offline the same pair scored 0.346 or 0.303 on that choice
# alone. Running both also stops update_tracks charging a miss to an object only
# the other model can see. DRONE_SET=BOTH_MODELS=1.
BOTH_MODELS = 0

# Two thresholds. mAP rewards ranked low-confidence guesses, so anything above
# DETECTION_CONFIDENCE is reported for the frame it was seen in. Only detections
# above NEW_TRACK_CONFIDENCE start a track that is remembered and reported on
# later frames; otherwise false alarms pile up (v1 reached 137 per frame).
DETECTION_CONFIDENCE = float(os.environ.get('DRONE_DET_CONF', '0.01'))
# Validation with v3: 0.10 -> 0.126, 0.25 -> 0.132, 0.40 -> 0.122.
NEW_TRACK_CONFIDENCE = float(os.environ.get('DRONE_TRACK_CONF', '0.25'))
# One-frame guesses rank below remembered objects of the same confidence.
TRANSIENT_WEIGHT = 0.5
MAX_TRACKS = 120
NMS_IOU = 0.5
MAX_DETECTIONS = 100

# Per-frame ground motion in source pixels, fitted on the Helsinki frames.
# The drone flies straight, so every point drifts down and slightly away from
# the centre as the ground gets closer:
#   dx = a + b*x + c*y,  dy = d + e*x + f*y
# This is only the starting guess. It was fitted on one flight, and a flight at
# a different altitude or speed moves the ground by a different number of
# pixels: measured against the recorded validation flight this is ~3 px/frame
# short vertically (66.2 vs 69.4 at the frame centre). That is under 5 %, but it
# compounds every frame, and a carried box that is 15 px out no longer overlaps
# a 37 px object at IoU 0.5 - so a track that is not re-detected for ~10 frames
# stops scoring. Since most answers come from memory rather than the current
# view, the motion is re-fitted from our own re-detections during the run.
MOTION = (-13.62, 0.00708, 0.00007, 51.25, 0.00029, 0.01334)

# Online motion fitting.
MOTION_MIN_SAMPLES = 8          # before that, the Helsinki prior stands alone
MOTION_SAMPLE_MEMORY = 240      # most recent samples kept
MOTION_MAX_GAP = 12             # frames between two sightings used as a sample
MOTION_TRIM = 0.75              # share of samples kept: vehicles move on their own
MOTION_PRIOR_STRENGTH = 12.0    # samples needed to outweigh the prior
MOTION_MAX_CORRECTION = 15.0    # px/frame at the frame centre; beyond this, distrust

# How much a detection at each level is trusted. It scales class votes only
# (add_votes), never a track's confidence, so scaling all three together
# cancels out -- what matters is the ratio between levels.
#
# Level 0 was at 0.4 on the intuition that a coarse whole-frame view is less
# reliable. Measured on five recorded runs (19 Sep), raising it to 1.0 helps in
# every one, and helps most where there is most evidence to judge on:
#
#   run        L0 views   0.4      0.8      1.0
#   5ace5364          3   0.288   +0.004   +0.010
#   8a1d65ee        130   0.333   +0.034   +0.040
#   a1c00d7c         90   0.366   +0.030   +0.039
#   81bf6bd3          3   0.282   +0.002   +0.009
#   04bef8d0          2   0.287   +0.001   +0.005
#
# The served `full` camera takes only 3 Level-0 views, so expect the top row
# (+0.010 offline, inside the real run-to-run noise of +/-0.01), not the +0.04.
# 2.0 was tried and rejected: it scores higher on the three-view runs than on
# the well-sampled ones, which is noise, not effect.
# DRONE_LEVEL0_WEIGHT=0.4 restores the old value for an A/B. It is a separate
# variable because DRONE_SET takes numeric settings only -- it splits on commas
# and rejects anything that is not already an int or float, so passing a dict
# literal there makes the service exit at startup rather than run with the old
# value.
LEVEL_WEIGHT = {0: float(os.environ.get('DRONE_LEVEL0_WEIGHT', '1.0')), 1: 0.8, 2: 1.0}
MATCH_IOU = 0.2
# A track the camera looked at without finding it this many times is dropped.
MAX_MISSES = 6
# Confidence of a remembered track fades with every frame it goes unseen.
UNSEEN_DECAY = 0.97
# Also report up to this many runner-up classes, when their vote is at least
# this share of the best one, at a confidence scaled by that share.
# Validation with v3 + full camera: 0 runner-ups 0.125, 2 (share 0.15) 0.130,
# 4 (share 0.03) 0.132.
RUNNER_UPS = 4
RUNNER_UP_SHARE = 0.03
# Class scores below this are not counted as votes.
MIN_VOTE_SCORE = 0.02
# Every reported box is scaled about its centre by this. Objects in the
# validation flight measure 0.55-0.85x their Helsinki box diagonal, so our
# boxes may be systematically too big for the 0.50 IoU the scorer needs.
BOX_SCALE = 1.0
# Box growth toward the official box convention, applied to REPORTED boxes only.
#
# The evaluator's boxes look like the projected 3D box of each object -- rotor
# span, wingtips and height included -- while make_dataset.py labels every
# pasted cut-out with the tight box around its alpha mask. So our models learned
# tight boxes and are scored against loose ones.
#
# The strongest evidence is already in our own history: DRONE_BOX_SCALE=0.8
# collapsed a real run from 0.143 to 0.017. Shrinking a well-matched box by 20%
# gives IoU ~0.64, comfortably over the 0.50 threshold and worth a few points at
# most. An 88% collapse only happens if the boxes were already sitting just
# above the threshold -- which is what a systematic size mismatch looks like.
#
# Measured on this machine's Helsinki cut-outs (official patch box / tight mask
# box, median per class; training/measure_box_convention.py re-measures):
# scoring an offline replay against truth grown by these factors cuts the mean
# error against six real validation scores from 0.134 to 0.075.
#
#   DRONE_BOX_GROW=helsinki   the isotropic factors below, capped
#   DRONE_BOX_GROW_CAP=1.3    ... at this (default 1.3)
#   DRONE_BOX_GROW=1.2        one factor for every class
#   DRONE_BOX_GROW=tank=1.3,jet_plane=1.5     explicit per class
#   DRONE_BOX_GROW unset      off, the 0.3048 behaviour
#
# NOT confirmed on a real run yet. The offline case rests on a truth file grown
# by these same factors, which cannot be fully independent. One validation run
# decides it, and the effect should be far outside the +/-0.01 noise either way.
# ISOTROPIC, and deliberately so: these are the exact values that scored 0.4618
# on validation (branch BEST-WORKING-VERSION, commit 945e89b). The convention is
# really anisotropic -- small_launcher measures 1.75 wide against 2.14 tall,
# helicopter 1.38 against 1.70 -- and DRONE_BOX_GROW_WH=1 switches to the
# measured per-dimension factors below. That form is NOT validated and showed no
# offline gain (0.488 either way, because cap 1.3 clamps almost everything), so
# it stays off. Do not make it the default without a real run that beats 0.4618.
HELSINKI_BOX_FACTORS = {
    'condor': 1.496, 'hangar': 1.152, 'helicopter': 1.531, 'jammer': 1.078,
    'jet_plane': 1.489, 'large_launcher': 1.115, 'large_tower': 1.078,
    'medium_launcher': 2.302, 'medium_plane': 1.320, 'mine_roller': 1.077,
    'small_launcher': 1.936, 'small_plane': 1.075, 'small_tower': 1.184,
    'spacecraft': 1.104, 'ta-ta': 1.147, 'tank': 1.309,
}
HELSINKI_BOX_FACTORS_WH = {
    'condor': (1.57, 1.43), 'hangar': (1.08, 1.23), 'helicopter': (1.38, 1.70),
    'jammer': (1.07, 1.09), 'jet_plane': (1.41, 1.57), 'large_launcher': (1.23, 1.01),
    'large_tower': (1.07, 1.08), 'medium_launcher': (2.24, 2.37), 'medium_plane': (1.49, 1.17),
    'mine_roller': (1.08, 1.07), 'small_launcher': (1.75, 2.14), 'small_plane': (1.13, 1.02),
    'small_tower': (1.10, 1.28), 'spacecraft': (1.07, 1.14), 'ta-ta': (1.16, 1.13),
    'tank': (1.30, 1.32),
}
BOX_GROW_WH = os.environ.get('DRONE_BOX_GROW_WH', '') == '1'
BOX_GROW_CAP = float(os.environ.get('DRONE_BOX_GROW_CAP', '1.3'))


def _parse_box_grow(spec: str):
    """'helsinki' | '1.2' | 'tank=1.3,...' -> {class: (width factor, height factor)}.

    The cap applies per dimension, so a class stretched in one axis keeps that
    asymmetry up to the cap instead of being averaged away.
    """
    spec = (spec or '').strip()
    if not spec:
        return {}
    cap = lambda f: min(f, BOX_GROW_CAP)
    if spec == 'helsinki':
        if BOX_GROW_WH:
            return {n: (cap(w), cap(h)) for n, (w, h) in HELSINKI_BOX_FACTORS_WH.items()}
        return {n: (cap(f), cap(f)) for n, f in HELSINKI_BOX_FACTORS.items()}
    try:
        both = cap(float(spec))
        return {n: (both, both) for n in OBJECT_CLASSES}
    except ValueError:
        pass
    out = {}
    for item in spec.split(','):
        name, _, value = item.partition('=')
        name = name.strip()
        if name not in OBJECT_CLASSES:
            raise SystemExit(f'DRONE_BOX_GROW: unknown class {name!r}')
        out[name] = (cap(float(value)), cap(float(value)))
    return out


BOX_GROW = _parse_box_grow(os.environ.get('DRONE_BOX_GROW', ''))
# A response may carry 500 annotations and we send ~10, so naming every class
# on every object looked free. It is not: validation with v4 scored 0.134 with
# it at 0.01, against 0.143 without, because those floor boxes outrank genuine
# low-confidence detections of the same class in other frames. Off by default.
# Note for anyone tempted to retry this: average precision pools every frame
# before ranking, so "below the real answers" has to hold across the whole
# flight, not within one response. At 0.01 these boxes land in the same band as
# our own faint detections elsewhere, which is what cost the 0.009. The variant
# that has *not* been measured is a floor strictly under the 0.001 clip - low
# enough that it can never outrank a real answer in any frame. That may be
# worth one run; naming classes at 0.01 is not.
FLOOR_ALL_CLASSES = float(os.environ.get('DRONE_FLOOR_ALL', '0'))
# The floor variant the comment above calls unmeasured, built so that it cannot
# do what FLOOR_ALL_CLASSES did. Every unnamed class gets a box at confidence
# EXACTLY 0.0, which is strictly below every real answer we emit (measured on
# the best v4 run: 3272 answers, none below 0.0015, none at the 0.001 clip).
#
# Probed against faster_coco_eval itself on 19 Sep, because this rests on the
# scorer's exact semantics rather than on an argument about them:
#   * maxDets is 100 PER IMAGE PER CATEGORY, and we emit at most 8 of one class
#     in a frame, so the band has room;
#   * 150 junk boxes a frame at score 0.0 left a perfect class at AP 1.000 -
#     detections ranked strictly below every real one cannot lower AP;
#   * a class with no answers at all went 0.000 -> 0.121 when ten true boxes
#     rode in on such a band.
# Both conditions are load-bearing: tied at 0.0 and listed AFTER the junk, the
# true box was truncated away by maxDets and the class fell back to 0.000. So
# the band is appended after the real answers are sorted, never sorted with
# them, and the 0.0 must not pass through the np.clip(..., 0.001, ...) that
# every other confidence here does.
#
# What it is for: small_launcher was emitted 8 times in 247 frames. Its AP of
# 0.000 is a no-answer problem, and no re-ranking can fix a class we never
# answer for. The score is a mean over the classes present in the truth, so a
# class we never name is a free zero -- which matters more on the evaluation
# flight, whose class mix is unknown, than on validation.
FLOOR_ZERO = os.environ.get('DRONE_FLOOR_ZERO', '0') == '1'
# A track only gets floor boxes for classes whose size is plausible for it:
# emitting hangar on a 30 px track spends precision for nothing, and AP is
# precision at the recall achieved. The tolerance is deliberately loose because
# the validation flight renders objects at roughly 0.5-0.9x their Helsinki size
# -- too tight a filter drops the very class the band exists to answer.
# Lowering it raises the band's precision; 1.8 is the value to try second.
FLOOR_SIZE_TOL = float(os.environ.get('DRONE_FLOOR_SIZE_TOL', '2.5'))
# Median official box size per class, sqrt(w*h) in source pixels, measured on
# the 25 official Helsinki frames (src/helsinki/annotations) on 19 Sep. Used as
# a size prior only, never as a box.
CLASS_SIZE = {
    'condor': 169.5, 'hangar': 154.3, 'helicopter': 104.4, 'jammer': 37.7,
    'jet_plane': 79.5, 'large_launcher': 128.0, 'large_tower': 63.0,
    'medium_launcher': 46.0, 'medium_plane': 52.4, 'mine_roller': 56.4,
    'small_launcher': 25.7, 'small_plane': 46.4, 'small_tower': 58.5,
    'spacecraft': 46.4, 'ta-ta': 23.3, 'tank': 48.5,
}
# Average precision is ranking and nothing else, and two signals the tracker
# already holds never reach the reported confidence: best_confidence is a
# running maximum over sightings that never falls, and `misses` -- the number of
# times the camera looked straight at a track and did not find it -- only ever
# deletes the track at MAX_MISSES. So a track refuted five times is ranked
# exactly as high as one just seen. 1.0 is the served behaviour; 0.85 is the
# value to try.
MISS_PENALTY = float(os.environ.get('DRONE_MISS_PENALTY', '1.0'))
# How much a track is trusted when only SOME of the loaded models have ever
# found it. With BOTH_MODELS every model sees every frame, so a real object is
# normally found by several of them and a false alarm on a bush or a rooftop
# often by one -- and until now detect() concatenated all their detections and
# threw the model index away, so that signal was never used.
#
# Why it matters, measured 19 Sep on the best recorded run: mean recall over
# the twelve scored classes is 0.661 while mean AP is 0.494. We are not failing
# to FIND objects -- large_launcher is detected in 100 % of its frames,
# large_tower 93 %, mine_roller 84 % -- we bury them under our own false
# positives. 0.168 of score is pure ranking loss, concentrated in mine_roller
# (0.398), tank (0.338), small_plane (0.310) and large_launcher (0.275).
#
# base is multiplied by AGREEMENT_WEIGHT once per model that has NEVER matched
# the track, so a track all three models have seen is untouched and a
# single-model track is scaled by AGREEMENT_WEIGHT ** 2. 1.0 is the served
# behaviour; 0.7 is the value to try.
AGREEMENT_WEIGHT = 1.0
# The confidence of a track with few sightings. base *= min(1, HITS_BASE +
# HITS_STEP * hits), so the defaults give 0.8 at one hit and saturate at three.
# Phantom tracks are mostly one- and two-hit tracks, so lowering HITS_BASE
# demotes them; these are the served values.
HITS_BASE = 0.7
HITS_STEP = 0.1
# A box partly outside the view is a guess at the object's size: report it
# lower, and let any whole sighting replace it.
TRUNCATED_WEIGHT = 0.5
# Two boxes are the same object if they overlap this much, or if one covers
# this share of the smaller one (a cut-off half inside the whole object).
MATCH_MIN_COVER = 0.6
# Views whose edge is this close to a box do not count as missing it.
EDGE_MARGIN = 20

RESTART_GAP = 5
MAX_SEQUENCES = 8
# A detection this close to the view edge is probably cut off.
CUT_OFF_PIXELS = 3

# Level-1 sweep patterns; every step is at most 1080 px, inside the 1102 px
# L1 limit, including the step from the last point back to the first.
# New objects enter at the top edge and memory carries them down, so the top
# row matters most once the whole frame has been seen.
TL, TM, TR = (960, 540), (1920, 540), (2880, 540)
BL, BM, BR = (960, 1620), (1920, 1620), (2880, 1620)
FULL_SWEEP = [TL, TM, TR, BR, BM, BL]
TOP_SWEEP = [TL, TM, TR, TM]
SWEEPS = {
    'full': FULL_SWEEP,
    'top': FULL_SWEEP + TOP_SWEEP * 1000,          # one full look, then the top
    'mixed': FULL_SWEEP + TOP_SWEEP * 3,           # repeats: full now and then
    # Each point twice in a row: what the code did before moves were planned
    # from the pending view (the camera moved every second answer).
    'dwell': [point for point in FULL_SWEEP for _ in range(2)],
    # The whole frame every other answer, a Level-1 quarter in between. Level 0
    # is exempt from the distance limit, so the quarters need no middle stops.
    'quad0': [p for corner in (TL, TR, BR, BL) for p in ((0, 1920, 1080), (1, *corner))],
    # The full sweep with a whole-frame look after every second step.
    'full0': [TL, TM, (0, 1920, 1080), TR, BR, (0, 1920, 1080), BM, BL, (0, 1920, 1080)],
}
# Chosen on validation runs with v3 (same flight, same model):
# full 0.130, quad0 0.126, full0 0.119, dwell 0.117, top 0.108.
CAMERA = os.environ.get('DRONE_CAMERA', 'full')
# Every point as (level, x, y); plain (x, y) points are Level 1.
SWEEP = [p if len(p) == 3 else (1, *p) for p in SWEEPS.get(CAMERA, FULL_SWEEP)]

# 'survey': a data-collection pattern, not a scoring one. Level-2 views
# (native resolution) snake along two rows covering the top half, where every
# object enters; steps are at most 550 px, inside the 551 px L2 limit.
SURVEY_ROW_TOP = [(x, 270) for x in (480, 1030, 1580, 2130, 2680, 3230, 3360)]
SURVEY_ROW_LOW = [(x, 810) for x in (3360, 2810, 2260, 1710, 1160, 610, 480)]
SURVEY = SURVEY_ROW_TOP + SURVEY_ROW_LOW

# 'hybrid': acquire small objects at native resolution, keep Level-1 coverage.
#
# Why it exists. Every scored object has a fixed size in source pixels, and at
# Level 1 the transmitted view halves it: small_launcher 6.4 px, spacecraft
# 12.2, mine_roller 13.4, large_tower 14.8, tank 17.2. YOLO's finest stride is
# 8 px, so those five sit at or under the detection floor -- and they are 382 of
# 911 scored object-frames, all of them at ~0.00 AP in every configuration ever
# measured here. Level 2 is 1:1 and doubles every one of them.
#
# Measured, on the recorded survey run replayed through v4: mine_roller
# 0.000 -> 0.180 and small_launcher 0.000 -> 0.052, the first non-zero either
# class has ever scored, and tank 0.043 -> 0.084. But pure survey totals 0.080
# against 0.277, because it works the top half only and abandons jet_plane and
# large_tower entirely. Hence a hybrid rather than a switch.
#
# The shape follows the flight: the ground scrolls down ~69 px/frame, so every
# object enters at the top edge and crosses the whole frame in ~31 frames. An
# object needs to be caught once, near the top, while it is over the L2 rows;
# memory carries it down. The Level-1 phase is what stops those tracks drifting
# out of IoU and picks up the large classes L2's narrow view walks past.
#
# Both phase lengths are DRONE_SET-tunable, because the split between acquiring
# and covering cannot be measured offline -- a camera that would have looked
# somewhere else has no recorded view to replay -- so it has to be tuned on runs.
HYBRID_ACQUIRE = 9     # frames per Level-2 acquisition pass
HYBRID_COVER = 3       # frames per Level-1 coverage pass

# Level-2 inspection: a small object whose class is still unsure gets one
# close look, then the sweep resumes.
INSPECT = os.environ.get('DRONE_INSPECT', '0') == '1'
INSPECT_MAX_SIDE = 60          # source pixels: bigger ones read fine at L1
INSPECT_SURE_SHARE = 0.75      # vote share above which a class counts as settled
INSPECT_MAX_Y = 1500           # only while there is time left to use the answer
INSPECT_EVERY = 4              # answered frames between inspections, at least
INSPECT_LEAD = 3               # frames between deciding and the view arriving


for _item in filter(None, os.environ.get('DRONE_SET', '').split(',')):
    _name, _value = _item.split('=', 1)
    if _name not in globals() or not isinstance(globals()[_name], (int, float)):
        raise SystemExit(f'DRONE_SET: unknown numeric setting {_name}')
    globals()[_name] = type(globals()[_name])(float(_value))
    logger.warning('Setting %s = %s', _name, globals()[_name])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

_model = None
_models = []
_model_lock = threading.Lock()


def size_for(which: int) -> int:
    """Inference size for model ``which``; the last size repeats if fewer given."""
    return IMGSZ_LIST[min(which, len(IMGSZ_LIST) - 1)]


def _load_one(path: Path, imgsz: int = None):
    from ultralytics import YOLO

    yolo = YOLO(str(path))
    # One ordinary prediction builds Ultralytics' inference wrapper, which runs
    # the network ~30 % faster on this CPU than calling the module directly.
    yolo.predict(np.zeros((540, 960, 3), np.uint8), imgsz=imgsz or IMGSZ, device=DEVICE, verbose=False)
    order = [OBJECT_CLASSES.index(yolo.names[i]) for i in range(len(yolo.names))]
    return yolo.predictor.model, order


def load_model():
    """Load and warm up the detector(s) once; None if the weights are missing."""
    global _model, _models
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        logger.error('No model at %s: answering with empty detections', MODEL_PATH)
        return None
    import torch

    torch.set_num_threads(THREADS)
    _models = [_load_one(MODEL_PATH, size_for(0))]
    _model = _models[0]
    # The first inference is the slow one; pay for it before the clock starts.
    for _ in range(2):
        raw_detections(np.zeros((540, 960, 3), np.uint8))
    logger.info('Loaded %s on %s', MODEL_PATH, DEVICE)

    for index, path in enumerate(ALT_MODEL_PATHS, start=1):
        if not path.exists():
            # Do not quietly serve fewer models than asked for: a half-loaded
            # pair answers 200s with plausible boxes and has cost an attempt.
            logger.error('No alternate model at %s: running %d model(s) only', path, len(_models))
            continue
        _models.append(_load_one(path, size_for(index)))
        for _ in range(2):
            raw_detections(np.zeros((540, 960, 3), np.uint8), index)
        logger.info('Loaded alternate %s at imgsz %d', path, size_for(index))
    if len(_models) > 1:
        logger.info('%d models loaded; %s', len(_models),
                    'all run on every frame' if BOTH_MODELS else 'they take alternate frames')
    return _model


def raw_detections(image: np.ndarray, which: int = 0):
    """YOLO on one image: (boxes xyxy in image pixels, per-class scores).

    Ultralytics' own predictor keeps only the best class of each box. Doing the
    letterbox and NMS here keeps every class score, which lets the tracker
    report second guesses; mAP pays well for a right answer ranked lower.
    """
    import torch
    import torchvision

    index = which % len(_models) if _models else 0
    net, order = _models[index] if _models else _model
    height, width = image.shape[:2]
    ratio = size_for(index) / max(height, width)
    new_h, new_w = round(height * ratio), round(width * ratio)
    pad_h, pad_w = math.ceil(new_h / 32) * 32, math.ceil(new_w / 32) * 32
    top, left = (pad_h - new_h) // 2, (pad_w - new_w) // 2
    resized = image if ratio == 1 else cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((pad_h, pad_w, 3), 114, np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
    tensor = tensor.to(DEVICE).float().div_(255).unsqueeze(0)

    with torch.inference_mode():
        out = net(tensor)
        out = out[0] if isinstance(out, (list, tuple)) else out
        pred = out[0].transpose(0, 1)                     # (anchors, 4 + classes)
        scores = pred[:, 4:]
        best = scores.max(dim=1).values
        keep = best >= DETECTION_CONFIDENCE
        pred, scores, best = pred[keep], scores[keep], best[keep]
        xy, wh = pred[:, :2], pred[:, 2:4]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
        kept = torchvision.ops.nms(xyxy, best, NMS_IOU)[:MAX_DETECTIONS]
        xyxy, scores = xyxy[kept].cpu().numpy(), scores[kept].cpu().numpy()

    xyxy = (xyxy - [left, top, left, top]) / ratio
    probabilities = np.zeros((len(scores), len(OBJECT_CLASSES)), np.float32)
    probabilities[:, order] = scores
    return xyxy, probabilities


def detect(image: np.ndarray, source_region, frame: int = 0) -> list:
    """Detections on one view as (class, confidence, source box, class scores).

    With DRONE_MODEL_ALT set, the two models take alternate frames. They fail on
    different classes, and every detection goes into the same object memory, so
    a run sees the union of what both can find without paying for both on any
    one frame. Measured on the recorded flight (v4 + v5): the per-class mean hit
    rate is 42.6 % for v4 alone, 37.9 % for v5 alone and 46.2 % alternating.
    """
    if load_model() is None:
        return []
    with _model_lock:
        if BOTH_MODELS and len(_models) > 1:
            parts = [raw_detections(image, which) for which in range(len(_models))]
            xyxy = np.concatenate([part[0] for part in parts])
            probabilities = np.concatenate([part[1] for part in parts])
            # Which model produced each box. Concatenating threw this away.
            source = np.concatenate([np.full(len(part[0]), i, np.int8)
                                     for i, part in enumerate(parts)])
        else:
            xyxy, probabilities = raw_detections(image, frame)
            source = np.full(len(xyxy), frame % max(1, len(_models)), np.int8)
    rx1, ry1, rx2, ry2 = source_region
    height, width = image.shape[:2]
    scale = np.array([(rx2 - rx1) / width, (ry2 - ry1) / height] * 2)
    boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
    return [
        (OBJECT_CLASSES[int(p.argmax())], float(p.max()), box, p, int(m))
        for box, p, m in zip(boxes, probabilities, source)
    ]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def advance(box: np.ndarray, steps: int, motion=MOTION) -> np.ndarray:
    """Move a source-pixel box forward by ``steps`` frames of ground motion."""
    a, b, c, d, e, f = motion
    x1, y1, x2, y2 = box
    for _ in range(max(0, steps)):
        x1, y1 = x1 + a + b * x1 + c * y1, y1 + d + e * x1 + f * y1
        x2, y2 = x2 + a + b * x2 + c * y2, y2 + d + e * x2 + f * y2
    return np.array([x1, y1, x2, y2])


def drift_at_centre(motion) -> float:
    """How far the frame centre moves in one frame, under ``motion``."""
    a, b, c, d, e, f = motion
    x, y = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
    return float(np.hypot(a + b * x + c * y, d + e * x + f * y))


def fit_motion(samples, prior):
    """Ground motion fitted on our own re-detections, or ``prior`` if unsure.

    Each sample is (x, y, dx, dy): where an object was, and how many pixels a
    frame it has moved since we last saw it there. Most objects are ground, so
    the fit is the ground's motion - but vehicles and aircraft move on their
    own, so the worst quarter of the residuals is dropped before the final fit,
    and the result is shrunk toward the prior while samples are few.
    """
    if len(samples) < MOTION_MIN_SAMPLES:
        return prior
    data = np.asarray(samples[-MOTION_SAMPLE_MEMORY:], float)
    design = np.column_stack([np.ones(len(data)), data[:, 0], data[:, 1]])

    fitted = []
    for target in (data[:, 2], data[:, 3]):
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        keep = np.abs(design @ coefficients - target) <= np.quantile(
            np.abs(design @ coefficients - target), MOTION_TRIM
        )
        if keep.sum() >= MOTION_MIN_SAMPLES:
            coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        fitted.append(coefficients)

    estimate = (*fitted[0], *fitted[1])
    weight = len(data) / (len(data) + MOTION_PRIOR_STRENGTH)
    blended = tuple(weight * new + (1 - weight) * old for new, old in zip(estimate, prior))
    # A fit that disagrees wildly with the prior is more likely to be a handful
    # of moving objects than a real flight: no fit beats a bad one.
    if abs(drift_at_centre(blended) - drift_at_centre(prior)) > MOTION_MAX_CORRECTION:
        return prior
    return blended


def cover(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def add_votes(track, name, confidence, probabilities, weight) -> None:
    if probabilities is None:
        track.votes[name] = track.votes.get(name, 0.0) + weight * confidence
        return
    for index in np.flatnonzero(probabilities >= MIN_VOTE_SCORE):
        cls = OBJECT_CLASSES[index]
        track.votes[cls] = track.votes.get(cls, 0.0) + weight * float(probabilities[index])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    box: np.ndarray                 # source pixels at ``frame``
    frame: int
    votes: Dict[str, float] = field(default_factory=dict)
    best_confidence: float = 0.0
    last_seen: int = 0
    best_level: int = 0
    hits: int = 0
    misses: int = 0
    inspected: bool = False
    truncated: bool = False
    # Indices of the models that have ever matched this track. See AGREEMENT_WEIGHT.
    models: set = field(default_factory=set)
    # The last box as actually *observed*, not carried: one half of a motion sample.
    seen_box: Optional[np.ndarray] = None
    seen_frame: int = -1

    def label(self) -> Tuple[str, float]:
        name = max(self.votes, key=self.votes.get)
        return name, self.votes[name]


@dataclass
class Sequence:
    tracks: List[Track] = field(default_factory=list)
    sweep_index: int = 0
    last_frame: int = -1
    # The last view we asked for: (level, x, y).
    pending: Optional[Tuple[int, int, int]] = None
    answers_since_inspection: int = 0
    hybrid_step: int = 0
    # Where the hybrid camera's Level-1 coverage phase has got to. Separate from
    # sweep_index, which the hybrid uses for the Level-2 snake.
    cover_index: int = 0
    # This flight's ground motion, re-fitted as re-detections come in.
    motion: Tuple[float, ...] = MOTION
    motion_samples: List[Tuple[float, float, float, float]] = field(default_factory=list)


_sequences: Dict[str, Sequence] = {}
_sequences_lock = threading.Lock()


def update_tracks(state: Sequence, frame: int, level: int, region, detections) -> list:
    """Fold this view's detections into memory; return the unremembered ones."""
    # Bring every track to this frame; forget the ones that left the ground.
    alive = []
    for track in state.tracks:
        track.box = advance(track.box, frame - track.frame, state.motion)
        track.frame = frame
        x1, y1, x2, y2 = track.box
        if x2 > 0 and y2 > 0 and x1 < IMAGE_WIDTH and y1 < IMAGE_HEIGHT:
            alive.append(track)
    state.tracks = alive

    weight = LEVEL_WEIGHT[level]
    rx1, ry1, rx2, ry2 = region
    cut_off = CUT_OFF_PIXELS * (rx2 - rx1) / 960
    matched = set()
    transient = []
    for detection in sorted(detections, key=lambda d: -d[1]):
        name, confidence, box = detection[:3]
        probabilities = detection[3] if len(detection) > 3 else None
        source = detection[4] if len(detection) > 4 else 0
        # Cut short by the view edge (the frame edge cuts the true box too).
        truncated = (
            (box[0] <= rx1 + cut_off and rx1 > 0) or (box[1] <= ry1 + cut_off and ry1 > 0)
            or (box[2] >= rx2 - cut_off and rx2 < IMAGE_WIDTH)
            or (box[3] >= ry2 - cut_off and ry2 < IMAGE_HEIGHT)
        )

        best, best_score = None, 0.0
        for track in state.tracks:
            overlap = iou(track.box, box)
            covered = cover(track.box, box)
            if overlap > MATCH_IOU or covered > MATCH_MIN_COVER:
                score = max(overlap, covered * 0.9)
                if score > best_score:
                    best, best_score = track, score
        if best is None:
            if confidence < NEW_TRACK_CONFIDENCE:
                transient.append((name, confidence * TRANSIENT_WEIGHT * (TRUNCATED_WEIGHT if truncated else 1), box))
                continue
            best = Track(box=box, frame=frame, best_level=level, truncated=truncated)
            best.models.add(source)
            if not truncated:
                best.seen_box, best.seen_frame = box.copy(), frame
            state.tracks.append(best)
        elif id(best) in matched:
            # A second detection of an object already handled this frame is a
            # class vote only; the first (most confident) one set the box. It is
            # also where model agreement shows up -- this is another model
            # finding the same object -- so the source is recorded before the
            # early return.
            best.models.add(source)
            add_votes(best, name, confidence, probabilities, weight)
            continue
        elif truncated:
            # Two partial views: the object spans at least both.
            if best.truncated:
                best.box = np.array([min(best.box[0], box[0]), min(best.box[1], box[1]),
                                     max(best.box[2], box[2]), max(best.box[3], box[3])])
        elif best.truncated or level >= best.best_level:
            # A whole sighting beats a partial one; finer views beat coarser.
            best.box = box
            best.best_level = level
            best.truncated = False
        else:
            best.box = 0.7 * best.box + 0.3 * box
        matched.add(id(best))
        best.models.add(source)
        # Seeing the same object twice measures how far the ground really moved
        # between those frames. Only whole sightings: a box the view edge cut
        # short has a centre that says more about the edge than the object.
        if not truncated:
            gap = frame - best.seen_frame
            if best.seen_box is not None and 0 < gap <= MOTION_MAX_GAP:
                was = best.seen_box
                state.motion_samples.append((
                    (was[0] + was[2]) / 2, (was[1] + was[3]) / 2,
                    ((box[0] + box[2]) - (was[0] + was[2])) / 2 / gap,
                    ((box[1] + box[3]) - (was[1] + was[3])) / 2 / gap,
                ))
            best.seen_box, best.seen_frame = box.copy(), frame
        add_votes(best, name, confidence, probabilities, weight * (TRUNCATED_WEIGHT if truncated else 1))
        best.best_confidence = max(best.best_confidence, confidence * (0.6 + 0.4 * weight))
        best.last_seen = frame
        best.hits += 1
        best.misses = 0

    # Tracks well inside this view that nothing matched were probably wrong,
    # unless the view is coarser than the one that found them.
    for track in state.tracks:
        if id(track) in matched or level < track.best_level:
            continue
        x1, y1, x2, y2 = track.box
        if (x1 > rx1 + EDGE_MARGIN and y1 > ry1 + EDGE_MARGIN
                and x2 < rx2 - EDGE_MARGIN and y2 < ry2 - EDGE_MARGIN):
            track.misses += 1
    state.tracks = [t for t in state.tracks if t.misses < MAX_MISSES]
    if len(state.tracks) > MAX_TRACKS:
        state.tracks.sort(key=lambda t: -t.best_confidence)
        del state.tracks[MAX_TRACKS:]

    if len(state.motion_samples) > MOTION_SAMPLE_MEMORY * 2:
        del state.motion_samples[:-MOTION_SAMPLE_MEMORY]
    state.motion = fit_motion(state.motion_samples, MOTION)
    return transient


def annotations_for(state: Sequence, frame: int, transient=()) -> List[DroneFlybyPredictionDto]:
    annotations = []
    # Kept apart from `annotations` on purpose: the floor band is appended after
    # the real answers are sorted, never sorted together with them. See FLOOR_ZERO.
    floor = []

    def scaled(box):
        if BOX_SCALE == 1.0:
            return box
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * BOX_SCALE / 2, (y2 - y1) * BOX_SCALE / 2
        return np.array([cx - w, cy - h, cx + w, cy + h])

    def reported(box, name):
        """Track box -> the bbox we answer with, grown for this class.

        Growth is per class and a track answers several classes (the winner plus
        runner-ups), so it cannot be folded into the single box computed per
        track. It is applied here and nowhere else: the stored track box must
        stay tight, or matching, motion fitting and truncation all shift with it.
        """
        grow = BOX_GROW.get(name)
        if grow and grow != (1.0, 1.0):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1) * grow[0] / 2, (y2 - y1) * grow[1] / 2
            box = np.array([cx - w, cy - h, cx + w, cy + h])
        return clip_bbox_to_frame((
            box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT,
            box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT,
        ))

    for name, confidence, box in transient:
        bbox = reported(scaled(box), name)
        if bbox is not None:
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
    for track in state.tracks:
        box = scaled(track.box)
        # Whether the track is on screen at all is judged on the ungrown box, so
        # turning growth on never changes which tracks are answered.
        if reported(box, '') is None:
            continue
        ranked = sorted(track.votes.items(), key=lambda item: -item[1])
        base = track.best_confidence
        base *= min(1.0, HITS_BASE + HITS_STEP * track.hits)
        # Models that have never found this track. At the served
        # AGREEMENT_WEIGHT of 1.0 this is a no-op, byte for byte.
        if AGREEMENT_WEIGHT != 1.0 and len(_models) > 1:
            base *= AGREEMENT_WEIGHT ** max(0, len(_models) - len(track.models))
        # max(0, ...): a stale frame is behind tracks already moved to a later
        # one, and a negative exponent would *raise* the confidence above
        # best_confidence instead of decaying it.
        base *= UNSEEN_DECAY ** max(0, frame - track.last_seen)
        # Times the camera looked at this track and did not find it. At the served
        # MISS_PENALTY of 1.0 this is a no-op, byte for byte.
        if MISS_PENALTY != 1.0 and track.misses:
            base *= MISS_PENALTY ** track.misses
        if track.truncated:
            base *= TRUNCATED_WEIGHT
        # A track can hold only zero-weight votes (LEVEL_WEIGHT of 0 for the
        # level it was seen at), and dividing by that raised ZeroDivisionError
        # inside the caller's try, which discards the WHOLE frame's annotations
        # and its camera command -- a silent, total loss that looks like a
        # quiet frame. Fall back to equal shares instead.
        top_vote = ranked[0][1] or 1.0
        named = set()
        for rank, (name, vote) in enumerate(ranked[:1 + RUNNER_UPS]):
            share = vote / top_vote
            if rank and share < RUNNER_UP_SHARE:
                break
            named.add(name)
            confidence = base if rank == 0 else base * 0.9 * share
            bbox = reported(box, name)
            if bbox is None:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=name,
                bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
        if FLOOR_ALL_CLASSES:
            for name in OBJECT_CLASSES:
                if name not in named:
                    bbox = reported(box, name)
                    if bbox is None:
                        continue
                    annotations.append(DroneFlybyPredictionDto(
                        object_id=name,
                        bbox=[round(c, 6) for c in bbox],
                        confidence=round(float(np.clip(base * FLOOR_ALL_CLASSES, 0.001, 1.0)), 4),
                    ))
        if FLOOR_ZERO:
            # sqrt(w*h) of the track box, against the class's Helsinki size prior.
            side = math.sqrt(max(1e-6, float((box[2] - box[0]) * (box[3] - box[1]))))
            for name in OBJECT_CLASSES:
                if name in named:
                    continue
                prior = CLASS_SIZE.get(name)
                if prior and not (1 / FLOOR_SIZE_TOL <= side / prior <= FLOOR_SIZE_TOL):
                    continue
                bbox = reported(box, name)
                if bbox is None:
                    continue
                # Exactly 0.0, and deliberately NOT through the np.clip above:
                # clipping it to 0.001 would put it back in the band our own
                # faint answers live in, which is what cost 0.009 last time.
                floor.append(DroneFlybyPredictionDto(
                    object_id=name,
                    bbox=[round(c, 6) for c in bbox],
                    confidence=0.0,
                ))
    annotations.sort(key=lambda a: -a.confidence)
    # After the sort: every real answer outranks the whole band, and the 500 cap
    # then drops floor boxes rather than anything we actually believe.
    annotations.extend(floor)
    return annotations[:500]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

def legal(base: Tuple[int, int, int], level: int, x: int, y: int) -> bool:
    return describe_camera_rejection(base[0], (base[1], base[2]), level, (x, y)) is None


def inspection_target(state: Sequence, base, frame: int) -> Optional[Tuple[int, int, int]]:
    """The most doubtful small object reachable from ``base``, as an L2 view."""
    best, best_doubt = None, 0.0
    for track in state.tracks:
        if track.inspected or track.best_level >= 2:
            continue
        box = advance(track.box, INSPECT_LEAD, state.motion)
        width, height = box[2] - box[0], box[3] - box[1]
        if max(width, height) > INSPECT_MAX_SIDE or box[1] > INSPECT_MAX_Y or box[3] < 0:
            continue
        share = max(track.votes.values()) / sum(track.votes.values())
        doubt = (1 - share) + (1 - track.best_confidence)
        if share >= INSPECT_SURE_SHARE and track.best_confidence >= 0.6:
            continue
        x = int(min(max((box[0] + box[2]) / 2, 480), 3360))
        y = int(min(max((box[1] + box[3]) / 2, 270), 1890))
        if doubt > best_doubt and legal(base, 2, x, y):
            best, best_doubt = (track, (2, x, y)), doubt
    if best is None:
        return None
    best[0].inspected = True
    return best[1]


def step_towards(base, level: int, x: int, y: int) -> Optional[Tuple[int, int, int]]:
    """A legal move from ``base`` towards (level, x, y), shortened if too far.

    The evaluator checks three things in order -- the level transition, the
    centre bounds for the requested level, then the distance against the limit
    for the level you are *currently* at (L0 2203, L1 1102, L2 551). A move that
    fails any of them is refused and the camera idles, so this shortens the step
    until it passes rather than asking for something that cannot be granted.
    """
    if level not in ALLOWED_RESOLUTION_LEVELS.get(base[0], ()):
        # Levels change one step at a time; 1 is reachable from both 0 and 2.
        level = 1
    if level == 0:
        return (0, *FULL_FRAME_CENTER)
    low_x, high_x, low_y, high_y = center_bounds_for_level(level)
    x = int(min(max(x, low_x), high_x))
    y = int(min(max(y, low_y), high_y))
    if legal(base, level, x, y):
        return (level, x, y)
    # Too far: walk in from the target until the step fits. Clipping to bounds
    # can itself lengthen the move, so this checks rather than computes a ratio.
    dx, dy = x - base[1], y - base[2]
    for ratio in (0.9, 0.75, 0.6, 0.45, 0.3, 0.15):
        near_x = int(min(max(base[1] + dx * ratio, low_x), high_x))
        near_y = int(min(max(base[2] + dy * ratio, low_y), high_y))
        if legal(base, level, near_x, near_y):
            return (level, near_x, near_y)
    return None


def hybrid_next_view(base, state: Sequence) -> Optional[RequestedViewDto]:
    """Alternate Level-2 acquisition over the top rows with Level-1 coverage."""
    state.hybrid_step += 1
    cycle = max(1, HYBRID_ACQUIRE + HYBRID_COVER)
    acquiring = (state.hybrid_step % cycle) < HYBRID_ACQUIRE

    if acquiring:
        # Advance along the L2 snake only once the camera has actually arrived,
        # so a shortened step resumes towards the same point instead of skipping it.
        point = SURVEY[state.sweep_index % len(SURVEY)]
        if base[0] == 2 and (base[1], base[2]) == point:
            state.sweep_index = (state.sweep_index + 1) % len(SURVEY)
            point = SURVEY[state.sweep_index % len(SURVEY)]
        target = step_towards(base, 2, *point)
    else:
        # Coverage. Leaving L2 costs one move, so rejoin at the nearest point --
        # but then ADVANCE along the sweep on every following coverage frame.
        #
        # This used to pick the nearest reachable Level-1 point every time, and
        # once the camera had arrived the nearest point was the one it was
        # already standing on. Driving choose_next_view through the evaluator's
        # own Camera showed the whole coverage phase as L1(2880,540) three times
        # running: HYBRID_COVER=3 bought one sixth of the frame, looked at
        # three times. Worst 60 px cell over a flight: 1 look, against 42 for
        # `full`. The 0.1234 the hybrid scored measured that, not Level 2.
        l1 = [p for p in SWEEP if p[0] == 1] or [(1, *FULL_FRAME_CENTER)]
        target = None
        if base[0] != 1:
            reachable = [(i, p) for i, p in enumerate(l1) if legal(base, *p)]
            if reachable:
                state.cover_index, target = min(
                    reachable,
                    key=lambda item: (item[1][1] - base[1]) ** 2 + (item[1][2] - base[2]) ** 2)
        else:
            for step in range(1, len(l1) + 1):
                index = (state.cover_index + step) % len(l1)
                if legal(base, *l1[index]):
                    state.cover_index, target = index, l1[index]
                    break
        if target is None:
            target = step_towards(base, 1, base[1], base[2])

    if target is not None and legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    state.pending = None
    return None


def survey_next_view(base, state: Sequence) -> Optional[RequestedViewDto]:
    if base[0] == 2 and (base[1], base[2]) == SURVEY[state.sweep_index % len(SURVEY)]:
        state.sweep_index = (state.sweep_index + 1) % len(SURVEY)
    elif base[0] == 2:
        # Off the pattern: rejoin at the nearest point that can be reached.
        state.sweep_index = min(
            range(len(SURVEY)),
            key=lambda i: (SURVEY[i][0] - base[1]) ** 2 + (SURVEY[i][1] - base[2]) ** 2,
        )
    x, y = SURVEY[state.sweep_index % len(SURVEY)]
    if base[0] == 0:
        # Zoom in one step, towards the first survey point.
        target = (1, int(min(max(x, 960), 2880)), int(min(max(y, 540), 1620)))
    else:
        target = (2, x, y)
        if not legal(base, *target):
            # Too far for one move: step towards it in a straight line.
            limit = 1100 if base[0] == 1 else 550
            dx, dy = x - base[1], y - base[2]
            ratio = min(1.0, limit / max(1.0, math.hypot(dx, dy)))
            target = (2, int(base[1] + dx * ratio), int(base[2] + dy * ratio))
            target = (2, int(min(max(target[1], 480), 3360)), int(min(max(target[2], 270), 1890)))
    if legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    state.pending = None
    return None


def choose_next_view(request: DroneFlybyPredictRequestDto, state: Sequence) -> Optional[RequestedViewDto]:
    view = request.view
    here = (view.resolution_level, view.center_x, view.center_y)
    # Where the camera will be when this command is applied: our previous
    # request, unless the evaluator refused it.
    base = here
    if state.pending is not None and request.camera_command_feedback is None:
        base = state.pending

    if CAMERA == 'survey':
        return survey_next_view(base, state)
    if CAMERA == 'hybrid':
        return hybrid_next_view(base, state)

    target = None
    state.answers_since_inspection += 1
    if INSPECT and base[0] == 1 and state.answers_since_inspection >= INSPECT_EVERY:
        target = inspection_target(state, base, request.frame)
    if target is not None:
        state.answers_since_inspection = 0
    elif base[0] == 2:
        # Back up to Level 1 as close as the 551 px limit allows.
        x = int(min(max(base[1], 960), 2880))
        y = int(min(max(base[2], 540), 1620))
        target = (1, x, y)
    else:
        position = SWEEP[state.sweep_index % len(SWEEP)]
        if base == position:
            state.sweep_index = (state.sweep_index + 1) % len(SWEEP)
        elif base in SWEEP and base[0] == 1:
            # Off the pattern (a refusal, a restart): carry on from the nearest
            # matching point at or after the current index, so repeated points
            # in a pattern do not send the sweep back to its opening pass.
            offsets = range(len(SWEEP))
            step = next((k for k in offsets if SWEEP[(state.sweep_index + k) % len(SWEEP)] == base), 0)
            state.sweep_index = (state.sweep_index + step + 1) % len(SWEEP)
        target = SWEEP[state.sweep_index % len(SWEEP)]
        if base[0] == 1 and target[0] == 1 and not legal(base, *target):
            # Too far for one move: rejoin the pattern at the nearest Level-1 point.
            nearest = min(
                (i for i, p in enumerate(SWEEP) if p[0] == 1 and legal(base, *p)),
                key=lambda i: (SWEEP[i][1] - base[1]) ** 2 + (SWEEP[i][2] - base[2]) ** 2,
                default=None,
            )
            if nearest is not None:
                state.sweep_index = nearest
                target = SWEEP[nearest]

    if target is not None and legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    # Something unexpected: the full view is always one step from L0 and L1.
    full = (0, *FULL_FRAME_CENTER)
    if legal(base, *full):
        state.pending = full
        return RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1])
    state.pending = None
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    if request.camera_command_feedback is not None:
        logger.warning('Camera command ignored: %s', request.camera_command_feedback.reason)

    with _sequences_lock:
        state = _sequences.get(request.sequence_id)
        # A new attempt, or the same id replayed from the start: forget.
        if state is None or request.frame_index == 0 or request.frame < state.last_frame - RESTART_GAP:
            state = _sequences[request.sequence_id] = Sequence()
            while len(_sequences) > MAX_SEQUENCES:
                _sequences.pop(next(iter(_sequences)))

    view = request.view
    try:
        detections = detect(decode_view(view), view.source_region_xyxy, request.frame)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        detections = []

    annotations, requested_view = [], None
    try:
        # Requests can overlap when one runs long; the newest frame wins.
        with _sequences_lock:
            transient = []
            stale = request.frame < state.last_frame
            if not stale:
                transient = update_tracks(state, request.frame, view.resolution_level, view.source_region_xyxy, detections)
                state.last_frame = request.frame
            # Separate try blocks: these two answer different questions, and a
            # failure building annotations used to discard the camera command as
            # well, which steers the rest of the run, not just this frame.
            try:
                annotations = annotations_for(state, request.frame, transient)
            except Exception:
                logger.exception('Building annotations failed on frame %s', request.frame)
            # A late frame's answer is still scored, but its view is out of
            # date: leave the camera plan to the newest frame.
            if not stale:
                try:
                    requested_view = choose_next_view(request, state)
                except Exception:
                    logger.exception('Camera planning failed on frame %s', request.frame)
    except Exception:
        logger.exception('Tracking failed on frame %s', request.frame)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
```

## `training/make_dataset.py`

Synthetic training-data generator (cut-and-paste onto backgrounds, views cut like the evaluator).

```python
"""Build a synthetic YOLO dataset from the supplied frames and object patches.

    python training/extract_patches.py                    # once, makes data/patches
    python training/make_dataset.py --scenes 20 --preview # quick look
    python training/make_dataset.py --scenes 1200         # the real thing

For every synthetic scene:

1. take a background: one of the 4K Helsinki frames (randomly flipped, real
   objects and labels kept), or, with --backgrounds, a random 3840x2160 cut
   from any large aerial photo (no objects of ours in it, so no labels);
2. paste extra objects from data/patches at free spots, classes drawn evenly,
   each randomly rotated, scaled and recoloured a little;
3. cut camera views out of it exactly the way the evaluator does (the crop for
   the level, then INTER_AREA down to 960x540) and write YOLO labels.

Output: data/yolo/{images,labels}/{train,val}/ and data/yolo/data.yaml.
"""

import argparse
import math
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

# GeoTIFF photos carry map tags OpenCV does not know; the warnings are noise.
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    center_bounds_for_level,
    frame_numbers,
    load_sample,
    source_region_for_view,
)

VIEW_SIZE = (960, 540)
# How many views to cut from each synthetic scene, per resolution level.
# The camera served Level 1 for 244 of 247 views in the recorded flight, so that
# is the resolution the detector has to be good at; the old 1/2/3 split spent
# half its views on Level 2, which we barely request. Level 0 and Level 2 stay in
# at one view each: DRONE_INSPECT is off today but is one experiment from being
# on, and Level 0 is what the camera falls back to when it loses the target.
# Six views per scene either way, so dataset size and GPU cost do not move.
VIEWS_PER_LEVEL = {0: 1, 1: 4, 2: 1}
PASTES_PER_SCENE = (8, 30)
UNTOUCHED_SCENE_SHARE = 0.15   # scenes left exactly as supplied
SCALE_RANGE = (0.85, 1.15)     # cut-outs taken from the validation flight
# Helsinki cut-outs, which are photographed much closer than the flight objects.
# Every confirmed validation object measured against its Helsinki counterpart
# (box diagonal, source pixels) came out smaller: hangar 0.98, jammer 0.89,
# small_plane 0.83, tank 0.75, large_tower 0.73, mine_roller 0.73, helicopter
# 0.67, small_tower 0.64, spacecraft 0.58, jet_plane 0.56, small_launcher 0.49,
# large_launcher 0.45 - median 0.70, and only hangar inside (0.85, 1.15). Pasting
# at the old range put every Helsinki object 1.2-1.6x too large. That matters
# more now than it did for v5: on the real backgrounds the terrain scale is
# exact, so paste size alone decides how big an object looks.
# Not to be confused with BOX_SCALE, which shrinks the box we *report* at
# inference - that one collapsed a run from 0.143 to 0.017 and stays at 1.0.
HELSINKI_SCALE_RANGE = (0.45, 1.00)
# Hue rotation in degrees: whole backgrounds, and pasted objects (whose colour
# is mostly their own, so less).
BACKGROUND_HUE = 20
PATCH_HUE = 8
MIN_VISIBLE = 0.5              # keep a box cut by the view edge if this much shows
MIN_BOX_VIEW_PIXELS = 2
L2_ON_OBJECT_SHARE = 0.7       # zoomed views mostly look at something
FILLED_MASK = 0.7              # mask this full is the ellipse fallback

CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}

# With --backgrounds, this share of scenes still uses the Helsinki frames.
HELSINKI_SHARE = 0.2
BACKGROUND_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
# Label rasters live next to the photos in segmentation datasets.
BACKGROUND_SKIP_WORDS = ('mask', '/gt/', 'drone-flyby-code')
BACKGROUND_MIN_SIDE = 2000   # smaller photos would need blurry upscaling
BACKGROUND_SCALE = (0.8, 1.25)
# Share of background scenes taken from the recorded flight rather than the
# stock aerial photo sets, when --real-backgrounds is given.
REAL_BACKGROUND_SHARE = 0.6

# Patches cut from recorded validation views (--extra-patches) are used for
# this share of pastes of their class; they carry that scene's lighting.
EXTRA_PATCH_SHARE = 0.4
# Photometric and shadow variation per pasted object, so a model does not learn
# one scene's light: validation objects are darker and cast hard shadows.
GRADE_SHARE = 0.6
BLUR_SHARE = 0.3
SHADOW_SHARE = 0.5
NOISE_SHARE = 0.25
# Share of well-masked patches that are also Poisson-blended (the badly masked
# ones always are), so a model never learns one pasting style.
BLEND_SHARE = 0.25
BLEND_MIN_TEXTURE = 8.0    # grey-level std range of the ground under a blended patch
BLEND_MAX_TEXTURE = 22.0

# Filled in per worker by _init_worker.
_patches = {}
_extra_patches = {}
_class_weights = None
_frames = []
_backgrounds = []
_real_backgrounds = []


def load_patches(folder: Path, required: bool = True):
    patches = {}
    for name in OBJECT_CLASSES:
        files = sorted((folder / name).glob('*.png'))
        if not files:
            if required:
                raise SystemExit(f'No patches for {name} in {folder}: run extract_patches.py first')
            continue
        patches[name] = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in files]
    return patches


def pick_patch(name: str, rng: random.Random):
    """A patch of this class and the scale range that suits its source."""
    extra = _extra_patches.get(name)
    if extra and rng.random() < EXTRA_PATCH_SHARE:
        return rng.choice(extra), SCALE_RANGE
    return rng.choice(_patches[name]), HELSINKI_SCALE_RANGE


def rotate_hue(bgr: np.ndarray, degrees: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + int(round(degrees / 2))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def find_backgrounds(folders) -> List[Path]:
    found = []
    for folder in folders:
        for path in Path(folder).rglob('*'):
            text = str(path).lower()
            if path.suffix.lower() in BACKGROUND_SUFFIXES and not any(w in text for w in BACKGROUND_SKIP_WORDS):
                found.append(path)
    return sorted(found)


def _init_worker(patch_folder: Path, backgrounds, extra_folders=(), class_weights=None, helsinki_share=None,
                 real_backgrounds=(), real_share=None):
    global _patches, _frames, _backgrounds, _extra_patches, _class_weights, HELSINKI_SHARE
    global _real_backgrounds, REAL_BACKGROUND_SHARE
    if real_share is not None:
        REAL_BACKGROUND_SHARE = real_share
    _real_backgrounds = list(real_backgrounds)
    if helsinki_share is not None:
        HELSINKI_SHARE = helsinki_share
    _class_weights = [class_weights.get(name, 1.0) for name in OBJECT_CLASSES] if class_weights else None
    _patches = load_patches(patch_folder)
    _extra_patches = {}
    for folder in extra_folders:
        for name, images in load_patches(folder, required=False).items():
            _extra_patches.setdefault(name, []).extend(images)
    _frames = frame_numbers()
    _backgrounds = backgrounds


def load_background(rng: random.Random) -> Optional[np.ndarray]:
    """A random 3840x2160 cut from one of the aerial photos, or None."""
    for _attempt in range(5):
        # Terrain from the flight itself is the closest background we have to the
        # scene being scored, so it is drawn far more often than the stock photos.
        pool = (_real_backgrounds if _real_backgrounds and rng.random() < REAL_BACKGROUND_SHARE
                else _backgrounds or _real_backgrounds)
        path = rng.choice(pool)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < BACKGROUND_MIN_SIDE:
            continue
        scale = rng.uniform(*BACKGROUND_SCALE)
        # Never smaller than the frame.
        scale = max(scale, IMAGE_WIDTH / image.shape[1], IMAGE_HEIGHT / image.shape[0])
        if abs(scale - 1) > 0.01:
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        y = rng.randint(0, image.shape[0] - IMAGE_HEIGHT)
        x = rng.randint(0, image.shape[1] - IMAGE_WIDTH)
        image = image[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH]
        if rng.random() < 0.5:
            image = image[::-1, ::-1]
        # Nudge the colours a little so photos from one city do not all look alike.
        image = image.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
        return rotate_hue(image, rng.uniform(-BACKGROUND_HUE, BACKGROUND_HUE))
    return None


def transform_patch(patch: np.ndarray, rng: random.Random, scale_range=SCALE_RANGE) -> np.ndarray:
    """Rotate, scale, flip and recolour one BGRA patch; crop to its mask."""
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    filled = (patch[:, :, 3] > 0).mean() > FILLED_MASK
    # An ellipse mask carries ground in its corners; only right angles keep
    # its box honest.
    angle = rng.choice([0, 90, 180, 270]) if filled else rng.uniform(0, 360)
    scale = rng.uniform(*scale_range)

    height, width = patch.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(math.ceil(height * sin + width * cos)) + 2
    new_height = int(math.ceil(height * cos + width * sin)) + 2
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    patch = cv2.warpAffine(
        np.ascontiguousarray(patch), matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    ys, xs = np.nonzero(patch[:, :, 3] > 127)
    if len(xs) == 0:
        return None
    patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    bgr = patch[:, :, :3].astype(np.float32)
    bgr = bgr * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    bgr *= np.array([rng.uniform(0.95, 1.05) for _ in range(3)], np.float32)
    if rng.random() < GRADE_SHARE:
        # Another scene's light: gamma (mostly darker), flatter, greyer.
        bgr = 255.0 * (np.clip(bgr, 0, 255) / 255.0) ** rng.uniform(0.8, 1.8)
        mean = bgr.mean()
        bgr = mean + (bgr - mean) * rng.uniform(0.6, 1.1)
        grey = bgr.mean(axis=2, keepdims=True)
        bgr = grey + (bgr - grey) * rng.uniform(0.4, 1.2)
    patch[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    if PATCH_HUE:
        patch[:, :, :3] = rotate_hue(np.ascontiguousarray(patch[:, :, :3]), rng.uniform(-PATCH_HUE, PATCH_HUE))
    if rng.random() < BLUR_SHARE:
        patch[:, :, :3] = cv2.GaussianBlur(patch[:, :, :3], (0, 0), rng.uniform(0.4, 1.2))
    return patch


def overlaps(box, boxes, gap: int = 8) -> bool:
    x1, y1, x2, y2 = box
    return any(
        x1 - gap < bx2 and bx1 < x2 + gap and y1 - gap < by2 and by1 < y2 + gap
        for bx1, by1, bx2, by2 in boxes
    )


def cast_shadow(scene: np.ndarray, patch: np.ndarray, x: int, y: int, rng: random.Random) -> None:
    """Darken the ground where the object's silhouette would throw a shadow."""
    height, width = patch.shape[:2]
    length = rng.uniform(0.15, 0.6) * max(height, width)
    angle = rng.uniform(0, 2 * math.pi)
    dx, dy = int(length * math.cos(angle)), int(length * math.sin(angle))
    sx1, sy1 = max(0, x + dx), max(0, y + dy)
    sx2, sy2 = min(scene.shape[1], x + dx + width), min(scene.shape[0], y + dy + height)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    mask = patch[sy1 - y - dy:sy2 - y - dy, sx1 - x - dx:sx2 - x - dx, 3].astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.8, 2.5))[:, :, None]
    darkness = rng.uniform(0.3, 0.65)
    region = scene[sy1:sy2, sx1:sx2].astype(np.float32)
    scene[sy1:sy2, sx1:sx2] = (region * (1 - darkness * mask)).astype(np.uint8)


def paste_blended(scene: np.ndarray, patch: np.ndarray, x: int, y: int) -> bool:
    """Poisson-blend a patch whose mask is a disc or rectangle of its old ground.

    Helicopters, towers and launchers are thin and camouflaged, so their cut-outs
    keep a disc of Helsinki grass. Pasted as is, that disc is what a model learns.
    Mixed cloning keeps the strongest gradients of either image: the object's
    edges survive, the old ground takes the new ground's colour and texture.
    """
    height, width = patch.shape[:2]
    mask = (patch[:, :, 3] > 0).astype(np.uint8) * 255
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0
    if (x < 1 or y < 1 or x + width >= scene.shape[1] - 1 or y + height >= scene.shape[0] - 1
            or height < 5 or width < 5 or not mask.any()):
        return False
    try:
        # Flatten the old ground's fine texture; strong edges (rotors, lattice) stay.
        source = cv2.edgePreservingFilter(np.ascontiguousarray(patch[:, :, :3]),
                                          flags=cv2.RECURS_FILTER, sigma_s=20, sigma_r=0.25)
        blended = cv2.seamlessClone(source, scene, mask,
                                    (x + width // 2, y + height // 2), cv2.MIXED_CLONE)
    except cv2.error:
        return False
    scene[:] = blended
    return True


def paste(scene: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    height, width = patch.shape[:2]
    alpha = patch[:, :, 3].astype(np.float32) / 255.0
    # A one-pixel feather hides the cut line without smearing the object.
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[:, :, None]
    region = scene[y:y + height, x:x + width].astype(np.float32)
    blended = region * (1 - alpha) + patch[:, :, :3].astype(np.float32) * alpha
    scene[y:y + height, x:x + width] = blended.astype(np.uint8)


def build_scene(rng: random.Random):
    """One 4K image plus its labels as [(class, x1, y1, x2, y2)]."""
    image = None
    if (_backgrounds or _real_backgrounds) and rng.random() >= HELSINKI_SHARE:
        image = load_background(rng)
    if image is not None:
        labels = []
    else:
        image, annotations = load_sample(rng.choice(_frames))
        labels = [(a['object_id'], *a['bbox']) for a in annotations]

    if rng.random() < 0.5:
        image = image[:, ::-1]
        labels = [(n, IMAGE_WIDTH - x2, y1, IMAGE_WIDTH - x1, y2) for n, x1, y1, x2, y2 in labels]
    if rng.random() < 0.5:
        image = image[::-1]
        labels = [(n, x1, IMAGE_HEIGHT - y2, x2, IMAGE_HEIGHT - y1) for n, x1, y1, x2, y2 in labels]
    image = np.ascontiguousarray(image)

    if rng.random() < UNTOUCHED_SCENE_SHARE:
        return image, labels

    occupied = [box[1:] for box in labels]
    for _ in range(rng.randint(*PASTES_PER_SCENE)):
        name = rng.choices(OBJECT_CLASSES, weights=_class_weights)[0]
        source, scale_range = pick_patch(name, rng)
        patch = transform_patch(source, rng, scale_range)
        if patch is None:
            continue
        height, width = patch.shape[:2]
        filled = (patch[:, :, 3] > 0).mean() >= FILLED_MASK
        for _attempt in range(20):
            x = rng.randint(0, IMAGE_WIDTH - width)
            y = rng.randint(0, IMAGE_HEIGHT - height)
            box = (x, y, x + width, y + height)
            if filled and _attempt < 15:
                # Blending loses a thin object in busy texture (forest) and
                # leaves the old ground visible on flat ground (water): look
                # for moderately textured ground first.
                region = image[y:y + height:2, x:x + width:2]
                texture = region.mean(axis=2).std() if region.size else 0.0
                if not BLEND_MIN_TEXTURE <= texture <= BLEND_MAX_TEXTURE:
                    continue
            if not overlaps(box, occupied):
                # Only cut-outs with a real silhouette get a shadow: a
                # rectangle's shadow would be a giveaway.
                if rng.random() < SHADOW_SHARE and (patch[:, :, 3] > 0).mean() < FILLED_MASK:
                    cast_shadow(image, patch, x, y, rng)
                if not ((filled or rng.random() < BLEND_SHARE) and paste_blended(image, patch, x, y)):
                    paste(image, patch, x, y)
                occupied.append(box)
                labels.append((name, *box))
                break
    return image, labels


def pick_centre(level: int, labels, rng: random.Random):
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    if level == 0:
        return IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
    if level == 2 and labels and rng.random() < L2_ON_OBJECT_SHARE:
        _, x1, y1, x2, y2 = rng.choice(labels)
        # Anywhere that still shows the object, not always dead centre.
        cx = (x1 + x2) / 2 + rng.uniform(-400, 400)
        cy = (y1 + y2) / 2 + rng.uniform(-220, 220)
    else:
        cx, cy = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)
    return int(min(max(cx, min_x), max_x)), int(min(max(cy, min_y), max_y))


def render_view(image, labels, level: int, cx: int, cy: int):
    """Crop and shrink like the evaluator; return (view, yolo label lines)."""
    rx1, ry1, rx2, ry2 = source_region_for_view(level, cx, cy)
    view = image[ry1:ry2, rx1:rx2]
    if view.shape[1] != VIEW_SIZE[0]:
        view = cv2.resize(view, VIEW_SIZE, interpolation=cv2.INTER_AREA)
    if NOISE_SHARE and random.random() < NOISE_SHARE:
        # Sensor-like grain, so the model does not rely on perfectly clean pixels.
        view = np.clip(view + np.random.normal(0, random.uniform(1.5, 5), view.shape), 0, 255).astype(np.uint8)
    region_width, region_height = rx2 - rx1, ry2 - ry1
    scale = VIEW_SIZE[0] / region_width

    lines = []
    for name, x1, y1, x2, y2 in labels:
        ix1, iy1, ix2, iy2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
            continue
        if min(ix2 - ix1, iy2 - iy1) * scale < MIN_BOX_VIEW_PIXELS:
            continue
        xc = ((ix1 + ix2) / 2 - rx1) / region_width
        yc = ((iy1 + iy2) / 2 - ry1) / region_height
        w, h = (ix2 - ix1) / region_width, (iy2 - iy1) / region_height
        lines.append(f'{CLASS_INDEX[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
    return view, lines


def make_scene(job):
    index, split, seed, out, extension = job
    rng = random.Random(seed)
    # render_view's noise uses the module generators: keep scenes reproducible.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    image, labels = build_scene(rng)
    written = 0
    for level, count in VIEWS_PER_LEVEL.items():
        for view_number in range(count if level else 1):
            cx, cy = pick_centre(level, labels, rng)
            view, lines = render_view(image, labels, level, cx, cy)
            stem = f's{index:05d}_L{level}_{view_number}'
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == 'png' else [cv2.IMWRITE_JPEG_QUALITY, 95]
            cv2.imwrite(str(out / 'images' / split / f'{stem}.{extension}'), view, params)
            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            written += 1
    return written


def write_preview(out: Path, split: str, count: int = 12):
    """Draw the labels onto a few views so they can be checked by eye."""
    tiles = []
    for image_path in sorted((out / 'images' / split).iterdir())[:count]:
        view = cv2.imread(str(image_path))
        label_path = out / 'labels' / split / f'{image_path.stem}.txt'
        for line in label_path.read_text().split('\n'):
            if not line:
                continue
            cls, xc, yc, w, h = line.split()
            xc, yc, w, h = float(xc) * 960, float(yc) * 540, float(w) * 960, float(h) * 540
            p1 = (int(xc - w / 2) - 2, int(yc - h / 2) - 2)
            p2 = (int(xc + w / 2) + 2, int(yc + h / 2) + 2)
            cv2.rectangle(view, p1, p2, (0, 0, 255), 1)
            cv2.putText(view, OBJECT_CLASSES[int(cls)], (p1[0], p1[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(view, image_path.stem, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(view)
    columns = 3
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    path = out / f'preview_{split}.jpg'
    cv2.imwrite(str(path), sheet)
    print(f'preview: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenes', type=int, default=1200, help='Synthetic 4K scenes (6 views each).')
    parser.add_argument('--val-share', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patches', type=Path, default=HERE.parent / 'data' / 'patches')
    parser.add_argument('--out', type=Path, default=HERE.parent / 'data' / 'yolo')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png',
                        help='png matches what the evaluator sends; jpg is ~6x smaller.')
    parser.add_argument('--real-backgrounds', type=Path, nargs='*', default=[],
                        help='Folders of backgrounds cut from the recorded flight '
                             '(training/make_real_backgrounds.py); drawn REAL_BACKGROUND_SHARE of the time.')
    parser.add_argument('--real-share', type=float, default=None,
                        help=f'Override that share (default {REAL_BACKGROUND_SHARE}).')
    parser.add_argument('--backgrounds', type=Path, nargs='*', default=[],
                        help='Folders searched for large aerial photos to paste onto.')
    parser.add_argument('--extra-patches', type=Path, nargs='*', default=[],
                        help='More patch folders (e.g. data/patches_val), used for part of the pastes.')
    parser.add_argument('--helsinki-share', type=float, default=HELSINKI_SHARE,
                        help='Share of scenes on Helsinki frames when --backgrounds is given.')
    parser.add_argument('--class-weights', default='',
                        help='name=weight,... pasted more (or less) often, e.g. small_launcher=2,ta-ta=2')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--preview', action='store_true', help='Also draw labels onto a few views.')
    args = parser.parse_args()

    for kind in ('images', 'labels'):
        for split in ('train', 'val'):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)

    real_backgrounds = find_backgrounds(args.real_backgrounds)
    if args.real_backgrounds:
        print(f'{len(real_backgrounds)} backgrounds from the recorded flight', flush=True)
        if not real_backgrounds:
            raise SystemExit('No backgrounds found in ' + ', '.join(map(str, args.real_backgrounds)))
    backgrounds = find_backgrounds(args.backgrounds)
    if args.backgrounds:
        print(f'{len(backgrounds)} background photos found', flush=True)
        for path in backgrounds[:5]:
            print('  e.g.', path)
        if not backgrounds:
            raise SystemExit('No background photos found in ' + ', '.join(map(str, args.backgrounds)))

    weights = {}
    for item in filter(None, args.class_weights.split(',')):
        name, value = item.split('=')
        if name not in OBJECT_CLASSES:
            raise SystemExit(f'unknown class {name}')
        weights[name] = float(value)

    val_count = max(1, int(args.scenes * args.val_share))
    jobs = [
        (i, 'val' if i < val_count else 'train', args.seed * 1_000_003 + i, args.out, args.format)
        for i in range(args.scenes)
    ]
    with Pool(args.workers, initializer=_init_worker, initargs=(args.patches, backgrounds, [p for p in args.extra_patches if p.exists()], weights,
                                                                   args.helsinki_share, real_backgrounds,
                                                                   args.real_share)) as pool:
        total = 0
        for done, written in enumerate(pool.imap_unordered(make_scene, jobs), 1):
            total += written
            if done % 50 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} scenes, {total} views', flush=True)

    names = '\n'.join(f'  {i}: {name}' for i, name in enumerate(OBJECT_CLASSES))
    (args.out / 'data.yaml').write_text(
        f'path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n'
    )
    print(f'dataset: {args.out / "data.yaml"}')

    if args.preview:
        write_preview(args.out, 'train')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

## `training/train_remote.sh` (excerpt: dataset build and training call)

`MODEL`, `EPOCHS` (40), `SCENES` (1600), `BATCH` (16), `IMGSZ`, `REAL_SHARE` (0.6) come from the environment; v8 used `MODEL=yolo11m.pt IMGSZ=1280 BATCH=12` and its own `WEIGHTS`.

```bash
WEIGHTS=${WEIGHTS:-tank=2.0,spacecraft=1.8,mine_roller=1.6,large_tower=1.6,small_launcher=1.3,condor=1.3,ta-ta=1.3,medium_plane=1.3,medium_launcher=1.3}
...
echo "== dataset"
cd "$FLYBY"
export OPENCV_LOG_LEVEL=ERROR
python training/extract_patches.py
REAL_ARGS=()
if [ -d "$REAL_BACKGROUNDS" ]; then
    echo "== real backgrounds: $(ls "$REAL_BACKGROUNDS" | wc -l) frames of the flight itself"
    REAL_ARGS=(--real-backgrounds "$REAL_BACKGROUNDS" --real-share "$REAL_SHARE")
else
    echo "!! $REAL_BACKGROUNDS missing - training on stock photos only, which is what v1-v5 did" >&2
fi
python training/make_dataset.py \
    --scenes "$SCENES" --out "$WORK/yolo" \
    --backgrounds "$WORK/backgrounds" \
    "${REAL_ARGS[@]}" \
    --extra-patches training/patches_val \
    --helsinki-share 0.1 \
    --class-weights "$WEIGHTS"

echo "== train $MODEL, $EPOCHS epochs"
MODEL="$MODEL" EPOCHS="$EPOCHS" BATCH="$BATCH" IMGSZ="$IMGSZ" WORK="$WORK" NAME="$NAME" \
  PRETRAINED="${PRETRAINED:-}" python - <<'PY'
import os, shutil, psutil
from ultralytics import YOLO

work, name = os.environ['WORK'], os.environ['NAME']
# Caching the images in RAM keeps a fast GPU from waiting on PNG decoding;
# ~1.6 GB per 1000 images at 960x540.
images = len(list((__import__('pathlib').Path(work) / 'yolo/images/train').glob('*')))
free_gb = psutil.virtual_memory().available / 1e9
cache = 'ram' if free_gb > images * 0.0017 + 8 else 'disk'
print(f'{images} training images, {free_gb:.0f} GB RAM free -> cache={cache}')

# MODEL may be a .pt (fine-tune) or an architecture .yaml (new head shape).
# With a .yaml, PRETRAINED names weights to warm-start from: the backbone and
# most of the neck transfer by matching state_dict keys, and layers whose index
# shifted (everything after the new P2 branch) start fresh. Expect ~60%.
model = YOLO(os.environ['MODEL'])
pretrained = os.environ.get('PRETRAINED', '')
if pretrained:
    before = sum(p.numel() for p in model.model.parameters())
    model = model.load(pretrained)
    print(f'warm-started from {pretrained} ({before/1e6:.1f}M param model)')
model.train(
    data=f'{work}/yolo/data.yaml',
    imgsz=int(os.environ['IMGSZ']),
    epochs=int(os.environ['EPOCHS']),
    batch=int(os.environ['BATCH']),
    device=0,
    cache=cache,
    workers=min(16, os.cpu_count() or 8),
    # The drone flies at a fixed altitude: sizes vary little, and the pasting
    # already rotates and recolours each object.
    scale=0.1,
    degrees=0.0,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.03,
    project=f'{work}/runs',
    name=name,
    plots=True,
)
shutil.copy(f'{work}/runs/{name}/weights/best.pt', f'{work}/{name}.pt')
print(f'\nDone: {work}/{name}.pt')
PY

echo
echo "Copy it off the box before destroying the instance, e.g.:"
echo "  scp -P <port> root@<host>:$WORK/$NAME.pt ."
```
