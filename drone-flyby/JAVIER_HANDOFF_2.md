# Drone Flyby — handoff 2 to Javier (19 Sep 2026, ~02:30 CEST)

Written by Franek's Claude after a night of measurement on your branch
`drone-flyby-real-backgrounds` (head `04c0fd3`). Part 1 is for you: what the
project is, where it stands, and what the new measurements say. Part 2 is a
prompt for your Claude Code: paste everything from "PART 2" down, and it has
all the context, reference code, commands and decision rules it needs.

**Deadline: 20 Sep 16:00 CEST.** Validation attempts are unlimited. The final
evaluation is **one** attempt on a **different** 250-frame flight.

---

# PART 1 — Summary for Javier

## The challenge, in five lines

A drone flies straight over a rendered landscape (600 m, 3 fps, 4K source
frames). Each frame we get one 960×540 view (Level 0 = whole frame at 1/4
resolution, L1 = a quarter at 1/2, L2 = a sixteenth at native). We answer with
boxes for the **whole** frame and pick the next view. The score is COCO mAP at
IoU 0.50, **macro-averaged over the classes present** in the ground truth.

## Where we stand

| | Ours (`main` / `franek-drone-flyby-motion-fit`) | Yours (`drone-flyby-real-backgrounds`) |
|---|---|---|
| Relationship | older | **superset**: merged our motion fit |
| Served models | v4 + v5 alternating | v4@960 + v6@1280, `BOTH_MODELS=1` |
| Best real validation | 0.14–0.24 | **0.3048** (Mac) |
| Extras | online motion fit | per-model imgsz, robustness fixes, `score_offline.py` + ignore regions, v6/v7 weights, real backgrounds |

**Your branch is the base for everything below.** Real validation history
(from `HANDOVER.md`): v2 0.097 → v3 0.132 → v4 0.1445 → v4+v6 alternating
0.2365 → v4@960+v6@1280 both 0.3048 (Mac; 0.2845 on the rented box). Dead:
hybrid camera 0.1234, v7 0.2611, `BOX_SCALE=0.8` 0.017.

Leaderboard (Drone Flyby, raw): CarlN 0.908, cosearch 0.844, Håkon Kjelseth
0.841, Elemental hero 0.776, Backprop Boys 0.749, Cybotrix 0.690, Iftikhar
Amiri 0.680, Lisan al-Gaibs 0.650 … **us 0.305**. The public competitor repos
([fierceviking/Nordic-AI-Cup-2026](https://github.com/fierceviking/Nordic-AI-Cup-2026)
branch `mtp_drone`: 0.10 → 0.15 real; [huyquoctrinh/drone-det](https://github.com/huyquoctrinh/drone-det):
Helsinki-only YOLO) are behind us; the 0.8–0.9 teams are not public. Part of
their lead is likely validation overfitting (the flight is deterministic and
recording it is allowed), which will not carry to the final flight.

## What the measurements say (all offline, on the full recorded run `2c09583d65c34d7db47b32c66c375b02`)

That is the only complete recording on Franek's laptop (camera `full`, 17 Sep).
Detections are v4@960 + v6@1280 (cached), replayed through `flyby.predict`.

### 1. The biggest loss is probably box shape, not detection

- The official boxes are the **projected 3D box** of each object: loose, with
  rotor span, wingtips and height included. Helsinki shows it plainly
  (helicopter 116×94 with rotors, jet 77×82 around a ~55 px jet). The frame
  metadata (`GimbalCamera`, NED pose, `z: -600`) looks like AirSim, whose 2D
  boxes are exactly that projection.
- `make_dataset.py` labels every pasted cut-out with the **tight box around its
  mask** (`transform_patch` crops to the alpha, and the pasted rectangle becomes
  the label). So our models learned tight boxes.
- On Helsinki, a *perfect* tight box overlaps the official box by:
  jet_plane **0.33**, small_launcher **0.23**, condor **0.42**, medium_plane
  0.50, small_plane 0.54, tank 0.55, large_launcher 0.59, small_tower 0.63,
  hangar 0.75, ta-ta 0.75, spacecraft 0.80, jammer 0.84, mine_roller 0.85;
  helicopter, large_tower and medium_launcher 1.0 (their cut-outs keep the
  whole box). The scorer needs 0.50.
- On validation our jet boxes are ~48×46. The validation jets sit at a yaw much
  like Helsinki's, where the official box is ~77×82, so our jet hits are near or
  below IoU 0.5. Our `validation_objects.json` truth is tight too (it came from
  our own detections), so `score_offline.py` cannot see any of this.
- **Falsification test that it passed.** Rebuild the truth "loose" (tight truth
  grown by the Helsinki per-class ratios) and replay:

  | truth used offline | served config | `BOX_SCALE=0.8` | ratio |
  |---|---|---|---|
  | tight (current `validation_objects.json`) | 0.380 | 0.351 | 0.93 |
  | loose (tight × Helsinki ratios) | **0.304** | 0.103 | **0.34** |
  | **real validation** | **0.3048** | 0.017 (v4, 18 Sep) | **0.12** |

  The loose truth reproduces the real score of the served config almost exactly,
  and reproduces the collapse your `BOX_SCALE=0.8` run showed. The tight truth
  reproduces neither.

### 2. v6's paste scale came from the same confusion

The comment above `HELSINKI_SCALE_RANGE` in `make_dataset.py` measured
validation objects at 0.45–0.98× their Helsinki box diagonal (median 0.70) and
set the paste range to (0.45, 1.00). Those ratios are what you get from comparing
**tight** validation boxes with **official** Helsinki boxes: tank 0.75 measured
vs 0.746 predicted by the convention alone, jet_plane 0.56 vs 0.58,
small_launcher 0.49 vs 0.46, jammer 0.89 vs 0.92. For those four classes, whose
masks are clean, the convention explains the whole ratio. For the rest it
explains part of it. Ground motion agrees: 68.37 px/frame at the centre against
Helsinki's 66.21. At the same speed per frame, validation objects are ~1.03×
Helsinki, not 0.7×. We have been pasting objects too small **and** labelling
them too tight.

### 3. Motion is biased, and would break on a different flight

- The whole flight is **one constant homography**: 393k SIFT matches between
  consecutive views, median residual 0.36 px per frame. True motion: 68.37
  px/frame at the centre (54.45 at the top, 82.86 at the bottom, ±12.85 in x at
  the sides).
- `flyby`'s online fit converges to **69.35** (1.4 % fast). `score_offline.py`
  carries its truth at 68.93 (also fast). A 1 px/frame error is ~20 px after
  20 frames, which kills IoU on tank-sized objects.
- Worse for the final: the fit only learns from detections matched to tracks,
  and tracks start out carried by the Helsinki prior. The organizers said the
  evaluation flight's direction may differ. In another direction, matches never
  form. More than 15 px/frame faster or slower, and `MOTION_MAX_CORRECTION`
  rejects the fit. Either way the tracker stays on the Helsinki prior and
  tracking collapses.
- **Fix, built and tested:** estimate motion by registering consecutive views
  (reference code in Part 2). First fix at frame 5 (67.8), within 0.2 px/frame by
  frame 15, 61 ms per frame on a laptop CPU (probably less on the M4). Offline:
  0.380 → **0.405**.

### 4. The evaluator sometimes re-sends the previous frame's render

In that run the view for frame 9 is pixel-identical (0.03 px) to frame 8's
ground, and 9 → 10 shows two frames of motion. Same at frame 239. The tracker
places those detections a whole frame (70–80 px) off. The registration step
detects it; moving that view's detections one frame forward: 0.405 → **0.415**.

### 5. Two answer-policy settings pay once motion is right

`UNSEEN_DECAY=1.0` and `NEW_TRACK_CONFIDENCE=0.15`: 0.415 → **0.445** together.
Everything else I swept moved ±0.015 or lost (`RUNNER_UPS=0` −0.021,
`NEW_TRACK_CONFIDENCE=0.35` −0.017).

### 6. False positives cost as much as misses

Deleting every false positive would lift offline AP from 0.405 to 0.557. They
are confident detector hallucinations: large_tower alone has 745 background
false positives (71 ranked above its median true hit), small_plane 273 (47
above), spacecraft 209 (59 above). No tracker knob fixes that; only the
detector can.

### 7. The detector misses objects that are plainly visible

Recall on confirmed objects: tank 44 %, mine_roller 47 %, spacecraft 16 %,
small_launcher 3 %. Crops show every one of them clearly at Level 1 (frame 54: a
tank in a clearing, no detection). It is not a resolution floor.

### Where that leaves us (loose-truth estimate)

| step | offline, loose truth |
|---|---|
| served today | 0.304 (real 0.3048) |
| + registration motion, re-sent fix, decay 1.0, new-track 0.15 | 0.347 |
| + box growth, isotropic Helsinki factor capped at 1.3 | 0.441 |
| + same, capped at 1.5 | 0.459 |

If the official boxes were tight after all, the cap-1.3 growth would cost ~0.07
instead. One real run decides it: the effect is +0.09 or −0.07, 7–9× the noise
floor either way.

## The plan

1. **19 Sep morning, no retraining (by ~10:00):** registration motion +
   re-sent fix + decay/new-track defaults + a box-growth setting + scorer fixes
   (Tasks 1–4).
2. **Two real runs on the Mac (by ~12:00):** A (tracker fixes) and B (A +
   growth). The decision tree is in Task 5.
3. **In parallel on the GPU box (start by ~10:00, trained by ~13:00):** v9 = the
   v6 recipe with official-convention labels and the paste scale fixed (Task 6).
   Judge it offline on the loose truth, then real run C (by ~18:00).
4. **20 Sep:** freeze by 12:00, confirmation run, evaluation by 14:00 (≥2 h
   buffer).

**What your Claude needs you for:** starting each validation attempt on the
competition site, keeping the Mac serving on a named tunnel, starting the GPU
box, and approving the final config.

---

# PART 2 — Prompt for Claude Code

> You are picking up the Drone Flyby entry of the Nordic AI Cup 2026 in this
> repo. Read `drone-flyby/HANDOVER.md` for history; this prompt supersedes its
> "what to do next" sections. Everything below was measured on 19 Sep by
> another Claude session on a teammate's laptop. Re-verify the offline numbers on
> this machine's recordings before relying on them. Work through the tasks in
> order, commit after each, and stop for the human wherever a task says so.

## 0. Ground rules

- **Branch:** create `drone-flyby-v9` from `origin/drone-flyby-real-backgrounds`
  and work there. Do not push to `main`. Commit after each task with a message
  that states the measured effect.
- **Deadline:** 20 Sep 16:00 CEST. The final evaluation is one attempt. Freeze
  the config by ~12:00 on 20 Sep.
- **Never start a validation or evaluation attempt yourself.** Prepare the exact
  server command, run `tools/preflight.py`, then tell the human the attempt is
  ready and what to compare it against. The human starts it.
- **Keep old behaviour reachable.** Every change gets a setting that restores the
  old behaviour, so one environment variable undoes it on the Mac.
- **Noise floor:** real runs vary about ±0.01. Compare real runs only on the same
  machine (the Mac). The rented box scores ~0.02 lower on an identical config.
- **Silent failures have cost attempts before.** After any restart, read `/api`
  and confirm what is really loaded (`models_loaded`, `imgsz`, and the new fields
  you add below). `docker compose up -d` does not rebuild; on the Mac the service
  runs without Docker (`DRONE_DEVICE=mps`).
- **Known doc bug:** `DRONE_SET` accepts numeric settings only. The comment in
  `flyby.py` suggesting `DRONE_SET=LEVEL_WEIGHT={0:0.4,...}` would make the
  service exit at startup. Fix the comment when you touch that area.

## 1. Facts you need

- The flight is one constant homography. Reference, validation flight, measured
  with SIFT over 393k matches (source pixels, one frame step):

  ```
  H = [[1.00661665, -0.00154514, -12.67095606],
       [0.00000212,  1.01320417,  53.11706381],
       [0.,         -0.0000008,    1.        ]]
  affine displacement fit (flyby.MOTION form a,b,c,d,e,f):
       (-14.212614, 0.007415, -0.000006, 52.535361, -0.000001, 0.01493)
  centre 68.37 px/frame, top (y=100) 54.45, bottom (y=2000) 82.86, x at the sides ±12.85
  ```

  `flyby.MOTION` (Helsinki prior) is 66.21 at the centre; the online fit
  converges to 69.35; `score_offline.py` carries its truth at 68.93. Carry error
  after 20 frames: a flight-fitted affine model, median 4 px; the Helsinki prior,
  46 px.
- **Box convention.** Official boxes = projected 3D box (loose). Our training
  labels and our truth file = tight mask boxes. Per-class Helsinki ratios
  (official ÷ tight mask box, medians over the cut-outs in `data/patches`,
  measured on 17 Sep patches; **re-measure on this machine's patches**, see
  Task 6a):

  | class | w | h | isotropic √(w·h) | IoU(tight, official) |
  |---|---|---|---|---|
  | jet_plane | 1.64 | 1.82 | 1.728 | 0.33 |
  | small_launcher | 1.92 | 2.33 | 2.115 | 0.23 |
  | condor | 1.52 | 1.59 | 1.555 | 0.42 |
  | medium_plane | 1.54 | 1.30 | 1.415 | 0.50 |
  | small_plane | 1.59 | 1.16 | 1.358 | 0.54 |
  | tank | 1.32 | 1.37 | 1.345 | 0.55 |
  | large_launcher | 1.23 | 1.39 | 1.308 | 0.59 |
  | small_tower | 1.20 | 1.33 | 1.263 | 0.63 |
  | hangar | 1.08 | 1.23 | 1.153 | 0.75 |
  | ta-ta | 1.18 | 1.12 | 1.150 | 0.75 |
  | spacecraft | 1.10 | 1.14 | 1.120 | 0.80 |
  | mine_roller | 1.09 | 1.08 | 1.085 | 0.85 |
  | jammer | 1.06 | 1.10 | 1.080 | 0.84 |
  | helicopter, large_tower, medium_launcher | 1.0 | 1.0 | 1.0 | 1.0 (cut-outs keep the whole box) |

  Our current-view boxes on validation (v4@960, v6@1280) are 0.9–1.17× the
  tight truth for every class except helicopter (1.4–1.6×, i.e. already loose).
- **Re-sent renders.** About 1 % of views are the previous frame's render under
  a new frame number (frames 9 and 239 in run `2c09583d…`).
- **Loss split** (tight truth, served pair with correct motion):

  | class | object-frames | AP | AP if no FPs | recall | misses: none / IoU 0.3–0.5 / wrong class |
  |---|---|---|---|---|---|
  | tank | 221 | 0.340 | 0.436 | 44 % | 89 / 14 / 19 |
  | helicopter | 100 | 0.346 | 0.564 | 57 % | 8 / 35 / 0 |
  | small_plane | 99 | 0.383 | 0.861 | 87 % | 3 / 8 / 2 |
  | jammer | 99 | 0.370 | 0.584 | 67 % | 10 / 23 / 0 |
  | large_tower | 98 | 0.217 | 0.644 | 64 % | 19 / 2 / 14 |
  | small_tower | 78 | 0.567 | 0.614 | 62 % | 23 / 7 / 0 |
  | hangar | 70 | 0.911 | 0.911 | 91 % | 6 / 0 / 0 |
  | jet_plane | 66 | 0.828 | 0.842 | 85 % | 9 / 0 / 0 |
  | mine_roller | 66 | 0.430 | 0.465 | 47 % | 34 / 1 / 0 |
  | spacecraft | 64 | 0.013 | 0.144 | 16 % | 48 / 1 / 2 |
  | small_launcher | 33 | 0.040 | 0.040 | 3 % | 31 / 1 / 0 |
  | large_launcher | 17 | 0.419 | 0.584 | 59 % | 6 / 0 / 1 |

  Helicopter's IoU 0.3–0.5 misses are a truth-file problem: our helicopter boxes
  are loose, the truth is tight.
- **Classes absent from the truth** (condor, medium_launcher, medium_plane,
  ta-ta) get 1380 false positives from us on validation. They are free there,
  because absent classes are not scored, but not on a final flight that contains
  them.
- The second half of the flight (frames ~150–249) is dense city and marina with
  almost no objects. The truth file is close to complete; its problem is box
  shape.

## 2. Task 1 — Motion from image registration, plus the re-sent render fix

**Goal:** replace the detection-fitted online motion with an estimate from
registering consecutive views. It must need no prior and assume no direction.
Expected offline effect on run `2c09583d…`: 0.380 → 0.405, then 0.415 with the
re-sent fix.

### 1a. New file `drone-flyby/registration.py` (tested reference implementation)

```python
"""Ground motion from registering consecutive views; no prior, no direction assumed.

update(frame, image, source_region) once per answered view, in frame order.
It returns the view's image lag: 1 means the evaluator re-sent the previous
frame's render under a new frame number, 0 means normal, None means it could
not tell. `.motion` is the per-frame affine displacement in flyby.MOTION's form,
    dx = a + b*x + c*y,  dy = d + e*x + f*y   (source pixels, per frame),
or None until two frame pairs agree.

Measured on the recorded validation flight: first fix at frame 5 (67.8 px/frame
at the centre against a true 68.37), within 0.2 px/frame by frame 15, rms 0.4-0.8
px; ~61 ms per view on a Ryzen AI 5 laptop CPU with SIFT at 1500 features.
"""
import cv2
import numpy as np


def _affine_fit(src, dst):
    d = dst - src
    design = np.column_stack([np.ones(len(src)), src[:, 0], src[:, 1]])
    cx = np.linalg.lstsq(design, d[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(design, d[:, 1], rcond=None)[0]
    return np.concatenate([cx, cy])


def _apply(m, pts, steps=1):
    out = pts.astype(np.float64).copy()
    for _ in range(steps):
        x, y = out[:, 0], out[:, 1]
        out = np.column_stack([x + m[0] + m[1] * x + m[2] * y, y + m[3] + m[4] * x + m[5] * y])
    return out


class RegistrationMotion:
    MIN_INLIERS = 40          # per accepted frame pair
    MIN_SPREAD = 250          # source px: inliers must span this much in x AND y
    MAX_PAIRS = 60            # most recent good pairs kept for the fit
    MAX_SPEED = 400           # source px/frame: anything faster is a bad match
    PAIR_TOLERANCE = 3.0      # px: a pair whose median residual exceeds this is an outlier
    RATIO = 0.75              # Lowe ratio test

    def __init__(self, features=1500):
        self.detector = cv2.SIFT_create(nfeatures=features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.prev = None          # (frame, points in source px, descriptors, scale)
        self.pairs = []           # (src, dst) inlier arrays of accepted 1-frame pairs
        self.motion = None
        self.rms = None

    def update(self, frame, image_bgr, region):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        rx1, ry1, rx2, ry2 = region
        scale = (rx2 - rx1) / gray.shape[1]
        if descriptors is None or len(keypoints) < 20:
            self.prev = None
            return None
        points = np.float32([(k.pt[0] * scale + rx1, k.pt[1] * scale + ry1) for k in keypoints])
        lag = None
        current = (frame, points, descriptors, scale)
        if self.prev is not None and frame - self.prev[0] == 1:
            lag = self._pair(self.prev, current)
        if lag == 1:
            return lag                 # keep the previous view as the reference
        self.prev = current
        return lag

    def _match(self, before, after):
        _, p1, d1, s1 = before
        _, p2, d2, s2 = after
        matches = self.matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in (x for x in matches if len(x) == 2) if m.distance < self.RATIO * n.distance]
        if len(good) < self.MIN_INLIERS:
            return None
        src = p1[[m.queryIdx for m in good]]
        dst = p2[[m.trainIdx for m in good]]
        fast = np.linalg.norm(dst - src, axis=1) <= self.MAX_SPEED
        src, dst = src[fast], dst[fast]
        if len(src) < self.MIN_INLIERS:
            return None
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5 * max(s1, s2))
        if H is None:
            return None
        inliers = inliers.ravel().astype(bool)
        src, dst = src[inliers].astype(np.float64), dst[inliers].astype(np.float64)
        if (len(src) < self.MIN_INLIERS or np.ptp(src[:, 0]) < self.MIN_SPREAD
                or np.ptp(src[:, 1]) < self.MIN_SPREAD):
            return None
        return src, dst

    def _pair(self, before, after):
        pair = self._match(before, after)
        if pair is None:
            return None
        src, dst = pair
        if np.median(np.linalg.norm(dst - src, axis=1)) < 1.0:
            return 1                   # identical ground: a re-sent render
        if self.motion is not None:
            errors = [np.median(np.linalg.norm(_apply(self.motion, src, k) - dst, axis=1)) for k in (1, 2)]
            if errors[1] < errors[0]:
                return None            # two frames of motion: the previous view was stale
            if errors[0] > 4 * self.PAIR_TOLERANCE:
                return None
        self.pairs.append((src, dst))
        del self.pairs[:-self.MAX_PAIRS]
        self._fit()
        return 0

    def _fit(self):
        if len(self.pairs) < 2:
            return
        good = list(range(len(self.pairs)))
        for _ in range(3):
            src = np.concatenate([self.pairs[i][0] for i in good])
            dst = np.concatenate([self.pairs[i][1] for i in good])
            model = _affine_fit(src, dst)
            residuals = [np.median(np.linalg.norm(_apply(model, s) - d, axis=1)) for s, d in self.pairs]
            new_good = [i for i, r in enumerate(residuals) if r <= self.PAIR_TOLERANCE]
            if len(new_good) < 2 or new_good == good:
                break
            good = new_good
        if len(good) < 2:
            return
        self.pairs = [self.pairs[i] for i in good] + [p for i, p in enumerate(self.pairs) if i not in good][-5:]
        src = np.concatenate([self.pairs[i][0] for i in range(len(good))])
        dst = np.concatenate([self.pairs[i][1] for i in range(len(good))])
        model = _affine_fit(src, dst)
        self.motion = tuple(float(v) for v in model)
        self.rms = float(np.sqrt(np.mean(np.sum((_apply(model, src) - dst) ** 2, axis=1))))
```

Why each guard exists, from what broke during testing:
- Without pair-level outlier rejection, the one stale pair (8 → 9, zero motion)
  plus its successor (9 → 10, double motion) dragged the centre estimate to 39.8
  px/frame until frame 30.
- `MIN_SPREAD` rejects the thin-strip overlaps you get on row changes of the
  `full` sweep (TR → BR, BL → TL). Their homographies are unconstrained.
- ORB is faster (47 ms) but three times noisier (rms 1.8 against 0.6). Keep SIFT.

### 1b. Wire it into `flyby.py`

1. Settings, next to the motion settings:

   ```python
   from registration import RegistrationMotion

   # Ground motion from registering consecutive views (registration.py) instead of
   # fitting it to our own re-detections. Measured on run 2c09583d: the detection
   # fit converges to 69.35 px/frame at the centre against a true 68.37, and it
   # can never converge on a flight in another direction. 0 restores the old fit.
   REGISTRATION = 1
   # The evaluator sometimes re-sends the previous frame's render under a new frame
   # number (frames 9 and 239 of run 2c09583d). Detections from such a view are
   # moved one frame of motion forward. 0 turns that off.
   FIX_RESENT = 1
   ```

2. `Sequence` gets `registration: Optional[RegistrationMotion] = None`,
   `registered_frame: int = -1`, `from_registration: bool = False`, and a
   `lock: threading.Lock = field(default_factory=threading.Lock)`.

3. `predict()`: decode the view **once** (today it decodes inside the `detect`
   call). If `REGISTRATION` is on and the image decoded, then under
   `state.lock`, and only when `request.frame > state.registered_frame`
   (requests can overlap and must be registered in frame order): create the
   estimator if needed, call `lag = state.registration.update(request.frame,
   image, view.source_region_xyxy)`, and set `state.registered_frame`. Wrap it in
   `try/except` with `logger.exception`: a registration failure must never cost
   the frame. Pass `lag` to `update_tracks`.
   Optional, only if preflight shows the round trip p90 above ~150 ms:
   registration is CPU and the detector is MPS, so run them concurrently in a
   small `ThreadPoolExecutor` and join before `update_tracks`.

4. `update_tracks(state, frame, level, region, detections, lag=0)`:
   - At the top, before tracks are advanced: if `REGISTRATION` and the estimator
     has a `.motion`, then on the **first** time only, re-place every track that
     has a `seen_box`, carried from its last observation with the new motion:
     `track.box = advance(track.seen_box, track.frame - track.seen_frame, reg)`
     (the existing loop then carries it to `frame`). Then set
     `state.motion = reg` and `state.from_registration = True`. On a flight in a
     new direction, tracks from frames 1–4 were carried the wrong way until now;
     this puts them back.
   - If `lag == 1 and FIX_RESENT`: replace each detection's box with
     `advance(box, 1, state.motion)` before matching, and **skip the miss
     counting** for this view (its region is one frame stale).
   - At the end, only run `state.motion = fit_motion(...)` when
     `not state.from_registration`. The old fit stays as the fallback if
     registration never produces a fix (e.g. featureless water).

5. `/api` in `api.py`: add `"registration": flyby.REGISTRATION` and
   `"fix_resent": flyby.FIX_RESENT`, so preflight shows what is live.

### 1c. Make the offline tools feed images to the estimator

`tools/score_offline.py` `from_replay()` stubs `flyby.decode_view` to `None`.
When `flyby.REGISTRATION` is on, stub it instead with a function that returns
`cv2.imread` of the recorded PNG for the current key (the replay already knows
`current['key']`). Detections still come from the cache.

### 1d. Acceptance

- `python tools/score_offline.py --replay --run <any full-camera run> --model
  models/drone-yolo11n-v4.pt:960 --model-alt models/drone-yolo11s-v6.pt:1280
  --set BOTH_MODELS=1` with `--set REGISTRATION=0` and then `=1`. Expect ≈ +0.025
  (on `2c09583d`: 0.380 → 0.405 with the truth carried by measured motion, see
  Task 4), and ≈ +0.01 more from `FIX_RESENT`. Run it on `5ace5364…` and on the
  recording of the 0.3048 run if it exists.
- **Direction test (must pass before serving):** feed a recording's views
  flipped vertically (`img[::-1]`, region `y → 2160 - y`, swapped). The
  estimator's centre dy must come out ≈ −68.4. Do the same with a horizontal
  flip for dx. Then replay the flipped run through the tracker: it must score
  about the same as the unflipped one. Flip the cached detections' boxes and the
  truth boxes the same way. With `REGISTRATION=0` the flipped run should
  collapse; that is the failure this task removes.
- Print the estimator's convergence on one run: the centre dy at frames 5, 10,
  20, 60 and 249 should be within 1 px of 68.4 from frame 5.
- `tools/preflight.py` against the Mac on localhost: round-trip p90 comfortably
  under 333 ms.

Commit: "Drone Flyby: ground motion from image registration; move re-sent renders
one frame".

## 3. Task 2 — Answer-policy defaults

`UNSEEN_DECAY = 1.0` (was 0.97) and `NEW_TRACK_CONFIDENCE = 0.15` (was 0.25;
env `DRONE_TRACK_CONF`). With registration on, together +0.030 offline on
`2c09583d` (0.415 → 0.445, tight truth). Measure both on your runs with
`--set` before changing the defaults. If either costs anything on the loose truth
(Task 4), leave it at its old value. Update the comments that quote the old
measurements. They were taken with a drifting motion, so a fading confidence was
compensating for boxes that had drifted off target.

## 4. Task 3 — Box growth toward the official convention (report-time)

Add, in `flyby.py`:

```python
# Box growth toward the official box convention. The evaluator's boxes are the
# projected 3D box of each object (see HANDOFF 2): on Helsinki a perfect tight box
# overlaps them by only IoU 0.33 for jet_plane, 0.23 small_launcher, 0.55 tank.
# Our models learned tight boxes from mask-cropped cut-outs, so reported boxes
# are grown about their centre by a per-class factor. Helicopter, large_tower and
# medium_launcher cut-outs keep their whole box, so those come out loose already.
#   DRONE_BOX_GROW=helsinki          the isotropic Helsinki factors below, capped
#   DRONE_BOX_GROW_CAP=1.3           ... at this (default 1.3)
#   DRONE_BOX_GROW=1.2               one factor for every class except the three
#   DRONE_BOX_GROW=jet_plane=1.5,tank=1.2    explicit per class
HELSINKI_BOX_FACTORS = {
    'condor': 1.555, 'hangar': 1.153, 'jammer': 1.080, 'jet_plane': 1.728,
    'large_launcher': 1.308, 'medium_plane': 1.415, 'mine_roller': 1.085,
    'small_launcher': 2.115, 'small_plane': 1.358, 'small_tower': 1.263,
    'spacecraft': 1.120, 'ta-ta': 1.150, 'tank': 1.345,
}
ALREADY_LOOSE = {'helicopter', 'large_tower', 'medium_launcher'}
```

Parse `DRONE_BOX_GROW` at import into `BOX_GROW: Dict[str, float]` (empty =
off). In `annotations_for`, per emitted `(name, box)`, including transient boxes
and runner-up classes: grow the box about its centre by `BOX_GROW.get(name, 1.0)`
**after** `BOX_SCALE`, then `clip_bbox_to_frame`. Growth applies to reported
boxes only, never to the stored track boxes: matching and motion must not see it.
Add `"box_grow": BOX_GROW` to `/api`. Replace the `HELSINKI_BOX_FACTORS` values
with the ones Task 6a measures on this machine's patches if they differ by more
than ~0.05.

Offline expectation on `2c09583d` (registration, re-sent fix, decay 1.0,
new-track 0.15):

| growth | tight truth | loose truth |
|---|---|---|
| none | 0.445 | 0.347 |
| uniform 1.2 | 0.399 | 0.402 |
| isotropic Helsinki, cap 1.3 | 0.377 | 0.441 |
| isotropic Helsinki, cap 1.5 | 0.284 | 0.459 |

The two columns disagree by design. Only a real run can say which truth is
right, which is Task 5's job.

## 5. Task 4 — Offline scorer upgrades (`tools/score_offline.py`)

1. **`--truth-motion registration` (new default).** Run `RegistrationMotion`
   over every view of the run in order and carry the truth with its final
   `.motion` (on `2c09583d`: `(-14.4365, 0.007486, 6.9e-05, 52.3468, 8e-06,
   0.014966)`, 68.51 at the centre). Keep `fitted` and `prior` as options. The
   current `fitted` truth (68.93) drifts with the same bias as the tracker.
2. **`--truth-boxes loose|tight` (default `loose`).** `loose` grows each truth
   box about its centre by the class's (w, h) Helsinki ratios from the table in
   section 1 (per dimension, not isotropic). It does not grow helicopter,
   large_tower or medium_launcher, then clips to the frame. Print one line
   saying which convention was used. Also print the tight-truth score on the same
   line, so nobody tunes on one without seeing the other.
3. **`--iou` (default 0.5).** Use `--iou 0.3` when comparing detectors whose box
   conventions differ (e.g. v6 vs v9): it measures finding and ranking objects,
   not box shape.
4. Update the module docstring and the "do not use it to tune box geometry"
   warning: tight truth cannot judge box size; loose truth is an estimate that
   matched the real 0.3048 and the `BOX_SCALE=0.8` collapse; real runs are the
   arbiter.

Acceptance: on `2c09583d`, the served config (`REGISTRATION=0`, old decay and
new-track values, no growth) scores ~0.30 with loose truth and ~0.38 with tight.

## 6. Task 5 — Real validation runs A and B (the human starts them)

Serve on the Mac, from `drone-flyby/`, recording on, nothing else running:

```bash
# Run A: tracker fixes only
DRONE_MODEL=models/drone-yolo11n-v4.pt DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt \
DRONE_IMGSZ=960,1280 DRONE_DEVICE=mps DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,UNSEEN_DECAY=1.0,NEW_TRACK_CONFIDENCE=0.15 \
.venv/bin/python api.py

# Run B: Run A + box growth
DRONE_BOX_GROW=helsinki DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt \
DRONE_IMGSZ=960,1280 DRONE_DEVICE=mps DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,UNSEEN_DECAY=1.0,NEW_TRACK_CONFIDENCE=0.15 \
.venv/bin/python api.py
```

Before each run: `curl -s localhost:9053/api` must show `models_loaded: 2`,
`imgsz: [960, 1280]`, `registration: 1`, and `box_grow` empty for A and filled
for B. Then run `python tools/preflight.py --url https://<named-tunnel>/predict`.
Then tell the human it is ready. After each run, write the real score, the run
id, frames answered and the preflight numbers into `HANDOVER.md` straight away.

Decision tree. Baseline is 0.3048 (Mac); expected A ≈ 0.33–0.35.
- **B ≥ A + 0.03:** the loose convention is confirmed. Run **B2** with
  `DRONE_BOX_GROW_CAP=1.6`, and if it wins again, `B3` uncapped. Serve the best.
  Task 6 (v9) must use official labels.
- **|B − A| < 0.03:** inconclusive. Run B2 (cap 1.6): a bigger dose gives a
  bigger effect either way.
- **B ≤ A − 0.03:** the official boxes are tighter than Helsinki suggests. Keep
  growth off. v9 keeps tight labels, but still gets the paste-scale fix. Switch
  `score_offline.py`'s default back to `--truth-boxes tight`.
- **A < 0.30:** something in Task 1/2 is wrong on the live system. Check `/api`,
  the log (look for registration exceptions and the lag=1 frames), and the
  recording before anything else. Do not stack more changes on it.

## 7. Task 6 — v9 detector: official-convention labels and the right paste scale (GPU box)

Start this in parallel with Task 5: the dataset build and training take ~2 h.
If Run B contradicts the loose convention, rebuild with `--box-convention
tight` (same recipe otherwise).

### 6a. `training/measure_box_convention.py` (new)

Run it after `training/extract_patches.py`. It writes
`training/box_convention.json`:

```python
"""Official box vs tight mask box, per class, on the Helsinki cut-outs."""
import json, sys
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).resolve().parent
patches = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'data' / 'patches'
out = {}
for folder in sorted(p for p in patches.iterdir() if p.is_dir()):
    rw, rh = [], []
    for path in folder.glob('*.png'):
        patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)   # stored = the official box, alpha = mask
        if patch is None or patch.shape[2] < 4:
            continue
        ys, xs = np.nonzero(patch[:, :, 3] > 127)
        if len(xs) == 0:
            continue
        rw.append(patch.shape[1] / (xs.max() - xs.min() + 1))
        rh.append(patch.shape[0] / (ys.max() - ys.min() + 1))
    if rw:
        w, h = float(np.median(rw)), float(np.median(rh))
        out[folder.name] = {'w': round(w, 3), 'h': round(h, 3), 'iso': round((w * h) ** 0.5, 3), 'n': len(rw)}
(HERE / 'box_convention.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
```

Compare with the table in section 1. If the masks were re-cut since 17 Sep, the
numbers will shift a little; use the new ones everywhere, in this script's JSON
and in `HELSINKI_BOX_FACTORS`.

### 6b. `training/make_dataset.py`

1. `--box-convention {tight,official}` (default `tight`, so old recipes are
   unchanged) and `--box-factors training/box_convention.json`. Pass both
   through `_init_worker` like the other options.
2. In `build_scene`, where a pasted cut-out's rectangle `box = (x, y, x + width,
   y + height)` becomes a label: in `official` mode, grow it about its centre by
   the class's `iso` factor and clip to the 4K frame. Use the grown box for the
   `overlaps` check, `occupied` and `labels`, so labels do not overlap. Filled
   cut-outs (helicopter, large_tower, medium_launcher) measure 1.0 and stay as
   they are. Helsinki scenes keep their own annotations unchanged: they already
   are the official convention.
3. The validation cut-outs (`--extra-patches training/patches_val`) are
   mask-cropped in the same way, so the same growth applies to them.
4. `--helsinki-scale LO,HI` overriding `HELSINKI_SCALE_RANGE`; v9 uses
   **0.75,1.2**. Rewrite the comment above `HELSINKI_SCALE_RANGE`: the 0.45–1.00
   range came from comparing tight validation boxes with official Helsinki boxes
   (tank 0.75 measured vs 0.746 from the convention alone, jet 0.56 vs 0.58,
   small_launcher 0.49 vs 0.46, jammer 0.89 vs 0.92). Ground motion (68.37 vs
   66.21 px/frame) says validation objects are ~1.03× Helsinki. Mention that the
   final flight is said to be at 600 m too.
5. Check `--preview` output by eye before training: pasted jets and tanks should
   now have boxes as loose as the Helsinki objects in the same view.

### 6c. `training/train_remote.sh`

Add `BOX_CONVENTION` (default `tight`) and `HELSINKI_SCALE` (default empty =
code default). Run `measure_box_convention.py` after `extract_patches.py`, and
pass `--box-convention "$BOX_CONVENTION" --box-factors
training/box_convention.json` plus `--helsinki-scale` when set. Put a "v9 RECIPE"
block at the top of the script, above the v8 notes, explaining the change and
superseding v8. v8's class weights and yolo11m are untested, and it would repeat
both label problems.

v9 = the v6 recipe with the two fixes, so the effect is attributable:

```bash
MODEL=yolo11s.pt IMGSZ=960 EPOCHS=40 SCENES=1600 NAME=drone-yolo11s-v9 \
BOX_CONVENTION=official HELSINKI_SCALE=0.75,1.2 REAL_SHARE=0.6 \
bash training/train_remote.sh
```

(`WEIGHTS` left at the script default, which is v6's. Raise `--helsinki-share`
from 0.1 to 0.2 only if there is time for a second run: the Helsinki frames are
the only exactly-labelled real renders we have.)

### 6d. Judge v9, then Run C

- Offline, loose truth, IoU 0.5, registration on, and at `--iou 0.3`:
  1. served pair with growth (the Run A/B winner),
  2. v9@1280 alone,
  3. v9@960 + v9@1280 (`DRONE_IMGSZ=960,1280`, the same weights twice),
  4. v4@960 + v9@1280.
- Growth is report-time and per class, not per model. So for options 2 and 3
  serve **without** growth (v9's boxes should already be official). For option
  4, v4's tight boxes and v9's official boxes would mix inside the tracks. Only
  consider it if you move growth into `detect()` per model index (grow the boxes
  of `DRONE_BOX_GROW_MODELS=0` only, computing truncation on the ungrown box). Do
  that only if option 4 is clearly the best offline.
- Real **Run C** with the best offline option, if it beats the Run A/B winner
  offline by more than ~0.02 on the loose truth. Keep whichever wins on the
  real score.

## 8. Task 7 — Freeze and the evaluation (20 Sep)

- By ~12:00 CEST: freeze the config (models, `DRONE_IMGSZ`, `DRONE_SET`,
  `DRONE_BOX_GROW`, camera `full`). Write it verbatim into `HANDOVER.md` and
  commit.
- Serve from the Mac, on power, sleep disabled, nothing else running, **named**
  Cloudflare tunnel. The quick tunnel dropped frames before: 68/249 answered.
  Recording on: it is cheap and useful afterwards.
- `/api` check, `tools/preflight.py` against the public URL, one confirmation
  validation run on the exact setup. The human starts it.
- Then the evaluation attempt, no later than ~14:00, leaving ≥2 h of buffer. The
  human starts it.
- Fallback if anything misbehaves on the day: the 0.3048 config. Spell every
  changed default out explicitly:
  `DRONE_SET=BOTH_MODELS=1,REGISTRATION=0,FIX_RESENT=0,UNSEEN_DECAY=0.97,NEW_TRACK_CONFIDENCE=0.25`,
  `DRONE_BOX_GROW` unset, same models and `DRONE_IMGSZ=960,1280`. Make sure that
  exact command still starts and serves after your changes.

## 9. Do not

- Do not tune box geometry on the tight truth, and do not quote a tight-truth
  score as progress on box shape.
- Do not change the camera. `hybrid` lost 0.16, there is no offline way to test
  cameras, and `full` is direction-agnostic, which the final flight may need.
- Do not train v8 as written. It repeats tight labels and the 0.45–1.00 paste
  scale.
- Do not add speculative labels to `validation_objects.json`. The truth's
  weakness is box shape, not missing objects.
- Do not serve from the rented box (it measures ~0.02 lower), and do not spend
  GPU time on TensorRT.
- Do not bundle untested changes into the final attempt. Every change must have
  had one real validation run on the Mac.

## 10. Reference: how the numbers above were produced

Everything was run on Franek's laptop against the recording
`data/recordings/2c09583d65c34d7db47b32c66c375b02` (248 views, camera `full`,
17 Sep). Detections were cached by `tools/bench_recordings.cached_detections`
for v4@960 and v6@1280 and concatenated per frame (`BOTH_MODELS=1`). The truth
was `training/validation_objects.json` with `validation_ignore.json` regions of
verdict `confirmed`. Scoring used `score_offline.score` (COCO, IoU 0.5, macro
over present classes).
- **Motion:** SIFT (5000 features) on consecutive recorded views, matches lifted
  to source pixels through `source_region_xyxy`, per-pair RANSAC homography,
  then one global homography and one affine fit over all inliers.
- **Re-sent renders:** for each consecutive pair, the number of frames of
  motion under the global homography that best explains the matches (0, 1 or 2)
  compared with the labelled gap.
- **Loose truth:** the tight truth boxes grown per class by the (w, h) Helsinki
  ratios about their centre (helicopter, large_tower, medium_launcher
  unchanged).
- **Loss split:** COCO-style greedy matching per frame and class. A false
  positive is `dup` (second box on a matched object), `loc` (right class, IoU
  0.1–0.5), `cls` (IoU ≥ 0.5 with another class's object) or `bg` (nothing
  there). "AP if no FPs" re-scores with every unmatched prediction deleted.
