# The configuration to ship — 20 Sep

## The finding that matters

Per-class AP on a real run, best config:

    large_launcher .956  large_tower .945  jet_plane .942  hangar .696
    helicopter .634  small_tower .530  tank .474  mine_roller .349
    small_launcher .280  small_plane .194  spacecraft .090  jammer .017

The top three are nearly maxed. Every knob we tuned for two days moved those.
**The score lives entirely in the bottom five**, each worth 1/12 = 0.083.

Decomposing each truth object-frame into MISSED / MISNAMED / FOUND:

* **MISNAMED is ~zero everywhere** (only tank->mine_roller, 14%). The
  misnaming story that drove the whole Level-2 investigation is irrelevant.
* jammer is **0.06 at IoU>=0.5 but 0.60 at IoU 0.1-0.5**. We *find* it two
  thirds of the time and draw the box wrong.
* Our boxes are 1.24-1.42x too big on exactly the dead classes, and
  0.94-1.07x on the healthy ones.
* Size is not the cause: large_tower is 49 px and found 97%; jammer is 48 px
  and found 6%.

We were inflating every box by a flat `DRONE_BOX_GROW=1.3`. That paid +0.017
overall because it helps the large classes, and it was strangling the small
ones the whole time.

## Measured result

Per-class isotropic + per-class height, fitted on three runs, scored on two
runs it was **never fitted to**:

    89f751a2  0.435 -> 0.515  (+0.080)
    95f5a5c9  0.439 -> 0.521  (+0.082)

jammer .001 -> .267 · small_plane +0.175 · helicopter +0.157 ·
large_tower +0.135 · mine_roller +0.080 · large_launcher +0.072

Every class up, none regressed, and the held-out gain is *larger* than the
fitted gain. A single global height factor is a wash (+0.002) — the effect is
genuinely per class.

## Ship this

```bash
PYTHON=python3 PORT=6006 tools/arm.sh SHIP \
  DRONE_CAMERA=row0 DRONE_LEVEL0_WEIGHT=1.5 \
  DRONE_BOX_GROW_WH=1 DRONE_BOX_GROW_CAP=1.9 \
  DRONE_BOX_GROW=condor=1.30x1.30,hangar=1.30x1.30,helicopter=1.17x1.87,jammer=0.78x1.09,jet_plane=1.37x1.37,large_launcher=1.17x1.40,large_tower=1.23x1.23,medium_launcher=1.30x1.30,medium_plane=1.30x1.30,mine_roller=1.10x1.66,small_launcher=1.30x1.30,small_plane=1.04x1.25,small_tower=1.04x1.14,spacecraft=1.30x1.30,ta-ta=1.30x1.30,tank=1.30x1.43 \
  DRONE_MODEL=models/drone-yolo11n-v4.pt \
  DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt \
  DRONE_IMGSZ=960,1280,1280,2560,1280 \
  DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10,MAX_MISSES=12
```

**The cap must be 1.9**, or helicopter's 1.87 height is clipped back to 1.3 and
most of the gain disappears. `arm.sh` checks 16 classes are present.

## Fallback — the previously validated config (real 0.5450, n=4)

Same as above but `DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3`, no
`DRONE_BOX_GROW_WH`. Use it if the new geometry does not reproduce.

Frame loss costs 0.0022/frame, calibrated to 40 frames. Do NOT extrapolate
past that — discard heavier losses instead (a 120-frame loss scored 0.2875).
