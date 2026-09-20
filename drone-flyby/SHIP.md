# The configuration to ship — 20 Sep

Validated. Corrected mean **0.5450** (n=4) at MAX_MISSES=6; **0.5502** (n=3) at
MAX_MISSES=12. The two are inside the ±0.005 noise floor, so either is safe;
12 is the point estimate peak of the 6/12/20 bracket (20 gave 0.5385).

```bash
PYTHON=python3 PORT=6006 tools/arm.sh SHIP \
  DRONE_CAMERA=row0 DRONE_LEVEL0_WEIGHT=1.5 \
  DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
  DRONE_MODEL=models/drone-yolo11n-v4.pt \
  DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt \
  DRONE_IMGSZ=960,1280,1280,2560,1280 \
  DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10,MAX_MISSES=12
```

`arm.sh` reads `/api` back and exits non-zero unless the service really is
serving this. Never start the service by hand.

Best single run: 0.5788. Frame loss costs 0.0022/frame (calibrated to 40
frames; do NOT extrapolate past that — discard heavier losses instead).
