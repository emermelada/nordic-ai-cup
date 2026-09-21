# The exact environment the Drone Flyby endpoint ran with for the graded
# evaluation (2026-09-20 12:05 UTC, score 0.2630, 249/249 frames answered).
# Read back from the box's saved config and confirmed against GET /api.
#
#   cd drone-flyby && set -a && . configs/evaluated_serve_env.sh && set +a && python3 api.py
#
# Needs a CUDA GPU (RTX 5090 here) and the weights under models/, which are
# committed. Allow ~2 minutes for the five models to load before the first frame.
export DRONE_PORT=10200
export DRONE_CAMERA=row0x2
export DRONE_BOX_GROW=1.3
export DRONE_BOX_GROW_CAP=1.3
export DRONE_MODEL=models/drone-yolo11n-v4.pt
export DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt
export DRONE_IMGSZ=960,1280,1280,2560,1280
export DRONE_DEVICE=cuda
export DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10
