"""Time YOLO inference on this machine.

    python scripts/bench_yolo.py                 # 200 synthetic 640x640 frames
    python scripts/bench_yolo.py path/to/video.mp4
"""
import platform
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
N_FRAMES = 200
WARMUP = 10

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = YOLO(WEIGHTS)


def frames():
    if len(sys.argv) > 1:
        import cv2

        cap = cv2.VideoCapture(sys.argv[1])
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    else:
        rng = np.random.default_rng(0)
        for _ in range(N_FRAMES + WARMUP):
            yield rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)


times = []
for i, frame in enumerate(frames()):
    t0 = time.perf_counter()
    model.predict(frame, device=device, verbose=False)
    if i >= WARMUP:
        times.append(time.perf_counter() - t0)

times = np.array(times)
print(f"machine : {platform.node()} ({platform.processor() or platform.machine()})")
print(f"device  : {device}  torch {torch.__version__}  threads {torch.get_num_threads()}")
print(f"frames  : {len(times)}")
print(f"latency : mean {times.mean() * 1000:.1f} ms  p95 {np.percentile(times, 95) * 1000:.1f} ms")
print(f"fps     : {1 / times.mean():.1f}")
