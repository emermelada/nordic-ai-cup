# Drone Flyby — what to serve, and how to spend the one evaluation attempt

Written 2026-09-20 ~04:30 CEST. Deadline 16:00 CEST.

## The one-line answer

If nothing below has been verified by a real run, serve **exactly this**, which
is the configuration measured at mean 0.5270 over four complete runs:

```bash
cd /workspace/drone-flyby
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt \
DRONE_IMGSZ=960,1280,1280,2560 \
DRONE_DEVICE=cuda DRONE_PORT=10200 \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 python3 api.py
```

`DRONE_CAMERA` unset means `full`. **Only change it to `top_mostly` if a real
validation run beat 0.5270 with it** — the expectation is +0.05 to +0.09 but it
was still unverified when this was written.

## Before the attempt, in order

1. **Check `/api` on the URL you will submit**, not the launch command:
   `models_loaded` must equal `models_requested` (4), `imgsz` must be
   `[960, 1280, 1280, 2560]`, `box_grow` must list 16 classes, `box_grow_cap`
   1.3, `new_track_confidence` 0.1, and `camera` must be what you intend.
   An empty `box_grow` means the baseline is being served and the attempt
   teaches nothing.
2. **Warm it up.** Four models warm slowly and the first run after a restart
   lost frames 2-5 once. Send a dozen throwaway frames first (any recorded
   frame through `/predict`) and confirm the per-frame cost in `data/serve.log`
   is ~76-82 ms.
3. **Gate the host.** Median RTT to `hel1-speed.hetzner.com` under 35 ms with
   p95 under 2.5x the median, and three consecutive 100 MB pulls holding
   ~20 MB/s. A host 60-90 ms away costs 0.05-0.08 through camera stalls alone.
   The box used here (93.91.156.104) measured 30.3 ms / 83-92 MB/s.
4. **One validation run on the exact setup you will evaluate**, and count the
   frames: 249/249 or discard it. Two missing frames cost up to 0.026.
5. **Nobody else validates at the same time.** One attempt runs at a time per
   team; a collision wastes both.

## Do not

* Do not restart the service while a run is in flight. And never kill it with
  `pkill -f probe_server.py` or `pkill -f api.py` over ssh: `-f` matches the
  ssh session's own command line and kills the shell mid-command. Use the PID
  files (`data/serve.pid`, `data/probe/server.pid`).
* Do not click "Test endpoint" in the portal while the probe queue matters: it
  sends one frame and consumes a queued job.
* Do not submit the evaluation while the service is behaving oddly — the
  competition service has returned attempts with broken timestamps and
  single-frame runs before.

## What the measurements say, if you need to choose under time pressure

Score ~= coverage x precision. Measured: coverage 0.683, implied precision
0.77, real score 0.527.

| change | expected | status |
|---|---|---|
| `DRONE_CAMERA=top_mostly` | +0.05-0.09 | ready, unverified |
| v9 added as a 5th pass at 1280 | unknown | `bash serve_v9.sh`, refuses a 22 MB nano |
| anything touching box growth | 1.3 is the measured optimum | closed |
| suppression of weak tracks | measured dead repeatedly | closed |

Seven things were tested and rejected in the 19-20 Sep session: a full 3D
tracker, per-track yaw, multi-frame super-resolution, class-vote-share ranking,
a physical-size class prior, a homography carry with a global ground level, and
the Level-2 entry ring. The details are in the git log on
`drone-flyby-breakthrough`.

## The probe harness (useful after the deadline too)

`probe/probe_server.py` answers any frame from a precomputed file and parks the
camera on a chosen tile, so one validation run measures exactly one set of
boxes against the real ground truth, with no model and no latency in the loop.
`data/probe/jobs.json` is a queue; each new run takes the next job. That is how
the 0.4025 number for the mined objects was obtained, and it is the only way we
have of measuring anything against the real truth.
