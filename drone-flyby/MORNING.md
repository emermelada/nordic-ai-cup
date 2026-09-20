# Morning runbook — 20 Sep, deadline 16:00 CEST

Read this first. `PLAN_TO_08.md` has the reasoning; this is the actions.

---

## Before anything else

**The serving box died overnight** (SSH and the service port both refused around
03:00). Nothing is lost — everything is on GitHub. It was rented at `$0.083/hr`,
which is an **interruptible/spot** rate; those get reclaimed when outbid. **Take
on-demand this time**, it is about $1/hr and the difference is irrelevant next to
losing the endpoint mid-evaluation.

```bash
# 1. rent Czechia datacenter:214845 (m:53164 / m:40773), ON-DEMAND, 1 GPU
# 2. then, from the repo:
tools/bootstrap_remote.sh -p <port> root@<ip>
```

It gates on **both** latency and sustained bandwidth to Hetzner Helsinki before
uploading anything. A bad host cost 0.05–0.30 yesterday — more than two days of
tuning gained. Do not skip it.

**Then agree the URL with Franek.** A submission silently went to his box last
night and scored 0.0008. On the one-shot evaluation that is unrecoverable.
Ideally whoever is not serving shuts their service down for the final window.

---

## The validated configuration — ship this if nothing else works

```bash
DRONE_CAMERA=row0 DRONE_LEVEL0_WEIGHT=1.5 \
DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3 \
DRONE_MODEL=models/drone-yolo11n-v4.pt \
DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt \
DRONE_IMGSZ=960,1280,1280,2560,1280 \
DRONE_DEVICE=cuda DRONE_RECORD_DIR=data/recordings \
DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10 python3 api.py
```

Start it with `tools/arm.sh`, which refuses to hand over a service that is not
serving what you asked for. **~0.55 all-runs, best single run 0.5788.**

---

## The order of the day

| time | what | gate |
|---|---|---|
| 08:00 | re-rent, `bootstrap_remote.sh`, serve the config above | both network gates pass |
| 08:30 | **3 runs** on it | reproduces ~0.55, else the box is wrong |
| 09:00 | **3 runs** on the hybrid retest (below) | see the three outcomes |
| 09:30 | branch: retrain, or the knob queue | — |
| **14:30** | **HARD STOP.** Lock, one validation run, evaluate. | — |

Budget **4–5 submissions per 3 usable runs** — the evaluator dropped frames on
roughly two runs in three overnight, and it is not our service (we answered
every frame received in 78 ms median, link 29.7 ms, 88–91 MB/s).

---

## The one experiment that matters: the hybrid retest

```bash
# same as the validated config, plus:
DRONE_CAMERA=hybrid DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10,HYBRID_ACQUIRE=1,HYBRID_COVER=7
```

At 1 acquire : 7 cover the coverage is identical to `full` (simulated: 217 L1,
31 L2, 0 refused, median 63 looks per cell), so this isolates the Level-2
*detections* from any coverage cost.

**Why retest something that scored −0.049:** that measurement used v4/v6/v8 with
no 2560 pass and no v9. A Level-2 view is *native* — L1 throws away half the
linear resolution before we ever see it — and size is the dominant predictor of
per-class recall (+0.582 on log object size, measured). L2@1280 gives the same
apparent object size as L1@2560 **with real detail at a quarter of the compute**.

**Read the per-class column, not the total:**

| outcome | meaning | next |
|---|---|---|
| > 0.56, or `small_launcher`/`spacecraft` up | v9 handles native scale | raise duty cycle: 2:6, then 3:5 |
| ~0.55 and `hangar`/`mine_roller` drop | blurry-patch mechanism confirmed | the retrain is the unlock — `PLAN_TO_08.md` §2, §5 |
| < 0.53 | L2 is dead on this detector | knob queue below |

---

## Knob queue (if the above fails) — ~+0.01 each, no code

1. `DRONE_SET=MAX_MISSES=3` then `=12` — the last untouched v3-era knob
2. `DRONE_SET=RUNNER_UP_SHARE=0.10` — `RUNNER_UPS=1` failed, but the *share*
   threshold is a different cut and is untested
3. `DRONE_SET=MATCH_IOU=0.15` then `=0.30` — v3-era association threshold

---

## Do not spend runs on these — all measured, all dead

`AGREEMENT_WEIGHT` (−0.005) · `MISS_PENALTY` (−0.013) · `RUNNER_UPS=1` (−0.003) ·
the 0.0 floor band (+0.0015) · `HITS_BASE` (same family, assume dead) ·
`UNSEEN_DECAY=1.0` (wash, higher variance) · `LEVEL0_WEIGHT` 1.0 and 2.0 (1.5 is
the bracketed peak) · box growth 1.2/1.45 (1.3 is the peak) · `DET_CONF=0.003`
(wash) · a 5th pass at 3200 (blows the budget, 0.264) · `v4@2560` as a 5th model
(0.4994) · per-class confidence calibration (**mathematically a no-op**) · WBF
(+0.004) · SAHI tiling (we already have its gain via the 2560 pass).

---

## Before the evaluation — the checklist that has actually cost us

1. `/api` shows the exact config — `models_loaded` equals `models_requested`,
   `box_grow` has 16 classes, `imgsz` has one entry per model
2. The registered URL is **ours**, agreed with Franek
3. The service has been warm and idle for a few minutes (a cold connection
   costs frame 2 on about one run in four, and cost eight timeouts once)
4. One validation run on the **exact** final setup, count the frames
5. Then evaluate — with buffer, not at 15:50
