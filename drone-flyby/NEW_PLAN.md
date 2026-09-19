# Retest the Level-0 camera: it was rejected on a detector that could not see at Level 0

Written 2026-09-19 evening. Every number was measured today: offline with
`tools/recall_replay.py` and `tools/camera_sim.py` (both new), the camera A/B on
the rented 5090 alongside the v9 training run. Six of my own hypotheses were
falsified on the way and are recorded at the end.

`flyby.py` is UNCHANGED -- nothing here needs a code edit.

## The recommendation

**Run `DRONE_CAMERA=full0` on the served 4-model stack, three complete runs,
paired against `full`.**

Simulated end to end through the served code -- the real camera policy, the real
tracker, the four real inference passes, views rendered out of the rebuilt 4K
scene, scored on the 228 fully covered frames:

| pattern | L0 share | recall@0.5 | vs served |
|---|---|---|---|
| `TL TM TR L0 BR BM BL L0` | 2 of 8 | **0.545** | **+0.079** |
| L0 every 2nd | 6 of 12 | 0.534 | +0.068 |
| L0 only | 1 of 1 | 0.530 | +0.064 |
| `full0` | 3 of 9 | 0.518 | +0.052 |
| `quad0` | 4 of 8 | 0.494 | +0.028 |
| L0 every 7th | 1 of 7 | 0.494 | +0.028 |
| `full` (served) | none | 0.466 | - |
| `dwell` | none | 0.458 | -0.008 |
| `top` | none | 0.443 | -0.023 |
| `mixed` | none | 0.439 | -0.027 |

Every pattern, same stack, same frames, no refused camera command in any of them.
**Every pattern with a meaningful Level-0 share beats every pattern without one.**
Level-0-ONLY -- never leaving the full frame -- already beats the served sweep,
which is the mechanism stated as plainly as it can be: an object is visible
every frame instead of 10 frames in 33.

**Do not read the ordering inside the L0 family.** 253 object-frames puts the
resolution at about 0.03, so 0.545 / 0.534 / 0.530 are the same number and
picking the top row would be selecting on noise. The L0-vs-no-L0 split is the
result; the cadence is for the real runs to settle.

Cross-checked on two machines: `full` scores 0.466 on this CPU box and 0.466 on
the 5090. The simulator is deterministic and device-independent.

**On v3 the order was exactly the opposite: full 0.130, quad0 0.126, full0
0.119.** That is the whole argument. `full0` and `quad0` put a whole-frame
Level-0 look into the sweep, and they were measured -- and rejected -- with v4
as the only model. v4@960 detects **0 of 18** objects at Level 0. v6@1280
detects 12 of 18 (0.67) at mean IoU 0.73. The camera pattern was ruled out on a
stack that was blind at the level the pattern exists to exploit.

This is the third time this exact staleness has cost the project: the 1600 pass
lost as a *replacement* and won as an *addition*; `NEW_TRACK_CONFIDENCE` was
tuned against v3 and was worth +0.013 when retuned.

**What the simulation is and is not.** It is the served code on real pixels, so
the comparison between policies is sound and it is deterministic -- no run-to-run
noise. It is *recall on the 32 confirmed objects*, not AP, so it does not
predict the score, only the direction and rough size. The scene is mosaicked
from Level-1 views upsampled back to 4K, so an L0 view there has been through
one extra resample: **L0 is if anything under-rated here, not flattered.**

## Why it should pay: coverage is the binding constraint, not detection

A Level-1 view is exactly a quarter of the frame, so **average coverage is 25 %
whatever the sweep pattern** -- a geometric limit, not a tuning choice.
Measured on the confirmed objects:

| | |
|---|---|
| object's in-frame life | median **33 frames** (ground moves ~60 px/frame; they enter at the top) |
| frames the object is inside a view | median **10 of 33** (0.34) |
| acquisition lag once it is in view | median **4 frames** |
| in-view frames where we answer the right class | **70 / 71** |

**When an object is in view we essentially always answer it.** Cross-tabulating
every confirmed object-frame against the raw detections:

| answer | in view | carried | |
|---|---|---|---|
| hit | 43 | 94 | |
| near miss | 27 | 50 | |
| **nothing** | **1** | **36** | every real miss is a carried frame |

So the loss is not the detector. It is that each object is only *looked at* 10
times in 33 frames, and the other 23 are carried.

## Level 0 works now, and that is the new fact

Rendering the evaluator's own view geometry out of `data/scene` (same pixels,
same objects, L0 = whole frame into 960x540, a 4x downsample):

| pass | L0 detection rate | L0 mean IoU |
|---|---|---|
| v4 @ 960 | **0.00** | - |
| v6 @ 1280 | **0.67** | 0.73 |
| v8 @ 1280 | 0.50 | 0.75 |
| v8 @ 2560 | 0.50 | 0.71 |

v4@960 -- the only model the camera sweep was ever chosen with -- is blind at
Level 0. v6@1280 is not. Boxes from L0 are as good as from L1 (mean IoU 0.73
against L1's 0.75-0.80), so an L0 sighting is a usable re-anchor, not a guess.

`quad0` alternates a whole-frame look with a Level-1 quarter, so every object is
looked at **every second frame instead of every sixth to twelfth**. Two details
already line up in our favour: Level 0 is exempt from the camera distance limit
(`full_view_reset_exempt_from_delta`), so the quarters need no middle stops; and
`update_tracks` does not charge a miss to a track whose `best_level` is finer
than the current view, so L0 frames cannot prune L1 tracks.

`LEVEL_WEIGHT[0]` is already 1.0 (env `DRONE_LEVEL0_WEIGHT`), so L0 sightings
are not demoted. If `quad0` helps but its boxes look loose, that knob is the
follow-up, not a code change.

## The runs to make, in order

Two arms. The first needs **no code change at all**:

```bash
DRONE_CAMERA=full     ...served command...   # paired baseline, same host, same day
DRONE_CAMERA=full0    ...served command...   # arm 1, zero code change
```

If `full0` clears the baseline, the best measured cadence is worth one line in
`SWEEPS` (`flyby.py`, beside `full0`) -- applied by whoever runs it, not by me:

```python
    # Level 0 after each row of the full sweep. Measured best of ten patterns
    # offline (recall 0.545 against full's 0.466); inside noise of full0.
    'row0': [TL, TM, TR, (0, 1920, 1080), BR, BM, BL, (0, 1920, 1080)],
```

Three complete runs per arm, 249/249 or discard, `/api` checked before each.
**Prediction: full0 +0.02 to +0.06 on the real score.** Recall is not AP, so
this is direction and magnitude, not a score. If `full0` loses by more than the
noise floor, then L0 boxes are re-anchoring tracks worse than carrying them
does, and the whole family is dead -- do not then try the other cadences.

Latency: an L0 pass is the same 960x540 input as any other, so the stack costs
what it already costs. Nothing here touches the 333 ms budget.

## What was falsified today -- do not re-run these

1. **Per-pass box growth.** The four passes size boxes identically (ratios
   1.050 / 1.038 / 1.061 / 1.078 against the same truth, so the comparison is
   convention-free). I predicted they would differ by ~0.25 and they do not.
   `DRONE_BOX_GROW_PASS` is implemented and **off by default**; leave it off.
2. **The motion prior.** Refitting `MOTION` on the flight's own objects changes
   offline recall by exactly nothing (0.873 -> 0.873): the online fit already
   converges. The prior being ~2 px/frame slow is real and harmless.
3. **The near-miss band as a real loss.** 28-38 % of object-frames are boxes
   centred to 0.08 box sides that miss IoU 0.5 on size alone -- but offline
   recall peaks at growth 1.0 (0.873) and collapses at 1.45 (0.145), while real
   runs peak at 1.3 and barely move. That calibrates the grader's boxes at
   **~1.3x the mined truth**, so most of that band is measurement error, not
   score. This is the documented "blind to box growth" trap and I walked into
   it; the calibration is the useful part.

## The tool

`tools/recall_replay.py` replays a recording through `flyby.predict` from a
cached detection set and reports **recall only** -- the truth file's missing
objects corrupt precision and AP, but cannot turn a real object into a miss.
`--carry` interpolates each object's own trajectory (quadratic, 2.6 px RMS over
21-31 frame spans) to score the carried frames too, which is where the misses
turned out to be. **Never quote a precision or an AP from it.**
