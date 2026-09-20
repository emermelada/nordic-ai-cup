# The configuration to ship — 20 Sep (REVISED 13:50, read the truth-file section first)

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


---

# Corroboration and the fallback (added after the held-out test)

`training/box_convention.json` is derived from the **25 official Helsinki
annotation frames**, and `score_offline.loosen()` already applies it to the
mined truth -- so the offline truth boxes are in the real ground-truth
convention, and the correction below is toward real labels, not toward our
own mining.

Four geometry policies, all five runs:

| policy | macro |
|---|---|
| A current, flat 1.3 | 0.473 |
| B `helsinki` isotropic -- **real labels, nothing fitted** | 0.507 |
| C `helsinki` w/h at cap 2.2 | 0.480 |
| **D fitted per-class w/h** | **0.559** |

**B is the corroboration that matters.** It is derived entirely from official
annotations with zero fitting and still beats current by +0.034, so the
direction is confirmed by two independent routes. D adds a further +0.052 and
held out cleanly (+0.080/+0.082 on two runs it was never fitted to).

Under D no class is worse than current.

## Fallback ladder

1. **D** -- the `DRONE_BOX_GROW` above. First choice.
2. **B** -- `DRONE_BOX_GROW=helsinki DRONE_BOX_GROW_CAP=1.3` (no `_WH`).
   Nothing fitted; use if D does not reproduce on real runs.
3. **A** -- `DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3`, the 0.5450 config.

## Rejected, measured, do not spend runs on

* **Global height factor**: wash, +0.002. The effect is genuinely per class.
* **Per-class centre shift**: our boxes sit off-centre on the weak classes
  (spacecraft -0.215 of its height, mine_roller +0.100, jammer +0.093 in x)
  and correcting it is worth only +0.007 held out -- and **spacecraft flips
  sign between fitted and held-out runs**, so that class is fitting noise.
  Not worth new serving-path code.
* **More resolution**: `large_tower` is 49 px and found 97%; `jammer` is 48 px
  and found 6%. Size is not the cause, which is why L2, imgsz 2560 and the P2
  head never moved these classes.

## What is still broken, for whoever picks this up

Roughly half of every object's frames are **out of view**, carried by memory.
Carrying is worth as much as detection and it is very uneven -- after the box
fix, `large_tower` carries 0.98 and `jet_plane` 0.93, but `spacecraft` 0.18
and `jammer` 0.21. `spacecraft` is also genuinely blind: a third of its
in-view frames have nothing near them at all.

Those need the tracking path (MAX_MISSES, track creation), which **cannot be
tested offline without a GPU** -- `score_offline --replay` needs model
detections. That is the next real lever.

---

# Rejected after testing: the per-class size gate (20 Sep, 12:40)

Demoting detections whose size falls outside a per-class band looked strong:
the false positives really are separable (a band keeps 90% of jammer's true
positives and only 13% of its false ones), and on one split it paid **+0.024**
held out.

**It does not survive the reversed split.** Fitting the bands on runs A,B,C and
scoring on D,E gives +0.024; fitting on D,E and scoring on A,B,C gives +0.004,
and the per-class signs invert:

| class | A,B,C -> D,E | D,E -> A,B,C |
|---|---|---|
| spacecraft | **-0.042** | **+0.016** |
| jet_plane | +0.014 | **-0.075** |
| large_tower | ~0 | -0.040 |
| helicopter | +0.008 | -0.037 |

The bands move with whichever runs they are fitted on. That is fitting noise,
not an effect. **Do not ship it and do not re-derive it** -- the separability
table that motivates it is genuinely true and will tempt you again.

Contrast with the box geometry fix, which gives +0.080 and +0.082 on the two
held-out runs, +0.086 across all five, and is independently corroborated by the
official annotation convention. That asymmetry is the whole reason to trust one
and not the other.

# Why the last three classes are still broken

`spacecraft` 0.200, `small_launcher` 0.218, `tank` 0.449 did not move.

Their **true detections carry weak confidence** -- median 0.22, 0.61 and 0.53
against 0.77-0.86 for every healthy class -- so false positives outrank them.
For `small_launcher`, 2.1 false positives sit above the median true positive
per true positive. AP punishes exactly that.

That is a detector calibration problem, not a geometry or tracking one, and
nothing that reranks our own output fixed it without overfitting. It needs
better weights.

# Untested, needs a GPU: memory carrying

About half of every object's frames are out of view and carried by memory.
After the box fix, carrying is still wildly uneven:

    large_tower 0.98   jet_plane 0.93   helicopter 0.90
    small_launcher 0.56   jammer 0.21   spacecraft 0.18

Carrying is worth as much as detection. `tools/sweep_remote.sh` runs
`score_offline --replay`, which exercises the real predict path (tracking,
memory, box growth) against cached detections -- seconds per run on a 5090, and
it spends no validation attempts. That is the next lever and it has never been
pulled.

# Robustness: the gain does not depend on the truth assumptions

The factors were fitted against 'loose' truth with unlabelled-object regions
ignored. Re-scored on the two held-out runs under all four combinations:

| truth | ignore | current | fixed | gain |
|---|---|---|---|---|
| loose | on  | 0.437 | 0.518 | **+0.081** |
| loose | off | 0.426 | 0.504 | +0.079 |
| tight | on  | 0.264 | 0.323 | +0.059 |
| tight | off | 0.255 | 0.312 | +0.057 |

`tight` is the opposite box convention and can barely see box geometry at all,
yet the gain survives there too. This was the main risk -- that the factors
were fitted to our own mining rather than to the real convention -- and it is
answered.

# Also ruled out, so nobody re-checks

* **Duplicate detections**: zero, in every class. The tracker merges correctly.
* **Precision / FP suppression**: the false positives are overwhelmingly
  background boxes (tank 4840, jammer 3761), but most are **real unlabelled
  objects** -- which is why all four previous suppression mechanisms failed and
  why the size gate above also failed. Precision is a dead end on this task.
  The remaining headroom in the last three classes is recall and detector
  quality, not ranking.

# Geometry is exhausted (leave-one-run-out, 20 Sep 13:20)

Refitting the factors on four runs and testing on the fifth, against the
shipped three-run fit on the same held-out run:

    held 7e441f26   shipped 0.597   refit-on-4 0.592   -0.005
    held 76bef7bb   shipped 0.580   refit-on-4 0.569   -0.011
    held b5544ad3   shipped 0.584   refit-on-4 0.584   +0.001
    held 89f751a2   shipped 0.515   refit-on-4 0.525   +0.010
    held 95f5a5c9   shipped 0.521   refit-on-4 0.530   +0.009
    mean            shipped 0.559   refit-on-4 0.560   +0.001

A wash. **Thin fitting data was not why `small_launcher` and `spacecraft` did
not move** -- more of it changes nothing (small_tower +0.034 and tank +0.017 are
cancelled by helicopter -0.029 and small_plane -0.011). The factors are
converged, and this also re-validates the shipped config at 0.559 leave-one-out.

Per-class **width** is likewise exhausted: the global optimum is exactly 1.00
and the per-class gains total +0.006, because the isotropic factor already
captures that axis. Height was the only axis with anything left in it.

**So every geometry lever is spent.** What remains for `spacecraft` (0.200),
`small_launcher` (0.218) and `tank` (0.449) is detector quality and memory
carrying -- the latter untested and needing a GPU (`tools/sweep_remote.sh`).


---

# CORRECTION (13:50): the offline proxy was the WRONG TRUTH FILE

`tools/score_offline.py` reads **`validation_objects.json`** (32 objects).
`tools/calibrate_truth.py` measured the two truth files against 12 runs of
known real score:

| truth | objects | Pearson | Spearman |
|---|---|---|---|
| `scene_objects.json` | 66 | **+0.894** | **+0.811** |
| `validation_objects.json` | 32 | **-0.039** | **-0.287** |

**Everything above was fitted against the anti-correlated one.** Re-scored
against `scene_objects.json`, the full fitted config gives **-0.007**, not
+0.086.

The two disagree about one class. Our jammer AP is **0.013** under
validation truth and **0.848** under scene truth -- the same detections. Box
conventions are near identical (jammer median 34.5 vs 38.1 px, small_plane
38.1 vs 38.1), so this is not scaling: they disagree about *which objects are
jammers*. `validation_objects.json` was labelled by "model v2 (conf >= 0.5)
... checked by eye"; `scene_objects.json` by v8+v6 at native resolution keeping
only chains of >= 6 consecutive sightings. The second is both the better method
and the one that correlates with reality.

## What is shipped instead: the factors that are safe under BOTH truths

jammer is the **only** class whose factor flips sign. Every other factor is
neutral or positive under both files, so dropping jammer alone gives a config
that does not depend on which mining run is right:

| config | validation truth | scene truth |
|---|---|---|
| full fitted | +0.083 | **-0.005** |
| **safe (jammer reverted to 1.30)** | **+0.062** | **+0.006** |

`SHIP_ARGS` carries the safe config. `jammer=1.30x1.30` is deliberate -- do not
"restore" it to 0.78x1.09.

## Honest expectation

The claimed gain is now **+0.006 to +0.062** depending on which truth is right,
not +0.086. It is positive under both, which is the most that can be said
without a real run. **The first validation run decides it.** If it does not
beat 0.5450, fall back: `DRONE_BOX_GROW=1.3 DRONE_BOX_GROW_CAP=1.3`.

`tools/diagnosis/scene.py` re-scores any config against either truth;
`tools/diagnosis/both.py` finds the factors safe under both. **Run every future
offline claim through both truth files before believing it.**

---

# Under the CORRECT truth: what is actually wrong (20 Sep 13:10)

Per-class AP against `scene_objects.json`:

    jammer .848   large_tower .594   small_tower .557   small_plane .531
    tank .494     helicopter .441    hangar .381        jet_plane .368
    large_launcher .335   mine_roller .334
    ta-ta .076    medium_launcher .002

`medium_launcher` and `ta-ta` do not exist in `validation_objects.json` at all,
so **they were invisible for the whole project**. They are worth 1/12 each.

MISSED / MISNAMED / FOUND against scene truth:

| class | FOUND | MISNAMED | MISSED |
|---|---|---|---|
| medium_launcher | 0.00 | 0.02 | **0.88** |
| ta-ta | 0.17 | 0.05 | **0.62** |
| large_launcher | 0.53 | **0.26** | 0.05 |

* `medium_launcher` and `ta-ta` are **blind spots** -- geometry does nothing
  (medium_launcher is 0.000 at every box scale from 0.6 to 1.3). Only training
  reaches them.
* `large_launcher` is **located correctly and called the wrong name** 26% of the
  time: mine_roller x93, tank x70, medium_plane x44. Misnaming IS real under
  this truth, unlike under `validation_objects.json` where it measured ~zero.

## Shipped: class aliasing (+0.010 scene, neutral validation)

Answer the same box again under a class we are confused with, at 0.30x
confidence. AP is per class, so the extra copy is one more low-ranked false
positive in a class that already has thousands, while the copy landing in the
right class is a new true positive -- the "additions work, replacements fail"
pattern, now six for six.

    DRONE_CLASS_ALIAS=mine_roller>large_launcher,tank>large_launcher,\
    medium_plane>large_launcher,small_tower>small_plane
    DRONE_CLASS_ALIAS_CONF=0.30

Every edge was tested individually under BOTH truths and only the four that
hurt neither were kept (16 of 20 candidates were dropped). Bracketed: 0.1 gives
+0.007, 0.3 +0.010, 0.45 +0.010, 0.6 +0.003, 1.0 **-0.048** -- above ~0.6 the
alias outranks real detections.

large_launcher .332 -> .412, small_plane .532 -> .572.

## Also added, untested: DRONE_CLASS_WEIGHT

Per-class multiplier on a track's class votes -- tilts WHICH class a track is
called without touching its confidence. Aimed at the same large_launcher
confusion. It needs `--replay` (GPU) to evaluate and has never been run.
Default empty = no change.

## Dead under the correct truth too

Top-K per class per frame (wash at every K) and a confidence floor (monotonic
loss). The harmful false positives are **high**-confidence, so trimming the
tail cannot help. Post-processing on what we emit is exhausted.
