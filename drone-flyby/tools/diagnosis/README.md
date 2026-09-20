# Per-class diagnosis (20 Sep 2026)

How the box-geometry bug was found and what was ruled out. Each script scores
**recorded answers** against the mined truth, so they need no GPU -- only
`numpy`, `opencv-python-headless`, `faster_coco_eval` and a run in
`data/recordings/<id>/*.json`.

Run order, and what each answered:

| script | question | answer |
|---|---|---|
| `diag.py` | is a missed object MISSED, MISNAMED or FOUND? | misnaming is ~zero; the dead classes are missed |
| `near.py` | missed at IoU 0.5 -- but is a box *near* it? | jammer 0.06 at IoU>=.5, **0.60 at 0.1-0.5**; boxes 1.24-1.42x oversized |
| `cover.py` | was the object even in view? | ~half of frames are out of view and carried by memory |
| `scale.py` | what rescale maximises the hit rate? | per-class, stable across 5 runs |
| `sweep.py` | what rescale maximises **AP**? | the shipped isotropic factors |
| `gh.py` / `gw.py` | per-class height / width on top | height pays, width does not |
| `cmp.py` | current vs helsinki vs fitted | fitted 0.559, helsinki-iso 0.507, current 0.473 |
| `robust.py` | does the gain survive other truth conventions? | yes, all four (+0.081/+0.079/+0.059/+0.057) |
| `loo.py` | does fitting on more runs help the thin classes? | see SHIP.md |

Ruled out -- **do not re-derive**:

| script | idea | why it failed |
|---|---|---|
| `shift.py` | correct the per-class centre offset | +0.007; spacecraft flips sign between splits |
| `gate.py` | demote out-of-band box sizes | +0.024 one split, +0.004 reversed, signs invert |
| `dup.py` | merge duplicate detections | there are none, in any class |
| `sep.py` | are FPs separable? | they are -- but most are real unlabelled objects |
| `rank.py` | why do some classes rank badly? | weak TP confidence: a detector problem |

`apply.py` and `wh.py` are earlier single-run versions kept for provenance;
prefer `sweep.py` and `gh.py`, which average over runs.
