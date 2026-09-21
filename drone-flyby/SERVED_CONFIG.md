# What was actually served for the graded evaluation

**Evaluation: 0.2630**, 2026-09-20 12:05 UTC, one attempt, all 249 frames answered
(0 lost, no errors). Validation mean for the same configuration: **0.5841**
(n = 7 runs, sd 0.0135), so about 45 % of the validation score carried over to
the unseen flight.

Launch it with [`configs/evaluated_serve_env.sh`](configs/evaluated_serve_env.sh).

## The configuration

| | |
|---|---|
| Detectors (5 passes per frame) | `v4@960`, `v6@1280`, `v8@1280`, `v8@2560`, `v9-p2@1280` |
| Camera | `row0x2`: the six Level-1 quadrant centres, each followed by a whole-frame Level-0 look (50 % Level 0) |
| Answer box growth | one flat factor, **1.3**, cap 1.3, the same for every class |
| New-track confidence | 0.10 |
| Tracker | ground motion refitted online during the flight from a Helsinki prior; other tracker constants at their code defaults |
| Per-class tables | none active: no per-class growth, thresholds, aliases or size gates |

Confirmed on the wire before submission: 5 of 5 models loaded, camera `row0x2`,
`box_grow_cap 1.3`, `level0_weight 1.0`, `band false`, `floor_all null`. Logged
inference was p50 154 ms, p95 192 ms against a 330 ms frame budget.

## Validation record for this configuration

| runs | scores |
|---|---|
| 7 runs before the final hour | mean 0.5841 (best 0.6071, worst 0.5683) |
| 2 confirmation runs | 0.5853, 0.5817 |
| 2 runs hit by external frame loss | 0.5562 (13 frames lost), 0.5063 (31 lost) |

The last two are network stalls on a shared box, not model differences: each
lost frame costs about 0.0022, which brings them to about 0.585 and 0.574.

## What is deliberately not in it

Everything below was measured on the real grader or against the official
annotations and rejected, so none of it is in the served configuration:

* per-class box growth with separate width and height factors: **-0.032** on the real grader
* single-model serving: -0.086 (the ensemble is worth that much)
* a sixth pass (`v9@2560`): -0.010
* Level-2 zoom with v9: -0.075
* graded sub-0.001 confidence band: 0.000
* per-class size gates and confidence thresholds: did not survive a reversed split

The reason is generalisation, not taste: the evaluation is a different flight,
and every fitted per-class constant is a fit to the validation flight. The two
tuned scalars that remain, box growth 1.3 and the weight of the whole-frame
view, describe the grader's box convention and how the detector degrades under
4x downsampling, not this flight's layout.

## Known caveats

* The box that served the run also carried a `WxH` box-growth parser patch that
  this repo does not contain. It is inactive under the flat `1.3` setting.
* Roughly 60 % of the training backgrounds are frames cut from the validation
  flight (`training/backgrounds_real`). The evaluation flight was never used.
* The 3.4 GB of recorded validation frames used for offline replay lived on the
  rented GPU instance and are not in this repository.
* The `HELSINKI_BOX_FACTORS` tables in `flyby.py` were measured on an older
  detector and are wrong for v9. They are dead code in this configuration.
