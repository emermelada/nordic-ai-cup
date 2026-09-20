# PRE-REGISTRATION — energy-conditioned blind-search speed (2026-09-20, written before the run)

## The one mechanism under test
V2's blind (no fruit visible, not committed) movement is a CONSTANT: `dist = speed * blind_explore_frac`
(0.12), unless `ef < low_energy_frac` (0.35), in which case an `elif` placed earlier OVERRIDES it with
`speed * 0.4`. So the fleet moves 3.3x FASTER blind as it starves and 5x SLOWER than it could when rich.
The mechanism replaces that constant with a monotone function of the agent's own energy fraction:

    blind_dist = speed * (esearch_lo + (esearch_hi - esearch_lo) * clip((ef - ef_lo)/(ef_hi - ef_lo), 0, 1))
    esearch=1.0, esearch_ef_lo=0.30, esearch_ef_hi=0.70, esearch_lo=0.12, esearch_hi=0.60

One knob (`esearch`), one structural change, no other parameter touched. Off => byte-identical to V2.

## Why this and not something else
It is the only mechanism on the table that is (a) absent from V2's structure, (b) specifically named by
the energy budget (movement spend is flat at 2,500-3,000 energy/1k while income falls 94%), and (c) large
enough to be visible above the measured per-seed noise (2,595 ticks).

## Arms (same seeds, paired, deterministic)
| arm | params |
|---|---|
| BASE | deployed 52-key GS set (= the serving controller 252f0ba1, behaviourally) |
| esearch1 (PRIMARY) | BASE + esearch=1.0 |
| esearch1_hi030 (secondary, dose diagnostic) | BASE + esearch=1.0, esearch_hi=0.30 |

## Design
- 160 seeds, 300000-300159. FRESH: no overlap with any seed ever used (screens 20000+, holdouts 90000+,
  arms 5000-5159, screens 1900-1949, holdouts 1301-1340, residual 8000-8139).
- horizon 18,000 ticks (median extinction ~7,500; the cap is above 95% of episodes).
- every arm on the SAME seed; paired per-seed differences; fresh process per episode (maxtasksperchild=1).
- cost: 3 arms x 160 seeds = 480 episodes, ~15-25 min on 60 workers of the 64-core box.
- the mechanism's firing is receipted: mean commanded distance on BLIND ticks is recorded per arm and
  compared with BASE, so "no effect" can be distinguished from "the block never ran".

## Thresholds (fixed now, before the data)
Noise reference: per-seed SD of the paired difference 2,595 ticks => SE(160 seeds) ~ 205 ticks = 2.6% of base.
- SUCCESS: paired mean delta >= +400 ticks (+5.1%) AND 95% CI lower bound > 0 AND arm p10 >= base p10 - 5%.
- FAILURE: paired mean delta < +400 ticks  =>  the energy-conditioned movement family is CLOSED, with no
  dose expansion and no parameter rescue.
- The secondary arm is a DOSE DIAGNOSTIC only. It can inform the interpretation of a null; it is not a
  second lottery ticket and a win there does not license deployment.

## Deployment rule
This run alone does NOT authorise any production change. A winner (>= +400, CI > 0) must then be confirmed
on a second, fully fresh 160-seed block (310000-310159) before any deploy decision. V2 stays deployed and
the serving container is not touched by this experiment.
