# PRE-REGISTRATION — search-effort direction, second structural test (2026-09-20, before the run)

## What the first test established (mech_ab, 160 fresh paired seeds, 300000-300159)
`esearch1` (energy-conditioned blind search, rich 0.12->0.60, broke 0.4->0.12) = **-1637 +- 258 ticks,
t = -6.36, W/L 52/108** against the deployed V2. Blind-tick mean commanded distance fell 3.99 -> 3.02.
So the arm mostly REDUCED blind travel (the broke regime dominates agent-ticks) and lost 21% of survival:
less search -> less coverage -> starvation. The energy-conservation reading of "movement is income-blind"
has the wrong sign.

## Therefore this is the other sign of the same lever, isolated
`blind_explore_frac` (0.12) applies ONLY when `ef >= low_energy_frac` (0.35): below that, an earlier `elif`
sets `dist = speed * 0.4`. So the first test confounded two regimes. This test moves ONLY the rich-regime
constant, i.e. "spend more on search when you can afford it", and leaves the broke regime untouched.

| arm | override | meaning |
|---|---|---|
| BASE | none | deployed V2 (52-key GS set) |
| bl40 | blind_explore_frac = 0.40 | rich blind search matches the broke regime's crawl |
| bl60 | blind_explore_frac = 0.60 | rich blind search 5x the baseline constant |

## Design (identical to the first test, so results are comparable)
160 FRESH seeds 300320-300479 (no overlap with 300000-300319 or any older block); horizon 18,000;
all arms on the same seed; paired per-seed; fresh process per episode; blind-tick distance receipted.
Cost: 480 episodes, ~15-20 min on 60 workers of the 64-core box.

## Pre-registered thresholds
SE(160 seeds) ~ 205-260 ticks.
- SUCCESS: paired mean delta >= +400 ticks AND 95% CI lower bound > 0 AND arm p10 >= base p10 - 5%.
- FAILURE (either arm): paired mean delta < +400  =>  the CONSTANT-search-effort family is CLOSED.
  Combined with test 1 (strongly negative when effort falls), a null here means search effort is already
  at its optimum in both directions, and no further movement-constant work is justified.
- A winner is NOT deployable on this run alone: it requires a second fresh 160-seed confirmation.

## Explicitly NOT in scope
No other knob joins this test. No dose expansion after a null. The learning tracks (ES with a
statistically usable seed regime, recurrent residual PPO) are the main compute consumers.
