# Survival v2: measured results

All scores are from `bench.py` on the fast simulator, which is bit-exact with the official one. Runs are
deterministic per seed, so seed sets are comparable across versions. One game's score is roughly the seconds
survived. Mean ± SE over the seeds listed.

## Simulator and infrastructure

| Item | Result |
|---|---|
| fastsim vs official, seed 1 × 3000 ticks and seed 11 to extinction | identical positions, energy, score and RNG state every tick |
| Speed | 1.1–3.5 ms/tick vs 9.6–17.5 official (8.5×) |
| Determinism | same seed + same policy gives the same score (checked twice on 6 seeds) |
| Server (orjson, bare ASGI) with the official client on this laptop, 42 agents | 7.2 ms mean per request, p99 15 ms, which projects to 215 s of the 600 s per-game budget |

## Controller progression (with predators)

| Version | Seeds | Mean score | Notes |
|---|---|---|---|
| v1 first smoke | 1 | 308 | pose tracking broken (stale observations) |
| + stale-observation guard, wall-fix fixes | 1–8 | 343 | tracking exact on most seeds |
| + rot-check fix (unknown-age fruit was skipped) | 1–8 | 437 | |
| + need-based fruit value, threat gating | 1–10 | 570 | |
| + ripe-only eating, fitness-ranked breeding | 1–10 | 709 | |
| + mechanics-based evasion (face within ±90°, back off) | 1–10 | 813–833 | kills still ~58–73/game |
| + sprint reserve, speed-weighted fitness, spread camping | 1–10 | 1014 | speed 15 at t=900 |
| + weak genomes may not dump at old age | 1–20 | 1012 | speed 15.1 at t=300, 19.7 at t=1200; kills 37 |
| + territorial food radius 90 px | 1–20 | 961 | |
| + retire weak genomes (no food) | 1–20 | 1016 | |
| + barren-tree detection, stale-cell exploration | 1–20 | 1034 | |
| + quantile breeding reference | 1–20 | 1023 | |

## Diagnostics that changed the plan

| Test | Result | Meaning |
|---|---|---|
| Kill post-mortem (3 seeds, 293 kills) | 98% of victims knew the predator beforehand, 97% died fleeing, 82% could not sprint | not a perception problem: speed/energy and herding |
| Trace of one kill | the predator charged a cluster and killed two in a row | spread out; keep sprint energy |
| No predators at all | mean 1497–1660, 0/10 reach 3000 | **the economy caps survival**, not predators |
| Energy ledger (no predators) | movement 36% of eaten energy, old-age drain 31% | energy is wasted, not missing |
| Old-age drain per agent | weak genomes 316 energy over 29 s of old age (they keep eating); good ones 76 | weak genomes must not eat |
| Harvest | 75–90% of spawned fruit eaten until t≈1300 | supply is harvested; conversion into children is the leak |
| Selection crash | at t=240, 21 of 23 agents retired when one better mutant appeared | measure "weak" against the 75th percentile, not the best |
| **Oracle** (true trees, fruit and fruit ages; no predators) | mean 1630 s, still extinct with fruiting trees left | **decision logic**, not perception |
| Oracle trace near extinction | campers starve with 13–26 ripe fruit > 100 px away; E 100–180 never reaches the 221 breeding threshold | food radius and late breeding threshold |
| Oracle with 500 px food radius | 7/10 seeds reach 1740–2190 s; seeds 4 and 7 now die early (792, 340 s) | radius helps late, hurts some openings |

## Afternoon, 19 Sep (seeds 1-20 unless noted; 40-seed rows are the reliable ones)

| Change | Mean | Notes |
|---|---|---|
| look-ahead flee over terrain (swamp 0.5, river 0.3) and walls | 1177 | 2 of 3 fast-agent kills were in swamps, 1 ran into walls |
| closest-prey threat filter (a predator chases only the closest agent it perceives) | 1186 | late flee time 40% -> 30% |
| colony birth brake (mean energy < 100) | 1154 | early collapses 2 -> 1 of 20; hurt the late game |
| brake only for colonies >= 15 agents | 1122 | 0 of 20 games under 500 s |
| 60 px spacing between campers | 1230 | chains of 4-5 kills in seconds came from adjacent campers |
| spacing 100 / 150 | 1218 / 1135 | 60 kept |
| retire weak genomes only in colonies >= 15 | 1019 | retiring at >= 6 is load-bearing |
| grab close fruit when starving under threat | 1069 | disabled |
| **baseline on seeds 101-140 (40)** | **1000 +- 52** | 6/40 under 500 s |
| **camp occupancy from claims instead of positions (seeds 101-140)** | **1239 +- 56** | switches 9-14 -> 3.6 per agent-minute; 2/40 under 500 s; 20/40 alive at 1200 (11) |
| no predators (seeds 101-120) | 1608 | the energy economy is the ceiling |
| tree choice favours trees with a long productive future (seeds 101-140) | 1160 +- 51 | paired diff vs claims -79 +- 69: reverted |

## Where the food goes (eatdiag.py, seeds 101-110, claims version)

| Eater's situation | Share of fruit | Mean energy |
|---|---|---|
| heading for that fruit, ripe | 32% | 58.5 |
| heading for that fruit, age unknown to the hive, turned out young | 28% | 31 |
| heading for that fruit, known young (hungry or old agents) | 14% | 33 |
| walked over it on the way to another fruit | 12% | 42 |
| camping, fleeing, exploring (walked over it) | 12% | 36-40 |

59% of all fruit is eaten under 15 s old. A fruit is worth 20 + 2/s up to 60 at 20 s and rots at 50 s, and
nothing else eats fruit, so waiting is almost free: this is the largest energy leak found so far (~30% of
the food energy).

## Platform validation 1 (tunnel, 19 Sep 11:16 UTC)

Score 320, no errors. Grader IP 46.62.240.126 (Hetzner Helsinki). 37 agents at t=100, 36 at t=200, game over at t~293:
the early boom-and-bust collapse seen in ~10% of local games (fixed since by the brake and spacing). Requests
~126 ms apart through the tunnel: the 600 s wait budget would cut a game at t~500-650, so the final run needs a
server in Helsinki.
