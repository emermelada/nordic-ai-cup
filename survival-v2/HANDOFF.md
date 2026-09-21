# Survival simulator — handoff

Branch `survival-v2` in `github.com/emermelada/nordic-ai-cup`, directory `survival-v2/`. Written 20 Sep 2026.
Nothing here depends on any other branch; the official simulator is vendored under `official/`.

**Where it stands**

| | |
|---|---|
| Controller | `hive.py`, served by `server.py`; best commit `3d1739a` |
| Local strength | **1484 +- 51** over 40 unseen seeds (101-140), one game swings about +- 350 |
| Best platform validation | **799.6**, cut by the grader's response-wait budget, not by the colony dying |
| Scoreboard context | 1743 / 1661 / 1403 / 1377 / 1312 for places 1-5 as of 19 Sep |
| Binding constraint today | **latency to the grader**, not the controller |

---

## 1. The rules that shape everything

Read from the vendored simulator, not from the README; several of these are not documented.

* Score = 0.1 per tick survived (3000 s cap) + fruit energy / 1000 - eaten agent energy / 100. Survival time
  dominates: everything else is worth a few dozen points.
* An agent pays **0.05 energy per commanded pixel** while walking and 0.5 per pixel above walking speed while
  sprinting, **1 energy per second** just to live, and `min(pi, |turn|) / 2pi` per turn. Walking at the evolved
  speed of 19-20 px/tick costs ~10 energy/s, ten times the living cost: **movement is the budget**.
* A biome multiplies the *distance covered*, not the cost: swamp 0.5, river 0.3, desert 0.8. Crossing a swamp
  costs twice the energy per pixel of progress. Energy drain per biome is identical everywhere (1.0).
* Fruit spawns 20-60 px from a tree, starts at 20 energy, gains 2/s up to 60 at 20 s old, and rots at 50 s.
  Nothing except our agents eats fruit, so **waiting for a fruit to ripen is nearly free and doubles it**.
* A tree fruits from age 20 until it dies (median age 58), producing ~0.1 fruit/s in forest and grassland.
  So a tree yields ~3.3 fruits ~ 200 energy over its life. Tree population halves every ~600 s:
  ~90 at t=100, 43 at 600, 20 at 1200, 10 at 1800, 5 at 2400.
* Agents die of old age: `max_age ~ U(60, 120)`, after which they lose an extra `0.01 * age` per tick, i.e.
  6-12 energy/s. An old agent's stored energy is worth far more as children than as extra seconds of life.
* Spawning costs the parent 100 and the child starts at 75 with 10%-per-trait mutations (x U(0.5, 1.5)),
  capped at speed 20, sprint 40, max_energy 1000, hearing 100, vision 400, cone pi/2.
* Predators: speed 11, sprint 15, hearing 60 px, vision 250 px in a +-30 deg cone, **occluded by walls**.
  Each chases only the closest agent it perceives. It charges when that agent looks away or is within 90 px;
  otherwise it circles at sprint speed, closing ~10.6 px/tick. It has 200 max energy, wakes at >100, and
  sprinting costs it 2.55/tick, so **a chase lasts at most ~4 s** before it rests ~3.4 s. Their number grows
  as 0.01 * t: 9 at t=900, 15 at 1500, 20 at 2000.
* Quirk: when an agent starves inside the update loop, the next agent in the list is skipped and receives last
  tick's observations. Detected by its age not advancing (`m.stale`), and ignored.

## 2. How the controller works

One `Hive` object decides for the whole colony each tick; agents share one world model.

**Pose tracking.** The platform sends only relative observations, so positions are dead reckoning: replay our
own action through the simulator's rules (caps, biome multiplier, collision deflection against mapped walls).
Frames merge when one agent sees another (the sighting gives an exact relative pose) and snap to world
coordinates when anyone sees a 1600 or 1200 px boundary wall. Corrections come from re-observed wall faces
(each face has its own random length, so a match gives the position error directly) and, new yesterday, from
peers: an agent that moved outside its own vision cone is flagged unverified, and any recently confirmed agent
that sees it sets its exact position.

**Shared map** per frame: wall faces as 30 px slabs, trees, fruit with spawn-time brackets, predator tracks,
a hearing raster recording when each 10 px cell was last within someone's hearing radius, and a 10 px
occupancy grid used for path planning.

**Foraging as expected value.** For every (agent, fruit) pair: the energy the fruit will be worth when eaten
at the best moment (now, or after waiting for ripeness), times the chance it has not rotted, minus walking
(0.06/px) and waiting (0.8/s), plus a hunger bonus, times 0.3 for agents that will never pass the energy on.
Agents that arrive early wait 18 px away and scan for predators. Fruit whose spawn time is unknown uses a
measured prior: the median unknown fruit is 3 s old when first seen, so it is treated as ripe 16 s after first
sighting. Moves are bent so they never *end* on a young fruit, since touching one eats it at 20-40 energy.

**Path planning.** A terrain-weighted grid Dijkstra field per target (scipy, cached per target cell), used
only when the straight line crosses a known wall. Before this, 12.5% of all walking energy went into moves
with no net progress — agents bouncing off obstacles.

**Camps.** Idle agents sit at the nearest free fruiting tree, claimed through other agents' targets rather
than their positions, with 60 px spacing so one predator cannot take several in a row.

**Breeding.** Fitness-weighted (speed 6, hearing 3, vision 0.7, cone 0.7, sprint 0.8), measured against the
75th percentile alive rather than the best. Elites breed with a reserve that scales with rank; anyone flagged
old dumps its energy into children immediately; weak genomes neither breed nor eat once the colony is big
enough. Old agents only eat fruit that lifts them over 100 so they can spawn at once.

**Evasion.** Threats are predators that can actually perceive the agent (their real geometry), filtered by
their closest-prey rule. Response depends on distance: full speed inside the 95 px charge zone, back off at
11 px/tick between 95 and 160 (just above the 10.6 px/tick a circling predator closes), drift otherwise, while
facing it so it circles instead of charging. Direction is chosen with a 6-tick look-ahead over 16 headings
that models terrain slowdown, walls, and each predator charging straight.

## 3. How it was measured

* `bench.py` runs the fast simulator (`fastsim.py`, bit-exact with the official one, 8.5x faster) over many
  seeds in parallel; `paired.py` reports per-seed differences between two runs.
* **40 seeds minimum** for any decision. One game's standard deviation is ~350, so 20 seeds cannot separate a
  100-point change from noise. Seeds 101-140 were never used for tuning.
* Diagnostics, each of which found something: `eatdiag.py` (fruit energy by eater's situation),
  `agediag.py` (how old fruit is when first seen), `deathdiag.py` (late-game deaths by cause, with walking
  per mode), `fleediag.py` (flee episodes and whether the predator was really chasing), `stuckdiag.py`
  (energy spent on moves with no progress), `trackfail.py` (first tick where a pose belief drifts),
  `lasttrace.py` (last seconds of agents that starved).

## 4. The measured path from 1239 to 1484

| Change | 40-seed mean | Paired diff | Why it worked or not |
|---|---|---|---|
| starting point (camp claims) | 1239 | | |
| ripe-aware fruit utility, never step on young fruit, scan while waiting | 1259 | +20 +- 85 | energy per fruit 42 -> 50; fewer fruits eaten, more rot |
| **A\* paths around known walls** | 1355 | **+96 +- 58** | recovered most of the 12.5% wasted walking |
| old agents eat only to spawn at once | 1391 | +36 +- 62 | old-age drain 14.9k -> 12.2k per game |
| **peer pose fix + 29 px wall fixes** | 1484 | **+94 +- 67** | removed the corrupted-map early deaths |

Rejected, all on the same 40 seeds: flee toward wall occlusion (-40), suppressing mapping by unverified
agents (-38), bigger late-game food radius (-5), late-game birth reserve 200 (-6), mid-game cap 12 (-6),
threat hysteresis (-69), predator-aware fruit and tree choice (-39), slower scanning (-83), less cautious
fleeing (-134), tree choice by expected remaining productivity (-79).

**The pattern worth remembering:** every change that made agents *more cautious* cut predator kills and still
lost score, and the one that made them less cautious lost more. Kills are not the binding cost; walking is.

## 5. Where the remaining loss is

From `deathdiag.py` on the current build (last 400 s of 12 games): 41% of deaths are young agents starving,
39% old age, 20% predators. The young starvers live 38 s, eat 2.3 fruits, and walk 1745 px, of which **847 px
is fleeing** — more than food and camp moves together. `fleediag.py`: ~3600 flee episodes per game with a
median length of 2 ticks, and half are against predators that are not chasing that agent.

So the late game is predator-*cost* limited, not predator-*kill* limited. The obvious fixes (flee less, avoid
danger) were tested and all lost, because they also cut foraging. What has not been tried:

1. **Stop the flee/return flicker without becoming cautious.** Hysteresis was tested as "treat it as a threat
   longer", which made agents flee more. The opposite framing is to let an agent finish its current fruit when
   the predator is not actually closing, i.e. commit to a short plan instead of re-deciding every tick.
2. **Estimate predator energy.** A predator can only sprint ~4 s from a fresh 100 energy, then must rest 3.4 s.
   Tracks already exist; adding an energy estimate from observed movement would let agents stop fleeing from
   an exhausted predator, which is most of what a 2-tick episode is reacting to.
3. **Route foraging around danger instead of avoiding targets.** The hazard test penalized *targets* near
   predators. Putting the hazard into the A* cost map instead would keep the fruit and change only the path.
4. **Give children a feeding plan.** 24% of young starvers never eat at all. A birth could be conditioned on a
   free fruiting tree or a claimed ripe fruit within reach of the parent's position.
5. **Retune the knobs the new mechanics changed.** `wait_w`, `undated_wait`, `dist_cost`, `food_range*` and
   `camp_spacing` were all tuned before ripeness-aware foraging and path planning existed. A CMA-ES run
   (`tune.py`) needs a many-core box: the laptop could not finish one generation in 40 minutes.

## 6. Infrastructure — the real blocker

The grader (46.62.240.126, Hetzner Helsinki) stops a game once it has spent **1200 s waiting for responses**
(raised from 600 s on 19 Sep). Our server answers in 3-10 ms, so the wait is almost entirely network.

| Path | Wait per step | Game ends at | Score |
|---|---|---|---|
| Cloudflare quick tunnel to the laptop | ~150 ms | t ~ 800 | 799 measured |
| Any European VPS (25-30 ms RTT from here to the grader) | ~35 ms | t ~ 3000 | model's real level |
| Same datacenter | ~5 ms | no cut | model's real level |

Tried and failed: direct IPv6 to the laptop (router or grader blocks it), IPv4 port forwarding (ISP appears to
share the public address), a teammate's VPS at 94.237.32.31 (unreachable from here, firewall or offline).

**What is needed:** any small VPS in Europe, Ubuntu, 2 vCPU. Deploy with
`docker build -f deploy/Dockerfile -t surv-v2 . && docker run -d -p 9053:9052 --restart unless-stopped surv-v2`,
or without Docker via `deploy/push.sh <ip> first` (systemd unit, uvicorn with uvloop). Submit
`http://<ip>:9053/predict`. `deploy/requirements-server.txt` is numpy, orjson, uvicorn[standard], scipy.

The evaluation runs three games back to back and averages them, so the server must stay up throughout, and
`server.py` already survives any exception by returning no-op actions for that tick.

## 7. Expected result

On a low-latency host, one game averages 1484 with a standard deviation of ~350; the three-game evaluation
average lands between roughly 1150 and 1800 with a median near 1480. Through the tunnel it is capped near 800
whatever the controller does. The single highest-value action left is therefore the VPS, not the algorithm.
