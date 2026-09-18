# PIVOT V2 — "UPGRADE THE SENSORS, SHRINK THE SWARM"
### (all insights · all constraints · all dead ends · the pivot · the honest ML verdict)
Written 2026-09-18 ~23:25Z, before any of it was built. Supersedes `PIVOT_DESIGN.md` §3 (that document's
W1–W6 portfolio stays valid as *lanes*; this adds the one mechanism none of them touched).

---

## 0. THE HEADLINE: the target is not a guess, it is occupied

Read from the official public leaderboard API (`cases.nordicaicup.com/api/v1/leaderboard/validation`,
fetched 2026-09-18 23:1xZ — 75 teams have a survival-simulator score):

| # | score | team (country) |
|---|---|---|
| 1 | **1794.35** | Cybotrix (IS) |
| 2 | 1743.34 | Elemental hero (DK) |
| 3 | 1712.05 | North GreatWall (SE) |
| 4 | 1667.59 | Iftikhar Amiri (NO) |
| 5 | 1661.32 | execve (DK) |
| 6 | 1633.63 | Hannah Family (FI) |
| 7 | 1550.18 | Childbeating Catboost Connoisseurs (SE) |
| 8-14 | 1418.75 · 1402.97 · 1386.41 · 1312.15 · 1300.46 · 1259.17 · 1205.72 | (FI/DK/NO/SE/IS) |
| … | | |
| **us** | **918.33** best official sample (798.42 with E3) | |

Median of the field **927.4**; 39 teams above 900, **14 above 1200, 7 above 1500**. So:
* 1,800 is not a stretch goal invented at home — team #1 is sitting on **1794.35 = ~17,940 ticks**;
* we are not behind because we lack ML. We are **at the field median**, ~2x off the front, and the front
  is reachable with the tools we already have (nothing in the leaderboard suggests anyone has more compute);
* the *evaluation* leaderboard (the one that is actually judged: 3 runs averaged, one try) has only
  **2** survival entries so far (529.36, 497.43) — almost nobody has spent their one shot.

## 0.1 The cap question, answered with numbers
* `score = sim_time` = ticks × 0.1, and the run ends on the last death OR at **30,000 ticks (score 3,000)**
  — there is no 1,800 ceiling in the environment.
* The 600 s **accumulated-wait** cap is the only other limit. Our measured per-tick wait is **9–18 ms**,
  so the wait budget buys **33,000–66,000 ticks** — it does not bind at 18,000 (that costs 162–324 s).
* It HAS bitten us once: attempt #3 scored 580.96 and ended with *"agent server bottleneck, accumulated wait
  exceeded 600 s"* — 5,452 requests / 686 s = 110 ms/tick, i.e. the tunnel era, not the policy.
* **Conclusion: you were right.** Raising "the cap" is not the lever — nothing is capping us. The agents
  die. And the sim's own ceiling is 3,000, so 1,794 is 60 % of everything available.

---

## 1. ALL THE INSIGHTS (each one measured in this repo; the number that proves it is included)

| # | insight | evidence |
|---|---|---|
| I1 | **Score is per RUN, not per agent**: `score += dt` fires once per tick regardless of fleet size. Fleet size is therefore a pure cost against a fixed reward (0.1/tick metabolism + 100/spawn + a −E/100 liability per death). | environment.py:760 |
| I2 | Survival dominates: a fruit is worth `+0.06` = **0.6 ticks**; a run is worth its ticks. | environment.py:671 |
| I3 | **Income is ACCESS-limited, not supply-limited**: 76–87 % of agent-ticks are blind, 0.26–0.42 fruits visible per agent, while **2,000–6,500 fruit energy stands in the world**. The fleet starves surrounded by food. | journal §4, §18.3 |
| I4 | The world **decays**: tree/fruit production × `0.5^(t/300)` — halves every 3,000 ticks; fruit rots at 100 s. Late income → 0. | environment.py:742 |
| I5 | Death mix at the official horizon: **starved 49–51 %, eaten 23–32 %, aged 16–27 %**. Starvation is the plurality cause. | journal §18.3 |
| I6 | **The sprint lockout is real**: below `max_energy/5` (100 base) the sim silently clamps you to walking speed, and 92 % of predation deaths happen inside that zone (mean energy at death 76–79, median 36–64). Predation kills the broke. | environment.py:512, journal §18.3 |
| I7 | …but **defending the reserve LOSES every time**: 5 independent arms, monotone in constraint strength (L1 −11/−17 %, L2 −9/−25 %, L3 −18/−19 %, L4 −24/−36 %). Movement and births ARE the income. | journal §18.1 |
| I8 | **Predators are never removed and spawn ∝ t²** (`(1/N)·dt·t·1e-4`): ~4 at t=900 s, ~**16 at t=1800 s (18,000 ticks)**. Measured on seed 100: 0 → 2 (t≈300 s) → 3 (t≈600) → 5 (t≈700). So the endgame is not "bank and coast" — it is an **avoidance race against a monotonically rising density** while production decays. | environment.py:763 + `experiments/_traits_seed100.out` |
| I9 | Predators **cannot kill while resting** (>100 energy wakes them; 30/s while resting) → a chase is a **~40-tick burst** followed by a **~33-tick sleep**. Surviving a predator is a timing problem, not a marathon. | environment.py:679-684 |
| I10 | The genome is **heritable, mutated and fully observable**: 10 %/trait per birth, `U(0.5,1.5)`, caps **2x** the start on vision (400), max_energy (1000), speed (20), sprint (40), and 2x hearing (100)/cone (π/2). The policy chooses who breeds → it can run **directed evolution inside its own fleet.** | environment.py:296-345, DTOs.py:18-24 |
| I11 | That axis is **live and already paying by accident**: with the deployed controller, 119 births in 7,292 ticks on seed 100 produced newborns with **mean max_energy 589.7 (+18 %), max 996.8 (≈ the 1,000 cap, 2x default)**; hearing +4 %, speed +2 %, but **vision barely moved (mean ~200, max 257.8)** — nothing selects for it. | `experiments/_traits_seed100.out` |
| I12 | A **speed-20 genome walks (0.05/unit) faster than any predator sprints (15)**, and cost per unit is genome-independent — so faster/higher-vision agents buy coverage and escape at no energy penalty. | environment.py:501-518 |
| I13 | The observation is **non-Markov** (no clock, no memory, no genome of others) — the same wall that killed IL ("the missing state is where the food I saw earlier") and memoryless search. | IMITATION_REPORT, RL_JOURNAL |
| I14 | macOS/ARM and Linux/x86 disagree wildly (4,472 vs 7,816 ticks, same seed+policy) → only Linux numbers count. | journal §6 |
| I15 | Official noise: same controller scored **768.68** then **918.33** five hours apart. Local ±500–1,000. Nothing under +500 on ≥20 paired seeds is a result. | journal §17.1 |

## 2. ALL THE CONSTRAINTS
* **C1 compute**: 4 vCPU VPS + a 10-core Mac. A 5-arm × 20-seed block at horizon 18,000 costs 55–70 min
  per container pair. From-scratch deep RL is off the table (the closest published win, Lux AI S3, needed
  ~20 B steps and a rented cluster — ~3 orders of magnitude more than we have).
* **C2 cardinality of proof**: ±500–1,000 local noise, ±150 official noise → a change must clear +500 on
  ≥20 paired held-out seeds AND win the majority of them.
* **C3 grading**: Linux only; validation = random seeds, unlimited attempts, one at a time; evaluation =
  **one** attempt, 3 runs averaged, preset seeds. Never rebuild/restart the endpoint while anything is queued.
* **C4 reproduction is load-bearing**: agents die of age at 600–1,200 ticks. Any gate that stops births
  kills the fleet (this is how 4 separate arms died).
* **C5 movement is the income mechanism**: every speed cut lost. "Movement = 47 % of burn" was the wrong
  reading — travel is how food is found.
* **C6 break-even economics**: the fleet lives at break-even, so every thrift measure reduces income faster
  than cost. This law has now been confirmed 5 times.
* **C7 code hygiene**: ONE canonical `best_controller.py`; any edit needs `tools/check_controller.py --update`
  or the Docker build refuses; new code must be off-by-default and inherit-identical when off.
* **C8 shared resources**: another agent has used the same VPS; `experiments/LANES.md` (max 2 test containers,
  never touch `nac-survival-vps`) governs.
* **C9 time**: the cup's national round runs 17–20 Sep; it is now 18 Sep ~23:25 → **~1.5 days**.

## 3. DEAD ENDS — already paid for; do not spend on them again
thrift/reserve/spawn-margins (5 arms, monotone loss) · population caps and repro gates (broke the age relay) ·
banking/late-phase switches (never reached, dead by ~4,600) · phase movement cuts (4,000–7,500 → −3,032) ·
memory search as priced (2x burn; −14.5 %) · tree chasing (−555) · bigger fleets (−1,017) ·
`face_predator` (physically unobservable: the wind-up state is invisible) · behaviour cloning (45 % of the
teacher) · from-scratch PPO (74 ± 9 vs heuristic 210).

## 4. THE PIVOT

**Claim: the fleet is a SEARCH SYSTEM with three binding limits, and two of the three are properties of the
agents' BODIES, which we have never made a decision about.**

| binding limit | why it binds | what fixes it | can the heuristic do it today? |
|---|---|---|---|
| ACCESS / blindness (I3) | vision cone 200 × ±30°, hearing 50 → 76–87 % of ticks blind | **vision 400 + cone π/2 + hearing 100 ≈ 6x the swept area** | no — the controller can only steer, never upgrade a sensor |
| the endgame bank (I4, I6) | metabolism 0.1/tick vs max_energy 500; 12,000 ticks drains 1,200 | **max_energy 1,000 = 2x the bank**, and the sprint threshold moves with it | no |
| the predation cascade (I6–I9) | below `max_energy/5` you cannot sprint; predator sprints 15 | **speed 20 walks faster than a predator; sprint 40 leaves it behind** | no |

All three are purchasable **without extra energy** — the mutation channel already exists, the caps are 2x the
starting values, and the genome of every agent is in the observation. **The only thing missing is a policy
that decides who reproduces.**

### V2 design (four layers, each independently switchable, all off-by-default)

**A. `genome_read`** — read the six heritable traits for the agent itself and for every observed
conspecific (already in the payload). Build a scalar *genome utility*:
`U = w_v·(vision/400) + w_c·(cone/(π/2)) + w_h·(hearing/100) + w_e·(max_energy/1000) + w_s·(sprint/40) + w_m·(speed/20)`
with the *perception* group dominant by default (it attacks I3, the plurality death cause).

**B. `breeder_select`** — a spawn is permitted only from an agent whose `U` is in the top-1/top-2 of the
observed fleet (plus the existing cooldown, plus an unconditional **lineage rescue** at pop ≤ 1 so the relay
can never be broken — the failure mode that killed four earlier arms). Every birth is a new draw from the
parent's genome, so a top-2 rule ratchets the population's genes while leaving the foraging engine intact.

**C. `swarm_target` (coupled, retested not assumed)** — because score is per RUN (I1), the optimum fleet is
"as small as income allows". The thin-relay idea was refuted **on the default genome**; with 3–6x perception
per agent it must be re-measured — a single high-vision, high-cap agent needs far less search than 12 blind
ones. Two arms (target 6 / target 12) decided by measurement, not by argument.

**D. `commit_search` (repriced)** — the earlier memory trial was mis-priced at ~2x burn (walk 0.5/tick vs the
live 0.225/tick). Retest at walk cost with an explicit acknowledge→commit→verify structure, and with the
higher-vision genome (which changes how much search is even needed).

### Why this is a pivot and not another knob
It changes the **state and the decision space**, not a scalar. Every one of the five refuted families was a
*constraint* on behaviour; this is an *expansion* of capability, and it is the only expansion the simulator
gives away for free. It also composes with everything already built: foraging, evasion (live), relay logic
(all kept), and it is orthogonal to the running residual-RL lane.

### Pre-registered experiment plan (all on Linux, horizon 18,000, 20 paired fresh seeds per arm)
| exp | arms | bar / kill criterion |
|---|---|---|
| E-V2-0 (diagnostic, 40 min) | `genome_read` ON, no behaviour change | **mechanism gate**: mean fleet `vision_range` must rise ≥50 % and stay up within 4,000 ticks. If it does not, the lane is dead regardless of score. |
| E-V2-1 (2 lanes, ~70 min) | LIVE vs breeder_select top-1 vs top-2 | +500 ticks AND ≥15/20 seeds won |
| E-V2-2 (~70 min) | selection objective: perception-only / balanced / energy-first | pick the best; report all three |
| E-V2-3 (~70 min) | thin swarm (6 / 12) × winning selection | decided by the same bar |
| E-V2-4 (concurrent) | W1 residual RL confirmation | its own pre-registered bar (income AND survival both up) |

### Weaknesses of this pivot, stated before starting
1. **Chicken-and-egg**: selection needs births, births need surplus, and the fleet dies at ~7k ticks — the
   ratchet may only start paying after the wall we cannot yet pass. Mitigation: selection never blocks the
   relay (top-2 + rescue), and the diagnostic arm measures trait drift *independently* of score.
2. **Selection pressure can fight income**: the richest parent is not necessarily the best genome. Mitigation:
   top-2, never top-1-only, and the E-V2-2 arm measures this directly.
3. **An unbiased ±50 % mutation is a random walk** — without strong selection it drifts nowhere (that is
   exactly what I11 measured passively: max_energy up, vision flat). Selection strength is the whole bet.
4. **Noise**: the effect must clear +500 on 20 paired seeds; a 12 % win in one block is not a result.
5. **Latency**: if the genome layer ever makes a tick cost >20 ms, 18,000 ticks stop fitting inside 600 s.
   Budget: keep the policy ≤20 ms/tick and log it.

## 5. THE ML VERDICT (the direct answer to "maybe no heuristic at all?")
* **Imitation**: teacher-bound. Measured 45 % of the teacher's survival; a top-3 Lux team says it in words —
  "nearly impossible to train an IL agent to outperform its teacher". Cannot reach 1,794 from a 918 teacher.
* **From-scratch MARL**: the winning published recipe (Lux S3, 1st place) = IMPALA + V-trace + self-play +
  a frozen teacher + an opponent pool at **~20 B steps**. We have ~3 orders of magnitude less compute and
  **no adversary** (predators are scripted, so self-play has no opponent to co-evolve with). Dead.
* **Residual RL on frozen H1 (W1, running now)**: alive and the right ML shape — zero-initialised so it
  cannot regress, ~10x sample efficiency vs from scratch, base policy needs no clone. But it can only
  **nudge distance and turn**: it cannot invent memory and it cannot invent selection. Treat it as a
  *refiner*, never as the pivot.
* **What a learned policy would actually need to win here** (if we do ML at all): recurrence (the failure is
  non-Markov — I13), the **genome features in the observation**, and a mid-game curriculum. That is a
  different observation space and a different architecture from the 31-number MLP that has been tried twice.
* **Recommendation**: keep the heuristic. Restructure what it can *see* (memory, genome) and what it can
  *choose* (who breeds). Then let residual RL polish it. "No heuristic at all" throws away the only part of
  the system that currently scores.

## 6. WHAT I NEED FROM YOU
1. **Approve V2-A/B (genome + breeder selection) as the primary lane**, with the mechanism gate E-V2-0 first.
2. **VPS access**: `ssh root@212.147.239.222` is denied for my key (`~/.ssh/vps_hermes` worked for the
   Frankfurt box). Give me the Helsinki user/key, or I will run the arms on the Mac and confirm the winner on
   the VPS later — but then the *first* measurement is on a platform that disagrees with the grader (I14).
3. **The residual-RL trainer is using ~600 % CPU on the Mac right now** (8 workers). Do I leave it, or stop
   it so the measured arms are not competing for the same cores?
4. **Anything already in flight that I would collide with** (another agent's lane / a validation you have
   queued) — I will follow `LANES.md` and claim a lane before launching.
