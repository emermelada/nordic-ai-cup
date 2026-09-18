# COMPLETE PARAMETER INVENTORY — survival-simulator (every knob, three layers)

Written 2026-09-18 ~23:20Z. Every value below is read out of the code in THIS repo (`src/`, `best_controller.py`)
or out of the organisers' published README (`README.md`, cross-checked against
https://raw.githubusercontent.com/amboltio/Nordic-AI-Cup-2026/main/survival-simulator/README.md — byte-identical
except our added serving section). Line numbers are the ones in this repo.

---

## LAYER 0 — GRADER (what our score actually is)

| item | value | source |
|---|---|---|
| score | `score = sim_time` (seconds) `+ fruit.energy/1000` (max +0.06/fruit) `− victim.energy/100` on being eaten | environment.py:671, 674, 726, 760 |
| score rate | `+dt` = **+0.1 per tick, ONCE per run regardless of fleet size** | environment.py:760 |
| run ends when | the LAST agent dies, OR the sim reaches **3000 s = 30,000 ticks** (score 3000) | README.md:19 |
| wait cap | accumulated response wait **600 s ends the run**; 10 s per single response | README.md:165 |
| effective ceiling | `score ≈ min(ticks/10, 60 s… i.e. 600/latency_per_tick/10)`. At our measured **9–18 ms/tick** the wait budget allows **33,000–66,000 ticks** → the 600 s cap does NOT bind at 18,000 ticks (needs 18,000×18 ms = 324 s). The sim's own 30,000-tick cap is the only ceiling. | measured 19.6/19.5 |
| validation | random seeds, unlimited attempts, one at a time | README.md:106 |
| evaluation | **3 runs averaged, preset seeds, ONE try** | README.md:110 |
| platform | Linux (macOS/ARM produces different trajectories — measured 4,472 vs 7,816 ticks, same seed+policy) | JOURNAL §6 |
| ordering | the official sim iterates entity **sets**, so its own episodes are not reproducible run-to-run; our copy sorts them for local determinism only | creature.py:128, environment.py:667 |

## LAYER 1 — WORLD (fixed by the organisers; we can only respond to it)

### 1.1 Episode construction (`simulation.py`, `core.py`)
| param | value |
|---|---|
| env size | 1600 × 1200 px, chunk_size 400, dt 0.1 |
| starting agents | 5, energy **150** (not 75), default traits |
| starting predators | 0 (they arrive later) |
| starting fruits | 32 (energy 20 each) |
| starting trees | 50, pre-grown `U(20,80)` age |
| obstacles | `width//20` = 80 rectangles, 30–100 px a side, + 4 boundary strips 30 px thick |

### 1.2 Agent body + metabolism
| param | value | line |
|---|---|---|
| size | 5 (contact radius) | agent.py |
| speed / sprint_speed | **10 / 20** (mutate to 20 / 40) | creature.py:25 |
| energy / max_energy | 75 (cohort 150) / **500** (mutate to 1000) | agent.py:17 |
| hearing / vision / cone | 50 / 200 / π·⅓ (±60°) (mutate to 100 / 400 / π·½) | agent.py:20-22 |
| max_age | `60 + U(0,60)` s = **600–1,200 ticks**, fixed at birth, NOT heritable | agent.py:27 |
| metabolism | `energy_drain_rate × dt` = **0.1/tick**, 1.0 in every biome | environment.py:639 |
| old age | above max_age: `energy -= 0.01 × age` per tick (age in seconds → at age 100 that is **1.0/tick**, 10x metabolism) | environment.py:646 |
| move cost | walking `0.05 × distance`; sprinting `speed×0.05 + (d−speed)×0.5` → 10 units = 0.5/tick, **20 units = 5.5/tick** (11x) | environment.py:501-518 |
| **sprint lockout** | `distance = speed` whenever `energy < max_energy/5` → below **100 energy** (base cap) an agent **physically cannot sprint** | environment.py:512 |
| turn cost | `min(π,|turn|)/(2π)` → 180° = 0.5 | environment.py:564 |
| biome move_penalty | multiplies the distance moved | environment.py:529 |
| obstacles | deflect in 10° steps until clear; bounds clamp to size | environment.py:541-557 |

### 1.3 Reproduction + heredity (THE UNEXPLOITED LAYER)
| param | value | line |
|---|---|---|
| spawn condition | `spawn_agent and agent.energy > 100` | environment.py:621 |
| spawn cost | parent `−100`, child starts at **75** | environment.py:623, creature.py:55 |
| spawn position | `U(10,30)` units from the parent (or the parent's own position if blocked) | environment.py:288-294 |
| **mutation rate** | **10 % per trait per birth**, factor `U(0.5, 1.5)` (±50 %) | environment.py:296-329 |
| heritable traits | `speed, sprint_speed, max_energy, hearing_radius, vision_radius, cone_angle` | environment.py:301-329 |
| caps | speed 20, sprint 40, max_energy 1000, hearing `chunk/4` = **100**, vision `chunk` = **400**, cone π/2 | environment.py:333-345 |
| **observability** | the agent state carries `speed, sprint_speed, hearing_radius, vision_angle, vision_range, max_energy` → **the policy can read the genotype of every visible agent AND its own** | DTOs.py:18-24 |
| what this means | nobody has to design the improvement — the sim already mutates it. **The only choice is WHO is allowed to breed.** A policy that spawns from the best-genome parent runs directed evolution inside its own fleet, at zero extra cost. | measured, see §3 |

### 1.4 Food chain
| item | value | line |
|---|---|---|
| fruit | energy 20 → grows 2/s to **60**, radius grows +0.2/s, **rots at age > 100 s** | fruit.py, environment.py:734-738 |
| fruit spawn | per tree with `age ≥ 20`: P = `dt × biome.fruit_spawn_rate` | environment.py:755 |
| tree spawn | P = `100/max(1, trees/2) × dt × 0.5^(time/300)` → **halves every 3,000 ticks** | environment.py:741-745 |
| tree life | dies after 50 s with P ∝ √U(0,1) (effectively ≤150 s); grows to radius 20 | environment.py:751 |
| biomes (10 seeds + 1 river, radius 20–100) | forest tree 1.0 / fruit **0.10** / move 1.0 · grassland 0.5 / **0.10** / 1.0 · swamp 0.9 / 0.08 / **0.5** · desert 0.1 / 0.05 / **0.8** · river 0 / **0** / 0.3 | biome.py:16-48 |
| energy_drain | **1.0 in every biome → there is no low-drain refuge** | biome.py:22 |

### 1.5 Predators (the wall)
| item | value | line |
|---|---|---|
| body | size 10, speed 11, sprint **15**, max_energy 200, hearing 60, vision 250 | predator.py:10 |
| start | energy 0, `resting = True` | predator.py:12 |
| resting | `energy += 3.0/tick` (30/s) and **cannot kill** (the interaction block is skipped); wakes above `max_energy/2` = 100 | environment.py:679-684 |
| awake | sprints at 15 (2.55/tick cost) → 100 energy buys ~40 ticks of chase, then it sleeps ~33 ticks | environment.py:705, 729 |
| targeting | chases the CLOSEST visible agent (hearing 60 any angle, vision 250 cone ±60°) | predator.py:31 |
| **charge rule** | charges only if the prey is **NOT facing it** (`|rel_dir| > π/2`) OR closer than `hearing×1.5 = 90`; otherwise it PIVOTS 45° off-target (radial closure `15·cos45 = 10.6/tick`) | predator.py:37-62 |
| kill | contact within `size_sum = 15`; predator gains the victim's energy capped at 200; **score −E/100** | environment.py:721-727 |
| **never removed** | there is no code path that deletes a predator | — |
| **spawn law** | P = `(1/max(1,N_pred)) × dt × time × 0.0001` per tick → deterministic drift `N ≈ 5e-6 × t²` (t in sim-seconds) → **~4 predators at t=900 s, ~16 at t=1800 s (18,000 ticks)**; measured 2 at t≈300 s, 3 at t≈600, 5 at t≈700 on seed 100 | environment.py:763 |

### 1.6 Sensing
Hearing sees any object (fruit/tree/agent/predator) within 50 units at any angle; vision sees within 200 units
inside ±30° unless blocked by an obstacle edge; walls are only *seen*, never heard. Agent→predator detection is
therefore **50 units at any angle or 200 inside the cone** — this is why the fleet is "blind" and why the
predator that kills you is usually one you never saw (journal §17).

## LAYER 2 — OUR CONTROLLER (`best_controller.py`, 63 knobs; live values in `best_controller/params.json`)

### steering
`fruit_weight` 1.148 · `predator_weight` 7.889 · `wall_weight` 0.0288 · `disperse_weight` 3.0 ·
`wander_weight` 0.0717 · `target_hyst` 0.5
### predator logic
`danger_dist` 167.87 · `flee_dist` 177.38 · `escape_dist` 252.26 · `fruit_risk_penalty` 0.677 ·
`concern_cone` 2.2 · `flee_speed_frac` 1.0
### forage
`walk_frac` 0.493 · `explore_frac` 0.2 · `blind_explore_frac` 0.290 · `tree_weight` 0.25 ·
`forage_nearest` 1.0 · `forage_speed` 1.0
### energy
`reserve_frac` 0.15 · `low_energy_frac` 0.35 · `boom_frac` 0.28
### reproduction
`repro_frac` 0.35 · `repro_frac_min` 0.22 · `repro_urgency` 1.0 · `repro_global_target` 12 ·
`repro_popcap` 6 · `spawn_cooldown` 120 · `repro_safe_radius` 253.5 · `repro_energy_abs` 0 ·
`repro_unlimited` 0 · `spawn_cap` 0
### relay / roles (dead axes, kept off)
`relay_age` 0 · `relay_energy_frac` 0.35 · `relay_cooldown` 700 · `role_by_age` 0 · `young_age` 25 ·
`old_age` 60 · `young_speed_frac` 1.0 · `old_speed_frac` 0.5
### state-memory search (off, measured losing)
`memory_search` 0 · `commit_len` 250 · `commit_speed_frac` 1.0 · `follow_agent_weight` 0 · `ars_speed_frac` 1.0
### predator-facing (off, refuted by observability)
`face_predator` 0 · `face_cone` 1.05 · `face_dist_max` 320 · `face_min_dist` 90
### retreat-while-facing evasion (LIVE)
`evade_mode` **1.0** · `evade_dist` **140** · `evade_disengage` **240** · `evade_speed_frac` **1.0** ·
`evade_energy_abs` **200**
### boom/famine phase (off, refuted)
`phase_mode` 0 · `famine_tick_lo` 10000 · `famine_tick_hi` 14000 · `famine_vis_guard` 0 ·
`famine_move_frac` 0.6 · `famine_blind_stop` 0 · `famine_pop_target` 3 · `famine_min_pop` 2 ·
`famine_bank_frac` 0.9 · **`famine_min_cap` 0.0 (the only knob that ever touched the genome — never enabled)**
### thin relay endgame (off, refuted)
`thin_relay` 0 · `thin_trig_tick` 6000 · `thin_trig_vis` 0.35 · `thin_trig_vis_off` 0.70 ·
`thin_target_pop` 2 · `thin_spawn_energy_abs` 250 · `thin_rescue_energy_abs` 160 ·
`thin_spawn_cooldown` 200 · `thin_move_frac` 0.7

**Knob-family verdict:** the search has had 63 knobs to play with and the live values are
the evolved survivors. Every *constraint* family (reserve, spawn margin, population cap, banking, phase
movement, memory search, tree chasing, larger fleet) was measured and LOST — five of them monotonically.
What has never been searched is the one axis the sim hands us for free: **which agents reproduce.**
