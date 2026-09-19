# Survival Simulator v2: plan

Fresh approach on branch `survival-v2` (worktree `C:\Users\franc\nordic-ai-cup-survival`), based on 7cd494e,
before any survival-simulator policy code was merged. None of the earlier survival code is used.
Official simulator vendored unchanged in `survival-v2/official/` (amboltio/Nordic-AI-Cup-2026 @ 2b39323).

Deadline: final evaluation before Sun 20 Sep 16:00 CEST. It runs one attempt of 3 games on preset seeds and scores the mean.

## 1. What wins

Current scoreboard (19 Sep): #1 1743, #2 1661, #3 1403, #4 1377, #5 1312. Nobody survives the full game.

Score (from `environment.py`, not the README):

| Term | Value | Scale |
|---|---|---|
| Survival | +0.1 per tick while any agent lives | max 3000 |
| Fruit | +fruit_energy/1000 per fruit eaten (20 at spawn → 60 at 20 s; rots at 50 s) | whole game ≈ 4000 fruit → max ≈ +240 |
| Predation | −agent_energy/100 per agent eaten | −0.75…−10 per kill |

Surviving all 3000 s is worth roughly +1250 over the current leader. The fruit bonus is small next to that,
so every design choice is judged by P(colony alive at t=3000) first, then by fruit and kills.

## 2. Measured facts (seeds 1, 2, idle immortal agents, official sim)

| t (s) | trees | productive trees | standing fruit | predators |
|---|---|---|---|---|
| 60 | 94–112 | 75–84 | 174–214 | 0 |
| 300 | 50–62 | 28–38 | 159–165 | 2–4 |
| 600 | 32–42 | 23–30 | 94–155 | 4–5 |
| 1200 | 20–25 | 15 | 43–51 | 7–9 |
| 1800 | 8 | 7 | 28–37 | 15 |
| 2400 | 4–7 | 3–4 | 12–18 | 19–21 |
| 2940 | 1–4 | 0–2 | 2–3 | 24–25 |

- Food supply ≈ 4 fruit/s early, ≈ 1/s at 1200, ≈ 0.3/s at 2400, ≈ 0.1/s at 2900. Late game needs a small colony (3–6) that eats almost every fruit.
- Agents die of old age: after a hidden max_age of 60–120 s the drain is 0.01·age per tick, so no agent lives past about 180 s. The colony has to keep reproducing.
- A child starts with 75 energy and the parent pays 100. An idle child lives 75 s with no food. **Endgame insurance:** any agent above 100 energy at t ≥ 2926 can spawn children that live to t=3000 without eating.
- Predators: speed 11, sprint 15, hearing 60, vision 250 in a 60° cone blocked by walls. Each one chases only the closest agent it perceives. It charges if the agent faces away or is within 90 px; otherwise it circles at sprint. It rests at 0 energy and wakes at 100. Its pose is fully observable to us (rel_dir gives its heading).
- **Traits mutate and we choose who breeds.** Each trait has a 10% chance per birth to be scaled by U(0.5, 1.5). Caps: speed 20, sprint 40, max_energy 1000, hearing 100, vision 400, cone 90°. Traits cost no energy. An agent with walk speed > 15 cannot be caught by a predator once it is detected (walking costs 0.05/px regardless of speed). About 20% of births are strictly better than the parent.
- `move_direction` is relative to heading. Dead reckoning is exact in each agent's own frame: heading changes only by our turns, and the biome speed penalty is observable. Only obstacle collisions are unknown.
- The official sim runs at 7.4–17.5 ms/tick, which is 4–9 min per game. Too slow for tuning, so we need a fast, RNG-faithful copy.
- Same-seed runs differ only because the sim hands observations in `set` order. A policy that sorts its observations should be deterministic per seed, which allows paired A/B tests.

## 3. Platform constraints

- The grader waits at most 10 s per step and **600 s accumulated per game**, which is 20 ms/tick including network.
  A public competitor (JBrO910) died at t=666 s serving through a Cloudflare tunnel (about 90 ms/tick).
- **The grader runs in Hetzner Helsinki (46.62.240.126).** Serve from a Hetzner `hel1` VM with plain `http://IP:port/predict` (about 1 ms RTT).
- The platform's connection-test sample has no `sim_time` or `n_agents` and uses lowercase observation types, so the parser must be tolerant.
- Evaluation = 3 games back to back from one server process. Controller state must reset per game (detect `sim_time` decreasing or new agent ids).
- Payload grows with population × observations. Budget: our processing < 3 ms/tick, and total per-tick wait well under 10 ms.

## 4. Research: approaches weighed

| Approach | Benefit | Drawback that would slow us | Verdict |
|---|---|---|---|
| Deep RL (PPO/MAPPO, shared policy) | Could discover novel behaviour | Slow Python sim, 30k-tick horizon, sparse survival reward, variable agent count. JBrO910 spent a day of PPO for +81 s and then plateaued | **No** |
| Neuroevolution of a policy net | No gradient needed | Hundreds of weights with 330 s/game noise needs thousands of games | **No** |
| Hand-built hierarchical controller (utility tiers) | µs inference, interpretable, robust, all exploits usable | Ceiling set by design quality | **Yes, the core** |
| CMA-ES on ~20–40 controller knobs | Best black-box optimiser at this dimension; parallel | Noisy ranking needs common random numbers, re-evaluation, and many cores | **Yes**, with paired seeds and racing |
| Directed evolution of agent traits (in-game truncation selection) | Structural fix for the predator ceiling that stops every public team at 1200–1750 s | Needs birth throughput early (food, population) | **Yes, the main lever** |
| Exact predator-model lookahead (short MPC) | The predator policy is deterministic and its pose is observable | CPU per endangered agent | **Yes**, only for agents under threat |
| Shared hivemind map (SLAM-lite: exact dead reckoning, parent→child frame registration, frame merge on mutual sighting, edge-anchored correction) | Predator early warning, fruit dating (3× energy per fruit), tree memory, late-game search | Complexity; JBrO910's drifting map hurt relocation | **Yes**, with exact odometry and ground-truth tests |
| Lévy-walk search for sparse targets | Optimal without memory | We have memory | Frontier/stale-cell search first; Lévy only as a fallback |
| Tunnels (cloudflared) | Easy | 90 ms/tick, dies at the 600 s budget | **No.** Serve from hel1 over plain HTTP |

## 5. Architecture

```
survival-v2/
  official/          vendored official simulator (unchanged)
  fastsim.py         drop-in subclass of the official Environment: no rendering, cached static geometry,
                     vectorised visibility, identical RNG call order (verified against official/)
  hive/              the controller
    parse.py         tolerant raw-JSON → arrays (case-insensitive types, missing fields)
    frames.py        per-agent pose; family frames (root agents, children registered from parent's sighting);
                     frame merge on cross sighting; edge/tree landmark correction
    worldmap.py      edges/obstacles, trees, fruit (spawn time → ripeness), predator tracks, biome grid
    genome.py        trait fitness, elite set, breeding schedule, old-age detection
    tactics.py       evasion (exact predator model lookahead), foraging assignment, exploration, facing
    hive.py          Hive.decide(step) -> actions; params dict (tunable)
  bench.py           parallel multi-seed evaluation (fast sim), metrics and paired stats
  tune.py            CMA-ES / racing over params with common random numbers
  server.py          ASGI + orjson endpoint (/predict, /, /api), timing logs, per-game reset
```

## 6. Step-by-step

Each step ends with a measurement logged in `RESULTS.md`. Nothing is kept unless it measures better.

### Step 1: Fast faithful simulator (target: ≤ 3 ms/tick with ~30 agents and ~25 predators)
1. Profile the official sim (done: `np.array` for edges, `compute_visibility`, shapely point checks, grid rebuilds).
2. Build `fastsim.py` by subclassing: skip pygame surfaces but consume the RNG exactly as `_render_biome_surface` does. Cache per-chunk edge lists (same set-build order, so the same array order). Use `shapely.contains_xy` for the vision polygon (same predicate). Do incremental grid updates. Keep the same list-with-removal loop quirks and rng call order.
3. Verify: same seed plus the same order-invariant policy gives identical world, score, positions and RNG state on official vs fast for 3 seeds × 3000 ticks, and matching statistics over full games.

### Step 2: Harness and server skeleton
1. `bench.py`: multiprocessing over seeds. Per game: score, survival time, fruit score, kills (count and energy), births, population curve, trait trajectory, decide ms/tick, and JSON bytes/tick.
2. `server.py`: orjson, never raises (falls back to safe no-op actions), per-game reset, logs client IP and wait/latency stats.
3. End-to-end check with the official `simulation_server.py` against `server.py`, timing `requests.post` around `.dict()` because that is most likely what the grader's 600 s budget counts.

### Step 3: Hive v1 (strong baseline, all pillars at low sophistication)
1. Parsing, per-agent exact odometry, family frames, frame merge, and a tracking-error test against sim ground truth (target: median error < 1 px, and recovery after collisions).
2. World map: fruit with spawn time (watched-spot dating), trees, predator tracks with heading and rest detection, edges, biome grid.
3. Behaviour tiers per tick: (a) threat evasion; (b) old-age energy dump into children; (c) breeding by elite genome; (d) fruit assignment (greedy on travel time + wait-until-ripe, one claimant per fruit); (e) tree camping, facing so tree and fruit stay in view or in hearing range; (f) exploration of stale, forest-biased cells.
4. Population targets by phase: many agents early (breeding and fruit), shrinking to what food supports late. Endgame insurance spawns from t ≈ 2926.
5. Gate: fast-sim bench on 48 seeds. Report mean/min survival and the P(survive 3000) histogram.

### Step 4: Make the breeding program work (the main lever)
1. Measure trait trajectories (speed, hearing, vision, cone, sprint, max_energy) of the elite line vs time.
2. Tune fitness weights, breeding energy thresholds, the elite margin and culling so that walk speed > 15.5 and hearing ≈ 100 are reached by t ≈ 600–900 on most seeds.
3. Switch evasion from "sprint and face" to "out-walk" once traits allow it.

### Step 5: Hardening the late game (t > 1800)
1. Death post-mortems on the fast sim (cause, traits, energy, predator count, biome, walls).
2. Fixes in order of measured impact: cornering (escape planning with walls), swamp/river avoidance under threat, sentinel coverage for new trees, lineage continuity and endgame insurance.

### Step 6: Tuning (overnight)
1. CMA-ES (pycma) over about 20–40 knobs. Each generation uses the same 24–48 seeds for every candidate (paired), rotated between generations, with a held-out seed set.
2. Objective: mean score, with extinctions counted in full. Accept the tuned set only if the held-out mean improves beyond 2 SE.
3. Compute: all local cores now. A hel1 box is better for both serving and tuning (see Asks).

### Step 7: Serve, validate, freeze
1. Deploy to hel1 (uvicorn + uvloop + httptools, 1 worker, plain HTTP). Smoke-test with the official client from the box.
2. Run a platform validation early (after Step 3) to confirm the latency budget and the full 3000 s run, then after each major gain.
3. Freeze by Sun 12:00 CEST: tag `survival-v2-final`, run 2 validations on the frozen build, then submit the evaluation before 16:00 and keep the box up through all 3 games.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Latency budget exceeded (600 s/game) | hel1 hosting, orjson, < 3 ms decide, population cap tied to measured payload cost, server-side timing logs |
| Map/odometry bugs silently hurting play | Ground-truth tracking-error tests in the fast sim; every map feature measured by ablation |
| Tuning overfits seeds | Paired seeds rotated per generation; held-out set; accept only > 2 SE gains |
| Breeding too slow on some seeds | Evasion does not depend on it (face-and-back-off and sprint rules first); breeding is a multiplier |
| One-shot evaluation fails on server issues | Frozen tag, rehearsal validations, restart-safe server, keep-alive, box monitored during the attempt |
