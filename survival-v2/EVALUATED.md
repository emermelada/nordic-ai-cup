# Evaluated artifact — Survival Simulator (graded run of 2026-09-20)

This file records **exactly** what the grader scored, so the code in this directory can be
hand-checked against the result.

## The graded result

| field | value |
|---|---|
| Evaluation score | **1405.2561882564453** |
| Service URL | `https://survival.zaitzev.com/predict` |
| Submitted | 2026-09-20T13:06:06Z |
| Started | 2026-09-20T14:44:14Z |
| Finished | 2026-09-20T15:12:44Z |
| Errors reported by the grader | `[]` (none) |
| Games | 3, run back to back and averaged (the competition's evaluation rule) |

Per-game scores as they arrived, from the server's own request trace:

    game 1   1319.9   (sim 1286.5 s, ended at 1 agent)
    game 2   1298.2   (sim 1358.4 s)
    game 3   1599.0   (sim 1534.3 s)
    average  1405.3   == the recorded 1405.2561882564453

## The exact files that served those three games

| file | sha256 | lines |
|---|---|---|
| `hive.py` | `56489acfff2cba36b8e1d871dc9bb9b0e9864818e62a66291e9a1b389b553c2e` | 1589 |
| `server.py` | `68fac18eec77223eb238b1aef807aa45b861a812423b76d51cd3ce987e581e12` | 198 |

How they were served (UpCloud Helsinki VM, 32 vCPU, Ubuntu, systemd unit `surv.service`):

    WorkingDirectory=/opt/surv
    ExecStart=/opt/nacv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 9052 --no-access-log
    Environment=SURV_LOG=/opt/surv/server_log.jsonl

Dependencies: `numpy`, `scipy` (imported lazily, only on the pathfinding branch), `fastapi`, `uvicorn`.
Reverse-proxied by Caddy (`survival.zaitzev.com -> 127.0.0.1:9052`).

## Relation to the rest of this directory

`hive.py` here is the controller described in `HANDOFF.md` / `RESULTS.md`, **plus** the
stray-payload guard from `evidence/FINDING_VALIDATION_PROBE_GUARD_2026-09-20.md`.

The guard is behaviourally inert on clean runs. It only reacts when a payload arrives with a
*backwards* `sim_time` — which is what the organisers' connectivity probe does, sometimes in the
middle of a live game. The previous rule reset the whole game on any backward jump, which had
already wiped a live attempt once. The guard instead suspends the game, serves the probe from a
fresh state, and restores the game if the real stream continues.

Proof that it does not change the model's results: the guarded file and the unguarded file
(`7f3467cfa5cf26dd`, 1526 lines) were run on the same 14 seeds at the full 30,000-tick horizon and
produced **bit-identical scores and tick counts, 14/14** (e.g. seed 1: 1557.030702 both; seed 164:
656.686564 both). During the graded run itself the guard counted 2 stray payloads and 0 errors, so
all three games ran on the intended policy.

## Measured behaviour of this policy (400 seeds, full 30,000-tick horizon)

    per-run    mean 1473   median 1477   sd 338   p10 1033   p90 1888   p95 2011   max 2709.6
    mean of 3  p5 1151    p50 1475      p95 1792      P(mean-of-3 < 1200) = 8.1%
    vs the previous deployed controller (200 paired seeds): +300.8 +- 30.8, t = +9.76, W/L 150/50
    400-seed pairing: +302.6 +- 21.1 (t = +14.33); truncation-free cross-check +242.1 +- 21.8

Determinism: the controller is exactly reproducible for a given map — 4 repeats x 8 seeds gave a
within-seed spread of 0.0000 (whole record byte-identical), against a between-seed sd of 481. So
all run-to-run variance is map variance, not controller noise. Details:
`evidence/NEW_CONTROLLER_TEST_2026-09-20.md`.

## Score ceiling (what is and is not reachable)

`score = ticks/10 + fruit_energy/1000`.

* The tick term is capped by the simulator: `score += dt` per tick and `sim_time` stops at 3000 s,
  so it contributes at most **3000 points**.
* The fruit term is not capped by the formula, but it is capped by the world: `Fruit.energy` tops out
  at **60** (`src/elements/fruit.py`), and fruit spawns from trees at ~0.1 fruit/s per tree with the
  tree population settling near ~39 — a total fruit supply of roughly 0.7 M energy over a full game,
  i.e. **about 700 points maximum**.
* Hence the maximum reachable score is roughly **3,700**, and the realistic maximum is ~3,200. Best
  ever observed with this policy is **2,709.6**; the best recorded validation across all builds is
  **1,815.55** (`evidence/GOLD_STANDARD_1815_2026-09-20.md`, which also explains why that run cannot
  be replayed: the grader's seeds are not in the payload).

## Honest limits

* The three graded games were mid-pack draws (1,320 / 1,298 / 1,599 against a per-run median of
  1,477); the average is what the competition scores, so map luck matters as much as the model.
* Two tail-targeted interventions were built, measured and **rejected**: bounding the localisation
  repair loop (target seeds +578, but representative seeds -30, net ~-7) and loosening the breeding
  brake `breed_colony_e` 100 -> 85 (targets +360, representative seeds -173, net ~-141). This
  controller sits at a local optimum where the tail and the middle are coupled, and both changes
  were left out of what was deployed.