# VALIDATION QUEUE — candidates ready for the user to validate / deploy

The user owns validation and deployment. Nothing here is deployed by an agent. Every entry carries the
paired evidence measured on x86-64 Linux at the official 18,000-tick horizon, and the deploy command.

---

## 1. E3 — retreat-while-facing evasion (READY, awaiting decision)

**What it is.** A config-only change on top of H1, plus one gated code addition: when a predator closes
inside 140 units, the agent moves directly away from it while turning to keep it in front (which stops
the predator charging and makes it pivot instead), at full sprint speed, and only when its own energy is
above 200 (so it can sustain the sprint — the sim silently forbids sprinting below 100).

```
evade_mode        1.0
evade_dist        140.0   (trigger; 220 was measured worse - concedes income too early)
evade_disengage   240.0   (hysteresis)
evade_speed_frac  1.0     (0.8 was measured worse: 16 units is not enough separation)
evade_energy_abs  200.0   (only agents that can afford a sustained sprint retreat)
```

**Why it is credible.** The mechanism is measured, not inferred: while facing a predator the gap OPENS at
0.477-1.583 units/tick versus ~0 when turned away (w_lockout_diag.py, 3 seeds, official horizon). And the
two losing arms fail in the predicted direction, which is what a real mechanism looks like.

**Paired evidence (H1 reference, identical seeds, horizon 18,000):**

| seed set | E3 | H1 | delta | seeds won |
|---|---|---|---|---|
| 1100-1109 | 8388.1 | 7468.2 | **+919.9 (+12.3%)** | 8/10 |
| 1110-1119 | 8260.1 | 7730.3 | **+529.8 (+6.9%)** | 5/10 |
| pooled (20) | 8324.1 | 7599.3 | **+724.9 (+9.5%)** | 13/20 |

Fruit eaten also rose (1361/1180 vs 1211/1169), so the gain is not from passivity.
Against the pre-registered bar: mean delta PASSES (>500), seeds-won MISSES (13/20 vs the 15/20 required).

**Deploy steps (once approved).**
1. `best_controller.py` gains one gated block (inert unless `evade_mode > 0`) — already written and
   unit-verified: OFF reproduces the incumbent's episode byte-for-byte, ON fires only in the designed case.
2. Install E3's params as `best_controller/params.json`, run `tools/check_controller.py --update` to
   re-record the sha256 (the Docker build refuses a mismatched ledger).
3. Rebuild and restart the endpoint (~1-2 min outage) — only when no validation is queued or running.
4. Rollback is one command: `git checkout HEAD -- best_controller.py best_controller/params.json` + rebuild.

**Known risk.** The evade branch adds a behaviour the tuned search never saw, so its interaction with the
other knobs (especially `flee_dist`/`escape_dist`, which also trigger on predators) is only tested at the
E3 settings. The two losing arms bound how bad a mis-set can be: -6 to -8%.

---

## Status of the other streams

- **W3 (access/income, matched price)** — sweep running on the VPS (nac-A1/nac-A2, seeds 1200-1219).
- **W2 (state-triggered thin relay)** — in development, code must stay off-by-default.
- **W1 (residual RL on frozen H1)** — training running, mid-game curriculum; confirmation will be on Linux.
- **W5 (curriculum spec)** — research/design only.


## Validation log (E3)

| # | when (UTC) | score | ticks | ms/tick | errors |
|---|---|---|---|---|---|
| 1 | 21:00:39-21:03:04 | 798.42 | 7,984 | 18.2 | none |

H1 on the same endpoint earlier: 768.68 (19.4 ms/tick), 918.33 (27.6 ms/tick) - mean 843.5.
Verdict so far: INCONCLUSIVE (1 sample). Repeating is cheap; if the leaderboard keeps the best attempt,
extra samples are free upside.
