# FINDING — validation was being lowered by the platform's own probe, not by the controller

2026-09-20, ~11:30 CEST. Question asked: *"validation isn't producing low scores because of the controller
being bad — it can be because of the file problem you surfaced. How to solve it, or at least validate
without a lowering in score?"*

Answer: **partly yes — there is a real, platform-only mechanism that throws score away, it is now fixed on
the graded endpoint, and the fix is verifiably free on a clean request stream.** The two lowest platform
draws so far were NOT caused by it (see §5), so the controller is still the main lever — but the mechanism
is real, was measurable, and would have cost us points in the graded 3-run mean.

---

## 1. What the deployed server's own trace shows (58,453 rows, /opt/surv/server_trace.csv)

Segmenting the trace by `sim_time` gives the per-attempt games. Two anomalies appear that no local harness
ever produces:

**(a) A lone synthetic payload injected into the stream.** 9 rows, always the same shape:

```
1789897199.174,0.0,1,123.4,running,1.20,1,edge|predator|tree
                ^sim  ^1 agent  ^score  ^game_status  ^decide ms  ^1 action
```

`sim_time 0.0`, one agent, `score 123.4`, `game_status "running"` (every real game row is `"ok"`),
observations `edge|predator|tree`. The organisers' README documents exactly this: *"the testing endpoint …
the `sim_time` and `n_agents` values are missing in this step"* — their connectivity check. It arrives
**before most attempts** (08:06:38, 08:38:38, 08:52:55, 09:42:41, 10:01:54 UTC) and, at least once, **in
the middle of a live graded game**: 09:39:59, when the in-flight attempt (final score 904.0) was at
`sim_time 784.7` and 22 agents.

**(b) A foreign client's whole game interleaved with ours.** 07:42:34-35: our 42-agent game alternating
with a 5-agent game from `127.0.0.1` — 1,336 backward `sim_time` jumps inside the gold run (that is the
known contamination of the gold trace's first 139 s).

Backward-jump counts per attempt, measured from the trace:

| attempt (score) | requests | backward sim_time jumps | mid-game probe |
|---|---|---|---|
| gold 1815.5 | 22,091 | 1,336 | (foreign client, 07:42) |
| 1356.4 | 12,975 | 0 | no |
| 282.6 | 2,610 | 0 | no |
| 258.2 | 2,410 | 0 | no |
| 904.0 | 8,676 | 2 | **yes, at sim_time 784.7** |
| 422.8 | 4,095 | 1 | no (pre-flight only) |

## 2. Why that lowered the score

`hive.decide()` (both the deployed controller and every earlier hive) had exactly one boundary rule:

```python
if t < self.last_t - 1e-9:
    self.reset()          # wipes rng, tick, mem, maps, next_frame, best_fit, stats
```

So **every** backward jump wiped a live game's entire internal state — the world map, every agent's memory
(frame, position estimate, current food target), the genome-selection ratchet and the RNG stream. In
addition, because the probe's tick was then processed from that fresh state, the probe's *synthetic* agent
(id 0, fake tree/edge/predator observations) was planted in the state that the real game inherited on its
very first tick — i.e. **every** attempt began with agent 0 carrying a bogus local map. The boom phase is
what determines a run (winners charge a bigger energy battery in t=0–300 s), which is precisely where the
probe lands.

## 3. The fix (deployed, 11:26:35 UTC)

A backward jump now only **suspends** the game:

* the payload is served from a fresh state as before (so a genuinely new game still starts clean),
* the live game's state is snapshotted and **restored exactly** when the stream resumes
  (`t >= that game's last sim_time`) — i.e. the stray payload cannot destroy it,
* if the stream instead starts a fresh game, the boundary payload's agent ids are dropped from memory, so
  the probe's synthetic agent cannot contaminate the new game,
* a `probe_guard` param (default 1.0) switches back to the old rule (`0.0`) for A/B testing,
* counters `strays / strays_restored / games_seen` are exported on `GET /` and in the 1000-request log
  line, so a validation can be proven clean afterwards.

Code: `experiments/hive_pf_guard.py` (= the controller that was live, plus the guard; nothing else
changed), server: `experiments/gold1815/server_next.py`, deployer: `tools/ops/deploy_probe_guard.sh`.

**Deployed state (verified live, not assumed):**

```
hive.py   56489acfff2cba36b8e1d871dc9bb9b0e9864818e62a66291e9a1b389b553c2e   (was 7f3467cf...)
server.py 68fac18eec77223eb238b1aef807aa45b861a812423b76d51cd3ce987e581e12   (was d94e68d0...)
GET https://survival.zaitzev.com/
  {"message":"Agent endpoint running!","requests":0,"errors":0,
   "controller_sha256":"56489acff...","params":71,"strays":0,"strays_restored":0,"games_seen":1}
systemd surv.service active, NRestarts=0, public 200 in 29 ms
rollback (one command, ~5 s):
  bash survival-simulator/tools/ops/deploy_probe_guard.sh --rollback
```

## 4. Verification (the fix is free where it must be, and does something where it must)

**(i) Logic level** — `experiments/hive_guard_test.py` (pure, no simulator):

```
1.  clean stream identical to old rule        : True (120 decisions)
1b. probe_guard=0 == old rule on stray stream : True
2.  mid-game probe: strays=1 restores=1       tick 300->301   mem 3->3   (state kept)
2b. old controller, same probe:               tick 300->1     mem 3->1   (state WIPED)
3.  new game after a probe: mem=[0,1,2]       (the probe's agent was dropped)
4.  game-to-game transition identical         : True
```

**(ii) Simulator level** — `experiments/probe_check.py --light`, 6 seeds, real `SimulationCore`, strays
injected at ticks 600/1200/2400 of a live game:

```
stray new == clean new         : 6/6
score change from strays, OLD  : mean -65.4   min -200.0   max +2.9   (n_worse=5)
score change from strays, NEW  : mean  +0.0   min   +0.0   max +0.0   (n_worse=0)
```

So: the mechanism costs the previously-deployed controller ~65 score points on average and up to 200 in
these 6 seeds, and the guarded controller loses **exactly nothing** (bit-identical outcome to an
interference-free run). Caveat, stated plainly: n=6 seeds at a 6,000-tick horizon is a mechanism test, not
a calibrated effect size — the per-seed variance here is 2,600–3,900 ticks, so the *average* penalty is
only suggestive; the exactness of the NEW arm (0 difference in 6/6) is what carries the argument.

**(iii) On the target host** — the staged server+hive were imported on the box and driven with a real tick
plus a real probe before the swap: `SMOKE OK: strays=1 restores=1 games=2 controller=56489acf…`.

## 5. What this does NOT explain (so it is not oversold)

* The two lowest draws (282.6 and 258.2) had **zero** backward jumps: those were genuine low draws of the
  controller, not probe damage. The controller remains the main lever.
* The 904.0 attempt was wiped once at `sim_time 784.7` and still finished as the second-best draw; one
  wipe is not automatically fatal, it is a gamble we no longer take.
* A probe whose payload omits `sim_time` entirely (the organisers' older DTO default) would not be
  detected as a boundary — it is treated as a normal tick, exactly as before. Not covered, deliberately:
  changing behaviour for a pattern that has not been observed is a bigger risk than the residual case.

## 6. Operational note found while doing this

`surv.service` was restarted at **11:01:17 UTC** while an attempt queued at **10:51:45** was in flight
(the trace stops at `sim_time 566.5`, request 60,000, and the log's last line is 10:49:34). That restart
was not made by this session; whoever deploys the pathfinding controller should note that a queued attempt
was lost to it. Restart rules unchanged: **never restart the container/service while a validation is in
flight**, and check `GET /` (`requests`, `games_seen`) plus the trace mtime first.
