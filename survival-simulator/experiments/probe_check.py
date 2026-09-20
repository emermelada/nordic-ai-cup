#!/usr/bin/env python3
"""probe_check.py - does a stray payload injected into a LIVE game cost score?

WHY. The graded platform's request stream is not one clean game at a time. Reading the deployed
server's own trace (server_trace.csv, 58,453 rows) shows two things the local harness never produces:

  1. a lone synthetic payload -- sim_time 0.0, ONE agent, score 123.4, observations edge|predator|tree
     -- arriving BEFORE most attempts (the organisers' README documents their connectivity probe) and,
     at least once, in the MIDDLE of a live graded game (09:39:59 UTC, attempt scoring 904.0, previous
     request sim_time 784.7);
  2. a foreign client's whole game interleaved with ours (07:42:34-35, our 42-agent game alternating
     with a 5-agent game from 127.0.0.1: 1,336 backward sim_time jumps inside the gold run).

hive's old rule was `if t < last_t - 1e-9: self.reset()`, so EVERY one of those backward jumps wiped a
live game's entire state -- world map, per-agent memory, genome ratchet, RNG. That is a platform-only
failure mode: it cannot appear in any local measurement, which is exactly why platform draws have been
sitting under local ones. It is also the answer to "is validation scoring low because the controller is
bad?" -- no: it is scoring low partly because the controller throws its game away when the platform
pokes it.

WHAT THIS MEASURES. For each seed, with the real simulator:
  clean      : one game, no interference                              (the reference)
  stray      : the same game with the probe injected at three ticks   (what the platform does)
  pre-probe  : one probe payload BEFORE the first tick, then the game  (the platform's pre-flight check)
for BOTH controllers -- the deployed one (hive_old.py, sha256 829e4147...) and the hardened one
(hive_next.py). Output per seed: (T, score) for each cell plus the internal-state diagnostics.

CORRECTNESS CONTRACT
  * new clean  == old clean            -> the hardening changes NOTHING on a clean stream (no score risk)
  * new stray  == new clean            -> a stray costs the hardened controller nothing
  * new pre    == new clean            -> the pre-flight probe no longer contaminates the game
  * old stray  vs  old clean           -> the size of the loss the platform has been paying

  PYTHONHASHSEED=0 python3 probe_check.py --seeds 1000-1011 --horizon 18000 --workers 8 --out /tmp/pc
"""
import argparse
import copy
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from multiprocessing import get_context  # noqa: E402

DT = 0.1

# The platform's connectivity probe, as the deployed trace describes it: sim_time 0.0, one agent,
# score 123.4, observations edge|predator|tree. Exact per-field values are not recoverable from the
# trace; the mechanism under test (a backward sim_time jump) does not depend on them.
PROBE = {
    "game_status": "ok", "score": 123.4, "sim_time": 0.0, "n_agents": 1,
    "agent_status": [{
        "agent_id": 0, "energy": 100.0, "biome": "forest", "age": 1.0, "speed": 10.0,
        "sprint_speed": 20.0, "hearing_radius": 60.0, "vision_angle": 1.0, "vision_range": 300.0,
        "max_energy": 500.0,
        "observations": [{"type": "edge", "distance": 100.0, "angle": 0.0},
                         {"type": "predator", "distance": 200.0, "angle": 0.5},
                         {"type": "tree", "distance": 50.0, "angle": -0.3}],
    }],
}


def play(hive, seed, horizon, strays=(), pre_probe=False):
    """One game with a fresh simulator. strays = ticks after which the probe payload is injected."""
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    random.seed(seed)
    np.random.seed(seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    env = core.env
    if pre_probe:
        hive.decide(copy.deepcopy(PROBE))       # the pre-flight connectivity check, before tick 1
    T = 0
    for i in range(horizon):
        live = list(env.agents)
        if not live:
            break
        states = [s for s in (env.get_agent_state(a.agent_id) for a in live) if s]
        if not states:
            break
        T = i + 1
        acts = hive.decide({"agent_status": states, "sim_time": i * DT, "n_agents": len(states)})
        byid = {s["agent_id"]: s for s in states}
        acted = []
        for d in (acts or []):
            aid = d.get("agent_id")
            if aid not in byid:
                continue
            acted.append((aid, ActionRequest(agent_id=aid,
                                             move_distance=float(d.get("move_distance", 0.0) or 0.0),
                                             move_direction=float(d.get("move_direction", 0.0) or 0.0),
                                             turn_angle=float(d.get("turn_angle", 0.0) or 0.0),
                                             spawn_agent=bool(d.get("spawn_agent", False)))))
        core.step(acted)
        if i in strays:
            hive.decide(copy.deepcopy(PROBE))   # injected mid-game, exactly as the platform did
    return T, float(env.score)


def run_seed(job):
    seed, horizon, light = job
    from hive_old import Hive as Old
    from hive_next import Hive as New

    strays = (600, 1200, 2400)
    hn = New(seed=0)
    nc = play(New(seed=0), seed, horizon)
    ns = play(hn, seed, horizon, strays=strays)
    os_ = play(Old(seed=0), seed, horizon, strays=strays)
    r = {"seed": seed, "new_clean": nc, "new_stray": ns, "old_stray": os_,
         "strays_seen": hn.counts["strays"], "restores": hn.counts["restores"],
         "stray_free": ns == nc,
         "old_clean": play(Old(seed=0), seed, horizon) if not light else None}
    r["same_clean"] = (r["old_clean"] == nc) if not light else None
    if not light:
        np_ = play(New(seed=0), seed, horizon, pre_probe=True)
        hp = New(seed=0)
        hp.decide(copy.deepcopy(PROBE))
        n2 = play(hp, seed + 5000, horizon)
        r.update({"new_pre": np_, "new_next_game_after_probe": n2,
                  "fresh_next_game": play(New(seed=0), seed + 5000, horizon),
                  "pre_free": np_ == nc})
        r["next_ok"] = r["new_next_game_after_probe"] == r["fresh_next_game"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1011", help="a-b inclusive")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--light", action="store_true", help="only the cells the guard can change")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    ctx = get_context("spawn")
    rows = []
    with open(os.path.join(args.out, "raw.jsonl"), "a") as f, ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_seed, [(s, args.horizon, args.light) for s in seeds]):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("seed %5d  clean new %.1f | stray old %.1f / new %.1f %s | strays=%s restores=%s%s"
                  % (r["seed"], r["new_clean"][1], r["old_stray"][1], r["new_stray"][1],
                     "OK" if r["stray_free"] else "DIFF", r["strays_seen"], r["restores"],
                     "" if r["same_clean"] is None else (" | clean old==new %s" % r["same_clean"])), flush=True)
    n = len(rows)
    d_old = [r["old_stray"][1] - r["new_clean"][1] for r in rows]
    d_new = [r["new_stray"][1] - r["new_clean"][1] for r in rows]
    print("\n=== %d seeds, horizon %d, %.0f s ===" % (n, args.horizon, time.time() - t0))
    if rows[0]["same_clean"] is not None:
        print("clean old == clean new         : %d/%d" % (sum(1 for r in rows if r["same_clean"]), n))
        print("pre-probe new == clean new     : %d/%d" % (sum(1 for r in rows if r["pre_free"]), n))
        print("game after probe == fresh game : %d/%d" % (sum(1 for r in rows if r["next_ok"]), n))
    print("stray new == clean new         : %d/%d" % (sum(1 for r in rows if r["stray_free"]), n))
    print("score change from strays, OLD  : mean %+.1f  min %+.1f  max %+.1f  (n_worse=%d)"
          % (sum(d_old) / n, min(d_old), max(d_old), sum(1 for x in d_old if x < -0.01)))
    print("score change from strays, NEW  : mean %+.1f  min %+.1f  max %+.1f  (n_worse=%d)"
          % (sum(d_new) / n, min(d_new), max(d_new), sum(1 for x in d_new if x < -0.01)))


if __name__ == "__main__":
    main()