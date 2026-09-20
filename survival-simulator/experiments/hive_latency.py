#!/usr/bin/env python3
"""hive_latency.py - measure the survival-v2 controller's per-tick decision latency in the real sim.

Deployment criterion #5 is "acceptable inference latency", and the competition budget is ~600 s of
total grader wait per game (measured: the public path is ~5.6 ms/tick, and p99 must stay low).
This runs ONE episode in the real simulator, timing hive.decide() on every tick, and reports
mean/p95/p99/max plus the agent count and payload size at that moment - the numbers a deploy
decision needs, measured rather than assumed.

    python3 hive_latency.py --seed 300560 --horizon 18000
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=300560)
    ap.add_argument("--horizon", type=int, default=18000)
    a = ap.parse_args()

    from hive_v2 import Hive
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    random.seed(a.seed)
    np.random.seed(a.seed)
    hive = Hive(seed=a.seed)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=a.seed)
    lat = []
    agents = []
    i = 0
    for i in range(a.horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
        if not states:
            break
        step = {"agent_status": states, "sim_time": i / 10.0, "n_agents": len(states)}
        t0 = time.perf_counter()
        acts = hive.decide(step)
        lat.append((time.perf_counter() - t0) * 1000.0)
        agents.append(len(states))
        byid = {s["agent_id"]: s for s in states}
        out = []
        for d in (acts or []):
            aid = d.get("agent_id")
            if aid not in byid:
                continue
            out.append((aid, ActionRequest(agent_id=aid,
                                           move_distance=float(d.get("move_distance", 0.0) or 0.0),
                                           move_direction=float(d.get("move_direction", 0.0) or 0.0),
                                           turn_angle=float(d.get("turn_angle", 0.0) or 0.0),
                                           spawn_agent=bool(d.get("spawn_agent", False)))))
        core.step(out)

    lat.sort()
    def pc(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))]
    res = {"seed": a.seed, "ticks": i + 1, "mean_agents": st.mean(agents) if agents else 0,
           "max_agents": max(agents) if agents else 0,
           "decide_ms_mean": st.mean(lat), "decide_ms_p95": pc(0.95), "decide_ms_p99": pc(0.99),
           "decide_ms_max": lat[-1],
           "projected_wait_s_per_game": st.mean(lat) * (i + 1) / 1000.0}
    print(json.dumps(res, indent=1))
    print(f"\nPROJECTION: mean {res['decide_ms_mean']:.2f} ms/tick x {i+1} ticks = "
          f"{res['projected_wait_s_per_game']:.1f} s of grader wait (budget ~600 s)", flush=True)
    json.dump(res, open(os.path.join(HERE, "hive_latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
