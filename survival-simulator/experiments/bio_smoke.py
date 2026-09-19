#!/usr/bin/env python3
"""bio_smoke.py - does the biome rule activate, and how big is the leak it fixes?

Measures, with the rule OFF: the share of SPAWNS made in zero-production terrain
(desert/river) and the share of agent-ticks spent there. Then compares survival.
"""
import json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import best_controller as bc
import oracle_probe as op
from env_wrapper import make_action

P0 = dict(bc.DEFAULT_PARAMS)
P0.update(json.load(open(os.path.join(os.path.dirname(HERE), "best_controller", "params.json"))))


def dead_biome(x, y):
    try:
        name = type(core.env.biome_map[int(x // 100), int(y // 100)]).__name__.lower()
    except Exception:
        return False
    return ("desert" in name) or ("river" in name)


def run(seed, over, ticks=14000):
    global core
    P = dict(P0)
    P.update(over)
    random.seed(seed)
    np.random.seed(seed)
    core = op.build_core(seed)
    f = bc.make_policy(P)
    bc.reset_memory()
    spawns_dead = spawns_all = dead_ticks = all_ticks = 0
    pops = []
    seen = set()
    for t in range(ticks):
        alive = list(core.env.agents)
        if not alive:
            return t, pops, spawns_dead, spawns_all, dead_ticks, all_ticks
        for a in alive:
            if a.agent_id not in seen:
                seen.add(a.agent_id)
                spawns_all += 1
                if dead_biome(a.x, a.y):
                    spawns_dead += 1
            all_ticks += 1
            if dead_biome(a.x, a.y):
                dead_ticks += 1
        states = [core.env.get_agent_state(a.agent_id) for a in alive]
        acts = [(s["agent_id"], make_action(s, f(s))) for s in states]
        core.step(acts)
        if t % 500 == 0:
            pops.append(len(core.env.agents))
    return ticks, pops, spawns_dead, spawns_all, dead_ticks, all_ticks


if __name__ == "__main__":
    for seed in (3700, 3701, 3702):
        t0, p0, sd0, sa0, td0, ta0 = run(seed, {})
        t1, p1, _, _, _, _ = run(seed, {"bio_mode": 1.0})
        mid0 = sum(p0[8:16]) / max(1, len(p0[8:16]))
        mid1 = sum(p1[8:16]) / max(1, len(p1[8:16]))
        leak = 100.0 * sd0 / max(1, sa0)
        dead_time = 100.0 * td0 / max(1, ta0)
        print(f"seed {seed}: OFF ticks={t0:6d} mid-pop={mid0:5.1f} | BIO ticks={t1:6d} mid-pop={mid1:5.1f} "
              f"| leak: {leak:4.1f}% of spawns in dead terrain, {dead_time:4.1f}% of agent-ticks there "
              f"| differs={'YES' if (t0 != t1 or p0 != p1) else 'NO <<< INERT'}")
