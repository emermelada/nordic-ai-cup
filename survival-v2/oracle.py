"""Diagnostic: the hive with perfect knowledge of trees and fruit (true positions, exact fruit spawn times).

    python oracle.py <seeds> [--pred]

If the colony survives far longer with the oracle, perception/exploration is the bottleneck; if not, the
decision and energy logic is.
"""
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastsim import SimulationCore, to_actions  # noqa: E402
from hive import Hive  # noqa: E402


def run(seed, predators):
    sim = SimulationCore(seed=seed)
    env = sim.env
    if not predators:
        env.spawn_predator = lambda *a, **k: None
    hv = Hive(seed=seed, params=__import__("json").loads(os.environ["HIVE_PARAMS"]) if os.environ.get("HIVE_PARAMS") else None)
    birth = {}
    actions = []
    for k in range(30000):
        st = sim.step(actions)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        for f in env.fruits:
            birth.setdefault(id(f), env.time - f.age / 2.0)   # fruit.age grows 2 per second
        # put every agent in the world frame at its true pose, then overwrite trees and fruit with the truth
        hv.decide({"sim_time": st["sim_time"], "agent_status": []}) if False else None
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        if "W" in hv.maps:
            fm = hv.maps["W"]
            for a in env.agents:
                m = hv.mem.get(a.agent_id)
                if m is not None and m.frame != "W":
                    pass
            fm.fx = np.array([f.x for f in env.fruits])
            fm.fy = np.array([f.y for f in env.fruits])
            b = np.array([birth[id(f)] for f in env.fruits])
            fm.flo, fm.fhi, fm.flast = b, b, np.full(len(b), env.time)
            fm.tx = np.array([t.x for t in env.trees])
            fm.ty = np.array([t.y for t in env.trees])
            ages = np.array([t.age for t in env.trees])
            fm.tfirst = env.time - ages
            fm.tlast = np.full(len(ages), env.time)
            fm.tfruit = np.where(ages >= 20, env.time, -1e9)
            fm.twatch = np.where(ages >= 20, 0.0, 100.0)
            fm._prod_tick = -1
            acts = hv._plan({aid: v for aid, v in _alive(hv, st)}, _by_frame(hv, st))
            for act in acts:
                m = hv.mem[act["agent_id"]]
                m.last = (act["move_distance"], act["move_direction"], act["turn_angle"], act["spawn_agent"])
        actions = to_actions(acts)
    return env.time, env.score


def _alive(hv, st):
    for a in st["observations"]:
        m = hv.mem.get(a["agent_id"])
        if m is not None:
            yield m.aid, (m, {"Fruit": [], "Agent": [], "Predator": [], "Tree": [], "Edge": []})


def _by_frame(hv, st):
    out = {}
    for a in st["observations"]:
        m = hv.mem.get(a["agent_id"])
        if m is not None:
            out.setdefault(m.frame, []).append((m, {}))
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    pred = "--pred" in sys.argv
    res = [run(s, pred) for s in seeds]
    for s, (t, sc) in zip(seeds, res):
        print(f"seed {s}: time {t:.0f} score {sc:.1f}")
    print("mean time", statistics.fmean(t for t, _ in res))
