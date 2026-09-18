"""DEATH-BUDGET DIAGNOSTIC: what actually kills the fleet, and how do predators accumulate?

Why this matters: environment.py:763 gives the predator spawn chance as
    (1/max(1,N_predators)) * dt * time * 0.0001
i.e. it grows LINEARLY WITH TIME and does not depend on our fleet size. Episodes start with zero
predators. Every experiment today assumed the fleet dies of STARVATION (when income decays), but if
predation dominates the second half then all the foraging work was aimed at the wrong failure.

For each agent that disappears we compare its last-seen energy: eaten agents vanish with energy
still in the tank, starved agents vanish at ~0. Also logs predator count and fleet energy over time.

Usage: python w_deaths.py <params.json> <seed> <horizon>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import SimulationCore                      # noqa: E402
import best_controller as bc                            # noqa: E402
from env_wrapper import make_action                     # noqa: E402


def main():
    path = sys.argv[1]
    seed = int(sys.argv[2])
    horizon = int(sys.argv[3])
    P = dict(bc.DEFAULT_PARAMS)
    P.update(json.load(open(path)).get("params", json.load(open(path))))
    bc.reset_memory()
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    prev = {a.agent_id: float(a.energy) for a in core.env.agents}
    deaths = {"eaten": 0, "starved": 0}
    rows = []
    i = 0
    for i in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in states]
        core.step(acts)
        alive = {a.agent_id: float(a.energy) for a in core.env.agents}
        for aid, e in prev.items():
            if aid not in alive:
                deaths["eaten" if e > 5.0 else "starved"] += 1
        prev = alive
        if i % 1000 == 0:
            rows.append({"t": i + 1, "agents": len(alive), "predators": len(core.env.predators),
                         "fleet_energy": round(sum(alive.values()), 1)})
    tot = max(1, deaths["eaten"] + deaths["starved"])
    print(f"seed={seed} steps={i + 1} deaths={deaths} "
          f"(eaten {100 * deaths['eaten'] / tot:.0f}% / starved {100 * deaths['starved'] / tot:.0f}%)", flush=True)
    print("t, agents, predators, fleet_energy", flush=True)
    for r in rows:
        print(f"  {r['t']:6d}  {r['agents']:3d}  {r['predators']:3d}  {r['fleet_energy']:9.1f}", flush=True)


if __name__ == "__main__":
    main()
