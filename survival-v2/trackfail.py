"""Find the first ticks where an agent's believed pose drifts from the truth (world frame).

    python trackfail.py <seed> [threshold_px] [max_reports]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastsim import SimulationCore, to_actions  # noqa: E402
Hive = __import__(os.environ.get("HIVE_MODULE", "hive")).Hive

seed = int(sys.argv[1])
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 6
sim = SimulationCore(seed=seed)
env = sim.env
hv = Hive(seed=seed)
actions = []
prev = {}
reported = set()
n = 0
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000:
        break
    before = {a.agent_id: (hv.mem[a.agent_id].x, hv.mem[a.agent_id].y, hv.mem[a.agent_id].frame, hv.mem[a.agent_id].mode)
              for a in env.agents if a.agent_id in hv.mem}
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    byact = {a["agent_id"]: a for a in acts}
    for a in env.agents:
        m = hv.mem.get(a.agent_id)
        if m is None or m.frame != "W":
            continue
        e = math.hypot(m.x - a.x, m.y - a.y)
        eh = abs((m.h - a.direction + math.pi) % (2 * math.pi) - math.pi)
        if (e > thr or eh > 0.01) and a.agent_id not in reported:
            reported.add(a.agent_id)
            n += 1
            ix, iy = min(max(int(a.x), 0), env.width - 1), min(max(int(a.y), 0), env.height - 1)
            b = before.get(a.agent_id)
            print(f"t {env.time:.1f} agent {a.agent_id} err {e:.1f} px, heading err {eh:.3f}; true ({a.x:.1f},{a.y:.1f}) "
                  f"belief ({m.x:.1f},{m.y:.1f}) prev belief {b[:2] if b else None} prev frame {b[2] if b else None} "
                  f"mode {m.mode} biome {env.biome_map[ix, iy].type} stale {m.stale} fixes {m.fixes} age {a.age:.1f} "
                  f"last act {m.last}")
            if n >= nmax:
                break
    if n >= nmax:
        break
    for a in env.agents:
        prev[a.agent_id] = (a.x, a.y)
    actions = to_actions(acts)
print("end", round(env.time))
