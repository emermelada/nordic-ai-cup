"""Find the first tick where a world-frame pose diverges from the simulator's truth."""
import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sim = SimulationCore(seed=seed); env = sim.env; hive = Hive(seed=seed)
actions = []
prev_frame = {}
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 600):
    st = sim.step(actions)
    if st["num_agents"] == 0: break
    acts = hive.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    bad = []
    for a in env.agents:
        m = hive.mem.get(a.agent_id)
        if m is None or m.frame != "W": continue
        e = math.hypot(m.x - a.x, m.y - a.y); dh = abs((m.h - a.direction + math.pi) % (2*math.pi) - math.pi)
        if e > 1.0 or dh > 1e-3:
            bad.append((a.agent_id, round(e, 1), round(dh, 3), m.mode, m.fixes, prev_frame.get(a.agent_id)))
    if bad:
        print("tick", k, "t", round(env.time, 1), "bad", bad[:6], "stats", hive.stats); break
    for a in env.agents:
        m = hive.mem.get(a.agent_id)
        if m: prev_frame[a.agent_id] = m.frame
    actions = to_actions(acts)
else:
    print("no divergence; stats", hive.stats)
