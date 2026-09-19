import sys, math, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive, wrap
import src.elements.predator as P
seed = int(sys.argv[1]); which = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
dec = {}
orig = P.Predator.step
def step(self, observation=None):
    sig = orig(self, observation)
    ags = [o for o in (observation or []) if o.get("type") == "Agent"]
    if ags:
        c = min(ags, key=lambda f: f["distance"])
        mode = "charge" if (abs(c["rel_dir"]) > math.pi/2 or c["distance"] < 90) else "pivot"
        dec[id(self)] = (mode, c.get("id"), round(c["distance"]), round(c["rel_dir"], 2), round(self.energy))
    else:
        dec[id(self)] = ("wander/edge", None, None, None, round(self.energy))
    return sig
P.Predator.step = step
trace = collections.defaultdict(lambda: collections.deque(maxlen=40))
actions = []; nk = 0; shown = 0
for k in range(30000):
    st = sim.step(actions)
    for (t, aid, pid) in env.kill_log[nk:]:
        if shown == which:
            print(f"KILL t={t:.1f} aid {aid}")
            for row in trace[aid]: print("  ", row)
            sys.exit()
        shown += 1
    nk = len(env.kill_log)
    if st["num_agents"] == 0: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    amap = {a["agent_id"]: a for a in acts}
    for a in env.agents:
        m = hv.mem.get(a.agent_id)
        if m is None: continue
        for p in env.predators:
            d = math.hypot(p.x - a.x, p.y - a.y)
            if d < 260 and not p.resting:
                face = round(wrap(math.atan2(p.y - a.y, p.x - a.x) - a.direction), 2)
                act = amap.get(a.agent_id, {})
                trace[a.agent_id].append((round(env.time,1), m.mode, round(a.energy), f"d={d:.0f}", f"bearing_vs_facing={face}", dec.get(id(p)),
                                          f"cmd dist={act.get('move_distance',0):.1f} turn={act.get('turn_angle',0):.2f}"))
                break
    actions = to_actions(acts)
