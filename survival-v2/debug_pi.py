import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
import hive as H
sim = SimulationCore(seed=3); env = sim.env; hv = H.Hive(seed=3)
orig_merge = H.Hive._merge_by_sighting
def merge(self, m, o, om):
    a = env.agents_dict[m.aid]; b = env.agents_dict[om.aid]
    true_d = math.hypot(b.x - a.x, b.y - a.y)
    true_ang = (math.atan2(b.y - a.y, b.x - a.x) - a.direction + math.pi) % (2*math.pi) - math.pi
    true_rel = (math.atan2(a.y - b.y, a.x - b.x) - b.direction + math.pi) % (2*math.pi) - math.pi
    print(f"tick {self.tick}: {m.aid}(f{m.frame}) sees {om.aid}(f{om.frame}) obs d {o['distance']:.3f} ang {o['angle']:.3f} rel {o.get('rel_dir'):.3f} | true d {true_d:.3f} ang {true_ang:.3f} rel {true_rel:.3f} | m.h err {(m.h - a.direction + math.pi) % (2*math.pi) - math.pi:.3f}")
    orig_merge(self, m, o, om)
H.Hive._merge_by_sighting = merge
actions = []
for k in range(3):
    st = sim.step(actions)
    for s in st["observations"]:
        if s["agent_id"] == 9: print("agent 9 age", s["age"], "energy", s["energy"])
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    for aid in (4, 9):
        m = hv.mem.get(aid); a = env.agents_dict.get(aid)
        if m and a: print(f"  after tick {hv.tick}: aid {aid} frame {m.frame} pose ({m.x:.2f},{m.y:.2f},{m.h:.3f}) stale {m.stale}")
