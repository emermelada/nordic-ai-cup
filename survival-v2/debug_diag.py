import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
import hive as H
sim = SimulationCore(seed=1); env = sim.env; hv = H.Hive(seed=1)
orig_map = H.Hive._map_edges
orig_merge = H.Hive._merge_by_sighting
orig_reg = H.Hive._register_world
def herr(m):
    a = env.agents_dict.get(m.aid)
    return None if a is None else round(abs((m.h - a.direction + math.pi) % (2*math.pi) - math.pi), 4)
def map_edges(self, m, edges):
    if self.maps[m.frame].world and herr(m) and herr(m) > 1e-3:
        print(f"tick {self.tick}: mapping edges from aid {m.aid} with heading error {herr(m)} frame {m.frame}")
    orig_map(self, m, edges)
def merge(self, m, o, om):
    fa, fb = m.frame, om.frame
    orig_merge(self, m, o, om)
    bad = [(q.aid, herr(q)) for q in self.mem.values() if q.frame in ("W",) and herr(q) and herr(q) > 1e-3]
    if bad: print(f"tick {self.tick}: after merge {m.aid}({fa}) sees {om.aid}({fb}) -> bad headings {bad[:5]}")
def reg(self, m, X, Y, Hd):
    a = env.agents_dict.get(m.aid)
    print(f"tick {self.tick}: world reg aid {m.aid} X {X:.1f} Y {Y:.1f} H {Hd:.3f} true {a.x:.1f} {a.y:.1f} {a.direction % (2*math.pi):.3f} frame {m.frame}")
    orig_reg(self, m, X, Y, Hd)
H.Hive._map_edges = map_edges; H.Hive._merge_by_sighting = merge; H.Hive._register_world = reg
actions = []
for k in range(600):
    st = sim.step(actions)
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
