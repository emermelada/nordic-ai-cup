import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
import hive as H
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sim = SimulationCore(seed=seed); env = sim.env; hv = H.Hive(seed=seed)
orig_fix = H.Hive._landmark_fix
orig_pred = H.Hive._predict
def pred(self, m):
    a = env.agents_dict.get(m.aid)
    before = (m.x, m.y)
    orig_pred(self, m)
    if a is not None and m.frame == "W":
        e = math.hypot(m.x - a.x, m.y - a.y)
        if e > 0.5:
            print(f"  predict err aid {m.aid} tick {hv.tick} err {e:.2f} last {m.last} biome {m.biome} blocked_pred {self.maps[m.frame].blocked(m.x, m.y)}")
def fix(self, m, edges):
    a = env.agents_dict.get(m.aid)
    bx, by = m.x, m.y
    orig_fix(self, m, edges)
    if a is not None and (abs(m.x - bx) + abs(m.y - by)) > 1e-3 and m.frame == "W":
        print(f"FIX aid {m.aid} tick {hv.tick}: before err {math.hypot(bx - a.x, by - a.y):.2f} after err {math.hypot(m.x - a.x, m.y - a.y):.2f} shift ({m.x-bx:.2f},{m.y-by:.2f})")
H.Hive._landmark_fix = fix; H.Hive._predict = pred
actions = []
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 600):
    st = sim.step(actions)
    if st["num_agents"] == 0: break
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
