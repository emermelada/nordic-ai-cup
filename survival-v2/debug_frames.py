import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
import hive as H
sim = SimulationCore(seed=1); env = sim.env; hv = H.Hive(seed=1)
orig_move = H.Hive._move_frame
def move(self, src, dst, phi, tx, ty):
    print(f"tick {self.tick}: move frame {src} -> {dst} phi {phi:.3f} members {[q.aid for q in self.mem.values() if q.frame == src]} edges {len(self.maps[src].edges)}")
    orig_move(self, src, dst, phi, tx, ty)
H.Hive._move_frame = move
actions = []
reported = set()
for k in range(600):
    st = sim.step(actions)
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    groups = {}
    for a in env.agents:
        m = hv.mem.get(a.agent_id)
        if m: groups.setdefault(m.frame, []).append(((a.direction - m.h) % (2*math.pi), a, m))
    for fid, g in groups.items():
        offs = [x[0] for x in g]
        if max(offs) - min(offs) > 1e-3 and (fid, k // 50) not in reported:
            reported.add((fid, k // 50))
            print(f"tick {hv.tick} frame {fid}: inconsistent heading offsets {[(x[1].agent_id, round(x[0],3)) for x in g][:8]}")
