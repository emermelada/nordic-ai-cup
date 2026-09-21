import math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
import hive as H
sim = SimulationCore(seed=1); env = sim.env; hv = H.Hive(seed=1)
actions = []
for k in range(567):
    st = sim.step(actions)
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
m = hv.mem[11]; a = env.agents_dict[11]
print("hive pose", round(m.x,2), round(m.y,2), round(m.h,3), "true", round(a.x,2), round(a.y,2), round(a.direction,3), "last", m.last)
fm = hv.maps["W"]
print("known edges near:")
for e in fm.edges:
    mx, my = (e[0]+e[2])/2, (e[1]+e[3])/2
    if math.hypot(mx - a.x, my - a.y) < 120: print("  ", [round(v,1) for v in e])
print("true obstacles near:")
for o in env.obstacles:
    if o.x - 60 < a.x < o.x + o.width + 60 and o.y - 60 < a.y < o.y + o.height + 60:
        print("   x", round(o.x,1), "y", round(o.y,1), "w", round(o.width,1), "h", round(o.height,1))
d, md, tn, sp = m.last
direction = a.direction + md
nx, ny = a.x + d*math.cos(direction), a.y + d*math.sin(direction)
print("next intended", round(nx,1), round(ny,1), "hive blocked?", fm.blocked(nx, ny))
