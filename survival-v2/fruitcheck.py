import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); T = float(sys.argv[2])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
env.spawn_predator = lambda *a, **k: None
actions = []
for k in range(30000):
    st = sim.step(actions)
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    if env.time >= T:
        fm = hv.maps["W"]
        print(f"t={env.time:.1f} true fruits {len(env.fruits)} known {fm.fx.size}")
        for a in env.agents:
            m = hv.mem[a.agent_id]
            near_true = [f for f in env.fruits if math.hypot(f.x - a.x, f.y - a.y) < 120]
            near_tree = min((math.hypot(t.x - a.x, t.y - a.y) for t in env.trees), default=None)
            known_near = sum(1 for x, y in zip(fm.fx, fm.fy) if math.hypot(x - m.x, y - m.y) < 120)
            obs_f = sum(1 for o in st["observations"][[s["agent_id"] for s in st["observations"]].index(a.agent_id)]["observations"] if o["type"] == "Fruit")
            print(f"  aid {a.agent_id} mode {m.mode} E {a.energy:.0f} hear {a.hearing_radius:.0f} vis {a.vision_radius:.0f} "
                  f"true fruit<120 {len(near_true)} known<120 {known_near} observed_now {obs_f} nearest tree {near_tree:.0f} target {m.target} pos ({a.x:.0f},{a.y:.0f}) est ({m.x:.0f},{m.y:.0f})")
        break
    actions = to_actions(acts)
