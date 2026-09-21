import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle
from fastsim import SimulationCore, to_actions
from hive import Hive
import numpy as np
seed = int(sys.argv[1]); T0 = float(sys.argv[2])
sim = SimulationCore(seed=seed); env = sim.env; env.spawn_predator = lambda *a, **k: None
hv = Hive(seed=seed); birth = {}; actions = []
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0: print("extinct at", round(env.time)); break
    for f in env.fruits: birth.setdefault(id(f), env.time - f.age / 2.0)
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    fm = hv.maps["W"]
    fm.fx = np.array([f.x for f in env.fruits]); fm.fy = np.array([f.y for f in env.fruits])
    b = np.array([birth[id(f)] for f in env.fruits]); fm.flo, fm.fhi, fm.flast = b, b, np.full(len(b), env.time)
    fm.tx = np.array([t.x for t in env.trees]); fm.ty = np.array([t.y for t in env.trees]); ages = np.array([t.age for t in env.trees])
    fm.tfirst = env.time - ages; fm.tlast = np.full(len(ages), env.time); fm.tfruit = np.where(ages >= 20, env.time, -1e9)
    fm.twatch = np.where(ages >= 20, 0.0, 100.0); fm._prod_tick = -1
    acts = hv._plan({aid: v for aid, v in oracle._alive(hv, st)}, oracle._by_frame(hv, st))
    for act in acts:
        m = hv.mem[act["agent_id"]]; m.last = (act["move_distance"], act["move_direction"], act["turn_angle"], act["spawn_agent"])
    if env.time >= T0 and k % 50 == 0:
        rows = []
        for act in acts:
            a = env.agents_dict[act["agent_id"]]; m = hv.mem[a.agent_id]
            ripe = sum(1 for f in env.fruits if math.hypot(f.x - a.x, f.y - a.y) < 100 and f.energy >= 59)
            rows.append(f"{a.agent_id}:{m.mode[:4]} a{a.age:.0f} E{a.energy:.0f} sp{act['spawn_agent']:d} br{m.spawned} ripe<100:{ripe}")
        prodt = sum(1 for t in env.trees if t.age >= 20)
        print(f"t={env.time:.0f} trees {len(env.trees)} prod {prodt} fruit {len(env.fruits)} ripe {sum(1 for f in env.fruits if f.energy >= 59)} | " + "  ".join(rows))
    actions = to_actions(acts)
