import sys, os, statistics
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); t0, t1 = float(sys.argv[2]), float(sys.argv[3])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
if not os.environ.get("WITH_PREDATORS"): env.spawn_predator = lambda *a, **k: None
actions = []
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > t1: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    if env.time >= t0 and k % 100 == 0:
        rows = []
        for a in env.agents:
            m = hv.mem[a.agent_id]
            good = m.fit >= hv.best_fit - hv.p["breed_gap"]
            rows.append(f"{a.agent_id}:{m.mode[:4]} age{a.age:.0f} E{a.energy:.0f} {'G' if good else 'w'}{'O' if m.old else ''}")
        fm = hv.maps.get("W")
        print(f"t={env.time:.0f} trees {len(env.trees)} fruit {len(env.fruits)} known_fruit {fm.fx.size if fm else '-'} | " + "  ".join(rows))
    actions = to_actions(acts)
