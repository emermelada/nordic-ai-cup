import sys, os, statistics
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
if not os.environ.get("WITH_PREDATORS"): env.spawn_predator = lambda *a, **k: None
actions = []; last = (0, 0, 0)
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000: break
    actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    if k % 1200 == 1199:
        eaten, rot = env.stat["fruit_n"], env.stat.get("rot_n", 0)
        e_known = sum(fm.fx.size for fm in hv.maps.values())
        prod = sum(1 for t in env.trees if t.age >= 20)
        print(f"t={env.time:5.0f} pop {len(env.agents):3d} eaten {eaten-last[0]:4d} rotted {rot-last[1]:4d} standing {len(env.fruits):3d} known {e_known:3d} trees {len(env.trees)} productive {prod} "
              f"E {statistics.fmean(a.energy for a in env.agents):.0f} ages>60 {sum(1 for a in env.agents if a.age > 60)}")
        last = (eaten, rot, 0)
