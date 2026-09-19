import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
if not os.environ.get("WITH_PREDATORS"): env.spawn_predator = lambda *a, **k: None
actions = []
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    if k % 600 == 599:
        fm = hv.maps.get("W")
        prod = [t for t in env.trees if t.age >= 20]
        known = 0; camped = 0; fruit_near_prod = 0
        for t in prod:
            if fm is not None and fm.tx.size and (abs(fm.tx - t.x) + abs(fm.ty - t.y)).min() < 4: known += 1
            if any(math.hypot(a.x - t.x, a.y - t.y) < 30 for a in env.agents): camped += 1
            fruit_near_prod += sum(1 for f in env.fruits if math.hypot(f.x - t.x, f.y - t.y) < 65)
        map_trees = fm.tx.size if fm is not None else 0
        dead_in_map = 0
        if fm is not None:
            for x, y in zip(fm.tx, fm.ty):
                if not any(abs(t.x - x) + abs(t.y - y) < 4 for t in env.trees): dead_in_map += 1
        modes = {}
        for m in hv.mem.values(): modes[m.mode] = modes.get(m.mode, 0) + 1
        print(f"t={env.time:5.0f} pop {len(env.agents):3d} productive trees {len(prod):3d} known {known:3d} camped {camped:3d} fruit-at-productive {fruit_near_prod:3d} "
              f"| map trees {map_trees:3d} of which dead {dead_in_map:3d} | modes {modes}")
    actions = to_actions(acts)
