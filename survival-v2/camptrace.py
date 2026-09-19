import sys, os, math, collections
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); T0 = float(sys.argv[2])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
env.spawn_predator = lambda *a, **k: None
actions = []; hist = collections.defaultdict(list)
for k in range(30000):
    st = sim.step(actions)
    if st["num_agents"] == 0: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    if env.time >= T0:
        for act in acts:
            m = hv.mem[act["agent_id"]]
            if m.mode == "camp":
                hist[m.aid].append((round(env.time, 1), m.target, round(act["move_distance"], 1), round(m.x), round(m.y)))
    if env.time >= T0 + 30: break
    actions = to_actions(acts)
for aid, h in list(hist.items())[:4]:
    moves = [x for x in h if x[2] > 0.5]
    targets = collections.Counter(x[1] for x in h)
    print(f"aid {aid}: camp ticks {len(h)} moving ticks {len(moves)} dist {sum(x[2] for x in h):.0f} targets {dict(targets)}")
    print("   ", h[::10][:12])
