import sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); T0 = float(sys.argv[2]); nshow = int(sys.argv[3])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed); actions = []
hist = collections.defaultdict(lambda: collections.deque(maxlen=30)); shown = 0
prev = {}
for k in range(30000):
    st = sim.step(actions)
    alive = {a.agent_id for a in env.agents}
    for aid in list(prev):
        if aid not in alive:
            a_e, a_old = prev[aid]
            killed = any(x[1] == aid for x in env.kill_log[-5:])
            if env.time > T0 and not killed and not a_old and shown < nshow:
                shown += 1
                print(f"=== starved aid {aid} t={env.time:.1f}")
                for r in list(hist[aid])[::3]: print("   ", r)
    prev = {a.agent_id: (a.energy, a.age > a.max_age) for a in env.agents}
    if st["num_agents"] == 0 or shown >= nshow: break
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    if env.time > T0 - 30:
        for a in env.agents:
            m = hv.mem.get(a.agent_id)
            prod = [t for t in env.trees if t.age >= 20]
            dt = min((math.hypot(t.x - a.x, t.y - a.y) for t in prod), default=-1)
            df = min((math.hypot(f.x - a.x, f.y - a.y) for f in env.fruits), default=-1)
            fm = hv.maps.get(m.frame) if m else None
            kf = fm.fx.size if fm is not None else 0
            hist[a.agent_id].append((round(env.time, 1), m.mode if m else None, round(a.energy), f"prodtree {dt:.0f}", f"fruit {df:.0f}", f"known_fruit {kf}", f"fit_ok {m.spawned if m else None}"))
    actions = to_actions(acts)
