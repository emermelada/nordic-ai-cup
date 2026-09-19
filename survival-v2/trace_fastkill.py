import sys, math, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive, wrap
import src.elements.predator as P
seeds = [int(x) for x in sys.argv[1].split(",")]; nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 2
dec = {}
orig = P.Predator.step
def step(self, observation=None):
    sig = orig(self, observation)
    ags = [o for o in (observation or []) if o.get("type") == "Agent"]
    if ags:
        c = min(ags, key=lambda f: f["distance"])
        dec[id(self)] = ("CHARGE" if (abs(c["rel_dir"]) > math.pi/2 or c["distance"] < 90) else "pivot", c.get("id"), round(c["distance"]), round(self.energy))
    else:
        dec[id(self)] = ("-", None, None, round(self.energy))
    return sig
P.Predator.step = step
shown = 0
for seed in seeds:
    sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
    trace = collections.defaultdict(lambda: collections.deque(maxlen=25))
    actions = []; nk = 0
    for k in range(30000):
        st = sim.step(actions)
        for (t, aid, pid) in env.kill_log[nk:]:
            h = list(trace[aid])
            if h and h[-1]["speed"] >= 17.5 and shown < nshow:
                shown += 1
                print(f"=== seed {seed} KILL t={t:.1f} aid {aid} speed {h[-1]['speed']}")
                for r in h: print("  ", {k2: v for k2, v in r.items() if k2 != 'speed'})
        nk = len(env.kill_log)
        if st["num_agents"] == 0 or shown >= nshow: break
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        amap = {a["agent_id"]: a for a in acts}
        for a in env.agents:
            m = hv.mem.get(a.agent_id)
            ps = sorted(((math.hypot(p.x - a.x, p.y - a.y), p) for p in env.predators if not p.resting), key=lambda q: q[0])
            if not ps or ps[0][0] > 200: continue
            d, p = ps[0]
            act = amap.get(a.agent_id, {})
            th = hv._threats(m, hv.maps[m.frame]) if m and m.frame in hv.maps else []
            trace[a.agent_id].append({"t": round(env.time, 1), "mode": m.mode if m else None, "E": round(a.energy), "d": round(d),
                "bear_vs_face": round(wrap(math.atan2(p.y - a.y, p.x - a.x) - a.direction), 2), "pred": dec.get(id(p)),
                "n_preds<200": len(ps if ps[-1][0] < 200 else [q for q in ps if q[0] < 200]), "threats_known": len(th),
                "cmd": (round(act.get("move_distance", 0), 1), round(act.get("turn_angle", 0), 2)), "speed": round(a.speed, 1),
                "biome": st["observations"][[s["agent_id"] for s in st["observations"]].index(a.agent_id)]["biome"]})
        actions = to_actions(acts)
    if shown >= nshow: break
