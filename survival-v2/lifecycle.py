"""Per-agent life accounting (no predators by default): eaten energy, children, energy at old-age onset, lifetime."""
import sys, os, statistics, json
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1]); params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(params=params, seed=seed)
if not os.environ.get("WITH_PREDATORS"): env.spawn_predator = lambda *a, **k: None
info = {}   # aid -> dict
actions = []
prev_alive = set()
for k in range(30000):
    before = {a.agent_id: a.energy for a in env.agents}
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000: break
    alive = {a.agent_id: a for a in env.agents}
    for aid, a in alive.items():
        d = info.setdefault(aid, {"born": env.time, "eaten": 0.0, "kids": 0, "E_old": None, "max_age": a.max_age, "fit": None, "E_max": 0.0})
        if a.age > a.max_age and d["E_old"] is None: d["E_old"] = a.energy
        d["E_max"] = max(d["E_max"], a.energy)
    for aid in prev_alive - set(alive):
        info[aid]["died"] = env.time
    prev_alive = set(alive)
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    for act in acts:
        if act["spawn_agent"] and alive.get(act["agent_id"]) is not None and alive[act["agent_id"]].energy > 100:
            info[act["agent_id"]]["kids"] += 1
        m = hv.mem.get(act["agent_id"])
        if m is not None: info[act["agent_id"]]["fit"] = round(m.fit, 2); info[act["agent_id"]]["good"] = m.fit >= hv.best_fit - hv.p["breed_gap"]
    actions = to_actions(acts)
dead = [d for d in info.values() if "died" in d]
life = [d["died"] - d["born"] for d in dead]
print(f"seed {seed} end t {env.time:.0f}; dead agents {len(dead)}")
print(f"lifetime mean {statistics.fmean(life):.0f}s; kids mean {statistics.fmean(d['kids'] for d in dead):.2f}; kids=0 share {sum(1 for d in dead if d['kids']==0)/len(dead):.2f}")
eo = [d["E_old"] for d in dead if d["E_old"] is not None]
print(f"reached old age {len(eo)}/{len(dead)}; energy at old-age onset mean {statistics.fmean(eo):.0f} median {statistics.median(eo):.0f}")
print(f"good-genome share of dead {sum(1 for d in dead if d.get('good'))/len(dead):.2f}; E_max mean {statistics.fmean(d['E_max'] for d in dead):.0f}")
for band in [(0, 600), (600, 1200), (1200, 3000)]:
    ds = [d for d in dead if band[0] <= d["born"] < band[1]]
    if ds:
        print(f"  born {band}: n {len(ds)} life {statistics.fmean(d['died']-d['born'] for d in ds):.0f}s kids {statistics.fmean(d['kids'] for d in ds):.2f} "
              f"E_old {statistics.fmean([d['E_old'] for d in ds if d['E_old'] is not None] or [0]):.0f} good {sum(1 for d in ds if d.get('good'))/len(ds):.2f} Emax {statistics.fmean(d['E_max'] for d in ds):.0f}")
