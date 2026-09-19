import sys, os, statistics
sys.path.insert(0, os.path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seed = int(sys.argv[1])
sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(seed=seed)
env.spawn_predator = lambda *a, **k: None
info = {}; actions = []
for k in range(30000):
    pre = {a.agent_id: (a.energy, a.age > a.max_age) for a in env.agents}
    st = sim.step(actions)
    if st["num_agents"] == 0 or env.time > 3000: break
    for a in env.agents:
        d = info.setdefault(a.agent_id, {"old_ticks": 0, "drain": 0.0, "eat_old": 0.0, "kids_old": 0, "detected": None, "onset": None, "good": None})
        if a.age > a.max_age:
            if d["onset"] is None: d["onset"] = env.time
            d["old_ticks"] += 1; d["drain"] += 0.01 * a.age
        m = hv.mem.get(a.agent_id)
        if m is not None and m.old and d["detected"] is None: d["detected"] = env.time
    acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
    for act in acts:
        a = env.agents_dict.get(act["agent_id"])
        if a is None: continue
        d = info[a.agent_id]
        m = hv.mem.get(a.agent_id)
        d["good"] = m.fit >= hv.best_fit - hv.p["breed_gap"]
        if a.age > a.max_age and act["spawn_agent"] and a.energy > 100.5: d["kids_old"] += 1
    actions = to_actions(acts)
old = [d for d in info.values() if d["onset"] is not None]
print("agents that reached old age", len(old))
print("mean old ticks", statistics.fmean(d["old_ticks"] for d in old), "mean drain", round(statistics.fmean(d["drain"] for d in old)))
lag = [d["detected"] - d["onset"] for d in old if d["detected"] is not None]
print("detected", len(lag), "of", len(old), "lag mean", round(statistics.fmean(lag), 2) if lag else None)
print("kids after onset mean", statistics.fmean(d["kids_old"] for d in old))
for g in (True, False):
    ds = [d for d in old if d["good"] is g]
    if ds: print("good" if g else "weak", len(ds), "drain mean", round(statistics.fmean(d["drain"] for d in ds)), "old ticks", round(statistics.fmean(d["old_ticks"] for d in ds)), "kids_old", round(statistics.fmean(d["kids_old"] for d in ds), 2))
