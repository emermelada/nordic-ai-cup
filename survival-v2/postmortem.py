"""Why agents get eaten: for every kill, what the hive knew and did in the ticks before."""
import sys, math, json, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from fastsim import SimulationCore, to_actions
from hive import Hive
seeds = [int(x) for x in sys.argv[1].split(",")]
params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
summary = collections.Counter(); rows = []
for seed in seeds:
    sim = SimulationCore(seed=seed); env = sim.env; hv = Hive(params=params, seed=seed)
    hist = collections.defaultdict(lambda: collections.deque(maxlen=60))   # aid -> recent (t, mode, E, dmin_true, pred_id, threat_known, sprintable, speed, facing_ok)
    actions = []; nk = 0
    for k in range(30000):
        st = sim.step(actions)
        for (t, aid, pid) in env.kill_log[nk:]:
            h = list(hist[aid]); nk += 1
            if not h: continue
            first = next((x for x in h if x[4] == pid and x[3] < 250), None)
            last = h[-1]
            known = any(x[4] == pid and x[5] for x in h)
            first_known = next((x for x in h if x[4] == pid and x[5]), None)
            rows.append(dict(seed=seed, t=round(t), mode=last[1], E=round(last[2]), speed=round(last[7],1), sprint_ok=last[6],
                             d_first250=round(first[3]) if first else None, known=known,
                             d_first_known=round(first_known[3]) if first_known else None, modes=[x[1] for x in h[-10:]]))
        nk = len(env.kill_log)
        if st["num_agents"] == 0 or env.time > 3000: break
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        preds = env.predators
        for a in env.agents:
            m = hv.mem.get(a.agent_id)
            if m is None or not preds: continue
            pd = [(math.hypot(p.x - a.x, p.y - a.y), id(p)) for p in preds if not p.resting]
            if not pd: continue
            dmin, pid = min(pd)
            th = hv._threats(m, hv.maps[m.frame]) if m.frame in hv.maps else []
            hist[a.agent_id].append((env.time, m.mode, a.energy, dmin, pid, bool(th), a.energy > a.max_energy / 5, a.speed))
        actions = to_actions(acts)
for r in rows:
    summary["kills"] += 1
    summary["known_before"] += r["known"]
    summary["mode_" + r["mode"]] += 1
    summary["cant_sprint"] += (not r["sprint_ok"])
    if r["d_first_known"] is not None: summary["known_at_<100"] += r["d_first_known"] < 100
print(dict(summary))
for r in rows[:25]: print(r)
