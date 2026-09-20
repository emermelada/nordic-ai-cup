"""Why do agents die late in the game? One line per death class, over many seeds.

    python deathdiag.py 101-110 [params_json] [late_from_s]

Per dead agent: cause (predator / starved young / starved old / retired), age, fruit eaten, children, energy
peak, time since last meal, share of life in each mode, and the energy it carried into the last 30 s.
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(job):
    seed, params, late = job
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(params=params, seed=seed)
    actions = []
    life = {}            # aid -> dict
    seen_eat = seen_death = seen_kill = 0
    deaths = []
    for k in range(30000):
        st = sim.step(actions)
        t = env.time
        for a in env.agents:
            L = life.get(a.agent_id)
            if L is None:
                L = life[a.agent_id] = {"born": t, "eat": 0, "eatE": 0.0, "last_eat": t, "peak": a.energy,
                                        "kids": 0, "modes": defaultdict(int), "E30": [], "max_age": a.max_age,
                                        "speed": a.speed, "maxE": a.max_energy, "px": defaultdict(float),
                                        "pos": (a.x, a.y), "mode_now": "?"}
            L["peak"] = max(L["peak"], a.energy)
            m = hv.mem.get(a.agent_id)
            if m is not None:
                L["modes"][m.mode] += 1
                L["mode_now"] = m.mode
                L["px"][m.mode] += ((a.x - L["pos"][0]) ** 2 + (a.y - L["pos"][1]) ** 2) ** 0.5
            L["pos"] = (a.x, a.y)
            L["E30"].append(a.energy)
            if len(L["E30"]) > 300:
                L["E30"].pop(0)
        for (te, aid, e, age, over, fx, fy) in env.eat_log[seen_eat:]:
            L = life.get(aid)
            if L:
                L["eat"] += 1
                L["eatE"] += e
                L["last_eat"] = te
        seen_eat = len(env.eat_log)
        for (td, aid, age, max_age) in env.death_log[seen_death:]:
            L = life.get(aid)
            if L:
                m = hv.mem.get(aid)
                cause = "old" if age > max_age else "starve"
                deaths.append((td, cause, L, age))
        seen_death = len(env.death_log)
        for (tk, aid, pid) in env.kill_log[seen_kill:]:
            L = life.get(aid)
            if L:
                deaths.append((tk, "killed", L, tk - L["born"]))
        seen_kill = len(env.kill_log)
        if st["num_agents"] == 0 or t > 3000:
            break
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        for a in acts:
            if a.get("spawn_agent"):
                L = life.get(a["agent_id"])
                if L:
                    L["kids"] += 1
        actions = to_actions(acts)
    end = env.time
    out = []
    for td, cause, L, age in deaths:
        if td < end - late:
            continue
        tot = sum(L["modes"].values()) or 1
        out.append({"cause": cause, "age": age, "eat": L["eat"], "eatE": L["eatE"], "kids": L["kids"],
                    "peak": L["peak"], "since_eat": td - L["last_eat"],
                    "E30": L["E30"][0] if L["E30"] else 0, "max_age": L["max_age"],
                    "modes": {k: v / tot for k, v in L["modes"].items()}, "speed": L["speed"], "maxE": L["maxE"],
                    "px": dict(L["px"]), "mode_now": L["mode_now"]})
    return {"seed": seed, "end": end, "deaths": out}


def main():
    a, b = sys.argv[1].split("-")
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    late = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
    with Pool(int(os.environ.get("DIAG_WORKERS", "10"))) as pool:
        rows = pool.map(run, [(s, params, late) for s in range(int(a), int(b) + 1)], chunksize=1)
    by = defaultdict(list)
    for r in rows:
        for d in r["deaths"]:
            by[d["cause"]].append(d)
    n = sum(len(v) for v in by.values())
    print(f"games {len(rows)}  mean end {sum(r['end'] for r in rows) / len(rows):.0f}  deaths in last {late:.0f} s: {n}")
    for c, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        f = lambda key: sum(d[key] for d in v) / len(v)
        modes = defaultdict(float)
        for d in v:
            for k, x in d["modes"].items():
                modes[k] += x / len(v)
        print(f"  {c:8s} n {len(v):4d}  age {f('age'):5.1f}  eaten {f('eat'):4.1f} ({f('eatE'):5.0f} E)  kids {f('kids'):4.2f}  "
              f"peak {f('peak'):5.0f}  E 30s before {f('E30'):5.0f}  since meal {f('since_eat'):5.1f}s  max_age {f('max_age'):5.1f}  "
              f"maxE {f('maxE'):5.0f}")
        print("           modes " + "  ".join(f"{k} {100 * x:.0f}%" for k, x in sorted(modes.items(), key=lambda kv: -kv[1])))
        px = defaultdict(float)
        for d in v:
            for k, x in d["px"].items():
                px[k] += x / len(v)
        print("           walk px/agent " + "  ".join(f"{k} {x:.0f}" for k, x in sorted(px.items(), key=lambda kv: -kv[1]))
              + f"  (total {sum(px.values()):.0f} px = {0.05 * sum(px.values()):.0f} E)")
        at = defaultdict(int)
        for d in v:
            at[d["mode_now"]] += 1
        print("           mode at death " + "  ".join(f"{k} {x}" for k, x in sorted(at.items(), key=lambda kv: -kv[1])))
        zero = sum(1 for d in v if d["eat"] == 0)
        nokid = sum(1 for d in v if d["kids"] == 0)
        print(f"           never ate {zero}  no children {nokid}")


if __name__ == "__main__":
    main()
