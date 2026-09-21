"""Where does the food go? Per eaten fruit: the eater's mode, the fruit's energy and age, overflow above max.

    python eatdiag.py 101-110 [params_json]
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(job):
    seed, params = job
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(params=params, seed=seed)
    actions = []
    seen = 0
    by_mode = defaultdict(lambda: [0, 0.0, 0.0])      # mode -> [n, energy, overflow]
    by_phase = defaultdict(lambda: [0, 0.0])
    unripe_modes = defaultdict(int)
    for k in range(30000):
        st = sim.step(actions)
        for (t, aid, e, age, over, fx, fy) in env.eat_log[seen:]:
            m = hv.mem.get(aid)
            mode = m.mode if m is not None else "?"
            if m is not None and m.old:
                mode += "/old"
            if m is not None and mode.startswith("food") and m.frame == "W":
                tg = m.food_tgt
                hit = tg is not None and abs(tg[0] - fx) + abs(tg[1] - fy) < 2.0
                mode += "/tgt" if hit else "/side"
                if hit and age < 15.0:
                    fm = hv.maps.get("W")
                    if fm is not None and fm.fx.size:
                        j = int((abs(fm.fx - fx) + abs(fm.fy - fy)).argmin())
                        dated = (fm.fhi[j] - fm.flo[j]) < 8.0
                        mode += "/dated" if dated else "/undated"
            r = by_mode[mode]
            r[0] += 1
            r[1] += e
            r[2] += over
            ph = int(t // 300) * 300
            by_phase[ph][0] += 1
            by_phase[ph][1] += e
            if age < 15.0:
                unripe_modes[mode] += 1
        seen = len(env.eat_log)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
    return {"seed": seed, "time": env.time, "by_mode": dict(by_mode), "by_phase": dict(by_phase),
            "unripe_modes": dict(unripe_modes)}


def main():
    a, b = sys.argv[1].split("-")
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    with Pool(10) as pool:
        rows = pool.map(run, [(s, params) for s in range(int(a), int(b) + 1)], chunksize=1)
    tot = defaultdict(lambda: [0, 0.0, 0.0])
    ph = defaultdict(lambda: [0, 0.0])
    un = defaultdict(int)
    for r in rows:
        for k, v in r["by_mode"].items():
            for i in range(3):
                tot[k][i] += v[i]
        for k, v in r["by_phase"].items():
            ph[int(k)][0] += v[0]
            ph[int(k)][1] += v[1]
        for k, v in r["unripe_modes"].items():
            un[k] += v
    n_all = sum(v[0] for v in tot.values())
    print(f"games {len(rows)}  mean time {sum(r['time'] for r in rows) / len(rows):.0f}  fruit eaten {n_all}")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:14s} n {v[0]:6d} ({100 * v[0] / n_all:4.1f}%)  mean E {v[1] / max(v[0], 1):5.1f}  "
              f"overflow {v[2] / max(v[0], 1):5.1f}/fruit  eaten <15 s old: {un.get(k, 0)}")
    for k in sorted(ph):
        print(f"  t {k:5d}+  n {ph[k][0]:6d}  mean E {ph[k][1] / max(ph[k][0], 1):5.1f}")


if __name__ == "__main__":
    main()
