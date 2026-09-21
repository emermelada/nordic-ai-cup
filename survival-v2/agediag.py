"""How old is a fruit when the hive first maps it, and how good is its [flo, fhi] birth bracket?

    python agediag.py 101-106
"""
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(seed):
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(seed=seed)
    actions = []
    known = set()
    rows = []     # (t, dated, true_age, fhi - flo, true_birth - flo)
    for k in range(30000):
        st = sim.step(actions)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        actions = to_actions(hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]}))
        fm = hv.maps.get("W")
        if fm is None or not fm.fx.size or not env.fruits:
            continue
        TX = np.array([f.x for f in env.fruits])
        TY = np.array([f.y for f in env.fruits])
        TA = np.array([f.age / 2.0 for f in env.fruits])
        TID = [id(f) for f in env.fruits]
        for j in range(fm.fx.size):
            d = np.abs(TX - fm.fx[j]) + np.abs(TY - fm.fy[j])
            i = int(d.argmin())
            if d[i] > 2.0 or TID[i] in known:
                continue
            known.add(TID[i])
            birth = env.time - TA[i]
            rows.append((env.time, (fm.fhi[j] - fm.flo[j]) < 8.0, TA[i], fm.fhi[j] - fm.flo[j], birth - fm.flo[j]))
    return rows


def main():
    a, b = sys.argv[1].split("-")
    with Pool(10) as pool:
        res = pool.map(run, range(int(a), int(b) + 1), chunksize=1)
    rows = [r for rr in res for r in rr]
    for label, sel in (("dated", True), ("undated", False)):
        v = [r for r in rows if r[1] == sel]
        if not v:
            continue
        ages = np.array([r[2] for r in v])
        width = np.array([r[3] for r in v])
        print(f"{label}: n {len(v)}  true age at first map: mean {ages.mean():.1f}  "
              f"p25 {np.percentile(ages, 25):.1f}  median {np.median(ages):.1f}  p75 {np.percentile(ages, 75):.1f}  "
              f"<10 s {100 * (ages < 10).mean():.0f}%  <20 s {100 * (ages < 20).mean():.0f}%  bracket width {width.mean():.1f}")
        hist = np.histogram(ages, bins=[0, 5, 10, 15, 20, 30, 40, 50])[0]
        print("   age histogram 0-5-10-15-20-30-40-50:", hist.tolist())
    by_t = defaultdict(list)
    for r in rows:
        if not r[1]:
            by_t[int(r[0] // 300) * 300].append(r[2])
    for t in sorted(by_t):
        v = np.array(by_t[t])
        print(f"   undated first seen t {t:5d}+: n {len(v):5d}  mean age {v.mean():.1f}  <20 s {100 * (v < 20).mean():.0f}%")


if __name__ == "__main__":
    main()
