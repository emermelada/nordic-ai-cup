"""How much energy goes into moves that do not happen (stuck on walls), and into births that leave the parent
nearly empty?   python stuckdiag.py 101-110
"""
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PEN = {"forest": 1.0, "grassland": 1.0, "swamp": 0.5, "desert": 0.8, "river": 0.3}


def run(seed):
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(seed=seed)
    actions = []
    prev = {}
    stuck_e = defaultdict(float)     # mode -> energy paid for moves that went nowhere
    stuck_ticks = defaultdict(int)
    move_e = 0.0
    births = []                      # (parent energy after, parent old flag, parent true old, age, t)
    false_old = 0
    true_old_flags = 0
    cmd = {}
    win = {}                          # aid -> [start x, start y, commanded px, energy, ticks, mode counts]
    osc_e = defaultdict(float)
    for k in range(30000):
        st = sim.step(actions)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        for a in env.agents:
            c = cmd.get(a.agent_id)
            if c is not None:
                dist, mode, x0, y0, pen = c
                moved = math.hypot(a.x - x0, a.y - y0)
                cost = min(dist, a.speed) * 0.05
                move_e += cost
                if dist > 3.0 and moved < 0.25 * dist * pen:
                    stuck_e[mode] += cost
                    stuck_ticks[mode] += 1
                w = win.get(a.agent_id)
                if w is None:
                    w = win[a.agent_id] = [x0, y0, 0.0, 0.0, 0, defaultdict(int)]
                w[2] += dist * pen
                w[3] += cost
                w[4] += 1
                w[5][mode] += 1
                if w[4] >= 20:
                    net = math.hypot(a.x - w[0], a.y - w[1])
                    if w[2] > 100.0 and net < 0.2 * w[2]:
                        md = max(w[5].items(), key=lambda kv: kv[1])[0]
                        osc_e[md] += w[3]
                    del win[a.agent_id]
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        cmd = {}
        byid = {a.agent_id: a for a in env.agents}
        for act in acts:
            a = byid.get(act["agent_id"])
            m = hv.mem.get(act["agent_id"])
            if a is None or m is None:
                continue
            ix = min(max(int(a.x), 0), env.width - 1)
            iy = min(max(int(a.y), 0), env.height - 1)
            pen = PEN.get(env.biome_map[ix, iy].type, 1.0)
            cmd[a.agent_id] = (float(act["move_distance"]), m.mode, a.x, a.y, pen)
            if act.get("spawn_agent") and a.energy > 100:
                births.append((a.energy - 100, m.old, a.age > a.max_age, a.age, env.time))
                if m.old and a.age <= a.max_age:
                    false_old += 1
                if m.old:
                    true_old_flags += 1
        actions = to_actions(acts)
    return {"seed": seed, "time": env.time, "stuck_e": dict(stuck_e), "osc_e": dict(osc_e), "stuck_ticks": dict(stuck_ticks),
            "move_e": move_e, "births": births, "false_old": false_old, "old_births": true_old_flags}


def main():
    a, b = sys.argv[1].split("-")
    with Pool(10) as pool:
        rows = pool.map(run, range(int(a), int(b) + 1), chunksize=1)
    n = len(rows)
    tot = defaultdict(float)
    tt = defaultdict(int)
    for r in rows:
        for k, v in r["stuck_e"].items():
            tot[k] += v / n
        for k, v in r["stuck_ticks"].items():
            tt[k] += v / n
    me = sum(r["move_e"] for r in rows) / n
    print(f"games {n}  mean time {sum(r['time'] for r in rows) / n:.0f}  walking energy/game {me:.0f}  "
          f"stuck energy/game {sum(tot.values()):.0f} ({100 * sum(tot.values()) / me:.1f}%)")
    print("  by mode: " + "  ".join(f"{k} {v:.0f} E / {tt[k]:.0f} ticks" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))
    osc = defaultdict(float)
    for r in rows:
        for k2, v in r["osc_e"].items():
            osc[k2] += v / n
    print(f"no-progress windows (20 ticks, >100 px commanded, net < 20%): {sum(osc.values()):.0f} E/game "
          f"({100 * sum(osc.values()) / me:.1f}% of walking)  " + "  ".join(f"{k2} {v:.0f}" for k2, v in sorted(osc.items(), key=lambda kv: -kv[1])))
    B = [x for r in rows for x in r["births"]]
    left = sorted(x[0] for x in B)
    print(f"births {len(B) / n:.0f}/game; parent energy left: p10 {left[len(left) // 10]:.0f}  median {left[len(left) // 2]:.0f}; "
          f"left < 40: {100 * sum(1 for x in left if x < 40) / len(left):.0f}%")
    old_b = [x for x in B if x[1]]
    print(f"births by agents flagged old: {len(old_b) / n:.0f}/game, of which not really old: "
          f"{sum(r['false_old'] for r in rows) / n:.1f}/game")
    early = [x for x in B if not x[1] and x[0] < 40]
    print(f"births leaving < 40 by agents not flagged old: {len(early) / n:.1f}/game")


if __name__ == "__main__":
    main()
