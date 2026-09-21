"""Flee episodes: how long, how costly, and was the predator really after us?

    python fleediag.py 101-110   (DIAG_WORKERS env for the pool size)

Per episode (consecutive flee ticks of one agent): duration, px commanded, energy spent, and the true state
of the nearest predator at the start: distance, resting, its energy, and whether this agent is the closest
agent that predator can perceive (a predator only ever chases that one).
"""
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def perceives(p, a, agents):
    """Closest agent the predator perceives (ignoring walls): hearing 60 or 250 px in +-30 deg."""
    best, bd = None, 1e9
    for q in agents:
        d = math.hypot(q.x - p.x, q.y - p.y)
        ang = abs((math.atan2(q.y - p.y, q.x - p.x) - p.direction + math.pi) % (2 * math.pi) - math.pi)
        if d < 60 or (d < 250 and ang <= math.pi / 6):
            if d < bd:
                best, bd = q, d
    return best


def run(seed):
    from fastsim import SimulationCore, to_actions
    from hive import Hive
    sim = SimulationCore(seed=seed)
    env = sim.env
    hv = Hive(seed=seed)
    actions = []
    ep = {}          # aid -> dict
    done = []
    killed = set()
    seen_k = 0
    for k in range(30000):
        st = sim.step(actions)
        for (tk, aid, pid) in env.kill_log[seen_k:]:
            killed.add(aid)
        seen_k = len(env.kill_log)
        if st["num_agents"] == 0 or env.time > 3000:
            break
        acts = hv.decide({"sim_time": st["sim_time"], "agent_status": st["observations"]})
        actions = to_actions(acts)
        byact = {a["agent_id"]: a for a in acts}
        alive_ids = set()
        for a in env.agents:
            alive_ids.add(a.agent_id)
            m = hv.mem.get(a.agent_id)
            if m is None:
                continue
            e = ep.get(a.agent_id)
            if m.mode == "flee":
                if e is None:
                    pr = min(env.predators, key=lambda p: math.hypot(p.x - a.x, p.y - a.y), default=None)
                    chasing = False
                    if pr is not None and not pr.resting:
                        chasing = perceives(pr, a, env.agents) is a
                    e = ep[a.agent_id] = {"t0": env.time, "E0": a.energy, "px": 0.0, "ticks": 0,
                                          "d0": math.hypot(pr.x - a.x, pr.y - a.y) if pr else 9999,
                                          "rest0": pr.resting if pr else True, "pE0": pr.energy if pr else 0,
                                          "chasing": chasing, "late": env.time > 900}
                e["ticks"] += 1
                e["px"] += byact[a.agent_id]["move_distance"] if a.agent_id in byact else 0.0
                e["E1"] = a.energy
            elif e is not None:
                e["outcome"] = "ok"
                done.append(e)
                del ep[a.agent_id]
        for aid in [x for x in ep if x not in alive_ids]:
            e = ep.pop(aid)
            e["outcome"] = "killed" if aid in killed else "starved"
            done.append(e)
    return done


def main():
    a, b = sys.argv[1].split("-")
    with Pool(int(os.environ.get("DIAG_WORKERS", "10"))) as pool:
        res = pool.map(run, range(int(a), int(b) + 1), chunksize=1)
    eps = [e for r in res for e in r]
    for late in (False, True):
        v = [e for e in eps if e["late"] == late]
        if not v:
            continue
        n = len(v)
        print(f"{'late (t>900)' if late else 'early'}: {n} episodes over {len(res)} games")
        groups = defaultdict(list)
        for e in v:
            key = ("resting" if e["rest0"] else ("chasing us" if e["chasing"] else "awake, not after us"))
            groups[key].append(e)
        for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            dur = sorted(x["ticks"] for x in g)
            px = sum(x["px"] for x in g) / len(g)
            en = sum(x["E0"] - x.get("E1", x["E0"]) for x in g) / len(g)
            kil = sum(1 for x in g if x.get("outcome") == "killed")
            print(f"  {key:20s} n {len(g):5d}  ticks median {dur[len(dur) // 2]:3d} p75 {dur[3 * len(dur) // 4]:3d}  "
                  f"px {px:6.0f}  energy {en:5.1f}  killed {kil}  d0 median "
                  f"{sorted(x['d0'] for x in g)[len(g) // 2]:.0f}  pred E0 median {sorted(x['pE0'] for x in g)[len(g) // 2]:.0f}")
        tot_px = sum(x["px"] for x in v)
        print(f"  total flee px {tot_px / len(res):.0f}/game")


if __name__ == "__main__":
    main()
