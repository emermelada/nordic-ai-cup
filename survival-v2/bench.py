"""Evaluate the Hive on the fast simulator, many seeds in parallel.

    python bench.py --seeds 1-24 --workers 10 [--params '{"pop_cap_early": 30}'] [--ticks 30000] [--out runs/x.json]

Per game: score, survival time, fruit (count, energy, score), predator kills (count, energy, penalty),
starvations, births, population / predators / mean traits at checkpoints, hive decide time per tick, JSON
payload bytes per tick (sampled), and pose tracking error of agents localized in the world frame.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHECKPOINTS = [60, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 2999]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def play(job):
    seed, params, ticks, policy = job
    import numpy as np
    import orjson
    from fastsim import SimulationCore, to_actions
    if policy == "hive":
        Hive = __import__(os.environ.get("HIVE_MODULE", "hive")).Hive
        hive = Hive(params=params, seed=seed)
    else:
        raise ValueError(policy)
    sim = SimulationCore(seed=seed)
    env = sim.env
    if os.environ.get("NO_PREDATORS"):
        env.spawn_predator = lambda *a, **k: None
    actions = []
    dec_t, dec_max = 0.0, 0.0
    payload, n_payload = 0, 0
    pop_max = 0
    track_err, track_n, world_n, agent_ticks = 0.0, 0, 0, 0
    track_max = 0.0
    cps = {}
    cp_i = 0
    k = 0
    for k in range(ticks):
        state = sim.step(actions)
        n = state["num_agents"]
        pop_max = max(pop_max, n)
        if cp_i < len(CHECKPOINTS) and env.time >= CHECKPOINTS[cp_i]:
            ags = env.agents
            cps[CHECKPOINTS[cp_i]] = {
                "pop": n, "preds": len(env.predators), "trees": len(env.trees), "fruits": len(env.fruits),
                "speed": round(statistics.fmean(a.speed for a in ags), 2) if ags else 0,
                "speed_max": round(max((a.speed for a in ags), default=0), 2),
                "hear": round(statistics.fmean(a.hearing_radius for a in ags), 1) if ags else 0,
                "vis": round(statistics.fmean(a.vision_radius for a in ags), 1) if ags else 0,
                "cone": round(statistics.fmean(a.cone_angle for a in ags), 3) if ags else 0,
                "maxe": round(statistics.fmean(a.max_energy for a in ags), 1) if ags else 0,
                "energy": round(statistics.fmean(a.energy for a in ags), 1) if ags else 0,
            }
            cp_i += 1
        if n == 0 or env.time > 3000:
            break
        step = {"game_status": "ok", "score": state["score"], "sim_time": state["sim_time"], "n_agents": n,
                "agent_status": state["observations"]}
        if k % 50 == 0:
            payload += len(orjson.dumps(step, option=orjson.OPT_SERIALIZE_NUMPY))
            n_payload += 1
        t0 = time.perf_counter()
        acts = hive.decide(step)
        dt = time.perf_counter() - t0
        dec_t += dt
        dec_max = max(dec_max, dt)
        actions = to_actions(acts)
        if k % 20 == 0:
            for a in env.agents:
                m = hive.mem.get(a.agent_id)
                if m is None:
                    continue
                agent_ticks += 1
                if m.frame == "W":
                    world_n += 1
                    e = math.hypot(m.x - a.x, m.y - a.y)
                    track_err += e
                    track_max = max(track_max, e)
                    track_n += 1
    st = env.stat
    return {
        "seed": seed, "score": round(env.score, 3), "time": round(env.time, 1), "ticks": k + 1,
        "fruit_n": st["fruit_n"], "fruit_score": round(st["fruit_e"] / 1000, 3),
        "fruit_mean_e": round(st["fruit_e"] / max(st["fruit_n"], 1), 1),
        "kill_n": st["kill_n"], "kill_pen": round(st["kill_e"] / 100, 3), "starve_n": st["starve_n"],
        "births": env._next_agent_id - 5, "pop_max": pop_max,
        "decide_ms": round(1000 * dec_t / max(k, 1), 3), "decide_max_ms": round(1000 * dec_max, 1),
        "payload_kb": round(payload / max(n_payload, 1) / 1024, 1),
        "world_frac": round(world_n / max(agent_ticks, 1), 3),
        "track_err": round(track_err / max(track_n, 1), 3), "track_max": round(track_max, 1),
        "cps": cps, "hive": dict(getattr(hive, "stats", {})),
        "ledger": {k: round(v, 1) for k, v in st.items()},
    }


def summarize(rows):
    def mean(key):
        return statistics.fmean(r[key] for r in rows)

    scores = [r["score"] for r in rows]
    n = len(rows)
    se = statistics.stdev(scores) / math.sqrt(n) if n > 1 else 0.0
    full = sum(1 for r in rows if r["time"] >= 3000)
    print(f"games {n}  score {mean('score'):.1f} +- {se:.1f}  min {min(scores):.1f}  full-survival {full}/{n}  "
          f"time {mean('time'):.0f}")
    print(f"fruit {mean('fruit_n'):.0f} ({mean('fruit_score'):.1f} pts, {mean('fruit_mean_e'):.1f} e/fruit)  "
          f"kills {mean('kill_n'):.1f} (-{mean('kill_pen'):.1f})  starve {mean('starve_n'):.1f}  "
          f"births {mean('births'):.0f}  pop_max {mean('pop_max'):.0f}")
    print(f"decide {mean('decide_ms'):.2f} ms/tick (max {max(r['decide_max_ms'] for r in rows):.0f})  "
          f"payload {mean('payload_kb'):.1f} KB  world {mean('world_frac'):.2f}  "
          f"track err {mean('track_err'):.2f} (max {max(r['track_max'] for r in rows):.0f})")
    keys = ["fruit_e", "move_e", "turn_e", "live_e", "old_e", "spawn_e", "kill_e", "starve_n", "old_death_n"]
    print("ledger " + "  ".join(f"{k} {statistics.fmean(r['ledger'][k] for r in rows):.0f}" for k in keys))
    modes = sorted({k for r in rows for k in r["hive"].get("mode_ticks", {})})
    print("modes " + "  ".join(f"{k}: ticks {statistics.fmean(r['hive'].get('mode_ticks', {}).get(k, 0) for r in rows):.0f} "
                               f"dist {statistics.fmean(r['hive'].get('mode_dist', {}).get(k, 0) for r in rows):.0f}" for k in modes))
    for cp in CHECKPOINTS:
        vals = [r["cps"].get(cp) or r["cps"].get(str(cp)) for r in rows]
        vals = [v for v in vals if v]
        if not vals:
            continue
        alive = sum(1 for v in vals if v["pop"] > 0)
        f = lambda k: statistics.fmean(v[k] for v in vals if v["pop"] > 0) if alive else 0.0
        print(f"  t={cp:5d} alive {alive:2d}/{n} pop {f('pop'):5.1f} preds {f('preds'):4.1f} trees {f('trees'):5.1f} "
              f"speed {f('speed'):5.2f} (max {f('speed_max'):5.2f}) hear {f('hear'):5.1f} vis {f('vis'):5.1f} "
              f"cone {f('cone'):.2f} maxE {f('maxe'):6.1f} E {f('energy'):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--params", default="")
    ap.add_argument("--ticks", type=int, default=30000)
    ap.add_argument("--policy", default="hive")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    params = json.loads(args.params) if args.params else None
    seeds = parse_seeds(args.seeds)
    jobs = [(s, params, args.ticks, args.policy) for s in seeds]
    t0 = time.time()
    if args.workers <= 1:
        rows = [play(j) for j in jobs]
    else:
        with Pool(args.workers) as pool:
            rows = pool.map(play, jobs, chunksize=1)
    rows.sort(key=lambda r: r["seed"])
    for r in rows:
        print(f"seed {r['seed']:4d} score {r['score']:8.1f} time {r['time']:6.0f} fruit {r['fruit_n']:4d} "
              f"kills {r['kill_n']:3d} (-{r['kill_pen']:.1f}) births {r['births']:4d} pop_max {r['pop_max']:3d} "
              f"dec {r['decide_ms']:.2f}ms pay {r['payload_kb']:.0f}KB trk {r['track_err']:.2f}/{r['track_max']:.0f}")
    summarize(rows)
    print(f"wall {time.time() - t0:.0f}s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"params": params, "rows": rows}, f)


if __name__ == "__main__":
    main()
