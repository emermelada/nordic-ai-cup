#!/usr/bin/env python3
"""mech_ab.py - ONE-mechanism paired A/B against the deployed incumbent. See PREREG_ESEARCH.md.

Plumbing is copied from ml_loop.run_one, the harness that has been verified end to end:
same SimulationCore, same seeding (random/np seeded with the seed), bc.reset_memory() before the
episode, and maxtasksperchild=1 so every episode runs in a fresh process (no cross-episode state).
No nets and no learning: this measures one parameterised structural change on paired seeds, so a
single run is interpretable.

Receipt: the mean commanded move_distance on BLIND ticks (no fruit visible) is recorded per arm, so
"the mechanism did nothing" is distinguishable from "the mechanism ran and survival did not move".

    python3 mech_ab.py --seeds 300000-300159 --horizon 18000 --workers 60 --out /opt/nac_h2h/mech_es
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# PRE-REGISTERED ARMS. base = the deployed controller; the other two are the same mechanism at two doses.
ARMS = {
    "base": {},
    "esearch1": {"esearch": 1.0},
    "esearch1_hi030": {"esearch": 1.0, "esearch_hi": 0.30},
}


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_one(job):
    """One episode, one arm, one policy. Returns ((arm, seed), ticks, receipt-or-error)."""
    arm, seed, horizon, params_path, overrides, policy = job
    try:
        import numpy as np
        from src.core import SimulationCore
        from src.utils.DTOs import ActionRequest

        random.seed(seed)
        np.random.seed(seed)

        if policy == "hive":
            # survival-v2 branch controller (hive.py), driven with the OFFICIAL payload shape:
            # agent_status = the per-agent state dicts, sim_time = ticks/10. Copied in read-only.
            from hive_v2 import Hive
            pol = Hive(params=(overrides or None), seed=seed)
        else:
            import best_controller as bc
            P = dict(bc.DEFAULT_PARAMS)
            with open(params_path) as f:
                P.update(json.load(f))
            P.update(overrides)
            bc.reset_memory()
            pol = bc.make_policy(P)

        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        n_blind = 0
        sum_blind = 0.0
        n_all = 0
        sum_all = 0.0
        i = 0
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                break
            states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
            if not states:
                break
            if policy == "hive":
                out = pol.decide({"agent_status": states, "sim_time": i / 10.0,
                                  "n_agents": len(states)})
                by = {a["agent_id"]: a for a in (out or [])
                      if isinstance(a, dict) and "agent_id" in a}
                byid = {s["agent_id"]: s for s in states}
                acted = []
                for aid, a in by.items():
                    s = byid.get(aid)
                    if s is None:
                        continue
                    nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
                    d = float(a.get("move_distance", 0.0) or 0.0)
                    n_all += 1
                    sum_all += d
                    if nf == 0:
                        n_blind += 1
                        sum_blind += d
                    acted.append((aid, ActionRequest(agent_id=aid, move_distance=d,
                                                     move_direction=float(a.get("move_direction", 0.0) or 0.0),
                                                     turn_angle=float(a.get("turn_angle", 0.0) or 0.0),
                                                     spawn_agent=bool(a.get("spawn_agent", False)))))
            else:
                acted = []
                for s in states:
                    dist, dr, turn, spawn = pol(s)
                    nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
                    n_all += 1
                    sum_all += float(dist)
                    if nf == 0:
                        n_blind += 1
                        sum_blind += float(dist)
                    acted.append((s["agent_id"],
                                  ActionRequest(agent_id=s["agent_id"], move_distance=float(dist),
                                                move_direction=float(dr), turn_angle=float(turn),
                                                spawn_agent=bool(spawn))))
            core.step(acted)
        return (arm, seed), i + 1, {
            "blind_ticks": n_blind,
            "blind_dist_mean": (sum_blind / n_blind) if n_blind else None,
            "all_dist_mean": sum_all / max(1, n_all),
        }
    except Exception as e:                                  # never let one episode stop the run
        return (arm, seed), None, f"ERR:{type(e).__name__}:{e}"


def summarize(ticks, receipts, seeds, arms):
    base = ticks.get("base", {})
    rows = []
    for a in arms:
        d = ticks.get(a, {})
        common = [s for s in seeds if s in d and s in base]
        if not common:
            rows.append({"arm": a, "n": 0})
            continue
        diffs = [d[s] - base[s] for s in common]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
        se = (var / n) ** 0.5
        vals = sorted(d[s] for s in common)
        bvals = sorted(base[s] for s in common)

        def pc(v, q):
            return v[min(len(v) - 1, int(q * len(v)))]
        bd = [receipts[(a, s)]["blind_dist_mean"] for s in common
              if receipts.get((a, s)) and receipts[(a, s)]["blind_dist_mean"] is not None]
        rows.append({
            "arm": a, "n": n,
            "mean": sum(vals) / n, "median": pc(vals, 0.5), "p10": pc(vals, 0.1), "p05": pc(vals, 0.05),
            "min": vals[0],
            "paired": mean, "se": se, "t": (mean / se) if se else 0.0,
            "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
            "W": sum(1 for x in diffs if x > 0), "L": sum(1 for x in diffs if x < 0),
            "base_mean": sum(bvals) / n, "base_p10": pc(bvals, 0.1),
            "blind_dist_mean": (sum(bd) / len(bd)) if bd else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="300000-300159")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "mech_es"))
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--arms-json", default="",
                    help='JSON dict overriding ARMS, e.g. \'{"bl40":{"blind_explore_frac":0.40}}\'')
    ap.add_argument("--policy", default="heuristic", choices=["heuristic", "hive"],
                    help="which controller to drive: the deployed best_controller, or the "
                         "survival-v2 branch controller (hive_v2.py, official payload shape)")
    a = ap.parse_args()

    if a.arms_json:
        ARMS.clear()
        ARMS.update(json.loads(a.arms_json))
        if "base" not in ARMS:
            ARMS["base"] = {}

    arms = [x for x in a.arms.split(",") if x]
    seeds = parse_seeds(a.seeds)
    os.makedirs(a.out, exist_ok=True)
    ledger = os.path.join(a.out, "ledger.jsonl")

    print(f"mech_ab | policy {a.policy} | arms {arms} | {len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f"| horizon {a.horizon} | workers {a.workers}", flush=True)
    dpl = json.load(open(a.params))
    print(f"  baseline params: {len(dpl)} keys from {a.params}", flush=True)
    for name in arms:
        print(f"  arm {name}: overrides {ARMS[name]}", flush=True)

    ticks, receipts = {}, {}
    todo = []
    done_keys = set()
    if os.path.exists(ledger):                       # crash-safe resume
        for line in open(ledger):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ticks") is None:
                continue
            ticks.setdefault(r["arm"], {})[r["seed"]] = r["ticks"]
            receipts[(r["arm"], r["seed"])] = {"blind_dist_mean": r.get("blind_dist_mean")}
            done_keys.add((r["arm"], r["seed"]))
    for name in arms:
        for s in seeds:
            if (name, s) not in done_keys:
                todo.append((name, s, a.horizon, a.params, ARMS[name], a.policy))
    print(f"  resuming: {len(done_keys)} episodes already in the ledger, {len(todo)} to run", flush=True)

    from multiprocessing import get_context
    ctx = get_context("spawn")
    t0 = time.time()
    if todo:
        with open(ledger, "a") as f, ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            done = 0
            for key, t, rec in pool.imap_unordered(run_one, todo, chunksize=1):
                arm, seed = key
                row = {"arm": arm, "seed": seed, "ticks": t}
                if isinstance(rec, str):
                    row["error"] = rec
                else:
                    row.update(rec)
                    receipts[key] = rec
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
                if t is not None:
                    ticks.setdefault(arm, {})[seed] = t
                done += 1
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"    {done}/{len(todo)} episodes  {done/el:.2f}/s  eta {max(0,len(todo)-done)/(done/el):.0f}s",
                          flush=True)
    else:
        for name in arms:
            for s in seeds:
                ticks.setdefault(name, {})[s] = ticks.get(name, {}).get(s)

    rows = summarize(ticks, receipts, seeds, arms)
    print(f"\n  {'arm':<16} {'n':>4} {'mean':>7} {'med':>7} {'p10':>7} {'p05':>7} {'min':>7} "
          f"{'paired':>8} {'+-se':>6} {'t':>6} {'95% CI':>18} {'W/L':>9} {'blind_dist':>10}")
    for r in rows:
        if not r.get("n"):
            print(f"  {r['arm']:<16} NO DATA")
            continue
        ci = f"[{r['ci_lo']:+.0f}, {r['ci_hi']:+.0f}]"
        bd = f"{r['blind_dist_mean']:.2f}" if r["blind_dist_mean"] is not None else "-"
        print(f"  {r['arm']:<16} {r['n']:>4} {r['mean']:>7.0f} {r['median']:>7.0f} {r['p10']:>7.0f} "
              f"{r['p05']:>7.0f} {r['min']:>7.0f} {r['paired']:>+8.0f} {r['se']:>6.0f} {r['t']:>6.2f} "
              f"{ci:>18} {str(r['W'])+'/'+str(r['L']):>9} {bd:>10}")
    print(f"\n  BASE mean {rows[0]['base_mean']:.0f} ticks, p10 {rows[0]['base_p10']:.0f}", flush=True)
    json.dump({"seeds": seeds, "horizon": a.horizon, "rows": rows},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"  wrote {os.path.join(a.out, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
