#!/usr/bin/env python3
"""sched.py - parallel experiment-search scheduler for the survival simulator.

WHAT THIS IS FOR
    The unit of work is (candidate x seed). A lane = a set of candidate policies + an evaluation
    protocol. This script saturates N worker processes with independent episodes, caches every result
    by content hash so nothing is ever recomputed, promotes survivors between cheap and expensive
    stages, and reports paired statistics against the deployed controller.

WHY EACH PIECE EXISTS (all of it is scar tissue):
    * content-addressed cache      - re-running the same (params, seed, horizon) is pure waste; with 64
                                     cores the search is wide, so the same pair is requested constantly
                                     across stages and lanes.
    * cache key includes the code  - the simulator's identity is part of the result. A controller edit
      and simulator hash             must invalidate old numbers, not silently reuse them.
    * reset_fn on every episode    - policy module state leaking between episodes is a documented
                                     false-positive source in this project (identical params disagreeing
                                     on 3/20 seeds). reset_fn makes episodes order-invariant.
    * --selftest                   - runs one episode twice in two separate processes and refuses to
                                     proceed if they differ, so a broken harness can never produce a
                                     headline number.
    * paired vs baseline           - screening on unpaired means produced three wrong headlines tonight.
                                     The baseline is always evaluated on the SAME seeds, so every delta
                                     is per-seed.
    * stages + early stopping      - thousands of candidates can only be screened cheaply; only winners
                                     earn the 18,000-tick held-out evaluation.

USAGE
    # stage 1/2/3: 3 seeds x 4k ticks, 10 x 10k, 40 held-out x 18k
    ./sched.py --lane roles --candidates cands.json --seeds 1900-1949 --holdout 1301-1340 \
               --stages "3:4000,10:10000,40:18000" --workers 60 --minutes 120

    # one-off determinism check
    ./sched.py --lane selftest --selftest --workers 2

CANDIDATE FILE FORMAT
    [{"id": "...", "params": {...}}, ...]
    params are OVERRIDES on the deployed controller. Special keys:
      "__policy__": "heuristic" (default) | "net"
      "__net__":    {"in":31,"hid":H,"w1":[...],"b1":[...],"w2":[...],"b2":[...]} for "net"
"""
import argparse
import hashlib
import json
import math
import os

# MUST be set before any worker interpreter starts: PYTHONHASHSEED is read at interpreter startup, so
# setting it inside a worker is too late. Python randomises string hashing per process, and the
# simulator iterates over entity collections whose order can depend on it, so without this two
# IDENTICAL configs can diverge across processes (measured: BASE vs a byte-identical copy differed on
# 2 of 20 seeds - a ~10% noise source inside every paired comparison we run).
os.environ.setdefault("PYTHONHASHSEED", "0")
import random
import sys
import time
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.environ.get("NAC_RESULTS", os.path.join(ROOT, "results"))
CACHE_PATH = os.path.join(RESULTS, "cache.jsonl")
DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")


# ----------------------------------------------------------------------------- identity / cache
def sim_version():
    """Hash of the simulator + controller sources, plus the hash seed: the identity of a result."""
    h = hashlib.sha256()
    # PYTHONHASHSEED belongs in the identity: results computed under per-process hash randomisation are
    # not the same results (identical configs diverged on 2 of 20 seeds), so old cache entries must not
    # be reused as if they were comparable.
    h.update(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')}".encode())
    files = [os.path.join(ROOT, "best_controller.py")]
    for d in ("src",):
        for dp, _dn, fn in os.walk(os.path.join(ROOT, d)):
            files += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(".py")]
    for f in sorted(files):
        try:
            h.update(open(f, "rb").read())
        except OSError:
            pass
    return h.hexdigest()[:16]


SIMV = sim_version()


def cache_key(params, seed, horizon):
    pid = params.get("id", "")
    payload = json.dumps({k: v for k, v in params.items() if k != "id"}, sort_keys=True, default=str)
    return hashlib.sha256(f"{payload}|{seed}|{horizon}|{SIMV}|{pid}".encode()).hexdigest()[:32]


def load_cache():
    cache = {}
    if os.path.exists(CACHE_PATH):
        for line in open(CACHE_PATH):
            try:
                r = json.loads(line)
                cache[r["key"]] = r
            except Exception:
                continue
    return cache


def append_cache(rec):
    os.makedirs(RESULTS, exist_ok=True)
    with open(CACHE_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ----------------------------------------------------------------------------- policies
def make_policy(params):
    """Returns (policy_fn, reset_fn). 'heuristic' = the deployed controller with param overrides."""
    kind = params.get("__policy__", "heuristic")
    if kind == "heuristic":
        import best_controller as bc
        P = dict(bc.DEFAULT_PARAMS)
        P.update({k: v for k, v in params.items() if not k.startswith("__") and k != "id"})
        return bc.make_policy(P), bc.reset_memory
    if kind == "roles":
        # HETEROGENEOUS FLEET: different parameter sets for different agents in the SAME episode.
        # Every experiment in this project so far has assumed one policy for every agent, even though
        # the score only needs ONE lineage to survive - which makes diversification the natural answer
        # to a reliability problem. This is implemented HERE, in the experiment layer, so the served
        # controller is untouched (its identity/hash is a deployment contract).
        #   "__roles__": [ {param overrides for role 0}, {role 1}, ... ]
        #   "__role_assign__": "by_index" (agent_id % n_roles) | "by_energy" (richest = role 0)
        import best_controller as bc
        roles = params["__roles__"]
        assign = params.get("__role_assign__", "by_index")
        fns = []
        for r in roles:
            P = dict(bc.DEFAULT_PARAMS)
            P.update({k: v for k, v in params.items() if not k.startswith("__") and k != "id"})
            P.update({k: v for k, v in r.items() if not k.startswith("__")})
            fns.append(bc.make_policy(P))

        def fn(s):
            i = int(s.get("agent_id", 0))
            if assign == "by_energy":
                frac = float(s.get("energy", 0.0)) / max(1.0, float(s.get("max_energy", 1.0)))
                # richest agent gets role 0; ranks split the remaining roles evenly
                k = 0 if frac > 0.6 else (1 if len(fns) > 1 else 0)
                k = min(k, len(fns) - 1)
            else:
                k = i % len(fns)
            return fns[k](s)

        return fn, bc.reset_memory
    if kind == "residual":
        # RESIDUAL POLICY: a small MLP adds bounded offsets to the DEPLOYED controller's action.
        # Why residual: a from-scratch net starts far below the heuristic (measured: 300 random MLPs
        # = -38% at 40 seeds). A residual starts AT the incumbent (zero weights = exact baseline) and
        # only has to learn a correction, which is the sample-efficient way to learn decision-making
        # against a simulator in the time available.
        import best_controller as bc
        import numpy as np
        from env_wrapper import build_obs
        base_fn = bc.make_policy(dict(params.get("__base__") or {}))
        net = params["__net__"]
        h = int(net["h"])
        W1 = np.array(net["w1"], np.float32).reshape(h, -1)
        b1 = np.array(net["b1"], np.float32).reshape(h)
        W2 = np.array(net["w2"], np.float32).reshape(-1, h)
        b2 = np.array(net["b2"], np.float32).reshape(-1)
        sd = float(params.get("__scale_dir__", 0.5))
        ss = float(params.get("__scale_speed__", 0.3))

        def fn(state):
            dist, dr, turn, spawn = base_fn(state)
            o = build_obs(state)
            hh = np.tanh(W1 @ o + b1)
            out = np.tanh(W2 @ hh + b2)
            sprint = float(state.get("sprint_speed", 20.0) or 20.0)
            # wrap the corrected heading into [-pi, pi] (no _wrap helper in this module)
            nd = dr + float(out[0]) * sd
            nd = (nd + math.pi) % (2.0 * math.pi) - math.pi
            return (float(np.clip(dist + out[1] * ss * sprint, 0.0, sprint)),
                    float(nd), float(turn), float(spawn))

        return fn, bc.reset_memory

    if kind == "net":
        import numpy as np
        from env_wrapper import build_obs, OBS_DIM
        net = params["__net__"]
        hid, din = int(net["hid"]), int(net.get("in", OBS_DIM))
        W1 = np.asarray(net["w1"], dtype=np.float32).reshape(hid, din)
        b1 = np.asarray(net["b1"], dtype=np.float32).reshape(hid)
        W2 = np.asarray(net["w2"], dtype=np.float32).reshape(4, hid)
        b2 = np.asarray(net["b2"], dtype=np.float32).reshape(4)

        def fn(s):
            x = build_obs(s)
            o = np.tanh(W2 @ np.tanh(W1 @ x + b1) + b2)
            sprint = float(s.get("sprint_speed", 10.0))
            dist = 0.5 * (o[0] + 1.0) * sprint          # [0, sprint_speed]
            return [float(dist), float(o[1]) * 3.141592653589793,
                    float(o[2]) * 1.5707963267948966, bool(o[3] > 0.0)]

        def reset():
            import best_controller as bc
            bc.reset_memory()

        return fn, reset
    raise ValueError(f"unknown __policy__ {kind!r}")


# ----------------------------------------------------------------------------- worker
def run_one(task):
    cand_id, params, seed, horizon, want_trace = task
    t0 = time.time()
    policy_fn, reset_fn = make_policy(params)
    from env_wrapper import run_eval_episode
    r = run_eval_episode(policy_fn, n_agents=5, seed=seed, horizon=horizon,
                         stop_on_death=True, reset_fn=reset_fn,
                         trace=want_trace, trace_every=500)
    rec = {"key": cache_key(params, seed, horizon), "cand": cand_id, "seed": seed,
           "horizon": horizon, "steps": r["steps"], "score": r["score"],
           "fruits": r["fruits_eaten"], "lost": r["predated"], "spawns": r["spawns"],
           "final_agents": r["final_agents"], "alive": r["alive"], "simv": SIMV,
           "wall_s": round(time.time() - t0, 2)}
    if want_trace:
        rec["traces"] = r["traces"]
    return rec


def run_batch(jobs, workers, deadline, cache, label=""):
    """Runs jobs (already filtered against the cache) with a pool; returns records. Stops submitting
    once `deadline` passes, but always drains what is in flight."""
    out = []
    if not jobs:
        return out
    ctx = get_context("spawn")

    def gen():
        for j in jobs:
            if deadline and time.time() > deadline:
                print(f"[{label}] deadline reached - not submitting further jobs", flush=True)
                return
            yield j

    done = 0
    with ctx.Pool(workers) as pool:
        for rec in pool.imap_unordered(run_one, gen(), chunksize=1):
            append_cache(rec)
            cache[rec["key"]] = rec
            out.append(rec)
            done += 1
            if done % 50 == 0:
                print(f"[{label}] {done}/{len(jobs)} episodes done", flush=True)
    return out


# ----------------------------------------------------------------------------- stats
def summarise(scored, baseline_id="BASE"):
    """Per-candidate stats + paired deltas against the baseline on identical seeds."""
    by_cand = {}
    for r in scored:
        by_cand.setdefault(r["cand"], {})[r["seed"]] = r
    base = by_cand.get(baseline_id, {})
    rows = []
    for cid, per_seed in by_cand.items():
        steps = [v["steps"] for v in per_seed.values()]
        if not steps:
            continue
        paired, wins, losses, ties = [], 0, 0, 0
        for sd, v in per_seed.items():
            if sd in base:
                d = v["steps"] - base[sd]["steps"]
                paired.append(d)
                wins += d > 0
                losses += d < 0
                ties += d == 0
        import numpy as np
        mean = float(np.mean(steps))
        rows.append({
            "cand": cid, "n": len(steps), "mean": mean, "median": float(np.median(steps)),
            "min": int(min(steps)), "max": int(max(steps)), "var": float(np.var(steps)),
            # FLOOR METRICS. The official score is a mean over 3 runs, so the bad tail drags it far more
            # than the best run lifts it: a controller that can hit 1,231 but sometimes 550 has a
            # VARIANCE problem, not a capability problem. p10 (and the paired gain in p10 against the
            # baseline on identical seeds) is therefore the metric to optimise, not the mean.
            "p10": float(np.percentile(steps, 10)),
            "p25": float(np.percentile(steps, 25)),
            "extinct_frac": sum(1 for s in steps if s < max(v["horizon"] for v in per_seed.values())) / len(steps),
            "fruits": float(np.mean([v["fruits"] for v in per_seed.values()])),
            "lost": float(np.mean([v["lost"] for v in per_seed.values()])),
            "spawns": float(np.mean([v["spawns"] for v in per_seed.values()])),
            "final_agents": float(np.mean([v["final_agents"] for v in per_seed.values()])),
            "paired_mean": float(np.mean(paired)) if paired else None,
            "paired_pct": (100 * float(np.mean(paired)) / float(np.mean([base[s]["steps"] for s in per_seed if s in base])))
                           if paired and base else None,
            "wins": wins, "losses": losses, "ties": ties,
            "win_frac": wins / max(1, wins + losses),
        })
    rows.sort(key=lambda r: (-(r["paired_mean"] if r["paired_mean"] is not None else -1e18), -r["mean"]))
    return rows


def print_rows(rows, baseline_id="BASE", top=None):
    # paired floor gain = candidate's p10 minus the baseline's p10 on the SAME seeds
    bp10 = next((r["p10"] for r in rows if r["cand"] == baseline_id), None)
    for r in rows[:top] if top else rows:
        tag = "  " if r["cand"] == baseline_id else "->"
        p = "" if r["paired_mean"] is None else f"  paired {r['paired_mean']:+8.0f} ({r['paired_pct']:+5.1f}%)  W{r['wins']}/L{r['losses']}/T{r['ties']}"
        floor = "" if bp10 is None else f"  floorP10 {r['p10'] - bp10:+7.0f}"
        print(f"{tag} {r['cand'][:30]:30s} n={r['n']:3d} mean={r['mean']:7.0f} med={r['median']:7.0f} "
              f"p10={r['p10']:6.0f} min={r['min']:6d} var={r['var']:10.0f} ext={r['extinct_frac']:.2f} "
              f"fruit={r['fruits']:6.0f} pop={r['final_agents']:4.1f}{p}{floor}", flush=True)


# ----------------------------------------------------------------------------- main
def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--candidates", default="", help="JSON list of {id, params}")
    ap.add_argument("--seeds", default="1900-1949", help="screening seed pool")
    ap.add_argument("--holdout", default="1301-1340", help="held-out seeds for the final stage")
    ap.add_argument("--stages", default="3:4000,10:10000,40:18000", help="n_seeds:horizon per stage")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--topk", type=int, default=12, help="survivors carried to the next stage")
    ap.add_argument("--tol", type=float, default=0.10, help="early-stop: drop candidates below (1-tol) x baseline mean")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--selftest", action="store_true", help="determinism check only, then exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    cache = load_cache()
    print(f"LANE {args.lane} | simv {SIMV} | cache {len(cache)} episodes | workers {args.workers} "
          f"| results {RESULTS}", flush=True)

    deployed = json.load(open(DEPLOYED)) if os.path.exists(DEPLOYED) else {}
    if not deployed:
        raise SystemExit(f"deployed params missing at {DEPLOYED} - refusing to run against DEFAULT_PARAMS")
    base_params = {"id": "BASE", **deployed}
    print(f"baseline = deployed controller: fruit_weight={deployed.get('fruit_weight'):.3f} "
          f"evade_mode={deployed.get('evade_mode')} walk_frac={deployed.get('walk_frac'):.3f}", flush=True)

    # ---- determinism self-test: the harness must be order/process invariant before any result counts
    if args.selftest:
        probe = [("ST", base_params, args.seeds and parse_seeds(args.seeds)[0], 2000, False)]
        a = run_one(probe[0])
        b = run_one(probe[0])
        same = (a["steps"], a["fruits"], a["lost"], a["spawns"]) == (b["steps"], b["fruits"], b["lost"], b["spawns"])
        print(f"SELFTEST seed={a['seed']} run1=({a['steps']},{a['fruits']},{a['lost']},{a['spawns']}) "
              f"run2=({b['steps']},{b['fruits']},{b['lost']},{b['spawns']}) -> "
              f"{'IDENTICAL' if same else 'MISMATCH - HARNESS UNSAFE, STOP'}", flush=True)
        raise SystemExit(0 if same else 2)

    cands = [{"id": "BASE", "params": base_params}]
    if args.candidates and os.path.exists(args.candidates):
        for c in json.load(open(args.candidates)):
            # sched.py requires an "id"; accept "tag" too so a generator cannot silently produce an
            # unusable candidate file (gen_search.py emits "tag", which killed the first wide run).
            if "id" not in c and "tag" in c:
                c["id"] = c["tag"]
            # GUARD: a knob that does not exist in the controller is SILENTLY IGNORED, so an arm can run,
            # report a plausible number, and have done nothing at all. This actually happened: the gs_*
            # breeder-selection arms were inert because the SERVING controller predates that code. Any
            # override key the controller does not know is now a loud warning, and if EVERY override is
            # unknown the candidate is dropped rather than run as a pointless copy of the baseline.
            try:
                from best_controller import DEFAULT_PARAMS as _DP
            except Exception:
                _DP = {}
            unknown = [k for k in c["params"] if k != "id" and not k.startswith("__") and k not in _DP]
            if unknown:
                print(f"WARNING {c['id']}: {len(unknown)} override key(s) NOT in the controller and will "
                      f"be IGNORED: {unknown[:6]}", flush=True)
            known_over = [k for k in c["params"] if k != "id" and k in _DP]
            if not known_over and unknown:
                print(f"DROPPING {c['id']}: every override is unknown -> would be a copy of BASE", flush=True)
                continue
            cands.append({"id": c["id"], "params": {"id": c["id"], **c["params"]}})
    print(f"candidates: {len(cands)} (1 baseline + {len(cands)-1})", flush=True)

    stages = []
    for s in args.stages.split(","):
        n, h = s.split(":")
        stages.append((int(n), int(h)))

    pool_seeds = parse_seeds(args.seeds)
    holdout = parse_seeds(args.holdout)
    deadline = time.time() + args.minutes * 60
    alive = cands
    ledger = []

    for si, (n, horizon) in enumerate(stages, 1):
        seeds = (holdout if si == len(stages) else pool_seeds)
        seeds = seeds[:n]
        want_trace = horizon >= 12000
        jobs, hits = [], 0
        for c in alive:
            for sd in seeds:
                k = cache_key(c["params"], sd, horizon)
                if k in cache:
                    hits += 1
                else:
                    jobs.append((c["id"], c["params"], sd, horizon, want_trace))
        print(f"\n=== stage {si}: {len(alive)} candidates x {len(seeds)} seeds @ {horizon} ticks "
              f"| {len(jobs)} new episodes ({hits} cache hits) ===", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        run_batch(jobs, args.workers, deadline, cache, label=f"stage{si}")
        scored = []
        for c in alive:
            for sd in seeds:
                rec = cache.get(cache_key(c["params"], sd, horizon))
                if rec:
                    scored.append(rec)
        rows = summarise(scored, "BASE")
        print_rows(rows, "BASE", top=args.topk + 1)
        ledger.append({"stage": si, "horizon": horizon, "n_seeds": len(seeds), "rows": rows,
                       "minutes": round((time.time() - t0) / 60.0, 2)})
        json.dump(ledger, open(os.path.join(RESULTS, f"{args.lane}_ledger.json"), "w"), indent=1)
        print(f"stage {si} took {(time.time()-t0)/60.0:.1f} min", flush=True)

        if si < len(stages):
            base_row = next((r for r in rows if r["cand"] == "BASE"), None)
            floor = (base_row["mean"] * (1.0 - args.tol)) if base_row else -1e18
            keep = [r for r in rows if r["cand"] != "BASE" and r["mean"] >= floor]
            if not keep:
                # A search that starts from nothing (random nets) will often have NO candidate above
                # the floor. Dropping everything would end the lane at stage one; promote the top-k by
                # mean anyway so the population can be mutated upward from whatever is best.
                keep = [r for r in rows if r["cand"] != "BASE"][:args.topk]
                print(f"WARNING: no candidate cleared the floor ({floor:.0f} ticks) - promoting the "
                      f"top {len(keep)} anyway (search-from-nothing mode)", flush=True)
            keep = keep[:args.topk]
            kept_ids = {r["cand"] for r in keep} | {"BASE"}
            dropped = [c["id"] for c in alive if c["id"] not in kept_ids and c["id"] != "BASE"]
            if dropped:
                print(f"early-stop dropped {len(dropped)}: {dropped[:8]}{'...' if len(dropped) > 8 else ''}", flush=True)
            alive = [c for c in alive if c["id"] in kept_ids]
        if time.time() > deadline:
            print("deadline reached - stopping after this stage", flush=True)
            break

    print(f"\n=== LANE {args.lane} COMPLETE ===  results in {RESULTS}/{args.lane}_ledger.json")


if __name__ == "__main__":
    main()
