#!/usr/bin/env python3
"""ml_loop.py - the unattended ML search: OpenAI-ES over residual GRUs on the deployed controller.

GATES ALREADY PASSED (do not remove; they are why this loop is trustworthy)
  1. rec_resid.py --stage zero : the zero-head residual reproduces BASE BIT-IDENTICALLY, 40/40.
     Candidate #0 is therefore the incumbent exactly, and the loop can never silently replace the
     base policy with an arbitrary learned one.
  2. A/A control (identical policy, two arms): 0/40 seeds differ. The simulator is DETERMINISTIC
     after fixing three address-dependent set iterations (predator eat order, entity-sort ties,
     fruit eat-order ties). A paired difference on fixed seeds is therefore a REAL difference on
     those seeds - which is what makes a cheap screen meaningful at all.

METHODOLOGY THAT MATTERS
  * SCREEN / CONFIRM split. Selection happens only on rotating screening blocks. A held-out block
    is never used for selection. Selecting on a fixed block is how this project previously produced
    champions that reversed sign when re-measured (E3 +9.5% -> +0.4%; C6 +18.1% -> +1.0%).
  * STAGED. Many candidates on few seeds at short horizon; only survivors get 40 seeds at 18k.
    Flat full evaluation would cap throughput at hundreds of candidates/day instead of thousands.
  * CACHED + RESUMABLE. Every episode is keyed (gen, cand, seed, horizon) in ledger.jsonl; a
    restart replays from cache and never recomputes.
  * RESILIENT. A candidate that crashes, times out, or emits a non-finite action is recorded as
    invalid and skipped; the population continues.
  * SHARDABLE. --shard i --nshards n takes a deterministic slice of the population so a second
    machine can run the same generation against the same centre and seeds.

    python3 ml_loop.py --gens 40 --pop 256 --shard 0 --nshards 1
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from probe_enc import build_v2, V2_DIM  # noqa: E402
from rec_resid import make_net, HEAD_SCALE, HID  # noqa: E402

WORK = os.environ.get("NAC_WORK", os.path.join(HERE, "ml"))
LEDGER = os.path.join(WORK, "ledger.jsonl")
CENTRE = os.path.join(WORK, "centre.pt")


# ------------------------------------------------------------------ centre / perturbation
def params_of(net):
    return [p.detach().clone() for p in net.parameters()]


def set_params(net, ps):
    with torch.no_grad():
        for p, q in zip(net.parameters(), ps):
            p.copy_(q)


def eps_seed(gen, idx):
    h = hashlib.sha256(f"{gen}:{idx}".encode()).hexdigest()[:8]
    return int(h, 16)


def save_centre(net, gen, path=CENTRE):
    torch.save({"sd": net.state_dict(), "gen": gen}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_centre(gen=None):
    if gen is None:
        return make_net()
    ck = torch.load(gen, weights_only=True)
    net = make_net()
    net.load_state_dict(ck["sd"])
    return net


# ------------------------------------------------------------------ evaluation
def run_one(job):
    """One episode. Returns (key, ticks, invalid). Never raises into the pool loop."""
    gen, idx, seed, horizon, params_path, weights, corr = job
    try:
        import random
        import best_controller as bc
        from src.core import SimulationCore
        from src.utils.DTOs import ActionRequest
        torch.set_num_threads(1)
        P = dict(bc.DEFAULT_PARAMS)
        with open(params_path) as f:
            P.update(json.load(f))
        random.seed(seed)
        np.random.seed(seed)
        bc.reset_memory()
        base = bc.make_policy(P)
        net = None
        if weights is not None:
            net = make_net()
            with torch.no_grad():
                net.load_state_dict(weights)
            net.eval()
        mem = {}
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        memo, hid = {}, {}
        invalid = 0
        # CORRECTION AUTHORITY MONITOR: how much of the allowed bound the net actually uses. If the
        # max sits just under the bound the net is being clipped and wants more authority; if it
        # sits at a few percent of the bound, the bound is irrelevant - a different finding from
        # "ML cannot improve the controller".
        cmax = np.zeros(3, np.float32)
        csum = np.zeros(3, np.float64)
        cn = 0
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                break
            states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
            acts = []
            for s in states:
                aid = s["agent_id"]
                dist, dr, turn, spawn = base(s)
                if net is not None:
                    m = memo.setdefault(aid, {})
                    energy = float(s.get("energy", 0.0) or 0.0)
                    pe = m.get("e")
                    inc = 0.0 if pe is None else max(0.0, energy - pe) + 0.1
                    m["inc_f"] = 0.9 * m.get("inc_f", 0.0) + 0.1 * inc
                    m["inc_s"] = 0.99 * m.get("inc_s", 0.0) + 0.01 * inc
                    me = max(float(s.get("max_energy", 500.0) or 500.0), 1.0)
                    ef = energy / me
                    m["ef_s"] = 0.99 * m.get("ef_s", ef) + 0.01 * ef
                    m["lock"] = 0.99 * m.get("lock", 0.0) + 0.01 * (1.0 if energy < me / 5.0 else 0.0)
                    nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
                    if nf > 0:
                        m["tf"], m["df"] = 0.0, 0.0
                    else:
                        m["tf"] = m.get("tf", 0.0) + 1.0
                        m["df"] = m.get("df", 0.0) + float(s.get("speed", 10.0) or 10.0) * 0.12
                    m["tick"] = float(bc._SIM_TICK)
                    m["n_agents"] = float(len(states))
                    if bc._GS_LAST is not None:
                        m["rank"] = float(bc._GS_LAST[1])
                    m["since_birth"] = float(i)
                    x = torch.tensor(build_v2(s, m))[None, None, :]
                    with torch.no_grad():
                        o, h = net(x, hid.get(aid))
                    h = torch.clamp(h, -10.0, 10.0)
                    hid[aid] = h
                    # tanh makes corr a TRUE bound (HEAD_SCALE alone was only a scale - the
                    # linear head was unbounded). tanh(0)=0 keeps the zero-change path exact.
                    o = torch.tanh(o[0, 0]).numpy() * corr
                    if not np.isfinite(o).all():
                        invalid += 1
                        o = np.zeros(3, np.float32)
                    cmax = np.maximum(cmax, np.abs(o))
                    csum += np.abs(o)
                    cn += 1
                    sprint = max(1.0, float(s.get("sprint_speed", 20.0) or 20.0))
                    if o[0] != 0.0:
                        dr = dr + float(o[0])
                    if o[1] != 0.0:
                        dist = dist + float(o[1]) * sprint
                    m["e"] = energy
                acts.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist),
                                                move_direction=float(dr), turn_angle=float(turn),
                                                spawn_agent=bool(spawn))))
            core.step(acts)
            keep = {a.agent_id for a in core.env.agents}
            hid = {k: v for k, v in hid.items() if k in keep}
        frac = (csum / max(1, cn)) / np.maximum(corr, 1e-9)
        return ((gen, idx, seed, horizon), i + 1, invalid,
                {"corr_abs_mean_frac": [round(float(x), 4) for x in frac],
                 "corr_abs_max_frac": [round(float(x), 4) for x in (cmax / np.maximum(corr, 1e-9))],
                 "corr_n": cn})
    except Exception as e:                                     # never let one candidate stop the run
        return (gen, idx, seed, horizon), None, f"ERR:{type(e).__name__}", None


def load_ledger():
    c = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                r = json.loads(line)
                c[(r["gen"], r["idx"], r["seed"], r["horizon"])] = r["ticks"]
            except Exception:
                continue
    return c


def append_ledger(rows):
    with open(LEDGER, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


STATS = {}          # (gen, idx, seed, horizon) -> correction-usage stats


def evaluate(pool, cache, jobs, label):
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in cache]
    new = []
    if todo:
        t0 = time.time()
        done = 0
        for key, ticks, invalid, st in pool.imap_unordered(run_one, todo, chunksize=1):
            cache[key] = ticks
            rec = {"gen": key[0], "idx": key[1], "seed": key[2], "horizon": key[3],
                   "ticks": ticks, "invalid": (invalid if isinstance(invalid, str) else 0)}
            if st:
                rec.update(st)
                STATS[key] = st
            new.append(rec)
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"    [{label}] {done}/{len(todo)} episodes  "
                      f"{done/max(1e-9,el):.1f}/s  eta {max(0,(len(todo)-done))/max(1e-9,done/el):.0f}s",
                      flush=True)
        append_ledger(new)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--screen-seeds", type=int, default=6)
    # 2,000 was tested and every candidate scored the horizon cap (median extinction is ~7,500),
    # so the screen separated nothing. 12,000 leaves ~80% of seeds extinct - enough signal.
    ap.add_argument("--screen-horizon", type=int, default=12000)
    ap.add_argument("--confirm-top", type=int, default=12)
    ap.add_argument("--confirm-seeds", type=int, default=40)
    ap.add_argument("--confirm-horizon", type=int, default=18000)
    ap.add_argument("--confirm-every", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--holdout-base", type=int, default=90000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--corr-heading", type=float, default=0.50,
                    help="max |heading| correction, radians - the residual's AUTHORITY")
    ap.add_argument("--corr-speed", type=float, default=0.30,
                    help="max speed correction as a fraction of sprint_speed")
    ap.add_argument("--corr-gate", type=float, default=0.25,
                    help="max reproduction-gate offset, fraction of max_energy")
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--resume-gen", type=int, default=None)
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    from multiprocessing import get_context
    ctx = get_context("spawn")
    cache = load_ledger()
    print(f"ml_loop | gens {a.gens} pop {a.pop}/shard sigma {a.sigma} alpha {a.alpha} "
          f"| shard {a.shard}/{a.nshards} | workers {a.workers} | cache {len(cache)} episodes",
          flush=True)

    corr = np.array([a.corr_heading, a.corr_speed, a.corr_gate], np.float32)
    print(f"  correction authority: heading +/-{a.corr_heading} rad  speed +/-{a.corr_speed}*sprint"
          f"  gate +/-{a.corr_gate}", flush=True)
    net = load_centre(a.resume_gen) if a.resume_gen else make_net()
    base_params = params_of(net)
    for gen in range(a.gens):
        # ROTATING screening block - never reuse a block for selection twice
        s0 = a.seed_base + gen * 1000
        screen_seeds = list(range(s0, s0 + a.screen_seeds))
        idxs = [i for i in range(1, a.pop) if i % a.nshards == a.shard]      # 0 = centre, always

        # candidate 0 = the unperturbed CENTRE (the incumbent while sigma has not moved it)
        cands = {0: None}
        for i in idxs:
            g = torch.Generator().manual_seed(eps_seed(gen, i))
            with torch.no_grad():
                ps = [p + torch.randn(p.shape, generator=g) * a.sigma for p in base_params]
            cands[i] = ps

        jobs = []
        for i, ps in cands.items():
            w = None
            if ps is not None:
                tmp = make_net()
                set_params(tmp, ps)
                w = tmp.state_dict()
            for s in screen_seeds:
                jobs.append((gen, i, s, a.screen_horizon, a.params, w, corr))
        print(f"\n=== GEN {gen} | {len(cands)} candidates x {len(screen_seeds)} seeds "
              f"@ {a.screen_horizon} | {len(jobs)} episodes", flush=True)
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            evaluate(pool, cache, jobs, f"g{gen}")

        # score: paired mean vs candidate 0 on the SAME seeds
        def sc(i):
            v = [cache.get((gen, i, s, a.screen_horizon)) for s in screen_seeds]
            b = [cache.get((gen, 0, s, a.screen_horizon)) for s in screen_seeds]
            pr = [(x - y) for x, y in zip(v, b) if x is not None and y is not None]
            if not pr:
                return None
            return float(np.mean(pr)), float(np.mean([x for x in v if x is not None]))

        scored = [(sc(i), i) for i in cands if sc(i) is not None]
        scored = [(s, i) for s, i in scored if s is not None]
        scored.sort(key=lambda t: -t[0][0])
        bm = sc(0)
        print(f"  GEN {gen}: centre mean {bm[1]:.0f} ticks | best paired "
              f"{scored[0][0][0]:+.0f} (cand {scored[0][1]}) | worst {scored[-1][0][0]:+.0f} "
              f"| {sum(1 for s,_ in scored if s[0] > 0)}/{len(scored)} beat the centre", flush=True)

        # ---- ES update: rank-normalised, weighted by the perturbation, scaled by 1/(N*sigma)
        fits = np.array([s[0] for s, _ in scored], float)
        ids = [i for _, i in scored]
        if len(fits) >= 4 and fits.std() > 0:
            rank = np.argsort(np.argsort(fits)).astype(float)
            w = (rank / (len(rank) - 1.0)) - 0.5
            acc = [torch.zeros_like(p) for p in base_params]
            for wj, i in zip(w, ids):
                g = torch.Generator().manual_seed(eps_seed(gen, i))
                with torch.no_grad():
                    for k, p in enumerate(base_params):
                        acc[k] += torch.randn(p.shape, generator=g) * a.sigma * float(wj)
            with torch.no_grad():
                for k in range(len(base_params)):
                    base_params[k] = base_params[k] + (a.alpha / (len(ids) * a.sigma)) * acc[k]
            set_params(net, base_params)
            save_centre(net, gen + 1)

        # ---- periodic held-out confirmation (never used for selection)
        if (gen + 1) % a.confirm_every == 0 or gen == a.gens - 1:
            top = ids[:a.confirm_top]
            hs = list(range(a.holdout_base + gen * 100, a.holdout_base + gen * 100 + a.confirm_seeds))
            cjobs = []
            for i in top:
                w = None
                if i != 0:
                    tmp = make_net()
                    set_params(tmp, cands[i])
                    w = tmp.state_dict()
                for s in hs:
                    cjobs.append((-1, i, s, a.confirm_horizon, a.params, w, corr))
            for s in hs:
                cjobs.append((-1, -1, s, a.confirm_horizon, a.params, None, corr))
            print(f"  CONFIRM gen {gen} on held-out seeds {hs[0]}..{hs[-1]} "
                  f"({len(cjobs)} episodes)", flush=True)
            with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
                evaluate(pool, cache, cjobs, f"confirm{gen}")

            def cf(i):
                v = [cache.get((-1, i, s, a.confirm_horizon)) for s in hs]
                b = [cache.get((-1, -1, s, a.confirm_horizon)) for s in hs]
                pr = [x - y for x, y in zip(v, b) if x is not None and y is not None]
                return (float(np.mean(pr)), float(np.mean([x for x in v if x is not None])),
                        len(pr)) if pr else None
            rows = []
            for i in top:
                r = cf(i)
                if r:
                    rows.append((r[0], i, r[1], r[2]))
            rows.sort(reverse=True)
            bmean = float(np.mean([x for x in
                                   [cache.get((-1, -1, s, a.confirm_horizon)) for s in hs]
                                   if x is not None]))
            # is the net CLIPPED by its authority, or using almost none of it?
            cres = {}
            for i in top:
                mf, xf = [], []
                for s_ in hs:
                    r = STATS.get((-1, i, s_, a.confirm_horizon))
                    if r:
                        mf.append(r["corr_abs_mean_frac"][0])
                        xf.append(r["corr_abs_max_frac"][0])
                if mf:
                    cres[i] = (float(np.mean(mf)), float(np.max(xf)))
            if cres:
                print("  CORRECTION USE (fraction of the allowed bound; heading channel):")
                for i in top[:6]:
                    if i in cres:
                        print(f"    cand {i:<4} mean |corr| = {100*cres[i][0]:5.1f}% of bound, "
                              f"max = {100*cres[i][1]:5.1f}%  "
                              f"{'<- CLIPPED, wants more authority' if cres[i][1] > 0.9 else ''}")
            print(f"  HELD-OUT (BASE mean {bmean:.0f}):")
            for p, i, m, n in rows[:6]:
                print(f"    cand {i:<4} mean {m:7.0f}  paired {p:+8.0f} ({100*p/bmean:+5.1f}%)  n={n}",
                      flush=True)
            json.dump({"gen": gen, "base_mean": bmean,
                       "rows": [{"idx": i, "paired": p, "mean": m, "n": n} for p, i, m, n in rows]},
                      open(os.path.join(WORK, f"confirm_gen{gen}.json"), "w"), indent=1)

    print("\nloop finished", flush=True)


if __name__ == "__main__":
    main()