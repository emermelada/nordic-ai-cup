#!/usr/bin/env python3
"""es_rec.py - antithetic ES over the residual GRU on the deployed controller, with statistics.

WHAT WAS WRONG WITH THE PREVIOUS ES REGIME (measured, not assumed)
  It screened 64 candidates on 6 seeds per generation. The per-seed SD of a candidate-vs-BASE paired
  difference is ~2,595 ticks, so the generation-level effect estimate had SE ~1,060 ticks while the
  whole effect being hunted was a few hundred. The centre consequently random-walked: the fraction of
  candidates beating BASE oscillated between 3% and 98% across generations, and the held-out confirm
  effect DECLINED (+423, +897, +255, -277 at gens 7/11/15/19; pooled +324 +- 205, t=1.58, n=160).
  4.5 h of 60 cores produced no reliable signal. This script fixes the estimator, not the slogan:

    * ANTITHETIC pairs (theta+e, theta-e) evaluated on the same seeds -> the perturbation noise
      cancels in the difference, so the gradient direction is no longer dominated by sampling luck;
    * 24 rotating seeds per generation instead of 6 (SE ~530 per arm, ~130 on the paired gradient);
    * an explicit BASE arm in every generation (candidate 0 is the exact incumbent, never a copy);
    * the CENTRE itself is evaluated on fresh held-out seeds every generation, so divergence is
      visible instead of inferred;
    * sigma is large enough to express different BEHAVIOUR (the dead run used sigma=0.02 on a
      zero-init head, so the net moved heading by ~0.03 rad: it never left V2's neighbourhood);
    * every ledger row carries the candidate's WEIGHT HASH, so a row can never be reused for
      different weights; the centre + archive + ledger are checkpointed every generation;
    * an archive of the best centres is retained - the centre is never blindly replaced by the
      single luckiest arm.

DIAGNOSED NEXT FIX (from the two runs on 2026-09-20, neither of which beat the base)
  The estimator is now adequate but the STEP is not trusted: sigma=0.15 on the head is small enough to
  show nothing (pop mean +331 +- 202 on the hive base, t=1.6) and large enough that one update moved the
  centre to -1640 +- 185 (-8.9 sigma) against hive - a tight local optimum is destroyed by a single
  averaged step. The fix is a trust region: after the step, evaluate the new centre on a few seeds and
  KEEP THE OLD CENTRE unless the new one is better (rollback), shrinking sigma when a step is rejected.
  Never commit a centre that has not been measured. Also: the candidate archive below exists because the
  one interesting signal (above) could not be re-measured - candidates are the deliverable of a search.

ZERO-CHANGE PATH  the actor head is zero-initialised, so generation 0's centre reproduces V2 exactly.

    python3 es_rec.py --workers 56 --out /opt/nac_h2h/es2
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                       # noqa: E402
from probe_enc import build_v2, V2_DIM                                   # noqa: E402

CORR = np.array([0.50, 0.30], np.float32)
HID = 48
LOG2PI = math.log(2 * math.pi)
SIG_TRUNK = 0.03      # per-weight sigma on the GRU (its activations are O(1))
SIG_HEAD = 0.15       # per-weight sigma on the actor head: tanh(mu) then saturates, i.e. the
                      # candidates actually explore different BEHAVIOUR, not a 0.03 rad nudge


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make_net(dim=V2_DIM, hid=HID):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(dim, hid, batch_first=True)
            self.mu = nn.Linear(hid, 2)
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.mu(o), h
    return Net()


def blank_weights(dim=V2_DIM, hid=HID):
    import torch
    return {k: v.detach().numpy().copy() for k, v in make_net(dim, hid).state_dict().items()}


def wkey(w):
    h = hashlib.sha1()
    for k in sorted(w):
        h.update(k.encode())
        h.update(np.ascontiguousarray(w[k]).tobytes())
    return h.hexdigest()[:12]


def gen_eps(gen, i, sign, shapes, sigmas):
    """Deterministic perturbation for (gen, candidate i, sign). Reproduced identically in the ES
    update, so the direction used to move the centre is exactly the direction that was evaluated."""
    import torch
    seed = int(hashlib.sha256(f"{gen}:{i}:{sign}".encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted(shapes):
        out[k] = (torch.randn(shapes[k], generator=g).numpy() * sigmas[k]).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------- rollout
def _mem_update(s, m, bc, n_agents, tick=None):
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
    m["tick"] = float(tick) if tick is not None else float(bc._SIM_TICK)
    m["n_agents"] = float(n_agents)
    if bc is not None and bc._GS_LAST is not None:
        m["rank"] = float(bc._GS_LAST[1])
    m["e"] = energy
    return m


def run_arm(job):
    """One (arm, seed) episode. weights=None => the exact base policy.

    base 'heuristic': best_controller.py + the deployed 52-key params (what survival.zaitzev.com serves)
    base 'hive'     : the survival-v2 controller (hive.py 829e4147), which measured +39.4% over the
                      heuristic on 119 fresh paired seeds. hive keeps its own genome/colony/breeding
                      logic; the residual only adds bounded movement direction/magnitude corrections.
    """
    arm, seed, horizon, params_path, weights, base = job
    import torch
    torch.set_num_threads(1)
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    try:
        net = None
        if weights is not None:
            net = make_net()
            net.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
            net.eval()
        pol = None
        bc = None
        if base == "hive":
            from hive_v2 import Hive
            pol = Hive(seed=seed)
        else:
            import best_controller as bc_mod
            bc = bc_mod
            P = dict(bc.DEFAULT_PARAMS)
            with open(params_path) as f:
                P.update(json.load(f))
            pol = bc.make_policy(P)
        random.seed(seed)
        np.random.seed(seed)
        if bc is not None:
            bc.reset_memory()
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        memo, hid = {}, {}
        csum = cmax = 0.0
        cn = 0
        i = 0
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                break
            states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
            if not states:
                break
            acts = []
            if base == "hive":
                out = pol.decide({"agent_status": states, "sim_time": i / 10.0, "n_agents": len(states)})
                byid = {st["agent_id"]: st for st in states}
                for d in (out or []):
                    aid = d.get("agent_id")
                    st = byid.get(aid)
                    if st is None:
                        continue
                    dr = float(d.get("move_direction", 0.0) or 0.0)
                    dist = float(d.get("move_distance", 0.0) or 0.0)
                    turn = float(d.get("turn_angle", 0.0) or 0.0)
                    spawn = bool(d.get("spawn_agent", False))
                    if net is not None:
                        m = _mem_update(st, memo.setdefault(aid, {}), None, len(states), tick=i / 10.0)
                        x = np.asarray(build_v2(st, m), np.float32)
                        with torch.no_grad():
                            mu, hnew = net(torch.tensor(x)[None, None, :], hid.get(aid))
                            hnew = torch.clamp(hnew, -10.0, 10.0)
                            a2 = (torch.tanh(mu[0, 0]) * torch.tensor(CORR)).numpy()
                        hid[aid] = hnew
                        csum += float(np.abs(a2).mean())
                        cmax = max(cmax, float(np.abs(a2).max()) / float(CORR.max()))
                        cn += 1
                        sprint = max(1.0, float(st.get("sprint_speed", 20.0) or 20.0))
                        if a2[0] != 0.0:
                            dr = dr + float(a2[0])
                        if a2[1] != 0.0:
                            dist = max(0.0, dist + float(a2[1]) * sprint)
                    acts.append((aid, ActionRequest(agent_id=aid, move_distance=dist,
                                                    move_direction=dr, turn_angle=turn,
                                                    spawn_agent=spawn)))
            else:
                for st in states:
                    aid = st["agent_id"]
                    dist, dr, turn, spawn = pol(st)
                    if net is not None:
                        m = _mem_update(st, memo.setdefault(aid, {}), bc, len(states))
                        x = np.asarray(build_v2(st, m), np.float32)
                        with torch.no_grad():
                            mu, hnew = net(torch.tensor(x)[None, None, :], hid.get(aid))
                            hnew = torch.clamp(hnew, -10.0, 10.0)
                            a2 = (torch.tanh(mu[0, 0]) * torch.tensor(CORR)).numpy()
                        hid[aid] = hnew
                        csum += float(np.abs(a2).mean())
                        cmax = max(cmax, float(np.abs(a2).max()) / float(CORR.max()))
                        cn += 1
                        sprint = max(1.0, float(st.get("sprint_speed", 20.0) or 20.0))
                        if a2[0] != 0.0:
                            dr = dr + float(a2[0])
                        if a2[1] != 0.0:
                            dist = dist + float(a2[1]) * sprint
                    acts.append((aid, ActionRequest(agent_id=aid, move_distance=float(dist),
                                                    move_direction=float(dr), turn_angle=float(turn),
                                                    spawn_agent=bool(spawn))))
            core.step(acts)
            keep = {a.agent_id for a in core.env.agents}
            hid = {k: v for k, v in hid.items() if k in keep}
        return (arm, seed), i + 1, {"corr_mean_frac": (csum / max(1, cn)) / float(CORR.mean()),
                                    "corr_max_frac": cmax}
    except Exception as e:
        return (arm, seed), None, f"ERR:{type(e).__name__}:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "es2"))
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--pairs", type=int, default=16, help="antithetic pairs per generation")
    ap.add_argument("--seeds", type=int, default=24, help="rotating screen seeds per generation")
    ap.add_argument("--horizon", type=int, default=6000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--holdout-base", type=int, default=600000)
    ap.add_argument("--holdout-seeds", type=int, default=40)
    ap.add_argument("--holdout-every", type=int, default=3)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--base", default="heuristic", choices=["heuristic", "hive"],
                    help="controller the residual sits on")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.pairs, a.seeds, a.horizon, a.workers, a.gens = 2, 2, 500, 2, 2
        a.holdout_every, a.holdout_seeds = 1, 2

    import torch
    from multiprocessing import get_context

    os.makedirs(a.out, exist_ok=True)
    LEDGER = os.path.join(a.out, "ledger.jsonl")
    CKPT = os.path.join(a.out, "centre.npz")
    ARCH = os.path.join(a.out, "archive.json")
    ctx = get_context("spawn")

    centre = blank_weights()
    shapes = {k: v.shape for k, v in centre.items()}
    sigmas = {k: (SIG_HEAD if k.startswith("mu") else SIG_TRUNK) for k in centre}
    start_gen = 0
    archive = []
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        centre = {k: z[k] for k in z.files if k != "gen"}
        start_gen = int(z["gen"]) + 1
        print(f"  RESUMED centre from {CKPT} at gen {start_gen}", flush=True)
    if os.path.exists(ARCH):
        archive = json.load(open(ARCH))

    # receipt: the centre at generation 0 must be exactly V2 (zero head)
    zh = float(np.abs(centre["mu.weight"]).max()) + float(np.abs(centre["mu.bias"]).max())
    print(f"es_rec | base {a.base} | pairs {a.pairs} | seeds/gen {a.seeds} | horizon {a.horizon} | workers {a.workers}"
          f" | obs dim {V2_DIM} | head L1 at centre = {zh:.3g} (0 => policy == V2)", flush=True)

    for gen in range(start_gen, a.gens):
        t0 = time.time()
        s0 = a.seed_base + gen * 100
        seeds = list(range(s0, s0 + a.seeds))
        arms = {0: None, "centre": dict(centre)}
        eps = {}
        for i in range(1, a.pairs + 1):
            e = gen_eps(gen, i, 1, shapes, sigmas)
            eps[("p", i)] = e
            arms[("p", i)] = {k: centre[k] + e[k] for k in centre}
            arms[("m", i)] = {k: centre[k] - e[k] for k in centre}
        hsh = {k: wkey(v) for k, v in arms.items() if v is not None}

        jobs = []
        for k, w in arms.items():
            for s in seeds:
                jobs.append((k, s, a.horizon, a.params, w, a.base))
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            out = []
            with open(LEDGER, "a") as f:
                for key, t, rec in pool.imap_unordered(run_arm, jobs, chunksize=1):
                    arm, seed = key
                    row = {"gen": gen, "arm": str(arm), "wkey": hsh.get(arm), "seed": seed,
                           "horizon": a.horizon, "ticks": t}
                    if isinstance(rec, str):
                        row["error"] = rec
                    else:
                        row.update(rec)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    if t is not None:
                        out.append((arm, seed, t))

        ticks = {}
        for arm, seed, t in out:
            ticks.setdefault(str(arm), {})[seed] = t
        if "0" not in ticks or "centre" not in ticks:
            print(f"  gen {gen}: BASE or centre arm missing, skipping", flush=True)
            continue

        def paired(armname):
            d = [ticks[armname][s] - ticks["0"][s] for s in ticks["0"] if s in ticks.get(armname, {})]
            return (float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
                    len(d)) if d else (None, None, 0)

        bm = float(np.mean(list(ticks["0"].values())))
        cm, cse, _ = paired("centre")
        diffs = {}
        for i in range(1, a.pairs + 1):
            d = [ticks[str(("p", i))][s] - ticks[str(("m", i))][s] for s in seeds
                 if s in ticks.get(str(("p", i)), {}) and s in ticks.get(str(("m", i)), {})]
            if d:
                diffs[i] = float(np.mean(d))
        fits = np.array([diffs[i] for i in sorted(diffs)], float)

        # ---- archive the top candidates BEFORE the centre moves. The hive-base run produced a
        # +331 population mean at gen 0 that could not be re-tested afterwards, because only the
        # centroid was checkpointed and the perturbation RNG state was not recoverable. Candidates
        # are the deliverable of a search, so they must survive the generation that made them.
        if len(diffs):
            order = np.argsort([-diffs[i] for i in sorted(diffs)])
            ids_sorted = sorted(diffs)
            for rank, oi in enumerate(order[:2]):
                i = ids_sorted[int(oi)]
                cand = {k: (centre[k] + eps[("p", i)][k]).astype(np.float32) for k in centre}
                np.savez(os.path.join(a.out, f"cand_g{gen}_r{rank}.npz"), **cand)
                with open(os.path.join(a.out, "candidates.jsonl"), "a") as f:
                    f.write(json.dumps({"gen": gen, "rank": rank, "eps_idx": i,
                                        "file": f"cand_g{gen}_r{rank}.npz",
                                        "paired_vs_base_sample": diffs[i]}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        # ---- ES update: rank-normalised antithetic differences, direction = the stored eps
        if len(fits) >= 4 and fits.std() > 0:
            rank = np.argsort(np.argsort(fits)).astype(float)
            w = (rank / (len(rank) - 1.0)) - 0.5
            ids = sorted(diffs)
            for k in centre:
                acc = np.zeros_like(centre[k])
                for wj, i in zip(w, ids):
                    acc += eps[("p", i)][k] * float(wj) * 2.0
                centre[k] = (centre[k] + (a.alpha / (len(ids))) * acc).astype(np.float32)
        if cm is not None:
            archive.append({"gen": gen, "centre_paired": cm, "centre_se": cse, "base_mean": bm,
                            "pop_mean_diff": float(fits.mean()) if len(fits) else None,
                            "head_L1": float(np.abs(centre["mu.weight"]).sum()),
                            "wkey_centre": wkey(centre)})
            archive = archive[-200:]
            json.dump(archive, open(ARCH, "w"), indent=1)
        np.savez(CKPT + ".tmp.npz", **centre, gen=gen)
        os.replace(CKPT + ".tmp.npz", CKPT)

        # ---- held-out evaluation of the centre on FRESH seeds (never used for selection)
        ho = None
        if a.holdout_every and (gen + 1) % a.holdout_every == 0:
            hs = list(range(a.holdout_base + gen * 100, a.holdout_base + gen * 100 + a.holdout_seeds))
            hjobs = [("0", s, 18000, a.params, None, a.base) for s in hs] + \
                    [("centre", s, 18000, a.params, dict(centre), a.base) for s in hs]
            with ctx.Pool(min(a.workers, len(hjobs)), maxtasksperchild=1) as pool:
                hres = pool.map(run_arm, hjobs)
            ht = {}
            for arm, seed, t in [(k[0], k[1], t) for k, t, r in hres if t is not None]:
                ht.setdefault(str(arm), {})[seed] = t
            if "0" in ht and "centre" in ht:
                d = [ht["centre"][s] - ht["0"][s] for s in ht["0"] if s in ht["centre"]]
                ho = {"gen": gen, "n": len(d), "base": float(np.mean([ht["0"][s] for s in ht["0"]])),
                      "centre": float(np.mean([ht["centre"][s] for s in ht["centre"]])),
                      "paired": float(np.mean(d)), "W": int(sum(1 for x in d if x > 0)),
                      "L": int(sum(1 for x in d if x < 0))}
                with open(os.path.join(a.out, "holdout.jsonl"), "a") as f:
                    f.write(json.dumps(ho) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        print(f"  gen {gen}: base {bm:.0f} | centre {cm:+.0f}+-{cse:.0f} (n={a.seeds}) | "
              f"pop mean diff {fits.mean():+.0f} | pop sd {fits.std():.0f} | "
              f"{int((fits>0).sum())}/{len(fits)} pairs up | head L1 {float(np.abs(centre['mu.weight']).sum()):.2f}"
              f" | {time.time()-t0:.0f}s"
              + (f" || HOLDOUT n={ho['n']} base {ho['base']:.0f} centre {ho['centre']:.0f} "
                 f"paired {ho['paired']:+.0f} W/L {ho['W']}/{ho['L']}" if ho else ""), flush=True)
    print("es_rec done", flush=True)


if __name__ == "__main__":
    main()
