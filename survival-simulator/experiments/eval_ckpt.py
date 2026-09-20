#!/usr/bin/env python3
"""eval_ckpt.py - paired verdict for a trained residual checkpoint against its frozen base.

Training reward is never evidence. This is the only thing that counts: the DETERMINISTIC policy (mean
correction, no exploration noise) versus the frozen base on SEEDS IT WAS NEVER TRAINED OR SELECTED ON,
paired per seed, same simulator.

    python3 eval_ckpt.py --ckpt /opt/nac_hm/ppo2/ckpt.pt --base heuristic \
        --seeds 310000-310079 --horizon 18000 --workers 52 --out /opt/nac_h2h/eval_ppo2
    python3 eval_ckpt.py --ckpt /opt/nac_h2h/es_hive/centre.npz --base hive \
        --seeds 310080-310159 --horizon 18000 --workers 52 --out /opt/nac_h2h/eval_eshive
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                       # noqa: E402


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_weights(ckpt, base):
    """Both lanes keep their weights in a checkpoint; es_rec writes npz, ppo_rec writes a torch dict."""
    import torch
    if ckpt.endswith(".npz"):
        z = np.load(ckpt)
        return {k: z[k] for k in z.files if k != "gen"}
    ck = torch.load(ckpt, weights_only=False)
    sd = ck.get("sd", ck)
    return {k: v.detach().numpy().copy() for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="heuristic", choices=["heuristic", "hive"])
    ap.add_argument("--seeds", default="310000-310079")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_ckpt"))
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    a = ap.parse_args()

    from multiprocessing import get_context
    if a.base == "hive":
        import ppo_hive as P
    else:
        import ppo_rec as P

    w = load_weights(a.ckpt, a.base)
    seeds = parse_seeds(a.seeds)
    os.makedirs(a.out, exist_ok=True)
    print(f"eval_ckpt | base {a.base} | {len(seeds)} seeds {seeds[0]}..{seeds[-1]} | horizon {a.horizon} "
          f"| workers {a.workers} | ckpt {a.ckpt}", flush=True)
    print(f"  actor head L1 = {np.abs(w.get('mu.weight', np.zeros(1))).sum():.4f} "
          f"(0 would mean the checkpoint IS the base policy)", flush=True)

    jobs = [("ppo", w, [s], a.horizon, a.params, True) for s in seeds] + \
           [("base", None, [s], a.horizon, a.params, True) for s in seeds]
    ctx = get_context("spawn")
    ticks = {}
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        # pool.map keeps job order, so the first len(seeds) results are the policy arm and the rest
        # are the base arm; rollout returns a list of episode dicts per job.
        results = pool.map(P.rollout, jobs, chunksize=1)
    for i, chunk in enumerate(results):
        mode = "ppo" if i < len(seeds) else "base"
        for e in chunk:
            if e.get("T") is not None:
                ticks.setdefault(mode, {})[e["seed"]] = e["T"]
    b, p = ticks.get("base", {}), ticks.get("ppo", {})
    common = sorted(set(b) & set(p))
    if not common:
        print("  NO PAIRED SEEDS - base and policy arms did not both complete", flush=True)
        return
    d = np.array([p[s] - b[s] for s in common], float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    res = {"ckpt": a.ckpt, "base": a.base, "n": len(common), "horizon": a.horizon,
           "base_mean": float(np.mean([b[s] for s in common])),
           "pol_mean": float(np.mean([p[s] for s in common])),
           "paired": float(d.mean()), "se": float(se), "t": float(d.mean() / se) if se else None,
           "W": int((d > 0).sum()), "L": int((d < 0).sum()),
           "p10_base": float(np.percentile([b[s] for s in common], 10)),
           "p10_pol": float(np.percentile([p[s] for s in common], 10)),
           "min_base": float(min(b[s] for s in common)), "min_pol": float(min(p[s] for s in common)),
           "per_seed_sd": float(d.std(ddof=1))}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    print(f"\n  PAIRED VERDICT: {res['paired']:+.0f} +- {res['se']:.0f} ticks "
          f"({100*res['paired']/res['base_mean']:+.1f}%), t={res['t']:.2f}, W/L {res['W']}/{res['L']}, n={res['n']}",
          flush=True)


if __name__ == "__main__":
    main()
