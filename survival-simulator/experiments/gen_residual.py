#!/usr/bin/env python3
"""gen_residual.py - generation-0 population of RESIDUAL policy networks.

The learning attempt the brief asks for: don't hand-write more rules, learn the decision-making.
Design choices forced by evidence:
  * residual, not from-scratch: 300 random MLPs scored -38% at 40 seeds, so a policy that starts below
    the heuristic wastes all its budget climbing back. Zero weights = the exact deployed controller,
    so every candidate is measured as a CORRECTION to the incumbent.
  * tiny net (31 -> 12 -> 2): the two outputs are bounded offsets to heading and speed only. Spawn
    gating, threat handling and phase logic stay with the controller, so a bad net cannot wreck the
    fleet wholesale.
  * sigma ladder 0.02/0.05/0.15: exploration radius. Whatever wins gets mutated for generation 1.
"""
import argparse
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from env_wrapper import OBS_DIM  # noqa: E402

DEPLOY = os.path.join(os.path.dirname(HERE), "best_controller", "params.json")
H = 12


def net_from(vec):
    i = 0
    w1 = vec[i:i + H * OBS_DIM].tolist(); i += H * OBS_DIM
    b1 = vec[i:i + H].tolist(); i += H
    w2 = vec[i:i + 2 * H].tolist(); i += 2 * H
    b2 = vec[i:i + 2].tolist()
    return {"h": H, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="par_res.json")
    ap.add_argument("--per-sigma", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    base = json.load(open(DEPLOY))
    rng = np.random.default_rng(a.seed)
    n = H * OBS_DIM + H + 2 * H + 2
    cands = [{"id": "R_zero", "params": {"__policy__": "residual", "__base__": base,
                                         "__net__": net_from(np.zeros(n)), "id": "R_zero"}}]
    for sig in (0.02, 0.05, 0.15):
        for k in range(a.per_sigma):
            v = rng.normal(0.0, sig, n)
            cid = f"R_s{sig:.2f}_{k:02d}"
            cands.append({"id": cid, "params": {"__policy__": "residual", "__base__": base,
                                                "__net__": net_from(v), "id": cid}})
    json.dump(cands, open(a.out, "w"))
    print(f"wrote {len(cands)} residual candidates ({n} weights each, obs_dim={OBS_DIM})")


if __name__ == "__main__":
    main()
