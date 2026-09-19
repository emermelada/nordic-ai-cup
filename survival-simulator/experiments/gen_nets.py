#!/usr/bin/env python3
"""gen_nets.py - generate small neural-policy candidates for sched.py ("__policy__": "net").

Why this lane: the heuristic family has produced five consecutive refutations for one shared reason
(they all cut income or mobility). A neural policy is a genuinely different representation: it can
express behaviours the potential field cannot, and it is small enough that inference is free.

Why random-then-evolve rather than gradients: prior art (Lux AI S3) found PPO over an imitation policy
collapsed, and this project's own from-scratch evolution attempt failed because its FITNESS was
gameable by dying - not because evolution is wrong. So: screen a wide random population with a
survival-based fitness (not gameable - dying yields fewer ticks), then promote winners.

The observation is env_wrapper.build_obs: 31 features, all already normalised to [0,1]-ish.
Output: 4 raw values -> tanh -> [move_distance fraction, relative direction, turn angle, spawn flag].

Usage:
    ./gen_nets.py --n 300 --hid 16 --seed 1 --out nets_rand_300.json
    ./gen_nets.py --n 400 --hid 24 --seed 2 --scale 1.6 --out nets_wide_400.json
"""
import argparse
import json
import math
import random

DIN = 31


def rand_net(rng, hid, scale):
    """He-style init scaled by `scale`; a larger scale = more saturated, more committed behaviours."""
    def rn(n):
        return [rng.gauss(0.0, scale / math.sqrt(n)) for _ in range(n)]
    w1 = [rng.gauss(0.0, scale / math.sqrt(DIN)) for _ in range(hid * DIN)]
    b1 = [rng.gauss(0.0, 0.35 * scale) for _ in range(hid)]
    w2 = [rng.gauss(0.0, 1.2 * scale / math.sqrt(hid)) for _ in range(4 * hid)]
    b2 = [rng.gauss(0.0, 0.35 * scale) for _ in range(4)]
    return {"in": DIN, "hid": hid, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--hid", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="nets.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cands = []
    for i in range(args.n):
        hid = args.hid if i % 4 else max(4, args.hid // 2)          # mix of widths
        sc = args.scale * (0.5 if i % 5 == 0 else (2.0 if i % 7 == 0 else 1.0))
        cands.append({"id": f"net{args.seed}_{i:04d}_h{hid}_s{sc:.2f}".replace(".", ""),
                      "params": {"__policy__": "net", "__net__": rand_net(rng, hid, sc)}})
    json.dump(cands, open(args.out, "w"))
    print(f"wrote {len(cands)} net candidates -> {args.out} (hid {args.hid}, scale {args.scale})")


if __name__ == "__main__":
    main()
