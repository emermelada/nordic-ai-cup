#!/usr/bin/env python3
"""gen_mutate.py - evolve the net population by MUTATING winners instead of resampling randomly.

The measured situation this responds to: 300 random MLP policies were screened and NOT ONE cleared the
floor; the best was -8.8% versus the deployed controller, most were -50% to -77%. Random search in
weight space is therefore hopeless for this representation at this population size - but the best net
DID survive the first 4,000 ticks (median 4,000, i.e. never died early) and only collapsed in the
mid-game. That makes it a legitimate starting point for hill-climbing, which is what this script feeds.

It reads the ledger written by sched.py (results/<lane>_ledger.json), takes the top-K candidates by
mean survival from the cheapest stage, and emits Gaussian weight perturbations at several sigma levels
- both directions (sigma, and a "big jump" sigma for escaping the local basin).

Usage:
    ./gen_mutate.py --ledger /opt/nac/results/neuro_ledger.json --candidates nets_rand_300.json \
                    --out nets_mut_240.json --topk 8
"""
import argparse
import json
import math
import random


def mutate(net, rng, sigma):
    out = {}
    for k, v in net.items():
        if not isinstance(v, list):
            out[k] = v
            continue
        # scale sigma by the weight's own layer magnitude so small layers are not obliterated
        scale = sigma * (1.0 + 0.5 * rng.random())
        out[k] = [w + rng.gauss(0.0, scale) for w in v]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="nets_mut.json")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--per-parent", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ledger = json.load(open(args.ledger))
    stage1 = min(ledger, key=lambda s: s["stage"])
    ranked = [r for r in stage1["rows"] if r["cand"] != "BASE"]
    ranked.sort(key=lambda r: -r["mean"])
    parents = [r["cand"] for r in ranked[: args.topk]]
    print(f"parents (best of stage {stage1['stage']}, horizon {stage1['horizon']}):")
    for r in ranked[: args.topk]:
        print(f"   {r['cand']:28s} mean {r['mean']:8.1f}  paired {r.get('paired_pct')}  ext {r['extinct_frac']:.2f}")

    by_id = {c["id"]: c for c in json.load(open(args.candidates))}
    missing = [p for p in parents if p not in by_id]
    if missing:
        raise SystemExit(f"parents not found in candidate file: {missing}")

    rng = random.Random(args.seed)
    sigmas = [0.02, 0.05, 0.10, 0.20]
    cands = []
    for p in parents:
        net = by_id[p]["params"]["__net__"]
        for i in range(args.per_parent):
            sig = sigmas[i % len(sigmas)]
            cands.append({"id": f"m_{p[:18]}_s{int(sig*100):02d}_{i:02d}",
                          "params": {"__policy__": "net", "__net__": mutate(net, rng, sig)}})
    json.dump(cands, open(args.out, "w"))
    print(f"wrote {len(cands)} mutants from {len(parents)} parents -> {args.out} "
          f"(sigmas {sigmas})")


if __name__ == "__main__":
    main()
