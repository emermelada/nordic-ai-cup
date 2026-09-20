#!/usr/bin/env python3
"""Analyse the ES ledger: is there a real population-level signal, or only max-over-12 noise?"""
import json, os, sys
from collections import defaultdict
import numpy as np

WORK = sys.argv[1] if len(sys.argv) > 1 else "/opt/nac_h2h/ml"
rows = defaultdict(dict)   # (gen,idx) -> {seed:ticks}   for screen
conf = defaultdict(dict)   # idx -> {seed: ticks} at gen -1
base = {}
inv = defaultdict(int)
H = defaultdict(set)
for line in open(os.path.join(WORK, "ledger.jsonl")):
    try:
        r = json.loads(line)
    except Exception:
        continue
    g, i, s, h = r["gen"], r["idx"], r["seed"], r["horizon"]
    t = r["ticks"]
    if isinstance(r.get("invalid"), str) or t is None:
        inv[(g, i)] += 1
        continue
    H[(g, h)].add(h)
    if g == -1:
        conf[i][s] = t
    else:
        rows[(g, i)][s] = t

gens = sorted({g for g, i in rows if g >= 0})
print(f"gens present: {min(gens)}..{max(gens)}  ({len(gens)} gens)   ledger rows: {sum(len(v) for v in rows.values())}")
print(f"invalid episodes: {dict(inv) if len(inv)<12 else f'{len(inv)} keys, total {sum(inv.values())}'}")

print("\n=== SCREEN: per-generation population health (6 seeds @12000) ===")
print(f"{'gen':>4} {'base_mean':>10} {'pop_mean_rel_base':>18} {'frac>0':>7} {'best':>8} {'worst':>8} {'spread':>8} {'n_cands':>7}")
for g in gens:
    idxs = [i for (gg, i) in rows if gg == g]
    if 0 not in idxs:
        continue
    b = rows[(g, 0)]
    seeds = sorted(b)
    diffs, vals = [], []
    for i in idxs:
        if i == 0:
            continue
        d = [rows[(g, i)][s] - b[s] for s in seeds if s in rows[(g, i)]]
        if d:
            diffs.append(np.mean(d)); vals.append(np.mean([rows[(g, i)][s] for s in seeds if s in rows[(g, i)]]))
    if not diffs:
        continue
    diffs = np.array(diffs)
    print(f"{g:>4} {np.mean(list(b.values())):>10.0f} {diffs.mean():>+18.0f} "
          f"{np.mean(diffs>0):>7.2f} {diffs.max():>+8.0f} {diffs.min():>+8.0f} {diffs.std():>8.0f} {len(diffs):>7}")

print("\n=== CONFIRM (gen -1) held-out blocks ===")
if conf:
    idxs = sorted(i for i in conf if i != -1)
    b = conf.get(-1, {})
    print(f"base arm present on {len(b)} seeds, mean {np.mean(list(b.values())) if b else float('nan'):.0f}")
    for i in idxs:
        seeds = sorted(conf[i])
        d = [conf[i][s] - b[s] for s in seeds if s in b]
        if not d:
            continue
        d = np.array(d)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  cand {i:>4} n={len(d):>3} paired {d.mean():>+8.0f}  se {se:>6.0f}  t={d.mean()/se if se else 0:>5.1f}  "
              f"W/L={int((d>0).sum())}/{int((d<0).sum())}  mean {np.mean([conf[i][s] for s in seeds]):.0f}")