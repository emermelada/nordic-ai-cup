#!/usr/bin/env python3
"""Correct analysis of the held-out confirm stages.

The printed tables show only the top 6 of 12 and treat candidates as independent. They are not:
all 12 candidates of a generation are perturbations of the SAME centre, so their paired effects
share a common component (the centre's own effect on those seeds). This computes:
  (a) per-confirm-block, per-seed mean over the 12 confirmed candidates -> an unbiased estimate of
      the 'ES output' effect, with the SE taken ACROSS SEEDS (correct correlation structure);
  (b) the same for the single best candidate (the actual deployable);
  (c) the pure-noise reference: what max-of-12 would look like if the centre effect were 0.
"""
import json, os, sys
from collections import defaultdict
import numpy as np

WORK = sys.argv[1] if len(sys.argv) > 1 else "/opt/nac_h2h/ml"
HOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 90000

by_seed = defaultdict(lambda: defaultdict(dict))  # block -> idx -> seed -> ticks
for line in open(os.path.join(WORK, "ledger.jsonl")):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r["gen"] != -1 or r["horizon"] != 18000 or r["ticks"] is None:
        continue
    b = (r["seed"] - HOLD) // 100
    by_seed[b][r["idx"]][r["seed"]] = r["ticks"]

print(f"{'block(gen)':>10} {'nseeds':>6} {'base_mean':>9} | {'pop mean+-SE':>16} {'t':>5} {'W/L':>9} | {'best popL':>10} {'t':>5} | {'best single':>11} {'t':>5}")
alleff, allbest = [], []
for b in sorted(by_seed):
    d = by_seed[b]
    if -1 not in d:
        continue
    seeds = sorted(d[-1])
    base = {s: d[-1][s] for s in seeds}
    cands = [i for i in d if i != -1]
    # per-seed effect for each candidate
    eff = {}
    for i in cands:
        for s in seeds:
            if s in d[i]:
                eff.setdefault(s, []).append(d[i][s] - base[s])
    seeds2 = [s for s in seeds if eff.get(s)]
    if len(seeds2) < 5:
        continue
    pop = np.array([np.mean(eff[s]) for s in seeds2])          # per-seed mean over candidates
    m, se = pop.mean(), pop.std(ddof=1) / np.sqrt(len(pop))
    # best single candidate measured on this block (post-hoc max; selection happened on screen)
    per_c = {i: np.array([d[i][s] - base[s] for s in seeds2 if s in d[i]]) for i in cands}
    per_c = {i: v for i, v in per_c.items() if len(v) >= len(seeds2) - 2}
    bi = max(per_c, key=lambda i: per_c[i].mean())
    bv = per_c[bi]
    bm, bse = bv.mean(), bv.std(ddof=1) / np.sqrt(len(bv))
    flat = np.concatenate([per_c[i] for i in sorted(per_c)])
    alleff.append(pop); allbest.append(bv)
    print(f"{b:>10} {len(seeds2):>6} {np.mean(list(base.values())):>9.0f} | {m:>+11.0f}+-{se:>4.0f} "
          f"{m/se:>5.1f} {int((pop>0).sum()):>4}/{int((pop<0).sum()):<4} | {bm:>+10.0f} "
          f"{bm/bse:>5.1f} | {flat.mean():>+11.0f} {flat.mean()/(flat.std(ddof=1)/np.sqrt(len(flat))):>5.1f}")

if alleff:
    pop = np.concatenate(alleff)
    print(f"\nPOOLED pop-effect over {len(pop)} held-out seeds: {pop.mean():+.0f} "
          f"+- {pop.std(ddof=1)/np.sqrt(len(pop)):.0f}  t={pop.mean()/(pop.std(ddof=1)/np.sqrt(len(pop))):.2f}  "
          f"W/L={int((pop>0).sum())}/{int((pop<0).sum())}")
    print(f"  per-seed SD of the pop effect: {pop.std(ddof=1):.0f} ticks  (this is the real noise unit)")
    allb = np.concatenate(allbest)
    print(f"  pooled ALL confirmed candidates (selected, so biased up): {allb.mean():+.0f} "
          f"+- {allb.std(ddof=1)/np.sqrt(len(allb)):.0f}  n={len(allb)}")