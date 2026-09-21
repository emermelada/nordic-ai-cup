"""Paired comparison of two bench outputs on their common seeds.   python paired.py base.json new.json"""
import json, statistics as S, sys
a = {r["seed"]: r for r in json.load(open(sys.argv[1]))["rows"]}
b = {r["seed"]: r for r in json.load(open(sys.argv[2]))["rows"]}
common = sorted(set(a) & set(b))
d = [b[s]["score"] - a[s]["score"] for s in common]
f = lambda rows, k: S.fmean(rows[s][k] for s in common)
print(f"n {len(common)}  base {f(a, 'score'):.1f}  new {f(b, 'score'):.1f}  diff {S.fmean(d):+.1f} +- {S.stdev(d) / len(d) ** 0.5:.1f}"
      f"  wins {sum(x > 0 for x in d)}/{len(d)}")
for k in ("time", "fruit_n", "fruit_mean_e", "kill_n", "kill_pen", "starve_n", "births"):
    print(f"  {k:13s} {f(a, k):8.1f} -> {f(b, k):8.1f}")
for k in ("move_e", "old_e", "live_e", "spawn_e"):
    print(f"  {k:13s} {S.fmean(a[s]['ledger'][k] for s in common):8.0f} -> {S.fmean(b[s]['ledger'][k] for s in common):8.0f}")
