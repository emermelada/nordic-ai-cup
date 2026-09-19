#!/usr/bin/env python3
"""arms_report.py - paired verdicts for a midgame_diag.py multi-arm run, plus the mechanism.

Reads <prefix>_sum.json (per-seed survival) and <prefix>.jsonl (per-seed 100-tick buckets).
Every arm runs on the SAME seed block, so all deltas are paired per seed against BASE.

Usage: arms_report.py diag/arms_gs
"""
import json
import statistics as st
import sys
from collections import defaultdict

WINDOW = 1000


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def main():
    prefix = sys.argv[1]
    sums = [json.loads(l) for l in open(prefix + "_sum.json")]
    rows = [json.loads(l) for l in open(prefix + ".jsonl")]

    ticks = defaultdict(dict)
    causes = defaultdict(lambda: defaultdict(int))
    for s in sums:
        ticks[s["arm"]][s["seed"]] = s["ticks"]
        for k, v in s["causes"].items():
            causes[s["arm"]][k] += v

    base = ticks.get("BASE", {})
    print("=" * 120)
    print(f"PAIRED VERDICT vs BASE   (all arms on the identical seed block, n={len(base)})")
    print("=" * 120)
    print(f"{'arm':<10} {'n':>4} {'mean':>7} {'median':>7} {'p10':>7} {'min':>6} {'max':>6} "
          f"{'paired':>8} {'paired%':>8} {'W':>4} {'L':>4} {'T':>4} {'win%':>6}")
    order = sorted(ticks, key=lambda a: -st.mean(list(ticks[a].values())))
    for a in order:
        d = ticks[a]
        ks = sorted(d)
        vals = [d[k] for k in ks]
        pd = [d[k] - base[k] for k in ks if k in base]
        w = sum(1 for v in pd if v > 0)
        l = sum(1 for v in pd if v < 0)
        t = sum(1 for v in pd if v == 0)
        bm = st.mean([base[k] for k in ks if k in base]) if pd else 0
        print(f"{a:<10} {len(ks):>4} {st.mean(vals):>7.0f} {st.median(vals):>7.0f} "
              f"{pct(vals,0.1):>7} {min(vals):>6} {max(vals):>6} "
              f"{(st.mean(pd) if pd else 0):>+8.0f} "
              f"{(100*st.mean(pd)/bm if pd and bm else 0):>+8.1f} {w:>4} {l:>4} {t:>4} "
              f"{(100*w/max(1,w+l)):>6.1f}")

    print()
    print("DEATHS BY CAUSE (pooled over seeds)")
    for a in order:
        c = causes[a]
        tot = sum(c.values()) or 1
        print(f"  {a:<10} n={tot:>6}  " + "  ".join(f"{k} {100*v/tot:5.1f}%"
                                                    for k, v in sorted(c.items(), key=lambda x: -x[1])))

    # ---------------- mechanism ----------------
    bw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        bw[r["arm"]][r["t0"] // WINDOW * WINDOW].append(r)

    def col(arm, w, key, agg="mean"):
        rs = bw[arm].get(w, [])
        v = [r[key] for r in rs if r.get(key) is not None]
        if not v:
            return float("nan")
        return st.mean(v)

    print()
    print("=" * 120)
    print("MECHANISM BY ARM (per 1,000-tick window; population/energy only over seeds STILL ALIVE)")
    print("=" * 120)
    hdr = (f"{'arm':<10} {'ticks':>6} {'pop':>5} {'e_abs':>6} {'ef':>5} {'rf':>5} {'cap':>6} "
           f"{'born/1k':>8} {'died/1k':>8} {'gateRef%':>8} {'gsRef%':>7} {'pred':>5}")
    print(hdr)
    print("-" * 120)
    for a in order:
        for w in sorted(bw[a]):
            if w > 10000:
                continue
            rs = bw[a][w]
            tk = sum(r["samples"] for r in rs) or 1
            elig = sum(r["elig"] for r in rs)
            born = sum(r["born"] for r in rs)
            died = sum(r["died"] for r in rs)
            gsref = 100 * sum(r["ref"]["gs_ok"] for r in rs) / max(1, elig)
            gateref = 100 * sum(r["ref"]["gate_ok"] for r in rs) / tk
            pred = sum(r["pred_kills"] for r in rs)
            print(f"{a:<10} {w:>6} {col(a,w,'pop'):>5.1f} {col(a,w,'e_mean'):>6.1f} "
                  f"{col(a,w,'ef_mean'):>5.3f} {col(a,w,'rf_mean'):>5.3f} {col(a,w,'cap_mean'):>6.0f} "
                  f"{born*1000/tk:>8.1f} {died*1000/tk:>8.1f} {gateref:>8.1f} {gsref:>7.1f} {pred:>5.0f}")
        print()


if __name__ == "__main__":
    main()