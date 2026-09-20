#!/usr/bin/env python3
"""compare_paired.py - paired comparison of two traj_map arms on a COMMON horizon.

Why a common horizon: the baseline arm ran with --horizon 18000 and a candidate arm may run with
30000. Survivors are truncated at their horizon, so comparing raw final scores would hand the longer
horizon a free win. Every score here is recomputed at the SHORTER of the two horizons:

    score_h(seed) = 0.1 * min(T, h)  +  (sum of fruit energy eaten while t <= min(T,h)/10) / 1000

(the corpse penalty is excluded: it is ~0.1% of the score and identical in both arms at the common h).

Reports, per arm: mean / median / p10 / p25 / p75 / p90 / max / P(>=1400) / P(>=1500) / P(>=1800),
and, paired by seed: mean delta +- SE, t, median delta, W/L, and the delta on the lower tail
(mean of the worst decile) plus how the two arms split the runs that changed band.

  python3 compare_paired.py --a /tmp/id_raw.jsonl --b /tmp/resc.jsonl
"""
import argparse
import json
import math


def load(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("buckets") and r["seed"] not in out:   # first record wins (duplicate-safe)
            out[r["seed"]] = r
    return out


def score_at(rec, h):
    T = min(int(rec["T"]), h)
    fruit = sum(float(b.get("fruit_E", 0.0)) for b in rec["buckets"] if b["t"] * 10 <= T)
    return 0.1 * T + fruit / 1000.0


def pct(xs, q):
    ys = sorted(xs)
    k = min(max(int(round(q * (len(ys) - 1))), 0), len(ys) - 1)
    return ys[k]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def describe(name, xs):
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(1, n - 1))
    print("  %-10s n=%3d  mean %7.1f  median %7.1f  sd %6.1f  p10 %6.1f  p25 %6.1f  p75 %6.1f  p90 %6.1f  max %6.1f"
          % (name, n, mean, pct(xs, .5), sd, pct(xs, .1), pct(xs, .25), pct(xs, .75), pct(xs, .9), max(xs)))
    for thr in (1400, 1500, 1800):
        k = sum(1 for x in xs if x >= thr)
        p, lo, hi = wilson(k, n)
        print("      P(>=%d) = %.3f [%.3f, %.3f]  (%d/%d)" % (thr, p, lo, hi, k, n))
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline arm raw.jsonl")
    ap.add_argument("--b", required=True, help="candidate arm raw.jsonl")
    ap.add_argument("--na", default="baseline")
    ap.add_argument("--nb", default="candidate")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    seeds = sorted(set(A) & set(B))
    h = min(max(int(A[s]["T"]) for s in seeds), max(int(B[s]["T"]) for s in seeds))
    print("paired seeds: %d   common horizon: %d ticks (%.0f sim-seconds)" % (len(seeds), h, h / 10.0))
    sa = [score_at(A[s], h) for s in seeds]
    sb = [score_at(B[s], h) for s in seeds]
    print("\nper-arm distribution at the common horizon:")
    ma = describe(args.na, sa)
    mb = describe(args.nb, sb)
    d = [y - x for x, y in zip(sa, sb)]
    dm = sum(d) / len(d)
    ds = math.sqrt(sum((x - dm) ** 2 for x in d) / max(1, len(d) - 1))
    se = ds / math.sqrt(len(d))
    t = dm / se if se else float("nan")
    print("\nPAIRED (candidate - baseline) on the same seeds:")
    print("  mean delta %+7.1f +- %.1f (SE)   t = %+0.2f   median delta %+7.1f" % (dm, se, t, pct(d, .5)))
    print("  W/L/T: %d/%d/%d   worst-decile mean delta %+7.1f   best-decile mean delta %+7.1f"
          % (sum(1 for x in d if x > 0), sum(1 for x in d if x < 0), sum(1 for x in d if x == 0),
             sum(sorted(d)[:max(1, len(d) // 10)]) / max(1, len(d) // 10),
             sum(sorted(d)[-max(1, len(d) // 10):]) / max(1, len(d) // 10)))
    # band movement: does the candidate convert LOW into MID/HIGH, and does it cost HIGH?
    def band(x):
        return 0 if x < 900 else (1 if x < 1400 else (2 if x < 1800 else 3))
    mv = {}
    for s, x, y in zip(seeds, sa, sb):
        key = (band(x), band(y))
        mv[key] = mv.get(key, 0) + 1
    names = ["LOW<900", "MID 900-1400", "HIGH 1400-1800", "TOP>=1800"]
    print("\nband transitions (baseline -> candidate):")
    for (x, y), n in sorted(mv.items()):
        tag = " (up)" if y > x else (" (down)" if y < x else "")
        print("   %-14s -> %-14s %3d%s" % (names[x], names[y], n, tag))
    print("\nverdict key: |t| >= 2.0 with mean delta > 0 = candidate wins; t < 2 = not distinguishable at this n")


if __name__ == "__main__":
    main()
