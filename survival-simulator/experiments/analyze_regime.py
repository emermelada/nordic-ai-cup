#!/usr/bin/env python3
"""analyze_regime.py - the regime/threshold questions of the 1905 update, answered from raw trajectories.

Answers, with n and confidence intervals printed:
  1. P(score > 1400/1500/1700/1800/1900) + Wilson 95% CI   (the objective, stated as probabilities)
  2. one regime or two?  2-means split on the score axis + bimodality coefficient + the overlap of the
     two clusters' score ranges (a "regime" must be separable, not merely a tail)
  3. window features at fixed times (slope of population, birth/death ratio, income vs expenditure,
     breeding capacity, energy variance) - AUC and Spearman against the final score
  4. the "small stable fleet" test: does the tail (t>=900) look like the high runs' pattern
     (pop in [2,12], births ~ deaths) and does that pattern separate >=1500 from 1100-1300?
  5. named-seed trajectory tables (the archived 1905/1891 runs vs typical and failed runs)
  6. conditional probabilities P(score>=1500 | feature at t) for the actionable features

Usage:
  python3 analyze_regime.py --raw /tmp/id_raw.jsonl --holdout 23,1 --times 100,200,300,400,500,700,900,1200
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def auc(pos, neg):
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    vals = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(vals)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n1, n0 = len(pos), len(neg)
    u = sum(ranks[k] for k in range(n) if vals[k][1] == 1) - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def spearman(xs, ys):
    n = len(xs)
    if n < 5:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def at(rec, t, feat):
    b = [x for x in rec["buckets"] if abs(x["t"] - t) < 0.05]
    return float(b[0].get(feat, 0.0)) if b else float("nan")


def window(rec, t0, t1, feat):
    """mean of a bucket feature over [t0, t1]"""
    xs = [float(x.get(feat, 0.0)) for x in rec["buckets"] if t0 <= x["t"] <= t1]
    return sum(xs) / len(xs) if xs else float("nan")


def slope(rec, t0, t1, feat="pop"):
    a, b = at(rec, t0, feat), at(rec, t1, feat)
    if math.isnan(a) or math.isnan(b) or t1 <= t0:
        return float("nan")
    return (b - a) / (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--holdout", default="", help="seeds to treat as permanent examples (listed, not tuned on)")
    ap.add_argument("--times", default="100,200,300,400,500,700,900,1200,1500")
    ap.add_argument("--named", default="23,1,35,73,3,19,48,49,74,68,52,27,33,42")
    args = ap.parse_args()
    times = [float(x) for x in args.times.split(",")]
    holdout = {int(x) for x in args.holdout.split(",") if x.strip()}

    recs = []
    for line in open(args.raw):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("buckets"):
            recs.append(r)
    n = len(recs)
    sc = sorted(float(r["score"]) for r in recs)
    print("=" * 78)
    print("REGIME ANALYSIS over %d runs of the immutable hive (real simulator, fresh seeds)" % n)
    print("=" * 78)
    print("\n1. OBJECTIVE AS PROBABILITIES (Wilson 95% CI):")
    for thr in (1400, 1500, 1700, 1800, 1900, 2000):
        k = sum(1 for s in sc if s > thr)
        p, lo, hi = wilson(k, n)
        print("   P(score > %-4d) = %5.3f   [%.3f, %.3f]   %d/%d" % (thr, p, lo, hi, k, n))
    mean = sum(sc) / n
    med = sc[n // 2]
    print("   mean %.0f  median %.0f  p10 %.0f  p25 %.0f  p75 %.0f  p90 %.0f  max %.0f"
          % (mean, med, sc[int(.1 * (n - 1))], sc[int(.25 * (n - 1))], sc[int(.75 * (n - 1))],
             sc[int(.9 * (n - 1))], sc[-1]))

    # ---- 2. one regime or two? 2-means on the score axis -------------------------------------------
    c1, c2 = sc[int(.25 * (n - 1))], sc[int(.75 * (n - 1))]
    for _ in range(200):
        g1 = [s for s in sc if abs(s - c1) <= abs(s - c2)]
        g2 = [s for s in sc if abs(s - c1) > abs(s - c2)]
        if not g1 or not g2:
            break
        n1, n2 = sum(g1) / len(g1), sum(g2) / len(g2)
        if abs(n1 - c1) < 1e-9 and abs(n2 - c2) < 1e-9:
            c1, c2 = n1, n2
            break
        c1, c2 = n1, n2
    g1 = sorted(s for s in sc if abs(s - c1) <= abs(s - c2))
    g2 = sorted(s for s in sc if abs(s - c1) > abs(s - c2))
    m3 = sum((s - mean) ** 3 for s in sc) / n
    m4 = sum((s - mean) ** 4 for s in sc) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in sc) / max(1, n - 1))
    bc = ((m3 / sd ** 3) ** 2 + 1) / (m4 / sd ** 4) if sd else float("nan")
    print("\n2. ONE REGIME OR TWO?")
    print("   2-means: cluster A n=%d mean %.0f range [%.0f, %.0f] | cluster B n=%d mean %.0f range [%.0f, %.0f]"
          % (len(g1), sum(g1) / max(1, len(g1)), g1[0] if g1 else 0, g1[-1] if g1 else 0,
             len(g2), sum(g2) / max(1, len(g2)), g2[0] if g2 else 0, g2[-1] if g2 else 0))
    print("   gap between the clusters: %.0f  (score units; <150 would be noise-level separation)" % (g2[0] - g1[-1] if g1 and g2 else 0))
    print("   bimodality coefficient %.3f  (bimodal if > 0.555); skew %.2f, kurtosis %.2f" % (bc, m3 / sd ** 3, m4 / sd ** 4))

    # ---- 3. window features ------------------------------------------------------------------------
    print("\n3. WINDOW / DERIVED FEATURES vs FINAL SCORE (Spearman rho; |rho|>0.25 with n>=60 is interesting)")
    feats = []
    for t in times:
        for f in ("pop", "E_mean", "E_med", "E_sd", "ef_lt35", "e_gt100", "e_gt140", "e_gt220",
                  "births", "d_sy", "d_so", "d_eat", "fruit_E", "c_move", "c_spawn",
                  "g_speed", "g_hear", "g_vis", "g_maxE", "sd_x", "disp", "req_d"):
            feats.append(("t%d.%s" % (t, f), [at(r, t, f) for r in recs]))
    feats.append(("slope_pop_100_300", [slope(r, 100, 300) for r in recs]))
    feats.append(("slope_pop_200_500", [slope(r, 200, 500) for r in recs]))
    feats.append(("birth/death_100_300", [window(r, 100, 300, "births") / max(1e-6, window(r, 100, 300, "births") + window(r, 100, 300, "d_sy") + window(r, 100, 300, "d_so") + window(r, 100, 300, "d_eat")) for r in recs]))
    feats.append(("birth/death_300_900", [window(r, 300, 900, "births") / max(1e-6, window(r, 300, 900, "births") + window(r, 300, 900, "d_sy") + window(r, 300, 900, "d_so") + window(r, 300, 900, "d_eat")) for r in recs]))
    feats.append(("net_income_100_300", [window(r, 100, 300, "fruit_E") - window(r, 100, 300, "c_move") - window(r, 100, 300, "c_biome") - window(r, 100, 300, "c_spawn") for r in recs]))
    feats.append(("net_income_300_900", [window(r, 300, 900, "fruit_E") - window(r, 300, 900, "c_move") - window(r, 300, 900, "c_biome") - window(r, 300, 900, "c_spawn") for r in recs]))
    feats.append(("move_per_fruit_100_300", [window(r, 100, 300, "c_move") / max(1e-6, window(r, 100, 300, "fruit_E")) for r in recs]))
    feats.append(("pop_min_600_1200", [min([float(x["pop"]) for x in r["buckets"] if 600 <= x["t"] <= 1200] or [0]) for r in recs]))
    feats.append(("tail_stability_900_1500", [window(r, 900, 1500, "births") / max(1e-6, window(r, 900, 1500, "d_sy") + window(r, 900, 1500, "d_so") + window(r, 900, 1500, "d_eat")) for r in recs]))
    scored = []
    for name, xs in feats:
        pairs = [(x, float(r["score"])) for x, r in zip(xs, recs) if not math.isnan(x)]
        if len(pairs) < 20:
            continue
        vals = [p[0] for p in pairs]
        if max(vals) == min(vals):      # column absent in this run's schema (older harness copy)
            continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        scored.append((abs(rho) if not math.isnan(rho) else 0.0, rho, name, len(pairs)))
    for a, rho, name, nn in sorted(scored, reverse=True)[:22]:
        print("   %-26s rho %+0.3f   n=%d" % (name, rho, nn))

    # ---- 4. small stable fleet test ----------------------------------------------------------------
    print("\n4. TAIL STABILITY TEST (does the high regime look like a small stable fleet?)")
    def tail(rec):
        return window(rec, 900, 1500, "pop"), window(rec, 900, 1500, "births"), window(rec, 900, 1500, "d_sy") + window(rec, 900, 1500, "d_so") + window(rec, 900, 1500, "d_eat")
    high = [r for r in recs if float(r["score"]) >= 1500 and r["T"] >= 15000]
    mid = [r for r in recs if 1000 <= float(r["score"]) < 1300 and r["T"] >= 15000]
    low = [r for r in recs if float(r["score"]) < 1000]
    for name, grp in ((">=1500 (T>=15000)", high), ("1000-1300 (T>=15000)", mid), ("<1000 (early death)", low)):
        if not grp:
            continue
        pops = [tail(r)[0] for r in grp]
        b = [tail(r)[1] for r in grp]
        d = [tail(r)[2] for r in grp]
        print("   %-22s n=%3d  tail pop %.1f  births %.1f  deaths %.1f  birth/death %.2f"
              % (name, len(grp), sum(pops) / len(pops), sum(b) / len(b), sum(d) / len(d),
                 (sum(b) / len(b)) / max(1e-9, sum(d) / len(d))))
    print("   NOTE: runs with T<15000 have no tail window; comparing them there is meaningless and is")
    print("         shown only to expose the circularity (a short run cannot have a 900-1500 s state).")

    # ---- 5. named trajectories ----------------------------------------------------------------------
    print("\n5. NAMED TRAJECTORIES (pop / median energy / breeding-capable agents at each t)")
    named = [int(x) for x in args.named.split(",") if x.strip()]
    by = {r["seed"]: r for r in recs}
    hdr = "   seed  score    T | " + " ".join("t%-4d" % t for t in (100, 200, 300, 500, 700, 900, 1200, 1500))
    print(hdr)
    for s in named:
        r = by.get(s)
        if not r:
            continue
        cells = []
        for t in (100, 200, 300, 500, 700, 900, 1200, 1500):
            pop = at(r, t, "pop")
            cells.append("%4.0f" % pop if not math.isnan(pop) else "   -")
        tag = " HOLDOUT" if s in holdout else ""
        print("   %-4d %6.0f %5d | %s%s" % (s, r["score"], r["T"], " ".join(cells), tag))
    print("   (population at each t; a run that dies early shows '-' from that t onward)")

    # ---- 6. conditional probabilities ----------------------------------------------------------------
    print("\n6. CONDITIONAL PROBABILITIES at the separator candidates")
    for feat, t in (("pop", 250), ("pop", 300), ("ef_lt35", 250), ("e_gt220", 300), ("e_gt140", 300)):
        vals = [(at(r, t, feat), float(r["score"])) for r in recs]
        vals = [(v, s) for v, s in vals if not math.isnan(v)]
        if len(vals) < 20:
            continue
        vv = sorted(v for v, _ in vals)
        for q in (0.5,):
            thr = vv[int(q * (len(vv) - 1))]
            hi = [s for v, s in vals if v >= thr]
            lo = [s for v, s in vals if v < thr]

            def stat(xs):
                if not xs:
                    return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
                k = sum(1 for s in xs if s >= 1500)
                p, l, h = wilson(k, len(xs))
                return (sum(xs) / len(xs), p, l, h, len(xs))

            mh, ph, lh, hh, nh = stat(hi)
            ml, pl, ll, hl, nl = stat(lo)
            print("   %-9s t=%-4.0f split %.2f | >= : mean %6.0f  P(>=1500) %.2f [%.2f,%.2f] n=%3d"
                  % (feat, t, thr, mh, ph, lh, hh, nh))
            print("   %-9s %6s       | <  : mean %6.0f  P(>=1500) %.2f [%.2f,%.2f] n=%3d"
                  % ("", "", ml, pl, ll, hl, nl))


if __name__ == "__main__":
    main()
