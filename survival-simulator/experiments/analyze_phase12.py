#!/usr/bin/env python3
"""analyze_phase12.py - paste-8 Phase 1 (score distribution) + Phase 2 (earliest separator) + Phase 7 (bimodality).

Reads the raw per-seed records written by traj_map.py (trajectories, economy, genes, behaviour) and:

  PHASE 1  mean/median/p10-p90/min/max/CV + histogram + a bimodality check, per-seed score table.
  PHASE 2  splits HIGH (top 25%) vs LOW (bottom 25%) by final score and, for every (feature, t) pair,
           reports the separation power: rank AUC of the feature at time t for the HIGH/LOW label,
           the best single-feature threshold (balanced accuracy), and the two conditional success
           rates P(final score >= 1500 | side of threshold).  The output names the EARLIEST t at which
           some feature reaches AUC >= threshold, i.e. the earliest measurable failure precursor.
  PHASE 7  checks whether the score distribution is one regime or two (bimodality coefficient +
           the histogram shape) before any averaging is done.
  GOLD     if --gold is given, prints where the 1815.55 run's population curve sits in the
           distribution at each t (percentile), which needs no seed identification.

No sklearn: AUC is the Mann-Whitney rank statistic, thresholds are brute-forced midpoints.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NUM = ("pop", "active", "E_mean", "E_med", "E_max", "E_min", "E_sd", "ef_lt35", "ef_lt20",
       "see_fruit_1", "fruit_seen", "pred_seen", "tree_seen", "agent_seen", "stat_frac", "req_d",
       "disp", "g_speed", "g_sprint", "g_hear", "g_vis", "g_maxE", "g_maxE_sd", "g_age", "sd_x", "sd_y")
RATE = ("births", "d_sy", "d_so", "d_eat", "fruit_n", "fruit_E", "c_move", "c_turn", "c_biome",
        "c_old", "c_spawn", "e_in_eaten")
FEATS = list(NUM) + list(RATE)


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(max(int(round(q * (len(ys) - 1))), 0), len(ys) - 1)
    return ys[k]


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) via the rank statistic. numpy-free."""
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
    s = sum(ranks[k] for k in range(n) if vals[k][1] == 1)
    u = s - n1 * (n1 + 1) / 2.0
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


def best_split(vals, label, skip=3):
    """Threshold maximising balanced accuracy of predicting `label` (1=HIGH) from vals."""
    pairs = sorted(zip(vals, label))
    n1 = sum(label)
    n0 = len(label) - n1
    if n1 < skip or n0 < skip:
        return None
    cum1 = 0
    best = None
    for i in range(len(pairs) - 1):
        cum1 += pairs[i][1]
        if pairs[i][0] == pairs[i + 1][0]:
            continue
        c0 = (i + 1) - cum1
        tpr = cum1 / n1
        tnr = 1 - c0 / n0
        bal = 0.5 * (tpr + tnr)
        if best is None or bal > best[0]:
            best = (bal, 0.5 * (pairs[i][0] + pairs[i + 1][0]), tpr, tnr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--high-q", type=float, default=0.75)
    ap.add_argument("--low-q", type=float, default=0.25)
    ap.add_argument("--amin", type=float, default=0.72, help="AUC that counts as a usable separator")
    ap.add_argument("--times", default="100,150,200,250,300,400,500,600,700,800,900,1000,1200")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    recs = []
    with open(args.raw) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    recs = [r for r in recs if r.get("buckets")]
    scores = sorted(float(r["score"]) for r in recs)
    n = len(scores)
    print("=" * 78)
    print("PHASE 1 - score distribution of IMMUTABLE hive, real simulator, %d fresh seeds" % n)
    print("=" * 78)
    mean = sum(scores) / n
    med = pct(scores, 0.5)
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(1, n - 1))
    print("mean %.1f   median %.1f   sd %.1f   CV %.3f" % (mean, med, sd, sd / mean))
    print("min %.1f  p10 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f  max %.1f"
          % (scores[0], pct(scores, .1), pct(scores, .25), med, pct(scores, .75), pct(scores, .9), scores[-1]))
    print("P(score>=1400) %.3f   P(score>=1700) %.3f   P(score>=1800) %.3f"
          % (sum(1 for s in scores if s >= 1400) / n, sum(1 for s in scores if s >= 1700) / n,
             sum(1 for s in scores if s >= 1800) / n))
    # histogram
    lo, hi = scores[0], scores[-1]
    nb = 12
    w = max(1e-9, (hi - lo) / nb)
    hist = [0] * nb
    for s in scores:
        hist[min(nb - 1, int((s - lo) / w))] += 1
    print("\nhistogram (bin width %.0f):" % w)
    for i, c in enumerate(hist):
        print("  %6.0f-%6.0f | %4d %s" % (lo + i * w, lo + (i + 1) * w, c, "#" * int(60 * c / max(hist))))
    # bimodality: coefficient = (skew^2 + 1) / kurtosis ; > 5/9 suggests bimodality
    m3 = sum((s - mean) ** 3 for s in scores) / n
    m4 = sum((s - mean) ** 4 for s in scores) / n
    skew = m3 / (sd ** 3 + 1e-12)
    kurt = m4 / (sd ** 4 + 1e-12)
    bc = (skew ** 2 + 1) / kurt if kurt else float("nan")
    # excess kurtosis of the sample (Fisher) drives the classic rule
    print("\nPHASE 7 - bimodality: skew %.3f  kurtosis %.3f  bimodality coefficient %.3f (bimodal if > 0.555)"
          % (skew, kurt, bc))
    print("  unimodal-vs-two-regime is judged from the histogram above + the coefficient; a coefficient")
    print("  barely above 0.555 with a smooth tail is a long tail, not a second mode.")

    hi_s = pct(scores, args.high_q)
    lo_s = pct(scores, args.low_q)
    HIGH = [r for r in recs if float(r["score"]) >= hi_s]
    LOW = [r for r in recs if float(r["score"]) <= lo_s]
    print("\nHIGH group: n=%d (score >= %.1f)   LOW group: n=%d (score <= %.1f)"
          % (len(HIGH), hi_s, len(LOW), lo_s))
    print("HIGH mean %.1f  LOW mean %.1f" % (sum(float(r["score"]) for r in HIGH) / max(1, len(HIGH)),
                                             sum(float(r["score"]) for r in LOW) / max(1, len(LOW))))

    times = [float(x) for x in args.times.split(",") if x.strip()]
    print("\n" + "=" * 78)
    print("PHASE 2 - separation power of each feature at each t (AUC for HIGH vs LOW)")
    print("=" * 78)
    results = []
    for t in times:
        for feat in FEATS:
            vh, vl = [], []
            sh, sl = [], []
            for r in HIGH:
                b = [x for x in r["buckets"] if abs(x["t"] - t) < 0.05]
                if b:
                    vh.append(float(b[0].get(feat, 0.0)))
                    sh.append(float(r["score"]))
            for r in LOW:
                b = [x for x in r["buckets"] if abs(x["t"] - t) < 0.05]
                if b:
                    vl.append(float(b[0].get(feat, 0.0)))
                    sl.append(float(r["score"]))
            if len(vh) < 4 or len(vl) < 4:
                continue
            a = auc(vh, vl)
            a = max(a, 1 - a)          # direction is not the point; power is
            rho = spearman(sh + sl, vh + vl)
            sp = best_split(vh + vl, [1] * len(vh) + [0] * len(vl))
            results.append({"t": t, "feat": feat, "auc": a, "rho": rho, "split": sp,
                            "hi_mean": sum(vh) / len(vh), "lo_mean": sum(vl) / len(vl)})
    # earliest t with a usable separator
    print("\nearliest usable separators (AUC >= %.2f):" % args.amin)
    first = {}
    for r in sorted(results, key=lambda z: z["t"]):
        if r["auc"] >= args.amin and r["feat"] not in first and not math.isnan(r["auc"]):
            first[r["feat"]] = r
    if not first:
        top = sorted([r for r in results if not math.isnan(r["auc"])], key=lambda z: -z["auc"])[:12]
        print("  NONE at AUC >= %.2f.  Best overall:" % args.amin)
        for r in top:
            print("    t=%4.0f %-12s AUC %.3f  HIGH %.2f vs LOW %.2f" % (r["t"], r["feat"], r["auc"], r["hi_mean"], r["lo_mean"]))
    else:
        for feat, r in sorted(first.items(), key=lambda kv: kv[1]["t"]):
            line = "  t=%4.0f %-12s AUC %.3f   HIGH %.2f vs LOW %.2f" % (r["t"], feat, r["auc"], r["hi_mean"], r["lo_mean"])
            if r["split"]:
                bal, thr, tpr, tnr = r["split"]
                line += "   threshold %.2f -> P(HIGH)=%.2f above / %.2f below" % (thr, tpr, 1 - tnr)
            print(line)

    print("\ntop 15 (feature, t) pairs by AUC:")
    for r in sorted([r for r in results if not math.isnan(r["auc"])], key=lambda z: -z["auc"])[:15]:
        print("  t=%4.0f %-12s AUC %.3f  rho %+.2f  HIGH %.2f vs LOW %.2f"
              % (r["t"], r["feat"], r["auc"], r["rho"] if not math.isnan(r["rho"]) else 0.0, r["hi_mean"], r["lo_mean"]))

    if args.gold:
        gold = json.load(open(args.gold))
        gs = {float(s["t"]): float(s["pop"]) for s in gold["series"]}
        print("\n" + "=" * 78)
        print("GOLD 1815.55 run vs the distribution (population percentile at each t)")
        print("=" * 78)
        for t in times:
            b = [(float(r["score"]), [x for x in r["buckets"] if abs(x["t"] - t) < 0.05]) for r in recs]
            b = [(s, x[0]["pop"]) for s, x in b if x]
            if not b or t not in gs:
                continue
            pops = sorted(p for _, p in b)
            gp = gs[t]
            rank = sum(1 for p in pops if p <= gp) / len(pops)
            print("  t=%4.0f gold pop %5.1f  percentile %5.1f%%  (median %.1f, p90 %.1f, max %.1f)"
                  % (t, gp, 100 * rank, pct(pops, .5), pct(pops, .9), pops[-1]))

    if args.json_out:
        json.dump({"n": n, "mean": mean, "median": med, "sd": sd, "scores": scores,
                   "hist": hist, "bin_width": w, "bimodality_coef": bc, "skew": skew, "kurt": kurt,
                   "results": [{k: v for k, v in r.items()} for r in results]},
                  open(args.json_out, "w"), indent=1)
        print("\nwrote %s" % args.json_out)


if __name__ == "__main__":
    main()
