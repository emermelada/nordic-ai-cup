#!/usr/bin/env python3
"""survival_analysis.py - does the small-fleet (2-6 agent) state actually predict long survival?

THE TAUTOLOGY PROBLEM. "Runs that are in a 2-6 agent state at t=600 live longer" is nearly circular: to
be measured at t=600 a run must already have survived 600 s. The version of the question that is NOT
circular conditions on being alive at t0 and then compares the LATER survival:

    among runs alive at t0, does population at t0 predict REMAINING ticks (T - t0*10)?

That is a conditional survival analysis (a hazard comparison), and it can be run on trajectories that
are alive at t0 without any run's future leaking into its own group label.

Questions answered:
  A. conditional hazard: for each t0 and each population bucket at t0, mean/median remaining ticks,
     P(remaining >= 600 s), P(extinction within the next 300 s).
  B. does entering the small-fleet state with BETTER genes last longer?  among runs in the 2-6 bucket at
     t0, Spearman(remaining ticks, trait/energy state at t0).
  C. the transition itself: for runs that are in the boom (pop >= 15) at t=400, what fraction reach the
     2-6 state alive at t=800, and how does surviving that transition split the final scores?
  D. is the 2-6 state a cause or a symptom?  compare runs that reach 2-6 at t0 and THEN die quickly
     against those that persist, on their state at t0 (energy, genes, income) - if nothing at t0
     separates them, the small fleet is a symptom of an environment/policy condition we have not found.

Usage: python3 survival_analysis.py --raw /tmp/id_raw.jsonl
"""
import argparse
import json
import math

TIMES = (300, 400, 500, 600, 700, 800, 900, 1000, 1200)
BUCKETS = ((1, 2), (3, 6), (7, 12), (13, 18), (19, 99))


def at(rec, t, feat):
    b = [x for x in rec["buckets"] if abs(x["t"] - t) < 0.05]
    return float(b[0].get(feat, 0.0)) if b else None


def spearman(xs, ys):
    n = len(xs)
    if n < 6:
        return float("nan")
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rk(xs), rk(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def atv(rec, t, feat, d=0.0):
    v = at(rec, t, feat)
    return d if v is None else v


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    recs = []
    for line in open(args.raw):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("buckets"):
            recs.append(r)
    n = len(recs)
    print("runs: %d   (mean score %.0f, median %.0f)" % (n, sum(r["score"] for r in recs) / n,
                                                         med([r["score"] for r in recs])))
    print("\nA. CONDITIONAL SURVIVAL: among runs ALIVE at t0, does population at t0 predict what happens NEXT?")
    for t0 in TIMES:
        t0t = t0 * 10
        rows = []
        for r in recs:
            p = at(r, t0, "pop")
            if p is None or p < 1.0 or r["T"] <= t0t:
                continue
            rows.append((p, r))
        if len(rows) < 25:
            continue
        print("\n   t0=%4d s   alive-at-t0 runs: %d   (median remaining %.0f ticks)"
              % (t0, len(rows), med([r["T"] - t0t for _, r in rows])))
        print("      %-8s %4s %14s %14s %12s %12s" % ("pop@t0", "n", "rem median", "rem mean", "P(rem>=600s)", "P(die<300s)"))
        for lo, hi in BUCKETS:
            g = [r for p, r in rows if lo <= p <= hi]
            if len(g) < 8:
                continue
            rem = [r["T"] - t0t for r in g]
            print("      %-8s %4d %14.0f %14.0f %12.2f %12.2f"
                  % ("%d-%d" % (lo, hi), len(g), med(rem), sum(rem) / len(rem),
                     sum(1 for x in rem if x >= 6000) / len(rem),
                     sum(1 for x in rem if x < 3000) / len(rem)))

    print("\nB. WITHIN THE SMALL-FLEET STATE: do better genes/energy at entry predict longer survival?")
    print("   (runs in the 3-6 population bucket at t0; rho of the t0 state vs remaining ticks)")
    for t0 in (500, 600, 700, 800):
        t0t = t0 * 10
        g = [(at(r, t0, "pop"), r) for r in recs]
        g = [r for p, r in g if p is not None and 3 <= p <= 6 and r["T"] > t0t]
        if len(g) < 15:
            continue
        rem = [r["T"] - t0t for r in g]
        out = []
        for f in ("g_hear", "g_speed", "g_vis", "g_maxE", "E_mean", "E_med", "ef_lt35", "fruit_E",
                  "c_move", "g_age", "disp", "see_fruit_1"):
            xs = [at(r, t0, f) for r in g]
            ok = [(x, y) for x, y in zip(xs, rem) if x is not None]
            if len(ok) < 15:
                continue
            rho = spearman([x for x, _ in ok], [y for _, y in ok])
            if not math.isnan(rho):
                out.append((abs(rho), rho, f))
        print("   t0=%4d n=%3d  " % (t0, len(g)) + "  ".join("%s %+0.2f" % (f, r) for _, r, f in sorted(out, reverse=True)[:6]))

    print("\nC. THE TRANSITION: boom (>=15 agents) at t=400 -> small fleet (2-6) alive at t=800")
    boom = []
    for r in recs:
        p4 = at(r, 400, "pop")
        if p4 is not None and p4 >= 15.0 and r["T"] > 4000:
            boom.append(r)
    made = [r for r in boom if (at(r, 800, "pop") or 0) >= 1.0 and (at(r, 800, "pop") or 99) <= 6.0]
    big = [r for r in boom if (at(r, 800, "pop") or 0) > 6.0]
    dead = [r for r in boom if r["T"] <= 8000]
    print("   entered boom at t=400: %d" % len(boom))
    for name, grp in (("reached 2-6 at t=800", made), ("still >6 at t=800", big), ("died before t=800", dead)):
        if grp:
            print("      %-22s n=%3d  mean score %6.0f  median %6.0f  median T %6d"
                  % (name, len(grp), sum(r["score"] for r in grp) / len(grp),
                     med([r["score"] for r in grp]), med([r["T"] for r in grp])))

    print("\nD. CAUSE OR SYMPTOM? among runs in 3-6 at t0=700, split by how it went afterwards")
    t0, t0t = 700, 7000
    g = [r for r in recs if (at(r, 700, "pop") or 0) >= 3 and (at(r, 700, "pop") or 99) <= 6 and r["T"] > t0t]
    if len(g) >= 12:
        g = sorted(g, key=lambda r: r["T"] - t0t)
        half = len(g) // 2
        lo = g[:half]
        hi = g[half:]
        print("   n=%d; split at median remaining %.0f ticks: early-death group (median rem %.0f) vs persisters (median rem %.0f)"
              % (len(g), med([r["T"] - t0t for r in g]), med([r["T"] - t0t for r in lo]), med([r["T"] - t0t for r in hi])))
        print("   state at t=700 (mean):        " + "   ".join(
            "%s early %.1f vs persist %.1f" % (f, sum(atv(r, t0, f) for r in lo) / len(lo), sum(atv(r, t0, f) for r in hi) / len(hi))
            for f in ("E_mean", "E_med", "ef_lt35", "g_hear", "g_maxE", "fruit_E", "c_move", "see_fruit_1", "pop")))
        # yield: how many agents are even present to carry genes forward
        print("   agents at t=700:              early %.2f vs persist %.2f   births next 300s: early %.2f vs persist %.2f"
              % (sum(atv(r, t0, "pop") for r in lo) / len(lo), sum(atv(r, t0, "pop") for r in hi) / len(hi),
                 sum(atv(r, 900, "births") for r in lo) / len(lo), sum(atv(r, 900, "births") for r in hi) / len(hi)))


if __name__ == "__main__":
    main()
