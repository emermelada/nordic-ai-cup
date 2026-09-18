"""Compact report over my parameter sweeps: bench_std_results.jsonl + robust_*.jsonl.

Filters to the tags I produced (R1-/R2-/R3-/ROB-) at horizon 16000, prints mean/median/std/min/p5/p20
and a two-fold (first-half vs second-half seed) stability check for the bench rows.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def show(tag, ticks, extra=""):
    if not ticks:
        return
    t = sorted(ticks)
    print("%-26s n=%d mean=%7.1f med=%7.1f std=%6.1f min=%6d p5=%7.1f p20=%7.1f max=%6d %s"
          % (tag, len(t), st.mean(t), st.median(t), st.pstdev(t), t[0], pct(t, .05), pct(t, .2),
             t[-1], extra))


def main():
    rows = []
    p = os.path.join(HERE, "bench_std_results.jsonl")
    for line in open(p):
        r = json.loads(line)
        if r["tag"].startswith(("R1-", "R2-", "R3-")) and r["horizon"] == 16000:
            rows.append(r)
    print("=========== bench_std (8 TRAIN seeds, H=16000), by mean ===========")
    for r in sorted(rows, key=lambda r: -r["ticks"]["mean"]):
        h = r["raw_ticks"][:4]; h2 = r["raw_ticks"][4:]
        extra = ("sp=%.0f pop=%.1f ms/t=%.2f folds %.0f/%.0f"
                 % (r["spawns_mean"], r["pop_peak_mean"], r["ms_per_tick"],
                    st.mean(h), st.mean(h2)))
        show(r["tag"], r["raw_ticks"], extra)
    print("\n=========== by p20 (floor-ish), descending ===========")
    for r in sorted(rows, key=lambda r: -pct(r["raw_ticks"], .2)):
        show(r["tag"], r["raw_ticks"])
    for f in sorted(os.listdir(HERE)):
        if f.startswith("robust_") and f.endswith(".jsonl"):
            rob = [json.loads(l) for l in open(os.path.join(HERE, f))]
            print("\n=========== %s ===========" % f)
            names = []
            for r in rob:
                names.append(r["shape"]["name"])
            for nm in sorted(set(names)):
                print("--- shape %s (%dx%d, n_agents=%d)" % (
                    nm, [r for r in rob if r["shape"]["name"] == nm][0]["shape"]["w"],
                    [r for r in rob if r["shape"]["name"] == nm][0]["shape"]["h"],
                    [r for r in rob if r["shape"]["name"] == nm][0]["shape"]["n"]))
                for r in sorted([x for x in rob if x["shape"]["name"] == nm],
                                key=lambda x: -x["ticks"]["mean"]):
                    show(r["tag"], r["raw_ticks"],
                         "sp=%.0f pop=%.1f ms/t=%.2f" % (r["spawns_mean"], r["pop_peak_mean"],
                                                         r["ms_per_tick"]))


if __name__ == "__main__":
    main()
