#!/usr/bin/env python3
"""midgame_report.py - aggregate midgame_diag.py output into the answer to "why do births stop?".

Reads the per-(seed,bucket) JSONL and prints, per 500-tick window:
  population, mean energy fraction, the energy gate it faces (rf), the fraction of agent-ticks
  refused by each gate, births and deaths per 1,000 ticks by cause, and the heritable-trait
  trajectory (the selection ratchet readout).

Usage: midgame_report.py midgame.jsonl [midgame_sum.json]
"""
import json
import sys
from collections import defaultdict

WINDOW = 500


def main():
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path)]
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["t0"]] = r

    # only seeds that are still alive in a window contribute to it (fair conditional view)
    win = defaultdict(list)
    for seed, bs in by_seed.items():
        for t0, r in bs.items():
            win[(t0 // WINDOW) * WINDOW].append(r)

    print("=" * 132)
    print("PER-WINDOW AGGREGATE  (mean over seeds still alive in that window)")
    print("=" * 132)
    hdr = (f"{'ticks':>6} {'seeds':>5} {'pop':>5} {'e':>6} {'ef':>5} {'rf':>5} "
           f"{'fvis':>5} {'pNear':>6} | {'elig%':>6} {'gate%':>6} {'cool%':>6} {'gsRef%':>7} "
           f"{'breed%':>7} | {'born/1k':>8} {'died/1k':>8} {'eat':>5} {'star':>5} {'aged':>5} "
           f"| {'cap':>6} {'capMax':>7} {'vis':>6}")
    print(hdr)
    print("-" * 132)
    for w in sorted(win):
        rs = win[w]
        n = len(rs)
        samp = sum(r["samples"] for r in rs) or 1
        elig = sum(r["elig"] for r in rs)
        g = lambda k: sum(r["ref"][k] for r in rs)  # noqa: E731
        born = sum(r["born"] for r in rs)
        died = sum(r["died"] for r in rs)
        eats = sum(r["deaths"]["eaten"] for r in rs)
        star = sum(r["deaths"]["starved"] for r in rs)
        aged = sum(r["deaths"]["aged"] for r in rs)
        breed = sum((r["breeder_frac"] or 0.0) * r["elig"] for r in rs)
        ticks = sum(r["samples"] for r in rs)
        per1k = 1000.0 / max(1, ticks)
        print(f"{w:>6} {n:>5} "
              f"{sum(r['pop'] for r in rs)/n:>5.1f} {sum(r['e_mean'] for r in rs)/n:>6.1f} "
              f"{sum(r['ef_mean'] for r in rs)/n:>5.3f} {sum(r['rf_mean'] for r in rs)/n:>5.3f} "
              f"{sum(r['fruits_vis'] for r in rs)/n:>5.2f} {sum(r['pred_near_frac'] for r in rs)/n:>6.3f} | "
              f"{100*elig/samp:>6.1f} {100*g('gate_ok')/samp:>6.1f} {100*g('cooldown')/samp:>6.1f} "
              f"{100*g('gs_ok')/max(1,elig):>7.1f} {100*breed/max(1,elig):>7.2f} | "
              f"{born*per1k:>8.1f} {died*per1k:>8.1f} {eats:>5} {star:>5} {aged:>5} | "
              f"{sum(r['cap_mean'] for r in rs)/n:>6.0f} {max(r['cap_max'] for r in rs):>7.0f} "
              f"{sum(r['vis_mean'] for r in rs)/n:>6.0f}")
    print("-" * 132)
    print("elig%   = share of agent-ticks where the agent was OFF COOLDOWN and could AFFORD a spawn")
    print("gate%   = share refused by the energy gate (ef <= rf);  cool% = refused by the spawn clock")
    print("gsRef%  = of the ELIGIBLE ticks, share refused by the GENOME gate (rank > gs_topk)")
    print("breed%  = of the ELIGIBLE ticks, share allowed to breed")
    print("cap/vis = mean heritable max_energy and vision_range (the selection-ratchet readout)")

    # ---- refusal detail ----
    print()
    print("REFUSALS BY GATE, as % of all agent-ticks")
    print(f"{'ticks':>6} {'pop':>5} {'gate_ok':>8} {'cooldown':>9} {'pop_ok':>7} {'crowd':>6} "
          f"{'unsafe':>7} {'gs_ok':>7}")
    for w in sorted(win):
        rs = win[w]
        n = len(rs)
        samp = sum(r["samples"] for r in rs) or 1
        g = lambda k: 100 * sum(r["ref"][k] for r in rs) / samp  # noqa: E731
        print(f"{w:>6} {sum(r['pop'] for r in rs)/n:>5.1f} {g('gate_ok'):>8.2f} {g('cooldown'):>9.2f} "
              f"{g('pop_ok'):>7.2f} {g('crowd_ok'):>6.2f} {g('safe'):>7.2f} {g('gs_ok'):>7.2f}")

    # ---- survival distribution ----
    if len(sys.argv) > 2:
        sums = [json.loads(l) for l in open(sys.argv[2])]
        import statistics as st
        ticks = sorted(s["ticks"] for s in sums)
        print()
        print("=" * 132)
        print(f"SURVIVAL over {len(ticks)} seeds: mean {st.mean(ticks):.0f} median {st.median(ticks):.0f} "
              f"p10 {ticks[int(0.1*len(ticks))]} p90 {ticks[int(0.9*len(ticks))]} "
              f"min {ticks[0]} max {ticks[-1]}")
        dead = dataset = defaultdict(int)
        for s in sums:
            for k, v in s["causes"].items():
                dead[k] += v
        tot = sum(dead.values())
        print(f"DEATHS n={tot}: " + "  ".join(f"{k} {100*v/tot:.1f}%" for k, v in
                                              sorted(dead.items(), key=lambda x: -x[1])))
        # time a seed spends below N agents (the remnant phase, as share of its own run)
        rem = defaultdict(list)
        for seed, bs in by_seed.items():
            t_end = max(bs) + 100
            for N in (1, 2, 5):
                t = sum(100 for r in bs.values() if r["pop"] <= N)
                rem[N].append(t / t_end)
        for N in (1, 2, 5):
            print(f"  share of run spent at pop<={N}: {100*st.mean(rem[N]):.1f}% "
                  f"(median {100*st.median(rem[N]):.1f}%)")


if __name__ == "__main__":
    main()