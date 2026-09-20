#!/usr/bin/env python3
"""analyze_gold.py - segment the frozen 1815.55 trace into games and extract the gold-standard run.

Input : experiments/gold1815/server_trace.csv   (wall_time, sim_time, n_agents, score, status, decide_ms,
                                                 n_actions, obs_types) - the deployed hive on :9052
Output: experiments/gold1815/gold_games.json    per-game summary (segmentation by sim_time reset)
        experiments/gold1815/gold_pop.json      the gold run's population/latency series, 10 s buckets
        stdout                                  the numbers cited in the write-up

The trace has no client column, so game boundaries are found by sim_time going backwards (which is also
exactly how hive itself detects a new game: server.py "resets itself when sim_time goes backwards").
The window in which a stray test client shared the process is flagged (rows whose sim_time jumps
non-monotonically INSIDE a segment) so the contaminated prefix is never silently trusted.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "gold1815", "server_trace.csv")
GOLD_SCORE = 1815.5488285690717


def load():
    rows = []
    with open(P) as f:
        for k, line in enumerate(f):
            parts = line.rstrip("\n").split(",")
            if len(parts) < 8:
                continue
            try:
                rows.append({"wall": float(parts[0]), "sim": float(parts[1]), "n": int(float(parts[2])),
                             "score": float(parts[3]), "status": parts[4], "ms": float(parts[5]),
                             "na": int(float(parts[6])), "obs": parts[7]})
            except ValueError:
                continue
    return rows


def segment(rows):
    games = []
    cur = []
    prev = None
    for r in rows:
        if prev is not None and r["sim"] < prev - 0.5:
            games.append(cur)
            cur = []
        cur.append(r)
        prev = r["sim"]
    if cur:
        games.append(cur)
    return games


def main():
    rows = load()
    games = segment(rows)
    out = []
    print("total rows %d -> %d segments" % (len(rows), len(games)))
    for gi, g in enumerate(games):
        if not g:
            continue
        sims = [r["sim"] for r in g]
        # monotonicity: fraction of rows that continue the previous game's timeline
        inc = sum(1 for a, b in zip(sims, sims[1:]) if b > a)
        d = {"i": gi, "rows": len(g), "wall0": g[0]["wall"], "wall1": g[-1]["wall"],
             "dur_s": round(g[-1]["wall"] - g[0]["wall"], 1),
             "sim0": round(sims[0], 1), "sim_end": round(max(sims), 1), "ticks": int(round(10 * max(sims))),
             "score_end": round(g[-1]["score"], 2), "score_max": round(max(r["score"] for r in g), 2),
             "n_first": g[0]["n"], "n_last": g[-1]["n"], "n_max": max(r["n"] for r in g),
             "mono_frac": round(inc / max(1, len(g) - 1), 3),
             "ms_mean": round(sum(r["ms"] for r in g) / len(g), 2), "statuses": sorted({r["status"] for r in g})}
        out.append(d)
        print(json.dumps(d))

    # gold game = the segment whose max score is closest to the official 1815.55
    gold = min(out, key=lambda d: abs(d["score_max"] - GOLD_SCORE))
    print("\nGOLD SEGMENT:", json.dumps(gold))

    g = games[gold["i"]]
    # de-interleave: keep the longest strictly-increasing run inside the gold segment (the grader's own game)
    best = []
    cur = []
    prev = None
    for r in g:
        if prev is not None and r["sim"] <= prev:
            if len(cur) > len(best):
                best = cur
            cur = []
        cur.append(r)
        prev = r["sim"]
    if len(cur) > len(best):
        best = cur
    print("gold segment rows %d, longest monotonic clean run %d rows (sim %.1f -> %.1f)"
          % (len(g), len(best), best[0]["sim"], best[-1]["sim"]))

    # 10 s buckets of the clean run
    buckets = {}
    for r in best:
        b = int(r["sim"] // 10) * 10
        buckets.setdefault(b, []).append(r)
    series = []
    for b in sorted(buckets):
        rs = buckets[b]
        series.append({"t": b, "pop": round(sum(x["n"] for x in rs) / len(rs), 4),
                       "pop_max": max(x["n"] for x in rs), "score": round(rs[-1]["score"], 3),
                       "ms_mean": round(sum(x["ms"] for x in rs) / len(rs), 3)})
    json.dump({"gold_segment": gold, "series": series, "clean_rows": len(best),
               "clean_sim": [best[0]["sim"], best[-1]["sim"]]},
              open(os.path.join(HERE, "gold1815", "gold_pop.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(HERE, "gold1815", "gold_games.json"), "w"), indent=1)

    # a few landmarks for the write-up
    def at(t):
        s = [x for x in series if x["t"] == t]
        return s[0] if s else None
    for t in (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700):
        r = at(t)
        if r:
            print("t=%4d  pop=%5.2f (max %2d)  score=%8.2f  ms=%.2f" % (t, r["pop"], r["pop_max"], r["score"], r["ms_mean"]))


if __name__ == "__main__":
    main()
