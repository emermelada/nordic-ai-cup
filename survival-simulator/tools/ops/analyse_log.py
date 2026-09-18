"""Extract every run's final score from a predict log (the endpoint logs each /predict call).

A new run is detected when the score RESETS (each episode restarts from 0), so this reconstructs the
full validation history from the server side -- independent of the platform's portal, which is a
JS SPA I cannot read without browser access.

Usage: python analyse_log.py <predict_log.jsonl>
"""
import datetime
import json
import sys

path = sys.argv[1]
runs = []
cur = None
bad = 0
with open(path) as fh:
    for line in fh:
        try:
            d = json.loads(line)
        except Exception:
            bad += 1
            continue
        s = float(d.get("score", 0.0))
        t = float(d.get("t", 0.0))
        if cur is None or s < cur["end"] - 1.0:      # score went backwards -> new episode
            if cur is not None:
                runs.append(cur)
            cur = {"start": t, "end": s, "ticks": 1, "n": int(d.get("n", 0))}
        else:
            cur["end"] = max(cur["end"], s)
            cur["ticks"] += 1
            cur["n"] = int(d.get("n", 0))
if cur:
    runs.append(cur)

print(f"{path}: {len(runs)} runs, {bad} unparsable lines")
print(f"{'started (UTC)':<20} {'final score':>11} {'ticks':>7} {'dur':>7} {'agents':>6}")
for r in runs:
    if r["ticks"] < 50:                              # ignore my hand-written curl probes
        continue
    ts = datetime.datetime.utcfromtimestamp(r["start"]).strftime("%m-%d %H:%M:%S")
    print(f"{ts:<20} {r['end']:>11.2f} {r['ticks']:>7} {r['end'] - r['start']:>6.0f}s {r['n']:>6}")
