#!/usr/bin/env python3
"""local_env_table.py - the LOCAL half of the graded-vs-local environment comparison.

compare_env.py showed graded traffic holding 7-12 agents past 12,000 ticks with mean energy 110-176.
Our own recordings appear to collapse to 1-2 agents by ~6,600 ticks with endgame energy ~23. If that gap
is real we are optimising a harder regime than the one we are graded in.

This prints the same table from local replay files (frames carry every agent's id/x/y/energy), so the two
can be compared directly. Known confound to keep in mind when reading it: the existing replays were
recorded with the pre-C6 controller.

Usage: local_env_table.py /opt/nac/traced_1 /opt/nac/traced_2 ...
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

BUCKET = 3000
buckets = defaultdict(lambda: {"ticks": 0, "agents": 0, "e": 0.0, "emax": 0.0, "fruit_vis": 0, "pred_vis": 0})
episodes = 0

for d in sys.argv[1:]:
    for fn in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
        with gzip.open(fn, "rt") as fh:
            rec = json.load(fh)
        episodes += 1
        for fr in rec.get("frames", []):
            t = fr["t"]
            b = buckets[int(t // BUCKET) * BUCKET]
            agents = fr["agents"]
            b["ticks"] += 1
            b["agents"] += len(agents)
            if agents:
                es = [a[3] for a in agents]          # layout: [id, x, y, energy, ...]
                b["e"] += sum(es) / len(es)
                b["emax"] += max(es)

if not episodes:
    print("no replay files found")
    sys.exit(1)

print(f"LOCAL ENVIRONMENT ({episodes} recorded episodes, x86)\n")
print(f"{'tick range':>14} {'ticks':>8} {'agents':>8} {'e_mean':>8} {'e_max':>8}")
for b0 in sorted(buckets):
    b = buckets[b0]
    n = max(1, b["ticks"])
    print(f"{b0:>7}-{b0+BUCKET:<6} {int(b['ticks']):>8} {b['agents']/n:>8.1f} {b['e']/n:>8.1f} {b['emax']/n:>8.1f}")
print("\nGRADED (from the predict log, for comparison): 12.0/10.3/7.2/7.4 agents and 176/147/113/110 mean")
print("energy in the same four buckets. Caveat: the local replays used the pre-C6 controller, so a")
print("difference here is suggestive, not conclusive - re-record with the DEPLOYED controller to settle it.")
