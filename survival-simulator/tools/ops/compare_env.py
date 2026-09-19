#!/usr/bin/env python3
"""compare_env.py - does the GRADER'S environment match our local simulator?

Why: local gains have been transferring weakly to the board (a +19% local effect shows up as a better
tail but not a better median). One candidate explanation is that the graded environment is not the same
as the one we optimise in - different population sizes, energy regimes, predator pressure or fruit
availability. The predict log contains the grader's own per-tick observations, so this measures it
instead of guessing.

Compares, per tick bucket: agents alive, mean/min energy, lockout share, predator-visible share,
fruit-visible share. Local reference values from our own recordings (replay_survey over 16 x86 episodes):
    lockout share of agent-life ~0.54, blind (no fruit visible) 76-87%, fruits visible 0.26-0.42/agent
Usage: compare_env.py /tmp/plog.jsonl
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/plog.jsonl"
buckets = defaultdict(lambda: {"n": 0, "agents": 0, "e": [], "lock": 0, "emax": [], "pred_vis": 0, "wall_vis": 0,
                              "fruit_vis": 0, "seen_pred": 0})


def bucket_of(tick):
    return int(tick // 3000) * 3000


rows = 0
for line in open(path):
    try:
        r = json.loads(line)
    except Exception:
        continue
    n = float(r.get("n", 0) or 0)
    if n <= 0:
        continue
    tick = float(r.get("sim", 0.0)) * 10.0
    b = buckets[bucket_of(tick)]
    b["n"] += 1
    b["agents"] += n
    e = float(r.get("emean", 0.0) or 0.0)
    mx = float(r.get("emax", 0.0) or 0.0)
    b["e"].append(e)
    b["emax"].append(mx)
    if mx and e < 0.2 * mx:
        b["lock"] += 1
    obs = r.get("obs") or {}
    b["fruit_vis"] += float(obs.get("Fruit", 0) or 0) / n
    b["pred_vis"] += float(obs.get("Predator", 0) or 0) / n
    b["wall_vis"] += float(obs.get("Edge", 0) or 0) / n
    rows += 1

if not rows:
    print("no usable rows (check the log's field names)")
    sys.exit(1)

print(f"GRADER ENVIRONMENT ({rows} graded ticks)\n")
print(f"{'tick range':>14} {'ticks':>7} {'agents/req':>11} {'e_mean':>8} {'e_max':>8} {'lockout':>8} "
      f"{'fruit/agent':>12} {'pred/agent':>11} {'wall/agent':>11}")
for b0 in sorted(buckets):
    b = buckets[b0]
    cnt = max(1, b["n"])
    print(f"{b0:>7}-{b0+3000:<6} {int(b['n']):>7} {b['agents']/cnt:>11.1f} "
          f"{sum(b['e'])/cnt:>8.1f} {sum(b['emax'])/cnt:>8.1f} {b['lock']/cnt:>8.2f} "
          f"{b['fruit_vis']/cnt:>12.2f} {b['pred_vis']/cnt:>11.2f} {b['wall_vis']/cnt:>11.2f}")
print("\nLOCAL REFERENCE (our own x86 recordings): lockout ~0.54 of agent-life; blind 76-87% of ticks so")
print("fruit visible 0.26-0.42 per agent; endgame mean energy ~23 with predators usually not visible.")
print("If the graded columns differ materially, local percentage gains transfer weakly and the board")
print("must be treated as the only instrument that counts.")
