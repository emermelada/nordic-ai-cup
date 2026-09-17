"""Analyze the grader's request log captured by agent_server (/app/predict_log.jsonl).

Usage:  python analyze_grader_log.py [path]
Reads JSONL records {sim, score, n, gs, ge, obs{type:count}, emax, emean, spawn} and prints
a compact report that reverse-engineers the grader environment + scoring.
"""
import sys, json, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/predict_log.jsonl"
recs = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

if not recs:
    print("no records in", path)
    sys.exit(0)

n = len(recs)
sim0, sim1 = recs[0]["sim"], recs[-1]["sim"]
sc0, sc1 = recs[0]["score"], recs[-1]["score"]
print(f"== grader request log: {path}")
print(f"total requests (ticks): {n}")
print(f"sim_time:   first={sim0}  last={sim1}  (span {sim1-sim0:.1f})")
print(f"score:      first={sc0}  last={sc1}")
# is score ~ sim_time? (score accumulates dt like sim_time; deviations = fruit - predation)
print(f"score/sim at last tick: {sc1/sim1:.4f}" if sim1 else "score/sim: n/a")
print(f"score - sim (net fruit/predation margin at end): {sc1 - sim1:+.3f}")

ns = [r["n"] for r in recs]
print(f"n_agents:   min={min(ns)} max={max(ns)} median={st.median(ns)}")
# collapse tick
coll = next((r for r in recs if r["n"] == 0), None)
print(f"collapse (first n==0): {'sim='+str(coll['sim'])+' score='+str(coll['score']) if coll else 'never (team alive at end)'}")

gs = defaultdict(int)
for r in recs:
    gs[r.get("gs")] += 1
print(f"game_status values: {dict(gs)}")

# observation environment: mean entity counts observed across agents
agg = defaultdict(list)
for r in recs:
    for k, v in (r.get("obs") or {}).items():
        agg[k].append(v)
print("mean observed entities per tick:", {k: round(st.mean(v), 2) for k, v in agg.items()})

# energy trajectory
em = [r["emean"] for r in recs if r.get("emean") is not None]
if em:
    print(f"mean energy per agent: first={em[0]:.1f} last={em[-1]:.1f} max={max(em):.1f}")

tot_spawn = sum(r.get("spawn", 0) for r in recs)
print(f"our total spawn flags returned: {tot_spawn}")

# downsampled trajectory (~12 rows)
print("\n-- trajectory (downsampled) --")
print(f"{'tick':>6} {'sim':>8} {'score':>9} {'n':>3} {'emean':>7} {'pred_obs':>8} {'fruit_obs':>9} {'spawn':>5}")
step = max(1, n // 12)
for i in range(0, n, step):
    r = recs[i]
    o = r.get("obs") or {}
    print(f"{i:>6} {r['sim']:>8.1f} {r['score']:>9.2f} {r['n']:>3} "
          f"{(r.get('emean') or 0):>7.1f} {o.get('Predator',0):>8} {o.get('Fruit',0):>9} {r.get('spawn',0):>5}")