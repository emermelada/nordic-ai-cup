"""oracle_healthy.py — re-condition the existing counterfactual-oracle dumps.

Q2's blocker was that the sample is dominated by knockout states (focal dead / energy 0), so
"an alternative wins" just means "any non-death branch wins". This re-analysis restricts to
HEALTHY snapshots (focal in no lockout, decent energy fraction) and asks the question Q3 needs:
among states where the controller itself survives the branch, does any fixed alternative beat it?

Usage: oracle_healthy.py [glob]   (default /opt/nac/oracle_*.json)
"""
import glob, json, sys, collections, statistics

paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/opt/nac/oracle_*.json"))
states = []
skipped = collections.Counter()
skipped_total = 0
for p in paths:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for s in d:
        if "features" not in s:          # oracle_noise / older probe schemas
            skipped[type(d).__name__] += 1
            continue
        s["_src"] = p.split("/")[-1]
        states.append(s)
print(f"loaded {len(states)} states from {len(paths)} dumps "
      f"({sum(skipped.values())} entries skipped: other schema)")

def healthy(s):
    f = s["features"]
    return (not f.get("in_lockout")) and f.get("energy_frac", 0) >= 0.4

def frac(num, den):
    return f"{num}/{den} = {100*num/den:.0f}%" if den else "0/0"

# split
by = collections.defaultdict(list)
for s in states:
    by[healthy(s)].append(s)
print()
for h in (True, False):
    g = by[h]
    if not g:
        continue
    ft = [s["features"]["tick"] for s in g]
    ef = [s["features"]["energy_frac"] for s in g]
    print(f"{'HEALTHY' if h else 'unhealthy'} snapshots: n={len(g)}  tick median {statistics.median(ft):.0f}"
          f"  energy_frac median {statistics.median(ef):.2f}")

print()
print("=== the Q3 question: among snapshots where the CONTROLLER branch survives the horizon ===")
for h in (True, False):
    g = [s for s in by[h] if s.get("controller", {}).get("alive")]
    if not g:
        print(f"{'HEALTHY' if h else 'unhealthy'}: controller never survived the branch -> no sample")
        continue
    beat = [s for s in g if any(a.get("alive") and a.get("energy", 0) > s["controller"].get("energy", 0)
                                for a in s["alternatives"].values())]
    print(f"{'HEALTHY' if h else 'unhealthy'}: controller survived {len(g)} of {len(by[h])} states;"
          f" an alternative alive AND richer beats it in {frac(len(beat), len(g))}")
    if beat:
        cnt = collections.Counter()
        for s in beat:
            for k, a in s["alternatives"].items():
                if a.get("alive") and a.get("energy", 0) > s["controller"].get("energy", 0):
                    cnt[k] += 1
        print("    winner tally:", cnt.most_common())

print()
print("=== branch outcomes by horizon survival (all states, healthy only) ===")
g = by[True]
if g:
    tally = collections.Counter()
    for s in g:
        for k, a in s["alternatives"].items():
            tally[k] += 1 if a.get("alive") else 0
        tally["(controller)"] += 1 if s["controller"].get("alive") else 0
    n = len(g)
    for k, v in tally.most_common():
        print(f"   {k:18s} alive after branch: {frac(v, n)}")
    # lockout vs healthy: is the knockout really the artefact?
    dead_all = sum(1 for s in g if not s["controller"].get("alive")
                   and all(not a.get("alive") for a in s["alternatives"].values()))
    print(f"   states where EVERY branch dies (including controller): {frac(dead_all, n)}")
