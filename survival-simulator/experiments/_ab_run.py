"""Driver: fresh-process A/B identity + timing for shipped vs best_controller_fast.

Part 1 (identity): for each seed and each policy, run TWO fresh processes and compare ticks / score /
spawns / fruits / predated / action digest. Identical digest => every action of the whole episode
matched exactly, end to end.
Part 2 (timing): report ms/tick per policy from those same runs.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
HORIZON = int(os.environ.get("AB_HORIZON", "16000"))
SEEDS = [int(x) for x in os.environ.get("AB_SEEDS", "100,200,300").split(",")]
REPS = int(os.environ.get("AB_REPS", "2"))

recs = {}
for seed in SEEDS:
    for pol in ("shipped", "fast"):
        out = []
        for rep in range(REPS):
            r = subprocess.run([PY, os.path.join(HERE, "_ab_episode.py"), pol, str(seed), str(HORIZON)],
                               capture_output=True, text=True, cwd=HERE)
            line = (r.stdout or "").strip().splitlines()
            if r.returncode != 0 or not line:
                print("RUN FAILED", pol, seed, rep, r.returncode, r.stdout[-500:], r.stderr[-800:])
                sys.exit(2)
            out.append(json.loads(line[-1]))
        recs[(seed, pol)] = out

print("=== PART 1: fresh-process identity (horizon=%d, n_agents=5, stop_on_death) ===" % HORIZON)
fields = ("ticks", "score", "spawns", "fruits", "predated", "final_agents", "n_actions", "action_sha")
ok = True
for seed in SEEDS:
    a, b = recs[(seed, "shipped")], recs[(seed, "fast")]
    rep_ok = all(a[0][k] == a[i][k] for i in range(1, len(a)) for k in fields)
    ab_ok = all(a[0][k] == b[0][k] for k in fields)
    ok = ok and ab_ok
    print("seed=%-4d shipped ticks=%-6d score=%-9.2f spawns=%-4d fruits=%-5d pred=%-5d  act_sha=%s"
          % (seed, a[0]["ticks"], a[0]["score"], a[0]["spawns"], a[0]["fruits"],
             a[0]["predated"], a[0]["action_sha"]))
    print("        fast    ticks=%-6d score=%-9.2f spawns=%-4d fruits=%-5d pred=%-5d  act_sha=%s"
          % (b[0]["ticks"], b[0]["score"], b[0]["spawns"], b[0]["fruits"],
             b[0]["predated"], b[0]["action_sha"]))
    print("        shipped-vs-fast: %s   (shipped run-to-run reproducible: %s, %d reps)"
          % ("EXACT MATCH" if ab_ok else "MISMATCH", rep_ok, len(a)))
print("PART 1 RESULT: %s" % ("EXACT identity on all %d seeds" % len(SEEDS) if ok else "FAILED"))

print("\n=== PART 2: ms/tick (same definition as bench_std.py: 1000*wall/total_ticks) ===")
for seed in SEEDS:
    for pol in ("shipped", "fast"):
        for r in recs[(seed, pol)]:
            print("  seed=%-4d %-8s ticks=%-6d wall=%6.2fs ms/tick=%.3f"
                  % (seed, pol, r["ticks"], r["wall_s"], r["ms_per_tick"]))
for pol in ("shipped", "fast"):
    tt = sum(r["ticks"] for s in SEEDS for r in recs[(s, pol)])
    ww = sum(r["wall_s"] for s in SEEDS for r in recs[(s, pol)])
    print("  AGG %-8s total_ticks=%-6d wall=%6.2fs ms/tick=%.3f" % (pol, tt, ww, 1000 * ww / tt))
sys.exit(0 if ok else 1)
