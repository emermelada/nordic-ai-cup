"""Distribution of observation-list sizes seen by the policy (worst case matters for CPU)."""
import os, sys, time
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from env_wrapper import run_eval_episode
import best_controller as bc

states = []
run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=100, horizon=6000,
                 stop_on_death=True, recorder=lambda i, s, a, o: states.extend(s),
                 reset_fn=bc.reset_memory)
cnt = Counter(); ncalls = 0
tot = 0
for s in states:
    o = s.get("observations") or []
    n = Counter(x.get("type") for x in o)
    cnt["edges"] += n["Edge"]; cnt["obs"] += len(o); ncalls += 1
    for t in ("Fruit", "Predator", "Agent", "Tree"):
        cnt[t] += n[t]
print("calls=%d  mean obs/call=%.2f" % (ncalls, cnt["obs"]/ncalls))
for t in ("edges", "obs", "Fruit", "Predator", "Agent", "Tree"):
    print("  %-9s mean/call=%.2f" % (t, cnt[t]/ncalls))
mx = 0; mxi = 0
for i, s in enumerate(states):
    k = sum(1 for x in s.get("observations") or [] if x.get("type") == "Edge")
    if k > mx: mx, mxi = k, i
print("max edges in one state: %d (state #%d)" % (mx, mxi))
