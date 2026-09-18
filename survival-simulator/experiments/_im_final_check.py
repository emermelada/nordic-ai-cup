"""Final checks: (1) the shipped ckpt loads through the served-style contract, (2) determinism.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imitation as im
import policy_imitation as pi
import env_wrapper as ew

pol = pi.ImitationPolicy(im.MODEL, device="cpu")
print("ckpt:", os.path.basename(pol.path), "obs_dim", pol.obs_dim, "thr", pol.spawn_threshold,
      "cd", pol.spawn_cooldown)

# contract check on a live episode: action shape/ranges + module-level entry point
calls = []
def probe(state):
    a = pol(state)
    calls.append(a)
    assert isinstance(a, list) and len(a) == 4, a
    assert 0.0 <= a[0] <= 20.0, a
    assert -4.0 * 3.15 <= a[1] <= 4.0 * 3.15, a
    assert a[2] == 0.0 and a[3] in (0.0, 1.0), a
    return a

r = ew.run_eval_episode(probe, n_agents=5, seed=1000, horizon=800, stop_on_death=True, reset_fn=pol.reset)
print("contract OK on %d calls; ticks=%d" % (len(calls), r["steps"]))

# determinism: identical seed twice -> identical ticks/spawns/scores
for rep in (1, 2):
    p2 = pi.ImitationPolicy(im.MODEL, device="cpu")
    rr = im.eval_policy(p2, [1000, 1100], horizon=3000, reset_fn=p2.reset, label="det%d" % rep)
    print("  run%d %s" % (rep, [(x["seed"], x["ticks"], x["spawns"], x["score"]) for x in rr]))
