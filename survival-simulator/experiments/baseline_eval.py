import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode, FleetEnv, HORIZON
from policies import random_policy, dummy_policy, heuristic_policy

SEEDS = [100, 200, 300, 400, 500]

def bench(name, fn, seeds=SEEDS):
    res = []
    for s in seeds:
        t0 = time.time()
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=HORIZON)
        r["t"] = time.time() - t0
        res.append(r)
    scs = [r["score"] for r in res]
    steps = [r["steps"] for r in res]
    fruits = [r["fruits_eaten"] for r in res]
    pred = [r["predated"] for r in res]
    alive = [r["alive"] for r in res]
    print(f"{name:12s} score={np.mean(scs):.1f}±{np.std(scs):.1f}  steps={np.mean(steps):.0f}±{np.std(steps):.0f}  fruits/eps={np.mean(fruits):.1f}  predated={np.mean(pred):.1f}  alive%={100*sum(alive)/len(alive):.0f}  avg {np.mean([r['t'] for r in res]):.1f}s/eps")
    return res

print("=== Multi-agent baselines (n_agents=5, horizon=3000 steps) ===")
bench("random", random_policy)
bench("dummy", dummy_policy)
bench("heuristic", heuristic_policy)
print()

# ---- EDA: single-agent reward distribution under heuristic ----
print("=== EDA (heuristic, single-agent, seeds 100,101,102) ===")
rewards = {"all": [], "base": [], "fruit": [], "neg": []}
for s in [100, 101, 102]:
    env = FleetEnv(seed=s, horizon=3000)
    obs, _ = env.reset()
    for i in range(3000):
        st = env._state_of(env.learner_id)
        if st is None:
            break
        act = heuristic_policy(st)
        # single-agent wrapper expects [dist, turn, spawn]
        a3 = [act[0], act[2], act[3]]
        obs, r, term, trunc, info = env.step(a3)
        rewards["all"].append(r)
        if r > 0.11:
            rewards["fruit"].append(r)
        elif r < 0.0:
            rewards["neg"].append(r)
        else:
            rewards["base"].append(r)
arr = np.array(rewards["all"])
fruit = np.array(rewards["fruit"])
neg = np.array(rewards["neg"])
print(f"steps collected={len(arr)}  mean_reward={arr.mean():.4f}/step")
print(f"  base dt reward: {len(rewards['base'])} steps ({100*len(rewards['base'])/len(arr):.0f}%) mean={np.mean(rewards['base']) if rewards['base'] else 0:.4f}")
print(f"  fruit-bonus steps: {len(fruit)} ({100*len(fruit)/len(arr):.2f}%) mean_bonus={fruit.mean():.4f}" if len(fruit) else "  no fruit events collected")
print(f"  negative steps (predated): {len(neg)} ({100*len(neg)/len(arr):.2f}%) mean={neg.mean():.3f}" if len(neg) else "  no predation events")
print(f"  so per-step reward is ~{arr.mean():.3f} (survival bonus 0.1 dominates)")