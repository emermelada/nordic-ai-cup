"""Compact (1+lambda) evolution strategy over the potential controller params.
FITNESS = SURVIVORSHIP score: run_eval_episode(stop_on_death=True, horizon=CAP) averaged over
a small held-out seed set. Because the +dt term only accumulates while the team is alive, this
directly rewards survival time (tick-to-collapse) plus fruit margin minus predation, i.e. the
real competition bottleneck (populations collapse ~4-6k of 30k).
Usage: python evolve.py <gens>
Writes best params to best_controller/params.json and evolve_history.json (with survival ticks).
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode
from best_controller import make_policy, DEFAULT_PARAMS

GENS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
H = 8000                      # survival cap: cheap because collapse stops the episode early
SEEDS = [100, 200, 300]       # multi-seed, not one

SPACE = {
    "fruit_weight":        (3.0,  1.0, 6.0, 0.25),
    "predator_weight":     (4.0,  1.0, 8.0, 0.25),
    "wall_weight":         (1.6,  0.0, 4.0, 0.35),
    "disperse_weight":     (0.9,  0.0, 3.0, 0.35),
    "wander_weight":       (0.05, 0.0, 0.25, 0.45),
    "danger_dist":         (210.0, 120.0, 280.0, 0.10),
    "flee_dist":           (150.0, 90.0, 200.0, 0.10),
    "escape_dist":         (260.0, 180.0, 380.0, 0.10),
    "fruit_risk_penalty":  (1.2,  0.0, 3.0, 0.35),
    "walk_frac":           (1.0,  0.6, 1.0, 0.12),
    "explore_frac":        (0.55, 0.3, 0.8, 0.22),
    "repro_frac":          (0.60, 0.40, 0.85, 0.10),
    "repro_global_target": (8.0, 4.0, 14.0, 0.12),
    "repro_popcap":        (3.0, 1.0, 6.0, 0.18),
    "spawn_cooldown":      (280.0, 150.0, 600.0, 0.18),
    "repro_safe_radius":   (260.0, 180.0, 400.0, 0.10),
}

rng = random.Random(0)


def clamp(P):
    out = dict(P)
    for k, (_, lo, hi, _) in SPACE.items():
        out[k] = min(hi, max(lo, out[k]))
    return out


def fitness(P):
    fn = make_policy(clamp(P))
    scores, tick = [], []
    for s in SEEDS:
        r = run_eval_episode(fn, n_agents=5, seed=s, horizon=H, stop_on_death=True)
        scores.append(r["score"])
        tick.append(r["steps"])
    return float(np.mean(scores)), float(np.mean(tick)), tick


def mutate(base, sigma_scale=1.0):
    P = dict(base)
    for k, (cur, lo, hi, sf) in SPACE.items():
        s = (hi - lo) * sf * sigma_scale
        P[k] = cur + rng.gauss(0, s)
    return clamp(P)


def main():
    best = dict(DEFAULT_PARAMS)
    for k, (cur, *_ ) in SPACE.items():
        best[k] = cur
    best = clamp(best)
    fb, tb, tickb = fitness(best)
    rows = [{"gen": 0, "surv_score": fb, "surv_ticks": tickb}]
    print(f"gen 0: surv.score={fb:.1f} mean_ticks={tb:.0f} ticks={tickb}", flush=True)
    start = time.time()
    for g in range(1, GENS + 1):
        cand = []
        for _ in range(2):
            Pc = mutate(best, max(0.4, 1.0 - g / GENS))
            cand.append((Pc,) + fitness(Pc))
        cand.sort(key=lambda t: t[1], reverse=True)
        if cand[0][1] >= fb - 1.0:
            best, fb, tb, tickb = cand[0][0], cand[0][1], cand[0][2], cand[0][3]
        rows.append({"gen": g, "surv_score": fb, "surv_ticks": tickb})
        print(f"gen {g}/{GENS}: best surv.score={fb:.1f} mean_ticks={tb:.0f} "
              f"cand={[round(c[1],1) for c in cand]} el={time.time()-start:.0f}s", flush=True)
    os.makedirs("best_controller", exist_ok=True)
    with open("best_controller/params.json", "w") as f:
        json.dump({k: best[k] for k in SPACE}, f, indent=1)
    with open("best_controller/evolve_history.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("SAVED best_controller/params.json")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()