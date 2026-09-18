"""Parallel (mu + lambda) evolution strategy over the FULL controller parameter vector.

Why a rewrite of evolve.py:
  * evolve.py evaluated WITHOUT reset_fn, so module-level policy memory leaked between episodes
    (the same contamination that made earlier sweeps order-dependent). Every episode here resets.
  * evolve.py wrote straight into best_controller/params.json -- one bug midway would overwrite the
    DEPLOYED parameters. This writes only to evolve_det_best.json; adopting a winner is a separate,
    explicit step.
  * evolve.py searched 16 knobs; the controller now exposes more, including the population target
    (which the fiscal sweep showed is load-bearing) and absolute-vs-fractional spawn gating.
  * fitness is reduced-horizon mean survival ticks on TRAIN seeds only; held-out EVAL seeds are
    never touched, so any winner still has to be confirmed independently.

Design: one candidate per task, its seeds evaluated inside the worker; N workers in parallel.
Ticks (not score) is the objective because score ~= ticks/10 and ticks is the cleaner signal.

Usage:
  python evolve_det.py --minutes 120 --workers 3 --pop 24 --horizon 6000 --seeds 100,200,300
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

# (low, high, sigma_as_fraction_of_range) -- bounds keep parameters physically sensible
SPACE = {
    "fruit_weight":        (0.5, 6.0, 0.30),
    "predator_weight":     (1.0, 8.0, 0.30),
    "wall_weight":         (0.0, 3.0, 0.35),
    "disperse_weight":     (0.0, 3.0, 0.35),
    "wander_weight":       (0.0, 0.30, 0.45),
    "target_hyst":         (0.2, 1.0, 0.30),
    "danger_dist":         (120.0, 300.0, 0.12),
    "flee_dist":           (80.0, 220.0, 0.12),
    "escape_dist":         (150.0, 400.0, 0.12),
    "fruit_risk_penalty":  (0.0, 3.0, 0.35),
    "walk_frac":           (0.3, 1.0, 0.15),
    "explore_frac":        (0.2, 1.0, 0.25),
    "blind_explore_frac":  (0.2, 1.0, 0.25),
    "tree_weight":         (0.0, 1.0, 0.40),
    "repro_frac":          (0.25, 0.85, 0.20),
    "repro_frac_min":      (0.15, 0.40, 0.25),
    "repro_urgency":       (0.0, 1.0, 0.40),
    "repro_global_target": (8.0, 34.0, 0.20),
    "repro_popcap":        (4.0, 18.0, 0.20),
    "spawn_cooldown":      (30.0, 500.0, 0.25),
    "repro_safe_radius":   (120.0, 420.0, 0.15),
    "repro_energy_abs":    (0.0, 300.0, 0.25),   # 0 = fractional gate; >=105 = absolute
    # --- ENDGAME CONSOLIDATION / BANKING (phase_mode 0 = off = exactly the old behaviour) ---
    # Why: every run ends by starvation as fruit production decays (0.5^(t/300)), and the first
    # attempt at this failed for TWO measurable reasons, both now fixed -- it fired mid-boom
    # (4,000-7,500) and its spawn gate silently forbade ALL breeding (gpop <= 3 while the fleet held
    # 10-16 agents), killing the relay. Arithmetic for the 1,800 target: metabolism alone is
    # ~0.1/tick, so one agent holding ~600 energy at zero/low movement survives ~6,000 extra ticks.
    # Rather than hand-set the switch points, expose them here and let the search choose.
    "phase_mode":          (0.0, 1.0, 0.50),     # 0 = off; scales the whole endgame effect
    "famine_tick_lo":      (6000.0, 16000.0, 0.30),
    "famine_tick_hi":      (8000.0, 20000.0, 0.30),
    "famine_move_frac":    (0.2, 1.0, 0.25),     # 1.0 = movement untouched in the endgame
    "famine_bank_frac":    (0.4, 1.0, 0.25),     # energy fraction needed to breed while banking
    "famine_min_pop":      (1.0, 6.0, 0.30),     # relay rescue threshold (never stop breeding)
    "famine_min_cap":      (0.0, 800.0, 0.30),   # 0 = any lineage; >0 = high-capacity lineages only
}

# Ranking = mean_ticks + FRUIT_W * mean_fruit. Kept SMALL and documented because the raw fruit term
# has a HEADCOUNT BIAS: a bigger fleet eats more in total simply by having more mouths, so a large
# weight would reward the over-breeding trap the finances sweep and the winner-vs-live comparison
# both warn about. 0.05/fruit caps the bonus near +100 ticks -- enough to break near-ties toward
# income (the low-variance causal quantity) without letting fleet size dominate survival.
FRUIT_W = 0.05

PARAMS_FILE = os.path.join(REPO, "best_controller", "params.json")
BEST_FILE = os.path.join(HERE, "evolve_det_best.json")
HIST_FILE = os.path.join(HERE, "evolve_det_history.jsonl")


def load_incumbent():
    import best_controller as bc
    P = dict(bc.DEFAULT_PARAMS)
    try:
        with open(PARAMS_FILE) as f:
            P.update(json.load(f))
    except Exception:
        pass
    for k in SPACE:                      # every searched key must exist as a float
        v = P.get(k)
        if k == "blind_explore_frac" and v is None:
            v = P.get("explore_frac", 0.35)
        if v is None:
            v = (SPACE[k][0] + SPACE[k][1]) / 2.0
        P[k] = float(min(SPACE[k][1], max(SPACE[k][0], float(v))))
    return P


def clamp(P):
    out = dict(P)
    for k, (lo, hi, _) in SPACE.items():
        out[k] = float(min(hi, max(lo, float(out.get(k, lo)))))
    return out


def eval_episode(task):
    """Worker: evaluate ONE (candidate, seed) episode.

    Fine-grained on purpose: episode lengths vary from ~4k to the 9k cap, so one-task-per-candidate
    left workers idle while the longest episode ran. Splitting per episode keeps every core busy.
    """
    idx, params, seed, horizon = task
    try:
        os.nice(19)                      # never outrank the live serving process
    except Exception:
        pass
    import best_controller as bc
    from env_wrapper import run_eval_episode
    fn = bc.make_policy(clamp(params))
    # run_eval_episode seeds random/np.random AND resets policy memory -> clean episode
    r = run_eval_episode(fn, n_agents=5, seed=seed, horizon=horizon, stop_on_death=True,
                         reset_fn=bc.reset_memory)
    return idx, seed, int(r["steps"]), int(r["fruits_eaten"])


def mutate(P, rng, scale):
    C = dict(P)
    for k, (lo, hi, sf) in SPACE.items():
        if rng.random() < 0.55:          # sparse: don't jitter every parameter every time
            C[k] = P[k] + rng.gauss(0.0, sf * (hi - lo) * scale)
    return clamp(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6000)
    ap.add_argument("--seeds", default="100,200,300")
    ap.add_argument("--seed_pool", default="")
    ap.add_argument("--seeds_per_gen", type=int, default=0)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    # Anti-overfitting: resample the fitness seeds every generation from a POOL of TRAIN seeds.
    # Tuning against one fixed small seed set is how three earlier hypotheses produced train-seed
    # "wins" that inverted on held-out seeds.
    pool_seeds = [int(x) for x in a.seed_pool.split(",")] if a.seed_pool else []
    n_per_gen = a.seeds_per_gen or 0
    deadline = time.time() + a.minutes * 60
    rng = random.Random(12345)

    incumbent = load_incumbent()
    if os.path.exists(BEST_FILE):
        try:
            incumbent = clamp(json.load(open(BEST_FILE))["params"])
            print("resuming from %s" % BEST_FILE)
        except Exception:
            pass

    print("EVOLVE-DET workers=%d pop=%d elite=%d horizon=%d seeds=%s minutes=%.0f"
          % (a.workers, a.pop, a.elite, a.horizon, seeds, a.minutes), flush=True)
    print("controller=%s" % os.path.join(REPO, "best_controller.py"), flush=True)

    pool = mp.Pool(a.workers)

    def score_candidates(cands, seeds):
        """Episode-level parallelism: one task per (candidate, seed), aggregated per candidate."""
        tasks = [(i, c, s, a.horizon) for i, c in enumerate(cands) for s in seeds]
        res = pool.map(eval_episode, tasks)
        per = {i: [] for i in range(len(cands))}
        fr = {i: [] for i in range(len(cands))}
        for idx, _seed, ticks, fruit in res:
            per[idx].append(ticks)
            fr[idx].append(fruit)
        return [(st.mean(per[i]), sorted(per[i]), st.mean(fr[i]), cands[i]) for i in range(len(cands))]

    _base = score_candidates([dict(incumbent)], seeds)[0]
    print("incumbent baseline: ticks_mean=%.1f per_seed=%s fruit=%.1f"
          % (_base[0], _base[1], _base[2]), flush=True)

    # The incumbent is PROTECTED: best_ticks starts at the incumbent's own fitness, so the saved
    # best can never be worse than the parameters we are already serving. (Without this, generation
    # 1 "improved" on a candidate that actually scored below the incumbent, because best_ticks
    # started as None.)
    best, best_ticks = dict(incumbent), _base[0]
    best_rank = _base[0] + FRUIT_W * (_base[2] if len(_base) > 2 else 0.0)
    gen = 0
    while time.time() < deadline:
        gen += 1
        if pool_seeds and n_per_gen:
            seeds = sorted(rng.sample(pool_seeds, min(n_per_gen, len(pool_seeds))))
        scale = max(0.25, 1.0 - 0.06 * gen)        # anneal the step size
        cands = [dict(incumbent)] + [mutate(incumbent, rng, scale) for _ in range(a.pop - 1)]
        t0 = time.time()
        # Rank on survival ticks, with FRUIT EATEN as a tie-breaker: when the horizon cap is hit the
        # tick count saturates and capped candidates become indistinguishable, but they still differ
        # in how much they ate -- which is the income that actually drives survival.
        # RANKING -- and this is the fix for the session's central failure. Survival ticks are a
        # HIGH-VARIANCE objective: the same policy on the same seed swung 5,922 vs 7,825 ticks
        # between runs, so ranking candidates on 3 seeds selects NOISE, which is why every sweep
        # today (tree/blind/population/phase) gained on train seeds and evaporated on held-out ones.
        # Fruit eaten is the same causal quantity with far lower variance (1,312/1,481/1,873/1,594/
        # 1,426 across seeds vs a 2x swing in ticks), and it is the throughput that ultimately feeds
        # survival. So: ticks stay primary, income gets a REAL weight (0.2/fruit ~ up to +500 ticks)
        # instead of the 0.001 tie-break, which was numerically negligible.
        scored = sorted(score_candidates(cands, seeds), key=lambda z: -(z[0] + FRUIT_W * z[2]))
        gen_best_ticks, gen_best_per, gen_best_fruit = scored[0][0], scored[0][1], scored[0][2]
        gen_rank = gen_best_ticks + FRUIT_W * gen_best_fruit
        elapsed = time.time() - t0
        # ADOPTION MUST MEASURE THE SAME QUANTITY AS THE RANKING (independent review finding):
        # ranking on ticks + FRUIT_W*fruit while adopting on ticks alone meant a candidate that won
        # the ranking on fruit could be recorded as "no gain" -- so the saved artifact could be
        # WORSE than the ranked winner, and the elite path then drifted from a worse base.
        improved = gen_rank > best_rank
        if improved:
            best, best_ticks, best_rank = dict(scored[0][3]), gen_best_ticks, gen_rank
            json.dump({"params": best, "fitness_ticks": best_ticks, "seeds": seeds,
                       "horizon": a.horizon, "generation": gen, "gen_per_seed": gen_best_per,
                       "gen_fruit": gen_best_fruit, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      open(BEST_FILE, "w"), indent=1)
        # (mu + lambda): next incumbent = best elite (weighted mean of top-k is also fine)
        elites = [z[3] for z in scored[:a.elite]]
        incumbent = clamp({k: st.mean([e[k] for e in elites]) for k in SPACE} | dict(incumbent))
        rec = {"gen": gen, "gen_best_ticks": round(gen_best_ticks, 1), "best_ticks": round(best_ticks, 1),
               "improved": improved, "gen_best_per_seed": gen_best_per, "fruit": round(gen_best_fruit, 1),
               "secs": round(elapsed, 1), "scale": round(scale, 3), "seeds": seeds}
        with open(HIST_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("gen %d: best_ticks=%.1f (gen best %.1f, %s) per_seed=%s fruit=%.1f  [%.0fs]"
              % (gen, best_ticks, gen_best_ticks, "IMPROVED" if improved else "no gain",
                 gen_best_per, gen_best_fruit, elapsed), flush=True)
    pool.close()
    pool.join()
    print("DONE. best_ticks=%.1f -> %s" % (best_ticks, BEST_FILE), flush=True)


if __name__ == "__main__":
    main()
