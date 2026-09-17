"""Phase-1 trajectory collector: full per-step, per-agent logs from the expert controller,
INCLUDING failure and edge cases (strategy: "do not only collect successful/typical trajectories").

Produces the dataset Phase 2 (imitation) and failure diagnosis need.

Usage:
    python collect_trajectories.py <horizon> <seeds: train|eval|csv> <out.npz> [subsample]

Records per (variant, seed, tick, agent):
    obs      31-dim encoder output (SAME encoder used at serve time)          float16
    act      [move_distance, move_direction(steer), turn_angle, spawn_flag]   float32  <- expert action
    energy, age, t (tick), agent_id, variant_id, seed
Episode-level metadata: ticks survived, score, spawns, deaths(predated), final agents, pop/energy trace.

Delay-variants (failure modes) are included so the policy sees predator encounters, low energy,
sparse fruit, crowding and collapse, not just clean runs.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from env_wrapper import run_eval_episode, build_obs
import best_controller as bc
from best_controller import DEFAULT_PARAMS

# Diversity: the deployed expert + deliberate failure/edge-case regimes.
VARIANTS = {
    "expert": {},                                                            # deployed NEW params
    "no_flee": {"use_flee": 0.0},                                            # predator encounters
    "no_fruit": {"use_fruit": 0.0},                                          # starvation / sparse food
    "under_repro": {"repro_frac": 0.9, "repro_global_target": 4,
                    "spawn_cooldown": 800},                                  # population collapse
    "no_predator_field": {"use_predator": 0.0},                              # unguarded movement
}
TRAIN_SEEDS = list(range(100, 900, 100))
EVAL_SEEDS = list(range(1000, 1800, 100))


def baseline():
    P = dict(DEFAULT_PARAMS)
    try:
        with open(os.path.join(HERE, "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    return P


def collect(horizon, seeds, out_path, subsample=2):
    obs_l, act_l, en_l, age_l, t_l, aid_l, var_l, seed_l = [], [], [], [], [], [], [], []
    episodes = []
    names = list(VARIANTS)
    for vi, name in enumerate(names):
        P = baseline()
        P.update(VARIANTS[name])
        fn = bc.make_policy(P)
        for s in seeds:
            buf = {"obs": [], "act": [], "energy": [], "age": [], "t": [], "aid": []}

            def recorder(i, livestates, acts, out, _b=buf):
                if i % subsample:
                    return
                for st, (aid, a) in zip(livestates, acts):
                    _b["obs"].append(build_obs(st))
                    _b["act"].append([float(a.move_distance), float(a.move_direction),
                                      float(a.turn_angle), 1.0 if a.spawn_agent else 0.0])
                    _b["energy"].append(float(st.get("energy", 0.0)))
                    _b["age"].append(float(st.get("age", 0.0)))
                    _b["t"].append(i)
                    _b["aid"].append(int(aid))

            r = run_eval_episode(fn, n_agents=5, seed=s, horizon=horizon,
                                 stop_on_death=True, trace=True, recorder=recorder)
            n = len(buf["act"])
            for k, dst in (("obs", obs_l), ("act", act_l), ("energy", en_l), ("age", age_l),
                           ("t", t_l), ("aid", aid_l)):
                dst.extend(buf[k])
            var_l.extend([vi] * n)
            seed_l.extend([s] * n)
            ep = {"variant": name, "seed": s, "ticks": r["steps"], "score": round(r["score"], 2),
                  "spawns": r["spawns"], "predated": r["predated"],
                  "final_agents": r["final_agents"], "samples": n, "trace": r["traces"]}
            episodes.append(ep)
            print("  %-18s seed=%-5d ticks=%6d score=%8.1f spawns=%-4d pred=%-4d samples=%d"
                  % (name, s, ep["ticks"], ep["score"], ep["spawns"], ep["predated"], n), flush=True)

    np.savez_compressed(
        out_path,
        obs=np.asarray(obs_l, dtype=np.float16),
        act=np.asarray(act_l, dtype=np.float32),
        energy=np.asarray(en_l, dtype=np.float32),
        age=np.asarray(age_l, dtype=np.float32),
        t=np.asarray(t_l, dtype=np.int32),
        agent_id=np.asarray(aid_l, dtype=np.int32),
        variant_id=np.asarray(var_l, dtype=np.int16),
        seed=np.asarray(seed_l, dtype=np.int32),
        variant_names=np.asarray(names),
    )
    meta = {"horizon": horizon, "subsample": subsample, "seeds": list(seeds),
            "variants": VARIANTS, "variants_used": names, "episodes": episodes,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": len(act_l)}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("saved %d samples -> %s (+ .meta.json)" % (len(act_l), out_path))
    return meta


if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    arg = sys.argv[2] if len(sys.argv) > 2 else "train"
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "traj_phase1.npz")
    sub = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    seeds = TRAIN_SEEDS if arg == "train" else (EVAL_SEEDS if arg == "eval"
                                                else [int(x) for x in arg.split(",")])
    print("COLLECT horizon=%d seeds=%s variants=%d subsample=%d"
          % (H, seeds, len(VARIANTS), sub))
    collect(H, seeds, out, sub)