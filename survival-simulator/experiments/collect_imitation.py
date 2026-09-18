"""Phase-2 collector: expert trajectories WITH the augmented observation (build_obs + private state).

Same regimes/seeds/harness as experiments/collect_trajectories.py (which is left untouched and stays
the Phase-1 asset), but each recorded sample carries obs_aug = [build_obs(state) (31) | aug (3)]
where the 3 aug features come from imitation_obs.AugObs, maintained from the expert's own rollout —
the exact same quantity the deployed policy maintains from its own history (see imitation_obs.py).

Usage:
    python collect_imitation.py <horizon> <seeds: train|eval|csv> <out.npz> [subsample]
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
from imitation_obs import AugObs
import best_controller as bc
from best_controller import DEFAULT_PARAMS

VARIANTS = {
    "expert": {},
    "no_flee": {"use_flee": 0.0},
    "no_fruit": {"use_fruit": 0.0},
    "under_repro": {"repro_frac": 0.9, "repro_global_target": 4, "spawn_cooldown": 800},
    "no_predator_field": {"use_predator": 0.0},
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
    obs31_l, aug_l, act_l, en_l, age_l, t_l, aid_l, var_l, seed_l = [], [], [], [], [], [], [], [], []
    episodes = []
    names = list(VARIANTS)
    for vi, name in enumerate(names):
        P = baseline()
        P.update(VARIANTS[name])
        fn = bc.make_policy(P)
        for s in seeds:
            buf = {"o31": [], "aug": [], "act": [], "energy": [], "age": [], "t": [], "aid": []}
            aug = AugObs()

            def recorder(i, livestates, acts, out, _b=buf, _a=aug):
                if i % subsample:
                    return
                for st, (aid, a) in zip(livestates, acts):
                    _b["o31"].append(build_obs(st))                 # encoder IDENTICAL to serve time
                    # aug features BEFORE this action's update (same order as in the policy wrapper)
                    _b["aug"].append(_a.features(st))
                    _a.update(st, bool(a.spawn_agent))
                    _b["act"].append([float(a.move_distance), float(a.move_direction),
                                      float(a.turn_angle), 1.0 if a.spawn_agent else 0.0])
                    _b["energy"].append(float(st.get("energy", 0.0)))
                    _b["age"].append(float(st.get("age", 0.0)))
                    _b["t"].append(i)
                    _b["aid"].append(int(aid))

            r = run_eval_episode(fn, n_agents=5, seed=s, horizon=horizon,
                                 stop_on_death=True, trace=True, recorder=recorder)
            n = len(buf["act"])
            for k, dst in (("o31", obs31_l), ("aug", aug_l), ("act", act_l), ("energy", en_l),
                           ("age", age_l), ("t", t_l), ("aid", aid_l)):
                dst.extend(buf[k])
            var_l.extend([vi] * n)
            seed_l.extend([s] * n)
            pop_peak = max([tr["n"] for tr in r["traces"]] or [0])
            ep = {"variant": name, "seed": s, "ticks": r["steps"], "score": round(r["score"], 2),
                  "spawns": r["spawns"], "predated": r["predated"],
                  "final_agents": r["final_agents"], "pop_peak": pop_peak,
                  "samples": n, "trace": r["traces"]}
            episodes.append(ep)
            print("  %-18s seed=%-5d ticks=%6d score=%8.1f spawns=%-4d pred=%-4d pop_peak=%-3d samples=%d"
                  % (name, s, ep["ticks"], ep["score"], ep["spawns"], ep["predated"], pop_peak, n),
                  flush=True)

    o31 = np.asarray(obs31_l, dtype=np.float16)
    aug = np.asarray(aug_l, dtype=np.float16)
    np.savez_compressed(
        out_path,
        obs=np.concatenate([o31, aug], axis=1),      # obs_aug = 34-dim, as the policy consumes it
        obs31=o31, aug=aug,
        act=np.asarray(act_l, dtype=np.float32),
        energy=np.asarray(en_l, dtype=np.float32),
        age=np.asarray(age_l, dtype=np.float32),
        t=np.asarray(t_l, dtype=np.int32),
        agent_id=np.asarray(aid_l, dtype=np.int32),
        variant_id=np.asarray(var_l, dtype=np.int16),
        seed=np.asarray(seed_l, dtype=np.int32),
        variant_names=np.asarray(names),
    )
    meta = {"horizon": horizon, "subsample": subsample, "seeds": list(seeds), "variants": VARIANTS,
            "variants_used": names, "episodes": episodes, "obs_dim": int(o31.shape[1] + aug.shape[1]),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": len(act_l)}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("saved %d samples (obs_aug dim %d) -> %s" % (len(act_l), meta["obs_dim"], out_path))
    return meta


if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    arg = sys.argv[2] if len(sys.argv) > 2 else "train"
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "traj_train_im_aug.npz")
    sub = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    seeds = TRAIN_SEEDS if arg == "train" else (EVAL_SEEDS if arg == "eval"
                                                else [int(x) for x in arg.split(",")])
    print("COLLECT(aug) horizon=%d seeds=%s variants=%d subsample=%d"
          % (H, seeds, len(VARIANTS), sub))
    collect(H, seeds, out, sub)
