#!/usr/bin/env python3
"""probe_enc.py - ENCODING ADEQUACY PROBE. Can the observation encoding represent the incumbent's
own behaviour at all?

WHY THIS EXISTS (a necessary condition, not a nice-to-have)
    A residual policy adds a correction to the deployed heuristic. That is only meaningful if the
    encoding it sees carries at least the information the heuristic itself uses. The current
    encoder (env_wrapper.build_obs, 31 floats) keeps only the NEAREST fruit / predator / agent plus
    counts, while the heuristic consumes the FULL observation list (every fruit, every tree, every
    edge). A correction term with less information than its base can only add noise.

    So before spending compute on training: can a model predict THE HEURISTIC'S OWN ACTION from the
    encoding? That is behavioural cloning of the incumbent against itself. If the encoding cannot
    recover it, training on that encoding cannot beat it.

METHOD
    One pass over the deployed controller. Every 5th tick, per agent, record
      label = the controller's own (move_direction, move_distance)
      V1    = the current 31-float encoder
      V2    = richer encoder: K-nearest per entity type (4 fruit / 2 predator / 2 tree), edge
              summary, own state fractional AND absolute, biome, counts, plus the economy/time
              state the audit identified as missing (income EMAs, ef slope, lockout EMA, ticks and
              distance since the last fruit, sim_tick, fleet size, genome rank).
    Split BY SEED into train/test - no leakage. Closed-form ridge per target. Report held-out R^2
    and mean absolute angular error for the heading.

READING THE RESULT
    high R^2 on both   -> encoding sufficient; a residual is well-posed
    V2 >> V1           -> the current encoding is the binding defect; fix it before training
    low R^2 on both    -> the action is not a function of the state, so a residual on this encoding
                          is flying blind and training is wasted compute

    python3 probe_enc.py --seeds 6000-6039 --horizon 18000 --workers 60 --out probe.json
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import random  # noqa: E402
import numpy as np  # noqa: E402

BIOME_ID = {"forest": 0, "grassland": 1, "swamp": 2, "desert": 3, "river": 4}
V2_DIM = 61
EVERY = 5


def _ent(obs_list, etype, k, with_rel=False):
    sel = sorted((o for o in obs_list if o.get("type") == etype), key=lambda o: o["distance"])[:k]
    out = []
    for i in range(k):
        if i < len(sel):
            o = sel[i]
            a = o["angle"]
            out += [min(o["distance"] / 300.0, 1.0), math.cos(a), math.sin(a)]
            if with_rel:
                r = o.get("rel_dir", 0.0)
                out += [math.cos(r), math.sin(r)]
        else:
            out += [1.0, 0.0, 0.0] + ([0.0, 0.0] if with_rel else [])
    return out


def build_v2(st, m):
    o = st.get("observations") or []
    max_e = max(float(st.get("max_energy", 500.0) or 500.0), 1.0)
    energy = float(st.get("energy", 0.0) or 0.0)
    ef = energy / max_e
    f = [ef, min(energy / 1000.0, 1.0), min(max_e / 1000.0, 1.0),
         min(float(st.get("age", 0.0) or 0.0) / 120.0, 1.0),
         float(st.get("speed", 10.0) or 10.0) / 20.0,
         float(st.get("sprint_speed", 20.0) or 20.0) / 40.0,
         float(st.get("vision_range", 200.0) or 200.0) / 400.0,
         min(float(st.get("vision_angle", 1.0) or 1.0) / (math.pi / 2), 1.0),
         float(st.get("hearing_radius", 50.0) or 50.0) / 100.0]
    onehot = [0.0] * 5
    onehot[BIOME_ID.get(st.get("biome"), 4)] = 1.0
    f += onehot
    nf = sum(1 for e in o if e.get("type") == "Fruit")
    npr = sum(1 for e in o if e.get("type") == "Predator")
    na = sum(1 for e in o if e.get("type") == "Agent")
    nt = sum(1 for e in o if e.get("type") == "Tree")
    f += [min(nf, 8) / 8.0, min(npr, 4) / 4.0, min(na, 8) / 8.0, min(nt, 8) / 8.0]
    f += _ent(o, "Fruit", 4)
    f += _ent(o, "Predator", 2, with_rel=True)
    f += _ent(o, "Tree", 2)
    edges = [e for e in o if e.get("type") == "Edge"]
    if edges:
        ds = [math.hypot(e["coords"][0][0], e["coords"][0][1]) for e in edges]
        c = edges[int(np.argmin(ds))]["coords"][0]
        n = max(1e-6, math.hypot(*c))
        cl, bc, bs = min(ds) / 300.0, c[0] / n, c[1] / n
    else:
        cl, bc, bs = 1.0, 0.0, 0.0
    f += [min(len(edges), 20) / 20.0, cl, bc, bs]
    f += [min(m.get("tick", 0.0) / 1000.0, 30.0),
          min(m.get("inc_f", 0.0) / 5.0, 1.0), min(m.get("inc_s", 0.0) / 5.0, 1.0),
          m.get("ef_s", ef), max(-1.0, min(1.0, (ef - m.get("ef_s", ef)) * 10.0)),
          m.get("lock", 0.0),
          min(m.get("tf", 0.0) / 500.0, 1.0), min(m.get("df", 0.0) / 1000.0, 5.0)]
    f += [min(m.get("n_agents", 0.0) / 20.0, 1.0), min(m.get("rank", 0.0) / 10.0, 1.0),
          m.get("since_birth", 0.0) / 1000.0]
    return np.array(f, dtype=np.float32)


def build_v3(st, m):
    """V2 plus the incumbent's OWN internal latches. Diagnostic only - at inference a learned
    policy would have to carry equivalents itself (recurrence), not read these."""
    base = build_v2(st, m)
    w = float(m.get("wand", 0.0))
    l = float(m.get("steer", 0.0))
    return np.concatenate([base, np.array(
        [math.cos(w), math.sin(w), math.cos(l), math.sin(l), float(m.get("flee", 0.0))],
        dtype=np.float32)])


def run_episode(args):
    seed, horizon, params_path = args
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    from env_wrapper import build_obs

    P = dict(bc.DEFAULT_PARAMS)
    with open(params_path) as f:
        P.update(json.load(f))
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    policy = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    memo, memo_birth = {}, {}
    X1, X2, X3, Y, MK = [], [], [], [], []
    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a.agent_id) for a in live) if s]
        rec = (i % EVERY == 0)
        acts = []
        for s in states:
            aid = s["agent_id"]
            m = memo.setdefault(aid, {})
            memo_birth.setdefault(aid, i)
            # the controller updates _SIM_TICK / _GS_LAST inside the call, so call FIRST
            dist, dr, turn, spawn = policy(s)
            energy = float(s.get("energy", 0.0) or 0.0)
            _mem = bc._MEM.get(aid) or {}
            m["wand"] = float(_mem.get("wander_ang", 0.0) or 0.0)
            m["steer"] = float(_mem.get("last_steer", 0.0) or 0.0)
            m["flee"] = 1.0 if _mem.get("flee") else 0.0
            if rec:
                prev_e = m.get("e")
                inc = 0.0 if prev_e is None else max(0.0, energy - prev_e) + 0.1
                m["inc_f"] = 0.9 * m.get("inc_f", 0.0) + 0.1 * inc
                m["inc_s"] = 0.99 * m.get("inc_s", 0.0) + 0.01 * inc
                max_e = max(float(s.get("max_energy", 500.0) or 500.0), 1.0)
                ef = energy / max_e
                m["ef_s"] = 0.99 * m.get("ef_s", ef) + 0.01 * ef
                m["lock"] = 0.99 * m.get("lock", 0.0) + 0.01 * (1.0 if energy < max_e / 5.0 else 0.0)
                nf = sum(1 for e in (s.get("observations") or []) if e.get("type") == "Fruit")
                if nf > 0:
                    m["tf"], m["df"] = 0.0, 0.0
                else:
                    m["tf"] = m.get("tf", 0.0) + EVERY
                    m["df"] = m.get("df", 0.0) + float(s.get("speed", 10.0) or 10.0) * 0.12 * EVERY
                m["tick"] = float(bc._SIM_TICK)
                m["n_agents"] = float(len(states))
                if bc._GS_LAST is not None:
                    m["rank"] = float(bc._GS_LAST[1])
                m["since_birth"] = float(i - memo_birth[aid])
                X1.append(build_obs(s))
                X2.append(build_v2(s, m))
                X3.append(build_v3(s, m))
                Y.append((dr, dist))
                # does the incumbent's heading come from a branch the FRAME can see? With
                # forage_nearest=1.0 the steer is the bearing to the nearest VISIBLE fruit, which
                # the frame carries. With no fruit visible it is a potential field damped by the
                # hidden held heading / steer hysteresis - invisible to any single frame.
                MK.append(1.0 if any(e.get("type") == "Fruit" for e in (s.get("observations") or [])) else 0.0)
            m["e"] = energy
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=dist, move_direction=dr,
                                            turn_angle=turn, spawn_agent=bool(spawn))))
        core.step(acts)
        keep = {a.agent_id for a in core.env.agents}
        memo = {k: v for k, v in memo.items() if k in keep}
        memo_birth = {k: v for k, v in memo_birth.items() if k in keep}
    return (seed, np.asarray(X1, np.float32), np.asarray(X2, np.float32), np.asarray(X3, np.float32),
            np.asarray(Y, np.float32), np.asarray(MK, np.float32))


def ridge_fit_eval(Xtr, Ytr, Xte, Yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    lam = 1.0
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    P = B @ W
    out = {}
    for j, name in enumerate(("heading", "speed")):
        y, p = Yte[:, j], P[:, j]
        ss = ((y - y.mean()) ** 2).sum()
        out[name + "_R2"] = float(1.0 - ((y - p) ** 2).sum() / max(ss, 1e-9))
        if j == 0:
            d = np.arctan2(np.sin(y - p), np.cos(y - p))
            out["heading_mae_rad"] = float(np.abs(d).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="6000-6039")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "probe.json"))
    a = ap.parse_args()
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    n_train = len(seeds) // 2
    tr_seeds, te_seeds = set(seeds[:n_train]), set(seeds[n_train:])
    print(f"probe_enc | {len(seeds)} seeds ({n_train} train / {len(seeds)-n_train} test) | "
          f"V2_DIM={V2_DIM} | horizon {a.horizon} | workers {a.workers}", flush=True)

    from multiprocessing import get_context
    ctx = get_context("spawn")
    jobs = [(s, a.horizon, a.params) for s in seeds]
    parts = {"tr": ([], [], [], [], []), "te": ([], [], [], [], [])}   # X1,V2,V3,Y,mask
    done = 0
    with ctx.Pool(a.workers) as pool:
        for seed, x1, x2, x3, y, mk in pool.imap_unordered(run_episode, jobs, chunksize=1):
            k = "tr" if seed in tr_seeds else "te"
            for j, arr in enumerate((x1, x2, x3, y, mk)):
                parts[k][j].append(arr)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} episodes", flush=True)

    def cat(k, j):
        return np.concatenate(parts[k][j]) if parts[k][j] else np.zeros((0, 1), np.float32)

    res = {}
    MKtr, MKte = cat("tr", 4).ravel(), cat("te", 4).ravel()
    for tag, j in (("V1_31float", 0), ("V2_rich", 1), ("V3_+memory", 2)):
        A, B = cat("tr", j), cat("te", j)
        Ytr_, Yte_ = cat("tr", 3), cat("te", 3)
        r = ridge_fit_eval(A, Ytr_, B, Yte_)
        r["n_train"], r["n_test"], r["dim"] = int(len(A)), int(len(B)), int(A.shape[1])
        r["fruit_visible_frac"] = float(MKte.mean())
        for lab, sel in (("fruit_visible", MKte > 0.5), ("blind", MKte <= 0.5)):
            if sel.sum() > 100:
                sub = ridge_fit_eval(A[MKtr > 0.5] if lab == "fruit_visible" else A[MKtr <= 0.5],
                                     Ytr_[MKtr > 0.5] if lab == "fruit_visible" else Ytr_[MKtr <= 0.5],
                                     B[sel], Yte_[sel])
                r[lab + "_heading_R2"] = sub["heading_R2"]
                r[lab + "_heading_mae"] = sub["heading_mae_rad"]
                r[lab + "_speed_R2"] = sub["speed_R2"]
                r[lab + "_n"] = int(sel.sum())
        res[tag] = r
    res["split"] = {"train_seeds": sorted(tr_seeds), "test_seeds": sorted(te_seeds)}
    json.dump(res, open(a.out, "w"), indent=1)

    print()
    print("ENCODING ADEQUACY  (held-out seeds; predicting the INCUMBENT'S OWN action)")
    print(f"{'encoder':<12} {'dim':>4} {'heading R2':>11} {'MAE rad':>8} | "
          f"{'FRUIT VISIBLE R2':>16} {'MAE':>7} {'n':>8} | {'BLIND R2':>9} {'MAE':>7} {'n':>8}")
    for tag in ("V1_31float", "V2_rich", "V3_+memory"):
        r = res[tag]
        print(f"{tag:<12} {r['dim']:>4} {r['heading_R2']:>11.4f} {r['heading_mae_rad']:>8.3f} | "
              f"{r.get('fruit_visible_heading_R2', float('nan')):>16.4f} "
              f"{r.get('fruit_visible_heading_mae', float('nan')):>7.3f} "
              f"{r.get('fruit_visible_n', 0):>8} | "
              f"{r.get('blind_heading_R2', float('nan')):>9.4f} "
              f"{r.get('blind_heading_mae', float('nan')):>7.3f} {r.get('blind_n', 0):>8}")
    print(f"\nfruit visible on {100*res['V1_31float']['fruit_visible_frac']:.1f}% of scored agent-ticks")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()