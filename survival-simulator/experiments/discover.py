#!/usr/bin/env python3
"""discover.py - quantitative counterfactual state->action discovery. NO vision, no viewer.

THE QUESTION, precisely: not "which single action wins?" (that produced no consistent winner) but
"which STATE FEATURES predict which actions are safe, and where does the current controller repeatedly
choose badly?" We want interpretable decision boundaries - "IF low energy AND predator near AND no fruit
visible THEN action A beats B" - not better numbers in a 44-knob heuristic.

WHY THE PREVIOUS ATTEMPT FAILED, and what is different here:
  1. Metric. It scored branches on (alive, then energy) - a KNOCKOUT. With 80% of sampled states ending
     dead or at zero, "winning" collapsed to a threshold event reproduced by no-ops and suicide actions
     (`still` won 60 times, `toward_predator` 53). Now every branch is scored at a FIXED tick T,
     regardless of when the fleet died, so outcomes are CONTINUOUS and comparable.
  2. Sampling. It sampled whatever state it landed on, dominated by extreme states. Now we condition on
     HEALTHY focal agents (energy >= 30% of max) and rotate the focal agent so we do not keep measuring
     one individual.
  3. Actions. It used 6 crude compass/simple options. Now the action set mirrors the controller's real
     decision space (speed, goal, turn, spawn) plus two SEQUENCE arms, because the user's question
     includes "does a short sequence of actions outperform the current heuristic?".
  4. Features. It assumed the controller's feature set. Now features are whatever the agent can actually
     observe (its own state, visible/heard/edge observations, its own recent history), and we let the
     analysis decide what predicts survival.

MODES
  gen      : sample states, branch K actions + the controller's own action, roll a fixed horizon, record
             features + full outcome vector. Parallelise by running several seeds ranges at once.
  analyse  : load the generated rows, fit interpretable models (decision tree / random forest) to answer
             (a) when does an alternative beat the controller, (b) in which feature regions is one action
             consistently best, (c) which features matter. Report leaves as candidate RULES, and check
             each rule's stability ACROSS SEEDS (a rule that only holds on one seed is noise).

Usage
  ./discover.py gen --seeds 3100-3105 --states-per-seed 8 --horizon 300 --out disc_a.jsonl
  ./discover.py analyse --inputs disc_*.jsonl --depth 3
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

import oracle_probe as op  # noqa: E402

HEALTHY_FRAC = 0.30          # only sample states where the focal agent is not already nearly dead
WALL_MOVE_EPS = 0.35         # commanded a move but barely displaced => blocked (wall)


# ---------------------------------------------------------------- actions
def action_set():
    """The controller's real decision space, plus the two sequence arms. Interpretable on purpose:
    any boundary we find must be encodable as a rule, not a network."""
    return {
        "ctrl": None,                                             # the controller itself (reference)
        "fruit_sprint": op.act_toward("Fruit", True),
        "fruit_walk": op.act_toward("Fruit", False),
        "flee_pred": op.act_away_from("Predator", True),
        "to_pred": op.act_toward("Predator", True),               # suicide probe: sanity check
        "still": (0.0, 0.0, 0.0, False),
        "spawn": (10.0, 0.0, 0.0, True),                          # the birth action is a real choice
        "seq_fruit_then_flee": "SEQ",
    }


def seq_action(horizon):
    """Two-phase sequence: forage for the first half, flee for the second. Tests whether a short
    SEQUENCE beats a single-minded controller. Returns a tuple (phase1, phase2, switch_tick)."""
    def f_phase1(st):
        return op.act_toward("Fruit", True)(st)

    def f_phase2(st):
        return op.act_away_from("Predator", True)(st)
    return (f_phase1, f_phase2, horizon // 2)


def act_of(name, act, seq, st, k):
    """Resolve the focal agent's action for this tick under the branch."""
    if name == "seq_fruit_then_flee":
        return seq[0](st) if k < seq[2] else seq[1](st)
    return act(st) if callable(act) else act


# ---------------------------------------------------------------- features
def features(st, core, my_hist):
    """ONLY what the agent can actually observe. Two predator channels are separated on purpose:
    visible vs only-heard. `seen` and `heard_only` have not been distinguished by the controller and
    that distinction is a prime suspect for why it cannot tell a threat from a distant animal."""
    obs = st.get("observations") or []
    preds = [o for o in obs if o.get("type") == "Predator"]
    fruits = [o for o in obs if o.get("type") == "Fruit"]
    edges = [o for o in obs if o.get("type") == "Edge"]
    cons = [o for o in obs if o.get("type") in ("Agent", "Conspecific", "Creature")]
    max_e = float(st.get("max_energy", 500.0) or 500.0)
    e = float(st.get("energy", 0.0))
    vr = float(st.get("vision_range", 0.0) or 0.0)
    seen = [o for o in preds if float(o.get("distance", 1e9)) <= vr]
    heard_only = [o for o in preds if float(o.get("distance", 1e9)) > vr]
    f = {
        "energy": e, "max_energy": max_e, "e_frac": e / max_e if max_e else 0.0,
        "in_lockout": 1.0 if e < 0.2 * max_e else 0.0,
        "age": float(st.get("age", 0.0)),
        "speed": float(st.get("speed", 0.0)), "sprint_speed": float(st.get("sprint_speed", 0.0)),
        "can_sprint": 1.0 if e >= 0.2 * max_e else 0.0,
        "pred_seen": float(len(seen)),
        "pred_heard_only": float(len(heard_only)),
        "pred_min_dist": min([float(o.get("distance", 1e9)) for o in seen], default=-1.0),
        "pred_min_bearing": min([abs(float(o.get("angle", 0.0))) for o in seen], default=-1.0),
        "fruit_n": float(len(fruits)),
        "fruit_min_dist": min([float(o.get("distance", 1e9)) for o in fruits], default=-1.0),
        "fruit_min_bearing": min([abs(float(o.get("angle", 0.0))) for o in fruits], default=-1.0),
        "edge_n": float(len(edges)),
        "conspecific_n": float(len(cons)),
        "conspecific_min_dist": min([float(o.get("distance", 1e9)) for o in cons], default=-1.0),
        "pop": float(len(core.env.agents)),
        "tick": int(round(float(core.env.time) * 10)),
        "recent_moved": float(np.mean([h[0] for h in my_hist])) if my_hist else 0.0,
        "recent_turn": float(np.mean([abs(h[2]) for h in my_hist])) if my_hist else 0.0,
        "recent_spawned": float(sum(1 for h in my_hist if h[3])) if my_hist else 0.0,
        "recent_ate": float(sum(1 for h in my_hist if h[4])) if my_hist else 0.0,
    }
    return f


# ---------------------------------------------------------------- one state's branch study
def study_state(core, snap, aid, st0, horizon, my_hist, out_rows, seed, tick):
    feats = features(st0, core, my_hist)
    acts = action_set()
    seq = seq_action(horizon)
    P = op.load_params()
    a0 = next((x for x in core.env.agents if x.agent_id == aid), None)
    if a0 is None:
        return
    e_start = float(a0.energy)
    score_start = float(core.env.score)
    pop_start = len(core.env.agents)

    for name, act in acts.items():
        op.restore(core, snap)
        f = op.bc.make_policy(P)
        op.bc.reset_memory()
        pred_min, lock_ticks, wall_block, ate = 1e9, 0, 0, 0
        died_at = None
        prev_e = e_start
        prev_xy = None
        for k in range(horizon):
            states = [core.env.get_agent_state(a.agent_id) for a in list(core.env.agents)]
            focal_st = next((s for s in states if s["agent_id"] == aid), None)
            if focal_st is None:
                died_at = k
                break
            focal_act = f(focal_st) if act is None else act_of(name, act, seq, focal_st, k)
            cmd_dist = float(focal_act[0])
            acts_tick = []
            for s in states:
                if s["agent_id"] == aid:
                    acts_tick.append((s["agent_id"], op.make_action(s, focal_act)))
                else:
                    acts_tick.append((s["agent_id"], op.make_action(s, f(s))))
            # telemetry at decision time
            d = [float(o.get("distance", 1e9)) for o in (focal_st.get("observations") or [])
                 if o.get("type") == "Predator"]
            if d:
                pred_min = min(pred_min, min(d))
            if float(focal_st.get("energy", 1e9)) < 0.2 * float(focal_st.get("max_energy", 500.0) or 500.0):
                lock_ticks += 1
            prev_xy = (float(focal_st.get("x", 0.0)), float(focal_st.get("y", 0.0)))
            core.step(acts_tick)
            cur = next((x for x in core.env.agents if x.agent_id == aid), None)
            if cur is None:
                died_at = k + 1
                break
            # wall blocking proxy: commanded a real move but barely displaced.
            # (A biome move-penalty can also slow an agent, so this is "blocked OR slowed", not a pure
            # wall-collision counter - named honestly rather than over-claimed.)
            dx, dy = float(cur.x) - prev_xy[0], float(cur.y) - prev_xy[1]
            if cmd_dist > 1.0 and math.hypot(dx, dy) < WALL_MOVE_EPS * cmd_dist:
                wall_block += 1
            de = float(cur.energy) - prev_e
            if de > 5.0:
                ate += 1
            prev_e = float(cur.energy)
        cur = next((x for x in core.env.agents if x.agent_id == aid), None)
        alive = cur is not None
        e_end = float(cur.energy) if alive else 0.0
        out_rows.append({
            "seed": seed, "tick": tick, "agent": aid, "action": name,
            "features": feats,
            "outcome": {
                "alive": alive,
                "died_at": died_at,
                "ticks": horizon if died_at is None else died_at,
                "energy": e_end,
                "e_frac_end": e_end / feats["max_energy"] if feats["max_energy"] else 0.0,
                "d_energy": e_end - e_start,
                "ate": ate,
                "lock_ticks": lock_ticks,
                "wall_block": wall_block,
                "pred_min_dist": (None if pred_min == 1e9 else pred_min),
                "pop_end": len(core.env.agents),
                "d_pop": len(core.env.agents) - pop_start,
                "score_gain": float(core.env.score) - score_start,
            },
        })


def gen(args):
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            seeds.append(int(part))
    out = open(args.out, "a")
    n_rows = 0
    for seed in seeds:
        P = op.load_params()
        fn = op.bc.make_policy(P)
        core = op.build_core(seed)
        op.bc.reset_memory()
        hist = []
        made = 0
        tick = 0
        while made < args.states_per_seed and tick < args.max_tick:
            op.roll(core, fn, args.every)
            tick += args.every
            if not core.env.agents:
                break
            healthy = [a for a in core.env.agents
                       if float(a.energy) >= HEALTHY_FRAC * float(getattr(a, "max_energy", 500.0) or 500.0)]
            if not healthy:
                continue
            focal = sorted(healthy, key=lambda a: a.agent_id)[made % len(healthy)]
            snap = op.snapshot(core.env)
            st0 = core.env.get_agent_state(focal.agent_id)
            rows = []
            study_state(core, snap, focal.agent_id, st0, args.horizon, hist, rows, seed, tick)
            for r in rows:
                out.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            made += 1
            op.restore(core, snap)
            op.roll(core, fn, args.every)
            tick += args.every
            cur = next((a for a in core.env.agents if a.agent_id == focal.agent_id), None)
            hist = (hist + [(0.0, 0.0, 0.0, False, False)])[-8:]
        print(f"seed {seed}: {made} states, {n_rows} rows total", flush=True)
        out.flush()
    out.close()
    print(f"wrote {n_rows} rows to {args.out}")


# ---------------------------------------------------------------- analysis
def utility(o):
    """A continuous utility for ranking branches. Deliberately transparent: survival first, then energy
    held, then fruit converted, minus lockout exposure and predator proximity. Documented so that the
    weighting can be argued about rather than hidden."""
    u = 0.0
    u += 1000.0 if o["alive"] else 0.0
    u += o["ticks"]
    u += 0.5 * o["e_frac_end"] * 100.0
    u += 40.0 * o["ate"]
    u -= 5.0 * o["lock_ticks"]
    if o["pred_min_dist"] is not None:
        u += min(o["pred_min_dist"], 200.0) / 20.0
    u += 10.0 * o["score_gain"]
    return u


def analyse(args):
    rows = []
    for pat in args.inputs:
        for fn in glob.glob(pat):
            with open(fn) as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    if not rows:
        print("no rows")
        return
    keys = sorted({r["seed"] for r in rows})
    print(f"loaded {len(rows)} branch rows from {len(keys)} seeds "
          f"({len(rows)//max(1,len(keys))} rows/seed)")

    # group by (seed, tick, agent) = a decision point
    points = {}
    for r in rows:
        points.setdefault((r["seed"], r["tick"], r["agent"]), []).append(r)
    print(f"decision points: {len(points)}")

    featnames = sorted(next(iter(rows))["features"].keys())
    X, y_beat, best_action, seed_of = [], [], [], []
    for (sd, tk, ag), rs in points.items():
        by = {r["action"]: utility(r["outcome"]) for r in rs}
        if "ctrl" not in by:
            continue
        ctrl_u = by["ctrl"]
        alts = {k: v for k, v in by.items() if k != "ctrl"}
        if not alts:
            continue
        best = max(alts, key=lambda k: alts[k])
        f = next(r for r in rs if r["action"] == "ctrl")["features"]
        X.append([f[k] for k in featnames])
        y_beat.append(1 if alts[best] > ctrl_u + args.margin else 0)
        best_action.append(best)
        seed_of.append(sd)

    X = np.asarray(X, dtype=float)
    y_beat = np.asarray(y_beat)
    print(f"\n=== does ANY alternative beat the controller? ===")
    print(f"  {int(y_beat.sum())}/{len(y_beat)} decision points ({100*y_beat.mean():.0f}%) have an "
          f"alternative worth > {args.margin} utility")
    if y_beat.mean() < 0.05:
        print("  -> alternatives essentially never win: the controller is already good on this action "
              "set, and this direction should be killed rather than expanded.")
        return

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:
        print(f"sklearn unavailable ({exc}); install scikit-learn to get the decision-boundary analysis")
        return

    # (a) WHICH FEATURES predict that an alternative beats the controller?
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=20, random_state=0)
    rf.fit(X, y_beat)
    imp = sorted(zip(featnames, rf.feature_importances_), key=lambda kv: -kv[1])[:10]
    print("\n=== which observable features predict that an alternative wins? (RF importance) ===")
    for k, v in imp:
        print(f"  {k:24s} {v:.3f}")

    # (b) DECISION BOUNDARIES: a shallow tree whose leaves say "here, action A is the best choice"
    print(f"\n=== decision boundaries (tree depth {args.depth}) ===")
    dt = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=args.min_leaf, random_state=0)
    dt.fit(X, best_action)
    print(export_text(dt, feature_names=featnames, max_depth=args.depth))

    # (c) STABILITY: does each leaf's advice hold on unseen seeds? A rule that works on one seed is noise.
    leaves = dt.apply(X)
    print(f"=== per-leaf stability across seeds (a rule must hold on a MAJORITY of seeds) ===")
    print(f"{'leaf':>5} {'n':>5} {'advises':>18} {'share':>7} {'seeds OK':>9}  verdict")
    for leaf in sorted(set(leaves)):
        idx = np.where(leaves == leaf)[0]
        if len(idx) < args.min_leaf:
            continue
        acts = [best_action[i] for i in idx]
        top = max(set(acts), key=acts.count)
        share = acts.count(top) / len(acts)
        per_seed = {}
        for i in idx:
            per_seed.setdefault(seed_of[i], []).append(best_action[i])
        ok = sum(1 for _, a in per_seed.items() if max(set(a), key=a.count) == top)
        verdict = "RULE" if (ok / max(1, len(per_seed)) >= 0.6 and share >= 0.5) else "noise"
        print(f"{leaf:>5} {len(idx):>5} {top:>18} {share:>6.2f} {f'{ok}/{len(per_seed)}':>9}  {verdict}")

    print("\nread this as: leaves marked RULE are candidate decision boundaries; encode one as a rule "
          "arm and test it against BASE on paired unseen seeds. Leaves marked noise are not reported.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seeds", default="3100-3105")
    g.add_argument("--states-per-seed", type=int, default=8)
    g.add_argument("--horizon", type=int, default=300)
    g.add_argument("--every", type=int, default=600)
    g.add_argument("--max-tick", type=int, default=12000)
    g.add_argument("--out", default="disc_rows.jsonl")
    a = sub.add_parser("analyse")
    a.add_argument("--inputs", nargs="+", default=["disc_rows.jsonl"])
    a.add_argument("--depth", type=int, default=3)
    a.add_argument("--min-leaf", type=int, default=25)
    a.add_argument("--margin", type=float, default=25.0)
    args = ap.parse_args()
    gen(args) if args.mode == "gen" else analyse(args)


if __name__ == "__main__":
    main()
