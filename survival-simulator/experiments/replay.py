#!/usr/bin/env python3
"""replay.py - record, replay and TEXTUALLY digest simulator episodes. READ-ONLY observability.

Purpose (Phase 1 of the observability programme): we have plenty of aggregate statistics and very
little understanding of what the agents physically DO. This records complete episodes - every entity,
every tick, plus the events and derived counters that explain HOW an agent got into the state where it
died - so a human (or later a model) can watch instead of inferring from means.

DESIGN CONSTRAINTS (deliberate):
  * read-only: the served controller and parameters are only ever READ; nothing is deployed or changed.
  * replayable: a recording is self-contained. Watching it needs no simulator, so the recorded bytes
    are evidence rather than a re-simulation that might differ.
  * text-first: every recording also renders to ASCII and to a numeric digest. Aggregate statistics hid
    the mechanism for two days; a text renderer means the same observation channel is available to a
    human in a terminal, to a text model, and to a vision model (frames->PNG) without three codepaths.
  * derived counters are recorded alongside raw state, because "they die below 100 energy" (measured)
    does not say WHY they got there. Wall blocking, turning away from visible fruit, oscillation, and
    predator-approach-while-facing-away are all measurable per agent per tick.

USAGE
    ./replay.py record  --seed 1500 --horizon 18000 --out /tmp/rp/seed1500.json.gz
    ./replay.py digest  /tmp/rp/seed1500.json.gz
    ./replay.py ascii   /tmp/rp/seed1500.json.gz --at 8000 --ticks 12
    ./replay.py corpus  --seeds 1500-1529 --horizon 18000 --outdir /tmp/rp --n 6
"""
import argparse
import gzip
import json
import math
import os
import random
import sys

# Reproducibility: Python randomises string hashing per process, and the simulator's entity iteration
# order can depend on it, so two identical runs in different processes can diverge on a minority of
# seeds (measured: 2 of 20 seeds, up to ~5,000 ticks). Recordings must not carry that noise.
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import best_controller as bc                                     # noqa: E402
from src.core import SimulationCore                              # noqa: E402
from env_wrapper import make_action                              # noqa: E402

DEPLOYED = os.path.join(ROOT, "best_controller", "params.json")
REST_GAIN = 3.0          # a resting predator gains ~dt*30; a jump above this means it ate someone
NEAR_PRED = 150.0        # "a predator is near"
FRUIT_NEAR = 220.0       # "fruit was visible and close enough to matter"
WALL_BLOCK_RATIO = 0.35  # moved less than this fraction of the commanded distance => blocked


# --------------------------------------------------------------------------- recording
def load_params(path=DEPLOYED):
    P = dict(bc.DEFAULT_PARAMS)
    blob = json.load(open(path))
    P.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    return P


def _ent(e):
    """Compact tuple for one entity."""
    return [round(float(getattr(e, "x", 0.0)), 1), round(float(getattr(e, "y", 0.0)), 1),
            round(float(getattr(e, "energy", 0.0)), 1)]


def record_episode(seed, horizon, params_path=DEPLOYED, every=2, agent_hist=12, traces=False):
    P = load_params(params_path)
    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)

    # biome_map holds biome OBJECTS (e.g. Desert_biome), not ids -> store a name->index legend
    bm = np.asarray(core.env.biome_map, dtype=object)
    names, legend = {}, []
    for cell in bm.ravel():
        n = type(cell).__name__
        if n not in names:
            names[n] = len(legend)
            legend.append(n)
    biome_idx = [[names[type(c).__name__] for c in row] for row in bm]

    rec = {"seed": int(seed), "horizon": int(horizon), "every": int(every),
           "params_sha": os.path.basename(params_path), "w": 1600, "h": 1200,
           "biome_legend": legend, "biome": biome_idx, "frames": [],
           "events": [], "edges": [],
           # ENERGY LEDGER (one row per tick). The simulator's accounting is fully known
           # (environment.py: metabolism `dt*biome_modifier` ~0.1/tick, age penalty 0.01*age,
           # movement 0.05/unit walking or speed*0.05+(extra)*0.5 sprinting, 100 per birth, fruit
           # absorption capped by max_energy). Recording the observable side of that balance per tick
           # lets us ask the question the aggregate statistics could not: does income per agent fall
           # BEFORE the population collapses (overshoot), or does the collapse happen at flat income
           # (something else)? Columns: t, pop, energy_sum, score, cmd_cost, births, deaths, lockout
           "ledger": []}

    prev_pred_e = {id(p): float(p.energy) for p in core.env.predators}
    prev_states = {}
    hist = {}          # agent_id -> ring of recent (dist, dir, turn, spawn)
    counters = {}      # agent_id -> derived counters
    edges_seen = {}
    t = -1
    alive_now = {}
    # PER-AGENT TRACES (opt-in): the raw material for "what do unusually successful agents DO
    # differently". Selection uses LINEAGE, not individual lifetime: within one episode, individual
    # outcomes are heavily contaminated by luck (spawn position, inherited max_energy, a predator
    # happening to be near at birth), so an agent is only "successful" if its DESCENDANTS are still
    # alive at the end. Parentage is inferred from the spawn cost: the simulator charges the parent
    # exactly 100 energy (environment.py:623), so the agent whose energy drops by ~100 on a birth tick
    # is the parent.
    A = {}

    def rec_agent(aid, tick):
        return A.setdefault(aid, {"id": aid, "birth": tick, "death": None, "ticks": 0, "travel": 0.0,
                                  "fruits": 0, "spawns": 0, "parent": None, "children": [],
                                  "lockout_ticks": 0, "obs_fruit": 0, "obs_pred": 0,
                                  "e_max": 0.0, "e_min": 1e9, "sprint_ticks": 0, "last_e": 0.0})

    for t in range(horizon):
        live = list(core.env.agents)
        if not live:
            break
        states = []
        for a in live:
            st = core.env.get_agent_state(a.agent_id)
            if st:
                st["x"] = float(getattr(a, "x", 0.0))
                st["y"] = float(getattr(a, "y", 0.0))
                st["direction"] = float(getattr(a, "direction", 0.0))
                st["max_age"] = float(getattr(a, "max_age", 999.0))
                states.append(st)
        if not states:
            break

        acts = []
        for st in states:
            aid = st["agent_id"]
            action = fn(st)
            acts.append((aid, make_action(st, action)))
            hist.setdefault(aid, []).append([round(float(action[0]), 2), round(float(action[1]), 3),
                                             round(float(action[2]), 3), bool(action[3])])
            if len(hist[aid]) > agent_hist:
                hist[aid].pop(0)
            c = counters.setdefault(aid, {"wall_block": 0, "turned_from_fruit": 0, "approach_pred": 0,
                                          "oscillate": 0, "lockout_ticks": 0, "ticks": 0,
                                          "fruit_ticks": 0, "saw_fruit": 0, "descend": 0})
            c["ticks"] += 1
            max_e = float(st.get("max_energy", 500.0) or 500.0)
            if float(st.get("energy", 0.0)) < 0.2 * max_e:
                c["lockout_ticks"] += 1

            obs = st.get("observations") or []
            for o in obs:
                if o.get("type") == "Edge":
                    xy = o.get("coords")
                    if xy:
                        # DETERMINISM/GEOMETRY: the simulator emits Edge coords RELATIVE to the agent
                        # and rotated by -direction (creature.py:181-184:
                        #   dx = s - self.x ; r = rot(-direction) * d ). Watching surfaces in the
                        # viewer needs ABSOLUTE world coordinates, so invert the transform here.
                        # Without this the "walls" were 64k distinct bogus segments - a rendering bug
                        # that looked like a world full of walls.
                        (rxs, rys), (rxe, rye) = xy
                        cd, sd = math.cos(st["direction"]), math.sin(st["direction"])
                        ax, ay = st["x"], st["y"]
                        key = ((round(ax + rxs * cd - rys * sd), round(ay + rxs * sd + rys * cd)),
                               (round(ax + rxe * cd - rye * sd), round(ay + rxe * sd + rye * cd)))
                        edges_seen[key] = 1
            fruit = [o for o in obs if o.get("type") == "Fruit"]
            near_fruit = [o for o in fruit if o["distance"] < FRUIT_NEAR]
            if near_fruit:
                c["saw_fruit"] += 1
                c["fruit_ticks"] += 1
                # "turned away": a close fruit exists and the agent's chosen direction points away from it
                angs = [o.get("angle", 0.0) for o in near_fruit]
                if angs and min(abs(a) for a in angs) > math.pi / 2:
                    c["turned_from_fruit"] += 1
            preds = [o for o in obs if o.get("type") == "Predator"]
            if preds:
                d = min(o["distance"] for o in preds)
                if d < NEAR_PRED:
                    c["approach_pred"] += 1
            if traces:
                ar = rec_agent(aid, t)
                ar["ticks"] += 1
                e_now = float(st.get("energy", 0.0))
                sp = float(st.get("speed", 10.0))
                d_cmd = float(action[0])
                ar["travel"] += d_cmd
                if d_cmd > sp:
                    ar["sprint_ticks"] += 1
                if e_now > ar["last_e"] + 5.0:            # energy jump = absorbed fruit
                    ar["fruits"] += 1
                ar["last_e"] = e_now
                ar["e_max"] = max(ar["e_max"], e_now)
                ar["e_min"] = min(ar["e_min"], e_now)
                ar["obs_fruit"] += len(fruit)
                ar["obs_pred"] += len(preds)
                if e_now < 0.2 * max_e:
                    ar["lockout_ticks"] += 1
            if len(hist[aid]) >= 3:
                turns = [h[2] for h in hist[aid][-3:]]
                if turns[0] * turns[1] < 0 and turns[1] * turns[2] < 0:
                    c["oscillate"] += 1

        before_pos = {st["agent_id"]: (st["x"], st["y"]) for st in states}
        prev_e = {st["agent_id"]: float(st["energy"]) for st in states}
        prev_pred_e = {id(p): float(p.energy) for p in core.env.predators}
        out = core.step(acts)
        alive_now = {a.agent_id: float(a.energy) for a in core.env.agents}

        # ---- wall blocking: commanded a move but barely moved ----
        for st in states:
            aid = st["agent_id"]
            if aid not in alive_now:
                continue
            cmd = next((a[1] for a in acts if a[0] == aid), None)
            if cmd is None:
                continue
            x0, y0 = before_pos[aid]
            a2 = next((a for a in core.env.agents if a.agent_id == aid), None)
            if a2 is None:
                continue
            moved = math.hypot(float(a2.x) - x0, float(a2.y) - y0)
            if float(cmd.move_distance) > 0.6 and moved < WALL_BLOCK_RATIO * float(cmd.move_distance):
                counters[aid]["wall_block"] += 1

        # ---- events: deaths (classified), births, eating ----
        ev = []
        for aid, st in ((s["agent_id"], s) for s in states):
            if aid in alive_now:
                continue
            eaten = any(prev_pred_e.get(id(p)) is not None
                        and float(p.energy) - prev_pred_e[id(p)] > REST_GAIN + 5.0
                        for p in core.env.predators)
            max_e = float(st.get("max_energy", 500.0) or 500.0)
            cause = "eaten" if eaten else ("aged" if float(st.get("age", 0.0)) > float(st.get("max_age", 999.0))
                                          else "starved")
            preds = [o for o in (st.get("observations") or []) if o.get("type") == "Predator"]
            if traces:
                rec_agent(aid, t)["death"] = t
            ev.append({"k": "die", "t": t, "a": aid, "cause": cause,
                       "x": round(st["x"], 1), "y": round(st["y"], 1),
                       "e": round(float(st.get("energy", 0.0)), 1),
                       "max_e": round(max_e, 1), "age": round(float(st.get("age", 0.0)), 1),
                       "lockout": bool(float(st.get("energy", 0.0)) < 0.2 * max_e),
                       "pred_dist": round(min([o["distance"] for o in preds]), 1) if preds else None,
                       "fruit_vis": len([o for o in (st.get("observations") or []) if o.get("type") == "Fruit"]),
                       "counters": counters.get(aid, {}), "recent": hist.get(aid, [])})
        for aid in alive_now:
            if aid not in prev_e:
                ev.append({"k": "born", "t": t, "a": aid})
                if traces:
                    # parentage by the spawn cost: the parent pays exactly 100 energy (environment.py:623)
                    par = None
                    best = 0.0
                    for a2 in prev_e:
                        if a2 in alive_now:
                            d_e = prev_e[a2] - alive_now[a2]
                            if d_e > 90.0 and d_e > best:
                                par, best = a2, d_e
                    if par is not None:
                        rec_agent(aid, t)["parent"] = par
                        rec_agent(par, t)["spawns"] += 1
                        A[par]["children"].append(aid)
        # score jump = fruit eaten (dt term is 0.1)
        if out["score"] and t > 0:
            pass
        rec["events"] += ev

        if t % every == 0:
            rec["frames"].append({
                "t": t, "score": round(float(out["score"]), 4),
                "n": len(alive_now),
                "e_mean": round(float(np.mean(list(alive_now.values()))), 1) if alive_now else 0.0,
                "e_min": round(float(min(alive_now.values())), 1) if alive_now else 0.0,
                "lock": sum(1 for st in states if float(st.get("energy", 0.0)) < 0.2 * float(st.get("max_energy", 500.0) or 500.0)),
                "agents": [[st["agent_id"], round(st["x"], 1), round(st["y"], 1),
                            round(float(st.get("energy", 0.0)), 1), round(st.get("direction", 0.0), 3),
                            float(st.get("vision_range", 300.0)), float(st.get("vision_angle", 1.0)),
                            float(st.get("max_energy", 500.0) or 500.0)]
                           for st in states if st["agent_id"] in alive_now],
                "preds": [_ent(p) for p in core.env.predators],
                "fruits": [_ent(f) for f in core.env.fruits],
                "trees": [[round(float(getattr(tr, "x", 0)), 0), round(float(getattr(tr, "y", 0)), 0)]
                          for tr in core.env.trees],
            })

        # ---- energy ledger row: the observable side of the simulator's energy balance ----
        n_births = sum(1 for e in ev if e["k"] == "born")
        n_deaths = sum(1 for e in ev if e["k"] == "die")
        cmd_cost = 0.0
        for st in states:
            d = float(next((a[1].move_distance for a in acts if a[0] == st["agent_id"]), 0.0))
            sp = float(st.get("speed", 10.0))
            # exactly the simulator's formula: walk 0.05/unit, sprint speed*0.05 + extra*0.5
            cmd_cost += min(d, sp) * 0.05 + max(0.0, d - sp) * 0.5
        rec["ledger"].append([t, len(alive_now), round(sum(alive_now.values()), 1),
                              round(float(out["score"]), 4), round(cmd_cost, 2), n_births, n_deaths,
                              sum(1 for st in states
                                  if float(st.get("energy", 0.0)) < 0.2 * float(st.get("max_energy", 500.0) or 500.0))])

        if int(out.get("num_agents", len(alive_now))) == 0:
            # STOP CONDITION FIDELITY: env_wrapper.run_eval_episode ends the episode on the simulator's
            # own `num_agents == 0` signal, not on len(env.agents). Without this the recorder could
            # append a phantom extra tick (measured on x86: seed 1500 gave 3883 vs the reference's
            # 3882, with identical score). A replay that shows a tick the real run never had is exactly
            # the "diff synth" failure mode the viewer must not have.
            break

    rec["edges"] = [[list(k[0]), list(k[1])] for k in edges_seen.keys()]
    # Strict-comparison fields. `score` is rounded for compactness in the recording, so a tolerance of
    # 1e-3 is the right test of BEHAVIOUR (the strict 1e-6 test fails on formatting, not on fidelity).
    rec["final"] = {"score": round(float(core.env.score), 4), "ticks": t + 1,
                    "survived": bool(alive_now),
                    "eaten": sum(1 for e in rec["events"] if e["k"] == "die" and e["cause"] == "eaten"),
                    "starved": sum(1 for e in rec["events"] if e["k"] == "die" and e["cause"] == "starved"),
                    "aged": sum(1 for e in rec["events"] if e["k"] == "die" and e["cause"] == "aged")}
    if traces:
        # Lineage success = how many DESCENDANTS an agent left, and how many were still alive at the end.
        # This is the selector for "unusually successful agent": an agent whose line survives is the one
        # whose behaviour (and genes) actually worked, whereas individual lifetime is contaminated by
        # luck (spawn position, inherited max_energy, a predator near at birth).
        alive_ids = set(alive_now)

        def descendants(aid):
            seen, stack = set(), list(A.get(aid, {}).get("children", []))
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                stack += A.get(c, {}).get("children", [])
            return seen

        rows = []
        for aid, a in A.items():
            d = descendants(aid)
            b = dict(a)
            b["descendants"] = len(d)
            b["descendants_alive_end"] = len(d & alive_ids)
            b["alive_end"] = aid in alive_ids
            b["travel_per_fruit"] = (a["travel"] / a["fruits"]) if a["fruits"] else None
            b["lockout_frac"] = (a["lockout_ticks"] / a["ticks"]) if a["ticks"] else 0.0
            rows.append(b)
        rec["agents"] = rows
    return rec


# --------------------------------------------------------------------------- text renderers
def ascii_frame(frame, cols=96, rows=34, w=1600, h=1200):
    grid = [[" "] * cols for _ in range(rows)]
    sx, sy = cols / w, rows / h

    def put(x, y, ch):
        c, r = int(x * sx), int(y * sy)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = ch

    for e in frame.get("edges", []):
        (x0, y0), (x1, y1) = e
        n = max(2, int(math.hypot((x1 - x0) * sx, (y1 - y0) * sy)))
        for i in range(n + 1):
            put(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, "#")
    for t in frame.get("trees", []):
        put(t[0], t[1], "T")
    for f in frame.get("fruits", []):
        put(f[0], f[1], ".")
    for p in frame.get("preds", []):
        put(p[0], p[1], "P")
    for a in frame.get("agents", []):
        ch = "@" if a[3] > 0.2 * (a[7] or 500.0) else ("o" if a[3] > 100 else "!")   # ! = lockout zone
        put(a[1], a[2], ch)
    lines = ["+" + "-" * cols + "+"]
    lines += ["|" + "".join(r) + "|" for r in grid]
    lines += ["+" + "-" * cols + "+"]
    hud = (f"t={frame['t']} score={frame['score']} agents={frame['n']} "
           f"e_mean={frame['e_mean']} e_min={frame['e_min']} in_lockout={frame['lock']} "
           f"fruit_on_map={len(frame.get('fruits', []))} predators={len(frame.get('preds', []))}")
    return "\n".join(lines) + "\n" + hud


def digest(rec, every_bucket=1000):
    print(f"=== REPLAY seed={rec['seed']} ticks={rec['final']['ticks']} "
          f"score={rec['final']['score']} survived_to_horizon={rec['final']['survived']} ===")
    print(f"frames {len(rec['frames'])} | events {len(rec['events'])} | walls(edges) {len(rec['edges'])}")
    cause = {}
    for e in rec["events"]:
        if e["k"] == "die":
            cause[e["cause"]] = cause.get(e["cause"], 0) + 1
    print(f"deaths by cause: {cause}")
    # The lockout share is only meaningful for PREDATION deaths: a starved or aged agent is at ~0
    # energy by definition, so counting it would over-claim the meaning of the statistic.
    eaten = [e for e in rec["events"] if e["k"] == "die" and e["cause"] == "eaten"]
    if eaten:
        lock = sum(1 for e in eaten if e.get("lockout"))
        print(f"  predation deaths: {len(eaten)}, of which in the <20% energy sprint-lockout zone: "
              f"{lock} ({100*lock/len(eaten):.0f}%)")
    other = [e for e in rec["events"] if e["k"] == "die" and e["cause"] != "eaten"]
    if other:
        print(f"  non-predation deaths: {len(other)} (starved/aged - trivially at low energy, "
              f"so not counted as lockout evidence)")
    print("\n  t     pop  e_mean  e_min  lock  fruit  preds")
    for i, fr in enumerate(rec["frames"]):
        if i % max(1, every_bucket // max(1, rec["every"])) == 0:
            print(f"  {fr['t']:6d} {fr['n']:4d} {fr['e_mean']:8.1f} {fr['e_min']:6.1f} "
                  f"{fr['lock']:5d} {len(fr['fruits']):6d} {len(fr['preds']):6d}")
    print("\n--- how agents died (the WHY context, not just the count) ---")
    for e in rec["events"]:
        if e["k"] == "die":
            c = e.get("counters", {})
            print(f"  t={e['t']:6d} agent {e['a']:3d} {e['cause']:8s} energy {e['e']:6.1f}/{e['max_e']:.0f} "
                  f"age {e['age']:6.1f} lockout={int(bool(e['lockout']))} "
                  f"pred_dist={e['pred_dist']} fruit_visible={e['fruit_vis']} | "
                  f"wall_blocked {c.get('wall_block', 0)}x turned_from_fruit {c.get('turned_from_fruit', 0)}x "
                  f"oscillated {c.get('oscillate', 0)}x lockout_ticks {c.get('lockout_ticks', 0)}/{c.get('ticks', 0)}")
    print("\n--- final frame ---")
    print(ascii_frame(rec["frames"][-1]))


# --------------------------------------------------------------------------- corpus
def make_corpus(seeds, horizon, outdir, n=6, traces=False):
    os.makedirs(outdir, exist_ok=True)
    index = []
    for sd in seeds:
        rec = record_episode(sd, horizon, traces=traces)
        idx = {"seed": sd, "ticks": rec["final"]["ticks"], "score": rec["final"]["score"],
               "causes": {}, "wall_blocked_total": 0, "pred_deaths": 0, "min_energy_seen": 9e9}
        for e in rec["events"]:
            if e["k"] == "die":
                idx["causes"][e["cause"]] = idx["causes"].get(e["cause"], 0) + 1
                idx["wall_blocked_total"] += e.get("counters", {}).get("wall_block", 0)
        idx["min_energy_seen"] = min((fr["e_min"] for fr in rec["frames"]), default=0.0)
        path = os.path.join(outdir, f"seed{sd}_{rec['final']['ticks']}t.json.gz")
        with gzip.open(path, "wt") as fh:
            json.dump(rec, fh)
        idx["path"] = path
        index.append(idx)
        print(f"recorded seed {sd}: {rec['final']['ticks']} ticks, causes {idx['causes']} -> {path}",
              flush=True)
    # pick representatives
    alive = [i for i in index]
    picks = {"base": sorted(alive, key=lambda i: abs(i["ticks"] - horizon // 2))[:1],
             "short_failure": sorted(alive, key=lambda i: i["ticks"])[:1],
             "long_survival": sorted(alive, key=lambda i: -i["ticks"])[:1],
             "predator_death": sorted(alive, key=lambda i: -i["pred_deaths"])[:1],
             "wall_heavy": sorted(alive, key=lambda i: -i["wall_blocked_total"])[:1],
             "low_energy": sorted(alive, key=lambda i: i["min_energy_seen"])[:1]}
    man = {"horizon": horizon, "seeds": seeds, "index": index,
           "representatives": {k: v[0]["path"] if v else None for k, v in picks.items()}}
    json.dump(man, open(os.path.join(outdir, "MANIFEST.json"), "w"), indent=1)
    print("\n=== representatives ===")
    for k, v in man["representatives"].items():
        print(f"  {k:15s} {v}")
    return man


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--seed", type=int, required=True)
    r.add_argument("--horizon", type=int, default=18000); r.add_argument("--out", required=True)
    r.add_argument("--every", type=int, default=2); r.add_argument("--params", default=DEPLOYED)
    r.add_argument("--traces", action="store_true", help="also record per-agent traces + lineage")
    d = sub.add_parser("digest"); d.add_argument("path")
    a = sub.add_parser("ascii"); a.add_argument("path"); a.add_argument("--at", type=int, default=None)
    a.add_argument("--ticks", type=int, default=6)
    c = sub.add_parser("corpus"); c.add_argument("--seeds", required=True)
    c.add_argument("--horizon", type=int, default=18000); c.add_argument("--outdir", required=True)
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--traces", action="store_true", help="also record per-agent traces + lineage")
    args = ap.parse_args()

    if args.cmd == "record":
        rec = record_episode(args.seed, args.horizon, args.params, every=args.every,
                             traces=getattr(args, "traces", False))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with gzip.open(args.out, "wt") as fh:
            json.dump(rec, fh)
        print(f"recorded {len(rec['frames'])} frames, {len(rec['events'])} events -> {args.out}")
        digest(rec)
    elif args.cmd == "digest":
        digest(json.load(gzip.open(args.path, "rt")))
    elif args.cmd == "ascii":
        rec = json.load(gzip.open(args.path, "rt"))
        frames = rec["frames"]
        if args.at is not None:
            i = min(range(len(frames)), key=lambda k: abs(frames[k]["t"] - args.at))
        else:
            i = len(frames) - 1
        for fr in frames[i:i + args.ticks]:
            print(ascii_frame(fr, w=rec.get("w", 1600), h=rec.get("h", 1200)))
            print()
    elif args.cmd == "corpus":
        seeds = []
        for part in args.seeds.split(","):
            if "-" in part:
                a2, b2 = part.split("-"); seeds += list(range(int(a2), int(b2) + 1))
            else:
                seeds.append(int(part))
        make_corpus(seeds, args.horizon, args.outdir, args.n, traces=getattr(args, "traces", False))


if __name__ == "__main__":
    main()
