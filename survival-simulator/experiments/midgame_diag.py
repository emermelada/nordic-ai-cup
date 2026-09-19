#!/usr/bin/env python3
"""midgame_diag.py - WHY do births stop in the mid-game (ticks 4,000-10,000)?

INSTRUMENTATION ONLY. This file never changes the controller's behaviour: it calls the real
deployed policy and then, separately, RECOMPUTES the spawn gate from the controller's own module
state (bc._MEM / bc._GC / bc._TRAITS / bc._GS_LAST) to attribute every refusal to a specific gate.
The controller source is imported, never edited.

Three hypotheses, each implying a different intervention:

  H1 BUDGET  births stop because the energy gate (ef > rf) can no longer be met as mean energy
             falls -> the fleet is over its carrying capacity -> intervention: scale the repro
             target with production (cc_k0 / cc_halflife / cc_floor).
  H2 GATE    births stop because the PERMISSION gates block them: repro_safe_radius (253.5) against
             a growing predator population, repro_popcap (6) local agents, or pop_ok (gpop < 12).
             -> intervention: relax the gate in the mid-game window only.
  H3 WAVE    the collapse is a synchronized cohort age-wave (births are bursty early; max_age is a
             60-120 s window), so deaths spike at a fixed offset from a birth spike with energy NOT
             depressed beforehand. -> intervention: desynchronise the relay.

They predict different observable things, so one run separates them.

OUTPUT: one JSON line per (seed, bucket) with the aggregate state, plus per-episode totals.
  python3 midgame_diag.py --seeds 2000-2199 --horizon 18000 --workers 64 --out midgame.jsonl
"""
import argparse
import json
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

BUCKET = 100  # ticks per aggregate row


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_episode(args):
    arm, seed, horizon, params_path, overrides = args
    import best_controller as bc
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    P = dict(bc.DEFAULT_PARAMS)
    with open(params_path) as f:
        P.update(json.load(f))
    P.update(overrides or {})

    random.seed(seed)
    np.random.seed(seed)
    bc.reset_memory()
    policy = bc.make_policy(P)

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)

    # ---- gate parameters mirrored from the controller's own DEFAULT_PARAMS + deployed overrides ----
    TARGET = float(P.get("repro_global_target", 10) or 0.0)
    POPCAP = float(P.get("repro_popcap", 2) or 0.0)
    SAFE_R = float(P.get("repro_safe_radius", 330.0) or 0.0)
    RF = float(P.get("repro_frac", 0.82) or 0.0)
    RF_MIN = float(P.get("repro_frac_min", 0.24) or 0.0)
    URGENCY = float(P.get("repro_urgency", 1.0) or 0.0)
    ABS_GATE = float(P.get("repro_energy_abs", 0.0) or 0.0)
    COOLDOWN = int(P.get("spawn_cooldown", 400) or 0)
    GS = float(P.get("genome_select", 0.0) or 0.0)
    GS_TOPK = float(P.get("gs_topk", 2.0) or 0.0)
    GS_MIN_KNOWN = float(P.get("gs_min_known", 3.0) or 0.0)
    GS_RESCUE_POP = float(P.get("gs_rescue_pop", 1.0) or 0.0)

    prev = {}          # agent_id -> (energy, age, max_age, min_pred_dist, near_pred)
    prev_pe = 0.0      # total predator energy last tick
    prev_npred = 0
    buckets = {}       # bucket index -> accumulator
    last_score = 0.0
    total_births = total_deaths = 0
    death_causes = {"eaten": 0, "starved": 0, "aged": 0}
    first_zero = None

    def acc(t):
        b = t // BUCKET
        if b not in buckets:
            buckets[b] = {"t0": b * BUCKET, "n": 0, "e_sum": 0.0, "ef_sum": 0.0, "age_sum": 0.0,
                          "pred_sum": 0.0, "pred_near": 0, "fruits_vis": 0, "rf_sum": 0.0,
                          "elig": 0, "born": 0, "died": 0,
                          "ref": {"gate_ok": 0, "cooldown": 0, "pop_ok": 0, "crowd_ok": 0,
                                  "safe": 0, "gs_ok": 0},
                          "ref_ef": 0.0, "ref_n": 0, "ref_pred_d": 0.0,
                          "deaths": {"eaten": 0, "starved": 0, "aged": 0},
                          "ages": [],
                          # genome ratchet readout + rank distribution among ELIGIBLE spawners
                          "rank_hist": {}, "breeders": 0, "trait_n": 0,
                          "cap_sum": 0.0, "cap_max": 0.0, "vis_sum": 0.0, "vis_max": 0.0,
                          "pred_kills": 0.0, "pred_e_sum": 0.0, "npred": 0, "ticks": 0,
                          "lockout": 0, "e_move": 0.0, "cmd": 0.0, "income": 0.0,
                          "mode": {}, "mode_e": {}}
        return buckets[b]

    for i in range(horizon):
        live = list(core.env.agents)
        if not live:
            first_zero = i
            break

        states = []
        for a in live:
            st = core.env.get_agent_state(a.agent_id)
            if st:
                st["_max_age"] = float(getattr(a, "max_age", 999.0))
                states.append(st)

        gpop = bc._global_alive()  # the controller's OWN cooperative population estimate

        acts = []
        for st in states:
            aid = st["agent_id"]
            energy = float(st.get("energy", 0.0) or 0.0)
            max_e = max(float(st.get("max_energy", 1.0) or 1.0), 1.0)
            ef = energy / max_e
            preds = [o for o in (st.get("observations") or []) if o.get("type") == "Predator"]
            agents_vis = [o for o in (st.get("observations") or []) if o.get("type") == "Agent"]
            fruits = [o for o in (st.get("observations") or []) if o.get("type") == "Fruit"]
            min_pd = min((p["distance"] for p in preds), default=-1.0)

            # --- call the REAL policy first: no behaviour is altered by anything below ---
            dist, dr, turn, spawn = policy(st)

            # --- then recompute the gate the controller used, for attribution only ---
            b = acc(i)
            b["n"] += 1
            b["e_sum"] += energy
            b["ef_sum"] += ef
            b["age_sum"] += float(st.get("age", 0.0) or 0.0)
            b["pred_sum"] += (min_pd if min_pd >= 0 else 0.0)
            b["pred_near"] += 1 if (0 <= min_pd < SAFE_R) else 0
            b["fruits_vis"] += len(fruits)
            b["ages"].append(float(st.get("age", 0.0) or 0.0))

            # sprint lockout: environment.py:512 clamps movement to walking speed below
            # energy < max_energy/5. Fraction below it is the exposure the ablation implicated.
            if energy < max_e / 5.0:
                b["lockout"] += 1
            # movement expenditure ACTUALLY charged. The ordering matters (environment.py:508-518):
            # the sim caps at sprint_speed, THEN clamps to walking speed if energy < max_energy/5,
            # and only then charges. Using the COMMANDED distance overstates the cost of every
            # sprint-locked agent, and lockout is 47-87% of agent-ticks.
            spd = float(st.get("speed", 10.0) or 10.0)
            spr = float(st.get("sprint_speed", 20.0) or 20.0)
            d = min(max(dist, 0.0), spr)
            if energy < max_e / 5.0 and d > spd:
                d = spd
            _em = d * 0.05 if d <= spd else spd * 0.05 + (d - spd) * 0.5
            b["e_move"] += _em
            b["cmd"] += dist

            # MOVEMENT MODE, attributed the same way as the gates: read the latch the controller
            # just set, otherwise recompute the branch conditions from the state.
            _m = bc._MEM.get(aid) or {}
            if _m.get("flee"):
                mode = "flee"                      # preempts everything (early return)
            elif float(P.get("forage_nearest", 0.0) or 0.0) > 0.0 and fruits:
                mode = "forage"                    # direct-to-nearest-fruit override
            elif fruits and min(f["distance"] for f in fruits) < 40.0:
                mode = "close"
            elif ef < float(P.get("low_energy_frac", 0.35) or 0.0):
                mode = "low"
            elif fruits:
                mode = "walk"
            else:
                mode = "blind"                     # the 0.12 x speed crawl
            b["mode"][mode] = b["mode"].get(mode, 0) + 1
            b["mode_e"][mode] = b["mode_e"].get(mode, 0.0) + _em

            deficit = max(0.0, min(1.0, (TARGET - gpop) / TARGET)) if TARGET > 0 else 0.0
            rf = RF - (RF - RF_MIN) * deficit * URGENCY
            b["rf_sum"] += rf

            # genome ratchet readout: the heritable traits this fleet is actually carrying
            b["trait_n"] += 1
            cap = max_e
            vis = float(st.get("vision_range", 0.0) or 0.0)
            b["cap_sum"] += cap
            b["cap_max"] = max(b["cap_max"], cap)
            b["vis_sum"] += vis
            b["vis_max"] = max(b["vis_max"], vis)

            # an agent is ELIGIBLE for a spawn this tick if it is off cooldown and can afford it
            m = bc._MEM.get(aid) or {}
            off_cd = (m.get("spawn_clock", 0) <= 0)
            gate_ok = (energy > ABS_GATE) if ABS_GATE > 0.0 else (ef > rf)
            if not off_cd:
                b["ref"]["cooldown"] += 1
            elif not gate_ok:
                b["ref"]["gate_ok"] += 1
                b["ref_ef"] += ef
                b["ref_n"] += 1
                b["ref_pred_d"] += (min_pd if min_pd >= 0 else SAFE_R)
            else:
                # it can afford it; did a PERMISSION gate refuse?
                b["elig"] += 1
                pop_ok = gpop < TARGET
                crowd_ok = len(agents_vis) <= POPCAP
                safe = all(p["distance"] >= SAFE_R for p in preds)
                gs_ok = True
                rank = None
                if GS > 0.0 and bc._GS_LAST is not None:
                    own_u, rank, n_alive = bc._GS_LAST
                    gs_ok = ((n_alive < GS_MIN_KNOWN) or (gpop <= GS_RESCUE_POP)
                             or (rank <= GS_TOPK))
                if rank is not None:
                    key = str(int(rank)) if rank <= 12 else ">12"
                    b["rank_hist"][key] = b["rank_hist"].get(key, 0) + 1
                    if rank <= GS_TOPK:
                        b["breeders"] += 1
                if not pop_ok:
                    b["ref"]["pop_ok"] += 1
                if not crowd_ok:
                    b["ref"]["crowd_ok"] += 1
                if not safe:
                    b["ref"]["safe"] += 1
                if not gs_ok:
                    b["ref"]["gs_ok"] += 1

            # EXACT predation detector: the sim kills an agent when a predator is within
            # predator.size + agent.size = 10 + 5 = 15 units (environment.py:720-727). This is
            # independent of the score arithmetic, which is polluted by the same-tick dt and fruit.
            near_pred = (0.0 <= min_pd < 15.0)
            prev[aid] = (energy, float(st.get("age", 0.0) or 0.0), st["_max_age"], min_pd, near_pred)
            acts.append((aid, ActionRequest(agent_id=aid, move_distance=dist, move_direction=dr,
                                            turn_angle=turn, spawn_agent=bool(spawn))))

        before = {a.agent_id for a in core.env.agents}
        out = core.step(acts)
        after = {a.agent_id for a in core.env.agents}
        score_delta = out["score"] - last_score
        last_score = out["score"]

        pe = sum(float(getattr(pp, "energy", 0.0)) for pp in core.env.predators)
        npred = len(core.env.predators)
        b = acc(i)
        # A resting predator gains dt*30 = 3/tick (environment.py:683). A jump beyond that is a
        # kill: predator.energy += agent.energy (environment.py:725). This detects predation
        # exactly, without the score arithmetic that the same-tick dt and fruit bonus pollute.
        jump = pe - prev_pe - 3.0 * min(npred, prev_npred)
        if jump > 1.0:
            b["pred_kills"] += 1
            b["pred_e_sum"] += jump
        # FRUIT INCOME in energy, from the score identity (environment.py:674 / 759 / 726):
        #   score_delta = dt + sum(fruit.energy)/1000 - sum(victim.energy)/100
        # the predation term is recovered from the predator energy jump (capped by predator
        # max_energy, so this is a slight underestimate when a predator is already full).
        b["income"] += 1000.0 * (score_delta - 0.1) + 10.0 * max(0.0, jump)
        b["npred"] += npred
        b["ticks"] = b.get("ticks", 0) + 1
        prev_pe, prev_npred = pe, npred
        born = len(after - before)
        gone = before - after
        b["born"] += born
        b["died"] += len(gone)
        total_births += born
        total_deaths += len(gone)

        for aid in gone:
            e, age, max_age, mpd, near_pred = prev.get(aid, (0.0, 0.0, 999.0, -1.0, False))
            if near_pred:
                cause = "eaten"
            elif age > max_age:
                cause = "aged"
            else:
                cause = "starved"
            b["deaths"][cause] += 1
            death_causes[cause] += 1
        prev = {a.agent_id: prev.get(a.agent_id, (0.0, 0.0, 999.0, -1.0, False))
                for a in core.env.agents}

    # ---- per-episode totals ----
    rows = []
    for bi in sorted(buckets):
        b = buckets[bi]
        n = max(1, b["n"])
        ages = sorted(b["ages"])
        rows.append({
            "arm": arm, "seed": seed, "t0": b["t0"], "samples": b["n"],
            "pop": n / BUCKET,
            "e_mean": b["e_sum"] / n, "ef_mean": b["ef_sum"] / n, "rf_mean": b["rf_sum"] / n,
            "age_mean": b["age_sum"] / n,
            "age_p10": ages[int(0.1 * (len(ages) - 1))] if ages else 0.0,
            "age_p90": ages[int(0.9 * (len(ages) - 1))] if ages else 0.0,
            "pred_near_frac": b["pred_near"] / n, "pred_d_mean": b["pred_sum"] / n,
            "fruits_vis": b["fruits_vis"] / n,
            "born": b["born"], "died": b["died"], "elig": b["elig"],
            "ref": b["ref"],
            "ref_ef_mean": (b["ref_ef"] / b["ref_n"]) if b["ref_n"] else None,
            "ref_pred_d_mean": (b["ref_pred_d"] / b["ref_n"]) if b["ref_n"] else None,
            "deaths": b["deaths"],
            "rank_hist": b["rank_hist"],
            "breeder_frac": (b["breeders"] / b["elig"]) if b["elig"] else None,
            "cap_mean": b["cap_sum"] / max(1, b["trait_n"]),
            "cap_max": b["cap_max"],
            "vis_mean": b["vis_sum"] / max(1, b["trait_n"]),
            "vis_max": b["vis_max"],
            "lockout_frac": b["lockout"] / max(1, b["n"]),
            "mode_pct": {k: 100.0 * v / max(1, b["n"]) for k, v in b["mode"].items()},
            "mode_e_pct": {k: 100.0 * v / max(1e-9, b["e_move"]) for k, v in b["mode_e"].items()},
            "e_move_per_tick": b["e_move"] / max(1, b["n"]),
            "cmd_per_tick": b["cmd"] / max(1, b["n"]),
            "income_per_tick": b["income"] / max(1, b["ticks"]),
            "income_per_1k": b["income"] * 1000.0 / max(1, b["ticks"]),
            "pred_kills": b["pred_kills"], "pred_e_gain": b["pred_e_sum"],
            "npred": b["npred"] / max(1, b["ticks"]),
        })
    summary = {"arm": arm, "seed": seed, "ticks": first_zero if first_zero is not None else horizon,
               "score": round(last_score, 3), "born": total_births, "died": total_deaths,
               "causes": death_causes}
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2000-2199")
    ap.add_argument("--horizon", type=int, default=18000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--params", default=os.path.join(ROOT, "best_controller", "params.json"))
    ap.add_argument("--arms", default="", help="JSON list of {id, params}; same format as sched.py")
    ap.add_argument("--out", default=os.path.join(HERE, "midgame_diag.jsonl"))
    ap.add_argument("--sumout", default=os.path.join(HERE, "midgame_diag_summary.json"))
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    # ARMS: the first arm is BASE. Without --arms we run the deployed controller alone.
    if args.arms:
        arms = [{"id": c.get("id") or c.get("tag"), "params": c.get("params", {})}
                for c in json.load(open(args.arms))]
    else:
        arms = [{"id": "BASE", "params": {}}]

    print(f"midgame_diag | {len(arms)} arms x {len(seeds)} seeds | horizon {args.horizon} "
          f"| workers {args.workers} | params {args.params}", flush=True)
    with open(args.params) as f:
        dep = json.load(f)
    print("  deployed params:", {k: dep.get(k) for k in
                                 ("genome_select", "gs_topk", "gs_w_energy", "evade_mode",
                                  "reserve_frac", "blind_explore_frac", "repro_global_target",
                                  "repro_safe_radius")}, flush=True)
    for a in arms:
        ov = {k: v for k, v in a["params"].items() if not k.startswith("__") and k != "id"}
        print(f"    {a['id']:<22} overrides: {ov or '(none = deployed)'}", flush=True)

    from multiprocessing import get_context
    ctx = get_context("spawn")
    jobs = [(a["id"], s, args.horizon, args.params, a["params"]) for a in arms for s in seeds]
    n_rows = n_done = 0
    with open(args.out, "w") as fh, open(args.sumout, "w") as sh:
        with ctx.Pool(args.workers) as pool:
            for rows, summary in pool.imap_unordered(run_episode, jobs, chunksize=1):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                n_rows += len(rows)
                sh.write(json.dumps(summary) + "\n")
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  {n_done}/{len(jobs)} episodes done ({n_rows} bucket rows)",
                          flush=True)
    print(f"done: {n_rows} bucket rows -> {args.out}; summaries -> {args.sumout}", flush=True)


if __name__ == "__main__":
    main()
