#!/usr/bin/env python3
"""traj_map.py - PHASE 1/2 forensics: map the score distribution of the IMMUTABLE hive controller.

Paste-8 Phase 1/2, no controller change, no hypothesis up front: run hive on a large FRESH seed
block in the REAL simulator and record, per seed, the full trajectory so the data can name the
earliest separator between HIGH and LOW runs.

WHAT IS RECORDED (per seed, every --bucket ticks = 10 sim-seconds)
  pop, active agents, births, deaths split starved-young / starved-old / eaten,
  energy: mean median max min sd, frac(ef<0.35) frac(ef<0.2)   (ef = energy/max_energy)
  economy: fruit eaten n + energy in, movement cost, turn cost, biome drain, old-age drain,
           spawn cost, eaten-loss                                     -- all EXACT, not estimated
  perception: agents seeing >=1 Fruit, mean fruit seen, predator seen, tree seen, agent seen
  behaviour: frac requested-stationary, mean requested distance, mean actual displacement,
             requested spawns, accepted spawns
  genes: mean speed/sprint/hearing/vision/max_energy, sd max_energy, mean age, spatial sd(x), sd(y)
  hive: mode_dist/mode_ticks snapshots (its own internal action-mode bookkeeping)

ENERGY ACCOUNTING IS EXACT AND SELF-CHECKED.  env.score accrues exactly dt=0.1/tick plus
fruit.energy/1000 per fruit eaten minus agent.energy/100 per agent eaten, so
    fruit_in = (score_delta - 0.1 + eaten_energy/100) * 1000
is exact.  Every other term (walk/sprint cost, turn cost, biome drain, old-age drain, spawn cost)
is recomputed from the pre-tick state with the simulator's own constants, and the per-seed
residual (post_sum - expected_sum) is stored as `closure` so a broken accounting is visible
instead of silently believed.  Kill events are captured by shadowing env.kill_agent on the
INSTANCE (observation only; no simulator source is modified).

RUN (on the remote boxes, never on the Mac)
  PYTHONHASHSEED=0 /opt/nacv/bin/python traj_map.py --seeds 50000-50199 --workers 60 --out /opt/nac_traj/out
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from multiprocessing import get_context  # noqa: E402

DT = 0.1
WALK_COST = 0.05          # environment.update_entity_position
SPRINT_COST = 0.5
NEWBORN_ENERGY = 75.0     # Agent default energy (spawn_agent's mutation path passes none)
SPAWN_DRAIN = 100.0

# metric columns kept per bucket (order matters for the compact numpy storage)
COLS = ["t", "pop", "active", "births", "d_sy", "d_so", "d_eat", "fruit_n", "fruit_E",
        "c_move", "c_turn", "c_biome", "c_old", "c_spawn", "e_in_eaten",
        "E_mean", "E_med", "E_max", "E_min", "E_sd", "ef_lt35", "ef_lt20",
        "see_fruit_1", "fruit_seen", "pred_seen", "tree_seen", "agent_seen",
        "stat_frac", "req_d", "disp", "spawn_req", "spawn_acc",
        "g_speed", "g_sprint", "g_hear", "g_vis", "g_maxE", "g_maxE_sd", "g_age",
        "sd_x", "sd_y"]


def move_cost(dist, e, speed, sprint, max_energy):
    """Exact copy of the simulator's cost rule (charged BEFORE the biome modifier is applied)."""
    if dist < 0:
        dist = 0.0
    if dist > sprint:
        dist = sprint
    if e < max_energy / 5.0 and dist > speed:
        dist = speed
    if dist <= speed:
        return dist * WALK_COST, dist
    return speed * WALK_COST + (dist - speed) * SPRINT_COST, dist


def run_seed(seed, horizon, bucket, hive_seed=0):
    import numpy as np
    from hive_v2 import Hive
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    random.seed(seed)
    np.random.seed(seed)
    hive = Hive(seed=hive_seed)   # deployed server.py uses Hive(params=PARAMS) => hive seed 0
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
    env = core.env

    # ---- observation-only hooks: exact kill records (tick is patched by the loop) -------------
    kills = []
    state = {"tick": 0}

    def _kill(agent):
        try:
            kills.append((state["tick"], int(agent.agent_id), float(agent.energy), float(agent.age),
                          float(getattr(agent, "max_age", 0.0)), float(agent.x), float(agent.y)))
        except Exception:
            pass
        env.real_kill(agent)

    env.real_kill = env.kill_agent
    env.kill_agent = _kill

    acc = {c: 0.0 for c in COLS if c != "t"}
    buckets = []
    events = {"births": [], "deaths": []}
    prev_ids = set()
    prev_pos = {}
    prev_energy = {}
    prev_score = env.score
    pre_biome_rate = {}
    disp_sum = {"n": 0.0, "d": 0.0}
    hive_snap = {}
    T = 0
    newborn_ids = set()
    for i in range(horizon):
        live = [a for a in env.agents]
        if not live:
            break
        states = [s for s in (env.get_agent_state(a.agent_id) for a in live) if s]
        if not states:
            break
        state["tick"] = i
        T = i + 1
        # pre-tick snapshot (for exact cost reconstruction)
        pre = {}
        for s in states:
            aid = s["agent_id"]
            a = env.agents_dict.get(aid)
            if a is None:
                continue
            ix = min(max(int(a.x), 0), env.width - 1)
            iy = min(max(int(a.y), 0), env.height - 1)
            pre[aid] = (float(a.energy), float(a.speed), float(a.sprint_speed), float(a.max_energy),
                        float(a.age), float(a.x), float(a.y), float(a.direction),
                        float(env.biome_map[ix, iy].energy_drain_rate), 0.0,
                        float(getattr(a, "max_age", 0.0)))
        acts_out = hive.decide({"agent_status": states, "sim_time": i * DT, "n_agents": len(states)})
        byid = {s["agent_id"]: s for s in states}
        acted = []
        req = {}
        for d in (acts_out or []):
            aid = d.get("agent_id")
            s = byid.get(aid)
            if s is None or aid not in pre:
                continue
            dist = float(d.get("move_distance", 0.0) or 0.0)
            dr = float(d.get("move_direction", 0.0) or 0.0)
            turn = float(d.get("turn_angle", 0.0) or 0.0)
            spawn = bool(d.get("spawn_agent", False))
            req[aid] = (dist, turn, spawn)
            acted.append((aid, ActionRequest(agent_id=aid, move_distance=dist, move_direction=dr,
                                             turn_angle=turn, spawn_agent=spawn)))
        # exact pre-charged costs
        c_move = c_turn = c_biome = c_old = c_spawn = 0.0
        for aid, (dist, turn, spawn) in req.items():
            e, sp, spr, mxE, age, x, y, _dir, bdrain, _unused, max_age = pre[aid]
            mc, _ = move_cost(dist, e, sp, spr, mxE)
            c_move += mc
            c_turn += min(np.pi, abs(turn)) / (2 * np.pi)
            c_biome += DT * bdrain
            if age > max_age:
                c_old += 0.01 * age
            if spawn and e > SPAWN_DRAIN:
                c_spawn += SPAWN_DRAIN
        E_pre = sum(p[0] for p in pre.values())
        pre_cost = {}
        for aid, (dist, turn, spawn) in req.items():
            e, sp, spr, mxE, age, x, y, _d, _b, _u, max_age = pre[aid]
            mc, _ = move_cost(dist, e, sp, spr, mxE)
            pre_cost[aid] = (mc + min(np.pi, abs(turn)) / (2 * np.pi)
                             + (SPAWN_DRAIN if (spawn and e > SPAWN_DRAIN) else 0.0))
        pre_fruit = {f.fruit_id: float(f.age) for f in env.fruits}
        pre_score = env.score
        core.step(acted)
        post_ids = {a.agent_id for a in env.agents}
        post_fruit = {f.fruit_id for f in env.fruits}
        # fruits that vanished while still fresh were EATEN; age > 100 is rot (environment.py)
        n_fruit_eaten = float(sum(1 for fid_, ag in pre_fruit.items()
                                  if fid_ not in post_fruit and ag <= 100.0))
        # ---- deaths this tick -------------------------------------------------------------
        eaten = starved_y = starved_o = 0
        eaten_energy = 0.0
        eaten_ids = set()
        starved_ids = []
        pred_pos = [(float(p.x), float(p.y), float(p.size)) for p in env.predators]
        new_kills = [k for k in kills if k[0] == i]
        kills[:] = []
        for (tk, aid, e_k, age_k, max_age_k, x_k, y_k) in new_kills:
            is_eaten = False
            for (px, py, psize) in pred_pos:
                if (px - x_k) ** 2 + (py - y_k) ** 2 < (psize + 5.0) ** 2:
                    is_eaten = True
                    break
            if is_eaten:
                eaten += 1
                eaten_energy += max(0.0, e_k)
                eaten_ids.add(aid)
                events["deaths"].append((i, aid, "eaten"))
            elif age_k > max_age_k:
                starved_o += 1
                starved_ids.append(aid)
                events["deaths"].append((i, aid, "starved_old"))
            else:
                starved_y += 1
                starved_ids.append(aid)
                events["deaths"].append((i, aid, "starved_young"))
        births = [a for a in env.agents if a.agent_id in post_ids - prev_ids]
        n_births = len(births)
        if n_births:
            events["births"].append((i, n_births))
        # ---- exact score algebra: fruit_in from the score delta and the exact corpse loss ----
        E_post = sum(float(a.energy) for a in env.agents)
        dscore = env.score - pre_score
        fruit_E = (dscore - DT + eaten_energy / 100.0) * 1000.0
        # perception + behaviour + genes
        see_f = 0.0
        fruit_seen = pred_seen = tree_seen = agent_seen = 0.0
        ef35 = ef20 = 0.0
        st = 0.0
        rd = 0.0
        sp_req = sp_acc = 0.0
        E = []
        maxEs = []
        ages = []
        spd = spr = hear = vis = 0.0
        xs = []
        ys = []
        for a in env.agents:
            ef = (a.energy / a.max_energy) if a.max_energy else 0.0
            E.append(float(a.energy))
            maxEs.append(float(a.max_energy))
            ages.append(float(a.age))
            spd += float(a.speed); spr += float(a.sprint_speed)
            hear += float(a.hearing_radius); vis += float(a.vision_radius)
            xs.append(float(a.x)); ys.append(float(a.y))
            if ef < 0.35:
                ef35 += 1.0
            if ef < 0.20:
                ef20 += 1.0
            r = req.get(a.agent_id)
            if r is not None:
                if r[0] <= 1e-9 and abs(r[1]) <= 1e-9:
                    st += 1.0
                rd += r[0]
                if r[2]:
                    sp_req += 1.0
                    if pre.get(a.agent_id, (0,))[0] > SPAWN_DRAIN:
                        sp_acc += 1.0
            obs = env.agent_observations.get(a.agent_id) or []
            nf = 0
            for e2 in obs:
                ty = e2.get("type")
                if ty == "Fruit":
                    nf += 1
                elif ty == "Predator":
                    pred_seen += 1.0
                elif ty == "Tree":
                    tree_seen += 1.0
                elif ty == "Agent":
                    agent_seen += 1.0
            fruit_seen += nf
            if nf > 0:
                see_f += 1.0
        n_pop = len(env.agents)
        n = max(1, n_pop)
        prev_ids = post_ids
        # ---- bucket accumulator (rate-like metrics are per-agent means or fractions) --------
        n_act = max(1, len(req))
        disp_tick = 0.0
        n_disp = 0
        for a in env.agents:
            p0 = prev_pos.get(a.agent_id)
            if p0 is not None:
                disp_tick += float(np.hypot(a.x - p0[0], a.y - p0[1]))
                n_disp += 1
        acc["pop"] += n_pop
        acc["active"] += float(len(req))
        acc["births"] += n_births
        acc["d_sy"] += starved_y
        acc["d_so"] += starved_o
        acc["d_eat"] += eaten
        acc["fruit_n"] += n_fruit_eaten
        acc["fruit_E"] += fruit_E
        acc["c_move"] += c_move
        acc["c_turn"] += c_turn
        acc["c_biome"] += c_biome
        acc["c_old"] += c_old
        acc["c_spawn"] += c_spawn
        acc["e_in_eaten"] += eaten_energy
        acc["E_mean"] += (sum(E) / n if E else 0.0)
        acc["E_med"] += (float(np.median(E)) if E else 0.0)
        acc["E_max"] += (max(E) if E else 0.0)
        acc["E_min"] += (min(E) if E else 0.0)
        acc["E_sd"] += (float(np.std(E)) if E else 0.0)
        acc["ef_lt35"] += ef35 / n
        acc["ef_lt20"] += ef20 / n
        acc["see_fruit_1"] += see_f / n
        acc["fruit_seen"] += fruit_seen / n
        acc["pred_seen"] += pred_seen / n
        acc["tree_seen"] += tree_seen / n
        acc["agent_seen"] += agent_seen / n
        acc["stat_frac"] += st / n_act
        acc["req_d"] += rd / n_act
        acc["disp"] += disp_tick / max(1, n_disp)
        acc["spawn_req"] += sp_req / n_act
        acc["spawn_acc"] += sp_acc / n_act
        acc["g_speed"] += spd / n
        acc["g_sprint"] += spr / n
        acc["g_hear"] += hear / n
        acc["g_vis"] += vis / n
        acc["g_maxE"] += (sum(maxEs) / n if maxEs else 0.0)
        acc["g_maxE_sd"] += (float(np.std(maxEs)) if maxEs else 0.0)
        acc["g_age"] += (sum(ages) / n if ages else 0.0)
        acc["sd_x"] += (float(np.std(xs)) if xs else 0.0)
        acc["sd_y"] += (float(np.std(ys)) if ys else 0.0)
        # ---- closure ------------------------------------------------------------------------
        # an agent that starves away takes its remaining (post-movement) energy out of the system;
        # an eaten agent's energy goes to the predator and is already counted in eaten_energy
        starved_pre_E = 0.0
        for aid in starved_ids:
            pc = pre_cost.get(aid, 0.0)
            starved_pre_E += max(0.0, pre.get(aid, (0.0,))[0] - pc)
        exp_post = (E_pre + fruit_E + NEWBORN_ENERGY * n_births - c_move - c_turn - c_biome
                    - c_old - c_spawn - eaten_energy - starved_pre_E)
        acc["_closure"] = acc.get("_closure", 0.0) + (E_post - exp_post)
        acc["_n"] = acc.get("_n", 0.0) + 1.0
        # close the bucket
        if (i + 1) % bucket == 0 or i == horizon - 1:
            nb = acc.pop("_n", 0.0) or 1.0
            row = {"closure": acc.get("_closure", 0.0) / nb}
            for c in COLS:
                if c == "t":
                    row[c] = (i + 1) / 10.0
                elif c in ("pop", "E_mean", "E_med", "E_max", "E_min", "E_sd", "ef_lt35", "ef_lt20",
                           "see_fruit_1", "fruit_seen", "pred_seen", "tree_seen", "agent_seen",
                           "stat_frac", "req_d", "disp", "g_speed", "g_sprint", "g_hear", "g_vis",
                           "g_maxE", "g_maxE_sd", "g_age", "sd_x", "sd_y"):
                    row[c] = acc.get(c, 0.0) / nb
                else:
                    row[c] = acc.get(c, 0.0)
            buckets.append(row)
            for c in COLS:
                if c != "t":
                    acc[c] = 0.0
            acc["_closure"] = 0.0
            acc["_n"] = 0.0
        prev_pos = {a.agent_id: (float(a.x), float(a.y)) for a in env.agents}
        if (i + 1) % 1000 == 0 and hasattr(hive, "stats"):
            hive_snap[int((i + 1) / 10)] = {k: v for k, v in hive.stats.items() if not isinstance(v, dict)}
    final = {"seed": seed, "T": T, "score": float(env.score), "n_final": len(env.agents),
             "fruit_E_total": float(sum(b["fruit_E"] for b in buckets)),
             "c_move": float(sum(b["c_move"] for b in buckets)),
             "c_turn": float(sum(b["c_turn"] for b in buckets)),
             "c_biome": float(sum(b["c_biome"] for b in buckets)),
             "c_old": float(sum(b["c_old"] for b in buckets)),
             "c_spawn": float(sum(b["c_spawn"] for b in buckets)),
             "eaten_E": float(sum(b["e_in_eaten"] for b in buckets)),
             "births": int(sum(b["births"] for b in buckets)),
             "d_sy": int(sum(b["d_sy"] for b in buckets)),
             "d_so": int(sum(b["d_so"] for b in buckets)),
             "d_eat": int(sum(b["d_eat"] for b in buckets)),
             "hive_stats": (hive.stats if hasattr(hive, "stats") else {}),
             "hive_snap": hive_snap,
             "buckets": buckets}
    return final


def _run(job):
    return run_seed(job[0], job[1], job[2], job[3])


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=30000)   # official cap: env.time <= 3000 s
    ap.add_argument("--bucket", type=int, default=100)       # 100 ticks = 10 sim-seconds
    ap.add_argument("--out", required=True)
    ap.add_argument("--hive-seed", type=int, default=0,
                    help="hive's own RNG seed; the deployed server.py uses the default 0")
    ap.add_argument("--chunk", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw.jsonl")
    done = set()
    if os.path.exists(raw):
        with open(raw) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    pass
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    print("[traj_map] %d seeds to run (%d already done), workers=%d horizon=%d"
          % (len(seeds), len(done), args.workers, args.horizon), flush=True)
    if not seeds:
        return
    t0 = time.time()
    ctx = get_context("spawn")
    jobs = [(s, args.horizon, args.bucket, args.hive_seed) for s in seeds]
    with open(raw, "a") as f, ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_run, jobs), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            el = time.time() - t0
            print("[traj_map] %d/%d seed=%d T=%d score=%.1f  (%.1f s elapsed, %.1f s/seed)"
                  % (k, len(seeds), res["seed"], res["T"], res["score"], el, el / k), flush=True)
    print("[traj_map] DONE in %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
