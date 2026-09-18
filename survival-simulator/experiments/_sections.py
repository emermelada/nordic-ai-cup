"""Section-level CPU breakdown of shipped best_controller.potential_controller.

The shipped function body is reproduced verbatim with a perf_counter around each section, then run
over a cached corpus of real per-agent states (collected once and pickled) so the shares are per
policy CALL, independent of sim noise. Usage: python _sections.py [seed] [horizon]
"""
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import best_controller as bc
import best_controller_fast as bcf
from env_wrapper import run_eval_episode

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100
H = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
CACHE = os.path.join(HERE, "_states_cache_%d_%d.pkl" % (SEED, H))


def get_states():
    if not os.path.exists(CACHE):
        states = []
        run_eval_episode(bc.make_policy(bc._load_params()), n_agents=5, seed=SEED, horizon=H,
                         stop_on_death=True,
                         recorder=lambda i, s, a, o: states.extend(s), reset_fn=bc.reset_memory)
        with open(CACHE, "wb") as f:
            pickle.dump(states, f)
    with open(CACHE, "rb") as f:
        return pickle.load(f)


SEC = {}

_SECTIONS = ["bookkeeping", "obs_filter", "pred_repel", "pred_facing", "fruit_loop",
             "tree_loop", "disperse_loop", "edge_loop", "combine_hyst", "dist_repro"]


def instrumented(state, P):
    """Shipped body, section-timed."""
    global _EPOCH
    import math
    t = time.perf_counter()
    _EPOCH = bc._EPOCH + 1
    bc._EPOCH = _EPOCH
    o = state.get("observations") or []
    energy = state.get("energy", 0.0)
    max_e = max(state.get("max_energy", 1.0) or 1.0, 1.0)
    ef = energy / max_e
    speed = state.get("speed", 10.0)
    sprint = state.get("sprint_speed", 20.0)
    aid = state.get("agent_id", 0)
    bc._maybe_new_episode(aid, state.get("age", 0.0))
    bc._GC[aid] = _EPOCH
    m = bc._mem(aid)
    m["clock"] += 1
    gpop = bc._global_alive()
    SEC["bookkeeping"] += time.perf_counter() - t
    t = time.perf_counter()
    fruits = [x for x in o if x.get("type") == "Fruit"]
    preds = [x for x in o if x.get("type") == "Predator"]
    agents = [x for x in o if x.get("type") == "Agent"]
    edges = [x for x in o if x.get("type") == "Edge"]
    trees = [x for x in o if x.get("type") == "Tree"]
    SEC["obs_filter"] += time.perf_counter() - t
    t = time.perf_counter()
    if m["spawn_clock"] > 0:
        m["spawn_clock"] -= 1
    px = py = 0.0
    threat_dist = float("inf")
    threat_dir = None
    use_pred = P.get("use_predator", True)
    for p in preds:
        if not use_pred:
            break
        d = p["distance"]
        if d > P["danger_dist"]:
            continue
        if d < threat_dist:
            threat_dist, threat_dir = d, p["angle"]
        a = p["angle"]
        fx = -math.cos(a)
        fy = -math.sin(a)
        weight = P["predator_weight"] / (max(d, 5.0) ** 2)
        px += fx * weight
        py += fy * weight
    SEC["pred_repel"] += time.perf_counter() - t
    t = time.perf_counter()
    facing_threat = False
    for p in preds:
        if not use_pred:
            break
        rdir = math.atan2(math.sin(p.get("rel_dir", 0)), math.cos(p.get("rel_dir", 0)))
        if abs(rdir) < P["concern_cone"] and p["distance"] < P["danger_dist"]:
            facing_threat = True
            break
    in_escape = m["flee"]
    if use_pred and threat_dist < P["flee_dist"]:
        in_escape = True
    elif threat_dist > P["escape_dist"]:
        in_escape = False
    flee_engage = (P.get("use_flee", True)) and (in_escape or (facing_threat and threat_dist < P["danger_dist"] * 0.7))
    SEC["pred_facing"] += time.perf_counter() - t
    t = time.perf_counter()
    if flee_engage:
        mag = math.hypot(px, py)
        if mag > 1e-6:
            away = math.atan2(py, px)
        elif threat_dir is not None:
            away = math.atan2(-math.sin(threat_dir), -math.cos(threat_dir))
        else:
            away = 0.0
        m["flee"] = True
        if ef > P["reserve_frac"]:
            dist = sprint * P["flee_speed_frac"]
        else:
            dist = speed * 0.9
        SEC["dist_repro"] += time.perf_counter() - t
        return [float(dist), float(away), 0.0, 0.0]
    m["flee"] = False
    fx = fy = 0.0
    best_fruit = None
    use_fruit = P.get("use_fruit", True)
    for f in fruits:
        if not use_fruit:
            break
        d = f["distance"]
        a = f["angle"]
        risk = 0.0
        for p in preds:
            if p["distance"] < P["danger_dist"] * 1.4:
                angdiff = abs(bc._wrap(p["angle"] - a))
                if angdiff < 1.0:
                    risk += (1.0 - angdiff) * P["fruit_risk_penalty"] * (1 - p["distance"] / (P["danger_dist"] * 1.4))
        w = 1.0 / (d + 8.0) * max(0.0, 1.0 - risk)
        fx += math.cos(a) * w
        fy += math.sin(a) * w
        if best_fruit is None or d < best_fruit["distance"]:
            best_fruit = f
    SEC["fruit_loop"] += time.perf_counter() - t
    t = time.perf_counter()
    for tt in trees:
        if not P.get("use_fruit", True):
            break
        d = tt["distance"]
        a = tt["angle"]
        w = 0.25 / (d + 20.0)
        fx += math.cos(a) * w
        fy += math.sin(a) * w
    SEC["tree_loop"] += time.perf_counter() - t
    t = time.perf_counter()
    for ag in agents:
        if not P.get("use_disperse", True):
            break
        d = ag["distance"]
        if d < 60.0 and d > 1.0:
            a = ag["angle"]
            w = P["disperse_weight"] / (d + 5.0)
            fx -= math.cos(a) * w
            fy -= math.sin(a) * w
    SEC["disperse_loop"] += time.perf_counter() - t
    t = time.perf_counter()
    if edges and P.get("use_wall", True):
        pts = [e["coords"][0] for e in edges] + [e["coords"][1] for e in edges]
        pxW = pyW = 0.0
        for (ex, ey) in pts:
            d = math.hypot(ex, ey)
            if d < 90.0 and d > 1.0:
                w = P["wall_weight"] / (d ** 2)
                pxW -= ex / d * w
                pyW -= ey / d * w
        fx += pxW
        fy += pyW
    SEC["edge_loop"] += time.perf_counter() - t
    t = time.perf_counter()
    rw_ang = m.get("wander_ang")
    if rw_ang is None or m["clock"] % 80 == 0:
        rw_ang = bc._det_rand(aid * 1000003 + m["clock"]) * 2 * math.pi
    m["wander_ang"] = rw_ang
    if P.get("use_explore", True):
        fx += math.cos(rw_ang) * P["wander_weight"]
        fy += math.sin(rw_ang) * P["wander_weight"]
    mag = math.hypot(fx, fy)
    if mag < 1e-6:
        steer = bc._wrap(rw_ang)
    else:
        steer = math.atan2(fy, fx)
    direct_forage = False
    if P.get("forage_nearest", 0.0) > 0.0 and fruits:
        tf = min(fruits, key=lambda f: f["distance"])
        steer = bc._wrap(tf["angle"])
        direct_forage = True
    if not direct_forage and (best_fruit is None or best_fruit["distance"] > 25.0):
        diff = bc._wrap(steer - m["last_steer"])
        steer = m["last_steer"] + diff * P["target_hyst"]
    m["last_steer"] = bc._wrap(steer)
    SEC["combine_hyst"] += time.perf_counter() - t
    t = time.perf_counter()
    use_energy = P.get("use_energy", True)
    if direct_forage:
        dist = speed * float(P.get("forage_speed", 1.0))
    elif best_fruit is not None and best_fruit["distance"] < 40.0:
        dist = min(speed, best_fruit["distance"])
    elif use_energy and ef < P["low_energy_frac"]:
        dist = speed * 0.4
    else:
        dist = speed * P["walk_frac"] if (best_fruit is not None) else speed * P["explore_frac"]
    if use_energy and ef < P["reserve_frac"]:
        dist = min(dist, speed * 0.25)
    if P.get("role_by_age", 0.0) > 0.0:
        _age = state.get("age", 0.0)
        if _age > P.get("old_age", 60.0):
            dist *= P.get("old_speed_frac", 0.5)
        elif _age < P.get("young_age", 25.0):
            dist *= P.get("young_speed_frac", 1.0)
    spawn = 0.0
    use_repro = P.get("use_repro", True)
    target = P.get("repro_global_target", 10)
    rf = P.get("repro_frac", bc.DEFAULT_PARAMS["repro_frac"])
    urgency = P.get("repro_urgency", 1.0)
    if urgency > 0.0 and target > 0:
        deficit = max(0.0, min(1.0, (target - gpop) / target))
        rf = rf - (rf - P.get("repro_frac_min", 0.24)) * deficit * urgency
    if P.get("repro_unlimited", 0.0) > 0.0:
        pop_ok = True
        crowd_ok = True
    else:
        pop_ok = gpop < target
        crowd_ok = len(agents) <= P.get("repro_popcap", 2)
    relay_age = P.get("relay_age", 0.0)
    relay = relay_age > 0.0 and state.get("age", 0.0) > relay_age
    if relay:
        pop_ok = True
        rf = min(rf, P.get("relay_energy_frac", 0.35))
        cd = P.get("relay_cooldown", 700)
    else:
        cd = P.get("spawn_cooldown", 400)
    abs_gate = P.get("repro_energy_abs", 0.0)
    gate_ok = (energy > abs_gate) if abs_gate > 0.0 else (ef > rf)
    if (use_repro and gate_ok and m["spawn_clock"] <= 0 and pop_ok and crowd_ok):
        safe = all(p["distance"] >= P.get("repro_safe_radius", 330.0) for p in preds)
        if safe:
            spawn = 1.0
            m["spawn_clock"] = cd
    SEC["dist_repro"] += time.perf_counter() - t
    return [float(dist), float(steer), 0.0, float(spawn)]


states = get_states()
P = bc._load_params()
print("corpus: %d policy calls (seed=%d, horizon=%d)" % (len(states), SEED, H))

bc.reset_memory()
for k in _SECTIONS:
    SEC[k] = 0.0
for s in states[:200]:
    instrumented(s, P)
for k in _SECTIONS:
    SEC[k] = 0.0
bc.reset_memory()
t0 = time.perf_counter()
for s in states:
    instrumented(s, P)
total = time.perf_counter() - t0

# ground truth totals for the same corpus
def timeit(module):
    fn = module.make_policy(module._load_params())
    best = 1e9
    for _ in range(3):
        module.reset_memory()
        t = time.perf_counter()
        for s in states:
            fn(s)
        best = min(best, time.perf_counter() - t)
    return 1000.0 * best / len(states)


ship = timeit(bc)
fast = timeit(bcf)

print("\n%-16s %10s %8s" % ("section", "us/call", "% of total"))
for k in _SECTIONS:
    print("%-16s %10.3f %7.1f%%" % (k, 1e6 * SEC[k] / len(states), 100 * SEC[k] / total))
print("%-16s %10.3f %7.1f%%" % ("(perf_counter tax)", 1e6 * (total - sum(SEC.values())) / len(states),
                               100 * (total - sum(SEC.values())) / total))
print("%-16s %10.3f" % ("instrumented tot", 1e6 * total / len(states)))
print("\nuninstrumented best-of-3: shipped=%.4f ms/call (%.2f us) | fast=%.4f ms/call (%.2f us) -> %.2fx (%.1f%% less)"
      % (ship, 1000 * ship, fast, 1000 * fast, ship / fast, 100 * (1 - fast / ship)))
