"""Potential-field / energy-aware steering controller (directive priorities 1-4).

contract: fn(agent_state_dict) -> [move_dist, move_dir(rel), turn_angle, spawn_flag]

Design (measured against source sim money-line mechanics):
  - score += dt is constant; the ONLY learnable margin = fruit(+~0.06) minus predation (-energy/100,
    big). So predator avoidance dominates; foraging sustains energy; reproduction keeps species alive.
  - Steering is a geometric potential field built directly from observation {distance, angle} vectors:
      desired_vec = w_f * sum(fruit attract) - w_p * sum(predator inverse-square repel)
                    + w_w * sum(wall repel) + w_a * sum(other-agent disperse) + w_x * wander
      steer = atan2(vec)
  - Flee override with hysteresis: if closest predator < flee_dist (or inside danger cone with
    rel_dir facing us), commit to FLEE (sprint straight away from the aggregate predator vector)
    until the predator is far enough away -> no tick-to-tick oscillation.
  - Move distance: sprint only when fleeing and energy allows; otherwise walk (cheap 0.05/unit).
  - Energy economics: below reserve_frac stop sprinting; below low_energy_frac prefer the nearest
    safe fruit and cut walking; never let projected living+move cost push energy to 0.
  - Reproduction: only when energy > repro_frac*max AND no predator within big radius AND not already
    crowded locally AND spawn cooldown elapsed -> treat spawn as an investment, not a habit.
All parameters are tunable (dict) so the policy is optimizeable (priority 8).
"""
import math, random
import numpy as np

# -----------------------------------------------------------------------------
# default parameterization (hand-tuned pre-search; refined by evolution)
# -----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # steering
    "fruit_weight": 1.0,       # attraction toward fruit
    "predator_weight": 3.0,    # inverse-square repulsion scale
    "wall_weight": 1.6,        # repulsion from visible edge points
    "disperse_weight": 0.6,    # repulsion from other agents (avoid clustering)
    "wander_weight": 0.12,     # exploration noise when food is scarce
    "target_hyst": 0.5,        # fruit target hysteresis (angle stickiness, radians)
    # predator
    "danger_dist": 180.0,      # repulsion begins this far out
    "flee_dist": 120.0,        # below this distance -> commit to sprint flee
    "concern_cone": 2.2,       # rad tolerance to treat FRO (facing-ish) predator as threat early
    "flee_speed_frac": 1.0,    # fraction of sprint_speed while fleeing
    "escape_dist": 200.0,      # hysteresis: stop fleeing only once predator beyond this
    # forage
    "fruit_risk_penalty": 1.2, # fraction weighting: avoid fruit in predator direction
    "walk_frac": 1.0,          # walking move distance = walk_frac*speed (<=1 => walking)
    "explore_frac": 0.55,      # wandering speed fraction
    # energy
    "reserve_frac": 0.15,      # below this single-agent fraction of max: no sprint, minimal walk
    "low_energy_frac": 0.35,   # below this: prioritize nearest safe fruit, cut speed 40%
    "boom_frac": 0.28,         # maximum sustainable walking+living energy burn per tick (not used directly)
    # reproduction
    "repro_frac": 0.82,        # energy fraction required to spawn
    "repro_safe_radius": 330.0,# no predator within this radius to allow spawn
    "repro_popcap": 2,         # max OTHER agents observed before refusing to spawn (dispersion)
    "repro_global_target": 10, # cooperative: don't spawn while estimated global population >= this
    "spawn_cooldown": 400,     # ticks between spawns (anti-overpopulation)
    "spawn_cap": 14,           # hard cap: never spawn while local(other) agents >= this
    # --- population-maintaining reproduction (2026-09-17 survival fix) ---
    "repro_frac_min": 0.24,    # floor for the energy gate when pop is far below target (sim needs >100)
    "repro_urgency": 1.0,      # 0 = flat repro_frac, 1 = scale gate down to repro_frac_min as pop->0
    "repro_unlimited": 0.0,    # 1 = ignore gpop/popcap gate entirely (rely on energy + cooldown only)
    # --- foraging efficiency (energy income; movement burn dominates the energy budget) ---
    "forage_nearest": 0.0,     # 1 = when fruit is visible, steer straight at the nearest safe fruit
    "forage_speed": 1.0,       # move-distance fraction used by the nearest-fruit override
    # --- generational relay (age-aware: survive past max_age by banking heirs) ---
    "relay_age": 0.0,          # sim seconds; spawn an heir once older than this (0 = off)
    "relay_energy_frac": 0.35, # energy gate used by the relay spawn (fraction of max_energy)
    "relay_cooldown": 700,     # ticks between relay spawns
    "repro_energy_abs": 0.0,   # ABSOLUTE energy gate (sim requires >100); 0 = use fractional rf
    # --- age-based roles, fully decentralised (each agent only knows its OWN age) ---
    "role_by_age": 0.0,        # 1 = young agents forage hard, old agents conserve
    "young_age": 25.0,         # sim seconds below which an agent is a "young forager"
    "old_age": 60.0,           # sim seconds above which an agent is "old" (conserve + bank heir)
    "young_speed_frac": 1.0,   # movement scale for young agents
    "old_speed_frac": 0.5,     # movement scale for old agents (they are a dying investment)
}

# module-level memory: per-agent-id last action / flee state / spawn clock.
# Persists across ticks inside one episode (fresh process), keyed by agent_id.
_MEM = {}
_EPOCH = 0
_GC = {}  # agent_id -> last-seen epoch (pruned via TTL -> cheap cooperative population estimate)
_AGE = {}  # agent_id -> last age (monotonic within an episode; a drop => new episode => reset memory)


def _maybe_new_episode(aid, age):
    """Age is monotonic within one episode, so an age decrease means the world reset.
    Purge all cross-agent memory on reset so consecutive episodes are independent."""
    global _EPOCH
    prev = _AGE.get(aid)
    if prev is not None and age < prev - 10.0:
        _MEM.clear()
        _GC.clear()
        _AGE.clear()
        _EPOCH = 0
    _AGE[aid] = age


def reset_memory():
    """Hard reset of ALL module-level policy state. Call at the START of every episode.

    Without this, evaluation is ORDER-DEPENDENT: `_MEM`/`_GC`/`_AGE`/`_EPOCH` are module globals,
    so an episode inherits the previous episode's per-agent memory and population estimate. The
    age-drop heuristic in `_maybe_new_episode` misses cases, which made the SAME seed produce
    different survival (e.g. 9362 vs 7581 vs 11873 ticks) depending on which seeds ran before it —
    silently biasing every multi-config sweep by config position."""
    global _EPOCH
    _MEM.clear()
    _GC.clear()
    _AGE.clear()
    _EPOCH = 0


def _global_alive(ttl=60):
    dead = [aid for aid, ep in _GC.items() if _EPOCH - ep > ttl]
    for aid in dead:
        _GC.pop(aid, None)
    return len(_GC)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _det_rand(i):
    """Deterministic pseudo-random in [0,1) from an integer (xorshift-ish).
    Replaces random.uniform for exploration: an unseeded RNG made the policy's search path
    non-reproducible run-to-run (same sim seed -> different collapse tick, +/-1.5k ticks)."""
    x = (int(i) * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 29
    return (x & 0xFFFFFFFF) / float(0x100000000)


def _mem(agent_id):
    return _MEM.setdefault(agent_id, {"flee": False, "spawn_clock": 0, "last_steer": 0.0, "clock": 0})


def potential_controller(state, P):
    global _EPOCH
    _EPOCH += 1
    o = state.get("observations") or []
    energy = state.get("energy", 0.0)
    max_e = max(state.get("max_energy", 1.0) or 1.0, 1.0)
    ef = energy / max_e
    speed = state.get("speed", 10.0)
    sprint = state.get("sprint_speed", 20.0)
    aid = state.get("agent_id", 0)
    _maybe_new_episode(aid, state.get("age", 0.0))
    _GC[aid] = _EPOCH
    m = _mem(aid)
    m["clock"] += 1
    gpop = _global_alive()

    fruits = [x for x in o if x.get("type") == "Fruit"]
    preds = [x for x in o if x.get("type") == "Predator"]
    agents = [x for x in o if x.get("type") == "Agent"]
    edges = [x for x in o if x.get("type") == "Edge"]

    # --- descend spawn clock ---
    if m["spawn_clock"] > 0:
        m["spawn_clock"] -= 1

    # ---- predator repulsion field (local frame vectors) ----
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

    # flee override with hysteresis
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

    if flee_engage:
        # flee directly away from aggregate predator vector; if pv~0 use closest threat
        mag = math.hypot(px, py)
        if mag > 1e-6:
            away = math.atan2(py, px)
        elif threat_dir is not None:
            away = math.atan2(-math.sin(threat_dir), -math.cos(threat_dir))
        else:
            away = 0.0
        m["flee"] = True
        # sprint only if we can afford it
        if ef > P["reserve_frac"]:
            dist = sprint * P["flee_speed_frac"]
        else:
            dist = speed * 0.9
        return [float(dist), float(away), 0.0, 0.0]

    m["flee"] = False

    # ---- fruit attraction (weighted by safety) ----
    fx = fy = 0.0
    best_fruit = None
    use_fruit = P.get("use_fruit", True)
    for f in fruits:
        if not use_fruit:
            break
        d = f["distance"]
        a = f["angle"]
        # risk: fruits in same direction as a close predator are less attractive
        risk = 0.0
        for p in preds:
            if p["distance"] < P["danger_dist"] * 1.4:
                angdiff = abs(_wrap(p["angle"] - a))
                if angdiff < 1.0:
                    risk += (1.0 - angdiff) * P["fruit_risk_penalty"] * (1 - p["distance"] / (P["danger_dist"] * 1.4))
        # weight: inverse distance, discounted by risk
        w = 1.0 / (d + 8.0) * max(0.0, 1.0 - risk)
        fx += math.cos(a) * w
        fy += math.sin(a) * w
        if best_fruit is None or d < best_fruit["distance"]:
            best_fruit = f

    # ---- tree attraction as weaker fallback (trees host fruit) ----
    for t in [x for x in o if x.get("type") == "Tree"]:
        if not P.get("use_fruit", True):
            break
        d = t["distance"]
        a = t["angle"]
        w = 0.25 / (d + 20.0)
        fx += math.cos(a) * w
        fy += math.sin(a) * w

    # ---- other-agent dispersion (avoid clustering -> predator multi-kills) ----
    for ag in agents:
        if not P.get("use_disperse", True):
            break
        d = ag["distance"]
        if d < 60.0 and d > 1.0:
            a = ag["angle"]
            w = P["disperse_weight"] / (d + 5.0)
            fx -= math.cos(a) * w
            fy -= math.sin(a) * w

    # ---- wall/edge repulsion ----
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

    # ---- wander ---
    rw_ang = m.get("wander_ang")
    if rw_ang is None or m["clock"] % 80 == 0:
        rw_ang = _det_rand(aid * 1000003 + m["clock"]) * 2 * math.pi
    m["wander_ang"] = rw_ang
    if P.get("use_explore", True):
        fx += math.cos(rw_ang) * P["wander_weight"]
        fy += math.sin(rw_ang) * P["wander_weight"]

    # ---- combine; hysteresis on steering angle to reduce oscillation ----
    mag = math.hypot(fx, fy)
    if mag < 1e-6:
        steer = _wrap(rw_ang)
    else:
        steer = math.atan2(fy, fx)

    # ---- direct-forage override: when fruit IS visible, spend the minimum travel distance to reach
    # it (movement costs 0.05/unit, the dominant energy sink). A weighted vector-sum of many fruits
    # can cancel out and produce a heading aimed at NO fruit; going straight at the nearest one
    # maximises energy income per distance travelled.
    direct_forage = False
    if P.get("forage_nearest", 0.0) > 0.0 and fruits:
        tf = min(fruits, key=lambda f: f["distance"])
        steer = _wrap(tf["angle"])
        direct_forage = True

    # sticky: don't flip heading wildly tick-to-tick unless food is very close
    if not direct_forage and (best_fruit is None or best_fruit["distance"] > 25.0):
        diff = _wrap(steer - m["last_steer"])
        steer = m["last_steer"] + diff * P["target_hyst"]
    m["last_steer"] = _wrap(steer)

    # ---- move distance (energy-aware) ----
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

    # ---- age-based role scaling (audit C2): every agent is called separately and knows only its
    # own age, so roles need no coordination at all. The young are the future -> forage hard; the
    # old are a dying investment (max_age 60-120 s) -> stop burning energy on movement and put it
    # into banking an heir instead.
    if P.get("role_by_age", 0.0) > 0.0:
        _age = state.get("age", 0.0)
        if _age > P.get("old_age", 60.0):
            dist *= P.get("old_speed_frac", 0.5)
        elif _age < P.get("young_age", 25.0):
            dist *= P.get("young_speed_frac", 1.0)

    # ---- reproduction (investment-gated; cooperative pop cap via shared estimate) ----
    spawn = 0.0
    use_repro = P.get("use_repro", True)
    target = P.get("repro_global_target", 10)
    # Population-maintaining gate: the energy bar drops toward repro_frac_min as the alive count
    # falls below target, so a depleted team spawns as soon as it can afford it (sim hard-requires
    # energy > 100 at the moment of spawning, so repro_frac_min is kept safely above that).
    rf = P.get("repro_frac", DEFAULT_PARAMS["repro_frac"])
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

    # --- generational relay (audit B1/B2, C2): never let the chain break ---
    # agent.py: max_age = 60 + U(0,60) sim-seconds and environment.py applies energy -= 0.01*age
    # per tick past it, so EVERY agent dies of age in 60-120 s no matter how well fed. Survival
    # past ~120 s therefore requires an unbroken line of heirs. An already-old agent is a dying
    # investment and must bank a successor NOW, even at/above the population target: population
    # size is NOT a score multiplier (score += dt fires once per tick regardless of agent count);
    # the metric is the time until the LAST agent dies.
    relay_age = P.get("relay_age", 0.0)  # sim seconds; 0 = disabled
    relay = relay_age > 0.0 and state.get("age", 0.0) > relay_age
    if relay:
        pop_ok = True  # population COUNT is not a score multiplier (score += dt once per tick)
        # ...but keep the LOCAL dispersion gate: a crowd is what gets multi-killed, and each extra
        # body is both a 100-energy cost and an edible -energy/100 liability. A relay needs ONE
        # competent heir, not a swarm.
        rf = min(rf, P.get("relay_energy_frac", 0.35))
        cd = P.get("relay_cooldown", 700)     # slower cadence for relay spawns than the normal gate
    else:
        cd = P.get("spawn_cooldown", 400)
    # --- reproduction gate, in the RIGHT unit ---
    # environment.py hard-requires energy > 100 to spawn. Our gate was a FRACTION of max_energy
    # (0.35 * 500 = 175), which is both the wrong unit (max_energy varies per agent) and
    # unreachable exactly when food decays and the relay matters most. Measured: ZERO births in
    # the final 2,000 ticks of a losing run, with mean energy 37-120 and fruit_vis 0-1.
    # Minimum viable reproduction: spend down to just above the sim's own requirement.
    abs_gate = P.get("repro_energy_abs", 0.0)
    gate_ok = (energy > abs_gate) if abs_gate > 0.0 else (ef > rf)
    if (use_repro and gate_ok
            and m["spawn_clock"] <= 0
            and pop_ok
            and crowd_ok):
        safe = all(p["distance"] >= P.get("repro_safe_radius", 330.0) for p in preds)
        if safe:
            spawn = 1.0
            m["spawn_clock"] = cd

    return [float(dist), float(steer), 0.0, float(spawn)]


def _load_params():
    import json, os
    P = dict(DEFAULT_PARAMS)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_controller", "params.json")
    try:
        with open(p) as f:
            P.update(json.load(f))  # partial file: merge over defaults so no KeyError
    except Exception:
        pass
    return P


# standalone importable policy function (contract) using saved/evolved params
def best_controller(state):
    return potential_controller(state, _load_params())


def make_policy(params):
    def fn(state):
        return potential_controller(state, params)
    return fn