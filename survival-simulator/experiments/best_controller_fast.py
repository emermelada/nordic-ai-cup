"""CPU-lean, BEHAVIOUR-IDENTICAL twin of experiments/best_controller.py.

Contract: fn(state_dict) -> [move_distance, move_direction(rel), turn_angle, spawn_flag]
Same params (best_controller/params.json), same module state, same arithmetic.

Why it exists
-------------
The shipped controller is ~2.5% of local ms/tick (the simulator core dominates the rest), so this
file targets the *per-request* CPU cost, which is what a grader aborting on accumulated response
wait actually pays for. Measured on this Mac (see experiments/best_controller_fast_report.md):
pure policy call 9.6 us -> ~2.6 us; served entry point `best_controller(state)` 25.0 us -> ~2.7 us.

Identical-behaviour argument (every change is a strict re-association-free rewrite)
----------------------------------------------------------------------------------
1. ONE pass over `observations` fills five buckets (Fruit/Predator/Agent/Tree/Edge) instead of four
   list comprehensions + a fifth inline comprehension re-scanning the list for trees. Relative
   order inside each bucket is preserved, so every downstream loop sees the exact same sequence.
2. The wall/edge pass no longer materialises `pts = [starts] + [ends]` (2 extra list allocations
   and 2 dict lookups per edge). It keeps the ORIGINAL two-part summation order (all starts, then
   all ends) because float addition is not associative. Points whose |x| or |y| already exceeds
   the 90-unit cutoff can never satisfy `d < 90` (hypot(x,y) >= max(|x|,|y|)), so they are skipped
   before `math.hypot`; every surviving point still computes `d = math.hypot(ex, ey)` and is tested
   with the ORIGINAL `d < 90.0 and d > 1.0` predicate. Exactly the same `d`, exactly the same sum.
3. Fruit-risk no longer recomputes `1 - pred_dist/(danger*1.4)` and re-wraps the predator bearing
   for every (fruit, predator) pair; those predator-only terms are hoisted into one list, in the
   same order, before the fruit loop. The multiplication order `(1-angdiff)*penalty*scale` is kept
   verbatim, so the accumulated `risk` is bit-identical.
4. Pure-read hoists (`P["danger_dist"]`, `P.get("use_*")`, `math.cos` ...) out of inner loops; a
   dict read is pure, so hoisting cannot change a value.
5. `_mem` avoids rebuilding a dict literal on every call; `_wrap`/`_det_rand` keep their exact
   formulas (inlined at the hot call sites, same arithmetic).
6. `_load_params()` is memoised on (params.json mtime, size). It returns the same *values* as the
   uncached loader; only the per-call file open/parse disappears. Same DEFAULT_PARAMS fallback.

Nothing else changed: the flee override/hysteresis, spawn clocks, `spawn_clock`/`last_steer`
bookkeeping, global-population estimate, repro gates and return tuple are the shipped logic.
"""
import json
import math
import os

# -----------------------------------------------------------------------------
# default parameterization — byte-for-byte the shipped table
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
    "fruit_risk_penalty": 1.2,  # fraction weighting: avoid fruit in predator direction
    "walk_frac": 1.0,          # walking move distance = walk_frac*speed (<=1 => walking)
    "explore_frac": 0.55,      # wandering speed fraction
    # energy
    "reserve_frac": 0.15,      # below this single-agent fraction of max: no sprint, minimal walk
    "low_energy_frac": 0.35,   # below this: prioritize nearest safe fruit, cut speed 40%
    "boom_frac": 0.28,         # maximum sustainable walking+living energy burn per tick
    # reproduction
    "repro_frac": 0.82,        # energy fraction required to spawn
    "repro_safe_radius": 330.0,  # no predator within this radius to allow spawn
    "repro_popcap": 2,         # max OTHER agents observed before refusing to spawn (dispersion)
    "repro_global_target": 10,  # cooperative: don't spawn while estimated global population >= this
    "spawn_cooldown": 400,     # ticks between spawns (anti-overpopulation)
    "spawn_cap": 14,           # hard cap: never spawn while local(other) agents >= this
    # --- population-maintaining reproduction (2026-09-17 survival fix) ---
    "repro_frac_min": 0.24,    # floor for the energy gate when pop is far below target
    "repro_urgency": 1.0,      # 0 = flat repro_frac, 1 = scale gate down to repro_frac_min as pop->0
    "repro_unlimited": 0.0,    # 1 = ignore gpop/popcap gate entirely (rely on energy + cooldown)
    # --- foraging efficiency ---
    "forage_nearest": 0.0,     # 1 = when fruit is visible, steer straight at the nearest safe fruit
    "forage_speed": 1.0,       # move-distance fraction used by the nearest-fruit override
    # --- generational relay (age-aware: survive past max_age by banking heirs) ---
    "relay_age": 0.0,          # sim seconds; spawn an heir once older than this (0 = off)
    "relay_energy_frac": 0.35,  # energy gate used by the relay spawn (fraction of max_energy)
    "relay_cooldown": 700,     # ticks between relay spawns
    "repro_energy_abs": 0.0,   # ABSOLUTE energy gate (sim requires >100); 0 = use fractional rf
    # --- age-based roles, fully decentralised ---
    "role_by_age": 0.0,        # 1 = young agents forage hard, old agents conserve
    "young_age": 25.0,         # sim seconds below which an agent is a "young forager"
    "old_age": 60.0,           # sim seconds above which an agent is "old" (conserve + bank heir)
    "young_speed_frac": 1.0,   # movement scale for young agents
    "old_speed_frac": 0.5,     # movement scale for old agents (they are a dying investment)
}

# module-level memory: per-agent-id last action / flee state / spawn clock.
_MEM = {}
_EPOCH = 0
_GC = {}   # agent_id -> last-seen epoch (pruned via TTL -> cheap cooperative population estimate)
_AGE = {}  # agent_id -> last age (monotonic within an episode; a drop => new episode => reset)

_MATH_COS = math.cos
_MATH_SIN = math.sin
_MATH_ATAN2 = math.atan2
_MATH_HYPOT = math.hypot
_MATH_PI = math.pi


def _maybe_new_episode(aid, age):
    """Age is monotonic within one episode, so an age decrease means the world reset."""
    global _EPOCH
    prev = _AGE.get(aid)
    if prev is not None and age < prev - 10.0:
        _MEM.clear()
        _GC.clear()
        _AGE.clear()
        _EPOCH = 0
    _AGE[aid] = age


def reset_memory():
    """Hard reset of ALL module-level policy state. Call at the START of every episode."""
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
    return _MATH_ATAN2(_MATH_SIN(a), _MATH_COS(a))


def _det_rand(i):
    """Deterministic pseudo-random in [0,1) from an integer (xorshift-ish)."""
    x = (int(i) * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 29
    return (x & 0xFFFFFFFF) / float(0x100000000)


def _mem(agent_id):
    m = _MEM.get(agent_id)
    if m is None:
        m = {"flee": False, "spawn_clock": 0, "last_steer": 0.0, "clock": 0}
        _MEM[agent_id] = m
    return m


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

    # --- single pass instead of four comprehensions + a fifth tree rescan ---
    fruits = []
    preds = []
    agents = []
    edges = []
    trees = []
    for x in o:
        t = x.get("type")
        if t == "Fruit":
            fruits.append(x)
        elif t == "Predator":
            preds.append(x)
        elif t == "Agent":
            agents.append(x)
        elif t == "Edge":
            edges.append(x)
        elif t == "Tree":
            trees.append(x)

    # --- descend spawn clock ---
    if m["spawn_clock"] > 0:
        m["spawn_clock"] -= 1

    P_get = P.get
    cos = _MATH_COS
    sin = _MATH_SIN
    atan2 = _MATH_ATAN2
    hypot = _MATH_HYPOT

    danger = P["danger_dist"]
    danger_risk = danger * 1.4

    # ---- predator repulsion field (local frame vectors) ----
    px = py = 0.0
    threat_dist = float("inf")
    threat_dir = None
    use_pred = P_get("use_predator", True)
    if use_pred:
        pweight = P["predator_weight"]
        for p in preds:
            d = p["distance"]
            if d > danger:
                continue
            a = p["angle"]
            if d < threat_dist:
                threat_dist = d
                threat_dir = a
            weight = pweight / (max(d, 5.0) ** 2)
            px += -cos(a) * weight
            py += -sin(a) * weight

    # flee override with hysteresis
    facing_threat = False
    if use_pred:
        cone = P["concern_cone"]
        for p in preds:
            rdir = atan2(sin(p.get("rel_dir", 0)), cos(p.get("rel_dir", 0)))
            if abs(rdir) < cone and p["distance"] < danger:
                facing_threat = True
                break
    in_escape = m["flee"]
    if use_pred and threat_dist < P["flee_dist"]:
        in_escape = True
    elif threat_dist > P["escape_dist"]:
        in_escape = False
    flee_engage = (P_get("use_flee", True)) and (in_escape or (facing_threat and threat_dist < danger * 0.7))

    if flee_engage:
        mag = hypot(px, py)
        if mag > 1e-6:
            away = atan2(py, px)
        elif threat_dir is not None:
            away = atan2(-sin(threat_dir), -cos(threat_dir))
        else:
            away = 0.0
        m["flee"] = True
        if ef > P["reserve_frac"]:
            dist = sprint * P["flee_speed_frac"]
        else:
            dist = speed * 0.9
        return [float(dist), float(away), 0.0, 0.0]

    m["flee"] = False

    # ---- fruit attraction (weighted by safety) ----
    fx = fy = 0.0
    best_fruit = None
    use_fruit = P_get("use_fruit", True)
    if use_fruit:
        # prey-only risk terms, in predator order, hoisted out of the n_fruit x n_pred double loop
        prisks = []
        for p in preds:
            pd = p["distance"]
            if pd < danger_risk:
                prisks.append((p["angle"], 1 - pd / danger_risk))
        if prisks:
            penalty = P["fruit_risk_penalty"]
            for f in fruits:
                d = f["distance"]
                a = f["angle"]
                risk = 0.0
                for pa, scale in prisks:
                    angdiff = abs(atan2(sin(pa - a), cos(pa - a)))
                    if angdiff < 1.0:
                        risk += (1.0 - angdiff) * penalty * scale
                w = 1.0 / (d + 8.0) * max(0.0, 1.0 - risk)
                fx += cos(a) * w
                fy += sin(a) * w
                if best_fruit is None or d < best_fruit["distance"]:
                    best_fruit = f
        else:
            for f in fruits:
                d = f["distance"]
                a = f["angle"]
                w = 1.0 / (d + 8.0)
                fx += cos(a) * w
                fy += sin(a) * w
                if best_fruit is None or d < best_fruit["distance"]:
                    best_fruit = f

    # ---- tree attraction as weaker fallback (trees host fruit) ----
    if use_fruit:
        for t in trees:
            d = t["distance"]
            a = t["angle"]
            w = 0.25 / (d + 20.0)
            fx += cos(a) * w
            fy += sin(a) * w

    # ---- other-agent dispersion (avoid clustering -> predator multi-kills) ----
    if P_get("use_disperse", True):
        disp = P["disperse_weight"]
        for ag in agents:
            d = ag["distance"]
            if d < 60.0 and d > 1.0:
                a = ag["angle"]
                w = disp / (d + 5.0)
                fx -= cos(a) * w
                fy -= sin(a) * w

    # ---- wall/edge repulsion ----
    # Original summation order (all edge STARTS, then all edge ENDS) is preserved verbatim.
    # Points outside the 90-unit box can never pass `d < 90.0` -> skip the hypot for them.
    if edges and P_get("use_wall", True):
        wall = P["wall_weight"]
        pxW = pyW = 0.0
        for e in edges:
            ex, ey = e["coords"][0]
            if -90.0 <= ex <= 90.0 and -90.0 <= ey <= 90.0:
                d = hypot(ex, ey)
                if d < 90.0 and d > 1.0:
                    w = wall / (d ** 2)
                    pxW -= ex / d * w
                    pyW -= ey / d * w
        for e in edges:
            ex, ey = e["coords"][1]
            if -90.0 <= ex <= 90.0 and -90.0 <= ey <= 90.0:
                d = hypot(ex, ey)
                if d < 90.0 and d > 1.0:
                    w = wall / (d ** 2)
                    pxW -= ex / d * w
                    pyW -= ey / d * w
        fx += pxW
        fy += pyW

    # ---- wander ---
    rw_ang = m.get("wander_ang")
    if rw_ang is None or m["clock"] % 80 == 0:
        rw_ang = _det_rand(aid * 1000003 + m["clock"]) * 2 * _MATH_PI
    m["wander_ang"] = rw_ang
    if P_get("use_explore", True):
        wander = P["wander_weight"]
        fx += cos(rw_ang) * wander
        fy += sin(rw_ang) * wander

    # ---- combine; hysteresis on steering angle to reduce oscillation ----
    mag = hypot(fx, fy)
    if mag < 1e-6:
        steer = _wrap(rw_ang)
    else:
        steer = atan2(fy, fx)

    # ---- direct-forage override (nearest fruit) ----
    direct_forage = False
    if P_get("forage_nearest", 0.0) > 0.0 and fruits:
        tf = fruits[0]
        tf_d = tf["distance"]
        for f in fruits:
            fd = f["distance"]
            if fd < tf_d:
                tf = f
                tf_d = fd
        steer = _wrap(tf["angle"])
        direct_forage = True

    # sticky: don't flip heading wildly tick-to-tick unless food is very close
    if not direct_forage and (best_fruit is None or best_fruit["distance"] > 25.0):
        last = m["last_steer"]
        diff = _wrap(steer - last)
        steer = last + diff * P["target_hyst"]
    m["last_steer"] = _wrap(steer)

    # ---- move distance (energy-aware) ----
    use_energy = P_get("use_energy", True)
    if direct_forage:
        dist = speed * float(P_get("forage_speed", 1.0))
    elif best_fruit is not None and best_fruit["distance"] < 40.0:
        dist = min(speed, best_fruit["distance"])
    elif use_energy and ef < P["low_energy_frac"]:
        dist = speed * 0.4
    else:
        dist = speed * P["walk_frac"] if (best_fruit is not None) else speed * P["explore_frac"]
    if use_energy and ef < P["reserve_frac"]:
        dist = min(dist, speed * 0.25)

    # ---- age-based role scaling ----
    if P_get("role_by_age", 0.0) > 0.0:
        _age = state.get("age", 0.0)
        if _age > P_get("old_age", 60.0):
            dist *= P_get("old_speed_frac", 0.5)
        elif _age < P_get("young_age", 25.0):
            dist *= P_get("young_speed_frac", 1.0)

    # ---- reproduction (investment-gated; cooperative pop cap via shared estimate) ----
    spawn = 0.0
    use_repro = P_get("use_repro", True)
    target = P_get("repro_global_target", 10)
    rf = P_get("repro_frac", DEFAULT_PARAMS["repro_frac"])
    urgency = P_get("repro_urgency", 1.0)
    if urgency > 0.0 and target > 0:
        deficit = max(0.0, min(1.0, (target - gpop) / target))
        rf = rf - (rf - P_get("repro_frac_min", 0.24)) * deficit * urgency
    if P_get("repro_unlimited", 0.0) > 0.0:
        pop_ok = True
        crowd_ok = True
    else:
        pop_ok = gpop < target
        crowd_ok = len(agents) <= P_get("repro_popcap", 2)

    # --- generational relay (never let the chain break) ---
    relay_age = P_get("relay_age", 0.0)
    relay = relay_age > 0.0 and state.get("age", 0.0) > relay_age
    if relay:
        pop_ok = True
        rf = min(rf, P_get("relay_energy_frac", 0.35))
        cd = P_get("relay_cooldown", 700)
    else:
        cd = P_get("spawn_cooldown", 400)
    abs_gate = P_get("repro_energy_abs", 0.0)
    gate_ok = (energy > abs_gate) if abs_gate > 0.0 else (ef > rf)
    if (use_repro and gate_ok
            and m["spawn_clock"] <= 0
            and pop_ok
            and crowd_ok):
        safe_radius = P_get("repro_safe_radius", 330.0)
        safe = True
        for p in preds:
            if not (p["distance"] >= safe_radius):   # verbatim negation of the shipped all() predicate
                safe = False
                break
        if safe:
            spawn = 1.0
            m["spawn_clock"] = cd

    return [float(dist), float(steer), 0.0, float(spawn)]


_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_controller", "params.json")
_PARAMS_CACHE = {"P": None, "key": None}


def _load_params():
    """Memoised params load: same values as the shipped loader, without a file open per request."""
    try:
        st = os.stat(_PARAMS_PATH)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if _PARAMS_CACHE["key"] != key or _PARAMS_CACHE["P"] is None:
        P = dict(DEFAULT_PARAMS)
        if key is not None:
            try:
                with open(_PARAMS_PATH) as f:
                    P.update(json.load(f))  # partial file: merge over defaults so no KeyError
            except Exception:
                pass
        _PARAMS_CACHE["P"] = P
        _PARAMS_CACHE["key"] = key
    return _PARAMS_CACHE["P"]


# standalone importable policy function (contract) using saved/evolved params
def best_controller(state):
    return potential_controller(state, _load_params())


def make_policy(params):
    P = params

    def fn(state):
        return potential_controller(state, P)
    return fn
