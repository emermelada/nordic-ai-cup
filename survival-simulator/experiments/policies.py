"""Policy functions: each takes an agent_state dict and returns an action tuple
[move_distance, move_direction, turn_angle, spawn_flag]. Shared by eval; the
single-agent gym env uses its own 3-d action for PPO training (adapted in launchers)."""
import math, random
import numpy as np

_rng = random.Random(0)


def random_policy(state):
    o = state.get("observations") or []
    return [float(_rng.uniform(0, 20)), float(_rng.uniform(-math.pi, math.pi)),
            float(_rng.uniform(-math.pi / 4, math.pi / 4)), 0.0]


def dummy_policy(state):
    # mirrors dummy_agent_policy.py (random move to sprint, turn 45deg, always spawn)
    o = state.get("observations") or []
    return [float(_rng.uniform(0.0, state.get("sprint_speed", 20.0))), 0.0,
            float(_rng.uniform(-math.pi / 4, math.pi / 4)), 1.0]


def _go(dirv):
    return math.atan2(math.sin(dirv), math.cos(dirv))


def heuristic_policy(state, safety=90.0, safe_at_dist=120.0, avoid_penalty=0.4):
    """Seek fruit (or trees), flee predators, avoid slow biomes, spawn when rich."""
    o = state.get("observations") or []
    energy = state.get("energy", 0.0)
    max_e = max(state.get("max_energy", 1.0), 1.0)
    speed = state.get("speed", 10.0)
    sprint = state.get("sprint_speed", 20.0)

    preds = [x for x in o if x.get("type") == "Predator"]
    # threat: near predator (within hearing*1.2 or facing us)
    threat = None
    if preds:
        p = min(preds, key=lambda x: x["distance"])
        rdir = math.atan2(math.sin(p.get("rel_dir", 0)), math.cos(p.get("rel_dir", 0)))
        if p["distance"] < safety or abs(rdir) < math.pi / 2:
            threat = p

    # flee: steer away, sprint
    if threat is not None and threat["distance"] < safety + 20:
        away = math.atan2(-math.sin(threat["angle"]), -math.cos(threat["angle"]))
        dist = sprint * (1.0 if threat["distance"] < safety / 2 else 0.55)
        return [float(dist), float(away), 0.0, 0.0]

    fruits = [x for x in o if x.get("type") == "Fruit"]
    trees = [x for x in o if x.get("type") == "Tree"]
    target = None
    steer = None
    if fruits:
        target = min(fruits, key=lambda x: x["distance"])
        steer = _go(target["angle"])
        move = min(speed, target["distance"])
    elif trees:
        target = min(trees, key=lambda x: x["distance"])
        steer = _go(target["angle"])
        move = speed * 0.7
    else:
        # wander
        steer = float(_rng.uniform(-0.4, 0.4))
        move = speed * 0.5

    # biome penalty: avoid slow biome if no immediate food
    bio = state.get("biome", "")
    if bio in ("swamp", "river") and not fruits:
        steer = steer + math.pi  # reverse
    spawn = 1.0 if (energy > max_e * 0.7 and threat is None) else 0.0
    return [float(move), float(steer), 0.0, float(spawn)]


def heuristic_v2(state, vision_ahead=170.0, flee_at=110.0, edge_clear=45.0,
                 low_energy=0.25, high_energy=0.85):
    """V2: energy-aware cruising + early predator drift + wall/edge avoidance + strict spawn."""
    o = state.get("observations") or []
    energy = state.get("energy", 0.0)
    max_e = max(state.get("max_energy", 1.0) or 1.0, 1.0)
    ef = energy / max_e
    speed = state.get("speed", 10.0)
    sprint = state.get("sprint_speed", 20.0)

    # ---- predator threat ----
    threat = None
    preds = [x for x in o if x.get("type") == "Predator"]
    if preds:
        p = min(preds, key=lambda x: x["distance"])
        rdir = math.atan2(math.sin(p.get("rel_dir", 0)), math.cos(p.get("rel_dir", 0)))
        in_cone = abs(rdir) < math.pi / 3 and p["distance"] < vision_ahead
        if p["distance"] < flee_at or in_cone:
            threat = p
    if threat is not None:
        away = math.atan2(-math.sin(threat["angle"]), -math.cos(threat["angle"]))
        d = threat["distance"]
        if d < flee_at / 2:
            dist = sprint
        elif d < flee_at:
            dist = sprint * 0.7
        else:
            dist = speed * 0.9
            away *= 0.6  # gentle drift, keep foraging intent
        return [float(dist), float(away), 0.0, 0.0]

    # ---- foraging (food) ----
    fruits = [x for x in o if x.get("type") == "Fruit"]
    trees = [x for x in o if x.get("type") == "Tree"]
    if fruits:
        f = min(fruits, key=lambda x: x["distance"])
        steer = _go(f["angle"]); move = min(speed, f["distance"])
    elif trees:
        t = min(trees, key=lambda x: x["distance"])
        steer = _go(t["angle"]); move = speed * 0.7
    else:
        steer = float(_rng.uniform(-0.35, 0.35)); move = speed * 0.45

    # ---- wall/edge avoidance: steer away from nearest visible edge point ----
    edges = [x for x in o if x.get("type") == "Edge"]
    if edges:
        pts = [e["coords"][0] for e in edges] + [e["coords"][1] for e in edges]
        n = min(pts, key=lambda p: math.hypot(p[0], p[1]))
        dmin = math.hypot(n[0], n[1])
        if dmin < edge_clear:  # coords are agent-local; steer away from the near wall
            edge_ang = math.atan2(n[1], n[0])
            steer = math.atan2(-math.sin(edge_ang), -math.cos(edge_ang)) * 0.8

    # ---- energy-aware cruising ----
    if ef < low_energy:
        move = min(move, speed * 0.4)
    elif ef > high_energy:
        move = min(speed, move * 1.2)

    # ---- conservative spawning ----
    danger = any((x.get("type") == "Predator" and x.get("distance", 1e9) < 200.0) for x in o)
    agents_near = sum(1 for x in o if x.get("type") == "Agent")
    spawn = 1.0 if (ef > high_energy and not danger and agents_near <= 1) else 0.0
    return [float(move), float(steer), 0.0, float(spawn)]