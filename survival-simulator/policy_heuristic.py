"""
Self-contained heuristic agent policy for the Survival Simulator.
Deployable candidate (no weights, deterministic-ish). Mirrors the validated heuristic in
experiments/policies.py so behavior is identical between local eval and serving.

Input: an agent observation as a plain dict with keys matching the served StepResponse /
src.utils.DTOs.ObservationResponse (agent_id, energy, biome, age, speed, sprint_speed,
hearing_radius, vision_angle, vision_range, max_energy, observations).
observations[] entries: {type, distance, angle[, rel_dir]}, Edge has coords.
Returns: (move_distance, move_direction, turn_angle, spawn_flag) as plain floats/bools.
"""
import math
import random

_rng = random.Random(0)
SAFETY = 90.0       # distance at which a predator is considered a threat
PENALTY_BIOMES = ("swamp", "river")


def _go(ang):
    return float(math.atan2(math.sin(ang), math.cos(ang)))


def decide(obs):
    """obs: full agent observation dict (see module docstring)."""
    o = obs.get("observations") or []
    energy = obs.get("energy", 0.0)
    max_e = max(obs.get("max_energy", 1.0) or 1.0, 1.0)
    speed = obs.get("speed", 10.0)
    sprint = obs.get("sprint_speed", 20.0)

    preds = [x for x in o if x.get("type") == "Predator"]
    threat = None
    if preds:
        p = min(preds, key=lambda x: x["distance"])
        rdir = math.atan2(math.sin(p.get("rel_dir", 0.0)), math.cos(p.get("rel_dir", 0.0)))
        if p["distance"] < SAFETY or abs(rdir) < math.pi / 2:
            threat = p

    # Flee predators
    if threat is not None and threat["distance"] < SAFETY + 20:
        away = math.atan2(-math.sin(threat["angle"]), -math.cos(threat["angle"]))
        dist = sprint * (1.0 if threat["distance"] < SAFETY / 2 else 0.55)
        return float(dist), float(away), 0.0, False

    fruits = [x for x in o if x.get("type") == "Fruit"]
    trees = [x for x in o if x.get("type") == "Tree"]
    if fruits:
        f = min(fruits, key=lambda x: x["distance"])
        return float(min(speed, f["distance"])), _go(f["angle"]), 0.0, False
    if trees:
        t = min(trees, key=lambda x: x["distance"])
        return float(speed * 0.7), _go(t["angle"]), 0.0, False
    # no food in view: wander, reverse out of slow biome
    steer = float(_rng.uniform(-0.4, 0.4))
    if obs.get("biome") in PENALTY_BIOMES:
        steer = steer + math.pi
    return float(speed * 0.5), steer, 0.0, False


def spawn_when_rich(obs):
    """Separate small decision so the endpoint can gate spawning cleanly."""
    energy = obs.get("energy", 0.0)
    max_e = max(obs.get("max_energy", 1.0) or 1.0, 1.0)
    danger = any(x.get("type") == "Predator" and x.get("distance", 1e9) < SAFETY * 1.5
                 for x in (obs.get("observations") or []))
    return bool(energy > max_e * 0.7 and not danger)


def decide_with_spawn(obs):
    d, m, t, _ = decide(obs)
    return d, m, t, spawn_when_rich(obs)