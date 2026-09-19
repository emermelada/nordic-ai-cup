"""Memoryless baseline: calibrates what the Hive's world model buys.

Per agent: flee from a predator closer than FLEE (full walk speed, sprint if slower than 15.5 and it is
close), else walk to the nearest visible fruit, else wander in a slowly turning direction. Breed when
energy > BREED and the colony is below CAP. Faces the nearest predator when fleeing.
"""
import math
import random

PI = math.pi


def wrap(a):
    return (a + PI) % (2 * PI) - PI


class Hive:
    def __init__(self, params=None, seed=0):
        p = {"flee": 110.0, "breed": 170.0, "cap": 14, "ripe_skip": 0.0}
        if params:
            p.update(params)
        self.p = p
        self.rng = random.Random(seed)
        self.wander = {}
        self.stats = {}
        self.mem = {}

    def decide(self, step):
        agents = step.get("agent_status") or []
        n = len(agents)
        out = []
        for a in sorted(agents, key=lambda q: q["agent_id"]):
            obs = a.get("observations") or []
            preds = [o for o in obs if str(o.get("type", "")).lower() == "predator"]
            fruits = [o for o in obs if str(o.get("type", "")).lower() == "fruit"]
            speed, sprint, E = a["speed"], a["sprint_speed"], a["energy"]
            spawn = E > self.p["breed"] and n < self.p["cap"] and not preds
            if spawn:
                n += 1
            if preds:
                pr = min(preds, key=lambda o: o["distance"])
                if pr["distance"] < self.p["flee"]:
                    dist = speed
                    if speed < 15.5 and pr["distance"] < 60 and E > a["max_energy"] / 5 + 10:
                        dist = sprint
                    turn = max(-1.0, min(1.0, pr["angle"]))
                    out.append({"agent_id": a["agent_id"], "move_distance": dist, "move_direction": wrap(pr["angle"] + PI),
                                "turn_angle": turn, "spawn_agent": False})
                    continue
            if fruits:
                f = min(fruits, key=lambda o: o["distance"])
                out.append({"agent_id": a["agent_id"], "move_distance": min(speed, f["distance"]),
                            "move_direction": f["angle"], "turn_angle": max(-0.5, min(0.5, f["angle"])),
                            "spawn_agent": spawn})
                continue
            w = self.wander.get(a["agent_id"])
            if w is None or self.rng.random() < 0.02:
                w = self.rng.uniform(-0.3, 0.3)
                self.wander[a["agent_id"]] = w
            out.append({"agent_id": a["agent_id"], "move_distance": min(speed, 6.0), "move_direction": 0.0,
                        "turn_angle": w, "spawn_agent": spawn})
        return out
