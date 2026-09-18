"""Augmented observation: build_obs (31-dim) + PRIVATE POLICY STATE (3 extra dims).

Why this exists
---------------
The expert controller (best_controller.potential_controller) is STATEFUL in ways the 31-dim
`build_obs` observation cannot express:
    m["spawn_clock"]   per-agent reproduction cooldown (decremented by the expert itself)
    m["flee"]          flee hysteresis / "currently fleeing" latch
    m["last_steer"]    heading stickiness
    m["wander_ang"]    exploration phase
    _global_alive()    TTL estimate of the live population (cooperative reproduction gate)
A feed-forward MLP trained on `build_obs` alone is therefore asked to imitate a partially-observed
policy.  Measured consequence: action match looked acceptable (dist MAE 0.77 of a mean 4.34,
fruit-step steering error 0.41 rad) but spawn decisions could not be reproduced
(7 spawns/episode vs the expert's 130 => the heir chain breaks and the population collapses at
~1,400 ticks, roughly the do-nothing floor).

Fix: the policy MAINTAINS THE SAME KIND OF PRIVATE STATE ITSELF (no oracle, no hidden sim access)
and the extra features are fed to the network as additional inputs.  Crucially every feature here is
computable IDENTICALLY (a) during the expert's rollout (from the recorded per-step state + the
expert's own spawn requests) and (b) inside the deployed policy from its own history — so train and
inference inputs match by construction:

    31: ticks_since_own_spawn_request / 600           (own reproduction clock)
    32: ticks_since_own_danger (nearest predator within flee_dist) / 300   (flee recency latch)
    33: estimated_live_population / 20                 (TTL estimate over recently-stepped agent ids)

All three are derived from `age` (sim seconds, 0.1 s/tick) and the observation list, so no global
RNG or wall-clock is involved: episodes stay deterministic and order-invariant after reset().
"""
import math

import numpy as np

MAX_AGE = 120.0
DT = 0.1                 # sim seconds per tick
SPAWN_RECENCY_TICKS = 600.0
DANGER_RECENCY_TICKS = 300.0
FLEE_DIST = 149.0        # the expert's flee_dist (best_controller/params.json)
POP_NORM = 20.0
POP_TTL_S = 2.0          # an agent whose `age` has not advanced within TTL seconds is considered dead

N_AUG = 3


class AugObs:
    """Per-episode private state -> 3 extra observation features. reset() clears it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_spawn_age = {}
        self.last_danger_age = {}
        self.age_seen = {}

    # --- feature computation (call BEFORE the action for this state is chosen) ---
    def features(self, state):
        aid = int(state.get("agent_id", 0))
        age = float(state.get("age", 0.0))
        last_spawn = self.last_spawn_age.get(aid)
        if last_spawn is None:
            rec = age                                   # no spawn request yet -> time since birth
        else:
            rec = max(0.0, age - last_spawn)
        f1 = min(rec / DT, SPAWN_RECENCY_TICKS) / SPAWN_RECENCY_TICKS

        dmin = math.inf
        for o in (state.get("observations") or []):
            if o.get("type") == "Predator":
                dmin = min(dmin, o["distance"])
        if dmin < FLEE_DIST:
            danger = 0.0                                # in danger right now
        else:
            last_d = self.last_danger_age.get(aid)
            danger = min(max(0.0, age - last_d) / DT, DANGER_RECENCY_TICKS) / DANGER_RECENCY_TICKS \
                if last_d is not None else 1.0
        f2 = danger

        # TTL population estimate: agents whose age advanced recently are alive this tick
        mx = max(self.age_seen.values()) if self.age_seen else age
        pop = sum(1 for a in self.age_seen.values() if a >= mx - POP_TTL_S)
        f3 = min(pop / POP_NORM, 1.0)
        return np.array([f1, f2, f3], dtype=np.float32)

    # --- state update (call AFTER the action for this state was taken) ---
    def update(self, state, spawn_requested):
        aid = int(state.get("agent_id", 0))
        age = float(state.get("age", 0.0))
        if spawn_requested:
            self.last_spawn_age[aid] = age
        dmin = math.inf
        for o in (state.get("observations") or []):
            if o.get("type") == "Predator":
                dmin = min(dmin, o["distance"])
        if dmin < FLEE_DIST:
            self.last_danger_age[aid] = age
        self.age_seen[aid] = age

    def step(self, state, spawn_requested):
        f = self.features(state)
        self.update(state, spawn_requested)
        return f
