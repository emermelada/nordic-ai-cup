"""
Shared observation encoder + Gymnasium single-agent training env + multi-agent evaluator.
Observation encoding MUST be identical between training and inference (server).
"""
import os, math, sys, random
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for `src.*`
# ^ APPEND, not insert(0): the repo root also holds a best_controller.py (deployment copy) and
#   prepending it here silently shadowed experiments/best_controller.py for every later import.
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

BIOME_ID = {"forest": 0, "grassland": 1, "swamp": 2, "desert": 3, "river": 4}
MAX_ENT = {"vision_range": 400.0, "hearing": 100.0, "speed": 20.0, "sprint": 40.0}
MAX_AGE = 120.0
MAX_TURN = math.pi
HORIZON = 30000  # default episode length in sim steps (3000 s sim time = 30,000 ticks per challenge directive)

OBS_DIM = 31
OBS_HI = 10.0


def _closest(obs_list, etype, bearing="angle", with_rel_dir=False):
    """Return features for nearest entity of a type from the observation list."""
    sel = [o for o in obs_list if o.get("type") == etype]
    if not sel:
        return None
    sel.sort(key=lambda o: o["distance"])
    o = sel[0]
    d = min(o["distance"] / 300.0, 1.0)
    a = o[bearing]
    feat = [d, math.cos(a), math.sin(a)]
    if with_rel_dir:
        r = o.get("rel_dir", 0.0)
        feat += [math.cos(r), math.sin(r)]
    return feat


def build_obs(agent_state):
    """agent_state: dict from get_agent_state (matches the served StepResponse fields)."""
    o = agent_state
    obs_list = o.get("observations") or []
    max_e = max(o["max_energy"], 1.0)
    f = []
    # ---- scalars ----
    f += [o["energy"] / max_e,
          min(o["age"] / MAX_AGE, 1.0),
          o["speed"] / MAX_ENT["speed"],
          o["sprint_speed"] / MAX_ENT["sprint"],
          o["vision_range"] / MAX_ENT["vision_range"],
          min(o["vision_angle"] / (math.pi / 2), 1.0),
          o["hearing_radius"] / MAX_ENT["hearing"]]
    # biome one-hot
    bio = BIOME_ID.get(o["biome"], 4)
    onehot = [0.0] * 5
    onehot[bio] = 1.0
    f += onehot
    # counts
    f += [sum(1 for e in obs_list if e.get("type") == "Fruit"),
          sum(1 for e in obs_list if e.get("type") == "Predator"),
          sum(1 for e in obs_list if e.get("type") == "Agent"),
          sum(1 for e in obs_list if e.get("type") == "Tree")]
    # nearest entities
    nf = _closest(obs_list, "Fruit")
    f += nf if nf else [1.0, 0.0, 0.0]
    np_ = _closest(obs_list, "Predator", with_rel_dir=True)
    f += np_ if np_ else [1.0, 0.0, 0.0, 0.0, 0.0]
    na = _closest(obs_list, "Agent", with_rel_dir=True)
    f += na if na else [1.0, 0.0, 0.0, 0.0, 0.0]
    # edges (obstacles visible / clearance)
    edges = [e for e in obs_list if e.get("type") == "Edge"]
    edge_flag = 1.0 if edges else 0.0
    if edges:
        # coords already in local frame; distance ~ hypot of nearest start point
        dists = [math.hypot(e["coords"][0][0], e["coords"][0][1]) for e in edges]
        clearance = min(min(dists) / 300.0, 1.0)
    else:
        clearance = 1.0
    f += [edge_flag, clearance]
    assert len(f) == OBS_DIM, f"obs dim mismatch {len(f)} != {OBS_DIM}"
    return np.array(f, dtype=np.float32)


class FleetEnv(gym.Env):
    """
    Single-agent training env: exactly ONE controlled agent in the world (clean signal).
    The learned policy is shared at inference across all real agents.
    """
    metadata = {"render_modes": []}

    def __init__(self, seed=0, horizon=HORIZON, shaped=True):
        super().__init__()
        self.seed0 = seed
        self.horizon = horizon
        self.shaped = shaped
        self.observation_space = spaces.Box(-OBS_HI, OBS_HI, (OBS_DIM,), np.float32)
        # [move_dist, turn_angle, spawn_flag(0..1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_TURN, 0.0]),
            high=np.array([20.0, MAX_TURN, 1.0]),
            dtype=np.float32)
        self.core = None
        self.learner_id = None
        self.step_i = 0
        self._last_score = 0.0
        self._reset(seed)

    def _reset(self, seed=None):
        self.core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                                   starting_agents=1, starting_predators=0,
                                   starting_fruits=32, starting_trees=50, seed=seed if seed is not None else self.seed0)
        self.learner_id = self.core.env.agents[0].agent_id
        self.step_i = 0
        self._last_score = 0.0

    def _state_of(self, agent_id):
        return self.core.env.get_agent_state(agent_id)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset(seed=seed if seed is not None else self.seed0)
        return self.build_obs_from_state(self._state_of(self.learner_id)), {}

    @staticmethod
    def build_obs_from_state(st):
        return build_obs(st)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        dist = float(a[0])
        turn = float(np.clip(a[1], -MAX_TURN, MAX_TURN))
        spawn = bool(a[2] > 0.5)
        req = ActionRequest(agent_id=self.learner_id, move_distance=dist,
                            move_direction=0.0, turn_angle=turn, spawn_agent=spawn)
        st = self.core.step([(self.learner_id, req)])
        self.step_i += 1
        r = st["score"] - self._last_score
        self._last_score = st["score"]
        if self.shaped:
            r = r - self.core.dt  # remove constant survival bonus so fruit/predation drive learning
        state = self._state_of(self.learner_id)
        terminated = state is None  # learner died
        truncated = (not terminated) and (self.step_i >= self.horizon)
        obs = self.build_obs_from_state(state) if state is not None else np.zeros(OBS_DIM, np.float32)
        info = {"score": st["score"], "sim_time": st["sim_time"],
                "n_agents": st["num_agents"], "terminated": terminated,
                "energy": (state or {}).get("energy", 0.0)}
        return obs, float(r), terminated, truncated, info


# ---------------- multi-agent evaluation (served / held-out) ----------------

def make_action(state, act):
    return ActionRequest(agent_id=state["agent_id"], move_distance=float(act[0]),
                         move_direction=float(act[1]), turn_angle=float(act[2]),
                         spawn_agent=bool(act[3]))


def run_eval_episode(policy_fn, n_agents=5, seed=0, horizon=HORIZON, env_width=1600, env_height=1200, stop_on_death=False, trace=False, trace_every=250, recorder=None, reset_fn=None):
    """policy_fn(state)->action tuple [dist, move_dir, turn, spawn]. Runs the REAL multi-agent world.
    By default runs the FULL horizon (constant +dt accrues regardless; fair grader proxy) and returns
    the score at horizon. If stop_on_death=True, halts early when all agents die (survivorship-biased)."""
    # DETERMINISM: policies here use the global `random` module (e.g. wander angles), and Python
    # auto-seeds it from OS entropy at import -> the SAME seed gave different episodes per process.
    # Seed both global RNGs here so (seed, policy) -> an identical, reproducible episode.
    random.seed(seed)
    np.random.seed(seed)
    if reset_fn is not None:  # clear policy-side module state -> episodes are order-invariant
        reset_fn()
    core = SimulationCore(env_width=env_width, env_height=env_height, chunk_size=400,
                          starting_agents=n_agents, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    last_score = 0.0
    fruits = eaten = spawns = 0
    steps = 0
    traces = []
    for i in range(horizon):
        steps = i + 1
        states = core.env.get_agent_state  # bound
        livestates = [states(a.agent_id) for a in core.env.agents]
        acts = [(s["agent_id"], make_action(s, policy_fn(s))) for s in livestates]
        before = len(core.env.agents)
        out = core.step(acts)
        after = len(core.env.agents)
        score_delta = out["score"] - last_score
        last_score = out["score"]
        if after < before:
            eaten += before - after
        elif after > before:
            spawns += after - before
        if score_delta > 0.11:  # fruit bonus on top of 0.1 dt
            fruits += 1
        if trace and steps % trace_every == 0:
            energies = [(states(a.agent_id) or {}).get("energy", 0.0) for a in core.env.agents]
            traces.append({"t": steps, "n": after,
                           "e_mean": round(sum(energies) / len(energies), 2) if energies else 0.0})
        if recorder is not None:  # (pre-step states, chosen actions, post-step out) - keeps ONE loop
            recorder(i, livestates, acts, out)
        if stop_on_death and out["num_agents"] == 0:
            break
    return {"score": core.env.score, "steps": steps, "fruits_eaten": fruits,
            "predated": eaten, "spawns": spawns, "final_agents": len(core.env.agents),
            "alive": len(core.env.agents) > 0, "traces": traces}