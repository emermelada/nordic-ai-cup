import time, sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

# Time a reset
t0 = time.time()
core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                      starting_agents=1, starting_predators=0,
                      starting_fruits=32, starting_trees=50, seed=1)
t_reset = time.time() - t0
print(f"reset: {t_reset:.2f}s  n_agents={len(core.env.agents)} fruits={len(core.env.fruits)} trees={len(core.env.trees)} score={core.env.score:.3f} time={core.env.time:.3f}")

# Time steps
def act_for(agent):
    return ActionRequest(agent_id=agent.agent_id, move_distance=10.0, move_direction=0.0,
                         turn_angle=0.1, spawn_agent=False)

N = 30
t0 = time.time()
for _ in range(N):
    actions = [(a.agent_id, act_for(a)) for a in core.env.agents]
    core.step(actions)
t_steps = time.time() - t0
print(f"{N} steps: {t_steps:.2f}s -> {N/t_steps:.2f} steps/s")

# Inspect one observation
state = core.step([(a.agent_id, act_for(a)) for a in core.env.agents])
obs0 = state["observations"][0]
print("obs keys:", sorted(obs0.keys()))
print("obs['observations'] n=", len(obs0["observations"]))
print("sample entity:", obs0["observations"][0] if obs0["observations"] else "none")
print("score now:", state["score"], "sim_time:", state["sim_time"], "num_agents:", state["num_agents"])
import shapely; print("shapely", shapely.__version__)