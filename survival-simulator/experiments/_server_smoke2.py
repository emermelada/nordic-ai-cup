import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest
import agent_server

print("ACTIVE POLICY:", getattr(agent_server, "_POLICY_NAME", "?"))

core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                      starting_agents=5, starting_predators=3,
                      starting_fruits=32, starting_trees=50, seed=303)
actions = [(a.agent_id, ActionRequest(agent_id=a.agent_id, move_distance=6.0, move_direction=0.0,
                                      turn_angle=0.1, spawn_agent=False)) for a in core.env.agents]
for _ in range(50):
    out = core.step(actions)
    actions = [(s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=6.0,
               move_direction=0.0, turn_angle=0.1, spawn_agent=False)) for s in out["observations"]]

st = [ObservationResponse(**{k: s[k] for k in ("agent_id","energy","biome","age","speed",
      "sprint_speed","hearing_radius","vision_angle","vision_range","max_energy","observations")})
      for s in out["observations"]]
step = StepResponse(game_status="running", score=out["score"], sim_time=out["sim_time"],
                    n_agents=len(st), agent_status=st)
# call predict twice to exercise the controller's per-agent/tick memory
for i in range(2):
    r = agent_server.predict(step)
acts = r["actions"]
print("n_actions:", len(acts))
for a in acts:
    assert set(a.keys()) == {"agent_id","move_distance","move_direction","turn_angle","spawn_agent"}, a.keys()
    assert 0.0 <= a["move_distance"] <= 20.0 and isinstance(a["spawn_agent"], bool)
print("sample:", json.dumps(acts[0]))
print("SERVER SMOKE PASSED with evolved controller")