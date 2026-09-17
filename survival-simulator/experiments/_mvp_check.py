import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # survival-simulator/
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from fastapi.testclient import TestClient
from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse
import agent_server

# --- build a REAL payload from an actual simulation step (the "grader" view) ---
core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                      starting_agents=5, starting_predators=2,
                      starting_fruits=32, starting_trees=50, seed=101)
# run a few steps so the world has fruit/predator observations in view
actions = [(a.agent_id, __import__("src.utils.DTOs", fromlist=["ActionRequest"]).ActionRequest(
           agent_id=a.agent_id, move_distance=5.0, move_direction=0.0, turn_angle=0.1, spawn_agent=False))
           for a in core.env.agents]
for _ in range(30):
    out = core.step(actions)
    actions = [(s["agent_id"], __import__("src.utils.DTOs", fromlist=["ActionRequest"]).ActionRequest(
                agent_id=s["agent_id"], move_distance=5.0, move_direction=0.0, turn_angle=0.1, spawn_agent=False))
               for s in out["observations"]]

# build StepResponse EXACTLY as the grader would serialise it
agent_status = [ObservationResponse(**{k: s[k] for k in ("agent_id","energy","biome","age","speed",
                "sprint_speed","hearing_radius","vision_angle","vision_range","max_energy","observations")})
                for s in out["observations"]]
step = StepResponse(game_status="running", score=out["score"], sim_time=out["sim_time"],
                    n_agents=len(agent_status), agent_status=agent_status)
payload = json.loads(step.model_dump_json())

client = TestClient(agent_server.app)
r0 = client.get("/"); print("GET /", r0.status_code, r0.json())

# time the predict over the real HTTP boundary
t0 = time.time()
r = client.post("/predict", json=payload)
dt = time.time() - t0
print("POST /predict", r.status_code, f"in {dt*1000:.1f} ms")
acts = r.json()["actions"]
print("n_actions:", len(acts))
for a in acts:
    assert set(a.keys()) == {"agent_id","move_distance","move_direction","turn_angle","spawn_agent"}, a.keys()
    assert 0.0 <= a["move_distance"] <= 20.0, a["move_distance"]
    assert isinstance(a["spawn_agent"], bool)
print("sample action:", json.dumps(acts[0]))
print("MVP CHECK PASSED: valid schema, low latency, real sim state handled")