import sys, os, time, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest

core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                      starting_agents=5, starting_predators=2,
                      starting_fruits=32, starting_trees=50, seed=101)
actions = [(a.agent_id, ActionRequest(agent_id=a.agent_id, move_distance=5.0, move_direction=0.0,
                                      turn_angle=0.1, spawn_agent=False)) for a in core.env.agents]
for _ in range(30):
    out = core.step(actions)
    actions = [(s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=5.0,
               move_direction=0.0, turn_angle=0.1, spawn_agent=False)) for s in out["observations"]]

agent_status = [ObservationResponse(**{k: s[k] for k in ("agent_id","energy","biome","age","speed",
                "sprint_speed","hearing_radius","vision_angle","vision_range","max_energy","observations")})
                for s in out["observations"]]
step = StepResponse(game_status="running", score=out["score"], sim_time=out["sim_time"],
                    n_agents=len(agent_status), agent_status=agent_status)
payload = json.dumps(json.loads(step.model_dump_json())).encode()

def post(path, data=None, method="POST"):
    req = urllib.request.Request("http://127.0.0.1:9077" + path, data=data,
                                 headers={"Content-Type":"application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())

# retry until server up
import time as _t
for _ in range(20):
    try:
        s, hello = post("/", method="GET")
        break
    except Exception:
        _t.sleep(0.5)
print("GET / ", s, hello)

t0 = _t.time()
s, body = post("/predict", payload)
dt = (_t.time() - t0) * 1000
print("POST /predict ", s, f"{dt:.1f} ms")
acts = body["actions"]
print("n_actions:", len(acts))
for a in acts:
    assert set(a.keys()) == {"agent_id","move_distance","move_direction","turn_angle","spawn_agent"}, a.keys()
    assert 0.0 <= a["move_distance"] <= 20.0
    assert isinstance(a["spawn_agent"], bool)
print("sample:", json.dumps(acts[0]))
print("MVP CHECK PASSED over real HTTP + real sim state")