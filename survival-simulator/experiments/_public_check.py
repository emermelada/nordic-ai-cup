import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest

core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                      starting_agents=5, starting_predators=2,
                      starting_fruits=32, starting_trees=50, seed=202)
actions = [(a.agent_id, ActionRequest(agent_id=a.agent_id, move_distance=6.0, move_direction=0.0,
                                      turn_angle=0.1, spawn_agent=False)) for a in core.env.agents]
for _ in range(40):
    out = core.step(actions)
    actions = [(s["agent_id"], ActionRequest(agent_id=s["agent_id"], move_distance=6.0,
               move_direction=0.0, turn_angle=0.1, spawn_agent=False)) for s in out["observations"]]
st = [ObservationResponse(**{k: s[k] for k in ("agent_id","energy","biome","age","speed",
      "sprint_speed","hearing_radius","vision_angle","vision_range","max_energy","observations")})
      for s in out["observations"]]
payload = json.dumps(json.loads(StepResponse(game_status="running", score=out["score"],
          sim_time=out["sim_time"], n_agents=len(st), agent_status=st).model_dump_json()))
open("/tmp/realistic_payload.json", "w").write(payload)
print(f"built payload: {len(st)} agents, {len(payload)} bytes")

r = subprocess.run(["curl", "-s", "--max-time", "30", "-X", "POST",
                    "https://survival.zaitzev.com/predict",
                    "-H", "content-type: application/json",
                    "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "-d", "@/tmp/realistic_payload.json"], capture_output=True, text=True)
print("curl exit:", r.returncode)
try:
    body = json.loads(r.stdout)
    acts = body["actions"]
    print("n_actions:", len(acts))
    for a in acts:
        assert set(a.keys()) == {"agent_id","move_distance","move_direction","turn_angle","spawn_agent"}, a.keys()
        assert 0.0 <= a["move_distance"] <= 20.0 and isinstance(a["spawn_agent"], bool)
    print("sample:", json.dumps(acts[0]))
    print("PUBLIC END-TO-END PASSED: real sim state -> deployed heuristic -> valid actions")
except Exception as e:
    print("PARSE FAIL:", e, "raw:", r.stdout[:300])