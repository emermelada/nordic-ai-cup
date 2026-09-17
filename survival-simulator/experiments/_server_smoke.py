import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # survival-simulator/
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest
import agent_server  # imports policy_heuristic + src

def mkobs(agent_id, energy, obs):
    return ObservationResponse(
        agent_id=agent_id, energy=energy, biome="grassland", age=10.0,
        speed=10.0, sprint_speed=20.0, hearing_radius=50.0,
        vision_angle=1.047197551, vision_range=200.0, max_energy=500.0,
        observations=obs)

# case 1: sees fruit + predator (exercises seek + also near-threat -> flee branch)
obs1 = [{"type": "Fruit", "distance": 80.0, "angle": 0.2},
        {"type": "Predator", "distance": 50.0, "angle": 2.5, "rel_dir": 0.1}]
# case 2: empty (wander branch)
obs2 = []
# case 3: rich energy + no threat (spawn branch)
obs3 = [{"type": "Tree", "distance": 120.0, "angle": -0.5}]

step = StepResponse(game_status="running", score=12.5, sim_time=125.0, n_agents=3,
                    agent_status=[mkobs(0, 150.0, obs1), mkobs(1, 40.0, obs2), mkobs(2, 420.0, obs3)])
out = agent_server.predict(step)
acts = out["actions"]
print("return keys:", list(out.keys()))
print("n actions:", len(acts))
for a in acts:
    print(json.dumps(a))
    # validate shape
    assert set(a.keys()) == {"agent_id","move_distance","move_direction","turn_angle","spawn_agent"}, a.keys()
    assert 0.0 <= a["move_distance"] <= 20.0, a["move_distance"]
# spawn branch should fire for agent 2 (energy 420 = 0.84*500)
assert acts[2]["spawn_agent"] is True, "expected spawn for rich agent"
assert acts[0]["spawn_agent"] is False, "expected no spawn when threatened"
print("OK: endpoint returns valid ActionRequest dicts; seek/flee/spawn branches exercised")