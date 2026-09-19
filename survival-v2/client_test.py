"""Official-style client against a running server: times requests.post around .dict() like the grader's budget."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "official"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import requests
url = sys.argv[1]; ticks = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
fast = len(sys.argv) > 4 and sys.argv[4] == "fast"
if fast:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fastsim import SimulationCore
else:
    from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest
# the platform's connection test: no sim_time / n_agents, lowercase types
verify = {"game_status": "ok", "score": 0.0, "agent_status": [{"agent_id": 0, "energy": 100.0, "biome": "forest", "age": 1.0,
          "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0, "vision_angle": 1.0, "vision_range": 200.0, "max_energy": 500.0,
          "observations": [{"type": "tree", "distance": 10.0, "angle": 0.1}, {"type": "fruit", "distance": 20.0, "angle": -0.2}]}]}
r = requests.post(url, json=verify, timeout=10); print("verify sample:", r.status_code, r.text[:200])
s = requests.Session()
sim = SimulationCore(seed=seed)
step = StepResponse(game_status="ok", score=0, sim_time=sim.env.time, n_agents=len(sim.env.agents), agent_status=[])
waits = []
for k in range(ticks):
    t0 = time.perf_counter()
    resp = s.post(url, json=step.dict(), timeout=10)
    waits.append(time.perf_counter() - t0)
    actions = [ActionRequest(**a) for a in resp.json().get("actions", [])]
    state = sim.step([(a.agent_id, a) for a in actions])
    status = [ObservationResponse(**{k2: o[k2] for k2 in ("agent_id", "observations", "energy", "biome", "age", "speed", "sprint_speed",
              "hearing_radius", "vision_angle", "vision_range", "max_energy")}) for o in state["observations"] if o is not None]
    step = StepResponse(game_status="ok" if state["num_agents"] > 0 and sim.env.time <= 3000 else "game_over", score=state["score"],
                        sim_time=state["sim_time"], n_agents=state["num_agents"], agent_status=status)
    if step.game_status == "game_over": break
waits.sort()
print(f"ticks {len(waits)} score {state['score']:.1f} agents {state['num_agents']} wait mean {1000*sum(waits)/len(waits):.2f} ms "
      f"p50 {1000*waits[len(waits)//2]:.2f} p99 {1000*waits[int(len(waits)*0.99)]:.2f} max {1000*waits[-1]:.1f} "
      f"-> projected per game {sum(waits)/len(waits)*30000:.0f} s of the 600 s budget")
