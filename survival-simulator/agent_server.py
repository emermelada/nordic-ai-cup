import os
import json
import time
import threading
from fastapi import FastAPI, Body
from src.utils.DTOs import StepResponse, ActionRequest

# ---------------------------------------------------------------------------
# Policy of record: the evolved potential-field controller
# (experiments/best_controller.py + experiments/best_controller/params.json).
# Falls back to the heuristic if anything is missing, so the endpoint can
# never 500 on import.
# ---------------------------------------------------------------------------
try:
    import best_controller as _bc

    def _policy(state):
        return _bc.best_controller(state)
    _POLICY_NAME = "best_controller(potential-field, evolved)"
except Exception:
    from policy_heuristic import decide_with_spawn as _ph

    def _policy(state):
        return _ph(state)
    _POLICY_NAME = "policy_heuristic(fallback)"

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")

# ---------------------------------------------------------------------------
# Grader instrumentation: log every incoming step so we can reverse-engineer
# the real environment + scoring from an actual attempt. Never breaks serving.
# ---------------------------------------------------------------------------
_LOG_PATH = os.environ.get("PREDICT_LOG", "/app/predict_log.jsonl")
_LOG_LOCK = threading.Lock()
_log_written = 0
_LOG_MAX = 200000  # hard cap on lines to bound file growth


def _as_dict(agent):
    """pydantic v2/1 -> plain dict matching get_agent_state keys."""
    try:
        return agent.model_dump()
    except AttributeError:
        return agent.dict()


def _log_request(step, n_spawn):
    global _log_written
    if _log_written >= _LOG_MAX:
        return
    try:
        counts = {}
        energies = []
        for a in step.agent_status:
            for e in (a.observations or []):
                k = e.get("type", "?")
                counts[k] = counts.get(k, 0) + 1
            energies.append(float(a.energy))
        n = len(step.agent_status)
        rec = {
            "t": round(time.time(), 3),
            "sim": round(float(step.sim_time), 2),
            "score": round(float(step.score), 4),
            "n": n,
            "gs": step.game_status,
            "ge": round(step.n_agents, 1),
            "obs": counts,
            "emax": round(max(energies), 1) if energies else None,
            "emean": round(sum(energies) / len(energies), 1) if energies else None,
            "spawn": n_spawn,
        }
        with _LOG_LOCK:
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
            _log_written += 1
    except Exception:
        pass


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    actions = []
    n_spawn = 0
    for agent in step.agent_status:
        state = _as_dict(agent)
        dist, move_dir, turn, spawn = _policy(state)
        if spawn:
            n_spawn += 1
        actions.append(
            ActionRequest(
                agent_id=state["agent_id"],
                move_distance=dist,
                move_direction=move_dir,
                turn_angle=turn,
                spawn_agent=bool(spawn),
            ).model_dump()
        )
    _log_request(step, n_spawn)
    # Must return {"actions": [...]} format
    return {"actions": actions}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!", "policy": _POLICY_NAME}


@app.get("/debug/stats")
def stats():
    """Lightweight introspection (not called by the grader)."""
    return {"policy": _POLICY_NAME, "log_lines": _log_written, "log_path": _LOG_PATH}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)