"""Agent endpoint for the survival simulator.

    uvicorn server:app --host 0.0.0.0 --port 9052 --loop uvloop --http httptools   (Linux)
    python server.py                                                              (anything)

POST /predict takes the StepResponse JSON and returns {"actions": [...]}. The body is parsed with orjson and
handed to the Hive as plain dicts (no per-observation pydantic models: those cost milliseconds per tick on
big colonies, and the grader allows 600 s of accumulated wait per game).

Never fails a tick: on any exception the previous plan is dropped and every agent gets a no-op, and the error
is logged. One Hive serves consecutive games; it resets itself when sim_time goes backwards.
Timing and the caller's address are logged every 1000 requests (SURV_LOG, default server_log.jsonl).
"""
import json
import math
import os
import sys
import time
import traceback

import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hive import Hive  # noqa: E402

LOG_PATH = os.environ.get("SURV_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_log.jsonl"))
PARAMS = json.loads(os.environ["HIVE_PARAMS"]) if os.environ.get("HIVE_PARAMS") else None

hive = Hive(params=PARAMS)
stats = {"n": 0, "decide_s": 0.0, "decide_max": 0.0, "bytes": 0, "errors": 0, "gap_s": 0.0, "last_done": None,
         "clients": {}}


def _log(rec):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _noop(step):
    return [{"agent_id": a.get("agent_id"), "move_distance": 0.0, "move_direction": 0.0, "turn_angle": 0.0,
             "spawn_agent": False} for a in (step.get("agent_status") or []) if isinstance(a, dict)]


TRACE_PATH = os.environ.get("SURV_TRACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_trace.csv"))
FIRST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_request.json")
_first_dumped = [False]


def _trace(step, dt, n_actions):
    """One compact line per request: wall time, sim time, agents, score, decide ms, observation types seen."""
    try:
        agents = step.get("agent_status") or []
        types = {}
        for a in agents[:3]:
            for o in (a.get("observations") or [])[:50]:
                types[str(o.get("type"))] = 1
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f},{step.get('sim_time')},{len(agents)},{step.get('score')},{step.get('game_status')},"
                    f"{1000 * dt:.2f},{n_actions},{'|'.join(sorted(types))}\n")
        if agents and not _first_dumped[0]:
            _first_dumped[0] = True
            with open(FIRST_PATH, "wb") as f:
                f.write(orjson.dumps(step, option=orjson.OPT_INDENT_2))
    except Exception:
        pass


def _clean(actions, step):
    """Every field a plain finite float/int/bool: a NaN would serialise as null and fail the grader's
    ActionRequest validation, which ends the game without an error message."""
    alive = {a.get("agent_id") for a in (step.get("agent_status") or []) if isinstance(a, dict)}
    out = []
    bad = 0
    for a in actions:
        try:
            aid = int(a["agent_id"])
            vals = []
            for k in ("move_distance", "move_direction", "turn_angle"):
                v = float(a.get(k, 0.0))
                if not math.isfinite(v):
                    v = 0.0
                    bad += 1
                vals.append(v)
            out.append({"agent_id": aid, "move_distance": max(0.0, vals[0]), "move_direction": vals[1],
                        "turn_angle": vals[2], "spawn_agent": bool(a.get("spawn_agent", False))})
        except Exception:
            bad += 1
    if bad:
        stats["bad_values"] = stats.get("bad_values", 0) + bad
        _log({"t": time.time(), "bad_values": bad, "sim_time": step.get("sim_time")})
    return out


def handle(body, client=None):
    t0 = time.perf_counter()
    if stats["last_done"] is not None:
        stats["gap_s"] += t0 - stats["last_done"]
    step = {}
    try:
        step = orjson.loads(body) if body else {}
        actions = _clean(hive.decide(step), step)
    except Exception:
        stats["errors"] += 1
        _log({"t": time.time(), "error": traceback.format_exc()[-2000:]})
        actions = _noop(step if isinstance(step, dict) else {})
    try:
        out = orjson.dumps({"actions": actions})
    except Exception:
        stats["errors"] += 1
        _log({"t": time.time(), "error": "dumps: " + traceback.format_exc()[-2000:]})
        out = orjson.dumps({"actions": _noop(step if isinstance(step, dict) else {})})
    dt = time.perf_counter() - t0
    if isinstance(step, dict):
        _trace(step, dt, len(actions))
    stats["n"] += 1
    stats["decide_s"] += dt
    stats["decide_max"] = max(stats["decide_max"], dt)
    stats["bytes"] += len(body or b"")
    if client:
        stats["clients"][client] = stats["clients"].get(client, 0) + 1
    stats["last_done"] = time.perf_counter()
    if stats["n"] % 1000 == 0:
        n = stats["n"]
        _log({"t": time.time(), "requests": n, "sim_time": step.get("sim_time") if isinstance(step, dict) else None,
              "agents": len(step.get("agent_status") or []) if isinstance(step, dict) else None,
              "decide_ms_mean": round(1000 * stats["decide_s"] / n, 3), "decide_ms_max": round(1000 * stats["decide_max"], 1),
              "kb_mean": round(stats["bytes"] / n / 1024, 1), "between_requests_ms_mean": round(1000 * stats["gap_s"] / max(n - 1, 1), 2),
              "errors": stats["errors"], "clients": stats["clients"]})
    return out


async def app(scope, receive, send):
    """Bare ASGI app: /predict, / and /api."""
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return
    path = scope.get("path", "/")
    method = scope.get("method", "GET")
    if method == "POST" and path.rstrip("/") == "/predict":
        chunks = []
        more = True
        while more:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        client = scope.get("client")
        out = handle(b"".join(chunks), client[0] if client else None)
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(out)).encode())]})
        await send({"type": "http.response.body", "body": out})
        return
    if path in ("/", "/api", "/health"):
        body = orjson.dumps({"message": "Agent endpoint running!", "requests": stats["n"], "errors": stats["errors"]})
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
        return
    await send({"type": "http.response.start", "status": 404, "headers": [(b"content-length", b"0")]})
    await send({"type": "http.response.body", "body": b""})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "9052")),
                log_level="warning", access_log=False)
