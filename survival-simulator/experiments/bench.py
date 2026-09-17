"""Shared benchmarking / failure analysis for the survival-sim experiments.
Mirrors env_wrapper.run_eval_episode but adds optional death-cause instrumentation.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env_wrapper import run_eval_episode, HORIZON
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

SEEDS = [100, 200, 300, 400, 500]


def multi_eval(policy_fn, seeds=SEEDS, horizon=HORIZON, n_agents=5, stop=False):
    res = []
    for s in seeds:
        t0 = time.time()
        r = run_eval_episode(policy_fn, n_agents=n_agents, seed=s, horizon=horizon, stop_on_death=stop)
        r["seed"] = s
        r["t"] = time.time() - t0
        res.append(r)
    return res


def summarize(res, name=""):
    sc = [r["score"] for r in res]
    return {
        "name": name,
        "mean": float(np.mean(sc)), "median": float(np.median(sc)),
        "std": float(np.std(sc)), "min": float(np.min(sc)), "max": float(np.max(sc)),
        "fruits": float(np.mean([r["fruits_eaten"] for r in res])),
        "predated": float(np.mean([r["predated"] for r in res])),
        "final_agents": float(np.mean([r["final_agents"] for r in res])),
        "alive": sum(r["alive"] for r in res),
        "steps": [r["steps"] for r in res],
        "t": float(np.mean([r["t"] for r in res])),
        "n": len(res),
    }


def print_summary(s, extra=""):
    s = dict(s)
    pop = s.pop("name", "")
    sc = s.pop("mean")
    std = s.pop("std")
    print(f"[{pop}] mean={sc:.1f}±{std:.1f} med={dict(s)['median']:.1f} "
          f"[{s['min']:.0f},{s['max']:.0f}] fruit={s['fruits']:.0f} pred={s['predated']:.1f} "
          f"final={s['final_agents']:.1f} alive={s['alive']}/{s['n']} {extra}")


def _predictor_contact(core, eps=1e-9):
    """ids of agents currently touching a predator."""
    contact = set()
    for p in core.env.predators:
        for a in core.env.agents:
            d = np.hypot(p.x - a.x, p.y - a.y)
            if d < p.size + a.size + eps:
                contact.add(a.agent_id)
    return contact


def failure_analysis(policy_fn, seed=100, horizon=20000, n_agents=5):
    """Classify every agent death: predator-contact vs energy/aging vs collapse."""
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=n_agents, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    deaths = {"predator": 0, "energy/aging": 0}
    births = 0
    peak = 0
    collapse_ticks = 0
    last_score = 0.0
    fruit_score = 0.0
    pred_penalty = 0.0
    steps = 0
    for i in range(horizon):
        steps = i + 1
        livestates = [core.env.get_agent_state(a.agent_id) for a in core.env.agents]
        if not livestates:
            collapse_ticks += 1
            acts = []
        else:
            acts = [(st["agent_id"], ActionRequest(
                agent_id=st["agent_id"], move_distance=float(policy_fn(st)[0]),
                move_direction=float(policy_fn(st)[1]), turn_angle=float(policy_fn(st)[2]),
                spawn_agent=bool(policy_fn(st)[3]))) for st in livestates]
        before_alive = set(core.env.agents_dict.keys())
        n_before = len(core.env.agents)
        contact = _predictor_contact(core)
        out = core.step(acts)
        delta = out["score"] - last_score
        last_score = out["score"]
        if delta > 0.11:
            fruit_score += delta - 0.1
        if delta < 0:
            pred_penalty += delta
        peak = max(peak, len(core.env.agents))
        births += max(0, len(core.env.agents) - n_before)
        now_alive = set(core.env.agents_dict.keys())
        for aid in (before_alive - now_alive):
            if aid in contact:
                deaths["predator"] += 1
            else:
                deaths["energy/aging"] += 1
    return {"deaths": deaths, "births": births, "peak": peak, "steps": steps,
            "score": core.env.score, "fruit_score": fruit_score, "pred_penalty": pred_penalty,
            "final_agents": len(core.env.agents), "collapse_ticks": collapse_ticks}