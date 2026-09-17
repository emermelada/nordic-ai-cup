"""Instrumented single-episode probe: trace population / energy / births / deaths / food supply.
Usage: python probe_surv.py <seed> <horizon> [param overrides as json]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
import best_controller as bc
from best_controller import DEFAULT_PARAMS


def build_params(overrides):
    P = dict(DEFAULT_PARAMS)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "best_controller", "params.json")) as f:
            P.update(json.load(f))
    except Exception:
        pass
    P.update(overrides)
    return P


def probe(seed=100, horizon=8000, overrides=None, outfile=None):
    P = build_params(overrides or {})
    fn = bc.make_policy(P)
    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400,
                          starting_agents=5, starting_predators=0,
                          starting_fruits=32, starting_trees=50, seed=seed)
    trace = []
    last_score = 0.0
    cum_deaths = 0
    cum_births = 0
    cum_fruit = 0
    eaten = 0
    spawn_req = 0
    prev_ids = set(core.env.agents_dict.keys())
    collapse = horizon
    for i in range(horizon):
        livestates = [core.env.get_agent_state(a.agent_id) for a in core.env.agents]
        acts = []
        for st in livestates:
            act = fn(st)
            if act[3] > 0.5:
                spawn_req += 1
            acts.append((st["agent_id"], ActionRequest(
                agent_id=st["agent_id"], move_distance=float(act[0]),
                move_direction=float(act[1]), turn_angle=float(act[2]),
                spawn_agent=bool(act[3]))))
        n_before = len(core.env.agents)
        out = core.step(acts)
        now_ids = set(core.env.agents_dict.keys())
        cum_births += max(0, len(core.env.agents) - n_before)
        if len(core.env.agents) < n_before:
            eaten += n_before - len(core.env.agents)
        cum_deaths += len(prev_ids - now_ids)
        if out["score"] - last_score > 0.11:
            cum_fruit += 1
        prev_ids = now_ids
        if (i + 1) % max(250, horizon // 30) == 0 or i == horizon - 1:
            states = [core.env.get_agent_state(a.agent_id) for a in core.env.agents]
            e = [s["energy"] for s in states if s]
            ages = [s["age"] for s in states if s]
            trace.append({
                "t": i + 1, "pop": len(core.env.agents),
                "e_mean": round(float(np.mean(e)), 1) if e else 0.0,
                "e_max": round(float(np.max(e)), 1) if e else 0.0,
                "age_mean": round(float(np.mean(ages)), 1) if ages else 0,
                "deaths": cum_deaths, "births": cum_births, "spawn_req": spawn_req,
                "trees": len(core.env.trees), "fruits": len(core.env.fruits),
                "preds": len(core.env.predators), "fruit_eaten": cum_fruit,
                "eaten_by_pred": eaten, "score": round(last_score, 1) })
        last_score = out["score"]
        if out["num_agents"] == 0:
            collapse = i + 1
            break
    print(f"seed={seed} horizon={horizon} collapse_ticks={collapse} score={core.env.score:.1f} "
          f"births={cum_births} deaths={cum_deaths} spawn_req={spawn_req} "
          f"fruit_eaten={cum_fruit} predated={eaten} preds_now={len(core.env.predators)}")
    print(f"{'t':>6} {'pop':>4} {'e_mean':>7} {'e_max':>7} {'age':>6} {'birth':>6} {'death':>6} "
          f"{'sreq':>5} {'trees':>6} {'fruit':>6} {'fed':>5} {'pred':>5}")
    for r in trace:
        print(f"{r['t']:>6} {r['pop']:>4} {r['e_mean']:>7} {r['e_max']:>7} {r['age_mean']:>6} "
              f"{r['births']:>6} {r['deaths']:>6} {r['spawn_req']:>5} {r['trees']:>6} {r['fruits']:>6} "
              f"{r['fruit_eaten']:>5} {r['preds']:>5}")
    if outfile:
        json.dump({"collapse": collapse, "trace": trace}, open(outfile, "w"), indent=1)
    return {"collapse": collapse, "births": cum_births, "deaths": cum_deaths, "spawn_req": spawn_req}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    probe(seed, horizon, ov)