"""TRAIT-DRIFT + PREDATOR-ACCUMULATION PROBE.

Two open questions this measures (nothing here is inferred from code reading):

1. TRAIT DRIFT. `spawn_agent(parent=...)` copies six heritable traits and mutates each with
   probability 0.1 by uniform(0.5, 1.5) (environment.py:296-345), with caps
   speed<=20, sprint<=40, max_energy<=1000, hearing<=chunk/4=100, vision<=chunk=400, cone<=pi/2.
   Defaults are speed 10, sprint 20, max_energy 500, hearing 50, vision 200, cone pi/3.
   So the fleet's genotype is a random walk that the POLICY controls, because the policy chooses
   which agents spawn. If the fleet's traits drift UP, evolution is already silently paying us;
   if they sit at the defaults, that axis is unexploited.

2. PREDATOR ACCUMULATION. spawn chance = (1/max(1,N)) * dt * time * 0.0001 (environment.py:763),
   predators are never removed, and they steal the victim's energy on a kill (so a kill refuels
   the burst). Deterministic drift dN/dt = t*1e-5 gives N ~ 5e-6 * t^2, i.e. ~16 predators by the
   t=1800 s (18,000-tick) horizon. Measure it instead of trusting the arithmetic.

Usage: python w_traits.py [params.json] [seed] [horizon]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import SimulationCore                      # noqa: E402
import best_controller as bc                            # noqa: E402
from env_wrapper import make_action                     # noqa: E402

TRAITS = ("speed", "sprint_speed", "max_energy", "hearing_radius", "vision_range", "vision_angle")
DEFAULTS = {"speed": 10.0, "sprint_speed": 20.0, "max_energy": 500.0,
            "hearing_radius": 50.0, "vision_range": 200.0, "vision_angle": 1.0471975511965976}


def summarize(states):
    d = {}
    for tr in TRAITS:
        vals = [s[tr] for s in states if tr in s]
        d[f"{tr}_mean"] = round(sum(vals) / len(vals), 2) if vals else None
        d[f"{tr}_max"] = round(max(vals), 2) if vals else None
    return d


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "best_controller/params.json"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 12000

    raw = json.load(open(path))
    P = dict(bc.DEFAULT_PARAMS)
    P.update(raw.get("params", raw))
    bc.reset_memory()
    fn = bc.make_policy(P)

    core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                          starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)

    seen_ids = {a.agent_id for a in core.env.agents}
    births = []
    rows = []
    i = 0
    for i in range(horizon):
        live = [a.agent_id for a in core.env.agents]
        if not live:
            break
        states = [s for s in (core.env.get_agent_state(a) for a in live) if s]
        if not states:
            break
        acts = [(s["agent_id"], make_action(s, fn(s))) for s in states]
        core.step(acts)

        new_ids = {a.agent_id for a in core.env.agents} - seen_ids
        cur = {}
        for a in core.env.agents:
            cur[a.agent_id] = a
        for nid in sorted(new_ids):
            a = cur.get(nid)
            if a is not None:
                births.append({"t": i + 1, **{tr: round(getattr(a, tr, float("nan")), 2) for tr in TRAITS}})
        seen_ids |= new_ids

        if i % 500 == 0:
            st = [s for s in (core.env.get_agent_state(a) for a in live) if s]
            row = {"t": i + 1, "agents": len(core.env.agents), "predators": len(core.env.predators),
                   "fleet_energy": round(sum(a.energy for a in core.env.agents), 1),
                   "births": len(births)}
            row.update(summarize(st))
            rows.append(row)

    print(f"seed={seed} params={path} steps={i + 1} births={len(births)}")
    hdr = ["t", "agents", "predators", "fleet_energy", "births",
           "speed_mean", "speed_max", "sprint_speed_mean", "sprint_speed_max",
           "max_energy_mean", "max_energy_max", "hearing_radius_mean", "hearing_radius_max",
           "vision_range_mean", "vision_range_max", "vision_angle_mean", "vision_angle_max"]
    print(",".join(hdr))
    for r in rows:
        print(",".join(str(r.get(h, "")) for h in hdr))
    if births:
        print("\n# births: mean trait of newborns vs the default genotype")
        for tr in TRAITS:
            vals = [b[tr] for b in births if b[tr] == b[tr]]
            if vals:
                print(f"  {tr:16s} n={len(vals):4d} mean={sum(vals)/len(vals):8.2f} "
                      f"min={min(vals):8.2f} max={max(vals):8.2f} default={DEFAULTS[tr]:8.2f}")
        print(f"  generations: {len(births)} births over {i + 1} ticks")


if __name__ == "__main__":
    main()
