"""w_lockout_diag.py — the instrument that decides WHICH bottleneck actually ends the run.

The question a code reading cannot answer: is the sprint lockout (environment.py:512 clamps
movement to walking speed below max_energy/5) the DOMINANT killer, or is it one real mechanic
among predator avoidance, food accessibility, movement expenditure and population structure?

Measured here, on the official eval configuration (5 agents, 1600x1200) at the official horizon,
with the DEPLOYED controller:
  1. deaths classified three ways — EATEN / AGED / STARVED — using the sim's own signals:
     a predator's energy jumping by more than its resting gain means a kill; age > max_age at the
     moment of death means the age drain; anything else is starvation. Each death records the
     agent's energy and whether it was inside the sprint-lockout zone.
  2. the share of agent-TICKS spent in the lockout zone, overall and while a predator is near.
  3. the MEASURED predator closure rate split by whether the agent was facing the predator
     (|bearing| < pi/2, which makes the predator PIVOT) or turned away (which makes it CHARGE).
     This tests the facing claim empirically instead of inferring it from the source.
  4. food: standing fruit energy in the world vs fruits VISIBLE per agent vs the blind fraction,
     per 1,000-tick bucket — supply versus access.
  5. income (fruits eaten) and fleet size/energy per bucket, to show where the decay bites.

Usage: python w_lockout_diag.py <params.json> <seed,seed,...> <horizon>
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import SimulationCore                      # noqa: E402
import best_controller as bc                            # noqa: E402
from env_wrapper import make_action                     # noqa: E402

BUCKET = 1000
NEAR = 150.0          # "a predator is near" for the exposure statistics
REST_GAIN = 3.0       # dt*30 while a predator rests; a jump above this means it ate someone


def classify_deaths(core, prev_pred_energy, prev_states, alive_now):
    """Return a list of (agent_id, cause, energy_at_death, in_lockout, age, max_age) for new deaths."""
    out = []
    for aid, st in prev_states.items():
        if aid in alive_now:
            continue
        eaten = False
        for p in core.env.predators:
            before = prev_pred_energy.get(id(p))
            if before is not None and float(p.energy) - before > REST_GAIN + 5.0:
                eaten = True
                break
        max_e = float(st.get("max_energy", 500.0) or 500.0)
        age = float(st.get("age", 0.0))
        max_age = float(st.get("max_age", 999.0))
        energy = float(st.get("energy", 0.0))
        if eaten:
            cause = "eaten"
        elif age > max_age:
            cause = "aged"
        else:
            cause = "starved"
        out.append((aid, cause, energy, energy < 0.2 * max_e, age, max_age))
    return out


def main():
    path, seeds_s, horizon_s = sys.argv[1], sys.argv[2], sys.argv[3]
    horizon = int(horizon_s)
    seeds = [int(s) for s in seeds_s.split(",") if s.strip()]
    blob = json.load(open(path))
    P = dict(bc.DEFAULT_PARAMS)
    P.update(blob.get("params", blob) if isinstance(blob, dict) else {})
    print(f"lockout diagnostic | horizon {horizon} | reserve_frac={P.get('reserve_frac')} "
          f"repro_energy_abs={P.get('repro_energy_abs')}", flush=True)

    for seed in seeds:
        bc.reset_memory()
        fn = bc.make_policy(P)
        core = SimulationCore(env_width=1600, env_height=1200, chunk_size=400, starting_agents=5,
                              starting_predators=0, starting_fruits=32, starting_trees=50, seed=seed)
        counts = {"eaten": 0, "aged": 0, "starved": 0}
        lockout_deaths = {"eaten": 0, "aged": 0, "starved": 0}
        death_energy = []
        ticks_total = ticks_lockout = 0
        near_ticks = near_lockout = 0
        closure = {"facing": [], "turned": []}
        buckets = {}
        prev_dist = {}
        prev_pred_e = {id(p): float(p.energy) for p in core.env.predators}
        i = 0
        for i in range(horizon):
            live = list(core.env.agents)
            if not live:
                break
            states = []
            for a in live:
                st = core.env.get_agent_state(a.agent_id)
                if st:
                    st["max_age"] = getattr(a, "max_age", 999.0)
                    states.append(st)
            if not states:
                break
            # ---- telemetry from the PRE-step state (this is what the policy sees) ----
            for st in states:
                ticks_total += 1
                max_e = float(st.get("max_energy", 500.0) or 500.0)
                e = float(st["energy"])
                in_lock = e < 0.2 * max_e
                if in_lock:
                    ticks_lockout += 1
                preds = [o for o in st.get("observations", []) if o.get("type") == "Predator"]
                if preds:
                    pn = min(preds, key=lambda o: o["distance"])
                    if pn["distance"] < NEAR:
                        near_ticks += 1
                        if in_lock:
                            near_lockout += 1
                if preds:
                    key = (st["agent_id"],)
                    d_now = min(o["distance"] for o in preds)
                    d_prev = prev_dist.get(key)
                    facing = abs(min(preds, key=lambda o: o["distance"])["angle"]) < math.pi / 2
                    if d_prev is not None and d_prev > 0 and d_now > 0:
                        rate = (d_prev - d_now)          # + means the predator is closing
                        closure["facing" if facing else "turned"].append(rate)
                    prev_dist[key] = d_now
                else:
                    prev_dist[(st["agent_id"],)] = None
            acts = [(st["agent_id"], make_action(st, fn(st))) for st in states]
            before_agents = {a.agent_id for a in core.env.agents}
            out = core.step(acts)
            alive_now = {a.agent_id: float(a.energy) for a in core.env.agents}
            for (aid, cause, energy, in_lock, age, max_age) in classify_deaths(
                    core, prev_pred_e, {s["agent_id"]: s for s in states}, alive_now):
                counts[cause] += 1
                if in_lock:
                    lockout_deaths[cause] += 1
                if cause == "eaten":
                    death_energy.append(round(energy, 1))
            prev_pred_e = {id(p): float(p.energy) for p in core.env.predators}
            # ---- per-bucket aggregates ----
            b = (i + 1) // BUCKET
            agg = buckets.setdefault(b, {"n": 0, "e": 0.0, "n_meas": 0, "standing": 0.0,
                                         "vis": 0, "blind": 0, "preds": 0, "fleet_e": 0.0})
            agg["n"] += 1
            agg["n_meas"] += len(states)
            agg["standing"] += sum(float(f.energy) for f in core.env.fruits)
            for st in states:
                fr = [o for o in st.get("observations", []) if o.get("type") == "Fruit"]
                agg["vis"] += len(fr)
                if not fr:
                    agg["blind"] += 1
            agg["preds"] = max(agg["preds"], len(core.env.predators))
            agg["fleet_e"] += sum(alive_now.values())
            agg["agents"] = max(agg.get("agents", 0), len(alive_now))
        steps = i + 1
        tot = max(1, sum(counts.values()))
        print(f"\nseed={seed} steps={steps} final_agents={len(core.env.agents)} "
              f"score={core.env.score:.2f}", flush=True)
        print(f"  deaths: {counts}  ("
              + " / ".join(f"{k} {100 * v / tot:.0f}%" for k, v in counts.items()) + ")", flush=True)
        print(f"  deaths inside the sprint-lockout zone: {lockout_deaths} "
              f"({100 * sum(lockout_deaths.values()) / tot:.0f}% of all deaths)", flush=True)
        if counts["eaten"]:
            print(f"  energy at predation death: n={len(death_energy)} "
                  f"mean={sum(death_energy) / len(death_energy):.1f} "
                  f"median={sorted(death_energy)[len(death_energy) // 2]:.1f} "
                  f"min={min(death_energy):.1f} max={max(death_energy):.1f}", flush=True)
        if ticks_total:
            print(f"  agent-ticks in lockout: {100 * ticks_lockout / ticks_total:.1f}%", flush=True)
        if near_ticks:
            print(f"  exposure: {100 * near_ticks / max(1, ticks_total):.1f}% of agent-ticks with a "
                  f"predator within {NEAR:.0f}; of those {100 * near_lockout / near_ticks:.1f}% in lockout",
                  flush=True)
        for k, v in closure.items():
            if v:
                print(f"  closure rate while {k:6s}: mean={sum(v) / len(v):+.3f} units/tick "
                      f"(n={len(v)})", flush=True)
        print("   tick    agents  fleet_E  standing_fruit  vis/agent  blind%  preds", flush=True)
        for b in sorted(buckets):
            a = buckets[b]
            if not a["n_meas"]:
                continue
            print(f"  {b * BUCKET:6d}  {a.get('agents', 0):6d}  {a['fleet_e'] / a['n']:8.0f}  "
                  f"{a['standing'] / a['n']:13.0f}  {a['vis'] / a['n_meas']:9.2f}  "
                  f"{100 * a['blind'] / a['n_meas']:6.1f}  {a['preds']:5d}", flush=True)


if __name__ == "__main__":
    main()
