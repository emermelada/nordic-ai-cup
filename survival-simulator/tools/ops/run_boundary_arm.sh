#!/usr/bin/env bash
# run_boundary_arm.sh - test the ONE decision boundary the counterfactual analysis found.
#
# The boundary (depth-3 tree, holds on 25/37 seeds):
#     tick > 2700  AND  energy > 323   ->  FLEE the predator
# i.e. only WEALTHY agents late in the game profit from fleeing. The deployed controller has evasion
# switched off entirely (evade_mode 0), and the earlier E3 line - which had evasion ON with a LOW energy
# gate (evade_energy_abs 200) - lost. This arm is the boundary's "high energy" half: re-enable evasion but
# only for agents above 350 energy, so the fleet does not spend escapes on agents that cannot sprint away
# anyway (the lockout means a poor agent fleeing is pointless: it cannot outrun anything).
#
# This is NOT a sweep. It is one pre-registered rule derived from data, tested against the DEPLOYED
# controller on 40 unseen paired seeds, with an A/A duplicate to measure this run's noise floor.
set -u
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy

cp /opt/nac/best_controller/params.json /opt/nac_gs/best_controller/params.json

/opt/nacv/bin/python - <<'PY'
import json
base = json.load(open('/opt/nac_gs/best_controller/params.json'))
assert base.get('genome_select') == 1.0, "baseline must be the DEPLOYED selection controller"
# E3's geometry (the evasion shape that was tested), but gated on WEALTH instead of the low 200 floor
EV = {"evade_mode": 1.0, "evade_dist": 140.0, "evade_disengage": 240.0,
      "evade_speed_frac": 1.0, "evade_energy_abs": 350.0}
arms = [("BASE_deployed", {})]
arms.append(("A1_evade_wealthy", EV))
arms.append(("A2_evade_wealthy_hi", {**EV, "evade_energy_abs": 420.0}))
arms.append(("A3_evade_short", {**EV, "evade_dist": 90.0}))
out = []
for tag, ov in arms:
    p = dict(base); p.update(ov)
    out.append({"id": tag, "params": p})
out.append({"id": "A4_dup_of_base", "params": dict(base)})     # A/A noise floor
# GUARD: every candidate must differ from the baseline (silent no-ops have burned us three times)
inert = [c["id"] for c in out if c["id"] not in ("BASE_deployed", "A4_dup_of_base")
         and all(c["params"].get(k) == base.get(k) for k in set(base) | set(c["params"]))]
if inert:
    raise SystemExit(f"ABORT: inert arms {inert}")
json.dump(out, open('/opt/nac_gs/experiments/par_boundary.json', 'w'), indent=1)
print(f"wrote par_boundary.json: {len(out)} arms (baseline + 3 boundary arms + A/A duplicate)")
PY

tmux kill-session -t boundary 2>/dev/null || true
tmux new-session -d -s boundary "cd /opt/nac_gs/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python sched.py --lane boundary --candidates par_boundary.json --seeds 3400-3439 --holdout 3440-3479 --stages 40:18000 --workers ${1:-24} --topk 5 --tol 0.0 --minutes 150 2>&1 | tee run_boundary.log"
sleep "${2:-45}"
echo "=== lane ==="; tmux ls 2>/dev/null | tr '\n' ' '; echo
head -5 /opt/nac_gs/experiments/run_boundary.log 2>/dev/null
tail -2 /opt/nac_gs/experiments/run_boundary.log 2>/dev/null
uptime
