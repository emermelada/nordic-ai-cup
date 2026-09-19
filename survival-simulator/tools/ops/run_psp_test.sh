#!/usr/bin/env bash
# run_psp_test.sh - 40-seed paired test of the ONE discovery-derived state rule.
#
# Rule (from the counterfactual boundary analysis, 8,440 branch rows): with fruit in sight, WALK before
# tick 900 and SPRINT in the 900-2700 window. 65% cross-seed consistency on the sprint half.
#
# Scope discipline: ONE arm with the discovered window, plus an A/A duplicate. Not a sweep of windows -
# the user has explicitly stopped parameter sweeps, and widening the window would turn a boundary test
# into one.
set -u
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy

cp /opt/nac/best_controller/params.json /opt/nac_gs/best_controller/params.json
# the patched controller (with the psp rule) must be in this tree
cp /opt/nac/best_controller.py /opt/nac_gs/best_controller.py

/opt/nacv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, '/opt/nac_gs'); sys.path.insert(0, '/opt/nac_gs/experiments')
import best_controller as bc
assert "psp_mode" in bc.DEFAULT_PARAMS, "patched controller missing from the gs tree"
base = json.load(open('/opt/nac_gs/best_controller/params.json'))
out = [{"id": "BASE_deployed", "params": dict(base)},
       {"id": "PSP_900_2700", "params": {**base, "psp_mode": 1.0, "psp_lo": 900.0, "psp_hi": 2700.0,
                                         "psp_mid_frac": 1.0}},
       {"id": "BASE_dup", "params": dict(base)}]
inert = [c["id"] for c in out if c["id"] not in ("BASE_deployed", "BASE_dup")
         and all(c["params"].get(k) == base.get(k) for k in set(base) | set(c["params"]))]
if inert:
    raise SystemExit(f"ABORT: inert arms {inert}")
json.dump(out, open('/opt/nac_gs/experiments/par_psp.json', 'w'), indent=1)
print(f"wrote par_psp.json: {len(out)} arms (baseline + rule + A/A duplicate)")
PY

tmux kill-session -t psp 2>/dev/null || true
tmux new-session -d -s psp "cd /opt/nac_gs/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python sched.py --lane psp --candidates par_psp.json --seeds 3600-3639 --holdout 3640-3679 --stages 40:18000 --workers ${1:-24} --topk 3 --tol 0.0 --minutes 120 2>&1 | tee run_psp.log"
sleep "${2:-40}"
tmux ls 2>/dev/null | tr '\n' ' '; echo
head -5 /opt/nac_gs/experiments/run_psp.log 2>/dev/null
tail -2 /opt/nac_gs/experiments/run_psp.log 2>/dev/null
uptime
