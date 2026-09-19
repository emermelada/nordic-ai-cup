#!/usr/bin/env bash
# run_gs_confirm.sh - 40-seed paired confirmation of the energy-breeding arms.
#
# The 4-seed mechanism screen showed the largest effect of the project: energy-dominant breeder
# selection raised fleet max_energy 1.32-1.53x, cut lockout time, raised fruit intake, and lifted
# survival 19-42% (en_top3 +41.7%). Mechanism and outcome agreed, which the pre-registered rule
# requires. But it is FOUR SEEDS, and four separate leads died at 40 seeds tonight, so this is the
# confirmation that decides it - on seeds never used for screening (2960-2999 screen, 3000-3039
# holdout), with an A/A duplicate to measure this run's noise floor.
#
# Included on purpose:
#   BASE_gs / BASE_dup  deployed C6 (genome_select 0) and its byte-identical twin -> noise floor
#   en_top2             the AGGRESSIVE V2 setting (topk 2) as a thinning-risk probe; if it also wins,
#                       the "selection thins the fleet" fear was unfounded
#   vis_ctrl            vision-dominant control: if energy is the real lever, this should NOT keep up
set -u
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy

/opt/nacv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, '/opt/nac_gs/experiments')
sys.path.insert(0, '/opt/nac_gs')
import mechanism_screen as ms

base = ms.load_params()
out = [{"id": "BASE_gs", "params": dict(base)}, {"id": "BASE_dup", "params": dict(base)}]
for k in ("en_top2", "en_top3", "en_top4", "en_top3_late", "en_top3_rescue3", "vis_ctrl"):
    if k in ms.ARMS:
        p = dict(base)
        p.update(ms.ARMS[k])
        out.append({"id": k, "params": p})
    else:
        print(f"  (arm {k} not defined in mechanism_screen.ARMS - skipped)")
json.dump(out, open('/opt/nac_gs/experiments/par_gs_confirm.json', 'w'), indent=1)
print(f"wrote par_gs_confirm.json: {len(out)} arms (2 baselines incl. A/A + {len(out)-2} candidates)")
PY

tmux kill-session -t gsconfirm 2>/dev/null || true
tmux new-session -d -s gsconfirm "cd /opt/nac_gs/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python sched.py --lane gs_confirm --candidates par_gs_confirm.json --seeds 2960-2999 --holdout 3000-3039 --stages 40:18000 --workers ${1:-26} --topk 8 --tol 0.0 --minutes 180 2>&1 | tee run_gsconfirm.log"

sleep "${2:-40}"
echo "=== lane ==="
tmux ls | tr '\n' ' '
echo
echo "=== progress ==="
head -4 /opt/nac_gs/experiments/run_gsconfirm.log 2>/dev/null
tail -2 /opt/nac_gs/experiments/run_gsconfirm.log 2>/dev/null
uptime
