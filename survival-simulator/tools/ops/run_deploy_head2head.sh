#!/usr/bin/env bash
# run_deploy_head2head.sh - is the LIVE artifact actually better than what it replaced?
#
# WHY THIS LANE: the queue is empty (three discovery-derived boundary rules all tested and rejected at
# 40 seeds: late-flee -0.5%, phase-speed +0.3%). The one consequential open question is whether the
# en_top3 genome-selection controller SERVING right now is better than the controller it replaced. Its
# own gate said +19.2% (W27/L13); the autopilot's independent 120-seed pool said 54% wins / p10 -285;
# the board recorded a +17% attempt (1,075.28). If the edge does not replicate on fresh seeds then that
# board number is variance and repeated validation - not more controllers - is the lever.
#
# INTEGRITY NOTE (found 2026-09-19 09:35 UTC): /opt/nac/best_controller.py on BOTH boxes holds the OLD
# C6 controller (f90cb4e3, 86-param, no genome_select) while the RUNNING container holds 252f0ba1 with
# genome_select=1.0 docker-cp'd in at 08:36. The host /opt/nac tree is therefore stale relative to the
# served artifact: "BASE = deployed params" applied in that tree silently IGNORES genome_select/gs_*
# (rule 7) and any rebuild of the image would silently revert the fleet to C6 behaviour. So this lane
# runs in an ISOLATED tree /opt/nac_h2h holding the TRUE in-container controller, and the ledger is
# written there.
#
# Arms (4 configs x 160 seeds = 640 episodes @18000 ticks):
#   BASE        - injected by sched.py = /opt/nac_h2h/best_controller/params.json = the DEPLOYED params
#   DEP_dup     - byte-identical copy of BASE -> this run's A/A noise floor
#   C6_pre      - deployed params MINUS genome_select / gs_* -> the pre-deploy controller
#   C6_pre_dup  - byte-identical copy of C6_pre -> A/A floor for the deprecated arm
# Seeds 4000-4159 are outside every seed used so far (top-level cache covered 1301-3419 only).
set -u
TREE=${NAC_TREE:-/opt/nac_h2h}
LANE=${NAC_LANE:-dephead}
SEEDS=${NAC_SEEDS:-4000-4159}
cd "$TREE/experiments"
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy
export NAC_RESULTS="$TREE/results"

/opt/nacv/bin/python - "$TREE" <<'PY'
import hashlib, json, sys
tree = sys.argv[1]
sys.path.insert(0, tree)
import best_controller as bc

base = json.load(open(f'{tree}/best_controller/params.json'))
for k in ("genome_select", "gs_w_energy", "gs_topk"):
    assert k in base, f"deployed params missing {k} - wrong tree?"
assert bc.DEFAULT_PARAMS.get("genome_select") == 0.0, \
    f"genome_select default is {bc.DEFAULT_PARAMS.get('genome_select')} - removing it is not a revert"
pre = {k: v for k, v in base.items() if k != "genome_select" and not k.startswith("gs_")}

out = [{"id": "DEP_dup", "params": dict(base)},
       {"id": "C6_pre", "params": dict(pre)},
       {"id": "C6_pre_dup", "params": dict(pre)}]
for c in out:
    diff = sorted(k for k in set(base) | set(c["params"]) if c["params"].get(k) != base.get(k))
    print(f"{c['id']:12s} differs from BASE in {len(diff)} keys: {diff[:6]}")
    if c["id"] == "DEP_dup":
        assert diff == [], "DEP_dup must be a byte-identical copy of BASE"
    else:
        assert diff, f"{c['id']} is inert vs BASE (rule 7)"
json.dump(out, open(f'{tree}/experiments/par_dephead.json', 'w'), indent=1)
sha_c = hashlib.sha256(open(f'{tree}/best_controller.py', 'rb').read()).hexdigest()[:12]
sha_p = hashlib.sha256(open(f'{tree}/best_controller/params.json', 'rb').read()).hexdigest()[:12]
print(f"wrote par_dephead.json: {len(out)} arms | controller sha256[:12]={sha_c} | params sha256[:12]={sha_p}")
assert sha_c == "252f0ba1e060", f"tree controller is {sha_c}, not the running container's 252f0ba1e060 - STOP"
PY
test $? -eq 0 || { echo "GUARD FAILED - not launching"; exit 1; }

tmux kill-session -t "$LANE" 2>/dev/null || true
tmux new-session -d -s "$LANE" "cd $TREE/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && export NAC_RESULTS=$TREE/results && /opt/nacv/bin/python sched.py --lane $LANE --candidates par_dephead.json --seeds $SEEDS --holdout $SEEDS --stages ${NAC_STAGE:-160}:18000 --workers ${1:-56} --topk 4 --tol 0.0 --minutes 180 2>&1 | tee run_$LANE.log"
sleep "${2:-45}"
tmux ls 2>/dev/null | tr '\n' ' '; echo
head -6 "$TREE/experiments/run_$LANE.log" 2>/dev/null
tail -2 "$TREE/experiments/run_$LANE.log" 2>/dev/null
uptime
