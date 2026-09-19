#!/usr/bin/env bash
# run_scan_test.sh - 40-seed paired test of the scan rule, ON THE SERVING BOX but in its own tree,
# with the latency guard armed: if /predict p95 crosses 30 ms the lane is killed automatically.
set -u
ssh -i /Users/zaitzev/.ssh/vps_hermes root@94.237.34.245 'rm -rf /opt/nac_scan && cp -r /opt/nac /opt/nac_scan'
scp -q -i /Users/zaitzev/.ssh/vps_hermes /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/best_controller.py root@94.237.34.245:/opt/nac_scan/best_controller.py
scp -q -i /Users/zaitzev/.ssh/vps_hermes /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments/sched.py root@94.237.34.245:/opt/nac_scan/experiments/sched.py
scp -q -i /Users/zaitzev/.ssh/vps_hermes /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/tools/ops/latency_guard.sh root@94.237.34.245:/opt/nac_scan/experiments/latency_guard.sh
ssh -i /Users/zaitzev/.ssh/vps_hermes root@94.237.34.245 'cd /opt/nac_scan/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python - <<PY
import json, sys
sys.path.insert(0, "/opt/nac_scan"); sys.path.insert(0, "/opt/nac_scan/experiments")
import best_controller as bc
base = json.load(open("/opt/nac/best_controller/params.json"))
out = [{"id": "BASE_deployed", "params": dict(base)},
       {"id": "BASE_dup", "params": dict(base)}]
p = dict(base); p["scan_mode"] = 1.0
out.append({"id": "SCAN_sweep", "params": p})
json.dump(out, open("/opt/nac_scan/experiments/par_scan.json", "w"), indent=1)
print("candidates:", [c["id"] for c in out])
PY'
ssh -i /Users/zaitzev/.ssh/vps_hermes root@94.237.34.245 'cd /opt/nac_scan/experiments && tmux new-session -d -s scan1 "cd /opt/nac_scan/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python sched.py --lane scan1 --candidates par_scan.json --seeds 4400-4439 --holdout 4440-4479 --stages 20:18000,40:18000 --workers 10 --topk 4 --tol 0.02 2>&1 | tee run_scan1.log"; sleep 8; tmux new-session -d -s guard_scan "cd /opt/nac_scan/experiments && bash latency_guard.sh 30 scan1 9052"; sleep 20; tmux ls; tail -3 run_scan1.log'
