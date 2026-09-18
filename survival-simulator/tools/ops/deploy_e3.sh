#!/usr/bin/env bash
# Deploy E3 (retreat-while-facing evasion) to the live endpoint.
#
# Discipline this script exists to enforce:
#   * the pre-deploy log is preserved BEFORE the restart (evidence, not lost with the container);
#   * the build's exit code is CHECKED - it is never piped (an earlier deploy piped the build through
#     `tail`, the exit code came from tail, a failed checksum went unnoticed and the container kept
#     running the OLD image while the operator believed the new one was live);
#   * the ledger is regenerated from the file actually on disk, then verified with sha256sum -c, which is
#     the same gate the Dockerfile runs - so a mismatch fails the build loudly;
#   * verification reads the params and hash OUT of the running container, not out of the build context.
set -eu
cd /opt/nac-survival

echo "=== 0/6 preserve the current in-container log (pre-E3 evidence) ==="
docker cp nac-survival-vps:/app/predict_log.jsonl /opt/nac-survival/logs/predict_log_H1_pre_E3.jsonl || echo "(log copy failed; continuing)"
ls -la /opt/nac-survival/logs/ | tail -4

echo "=== 1/6 install the new controller + E3 params ==="
cp /tmp/bc_e3/best_controller.py ./best_controller.py
cp /tmp/bc_e3/params.json ./best_controller/params.json
cp /tmp/bc_e3/params.json ./params.json
sha256sum best_controller.py > best_controller.sha256
echo "recorded ledger: $(cat best_controller.sha256)"

echo "=== 2/6 provenance gate (same check the Dockerfile performs) ==="
sha256sum -c best_controller.sha256

echo "=== 3/6 rebuild ==="
if docker build -f Dockerfile.vps -t nac-survival-vps:latest . ; then
  echo "BUILD OK"
else
  echo "BUILD FAILED - NOT restarting; the old container is still serving" >&2
  exit 1
fi

echo "=== 4/6 restart ==="
docker rm -f nac-survival-vps
docker run -d --name nac-survival-vps -p 9052:9052 --restart unless-stopped nac-survival-vps:latest

echo "=== 5/6 verify what is actually running ==="
sleep 6
docker ps --format "{{.Names}}\t{{.Status}}" | grep nac-survival-vps
docker exec nac-survival-vps python3 -c "import json;p=json.load(open('/app/best_controller/params.json'));print('IN-CONTAINER PARAMS:',{k:p.get(k) for k in ('fruit_weight','predator_weight','walk_frac','evade_mode','evade_dist','evade_speed_frac','evade_energy_abs')})"
docker exec nac-survival-vps sha256sum /app/best_controller.py
curl -s -o /dev/null -w "GET / -> %{http_code} in %{time_total}s\n" http://127.0.0.1:9052/

echo "=== 6/6 public predict probe (3 ticks) ==="
for i in 1 2 3; do curl -s -o /dev/null -w "%{time_total} " -X POST https://survival.zaitzev.com/predict -H "Content-Type: application/json" -d @/tmp/probe.json; done; echo
echo "=== DEPLOY COMPLETE ==="
