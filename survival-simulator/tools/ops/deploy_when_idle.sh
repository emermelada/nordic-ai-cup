#!/usr/bin/env bash
# deploy_when_idle.sh - install the SELECTION controller (en_top3) in the first clean gap between
# the user's queued validations, never mid-run.
#
# WHY THIS IS A CODE DEPLOY, not params-only: the breeder-selection machinery (gs_*) exists only in the
# working-tree controller (86 params). The confirmation measured en_top3 at +19.2% over 40 unseen paired
# seeds (W27/L13), floor p10 +342, worst case 5,050 vs 313 ticks - and its vision-dominant control also
# gained +17.4%, so what pays is ACTIVE SELECTION, not the specific trait. Effect size is ~6x the
# measured A/A noise floor (-3.0%).
#
# SAFETY: the live artifact is backed up inside the container first (code AND params), so rollback is
# two docker cp commands. A restart takes ~8s, so we only proceed when no graded request has arrived for
# IDLE_S seconds - the user has multiple validations queued and corrupting one costs real score.
set -u
KEY="$HOME/.ssh/vps_hermes"
HOST="root@94.237.34.245"
CONTAINER="nac-survival-vps"
IDLE_S="${1:-120}"
MAX_WAIT_S="${2:-600}"
COPIES="${3:-2}"

age_now() {
  ssh -i "$KEY" "$HOST" "docker exec $CONTAINER sh -c 'tail -1 /app/predict_log.jsonl'" 2>/dev/null | python3 -c "
import json,sys,time
try:
    r=json.loads(sys.stdin.read()); print(int(time.time()-r['t']))
except Exception:
    print(-1)"
}

echo "waiting for a >=${IDLE_S}s idle gap (max ${MAX_WAIT_S}s)..."
waited=0
while [ "$waited" -lt "$MAX_WAIT_S" ]; do
  a=$(age_now)
  echo "  t+${waited}s: last graded request ${a}s ago"
  if [ "$a" -ge "$IDLE_S" ]; then
    echo "-> IDLE GAP FOUND. Deploying."
    break
  fi
  sleep 15
  waited=$((waited + 15))
done
if [ "$waited" -ge "$MAX_WAIT_S" ]; then
  echo "NO IDLE GAP in ${MAX_WAIT_S}s - NOT deploying (the campaign is still running)."
  exit 0
fi

set -e
echo "=== 1. back up the LIVE code + params inside the container ==="
ssh -i "$KEY" "$HOST" "docker cp $CONTAINER:/app/best_controller.py /root/code_backup_f90cb4e3.py && docker cp $CONTAINER:/app/best_controller/params.json /root/params_backup_c6.json && echo '  backups written: /root/code_backup_f90cb4e3.py and /root/params_backup_c6.json'"

echo "=== 2. install the selection controller + params ==="
ssh -i "$KEY" "$HOST" "docker cp /tmp/gs_best_controller.py $CONTAINER:/app/best_controller.py && docker cp /tmp/DEPLOY_GS_params.json $CONTAINER:/app/best_controller/params.json && docker restart $CONTAINER >/dev/null && sleep 8 && echo '  installed and restarted'"

echo "=== 3. verify from INSIDE the running container ==="
ssh -i "$KEY" "$HOST" "docker exec $CONTAINER sha256sum /app/best_controller.py; docker exec $CONTAINER cat /app/best_controller/params.json" | python3 -c "
import json,sys
line=sys.stdin.read().split('\n',1)
print('  code hash:', line[0].split()[0][:16])
p=json.loads(line[1])
print('  params:', len(p), 'keys')
for k in ('genome_select','gs_w_energy','gs_w_vision','gs_topk','reserve_frac','evade_mode','blind_explore_frac'):
    print(f'    {k:20s} {p.get(k)}')
ok = p.get('genome_select')==1.0 and p.get('gs_w_energy')==3.0 and p.get('reserve_frac')==0.0
print('  SELECTION CONTROLLER LIVE' if ok else '  UNEXPECTED PARAMS - ROLL BACK')"

echo "=== 4. live checks ==="
ssh -i "$KEY" "$HOST" "for i in 1 2 3; do curl -s -o /dev/null -w '  predict %{time_total}s -> %{http_code}\n' --max-time 10 -X POST localhost:9052/predict -H 'Content-Type: application/json' -d '{\"game_status\":\"RUNNING\",\"score\":0.0,\"agent_status\":[{\"agent_id\":0,\"energy\":300.0,\"biome\":\"grassland\",\"age\":10.0,\"speed\":10.0,\"sprint_speed\":20.0,\"hearing_radius\":60.0,\"vision_angle\":1.5,\"vision_range\":300.0,\"max_energy\":500.0,\"observations\":[]}]}'; done; curl -s -o /dev/null -w '  public https -> %{http_code} in %{time_total}s\n' --max-time 15 https://survival.zaitzev.com/"
echo "ROLLBACK if needed: docker cp /root/code_backup_f90cb4e3.py $CONTAINER:/app/best_controller.py && docker cp /root/params_backup_c6.json $CONTAINER:/app/best_controller/params.json && docker restart $CONTAINER"
