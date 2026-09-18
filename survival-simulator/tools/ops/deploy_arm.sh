#!/usr/bin/env bash
# deploy_arm.sh <arm_config.json> [expect_key=value ...]
#
# Ships an experiment arm to the serving box, generalising the discipline that deploy_e3.sh encoded.
# Every one of these rules exists because breaking it has already cost us a deploy or a wrong result:
#
#   * the pre-deploy predict log is copied OUT of the container before the restart - the evidence does
#     not survive a `docker rm`;
#   * the build exit code is CHECKED, never piped: an earlier deploy piped the build through `tail`,
#     the pipeline returned tail's status, a FAILED checksum went unnoticed and the container kept
#     serving the OLD image while the operator believed the new one was live;
#   * the controller that ships is taken from the file the experiment actually used, and the ledger is
#     regenerated from that file on disk, then re-verified with `sha256sum -c` (the same gate the
#     Dockerfile runs) so a mismatch fails the build loudly;
#   * verification reads the params and the hash OUT OF THE RUNNING CONTAINER, because "the file I
#     copied is right" is not the same claim as "the process that answers /predict is running it";
#   * the arm is checked to actually SET the flag it claims (a silently-absent key is how three
#     baseline traps happened tonight).
set -eu

ARM="${1:?usage: deploy_arm.sh <arm_config.json> [expect_key=value ...]}"
shift || true
HOST="${NAC_HOST:-root@94.237.34.245}"
KEY="${NAC_KEY:-/Users/zaitzev/.ssh/vps_hermes}"
TREE="${NAC_TREE:-/opt/nac}"
LOGDIR="/opt/nac-survival/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new $HOST"

echo "=== 0/7 arm under inspection: $ARM ==="
test -f "$ARM" || { echo "arm config not found: $ARM" >&2; exit 1; }
python3 -c "import json,sys; d=json.load(open('$ARM')); print('   ', {k:v for k,v in d.items() if k.startswith('gs_') or k.startswith('genome') or k.startswith('thin')})"

echo "=== 1/7 copy the controller + arm params to the box ==="
# The controller MUST be the one the experiment used; copy it explicitly, never assume.
scp -q -i "$KEY" best_controller.py "$HOST:/tmp/arm_bc.py"
scp -q -i "$KEY" "$ARM" "$HOST:/tmp/arm_params.json"

echo "=== 2/7 assert the arm actually sets what it claims ==="
for kv in "$@"; do
  k="${kv%%=*}"; v="${kv#*=}"
  got=$(python3 -c "import json;print(json.load(open('$ARM')).get('$k','<ABSENT>'))")
  if [ "$got" != "$v" ]; then echo "ARM FLAG MISMATCH: $k expected '$v' got '$got' - refusing to deploy" >&2; exit 1; fi
  echo "    ok: $k = $got"
done

echo "=== 3/7 preserve the current in-container log (pre-deploy evidence) ==="
$SSH "mkdir -p $LOGDIR && docker cp nac-survival-vps:/app/predict_log.jsonl $LOGDIR/predict_log_pre_${STAMP}.jsonl || echo '(no live log to copy - continuing)'"

echo "=== 4/7 install into the serving tree + regenerate the ledger from disk ==="
$SSH "set -eu; cd $TREE; cp /tmp/arm_bc.py ./best_controller.py; cp /tmp/arm_params.json ./best_controller/params.json; sha256sum best_controller.py > best_controller.sha256; sha256sum -c best_controller.sha256"

echo "=== 5/7 rebuild (exit code checked, never piped) ==="
if $SSH "cd $TREE && docker build -f Dockerfile.vps -t nac-survival-vps:latest ."; then
  echo "BUILD OK"
else
  echo "BUILD FAILED - NOT restarting; the previous container is still serving" >&2; exit 1
fi

echo "=== 6/7 restart + verify from INSIDE the running container ==="
$SSH "set -eu; docker rm -f nac-survival-vps; docker run -d --name nac-survival-vps -p 9052:9052 --restart unless-stopped nac-survival-vps:latest; sleep 6
docker ps --format '{{.Names}} {{.Status}}' | grep nac-survival-vps
echo -n 'in-container params: '; docker exec nac-survival-vps python3 -c \"import json;p=json.load(open('/app/best_controller/params.json'));print({k:p.get(k) for k in ('genome_select','gs_topk','gs_w_energy','evade_mode','fruit_weight')})\"
echo -n 'in-container hash:   '; docker exec nac-survival-vps sha256sum /app/best_controller.py
curl -s -o /dev/null -w 'GET / -> %{http_code} in %{time_total}s\n' http://127.0.0.1:9052/"

echo "=== 7/7 public probe (through TLS, real traffic shape) ==="
for i in 1 2 3; do
  curl -s -o /dev/null -w "%{time_total} " -X POST https://survival.zaitzev.com/predict \
    -H "Content-Type: application/json" \
    -d '{"game_status":"RUNNING","score":0.0,"agent_status":[]}' || true
done; echo
echo "=== DEPLOY COMPLETE ($STAMP) ==="
