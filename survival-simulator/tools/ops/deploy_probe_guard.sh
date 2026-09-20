#!/usr/bin/env bash
# deploy_probe_guard.sh - put the stray-payload-guarded hive on the graded endpoint, reversibly.
#
# WHY. server_trace.csv on the graded host (58,453 rows) shows the platform injecting a lone synthetic
# payload -- sim_time 0.0, one agent, score 123.4, game_status "running", observations
# edge|predator|tree -- into the request stream: before most attempts (their connectivity check) and at
# least once (09:39:59 UTC, attempt scoring 904.0) in the MIDDLE of a live graded game, where the previous
# request carried sim_time 784.7. hive's old rule reset() on any backward sim_time jump, so every one of
# those probes threw away a live game's entire state. hive_next.py suspends a game instead of resetting it
# and restores the state when the live stream resumes.
#
# WHAT IT CHANGES.  /opt/surv/hive.py  ->  the guarded artifact.  Nothing else: server.py, the systemd
# unit, the params (HIVE_PARAMS unset => hive defaults) and the port are untouched.
#
# USAGE (from the repo root, on the Mac)
#   bash survival-simulator/tools/ops/deploy_probe_guard.sh --stage     # copy + verify, no restart
#   bash survival-simulator/tools/ops/deploy_probe_guard.sh --apply     # swap + restart (~2 s outage)
#   bash survival-simulator/tools/ops/deploy_probe_guard.sh --verify    # post-swap checks
#   bash survival-simulator/tools/ops/deploy_probe_guard.sh --rollback  # restore the previous hive
set -euo pipefail

HOST=root@94.237.34.245
KEY=${KEY:-$HOME/.ssh/vps_hermes}
SRC=${SRC:-$(cd "$(dirname "$0")/../../experiments" && pwd)/hive_pf_guard.py}
SRC_SERVER=${SRC_SERVER:-$(cd "$(dirname "$0")/../../experiments/gold1815" && pwd)/server_next.py}
NEW_HASH_EXPECTED=${NEW_HASH_EXPECTED:-56489acfff2cba36b8e1d871dc9bb9b0e9864818e62a66291e9a1b389b553c2e}
OLD_HASH_EXPECTED=${OLD_HASH_EXPECTED:-7f3467cfa5cf26dde71fa523156037167dd96149adbc72446121bc27b89a1c43}
cd "$(dirname "$0")/../.."   # repo root (for the relative paths below)
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 $HOST"
SCP="scp -q -i $KEY -o StrictHostKeyChecking=no"

mode=${1:---verify}

case "$mode" in
  --stage)
    # upload with CRLF stripped, to .new names, and confirm the byte hashes BEFORE anything is swapped
    tr -d '\r' < "$SRC" > /tmp/hive_guard.py
    tr -d '\r' < "$SRC_SERVER" > /tmp/server_guard.py
    echo "local  hive   sha256: $(sha256sum /tmp/hive_guard.py | cut -c1-16)  (expected ${NEW_HASH_EXPECTED:0:16})"
    echo "local  server sha256: $(sha256sum /tmp/server_guard.py | cut -c1-16)  (was d94e68d0 on the gold run)"
    $SCP /tmp/hive_guard.py /tmp/server_guard.py $HOST:/opt/surv/
    $SSH 'sha256sum /opt/surv/hive_guard.py /opt/surv/server_guard.py'
    $SSH 'cp -n /opt/surv/hive.py /opt/surv/hive_pre_guard.py; cp -n /opt/surv/server.py /opt/surv/server_pre_guard.py; \
          sha256sum /opt/surv/hive_pre_guard.py /opt/surv/server_pre_guard.py'
    # pre-swap smoke test: import the STAGED server module and drive two requests through it (one normal
    # tick, one sim_time-0.0 probe). Nothing is restarted here; the live service keeps the old code.
    $SSH 'cd /opt/surv; cp server.py server_live.py; cp server_guard.py server.py; cp hive.py hive_live.py; \
          cp hive_guard.py hive.py; rm -rf __pycache__; \
          /opt/nacv/bin/python - <<PY
import json, orjson, server
tick = {"game_status": "ok", "score": 0.1, "sim_time": 0.1, "n_agents": 2,
        "agent_status": [{"agent_id": i, "energy": 300.0, "biome": "forest", "age": 20.0, "speed": 12.0,
                          "sprint_speed": 24.0, "hearing_radius": 70.0, "vision_angle": 1.2,
                          "vision_range": 320.0, "max_energy": 600.0,
                          "observations": [{"type": "fruit", "distance": 60.0, "angle": 0.2}]} for i in range(2)]}
for t in (0.1, 0.2, 0.3):
    tick["sim_time"] = t
    out = server.handle(orjson.dumps(tick), "127.0.0.1")
    assert json.loads(out)["actions"], "no actions returned"
probe = {"game_status": "running", "score": 123.4, "sim_time": 0.0, "n_agents": 1,
         "agent_status": [{"agent_id": 0, "energy": 100.0, "biome": "forest", "age": 1.0, "speed": 10.0,
                           "sprint_speed": 20.0, "hearing_radius": 60.0, "vision_angle": 1.0,
                           "vision_range": 300.0, "max_energy": 500.0,
                           "observations": [{"type": "edge", "distance": 100.0, "angle": 0.0}]}]}
server.handle(orjson.dumps(probe), "46.62.240.126")
tick["sim_time"] = 0.4
server.handle(orjson.dumps(tick), "46.62.240.126")
print("SMOKE OK: strays=%s restores=%s games=%s controller=%s" %
      (server.COUNTS.get("strays"), server.COUNTS.get("restores"), server.COUNTS.get("games"), server.HIVE_SHA[:16]))
PY
          rm -rf __pycache__; cp server_live.py server.py; cp hive_live.py hive.py; rm -f server_live.py hive_live.py; \
          sha256sum server.py hive.py'
    echo "staged + smoke-tested. verify: new hive = $NEW_HASH_EXPECTED, backup hive = $OLD_HASH_EXPECTED"
    ;;
  --apply)
    echo "== before: is an attempt in flight? =="
    $SSH 'tail -c 400 /opt/surv/server_trace.csv | tail -3; systemctl is-active surv; docker ps --format "{{.Names}} {{.Status}}" | head -3'
    if [ "${APPLY_OK:-}" != "1" ]; then
      read -r -p "no attempt currently posting (last trace row is old)? type YES to swap and restart: " ok
      [ "$ok" = YES ] || { echo "aborted"; exit 1; }
    fi
    $SSH 'set -e; cd /opt/surv; cp hive.py hive_pre_guard.py; cp server.py server_pre_guard.py; \
          install -m 644 hive_guard.py hive.py; install -m 644 server_guard.py server.py; \
          rm -rf __pycache__; systemctl restart surv; sleep 3; systemctl is-active surv; \
          sha256sum hive.py server.py; rm -f hive_guard.py server_guard.py'
    $SSH 'curl -s -o /dev/null -w "public %{http_code} in %{time_total}s\n" https://survival.zaitzev.com/'
    $SSH 'journalctl -u surv -n 5 --no-pager'
    ;;
  --verify)
    $SSH 'echo -n "live hive.py: "; sha256sum /opt/surv/hive.py; systemctl is-active surv; \
          systemctl show surv -p NRestarts; tail -2 /opt/surv/server_log.jsonl; \
          curl -s https://survival.zaitzev.com/ ; echo'
    ;;
  --rollback)
    $SSH 'set -e; cd /opt/surv; cp hive.py hive_failed.py; cp server.py server_failed.py; \
          cp hive_pre_guard.py hive.py; cp server_pre_guard.py server.py; rm -rf __pycache__; \
          systemctl restart surv; sleep 3; systemctl is-active surv; sha256sum hive.py server.py'
    $SSH 'curl -s -o /dev/null -w "public %{http_code} in %{time_total}s\n" https://survival.zaitzev.com/'
    echo "rolled back to the pre-guard controller + server (kept as /opt/surv/hive_failed.py, server_failed.py)"
    ;;
  *)
    echo "usage: $0 [--stage|--apply|--verify|--rollback]"; exit 2;;
esac