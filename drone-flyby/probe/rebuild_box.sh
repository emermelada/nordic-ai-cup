#!/usr/bin/env bash
# Rebuild the whole serving setup on a fresh rented box, in one command.
#
#   probe/rebuild_box.sh -p 55010 root@93.91.156.104
#
# Everything here was learned the hard way on 19-20 Sep and is why it looks
# defensive: never `pkill -f` over ssh (it matches the ssh session's own
# command line and kills the shell mid-command), always detach with
# `setsid --fork`, and keep one file that says what is served.
set -eu
cd "$(dirname "$0")/.."
SSH_ARGS="$*"
HOST="${SSH_ARGS##* }"
# scp wants -P for the port, and must NOT be handed the host as an argument.
PORT=$(printf '%s' "$SSH_ARGS" | tr ' ' '\n' | grep -A1 -x -- '-p' | tail -1)
SCP_ARGS=${PORT:+-P $PORT}
KEY=${KEY:-$HOME/.ssh/vast_medical}
REMOTE=/workspace/drone-flyby
say() { printf '\n== %s\n' "$*"; }

say "reachable, and is it a GPU box?"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 $SSH_ARGS \
  'hostname; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; df -h /workspace | tail -1'

say "gate: latency and sustained bandwidth to the evaluator (Hetzner Helsinki)"
ssh -i "$KEY" $SSH_ARGS 'python3 - <<"PY"
import socket, time, statistics, urllib.request, sys
lat=[]
for _ in range(20):
    s=socket.socket(); s.settimeout(3); t=time.perf_counter()
    try: s.connect(("hel1-speed.hetzner.com",443)); lat.append((time.perf_counter()-t)*1000)
    except Exception: pass
    s.close(); time.sleep(0.2)
lat.sort(); med=statistics.median(lat); p95=lat[int(.95*len(lat))]
print(f"   RTT median {med:.1f} ms  p95 {p95:.1f} ms")
if med > 35: print("   WARNING: over 35 ms -- camera commands will miss deadlines (0.05-0.08 of score)")
for i in range(2):
    t=time.perf_counter(); n=0
    with urllib.request.urlopen("https://hel1-speed.hetzner.com/100MB.bin", timeout=60) as r:
        while True:
            b=r.read(1<<20)
            if not b: break
            n+=len(b)
            if time.perf_counter()-t>8: break
    print(f"   download {n/1e6/(time.perf_counter()-t):.1f} MB/s")
PY'

say "free the GPU if the template left vLLM running (it holds ~28 GB)"
ssh -i "$KEY" $SSH_ARGS 'supervisorctl stop vllm 2>/dev/null | tail -1 || true
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  if ps -p "$p" -o args --no-headers 2>/dev/null | grep -qi vllm; then kill "$p" 2>/dev/null || true; fi
done
sleep 6; nvidia-smi --query-gpu=memory.used --format=csv,noheader'

say "dependencies"
ssh -i "$KEY" $SSH_ARGS 'pip -q install fastapi "uvicorn[standard]" ultralytics 2>&1 | tail -1
python3 -c "import fastapi, uvicorn, cv2, torch, ultralytics; print(\"deps ok, cuda\", torch.cuda.is_available())"'

say "code, payloads and weights"
ssh -i "$KEY" $SSH_ARGS "mkdir -p $REMOTE/probe $REMOTE/models $REMOTE/data/probe $REMOTE/data/captures"
scp -i "$KEY" -q $SCP_ARGS dtos.py utils.py flyby.py api.py "$HOST:$REMOTE/"
scp -i "$KEY" -q $SCP_ARGS probe/*.py probe/*.sh "$HOST:$REMOTE/probe/"
scp -i "$KEY" -q $SCP_ARGS data/probe/*.json "$HOST:$REMOTE/data/probe/"
for m in drone-yolo11n-v4 drone-yolo11s-v6 drone-yolo11m-v8 drone-yolo11m-p2-v9; do
  printf '   uploading %s ... ' "$m"
  scp -i "$KEY" -q $SCP_ARGS "models/$m.pt" "$HOST:$REMOTE/models/" && echo done
done

say "config, services and watchdog"
ssh -i "$KEY" $SSH_ARGS "cd $REMOTE && cat > data/serve_env.sh <<'ENV'
export DRONE_PORT=10200
export DRONE_CAMERA=${CAMERA:-full}
export DRONE_BOX_GROW=1.3
export DRONE_BOX_GROW_CAP=1.3
export DRONE_MODEL=models/drone-yolo11n-v4.pt
export DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt
export DRONE_IMGSZ=960,1280,1280,2560,1280
export DRONE_DEVICE=cuda
export DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10
ENV
cat > watchdog.sh <<'WD'
#!/usr/bin/env bash
cd /workspace/drone-flyby
while true; do
  curl -sf -m 5 localhost:10100/api > /dev/null 2>&1 || {
    echo \"\$(date -Is) probe down\" >> data/watchdog.log
    setsid --fork env PROBE_PORT=10100 PROBE_JOBS=data/probe/jobs.json PROBE_OUT=data/captures \
      python3 probe/probe_server.py >> data/probe/server.log 2>&1 < /dev/null; }
  curl -sf -m 5 localhost:10200/api > /dev/null 2>&1 || {
    echo \"\$(date -Is) pipeline down\" >> data/watchdog.log
    ( set -a; . ./data/serve_env.sh; set +a; setsid --fork python3 api.py >> data/serve.log 2>&1 < /dev/null ); }
  sleep 20
done
WD
chmod +x watchdog.sh
for p in \$(ps -eo pid,cmd --no-headers | grep -E 'python3 (api|probe)|bash watch' | grep -v grep | awk '{print \$1}'); do kill \$p 2>/dev/null || true; done
sleep 3
( set -a; . ./data/serve_env.sh; set +a; setsid --fork python3 api.py > data/serve.log 2>&1 < /dev/null )
setsid --fork env PROBE_PORT=10100 PROBE_JOBS=data/probe/jobs.json PROBE_OUT=data/captures python3 probe/probe_server.py > data/probe/server.log 2>&1 < /dev/null
sleep 100
( set -a; . ./data/serve_env.sh; set +a; setsid --fork bash watchdog.sh > /dev/null 2>&1 < /dev/null )
echo '--- pipeline:'; curl -s localhost:10200/api | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in (\"models_loaded\",\"models_requested\",\"imgsz\",\"camera\",\"box_grow_cap\",\"new_track_confidence\")})'
echo '--- probe:';    curl -s localhost:10100/api | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"next job:\", d[\"next_jobs\"][0] if d[\"next_jobs\"] else None)'
echo '--- ports mapped by this template:'; env | grep ^VAST_TCP_PORT_ | sort"

say "done. Submit http://<public-ip>:<public port for 10200>  (and the probe server is on 10100)"
