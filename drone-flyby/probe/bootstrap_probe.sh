#!/usr/bin/env bash
# Put the probe server (capture + replay) on a fresh rented box and start it.
#
#   probe/bootstrap_probe.sh -p 41333 root@1.2.3.4 9053
#
# Argument order: everything but the LAST argument is passed to ssh/scp, the
# last one is the internal port to bind (must be a port the template mapped).
set -eu
cd "$(dirname "$0")/.."
PORT="${@: -1}"
SSH_ARGS="${@:1:$(($#-1))}"
# scp wants -P for the port where ssh wants -p.
SCP_ARGS=$(printf '%s' "$SSH_ARGS" | sed 's/-p /-P /')
REMOTE=/workspace/drone-flyby

say() { printf '\n== %s\n' "$*"; }

say "reachable?"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $SSH_ARGS \
  'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU"; df -h /workspace | tail -1'

say "latency and sustained bandwidth to the evaluator (Hetzner Helsinki)"
ssh $SSH_ARGS 'python3 - <<"PY"
import socket, time, statistics, urllib.request
lat=[]
for _ in range(20):
    s=socket.socket(); s.settimeout(3); t=time.perf_counter()
    try: s.connect(("hel1-speed.hetzner.com",443)); lat.append((time.perf_counter()-t)*1000)
    except Exception: pass
    s.close(); time.sleep(0.2)
lat.sort()
print(f"   Helsinki RTT median {statistics.median(lat):.1f} ms  p95 {lat[int(.95*len(lat))]:.1f} ms  n={len(lat)}")
for i in range(2):
    t=time.perf_counter(); n=0
    with urllib.request.urlopen("https://hel1-speed.hetzner.com/100MB.bin", timeout=60) as r:
        while True:
            b=r.read(1<<20)
            if not b: break
            n+=len(b)
            if time.perf_counter()-t>10: break
    dt=time.perf_counter()-t; print(f"   download {n/1e6/dt:.1f} MB/s")
PY'

say "dependencies"
ssh $SSH_ARGS 'pip -q install fastapi "uvicorn[standard]" pydantic opencv-python-headless 2>&1 | tail -2; python3 -c "import fastapi, uvicorn, cv2, pydantic; print(\"deps ok\")"'

say "code and probe payloads"
ssh $SSH_ARGS "mkdir -p $REMOTE/probe $REMOTE/data/probe $REMOTE/data/captures"
scp -q $SCP_ARGS dtos.py utils.py flyby.py api.py "${SSH_ARGS##* }:$REMOTE/"
scp -q $SCP_ARGS probe/*.py "${SSH_ARGS##* }:$REMOTE/probe/"
scp -q $SCP_ARGS data/probe/*.json "${SSH_ARGS##* }:$REMOTE/data/probe/"

say "start the probe server on port $PORT"
ssh $SSH_ARGS "cd $REMOTE && pkill -f probe_server.py || true; sleep 1; \
  PROBE_PORT=$PORT PROBE_JOBS=data/probe/jobs.json PROBE_OUT=data/captures \
  nohup python3 probe/probe_server.py > data/probe/server.log 2>&1 &
  sleep 4; curl -s localhost:$PORT/api; echo"
