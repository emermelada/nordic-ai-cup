#!/usr/bin/env bash
# Re-serve with a different camera, keeping whatever stack serve_env.sh names.
#   bash probe/serve_cam.sh row0        bash probe/serve_cam.sh full
set -eu
cd /workspace/drone-flyby
CAM=${1:?usage: serve_cam.sh <camera>}
sed -i "s|^export DRONE_CAMERA=.*|export DRONE_CAMERA=$CAM|" data/serve_env.sh
for p in $(ps -eo pid,cmd --no-headers | grep -E 'python3 api.py|bash watch' | grep -v grep | awk '{print $1}'); do kill "$p" 2>/dev/null || true; done
sleep 4
( set -a; . ./data/serve_env.sh; set +a; setsid --fork python3 api.py >> data/serve.log 2>&1 < /dev/null )
sleep 100
curl -s localhost:10200/api | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ("models_loaded","imgsz","camera","box_grow_cap","new_track_confidence")})'
( set -a; . ./data/serve_env.sh; set +a; setsid --fork bash watchdog.sh > /dev/null 2>&1 < /dev/null )
