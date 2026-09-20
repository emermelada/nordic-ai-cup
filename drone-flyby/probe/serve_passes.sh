#!/usr/bin/env bash
# Switch the detector stack. The camera is left as it is.
#   bash probe/serve_passes.sh 4    v4@960 v6@1280 v8@1280 v8@2560           (0.5270 baseline)
#   bash probe/serve_passes.sh 5    + v9@1280                                (0.5391 best single run)
#   bash probe/serve_passes.sh 6    + v9@2560 as well                        (untested)
set -eu
cd /workspace/drone-flyby
N=${1:?usage: serve_passes.sh <4|5|6>}
case "$N" in
  4) ALT='models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt'; SZ='960,1280,1280,2560' ;;
  5) ALT='models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt'; SZ='960,1280,1280,2560,1280' ;;
  6) ALT='models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-v8.pt,models/drone-yolo11m-p2-v9.pt,models/drone-yolo11m-p2-v9.pt'; SZ='960,1280,1280,2560,1280,2560' ;;
  *) echo "pick 4, 5 or 6"; exit 1 ;;
esac
sed -i "s|^export DRONE_MODEL_ALT=.*|export DRONE_MODEL_ALT=$ALT|; s|^export DRONE_IMGSZ=.*|export DRONE_IMGSZ=$SZ|" data/serve_env.sh
for p in $(ps -eo pid,cmd --no-headers | grep -E 'python3 api.py|bash watch' | grep -v grep | awk '{print $1}'); do kill "$p" 2>/dev/null || true; done
sleep 4
( set -a; . ./data/serve_env.sh; set +a; setsid --fork python3 api.py > data/serve.log 2>&1 < /dev/null )
sleep 110
curl -s localhost:10200/api | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ("models_loaded","models_requested","imgsz","camera")})'
( set -a; . ./data/serve_env.sh; set +a; setsid --fork bash watchdog.sh > /dev/null 2>&1 < /dev/null )
