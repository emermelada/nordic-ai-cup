#!/usr/bin/env bash
# Start one sweep arm and refuse to hand it over unless it is really serving it.
#
#   tools/arm.sh control
#   tools/arm.sh cap145   DRONE_BOX_GROW_CAP=1.45
#   tools/arm.sh inspect  DRONE_INSPECT=1
#   tools/arm.sh floor    DRONE_FLOOR_ZERO=1
#   tools/arm.sh frames                          # count the last runs' frames
#
# Every failure this project has paid for looked like a healthy service: a stale
# image, a silently ignored DRONE_MODEL_ALT, a failed bind leaving the OLD
# config serving while every log line reads fine. So this reads /api back and
# exits non-zero rather than letting an attempt be spent on the wrong thing.
set -u
cd "$(dirname "$0")/.."

# python3 on a rented box, .venv/bin/python on the Mac.
PYTHON=${PYTHON:-.venv/bin/python}
PORT=${PORT:-9053}
export DRONE_PORT=$PORT
RECORD=${DRONE_RECORD_DIR:-data/recordings}

if [ "${1:-}" = frames ]; then
    # 249/249 or the score does not count. Two missing frames cost 0.026.
    for d in "$RECORD"/*/; do
        n=$(ls "$d"/*.png 2>/dev/null | wc -l)
        [ "$n" -gt 100 ] && printf '%s  %s%s\n' "$(basename "$d")" "$n" \
            "$( [ "$n" -ge 249 ] && echo '  complete' || echo '   <- SHORT, discard this run')"
    done | tail -8
    exit 0
fi

ARM=${1:?usage: tools/arm.sh <arm-name> [VAR=value ...] | frames}
shift

# Kill OUR service and nothing else. `pkill -x python3` would also take the
# rented box's Jupyter, and `pkill -f api.py` matches this script's own command
# line over ssh -- which killed a shell mid-command here earlier today.
if [ -f .serve.pid ] && kill -0 "$(cat .serve.pid)" 2>/dev/null; then
    kill "$(cat .serve.pid)" 2>/dev/null
fi
# Not `ss`/`lsof`: neither exists in the NGC container, and a missing tool
# piped into `grep -q` reports "free" forever -- a guard that always passes is
# worse than no guard. A connect attempt needs nothing but python.
held() { "$PYTHON" -c "
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)"; }
for _ in $(seq 30); do held || break; sleep 1; done
if held; then
    echo "FAIL  port $PORT is still held by something that is not ours." >&2
    echo "      Pick a different DRONE_PORT, or stop whatever owns it." >&2
    exit 1
fi

# The 0.4788 baseline. Arm overrides are appended, so they win.
# nohup and </dev/null: started over ssh, a plain `&` job takes SIGHUP when the
# session closes and the service dies between arms.
nohup env \
  DRONE_BOX_GROW=helsinki \
  DRONE_BOX_GROW_CAP=1.3 \
  DRONE_MODEL=models/drone-yolo11n-v4.pt \
  DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt \
  DRONE_IMGSZ=960,1280,1280 \
  DRONE_DEVICE="${DRONE_DEVICE:-cuda}" \
  DRONE_RECORD_DIR="$RECORD" \
  DRONE_SET=BOTH_MODELS=1 \
  "$@" \
  "$PYTHON" api.py > "serve-$ARM.log" 2>&1 < /dev/null &
echo $! > .serve.pid
disown 2>/dev/null || true

for _ in $(seq 180); do
    curl -sf "http://localhost:$PORT/api" > /dev/null 2>&1 && break
    sleep 1
done

echo "== arm: $ARM   overrides: ${*:-none}"
"$PYTHON" - "$PORT" "$ARM" <<'PY'
import json, sys, urllib.request
port, arm = sys.argv[1], sys.argv[2]
try:
    s = json.load(urllib.request.urlopen(f'http://localhost:{port}/api', timeout=10))
except Exception as exc:
    sys.exit(f'FAIL  /api unreachable: {exc}')
grow = s.get('box_grow') or {}
for k in ('models_loaded', 'models_requested', 'imgsz', 'device', 'box_grow_cap',
          'new_track_confidence', 'inspect', 'floor_zero', 'floor_size_tol',
          'miss_penalty', 'camera', 'recording'):
    print(f'   {k:22s} {s.get(k)}')
print(f'   {"box_grow":22s} {len(grow)} classes')
bad = []
if s.get('models_loaded') != s.get('models_requested'):
    bad.append(f'{s.get("models_loaded")} of {s.get("models_requested")} models loaded')
if len(grow) != 16:
    bad.append(f'box_grow covers {len(grow)} classes, expected 16')
if len(s.get('imgsz') or []) != s.get('models_requested'):
    bad.append(f'imgsz {s.get("imgsz")} does not match {s.get("models_requested")} models')
if not s.get('recording'):
    bad.append('not recording: you will not be able to count frames afterwards')
for line in bad:
    print(f'FAIL  {line}')
print('PASS  serving this arm' if not bad else '')
sys.exit(1 if bad else 0)
PY
