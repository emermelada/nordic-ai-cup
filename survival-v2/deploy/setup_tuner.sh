#!/usr/bin/env bash
# Copy the simulator + tuner to the VPS and start CMA-ES there at low CPU priority.
#
#   bash deploy/setup_tuner.sh <ip> <workers> [start_params.json]
#
# The agent server (systemd unit "surv") keeps priority: the tuner runs under nice 15 and leaves 2 cores free.
set -euo pipefail
IP=$1
WORKERS=$2
START=${3:-}
KEY=~/.ssh/surv_vps
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new root@$IP"
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/sv"
( cd "$HERE" && tar --exclude=.venv --exclude=runs --exclude='*.jsonl' --exclude='*.txt' --exclude=__pycache__ -cf - \
    fastsim.py hive.py bench.py tune.py official ) | ( cd "$TMP/sv" && tar -xf - )
find "$TMP/sv" -name '*.py' -exec sed -i 's/\r$//' {} +
( cd "$TMP" && tar -czf sv.tgz sv )
scp -i $KEY -q "$TMP/sv.tgz" root@$IP:/opt/
rm -rf "$TMP"
EXTRA=""
if [ -n "$START" ]; then
  scp -i $KEY -q "$START" root@$IP:/opt/sv/start_params.json
  EXTRA="--start start_params.json"
fi
$SSH "set -e; cd /opt && rm -rf sv && tar -xzf sv.tgz && cd sv && \
  (test -d .venv || python3 -m venv .venv) && .venv/bin/pip install -q numpy scipy shapely pygame-ce numba cma orjson && \
  cp hive.py hive_tune.py && mkdir -p runs/tune && \
  HIVE_MODULE=hive_tune nohup nice -n 15 .venv/bin/python tune.py --workers $WORKERS --seeds-per-gen 24 --popsize 12 \
     --gens 400 --sigma 0.12 --holdout 5001-5048 $EXTRA --out runs/tune > runs/tune/stdout.txt 2>&1 &
  sleep 5; tail -3 runs/tune/stdout.txt"
echo "tuner started on $IP; progress: ssh -i $KEY root@$IP tail -5 /opt/sv/runs/tune/stdout.txt"
