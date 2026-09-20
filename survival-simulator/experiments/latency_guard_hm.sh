#!/usr/bin/env bash
# Latency guard for the GRADED box. The ML lane must never cost the score.
# Samples /predict 5x every 20 s; if any sample exceeds 30 ms the lane is killed immediately.
# (Serving baseline is ~2 ms; 30 ms is ~15x headroom and far below the per-tick budget.)
set -u
LIM=0.030
while true; do
  m=0
  for i in 1 2 3 4 5; do
    t=$(curl -s -o /dev/null -w '%{time_total}' -X POST http://127.0.0.1:9052/predict \
        -H 'Content-Type: application/json' -d '{}' 2>/dev/null || echo 9)
    m=$(awk -v a="$m" -v b="$t" 'BEGIN{print (b>a)?b:a}')
  done
  if awk -v x="$m" -v l="$LIM" 'BEGIN{exit !(x>l)}'; then
    echo "$(date -u +%H:%M:%S) LATENCY ${m}s > ${LIM}s - KILLING the ML lane to protect serving"
    for p in $(pgrep -f "experiments/mech_ab.py"); do kill -TERM "$p" 2>/dev/null; done
    sleep 3
    for p in $(pgrep -f "spawn_main"); do cwd=$(readlink /proc/$p/cwd 2>/dev/null); \
      [ "$cwd" = "/opt/nac_hm" ] && kill -9 "$p" 2>/dev/null; done
    echo "$(date -u +%H:%M:%S) lane killed; guard exiting"
    exit 0
  fi
  echo "$(date -u +%H:%M:%S) predict max ${m}s (limit ${LIM}s) ok"
  sleep 20
done
