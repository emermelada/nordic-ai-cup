#!/usr/bin/env bash
# latency_guard.sh [p95_threshold_ms] [tmux_session_to_kill]
#
# The 32-core box serves the live endpoint AND runs experiments - a deliberate decision to use both
# machines. That makes protecting serving latency a mechanical obligation, not a good intention:
# score ~= min(ticks, 600s / per-tick latency), so if a saturated CPU pushes p95 from 5 ms to 60 ms we
# lose score directly, and a validation running at the time would be wasted.
#
# This samples the endpoint every 30 s and KILLS the experiment session if p95 crosses the threshold.
# Killing loses nothing: results are content-addressed in results/cache.jsonl, so re-running the lane
# resumes from cache and only the in-flight episodes are repeated.
set -u
THRESH="${1:-30}"
SESS="${2:-neuro2}"
PORT="${3:-9052}"

echo "guarding session '$SESS' : kill if p95 > ${THRESH}ms (sampling ${PORT} every 30s)"
while true; do
  samples=$(for i in $(seq 1 20); do
              curl -s -o /dev/null -w "%{time_total}\n" --max-time 5 \
                -X POST "localhost:${PORT}/predict" -H 'Content-Type: application/json' \
                -d '{"game_status":"RUNNING","score":0.0,"agent_status":[]}'
            done)
  p95=$(printf '%s\n' "$samples" | sort -n | awk '{a[NR]=$1*1000} END{printf "%.1f", a[int(NR*0.95)]}')
  n=$(printf '%s\n' "$samples" | grep -c .)
  echo "$(date -u +%H:%M:%S) samples=$n p95=${p95}ms threshold=${THRESH}ms"
  if [ "$n" -lt 10 ]; then
    echo "endpoint not answering reliably ($n/20) -> killing experiments in '$SESS'"
    tmux kill-session -t "$SESS" 2>/dev/null
    break
  fi
  if awk "BEGIN{exit !($p95 > $THRESH)}"; then
    echo "SERVING DEGRADED: p95 ${p95}ms > ${THRESH}ms -> killing experiments in '$SESS'"
    tmux kill-session -t "$SESS" 2>/dev/null
    break
  fi
  sleep 30
done
