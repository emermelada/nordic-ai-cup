#!/usr/bin/env bash
# Hit every service's health + predict endpoint and print latency.
#   scripts/smoke-test.sh                      # local ports 8001-8003
#   scripts/smoke-test.sh https://xyz.trycloudflare.com   # one public URL
#   CONCURRENT=1 scripts/smoke-test.sh         # fire all at once (contention test)
set -u
payload=${PAYLOAD:-'{}'}

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=(http://localhost:8001 http://localhost:8002 http://localhost:8003)
fi

probe() {
  local base=$1
  local health predict
  health=$(curl -s -o /dev/null -w '%{http_code} %{time_total}s' --max-time 10 "$base/api")
  predict=$(curl -s -o /dev/null -w '%{http_code} %{time_total}s' --max-time 60 \
    -X POST "$base/predict" -H 'content-type: application/json' -d "$payload")
  printf '%-45s /api %-14s /predict %s\n' "$base" "$health" "$predict"
}

for t in "${targets[@]}"; do
  if [ "${CONCURRENT:-0}" = 1 ]; then probe "$t" & else probe "$t"; fi
done
wait
