#!/usr/bin/env bash
# Benchmark cloudflared protocol/edge variants against the live named tunnel hostname.
# NOTE: this kills the current tunnel process (incl. the up.sh/caffeinate wrapper) and restarts it.
FILTER='gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh'
U="https://survival.zaitzev.com/predict"
BODY='{"game_status":"ok","score":0,"sim_time":0,"n_agents":0,"agent_status":[]}'

measure() {  # $1 label
  local times=""
  for j in $(seq 1 15); do
    t=$(curl -s -o /dev/null -w '%{time_total}' --max-time 25 -H 'User-Agent: Mozilla/5.0' \
          -H 'content-type: application/json' -X POST "$U" -d "$BODY")
    times="$times $t"
  done
  echo "$times" | tr ' ' '\n' | grep -E '^[0-9]' | sort -n | awk -v L="$1" \
    '{a[NR]=$1*1000;s+=$1*1000} END{printf "  %-16s median=%.0fms  mean=%.0fms  min=%.0fms  max=%.0fms  (n=%d)\n", L, a[int((NR+1)/2)], s/NR, a[1], a[NR], NR}'
}

start_cf() {  # $1 proto  $2 ipver(optional)
  pkill -f "cloudflared tunnel run" 2>/dev/null; pkill -f "caffeinate" 2>/dev/null; sleep 2
  : > /tmp/cf_test.log
  if [ -n "$2" ]; then
    nohup cloudflared tunnel run --protocol "$1" --edge-ip-version "$2" nac-survival >/tmp/cf_test.log 2>&1 &
  else
    nohup cloudflared tunnel run --protocol "$1" nac-survival >/tmp/cf_test.log 2>&1 &
  fi
  disown 2>/dev/null
  for i in $(seq 1 40); do grep -q "Registered tunnel connection" /tmp/cf_test.log 2>/dev/null && break; sleep 1; done
  sleep 2
}

echo "== cloudflared variant benchmark (15 POSTs each) =="
for v in "quic auto" "http2 auto" "quic 4" "http2 4"; do
  set -- $v
  start_cf "$1" "$2"
  if grep -q "Registered tunnel connection" /tmp/cf_test.log 2>/dev/null; then
    measure "$1 ipv=$2"
  else
    echo "  $1 ipv=$2: FAILED to connect"; tail -3 /tmp/cf_test.log | grep -vE "$FILTER"
  fi
done

echo
echo "== restoring default (quic) tunnel =="
start_cf quic ""
grep -q "Registered tunnel connection" /tmp/cf_test.log && echo "  tunnel up (quic)" || echo "  WARN: tunnel not up"
curl -s -o /dev/null -w '  public GET / -> %{http_code}\n' --max-time 15 https://survival.zaitzev.com/