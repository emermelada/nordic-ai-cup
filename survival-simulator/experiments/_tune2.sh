#!/usr/bin/env bash
FILTER='gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh'
U="https://survival.zaitzev.com/predict"
BODY='{"game_status":"ok","score":0,"sim_time":0,"n_agents":0,"agent_status":[]}'

start_cf() {  # $1 = protocol
  pkill -f "cloudflared tunnel run" 2>/dev/null; sleep 3
  : > "/tmp/cf_$1.log"
  nohup cloudflared tunnel run --protocol "$1" nac-survival >"/tmp/cf_$1.log" 2>&1 &
  disown 2>/dev/null
  for i in $(seq 1 60); do grep -q "Registered tunnel connection" "/tmp/cf_$1.log" 2>/dev/null && break; sleep 1; done
  sleep 3
}
measure() {
  local times=""
  for j in $(seq 1 15); do
    t=$(curl -s -o /dev/null -w '%{time_total}' --max-time 25 -H 'User-Agent: Mozilla/5.0' \
          -H 'content-type: application/json' -X POST "$U" -d "$BODY")
    times="$times $t"
  done
  echo "$times" | tr ' ' '\n' | grep -E '^[0-9]' | sort -n | awk -v L="$1" \
    '{a[NR]=$1*1000;s+=$1*1000} END{printf "  %-8s median=%.0fms  mean=%.0fms  p90=%.0fms  min=%.0fms  max=%.0fms\n", L, a[int((NR+1)/2)], s/NR, a[int(NR*0.9)], a[1], a[NR]}'
}
up() { grep -q "Registered tunnel connection" "/tmp/cf_$1.log" 2>/dev/null && echo "  [$1 up]" || echo "  [$1 FAILED]"; }

echo "== A/B cloudflared protocol (15 POSTs each, same hostname) =="
start_cf quic;   up quic;   measure "quic"
start_cf http2;  up http2;  measure "http2"

echo
echo "== restoring quic (default) =="
start_cf quic; up quic
curl -s -o /dev/null -w '  public GET / -> %{http_code}\n' --max-time 15 https://survival.zaitzev.com/