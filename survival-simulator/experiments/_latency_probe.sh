#!/usr/bin/env bash
# Measure per-request round-trip latency through the PUBLIC path (Cloudflare -> tunnel -> Mac).
BODY='{"game_status":"ok","score":0.0,"sim_time":0.0,"n_agents":0,"agent_status":[]}'
U="https://survival.zaitzev.com/predict"
N=40
times=()
for i in $(seq 1 $N); do
  t=$(curl -s -o /dev/null -w '%{time_total}' --max-time 20 -H 'User-Agent: Mozilla/5.0' \
        -H 'content-type: application/json' -X POST "$U" -d "$BODY")
  times+=("$t")
done
printf '%s\n' "${times[@]}" | sort -n | awk '
  {a[NR]=$1*1000; s+=$1*1000}
  END{
    n=NR
    printf "requests=%d  min=%.1fms  median=%.1fms  p90=%.1fms  max=%.1fms  mean=%.1fms\n", n, a[1], a[int((n+1)/2)], a[int(n*0.9)], a[n], s/n
  }'
echo "(full path: this Mac -> Cloudflare edge -> cloudflared tunnel -> home Mac) "