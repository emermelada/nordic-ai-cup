#!/usr/bin/env bash
FILTER='gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh'
EDGE=172.67.223.65   # Cloudflare edge for survival.zaitzev.com (from DNS)
echo "== 1. home network RTT to Cloudflare (1.1.1.1) =="
ping -c 15 -q 1.1.1.1 2>&1 | tail -3
echo
echo "== 2. home network RTT to the Cloudflare edge serving our host =="
ping -c 15 -q $EDGE 2>&1 | tail -3
echo
echo "== 3. localhost baseline (origin only, no network) =="
for i in 1 2 3; do
  curl -s -o /dev/null -w '  total=%{time_total}s\n' --max-time 5 -X POST http://localhost:9052/predict \
    -H 'content-type: application/json' -d '{"game_status":"ok","score":0,"sim_time":0,"n_agents":0,"agent_status":[]}'
done
echo
echo "== 4. public path breakdown (DNS->TCP->TLS->TTFB->total) =="
for i in 1 2 3 4 5; do
  curl -s -o /dev/null --max-time 20 -H 'User-Agent: Mozilla/5.0' -H 'content-type: application/json' \
    -X POST https://survival.zaitzev.com/predict -d '{"game_status":"ok","score":0,"sim_time":0,"n_agents":0,"agent_status":[]}' \
    -w '  dns=%{time_namelookup} conn=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total}\n'
done
echo
echo "== 5. tailscale funnel status (alternative exposure) =="
tailscale funnel status 2>&1 | head -8 | grep -vE "$FILTER" || echo "(no funnel configured)"