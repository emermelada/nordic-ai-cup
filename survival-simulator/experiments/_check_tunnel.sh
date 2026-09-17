#!/usr/bin/env bash
REPO="/Users/zaitzev/Life/Projects/nordic-ai-cup"
cd "$REPO" || exit 1
FILTER='gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh'
echo "== container =="
docker compose ps 2>&1 | grep -v -E "$FILTER" || true
echo "== origin :9052 (must be 200 before tunnel matters) =="
echo "GET  /       : $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:9052/ 2>&1)"
echo "POST /predict: $(curl -s --max-time 8 -X POST http://localhost:9052/predict -H 'content-type: application/json' -d '{"game_status":"ok","score":0.0,"sim_time":0.0,"n_agents":0,"agent_status":[]}' 2>&1 | head -c 160)"
echo "new heuristic in container? $(docker compose exec -T survival-simulator sh -c 'test -f /app/policy_heuristic.py && echo YES || echo NO' 2>&1 | grep -v -E "$FILTER" | tail -1)"
echo "== cloudflared processes =="
pgrep -fl cloudflared 2>&1 | grep -v -E "$FILTER" || echo "(none running)"
echo "== tunnel config =="
if [ -f "$HOME/.cloudflared/config.yml" ]; then grep -v -E "$FILTER" "$HOME/.cloudflared/config.yml"; else echo "(no config.yml)"; fi
echo "== recent cloudflared logs =="
for f in /tmp/nac-*-cloudflared.log; do [ -f "$f" ] && { echo "-- $f"; tail -n 8 "$f"; }; done 2>/dev/null || true
echo "== public =="
echo "GET https://survival.zaitzev.com/ : $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://survival.zaitzev.com/ 2>&1)"