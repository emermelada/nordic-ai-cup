#!/usr/bin/env bash
# read-only deploy state check (no changes)
REPO="/Users/zaitzev/Life/Projects/nordic-ai-cup"
cd "$REPO" || exit 1
echo "== docker containers =="
docker compose ps 2>&1 | grep -v -E "gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh" || true
echo
echo "== local :9052 =="
echo "GET / : $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:9052/ 2>&1)"
echo "POST /predict: $(curl -s --max-time 8 -X POST http://localhost:9052/predict -H 'content-type: application/json' -d '{"game_status":"ok","score":0.0,"sim_time":0.0,"n_agents":0,"agent_status":[]}' 2>&1 | head -c 200)"
echo
echo "== does the RUNNING container have the new heuristic code? =="
docker compose exec -T survival-simulator sh -c 'ls -la /app/policy_heuristic.py 2>&1; grep -l policy_heuristic /app/agent_server.py 2>&1' 2>&1 | grep -v -E "gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh" || true
echo
echo "== public URL =="
echo "GET https://survival.zaitzev.com/ : $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://survival.zaitzev.com/ 2>&1)"