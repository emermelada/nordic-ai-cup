#!/usr/bin/env bash
U="https://survival.zaitzev.com"
BODY='{"game_status":"ok","score":0.0,"sim_time":0.0,"n_agents":0,"agent_status":[]}'
FILTER='gitstatus|zle|setopt|monitor|GITSTATUS|Restart|exec zsh'
echo "== GET / (curl) =="
curl -s -o /dev/null -w 'code=%{http_code}\n' --max-time 15 "$U/" 2>&1 | grep -vE "$FILTER"
echo
echo "== POST /predict empty body, curl default UA =="
curl -s -D - -o /tmp/pub_post_body.txt --max-time 20 -X POST "$U/predict" -H 'content-type: application/json' -d "$BODY" 2>&1 | grep -viE "$FILTER" | grep -iE '^(HTTP/|server:|cf-ray:|cf-mitigated:|content-type:|date:)' 
echo "body: $(head -c 200 /tmp/pub_post_body.txt)"
echo
echo "== POST /predict with a normal browser User-Agent =="
curl -s -o /dev/null -w 'code=%{http_code}\n' --max-time 20 -X POST "$U/predict" \
  -H 'content-type: application/json' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' -d "$BODY" 2>&1 | grep -vE "$FILTER"
echo
echo "== POST /predict localhost (bypass cloudflare) for comparison =="
curl -s -o /dev/null -w 'code=%{http_code}\n' --max-time 10 -X POST "http://localhost:9052/predict" -H 'content-type: application/json' -d "$BODY" 2>&1 | grep -vE "$FILTER"