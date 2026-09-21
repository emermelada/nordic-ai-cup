#!/bin/bash
# Queue one platform VALIDATION of the live endpoint and wait for its score.
#
# Validations only. The track allows a single evaluation attempt, it is irreversible, and
# running it is the user's decision -- this script must never be pointed at an evaluation
# endpoint.
#
# Usage: NORDIC_TOKEN=<token> tools/serving/validate_platform.sh [predict url]
set -u
: "${NORDIC_TOKEN:?set NORDIC_TOKEN to the team token}"
BASE=https://cases.nordicaicup.com/api/v1/usecases/medical-appointment
URL=${1:-http://87.236.196.76:40724/predict}

count() { curl -s -m 30 "$BASE/status" -H "x-token: $NORDIC_TOKEN" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["n_validations"])'; }

before=$(count)
echo "validations before: $before  url: $URL"
curl -s -m 60 "$BASE/validate/queue" --request POST --header "x-token: $NORDIC_TOKEN" \
  --header 'Content-Type: application/json' --data "{\"url\":\"$URL\"}"
echo
# A validation runs 19 hidden conversations in about 4-5 minutes.
for _ in $(seq 1 40); do
  sleep 20
  [ "$(count)" != "$before" ] && break
  printf '.'
done
echo
curl -s -m 30 "$BASE/status" -H "x-token: $NORDIC_TOKEN" | python3 -c '
import json, sys
rows = json.load(sys.stdin)["validations"]
for row in rows[:3]:
    print(row["score"], row.get("finished_at"), "errors:", len(row["errors"]) or "none")'
