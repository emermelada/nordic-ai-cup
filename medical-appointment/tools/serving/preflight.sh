#!/bin/bash
# Verify the served build really is the three-producer medoid, before anyone submits to it.
#
# Two silent failures cost real points on 2026-09-20: a deleted extractor checkpoint left the
# endpoint answering normally while the vote fell back to stage B spans, and a restart during a
# running attempt scored 0.8110 instead of 0.8308. This checks for both.
#
# Run it on the box:  /workspace/preflight.sh
set -u
LOG=${LOG:-/workspace/api.log}
API=${API:-http://127.0.0.1:3000}
VLLM=${VLLM:-http://127.0.0.1:18000}
fail=0
check() { # name, condition-output, expected
  if [ "$2" = "$3" ]; then printf 'PASS  %s\n' "$1"; else printf 'FAIL  %s (got %s, want %s)\n' "$1" "$2" "$3"; fail=1; fi
}

check "vLLM answers"        "$(curl -s -m 10 -o /dev/null -w '%{http_code}' $VLLM/v1/models)" 200
check "API answers"         "$(curl -s -m 10 -o /dev/null -w '%{http_code}' $API/api)" 200
check "extractor loaded"    "$(grep -c 'Span extractor loaded' $LOG)" 1
check "ranker loaded"       "$(grep -c 'Span ranker loaded' $LOG)" 1
check "no producer errors"  "$(grep -c 'unavailable; \|Producer vote failed' $LOG)" 0

echo '--- one real conversation ---'
before=$(grep -c 'Producer vote' $LOG || true)
cd /workspace/medical-appointment && python3 smoke.py 2>&1 | tail -4
vote=$(grep 'Producer vote' $LOG | tail -1)
echo "$vote" | grep -qE 'Producer vote: [1-9][0-9]* extractor span\(s\), [1-9][0-9]* ranker span\(s\)' \
  && printf 'PASS  vote used both producers\n' \
  || { printf 'FAIL  vote did not use both producers: %s\n' "$vote"; fail=1; }
after=$(grep -c 'Producer vote' $LOG)
check "the vote ran once more" "$((after - before))" 1

echo '--- is a platform attempt in flight? ---'
recent=$(grep '46.62.240.126' $LOG | tail -1)
[ -n "$recent" ] && echo "NOTE  the platform has called this endpoint; check /status before restarting" \
                 || echo "none seen in this log"

[ $fail -eq 0 ] && echo 'PREFLIGHT PASS -- safe to hand over the URL' \
                || echo 'PREFLIGHT FAIL -- do not submit against this endpoint'
exit $fail
