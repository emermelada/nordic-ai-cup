#!/bin/bash
# Determinism test: run the SAME sweep in two fresh processes and diff the results.
set -e
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/d1.log 2>&1
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/d2.log 2>&1
echo "=== run 1 ==="
cat /tmp/d1.log
echo "=== run 2 ==="
cat /tmp/d2.log
if diff -q /tmp/d1.log /tmp/d2.log >/dev/null; then
  echo "RESULT: IDENTICAL -> deterministic"
else
  echo "RESULT: DIFFERENT -> still nondeterministic"
  diff /tmp/d1.log /tmp/d2.log || true
fi