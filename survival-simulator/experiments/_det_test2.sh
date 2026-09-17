#!/bin/bash
# Same as _det_test.sh but with PYTHONHASHSEED pinned -> isolates hash-randomization effects.
set -e
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python
export PYTHONHASHSEED=0
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/h1.log 2>&1
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/h2.log 2>&1
echo "=== hashseed=0 run 1 ==="; cat /tmp/h1.log
echo "=== hashseed=0 run 2 ==="; cat /tmp/h2.log
if diff -q /tmp/h1.log /tmp/h2.log >/dev/null; then
  echo "RESULT: IDENTICAL -> hash randomization was the cause"
else
  echo "RESULT: DIFFERENT -> another source remains"
  diff /tmp/h1.log /tmp/h2.log || true
fi