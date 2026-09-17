#!/bin/bash
# Controller-level determinism: two fresh processes, hashing the controller file around each run
# so a concurrent edit cannot masquerade as nondeterminism.
set -e
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python

m1=$(md5 -q best_controller.py)
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/e1.log 2>&1
m2=$(md5 -q best_controller.py)
$PY batch_eval.py cfg_det.json 3000 100 > /tmp/e2.log 2>&1
m3=$(md5 -q best_controller.py)

echo "controller md5 before/after run1/after run2: $m1 $m2 $m3"
echo "=== run 1 ==="; cat /tmp/e1.log
echo "=== run 2 ==="; cat /tmp/e2.log
if [ "$m1" = "$m2" ] && [ "$m2" = "$m3" ]; then
  echo "=> controller file stable during test"
else
  echo "=> CONTROLLER WAS EDITED MID-TEST (results unattributable)"
fi
if diff -q /tmp/e1.log /tmp/e2.log >/dev/null; then
  echo "=> RESULT: IDENTICAL -> controller-level determinism restored"
else
  echo "=> RESULT: DIFFERENT -> real nondeterminism remains in the policy"
  diff /tmp/e1.log /tmp/e2.log || true
fi