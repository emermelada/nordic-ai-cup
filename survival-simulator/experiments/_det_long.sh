#!/bin/bash
# Two fresh processes, long horizon, same seeds -> detect residual nondeterminism.
set -e
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python
$PY _det_long.py > /tmp/L1.log 2>&1
$PY _det_long.py > /tmp/L2.log 2>&1
echo "=== run1 ==="; cat /tmp/L1.log
echo "=== run2 ==="; cat /tmp/L2.log
if diff -q /tmp/L1.log /tmp/L2.log >/dev/null; then echo "=> DETERMINISTIC at 16k"; else echo "=> NONDETERMINISTIC at 16k"; fi