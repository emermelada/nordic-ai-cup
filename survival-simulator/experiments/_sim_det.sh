#!/bin/bash
# Sim-only determinism: constant policy, two fresh processes, with and without hash randomization.
set -e
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator/experiments
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python

echo "--- default PYTHONHASHSEED ---"
$PY _sim_det.py > /tmp/s1.log 2>&1
$PY _sim_det.py > /tmp/s2.log 2>&1
printf 'run1: '; cat /tmp/s1.log
printf 'run2: '; cat /tmp/s2.log
if diff -q /tmp/s1.log /tmp/s2.log >/dev/null; then echo "=> SIM DETERMINISTIC"; else echo "=> SIM NONDETERMINISTIC"; fi

echo "--- PYTHONHASHSEED=0 ---"
PYTHONHASHSEED=0 $PY _sim_det.py > /tmp/s3.log 2>&1
PYTHONHASHSEED=0 $PY _sim_det.py > /tmp/s4.log 2>&1
printf 'run3: '; cat /tmp/s3.log
printf 'run4: '; cat /tmp/s4.log
if diff -q /tmp/s3.log /tmp/s4.log >/dev/null; then echo "=> SIM DETERMINISTIC (pinned)"; else echo "=> SIM NONDETERMINISTIC even pinned"; fi