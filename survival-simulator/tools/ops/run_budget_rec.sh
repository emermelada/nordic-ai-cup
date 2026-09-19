#!/usr/bin/env bash
set -u
cd /opt/nac_gs/best_controller && cp /opt/nac/best_controller/params.json ./params.json
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy
for i in 1 2 3 4; do
  sd=$((3400 + i))
  tmux kill-session -t bud$i 2>/dev/null
  tmux new-session -d -s bud$i "/opt/nacv/bin/python replay.py corpus --seeds $sd --horizon 18000 --traces --outdir /opt/nac/budget_$i 2>&1 | tee run_bud$i.log"
done
sleep 20
tmux ls | tr '\n' ' '
