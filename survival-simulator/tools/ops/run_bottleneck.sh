#!/usr/bin/env bash
# run_bottleneck.sh - one job per intervention so all eight finish in minutes, not an hour.
set -u
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy
mkdir -p /opt/nac/bottleneck
for m in control no_predators no_repro unlimited_sprint lower_move_cost perfect_info cheap_food fewer_agents; do
  tmux kill-session -t bn_$m 2>/dev/null
  tmux new-session -d -s bn_$m "/opt/nacv/bin/python bottleneck_one.py $m 3800,3801,3802,3803 2>&1 | tee /opt/nac/bottleneck/run_$m.log"
done
sleep 15
echo "jobs started: $(tmux ls 2>/dev/null | grep -c bn_)"
