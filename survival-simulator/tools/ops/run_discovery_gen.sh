#!/usr/bin/env bash
# run_discovery_gen.sh - generate the state/action/outcome dataset for counterfactual discovery.
#
# Uses the 64 cores for DATA GENERATION, as instructed. Each job covers a disjoint seed range, writes its
# own JSONL, and the analyser merges them. Design notes (all three failures of the previous attempt
# designed out):
#   * fixed-tick CONTINUOUS outcomes, not the alive/energy knockout that made 'still' and 'toward_predator'
#     look like winners
#   * only HEALTHY focal agents are sampled (energy >= 30% of max), rotating the focal agent
#   * features limited to what the agent can actually observe, with visible-vs-heard predators split
#   * 8 alternative behaviours + the controller's own action + a two-phase SEQUENCE arm
#
# Scale: JOBS x SEEDS_PER_JOB x STATES_PER_SEED states, each with 9 branches of HORIZON ticks.
set -u
cd /opt/nac/experiments
export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy
OUT=/opt/nac/discovery
mkdir -p "$OUT"

JOBS="${1:-20}"
SEEDS_PER_JOB="${2:-5}"
STATES_PER_SEED="${3:-25}"
HORIZON="${4:-300}"
BASE_SEED="${5:-3300}"

for i in $(seq 1 "$JOBS"); do
  lo=$((BASE_SEED + (i-1)*SEEDS_PER_JOB))
  hi=$((lo + SEEDS_PER_JOB - 1))
  tmux kill-session -t "dg$i" 2>/dev/null || true
  tmux new-session -d -s "dg$i" "cd /opt/nac/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python discover.py gen --seeds $lo-$hi --states-per-seed $STATES_PER_SEED --horizon $HORIZON --out $OUT/rows_$i.jsonl 2>&1 | tee /opt/nac/experiments/run_dg$i.log"
done
sleep "${6:-40}"
echo "=== jobs ==="
tmux ls 2>/dev/null | grep -c dg || true
uptime
echo "=== first job progress ==="
tail -2 /opt/nac/experiments/run_dg1.log 2>/dev/null
