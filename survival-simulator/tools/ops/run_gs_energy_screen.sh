#!/usr/bin/env bash
# run_gs_energy_screen.sh - test the ONE untested, physics-grounded hypothesis.
#
# WHY THIS EXISTS: the sprint lockout is `energy < max_energy/5`, and max_energy is HERITABLE and
# mutable at birth (cap 1000, default 500). 86% of our predation deaths happen inside the lockout, our
# own winner analysis found winners carry 5.0x higher max_energy (consistent 16/16 episodes), and the
# controller's own comment records newborns already reaching max_energy 996 by accident. So breeding for
# the energy bank should attack the dominant death mechanism directly.
#
# It was never tested: the SERVING controller predates the selection code, so every gs_* arm I ran
# earlier was silently inert (it was a byte-identical copy of the baseline). This runs in /opt/nac_gs,
# which carries the 86-param controller where gs_* is real, with the deployed C6 params as baseline.
#
# The old V2 failure mode was that selection THINNED the fleet and income collapsed (1141 -> 712). The
# arms here therefore include gs_rescue_pop (never gate when the fleet is tiny) and gs_mode 2 (an
# improving gate, which spawns MORE among improvers rather than fewer), plus a vision-dominant arm as a
# CONTROL: if the mechanism is genuinely about energy, vis_ctrl should ratchet VISION and not energy.
set -u
cd /opt/nac_gs/experiments
export PYTHONHASHSEED=0
export SDL_VIDEODRIVER=dummy

arms_for() {
  case "$1" in
    1) echo "base,en_top3,en_top4" ;;
    2) echo "base,en_top3_late,en_top3_rescue3" ;;
    3) echo "base,vis_ctrl" ;;
  esac
}

for i in 1 2 3; do
  A=$(arms_for "$i")
  tmux kill-session -t "gsen$i" 2>/dev/null || true
  tmux new-session -d -s "gsen$i" "cd /opt/nac_gs/experiments && export PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy && /opt/nacv/bin/python mechanism_screen.py --seeds ${1:-2920-2923} --arms $A --outdir /opt/nac_gs/gsen$i 2>&1 | tee run_gsen$i.log"
done
sleep "${2:-45}"
echo "=== lanes ==="
tmux ls | tr '\n' ' '
echo
echo "=== progress ==="
for i in 1 2 3; do printf 'gsen%s: %s\n' "$i" "$(tail -1 /opt/nac_gs/experiments/run_gsen$i.log 2>/dev/null | cut -c1-90)"; done
uptime
