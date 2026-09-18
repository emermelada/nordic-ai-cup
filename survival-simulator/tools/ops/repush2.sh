#!/bin/bash
# The previous repush used the repo-root cwd, so every survival-simulator/ pathspec failed:
# "fatal: pathspec 'best_controller.py' did not match any files". The commit that pushed therefore
# did NOT contain the controller/journal/evidence work. Redo it from the correct directory.
cd /Users/zaitzev/Life/Projects/nordic-ai-cup/survival-simulator || exit 1
export GIT_TERMINAL_PROMPT=0
PY=/Users/zaitzev/Life/Projects/nordic-ai-cup/.venv/bin/python

echo "=== what did the bad push actually contain? ==="
git show --stat --oneline HEAD | head -12

echo "=== provenance hash must match the controller (build gate) ==="
$PY tools/check_controller.py --update | tail -3
$PY tools/check_controller.py | tail -3; echo "GUARD_EXIT=$?"

echo "=== stage ONLY the intended files (cwd is now survival-simulator/) ==="
git add best_controller.py best_controller.sha256 best_controller/params.json JOURNAL_2026-09-18.md \
        experiments/evolve_det.py experiments/w_hybrid.py experiments/w_confirm.py experiments/w_deaths.py \
        experiments/paired_results.jsonl experiments/rl_fleet.py \
        experiments/M_walk_250.json experiments/M_walk_700.json experiments/B4_early_4500.json \
        experiments/P1_memory_only.json experiments/P2_memory_social.json experiments/P3_full_trio.json \
        experiments/B1_early_mild.json experiments/B2_mid_strong.json experiments/B3_late_hard.json \
        experiments/wH1_winEARN_liveSPEND.json experiments/wH2_liveEARN_winSPEND.json experiments/wH3_midbreed.json \
        experiments/F1_face.json experiments/F2_face_noflee.json experiments/F3_face_wide.json
echo "staged now: $(git diff --cached --name-only | wc -l | tr -d ' ') files"
git diff --cached --name-only

echo "=== hard guard: nothing over 2MB ==="
git diff --cached --name-only | while read -r f; do
  [ -f "$f" ] || continue
  sz=$(wc -c < "$f")
  if [ "$sz" -gt 2000000 ]; then echo "  UNSTAGING $f ($((sz/1048576)) MB)"; git reset -q HEAD -- "$f"; fi
done

echo "=== commit + push ==="
git commit -q -m "Predator-facing defence + memory search + review fixes (evidence on disk)

MEASURED MECHANIC (src/elements/predator.py:37): a predator CHARGES only when the agent is not
facing it (|agent_looking_dir| > pi/2) or when it is already inside hearing_radius*1.5 = 90 units;
if the agent faces it, the predator PIVOTS 45deg off-target instead of closing. Predators never
tire (Predator.step sprints with no energy accounting), so kiting is impossible. Turning costs
|turn|/(2pi) energy (180deg = 0.5) versus 5.5/tick to sprint away, and movement is applied BEFORE
the turn (environment.py:614/618), so an agent can hold a pursuer in its vision cone while still
travelling toward fruit. Implemented as _facing_turn() + face_predator/face_cone/face_dist_max/
face_min_dist, all default OFF.

WHY IT MATTERS (death-budget diagnostic, experiments/w_deaths.py on the live controller): the fleet
banks ~3,000 energy and then loses 14 of 16 agents inside a single 1,000-tick window, each kill
carrying off that agent's bank (fleet energy 2767 -> 51); predation is 35-43% of all deaths, and
predator spawn chance grows LINEARLY WITH TIME (environment.py:763). Runs end in a predation
cascade, not gradual starvation -- which is why every foraging/banking/memory experiment today
failed to help.

Also carried: memory-search knobs (memory_search, commit_len, commit_speed_frac,
follow_agent_weight, ars_speed_frac) recorded as UNPROVEN/LOSING (P1 -14.5%, M_walk_250 -6.2%);
evolve_det ranking/adoption now use the same quantity (FRUIT_W = 0.05, headcount bias documented);
experiments/paired_results.jsonl holds the 10 paired comparisons that previously existed only in
container logs; w_deaths.py is the death-budget diagnostic." || echo "NOTHING_TO_COMMIT"
git push origin survival-simulator-policy 2>&1 | tail -3
echo "LOCAL_HEAD=$(git rev-parse --short HEAD) STATUS=$(git status --porcelain | wc -l | tr -d ' ') dirty"