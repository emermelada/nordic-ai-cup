# AUTOPILOT — standing orders for unattended runs
**Read this first, every wake-up.** Mission: maximise the official validation score on
`cases.nordicaicup.com/survival-simulator`. Deadline: **Sunday 20 Sep 2026, 16:00 CEST.**

You are resuming work with NO memory of previous sessions. `STATUS_LOCKIN_2026-09-19.md` has the full
position; this file is the operating procedure. Do not re-litigate settled questions — the refuted list
in STATUS_LOCKIN §4 is closed.

---

## HARD RULES (violating these has already cost real score)

1. **Never deploy** a controller unless it has (a) ≥40 paired seeds on seeds it was never screened on,
   (b) win rate ≥55%, and (c) a non-negative floor gain (p10 vs baseline). A 20-seed screen is a screen,
   NOT a result. Four headlines in this project died on fresh seeds.
2. **Never run experiments on the serving box** (`94.237.34.245`) while any validation could be running.
   Serving latency IS score: `score ≈ ticks/10`, and ticks are consumed in wall-clock time. A missing
   `latency_guard.sh` once cost a run 117 ms/tick. Put lanes on the experiment box only.
3. **Never modify the served artifact** (`best_controller.py` on the serving box) outside a deploy.
   After any deploy, verify the hash inside the running container matches `best_controller.sha256`.
4. **Commit and push everything** at the end of every wake-up. Uncommitted research is a real risk —
   two fallback hosts were lost with work on them.
5. **No secrets in commits.** Never print or commit SSH private keys, tokens, or connection strings.
6. **Measure on x86 only.** macOS/ARM gives 4,472 ticks where x86 gives 7,816 for the same seed+policy.
7. **A knob that does not exist is silently ignored.** Before trusting any arm, confirm the override key
   is in `DEFAULT_PARAMS` (sched.py now warns) and that the arm actually differs from BASE.

## HOSTS AND PATHS

| purpose | host | notes |
|---|---|---|
| serving (graded) | `ssh -i ~/.ssh/vps_hermes root@94.237.34.245` | 32 vCPU, Caddy + container `nac-survival-vps`, `/opt/nac` |
| experiments | `ssh -i ~/.ssh/vps_hermes root@212.147.236.122` | 64 vCPU, repo `/opt/nac`, venv `/opt/nacv`, lanes in `tmux` |
| standby (tested) | same 64 vCPU box | `nac-survival-STANDBY` on :9053, Caddy DISABLED; see `tools/ops/ROLLBACK.md` |

Lane logs: `/opt/nac/experiments/run_*.log`. Lane results: `/opt/nac/results/*.jsonl` (+ cache).
Start a lane with:
`tmux new-session -d -s NAME "cd /opt/nac/experiments && SDL_VIDEODRIVER=dummy /opt/nacv/bin/python <cmd> 2>&1 | tee run_NAME.log"`
**Always export `PYTHONHASHSEED=0`** (sched.py and replay.py set it themselves; other scripts may not).

## THE WAKE-UP LOOP

1. **Status sweep** (one ssh per box):
   `tmux ls`; `tail -3 run_*.log` for live lanes; `uptime`.
   On the serving box: `docker exec nac-survival-vps sha256sum /app/best_controller.py` and
   `curl -s -o /dev/null -w '%{http_code} %{time_total}' https://survival.zaitzev.com/`
2. **Read any finished verdict.** Apply the decision rules below. Do not act on a 20-seed screen.
3. **Launch the next queue item** (below). One or two lanes at a time, never oversubscribe past ~60
   workers on the 64-core box.
4. **Log** to `AUTOPILOT_LOG.md` (append: time, what ran, what it showed, what you did next) and push.
5. **Report ≤12 lines** to Telegram: what finished, the numbers, what you launched, any blocker.

If nothing in the queue is actionable, do the cheapest useful measurement rather than idling: extend a
verdict's seed count, re-measure the noise floor, or collect more oracle disagreements.

## QUEUE (work top-down; re-order only with evidence)

**Q1 — Deploy holdout verdict (HIGHEST VALUE).** `survival-simulator/experiments/run_deploy.log` on the
experiment box: 8 arms, 20-seed screen then 40-seed held-out stage. Candidate to beat: **`C3_no_reserve`**
(`reserve_frac: 0.0`), which replicated on two independent seed sets (+5.9% and +9.3%, W12/L8 both) and
cut variance 68%. If it clears HARD RULE 1 → deploy with `tools/ops/deploy_arm.sh`, verify the in-container
hash and a real `POST /predict`, then tell the user to validate (the board keeps our best attempt).

**Q2 — Scale the counterfactual oracle.** `experiments/oracle_probe.py` — snapshot/restore is VERIFIED
(bit-identical continuations on x86; see the file header). Scale from 3 usable states to **hundreds**,
across many seeds, and record: the fraction of states where an alternative behaviour beats the controller,
WHICH behaviour wins, and how that fraction depends on the world state (energy, predator distance,
fruit visible, time). Save the disagreement states as replays for human inspection. Report the fraction
with its sample size, never a single example.

**Q3 — Distil, only after Q2.** If the oracle beats the controller on ≥20% of states with a *consistent*
winning behaviour, encode that behaviour as a rule arm (not a network) and test it at 40 paired seeds
against BASE. Behaviour of interest so far: sprinting to the nearest visible fruit beat the controller
for the focal agent (+117 energy) in the one state measured.

**Q4 — bet B tier-2.** The fleet-level meta-controller evolves its own population target downward. Its
held-out paired confirmation is the only lane that could move the score materially. Read and report it.

**Q5 — wide1 breadth.** 601 candidates, staged 12k → 18k. Any top candidate gets a 40-seed paired test
on unseen seeds before it means anything.

## DECISION RULES

* **Noise floor:** the simulator is NOT fully deterministic across processes — byte-identical configs have
  disagreed on up to 6 of 20 seeds. So: (a) always include a duplicate baseline in a run to MEASURE the
  noise floor for that run, (b) 20 seeds screens only, (c) 40-80 seeds to claim, (d) prefer the FLOOR
  (p10) over the mean, because the official score is a mean of 3 runs and the bad tail dominates.
* **Mechanism before score.** An arm must move the mechanism it claims to move (travel per fruit, fleet
  max_energy, lockout fraction) before its survival number is believed. `mechanism_screen.py`,
  `ledger_analysis.py`, `winner_signature.py`, `survival_autopsy.py` exist for this.
* **One writer per file.** Another agent may be editing `best_controller.py` and the dated journals —
  do not clobber them; work in `experiments/` and add new files.

## CLOSED — DO NOT REOPEN

reserve/repro floors; memory search / follow / ARS; thin relay & banking; E3 retreat-while-facing evasion
(+0.4% over 80 seeds); V2 genome-aware breeder selection (−17% over 40); random neural policies (−38%);
mutated nets (−23.7%); heterogeneous fleet roles (best −6.9%); generic PPO (no established requirement).
**The shared mechanism behind every failure: each idea cut income or mobility in an access-limited world
with a 15x energy surplus. That is a finding, not bad luck.**
