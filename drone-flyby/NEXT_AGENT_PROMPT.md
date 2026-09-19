# Prompt for the next session

Paste everything below the line into a fresh Claude Code session started in
`~/Proyectos/nordic-ai-cup/drone-flyby`, once the new GPU box is rented.

---

You are a senior ML engineer working on the Nordic AI Cup 2026 Drone Flyby
entry. **Deadline 20 Sep 2026, 16:00 CEST. Validation attempts are unlimited
(~2 min each); the final evaluation is ONE attempt on a different 250-frame
flight.** Best validated score so far: **0.5270**, the mean of four complete
249/249 runs, config frozen on branch `BEST-WORKING-VERSION`.

**Read `drone-flyby/NEW_PLAN.md` first, then the START HERE block at the top of
`drone-flyby/HANDOVER.md`.** Between them they hold every measured result; do
not re-derive them. `NEW_PLAN.md` §6 lists twelve things already measured dead —
spending a validation run on any of them is pure waste.

## Your job, in order

**BEFORE ANYTHING ELSE: the deadline is 16:00 CEST TODAY and both GPU boxes are
destroyed.** Re-rent and get three confirmation runs done with real buffer, not
at 15:50. Re-renting plus gating plus a first run is comfortably an hour.

1. **Re-rent the serving host and gate it.** Vast.ai, Czechia
   `datacenter:214845` (`m:53164` / `m:40773`, was `93.91.156.98`) — that host
   produced every 0.50+ score. Use `tools/bootstrap_remote.sh`, which gates on
   latency AND sustained bandwidth to the evaluator (Hetzner Helsinki). A bad
   host cost 0.05–0.30 in one day, three times what a day of tuning gained.
   Serve with `tools/arm.sh`, which refuses to hand over a service that is not
   serving what you asked for.

2. **Run the paired camera A/B.** Three complete runs each, same host, same
   session, `DRONE_CAMERA=full` then `DRONE_CAMERA=row0`, on the served
   4-model stack. `row0` is already in `SWEEPS`; the default is still `full`.

3. **Then add v9.** `models/drone-yolo11m-p2-v9.pt` as a FIFTH pass at 1280 —
   an addition, never a replacement. Three complete runs of the winner from
   step 2 plus v9.

   **Ship it if it beats 0.5270 at all**, and read the **per-class** column for
   `spacecraft` and `small_launcher` rather than the total. Both sessions agreed
   this bar after the simulation put v9 at only +0.0023: the mined truth
   contains none of `spacecraft`/`small_launcher`/`condor`/`medium_plane`, and
   `spacecraft` 2.0 and `small_launcher` 1.5 are v9's two highest class weights.
   Rejecting v9 on that number would repeat the exact error this project made
   by rejecting Level-0 cameras on a detector blind at Level 0. `small_launcher`
   is the precedent: 0.000 in every configuration ever measured until the 2560
   pass took it to 0.512, and no offline metric predicted that.

4. **Then, if runs remain:** mine the four classes missing from the truth file
   (`spacecraft`, `small_launcher`, `condor`, `medium_plane`) with
   `tools/mine_scene.py`, re-calibrate with `tools/calibrate_truth.py`, and only
   then look at the per-class ranking problem in `NEW_PLAN.md` §4b.

## Rules that are not negotiable

* **249/249 or the run does not count.** Two missing frames cost 0.026, more
  than most changes gain. Count frames after every run.
* **Check `/api` before every attempt** (`models_loaded`, `imgsz`, `box_grow`),
  never the launch command. Every expensive failure in this project has looked
  like a success: a stale Docker image, a silently ignored `DRONE_MODEL_ALT`, a
  poisoned detection cache.
* **Three complete runs per arm, judge on the mean.** Promoting the best of
  several draws is selecting on noise; sd is ~0.005–0.007 on complete runs.
  Runs are deterministic given the camera path, so a matched pair beats six
  unpaired runs.
* **Do not move `BEST-WORKING-VERSION`** unless a real validation run beats
  0.5270 on the mean of complete runs. It is the one-shot fallback.
* **`tools/preflight.py` is not a load test.** It replays sequentially against a
  warm service while the evaluator emits every 333 ms regardless. It passed the
  3200 stack at "0/11 over budget"; that stack then ran at 514 ms median and
  scored 0.264 with 115 frames unanswered.
* **Offline metrics lie in opposite directions.** Recall ignores what extra
  boxes cost; offline AP over-charges for them because the truth is incomplete.
  Use the mined truth for RANKING configurations (Spearman +0.811, constant
  −0.074 offset), never for absolute score, and never to tune `DRONE_BOX_GROW`
  (it inherits our own box convention and is blind to growth by construction).
* **The calibration set is `data/runs_20260919.tar.gz`** (6.3 MB packed, 49 MB
  unpacked, force-added past the `data/` gitignore): 31 runs' answers with their
  real scores. Any new or extended truth file must be re-calibrated against it
  with `tools/calibrate_truth.py`. Do not delete it — it existed on one laptop
  until yesterday.

## What to expect

`row0` +0.01 to +0.03. v9 unknown: it measures +0.0023 in simulation, but the
mined truth contains none of `spacecraft`/`small_launcher`/`condor`/
`medium_plane` and v9's two highest class weights are `spacecraft` 2.0 and
`small_launcher` 1.5 — the metric is blind to what it was built to fix, so that
number cannot reject it. Realistic landing zone is **0.55–0.60**, not 0.70.
Nothing measured supports +0.17 from configuration.

Report what you measure, including negative results, and say plainly when a run
does not clear the noise floor.
