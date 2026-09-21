# What we submitted, and where the code is

Team **Elysa's Secret**, University of Southern Denmark. Nordic AI Cup 2026.
Each use case had one graded evaluation attempt. These are the results and the
code behind them.

| Use case | Evaluation | Denmark rank | Points | Code |
|---|---|---|---|---|
| Medical Appointment | 0.822 | 1st | 25 | [`medical-appointment/`](medical-appointment/) |
| Survival Simulator | 1405.256 | 4th | 12 | [`survival-v2/`](survival-v2/) |
| Drone Flyby | 0.263 | 6th | 8 | [`drone-flyby/`](drone-flyby/) |

**Total 45 points, 1st in Denmark.**

Every model here does real inference or real search at request time. Nothing
answers from a stored table of validation or evaluation answers.
Third-party models, datasets and AI tools are listed in
[`AI_TOOLS_LOG.md`](AI_TOOLS_LOG.md).

## Drone Flyby  (fully documented)

The exact graded configuration is in
[`drone-flyby/SERVED_CONFIG.md`](drone-flyby/SERVED_CONFIG.md), with the launch
environment in `drone-flyby/configs/evaluated_serve_env.sh`. Five YOLO11 passes
per frame, a fixed six-quadrant camera sweep with a whole-frame look after each
step, ground motion refitted online during the flight, and one flat box-growth
factor for every class. Validation mean 0.5841 over 7 runs; evaluation 0.2630 on
an unseen flight with all 249 frames answered.

`SERVED_CONFIG.md` also lists what was tried on the real grader and rejected
(per-class box geometry, a sixth pass, Level-2 zoom, per-class thresholds),
because those were fits to the validation flight and would not generalise.

Not in the repository: 3.4 GB of recorded validation frames used for offline
replay. They lived on a rented GPU instance that has been shut down.

## Survival Simulator

Code in [`survival-v2/`](survival-v2/): a shared-map colony policy, a numba
simulator that matches the official one tick for tick (positions, energy, score
and RNG state), and the measured results in `survival-v2/RESULTS.md`.

The graded evaluation (1405.256) came from the exact artifact recorded in
[`survival-v2/EVALUATED.md`](survival-v2/EVALUATED.md): controller
`56489acfff2cba36...` served with `68fac18eec77223e...`, three games run back to
back and averaged (1319.9 / 1298.2 / 1599.0). That file lists the hashes, the
serving command, the measured behaviour over 400 seeds and the honest limits.

## Medical Appointment

Code in [`medical-appointment/`](medical-appointment/): faster-whisper `base`
word timestamps matched to the annotators' coordinates, and an LLM that answers
over the transcript. Defaults reproduce the 0.802 validation behaviour; the
branch history records the experiments run on top of it.

## Provenance we could not confirm from git alone

Two statements above rest on team notes and consistency checks, not on a commit
that pins the graded build. The team should replace this section with the exact
commits.

* **Survival:** the graded policy is believed to be `survival-v2` at its tip.
  Its local 40-seed mean was 1484 against a graded validation of 1815 and
  evaluation of 1405, which is consistent, but no record ties a commit to the run.
* **Medical:** the graded validation was 0.831 and the evaluation 0.822, while the
  branch defaults reproduce the 0.802 behaviour. The repository does not record
  which commit or settings produced 0.831.
