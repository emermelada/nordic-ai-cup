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
and RNG state), and the measured results in `survival-v2/RESULTS.md`. The
`survival-simulator/` directory holds an earlier, separate controller written
by another team member.

## Medical Appointment

Code in [`medical-appointment/`](medical-appointment/): faster-whisper `base`
word timestamps matched to the annotators' coordinates, a Qwen3.8-27B answer pass
over the numbered transcript, and evidence spans chosen as the medoid of three
independently trained producers -- that answer model's own quote, a fine-tuned
DeBERTa span extractor, and a listwise span ranker over an enumerated candidate
pool.

This is the build that was graded: **validated 0.8307887829198248 four times
identically, evaluated at 0.8222490889515863** with zero errors.
[`medical-appointment/REPRODUCE.md`](medical-appointment/REPRODUCE.md) states the
exact model, GPU and speech-recogniser configuration, the serving commands, and
the retraining recipe for both checkpoints; the weights themselves, 1.7 GB each,
are not in the repository. The earlier solution whose defaults reproduce the
0.802 validation behaviour is unchanged on the `medical-appointment-0.802` and
`medical-appointment-exact-evidence` branches.

## Provenance we could not confirm from git alone

One statement below rests on team notes and consistency checks, not on a commit
that pins the graded build. The team should replace it with the exact commit.

* **Survival:** the graded policy is believed to be `survival-v2` at its tip.
  Its local 40-seed mean was 1484 against a graded validation of 1815 and
  evaluation of 1405, which is consistent, but no record ties a commit to the run.
* **Medical:** resolved. The graded build is the merge that brought
  `medical-appointment/` to its current state, and its configuration is pinned in
  `medical-appointment/REPRODUCE.md`. Its four validations returned
  0.8307887829198248 byte-identically and the evaluation returned
  0.8222490889515863.
