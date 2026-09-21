# SOLUTION_JOURNEY — Survival Simulator (Nordic AI Cup 2026)

> **Note (2026-09-20, added during the hand-off):** the survival work was later consolidated into a
> single directory, [`survival-v2/`](survival-v2/) — see `survival-v2/EVALUATED.md` for the graded
> artifact. The paths below refer to the earlier two-directory layout and are kept as history.

**Last updated:** 17 Sep 2026 · **Repo:** `nordic-ai-cup/survival-simulator/`
**Policy of record:** episodic, evidence-driven. Every number below was measured from the simulator source or from runs — nothing is assumed. Anything not yet known is labeled **in progress**.

---

## 1. TL;DR — how we approached it

We refused to treat "build an RL controller" as the default. We first read the simulator source to learn the *exact* reward, then built a shared harness before any model, established strong baselines, ran a cheap PPO feasibility spike that invalidated from-scratch RL within our time budget, and discovered a critical ambiguity in the grading metric. We are now executing a directive-driven control/optimization program — decompose the problem, build a strong scripted + geometric-steering baseline, then tune and harden it experimentally — rather than betting the deadline on end-to-end training. A deployable self-contained heuristic is already the policy of record (agent on port 9052).

## 2. First principles — the domain we learned

We traced the reward directly in `src/elements/environment.py` (source, not guesswork):

- **`score += dt` fires ONCE per step regardless of agent count** (dt=0.1). So the survival bonus is a **constant w.r.t. strategy** at any fixed eval horizon — everyone is paid the same tax-free `horizon × 0.1` just for the world ticking.
- **Eating fruit** adds `fruit.energy/1000` (≈ +0.06 each). **Being eaten** subtracts `agent.energy/100` — and since eaten agents carry a lot of energy, predation is a large negative.
- => the **real learnable margin is fruit intake minus predation loss**. This is a foraging/evasion problem, NOT "survive as long as possible."
- Each agent sees a **partial observation** (energy, age, biome, movement/traits, and an entity list {Fruit, Tree, Agent, Predator, Edge} with distance/angle). Vision is a cone/raycast. Edge coordinates are agent-local.
- Actions: `move_distance` (≤ sprint_speed 20), `move_direction`/`turn_angle` (relative), `spawn_agent` (costs 100 energy). Walking costs 0.05/unit, sprinting 0.5/unit, turning costs energy, aging raises costs. Predators spawn over time and hunt. Biomes penalize movement (swamp 0.5, river 0.3, desert 0.8).

**The two grading hypotheses (unresolved, see §5).** The directive says the sim runs *up to 30,000 ticks*, and final grade averages 3 runs on preset deterministic seeds. That leaves two possible metrics:
- **Fixed-horizon**: total score at the horizon regardless of deaths. Everyone gets `horizon×0.1` baseline; the win margin is only fruit minus predation — *tiny but scales with time*.
- **Survivorship**: episode effectively ends when the team is wiped, so survival dominates.

These imply **opposite optimization targets**. We currently cannot tell which the grader uses. **Resolution path (in progress):** queue a validation of the heuristic and compare its score against the provided baseline under both assumptions.

## 3. What we built and why

**Harness first, before any model:**
- `experiments/env_wrapper.py` — `build_obs(agent_state)` produces a fixed **31-dim normalized vector**, shared *verbatim* between training and the served inference path (must match StepResponse fields exactly).
- `FleetEnv` — a single-agent gym env (one controlled agent) for a clean training signal; the learned/shared policy is reused across all agents at inference.
- `run_eval_episode()` — evaluates against the **real 5-agent world** for honest scoring.
- **Reward shaping** for training: `score_delta - dt` removes the constant survival bonus so only fruit(+) and predation(−) drive the gradient. True score is always reported separately.

**Deployed candidate:** `survival-simulator/policy_heuristic.py` (v1 behavior) is wired into `agent_server.py` and smoke-tested against a realistic StepResponse (valid `{"actions":[...]}` JSON; seek/flee/spawn branches exercised). Container listens on **9052**, stdlib-only so `requirements.txt` is unchanged.

## 4. What DIDN'T work (honestly)

**From-scratch PPO — measured, then invalidated.** Trained SB3 PPO (MlpPolicy 256×256, lr 3e-4, shaped reward) for 150k steps = 646s (~232 steps/s; early deaths caused many ~0.6s env resets). Eval (survivorship): **74.2 ± 9.0** over 5 seeds, teams wiped, ~0–1 fruit eaten. PPO did not learn to forage or survive in budget. Root causes: the fruit signal is far too sparse (~0.15% of steps in the EDA) for a dense-gradient learner, and wall-clock is ~5× slower than raw sim due to resets. Verdict recorded in `spikes/001-ppo-feasibility/README.md`. This **confirmed** the user's own intuition that we don't have to — or can't reliably — train from scratch.

**Heuristic v2 regression (negative result).** Our robustness-motivated v2 (energy-aware cruising, early predator drift, edge/avoidance, stricter spawn) *regressed* to ≈308 and ate fewer fruit than v1. A valuable negative: added sophistication can hurt; v1 remains the champion engineered policy.

**Constraints we hit and worked around.** Local M4 Mac, full sim ~4,400 steps/s but RL resets dropped it to ~230 steps/s. Competition runs 17–20 Sep (short). We corrected the eval horizon **3000 → 30,000** ticks after reading the directive, which made the earlier 3000-tick numbers acknowledge-ably understated. Environment friction surfaced mid-run (bare `python` missing, `numpy`/`cma` not on the default interpreter) — we resolved the interpreter by using `repo/.venv/bin/python` (Python 3.14.7, numpy 2.3.5, scipy 1.16.3, pygame 2.6.1) and flagged `cma` as not yet installed for the evolutionary step.

## 5. What worked

**The engineered heuristic crushed the baselines** (5-agent world, 3000 ticks, survivorship measure):

| policy | score (mean±std) | ~steps | fruit/eps | predated | alive% |
|---|---|---|---|---|---|
| random | 25.7±5.9 | 257 | 0.8 | 5.0 | 0 |
| dummy | 22.4±6.9 | 223 | 2.4 | 10.2 | 0 |
| **heuristic (v1)** | **210.1±106.4** | 2007 | 227.8 | 24.2 | 40 |

EDA confirmed *why* raw training is hopeless: per-step reward ≈ **0.1001** (the survival constant); fruit-bonus steps ~0.15% of steps. Reward shaping is required and fruit/predation are the only gradient.

**The metric insight.** Under fixed-horizon grading (`core.py` `run_eval_episode(..., stop_on_death=False)`) the margin is tiny: random/dummy = 300±0 vs heuristic ≈309±10; but under survivorship it's huge (300/25 vs 210). The two hypotheses demand opposite optimizations — so identifying the grader is now a first-order driver. A fresh 3000-tick timing run of the heuristic reached **score ≈ 313.8, 474 fruits eaten** at ~265 steps/s — consistent with the fixed-horizon 309 anchor.

**The directive-driven plan.** The user's charter (`paste_1_172113.txt`) reframes this as *control/planning/optimization*, not RL: decompose into perception / threat / target-selection / steering / energy / reproduction / coordination / long-horizon planning; build a **strong deterministic + geometric-steering baseline**; then apply **failure-mode analysis**, explicit energy economics, **action chunking**, a hierarchical hybrid, MPC if fast rollouts allow, and **CMA-ES/evolutionary** tuning — optimizing mean score across seeds, robustly, not a single seed. Its 15-step priority order is the experiment roadmap.

## 6. Current status & what's next

A dedicated experimentation subagent is **actively executing** the directive's priorities — currently bootstrapping the eval environment (interpreter/venv + dependencies) and building the failure-mode analysis and rule/steering baseline; it hit the `cma` import gap and the venv-path issue, both now resolved. It will record quantitative results to `experiments/DIRECTIVE_RESULTS.md` and the winner to `experiments/best_controller.{py,json}`. `anchor_30k.py` re-anchors the heuristic at 30,000 ticks under both grading metrics for verification.

**Immediate next steps:** (1) resolve the grading-metric ambiguity via a validation run vs the provided baseline; (2) land the strong steering baseline + failure-mode taxonomy; (3) tune the winning controller with CMA-ES across presets; (4) redeploy the survivor as the served policy on 9052.

## 7. Timeline

| Phase | What | Outcome |
|---|---|---|
| 0 | Read sim source; extracted exact reward & mechanics | Reward verified: survival = constant, learnable margin = fruit−predation |
| 1 | Built harness (obs encoder, FleetEnv, multi-agent eval) | Shared 31-dim encoding; shaped reward; honest evaluator |
| 2 | Baselines + EDA (3000 ticks) | Heuristic 210 vs random 26 / dummy 22; reward ~0.1001 const |
| 3 | PPO feasibility spike (150k steps, 646s) | **74±9 — INVALIDATED** for this budget (`spikes/001/`) |
| 4 | Metric analysis + horizon correction | 3000→30,000; two opposite grading hypotheses surfaced |
| 5 | Deployed self-contained heuristic (v1) on 9052 | Smoke-tested, stdlib-only, deployable |
| 6 | Directive experiment launched (in flight) | Failure analysis, baseline, energy, chunking, CMA-ES |

## 8. Experiments & results table

| experiment | setting | metric | result |
|---|---|---|---|
| random | 5-agent, 3000 | survivorship | 25.7±5.9 (floor) |
| dummy | 5-agent, 3000 | survivorship | 22.4±6.9 |
| heuristic v1 | 5-agent, 3000 | survivorship | **210.1±106.4**, ~228 fruit, 40% alive |
| heuristic v2 | 5-agent, 3000 | fixed-horizon | ≈308, fewer fruit than v1 → regression (v2 survivorship not separately measured) |
| PPO from scratch | 150k steps / 646s | survivorship | **74.2±9.0** → invalidated |
| heuristic v1 | fixed-horizon (3000) | fixed | ≈309±10 (vs 300±0 random/dummy) |
| heuristic v1 | fixed-horizon timing (3000) | fixed | ≈313.8, 474 fruit, ~265 steps/s |
| **horizon re-anchor** | 30,000 ticks | both | **in progress** (`anchor_30k.py`) |
| directive program | failure-mode / steering / energy / chunking / CMA-ES | mean score robust | **in progress** |

*Numbers marked "in progress" do not yet exist; all others are from the on-disk baseline, spike, and EDA code listed in §9.*

## 9. Evidence files
`RL_JOURNAL.md` (journal), `spikes/001-ppo-feasibility/README.md` (spike verdict), `experiments/env_wrapper.py` / `policies.py` / `baseline_eval.py` / `compare_heuristic.py` (numbers), `../.venv/` (ML stack), directive charter `~/.hermes/pastes/paste_1_172113.txt`.