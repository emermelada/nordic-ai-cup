# RL Journal — Survival Simulator (survival-simulator/)

Engineer: Hermes agent. Policy of record: episodic, evidence-driven. Everything below was
measured, not assumed.

## Reward mechanics (VERIFIED from source, not guessed)
From `src/elements/environment.py`:
- `env.score += dt` fires ONCE per step, regardless of agent count (line 757). dt=0.1.
  => survival bonus is **constant w.r.t. strategy** at a fixed eval horizon.
- `env.score += fruit.energy/1000` on eating fruit (line 671). fruit.energy grows to 60 -> +0.06 each.
- `env.score -= agent.energy/100` when a predator eats an agent (line 723). Eaten agents carry lots
  of energy, so predation is a big negative.
- CRITICAL: the real differentiators are FRUIT COLLECTION (+0.06) and AVOIDING PREDATION (-energy/100).
  The +dt is a constant. So it's a foraging/evasion problem, not "stay alive as long as possible."
- Agents: move_distance cap=sprint_speed(20), move_direction & turn_angle are RELATIVE angles added to
  heading, spawn_agent costs 100 energy (needs >100). Walking cost 0.05/u, sprint 0.5/u, turn cost.
  Energy drains by biome_energy_drain_rate*dt (rate=1.0 everywhere in this build). Predators spawn over
  time, chase/hunt (see predator.py). Biomes affect move_penalty (swamp 0.5, desert 0.8, river 0.3) and
  fruit/tree spawn rates. Note agent_server.py currently binds PORT 9052 (README says 8003) — check at deploy.

## Observations (per agent, from get_agent_state / StepResponse)
Keys: agent_id, observations(list), energy, biome, age, speed, sprint_speed, hearing_radius,
vision_angle(=cone), vision_range, max_energy.
`observations` entries: {type, distance, angle} + optional rel_dir, id; Edge has coords in agent-local frame.
Entity types: Fruit, Tree, Agent, Predator, Edge.

## Phase 1 — wrapper
`experiments/env_wrapper.py`:
- `build_obs(agent_state)` -> fixed 31-dim vector (scalars + biome onehot + nearest-entity feats + edge clearance).
  Shared verbatim by training and inference (matches served StepResponse fields).
- `FleetEnv` = single-agent gym env (starting_agents=1) for CLEAN training signal; shared policy reused at inference.
- `run_eval_episode()` = the REAL multi-agent world (5 agents) for scoring.
- Action (gym Box 3d): [move_distance 0..20, turn_angle -pi..pi, spawn 0..1]. move_direction pinned to 0.
- Reward shaping (default on): `score_delta - dt` so the flat survival constant is removed and only
  fruit(+)/predation(-) drive learning. TRUE score always reported during eval.

## Phase 2 — baselines + EDA (multi-agent n=5, horizon=3000 steps => 300 s sim)
| policy   | score(mean±std) | steps | fruit/eps | predated | alive% |
|----------|-----------------|-------|-----------|----------|--------|
| random   | 25.7±5.9        | 257   | 0.8       | 5.0      | 0      |
| dummy    | 22.4±6.9        | 223   | 2.4       | 10.2     | 0      |
| heuristic| 210.1±106.4     | 2007  | 227.8     | 24.2     | 40     |

EDA (heuristic, single-agent, 3000 steps): per-step reward ~0.1001 = the survival bonus. Fruit-bonus steps
0.15% of steps. => Training on raw score gives NO gradient (constant reward). Reward shaping is REQUIRED.
heuristic HUGELY beats random/dummy but is HIGH-VARIANCE (40% survive full horizon). Robustness is the lever.

## Phase 3 (in progress)
Candidate matrix (measured at 3000-step horizon before the horizon correction below):
- random (floor)              : 25.7
- dummy (env baseline)        : 22.4
- heuristic (engineered)      : 210.1  -> strong; needs variance reduction
- PPO from scratch (spike)    : 74±9   -> INVALIDATED. See spikes/001-ppo-feasibility.

## CRITICAL CORRECTION — horizon & grading metric
- Challenge directive states the simulation lasts UP TO 30,000 ticks (not 3000). HORIZON in
  env_wrapper.py is now 30000. Earlier numbers (3000 ticks) understate the real sim.
- REWARD DEPENDENCE: `score += dt` is constant per step. Two grading hypotheses:
  * FIXED-horizon grader: everyone gets horizon*0.1 base (300 at 3000 ticks / 3000 at 30000).
    The win margin is ONLY fruit(+6e-2 each) minus predation(-energy/100). Tiny but scales with time.
  * SURVIVORSHIP grader (episode ends when team wiped): survival dominates hugely.
  These imply OPPOSITE optimization targets. As of now we cannot tell which the grader uses.
  Resolution path: queue a validation with the heuristic and compare its score vs the provided
  baseline (3000-tick anchor gave heuristic 309-fixed/210-survivorship vs random 300/25).

## Deployed candidate (Phase 6, ready to validate)
- agent_server.py now runs a self-contained heuristic (policy_heuristic.py, = v1 behavior, seek fruit
  -> seek tree -> wander w/ biome reverse; spawn step gated to rich+undangerous). Smoke-tested vs a
  realistic StepResponse: returns valid {"actions":[...]}, seek/flee/spawn branches verified.
- Container listens 9052 (compose 9052:9052); needs only stdlib -> requirements.txt unchanged.
- To deploy: docker compose up -d --build survival-simulator ; curl localhost:9052/ ; cloudflared tunnel.
- LIVE as of now: container healthy, /predict 200, tunnel nac-survival up under caffeinate, public
  https://survival.zaitzev.com verified 200 with a real 5-agent payload -> 5 valid actions.
  (Note: Cloudflare 403s Python-urllib User-Agent on POST; curl/normal clients get 200. Not a bug.)

## Directive experiment (in flight)
- A subagent (sa-0-49fb4291) is executing the paste directive priorities 1-8 (failure analysis,
  strong steering/rule baseline, energy economics, action chunking, evolutionary (CMA-ES) tuning).
  Charter: /Users/zaitzev/.hermes/pastes/paste_1_172113.txt. It will save winner + params to
  experiments/best_controller.{py,json} and experiments/DIRECTIVE_RESULTS.md.
- anchor_30k.py re-anchors heuristic at 30000 ticks (both grading metrics) for verification.

## 30k re-anchor (heuristic, seeds 100-500, horizon=30000) — KEY INSIGHT
- FIXED-horizon: random 3000.0±0.0, heuristic 3008.3±8.2 (fruit 243, predated 36.4). Constant 3000 still
  dwarfs everything; heuristic beats do-nothing by only ~8 points at 30k. ANY fixed-horizon win is razor-thin and
  demands vastly more fruit intake + far less predation.
- SURVIVORSHIP: heuristic 349.2±213.8, team-wipe ticks [4893,5906,856,4275,1012] (~4-6k of 30k), fruit 391,
  predated 58. All teams wiped. Huge headroom: surviving another +10k ticks ≈ +1000 score.
- CONCLUSION THAT RESOLVES THE METRIC AMBIGUITY: the dominant failure is POPULATION COLLAPSE near ~4-6k ticks.
  A policy that keeps the population alive longer (reliable foraging + predator evasion + energy + reproduction)
  improves BOTH metrics (longer survival -> more fruit & less collapse -> higher fixed score AND much higher
  survivorship score). So we do NOT have to pick a grading mode: target long-population-survival. This is
  exactly what the directive subagent is working on.
- Verification baseline for the directive winner: heuristic = 3008.3 fixed / 349.2 survivorship at 30k.

## Directive result (subagent sa-49fb4291, COMPLETE) + deployment
- Winner: experiments/best_controller.py (geometric potential-field steering: fruit attract, inverse-square
  predator repel, wall repel, agent disperse, wander; flee-override w/ hysteresis; energy economics;
  reproduction-as-investment w/ cooperative global pop cap). 13 params evolved by (1+λ)-ES in evolve.py.
- Full-30k, seeds 100-500: survivorship score median heuristic 270.1 -> best_controller 548.8 (+103%);
  median survival ticks 2646 -> 5202 (~1.9x). Fixed-30k 3008.7 -> 3013.1. Neither survives full 30k.
- DEPLOYED: best_controller.py + best_controller/params.json copied to survival-simulator/, agent_server.py
  now uses it (fallback = policy_heuristic). Verified in-container active policy = best_controller(potential-field, evolved).

## Grader reality + instrumentation (CRITICAL)
- Validation attempt #1 (heuristic): score 78, clean. attempt #2: 47.8 BUT with a 502 error (endpoint went
  down mid-run - container restart) -> contaminated, ignore. Never rebuild while an attempt is queued.
- score == sim_time (verified: after 50 ticks sim=5.0 score=5.0). So the grader scores TICKS SURVIVED.
  Our 78 = ~780 ticks; leaderboard top 1500 = ~15,000 ticks. Grader is ~4x harsher than our local sim
  (heuristic survives ~2646 ticks locally vs ~780 in the grader). Chat: top scorers ~1k+, best 1500.
- Added grader instrumentation: agent_server logs every request to /app/predict_log.jsonl
  (sim, score, n, game_status, obs entity counts, energy, spawn). GET /debug/stats shows log_lines.
  Analyzer: experiments/analyze_grader_log.py. Plan: queue attempt #3, dump the log, decode the real
  environment (predator pressure, energy collapse, true collapse tick) and optimize against THAT.

## ATTEMPT #3 RESULT + ROOT CAUSE (major)
- score 580.96 (heuristic was 78 -> 7.4x). BUT ended with error: "Game over due to agent server
  bottleneck. Accumulated wait time for agent exceeded 600 seconds".
- Log analysis (5452 requests, /tmp/predict_log.jsonl): sim 0->545.1, n_agents 1-14 (median 8),
  team NEVER collapsed, energy healthy (85->155, max 411), predators ~1/tick, our spawns 50.
- => 5452 requests / 686 s = ~110 ms per HTTP round-trip. The grader caps ACCUMULATED wait at 600 s,
  so score ~= 600s / latency. 110 ms -> ~5.4k ticks (580). 40 ms -> ~15k (1500). 20 ms -> 30k cap (~3000).
- CONCLUSION: the policy is now good enough (never dies); the binding constraint is LATENCY, dominated
  by the Cloudflare tunnel -> home Mac uplink (our endpoint answers in 0.8 ms locally). NEXT LEVER:
  host the service close to the grader / drop the tunnel hop (e.g. the Frankfurt Ubuntu VPS with a
  public IP). Ticks-to-score is now the whole game.

## Latency measured + VPS bundle ready
- experiments/_latency_probe.sh (40 reqs through the public path): min 72ms, median 80ms, p90 163ms,
  mean 105ms, max 280ms. Endpoint answers 0.8ms locally -> ~all of it is the tunnel + home-uplink leg
  (and jittery). A datacenter origin sits ~1-5ms from the CF edge.
- Built + validated a minimal serving image: Dockerfile.vps + requirements-vps.txt (fastapi/uvicorn/
  pydantic/numpy only, no pygame/scipy/shapely). Built clean, ran on :9079, GET / reports the evolved
  controller, POST /predict OK. Runbook: vps_deploy.md (Option A: run the named tunnel ON the VPS;
  Option B: CF proxied A record straight to the VPS; Option C: direct TLS). Mac service left live as fallback.

## cloudflared tuning — DEAD END
- A/B on the live hostname (15 POSTs each): quic median 60 / mean 71 ms; http2 median 62 / mean 73 ms.
  Protocol makes no difference; the ~60-75 ms tunnel leg is inherent to hosting behind the tunnel on this
  connection. Home link to the CF edge is fine (14-18 ms); the extra ~50 ms is the edge->cloudflared->Mac
  round trip. Note the Mac serves on Wi-Fi (en0); Ethernet adapters en4/en5/en6 are unused.
- Remaining no-VPS options: (1) wire Ethernet + retest; (2) Cloudflare WAF skip rule for /predict;
  (3) run the policy at the CF edge as a Worker (port controller to JS) -> removes the tunnel leg entirely,
  latency ~ grader->edge only. User deferred the decision for now.
- After the tuning A/B the tunnel is running un-caffeinated (nohup). Re-run scripts/up.sh for the
  sleep-protected persistent tunnel before queuing attempts.

## VPS deploy (UpCloud) — BLOCKED on cloudflared
- VPS: UpCloud (sys_vendor=UpCloud), ubuntu-1cpu-1gb-de-fra1, Ubuntu 26.04, 1 vCPU/843MB, disk 91% full,
  public IP 94.237.81.173, Docker 29.7.2. SSH: ssh -i ~/.ssh/vps_hermes root@100.105.61.1 (works over Tailscale).
- Deployed: rsync'd minimal files to /opt/nac-survival, built nac-survival-vps, container running on :9052
  (origin 200, evolved controller). NOT yet publicly reachable.
- BLOCKER: UpCloud blocks OUTBOUND port 7844 (quic UDP + http2 TCP) -> cloudflared cannot connect
  ("Allow outbound QUIC traffic on port 7844 or use HTTP2"). Local firewall is permissive (OUTPUT ACCEPT);
  it's upstream. 443 is open.
- Routing options (no tunnel = also faster): (A) open outbound 7844 in UpCloud firewall (no DNS change);
  (B) Cloudflare proxied A record survival.zaitzev.com -> 94.237.81.173 on an allowed origin port
  (e.g. 8080); (C) direct A record + Caddy TLS on 443 (fastest). B/C need a Cloudflare DNS change.
- User also offered a new CPH VPS (2 cores/4GB/50GB) which may be closer to the grader + more headroom.
- Note: cloudflared flags are GLOBAL and must precede the subcommand (`cloudflared --config X tunnel run`);
  --config/--edge-ip-version are global; there is NO --protocol flag on this version (2026.9.1).
- Mac tunnel restored (nohup) so the public URL is back to 200 meanwhile.

## File map
- experiments/env_wrapper.py  (obs encoder, FleetEnv, run_eval_episode)
- experiments/policies.py     (random/dummy/heuristic policy fns; action = [dist,move_dir,turn,spawn])
- experiments/trainer.py      (PPO train + multi-agent true-score eval -> runs/<run>/)
- experiments/baseline_eval.py (Phase 2 table + EDA)