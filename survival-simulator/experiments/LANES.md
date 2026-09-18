# COMPUTE LANES — survival-simulator VPS (212.147.239.222)

Shared rule for every agent working on this project. Read this BEFORE you launch anything.

## The rule

- **Maximum TWO `nac-*` test containers at once** (2 × `--cpus=1.5` = 3 of 4 vCPU; the 4th vCPU must stay
  free for the live graded endpoint, which must never miss a per-request deadline).
- **Before launching:** run `docker ps` and count the `nac-*` test containers that are `Up`.
  If two are already `Up`, WAIT — poll every 60-120 s until one exits.
- **Then append a claim line to the table below** with your owner tag, container names, seeds, horizon.
- **Never touch, restart, rebuild or exec-mutate `nac-survival-vps`.** It serves the graded endpoint.
  Nothing may restart it while a validation is queued or running: a mid-run restart contaminates the score.
- **Never deploy.** Producing a candidate and its paired evidence is the end of an agent's job; the user
  owns validation and deployment.
- Do not edit another stream's config files or branch. One stream = one config family + one result file.

## Claim table (append, do not rewrite)

| claimed (UTC) | owner | containers | seeds | horizon | status |
|---|---|---|---|---|---|
| 2026-09-18 19:47 | orchestrator | nac-E1, nac-E2 (evade E1/E2/E3) | 1100-1109, 1110-1119 | 18000 | running |
| 2026-09-18 19:32 | orchestrator | nac-reach (reachability probe) | 1130-1132 | 18000 | finished |
| 2026-09-18 18:48 | orchestrator | nac-L1, nac-L2 (lockout factorial) | 1060-1069, 1070-1079 | 18000 | finished |
| 2026-09-18 19:32 | orchestrator | nac-diag (lockout/closure/access diagnostic) | 1060,1070,1080 | 18000 | finished |

| 2026-09-18 21:20 | W2 (thin relay) | nac-B1 (wB0 ref + wB1/wB2/wB3 + LIVE) | 1300-1319 | 18000 | running |
| 2026-09-18 21:40 | W2 (thin relay) | nac-B2 (wB0 ref + wB1 dose + wB4 mild-thin + LIVE) | 1300-1319 | 18000 | running |

## Who is in the queue

- **W3 (access arms)**: nac-A1, nac-A2 — seeds 1200-1209, 1210-1219, 4 arms, horizon 18000.
- **W2 (thin relay)**: nac-B1, nac-B2 — to be launched after W3 unless it wins the free lane first.
- **W1 (residual RL, orchestrator)**: trains on the Mac first; will claim a lane only for the final
  paired confirmation.
| 2026-09-18 21:32 UTC | W3 (access arms) | nac-A1, nac-A2 (access A1/A2/A3) | 1200-1209, 1210-1219 | 18000 | finished — LOSS (all 3 arms < H1; best A3_ars -2638.7) |
