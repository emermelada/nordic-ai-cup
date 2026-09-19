# LOCK-IN — position, evidence, and what happens next
**2026-09-19, ~01:25 UTC. Written on request ("lock in").**

Everything below is committed and pushed (`06047a6`). Nothing is deployed or changed by this document.

---

## 1. Position in the competition

| | |
|---|---|
| our best official validation | **918.33** (field median is 927.4 — we sit at the median) |
| leaderboard front | **2096.49**, then 1794, 1743, 1712, 1668, 1661, 1623; 15 teams ≥1200 |
| score mechanics | `score ≈ ticks/10`; 18,000 ticks = 1800; cap 30,000 ticks = 3000 |
| latency | serving answers in 6-20 ms; 18,000 ticks needs ≤33 ms → ~2x margin, so latency is not the binder |
| **honest summary** | no confirmed positive policy gain yet. What is locked in is infrastructure, three refutations, one mechanism with a consistent signature, and the discipline to tell them apart. |

## 2. Infrastructure (all verified in this session)

| role | state |
|---|---|
| **PRIMARY endpoint** `94.237.34.245` | Caddy + container serving `survival.zaitzev.com`, **hash f90cb4e3…**, public 200 in 18 ms, never touched tonight |
| **STANDBY** `212.147.236.122` | image built from the LIVE artifact, container created + stopped, Caddy installed/validated/disabled, **failover tested**; procedure in `tools/ops/ROLLBACK.md` (target <5 min) |
| **experiment box** | same 64 vCPU box: repo at `/opt/nac`, venv `/opt/nacv`, `sched.py` with content-addressed cache (relaunch resumes), determinism gate passing |
| **screening capacity** | 903 episodes (301 candidates × 3 seeds @4k) in **1.9 minutes** on 30 workers ≈ **28k episodes/hour** |
| guard rails | `tools/ops/latency_guard.sh` kills experiments if serving p95 > 30 ms; `_loyalty_check.py` proves recordings are the real simulation; `tools/ops/paired_stats.py` produces paired verdicts mechanically |
| old fallbacks | `212.147.239.222` and `94.237.81.173` are **both dead** (no ping/SSH) — hence the standby above |

## 3. What is proven

* Policy transfers: simulation and official grader agree, 0 errors across every validation.
* **Access-limited, not energy-limited**: world yields 55-57k energy over 18k ticks; one lineage costs
  ~3,800 (7%); fleets die with 42-129 fruit standing uneaten (15 of 16 fresh x86 episodes).
* **Mid-game extinction is modal**: peak ~20 agents, ≤2 by median tick 6,664.
* **The lockout is the killer**: agents spend a **median 54% of their lives** below the 20% sprint-lockout;
  **86% of predation deaths** (median over 16 episodes) happen there — and in 94% of those deaths the
  predator was *already visible*, so this is not a detection failure.
* **Movement dominates the budget**: income per agent ~550-830 per 1000 ticks against movement cost
  ≈76% of income; the fleet never banks a surplus (early surplus is 4.6-8.3x metabolic need and still
  gets spent). Income per agent falls *before* the population collapses in most episodes (overshoot).
* **Winners differ by behaviour, consistently** (586 winners vs 586 losers, lineage-selected, 16/16
  episodes): travel per fruit **0.34x**, time in lockout **0.36x**, fruit absorbed **3.1x**, fruit seen
  **3.2x**. Winners also inherit **5.0x** higher `max_energy` — so success is part genome, part behaviour.

## 4. What is refuted (stop spending here)

| family | result |
|---|---|
| reserve/repro energy floors (lockout) | 4/4 arms −9% to −36%, monotone dose-response |
| memory search / follow / ARS access | −18% to −50% (three independent refutations; best arm −31.2%) |
| thin relay / banking (W2) | every arm ≤ reference; the one seed where it armed: −28.5% |
| retreat-while-facing evasion (E3) | closed: 80 paired x86 seeds, **+0.4%**, 70/80 identical → originally selection on noise |
| V2 genome-aware breeder selection | +12.2% on 3 macOS seeds → **−17.2%** on 40 paired x86 seeds (CI [−2405,−80]) |
| random neural policies (300 screened) | best −8.8% cheap stage; 40-seed confirmation **−38.0%** (5/40 wins) |
| mutated nets from the best parent | −23.7% / −35.9% / −37.9% at 40 seeds |
| heterogeneous fleet roles (153 candidates) | every arm below BASE; best −6.9% |

**One shared mechanism:** every refuted family cut income or mobility in an access-limited world with a
15x energy surplus. That is now a finding, not bad luck — and it is why the current lead targets travel
efficiency instead: reduce what movement *costs per fruit*, rather than moving less.

## 5. The live bets

* **Mechanism screen (running)**: arms taking three different routes to a lower travel-per-fruit
  (stickier target hysteresis; shorter blind hops; both), plus a deliberately wrong-direction control
  that must *worsen* the metric or the screen is invalid. Pre-registered rule: **only an arm that moves
  the mechanism earns a paired survival test.** An arm that raises survival without moving the mechanism
  is not doing what made winners win and will not be believed.
* **bet B, fleet-level meta-controller (running, gen 10+)**: 8 evolved fleet genes over the global fleet
  state, zero serving cost. Independently evolving its population target *down* to ~1-3 agents and never
  arming its endgame mode. Its held-out paired confirmation decides it.
* **Winner-imitation (Stage 2)**: justified by Stage 1, but *behind* the mechanism screen — because if a
  hand-coded rule moves travel-per-fruit, we get the gain for free and skip the teacher-ceiling risk that
  Lux AI's 3rd place documented.

## 6. Next steps, in order

1. Read the mechanism screen. If the wrong-direction control fails to worsen travel-per-fruit, fix the
   screen before believing anything from it.
2. For any arm that moves the mechanism: paired survival test on ≥40 unseen seeds (minutes now).
3. bet B: read the tier-2 held-out confirmation; it is the only lane that could move the score materially.
4. If a candidate wins on both mechanism and paired survival: deploy via `tools/ops/deploy_arm.sh`
   (checks the flag actually set, regenerates the ledger, verifies from inside the running container),
   and re-validate officially several times — the board keeps the best attempt.
5. **Vision-model phases (4-6) are blocked on capability, not on code**: this model cannot see images.
   Frames can be dumped (`viewer.py --dump-frames`), but reading them needs a vision-capable model to be
   configured. That is a decision for the user, not something I can work around.

## 7. Measurement rules now enforced

* ≥40 paired unseen seeds for any claim; 20 seeds only screens.
* Baseline configs are verified key-by-key against the deployed params at run time; a missing baseline
  is a hard error, never a silent fallback (this bit us three times before it was enforced).
* Recordings are proven to be the real simulation (`_loyalty_check.py`), and that gate already caught a
  phantom-tick divergence on x86.
* Mechanism evidence is reported separately from score evidence, and inherited traits are separated from
  behaviour.
