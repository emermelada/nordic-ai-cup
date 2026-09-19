"""Potential-field / energy-aware steering controller (directive priorities 1-4).

contract: fn(agent_state_dict) -> [move_dist, move_dir(rel), turn_angle, spawn_flag]

Design (measured against source sim money-line mechanics):
  - score += dt is constant; the ONLY learnable margin = fruit(+~0.06) minus predation (-energy/100,
    big). So predator avoidance dominates; foraging sustains energy; reproduction keeps species alive.
  - Steering is a geometric potential field built directly from observation {distance, angle} vectors:
      desired_vec = w_f * sum(fruit attract) - w_p * sum(predator inverse-square repel)
                    + w_w * sum(wall repel) + w_a * sum(other-agent disperse) + w_x * wander
      steer = atan2(vec)
  - Flee override with hysteresis: if closest predator < flee_dist (or inside danger cone with
    rel_dir facing us), commit to FLEE (sprint straight away from the aggregate predator vector)
    until the predator is far enough away -> no tick-to-tick oscillation.
  - Move distance: sprint only when fleeing and energy allows; otherwise walk (cheap 0.05/unit).
  - Energy economics: below reserve_frac stop sprinting; below low_energy_frac prefer the nearest
    safe fruit and cut walking; never let projected living+move cost push energy to 0.
  - Reproduction: only when energy > repro_frac*max AND no predator within big radius AND not already
    crowded locally AND spawn cooldown elapsed -> treat spawn as an investment, not a habit.
All parameters are tunable (dict) so the policy is optimizeable (priority 8).
"""
import math, random
import numpy as np

# -----------------------------------------------------------------------------
# default parameterization (hand-tuned pre-search; refined by evolution)
# -----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # steering
    "fruit_weight": 1.0,       # attraction toward fruit
    "predator_weight": 3.0,    # inverse-square repulsion scale
    "wall_weight": 1.6,        # repulsion from visible edge points
    "disperse_weight": 0.6,    # repulsion from other agents (avoid clustering)
    "wander_weight": 0.12,     # exploration noise when food is scarce
    "target_hyst": 0.5,        # fruit target hysteresis (angle stickiness, radians)
    # predator
    "danger_dist": 180.0,      # repulsion begins this far out
    "flee_dist": 120.0,        # below this distance -> commit to sprint flee
    "concern_cone": 2.2,       # rad tolerance to treat FRO (facing-ish) predator as threat early
    "flee_speed_frac": 1.0,    # fraction of sprint_speed while fleeing
    "escape_dist": 200.0,      # hysteresis: stop fleeing only once predator beyond this
    # forage
    "fruit_risk_penalty": 1.2, # fraction weighting: avoid fruit in predator direction
    "walk_frac": 1.0,          # walking move distance = walk_frac*speed (<=1 => walking)
    # --- DISCOVERY-DERIVED PHASE SPEED RULE (OFF by default; see the move-distance block) ---
    # Counterfactual boundary analysis found: with fruit in sight, WALK before tick 900 and SPRINT in the
    # mid-game (900-2700). A state-dependent rule, not a tunable value - no single walk_frac beats the
    # baseline, which is why the phase gate exists.
    "psp_mode": 0.0,           # 0 = current behaviour; 1 = enable the phase speed rule
    "psp_lo": 900.0,           # tick at which the mid-game sprint window opens
    "psp_hi": 2700.0,          # tick at which it closes
    "psp_mid_frac": 1.0,       # speed as a fraction of sprint_speed inside the window
    "explore_frac": 0.55,      # wandering speed fraction
    # energy
    "reserve_frac": 0.15,      # below this single-agent fraction of max: no sprint, minimal walk
    "low_energy_frac": 0.35,   # below this: prioritize nearest safe fruit, cut speed 40%
    "boom_frac": 0.28,         # maximum sustainable walking+living energy burn per tick (not used directly)
    # reproduction
    "repro_frac": 0.82,        # energy fraction required to spawn
    "repro_safe_radius": 330.0,# no predator within this radius to allow spawn
    "repro_popcap": 2,         # max OTHER agents observed before refusing to spawn (dispersion)
    "repro_global_target": 10, # cooperative: don't spawn while estimated global population >= this
    # --- CARRYING-CAPACITY PHASE SCALING (OFF by default: cc_k0 = 0.0) ---
    # The receipt for mid-game death (measured, budget.py): production per 1k ticks falls
    # 9,983 -> 6,963 -> 4,882 while spawns stay 12.5-16.0 per 1k, so the fleet ends up ~75% over the
    # measured sustainable N (7.2 agents vs 5.9 at 6-9k; 6.3 vs 3.6 at 9-12k) and drains its standing
    # stock (+1,288 early, then -435, -213, -194). Production halves every 3,000 ticks, so the
    # reproduction TARGET must halve on the same schedule. cc_k0 = the target at tick 0 (12 matches
    # the fleet's current early size); cc_floor keeps a lineage alive when the target decays.
    "cc_k0": 0.0,              # 0 = OFF. Set to ~12 to enable the carrying-capacity schedule.
    "cc_halflife": 3000.0,     # ticks for the target to halve (= the world's production halflife)
    "cc_floor": 2.0,           # never drive the reproduction target below this
    "cc_margin": 1.0,          # safety margin on the measured sustainable population
    # --- BIOME-AWARE TERRAIN HANDLING (OFF by default: bio_mode = 0.0) ---
    # The observation payload carries `biome` (src/utils/DTOs.py: "biome: str") and this controller has
    # never used it. MEASURED map mix: desert 27.9% + river 5.9% produce ZERO fruit; grassland 16.9% is
    # the best (0.1 fruit/s per 100x100). With agents blind 76-87% of ticks and starvation our largest
    # death category (32-51%), bio_mode stops paying 100 energy to spawn a body into dead terrain, and
    # crosses dead terrain fast when nothing is visible.
    "bio_mode": 0.0,           # 0 = OFF. Set to 1.0 to enable biome-aware spawning + dead-terrain transit.
    "bio_sprint_frac": 0.85,   # speed fraction used to cross zero-production terrain with nothing visible
    "scan_mode": 0.0,          # OFF. 1.0 = sweep-while-approaching fruit (cheap LOCAL information)
    "scan_period": 10.0, "scan_ticks": 4.0, "scan_slow": 0.25, "scan_turn": 0.45,
    "spawn_cooldown": 400,     # ticks between spawns (anti-overpopulation)
    "spawn_cap": 14,           # hard cap: never spawn while local(other) agents >= this
    # --- population-maintaining reproduction (2026-09-17 survival fix) ---
    "repro_frac_min": 0.24,    # floor for the energy gate when pop is far below target (sim needs >100)
    "repro_urgency": 1.0,      # 0 = flat repro_frac, 1 = scale gate down to repro_frac_min as pop->0
    "repro_unlimited": 0.0,    # 1 = ignore gpop/popcap gate entirely (rely on energy + cooldown only)
    # --- foraging efficiency (energy income; movement burn dominates the energy budget) ---
    "forage_nearest": 0.0,     # 1 = when fruit is visible, steer straight at the nearest safe fruit
    "forage_speed": 1.0,       # move-distance fraction used by the nearest-fruit override
    # --- search behaviour when BLIND (no fruit in view) ---
    # Measured on the deterministic harness: in the endgame the world holds 3-4x more fruit energy
    # than the whole fleet owns, yet agents see only 0.3-0.6 fruits each and in 12-29% of ticks see
    # none at all -> income is ACCESS-limited, not supply-limited. Trees are the fruit source in this
    # sim (forest fruit_spawn_rate 0.08/s per 100x100), so these knobs steer the SEARCH.
    # Both default to the pre-existing behaviour: 0.25 was the hardcoded tree weight, and None means
    # "keep using explore_frac", so adding them cannot change any already-measured result.
    "tree_weight": 0.25,        # attraction to Tree observations (0.25 = the original hardcoded value)
    "blind_explore_frac": None, # move-distance fraction when NO fruit is visible (None = explore_frac)
    # --- MEMORY-BASED SEARCH: the optimal-foraging behaviours the controller never had. ---------
    # WHY (and why this is the interesting fix): our controller is a MEMORYLESS potential field --
    # chase the nearest visible fruit, otherwise wander randomly. Foraging ecology says that is the
    # wrong shape for a patchy world, and our own measurements agree: income is ACCESS-limited
    # (4,518 fruit energy standing in the world vs 1,109 eaten; 0.3 fruits visible per agent;
    # 12-29% of ticks with NO fruit in view), so the binding problem is SEARCH, not appetite.
    # The ML track independently reached the same conclusion from the other side: behaviour cloning
    # of this controller failed because the observation is not Markov -- the missing state is
    # exactly "where was the food I saw earlier". So give the heuristic that memory directly.
    #   1. BALLISTIC RELOCATION (Levy-like): after losing sight of food, COMMIT to a straight run in
    #      the remembered food direction instead of re-randomising the heading every tick, then
    #      commit to a fresh straight run -- cover ground instead of milling in place.
    #   2. SOCIAL FORAGING: when blind, steer toward a visible conspecific (the observation already
    #      carries other agents' bearings and we ignored them entirely).
    #   3. AREA-RESTRICTED SEARCH: while food IS visible, move slower so the agent stays inside the
    #      patch it just found rather than walking straight out of it.
    "memory_search": 0.0,       # 0 = off (exactly the old behaviour); 1 = enable the block below
    "commit_len": 250.0,        # ticks to hold the remembered heading before re-committing
    "commit_speed_frac": 1.0,   # speed fraction during a committed relocation run
    "follow_agent_weight": 0.0, # 0 = off; >0 blends toward the nearest visible agent when blind
    "ars_speed_frac": 1.0,      # 1 = off; <1 slows movement while food is visible (stay in patch)
    # --- PREDATOR-FACING DEFENCE (face_predator 0.0 = OFF) -------------------------------------
    # MEASURED IN THE SIM (src/elements/predator.py:37): a predator CHARGES only when the agent is
    # NOT facing it (|agent_looking_dir| > pi/2) or when it is already inside hearing_radius*1.5 =
    # 90 units. If the agent IS facing it, the predator instead PIVOTS 45 degrees off-target and
    # does not close. Turning costs |turn|/(2pi) energy (a full 180deg turn = 0.5) against 5.5/tick
    # to sprint away, and movement is applied BEFORE the turn (environment.py:614/618), so an agent
    # can hold a pursuer inside its vision half-plane while still travelling toward fruit.
    # Predators never tire (Predator.step sprints with NO energy accounting), so kiting is
    # impossible -- facing is the cheap counter to the terminal predation cascade that ends runs.
    # Death budget measured on the live controller: 35-43% of deaths are predation, and the fleet
    # banks ~3,000 energy then loses 14 of 16 agents in a single 1,000-tick window, each kill
    # carrying off that agent's bank. That cascade, not gradual starvation, ends the run.
    "face_predator": 0.0,       # 0 = off; 1 = keep the nearest predator inside the facing cone
    "face_cone": 1.05,          # half-plane kept clear, radians (~60 deg)
    "face_dist_max": 320.0,     # only face predators within this range
    "face_min_dist": 90.0,      # closer than this it charges anyway, so facing cannot help
    # --- RETREAT-WHILE-FACING EVASION (evade_mode 0.0 = OFF, i.e. byte-identical behaviour) -----
    # MEASURED, not inferred (w_lockout_diag.py on the deployed H1, 3 seeds, official 18k horizon):
    # while the agent FACES a nearby predator the gap OPENS at 1.583 / 1.359 / 0.477 units per tick;
    # while turned away it is -0.126 / +0.091 / +4.221, i.e. neutral to CLOSING. Facing is worth
    # ~1.6 units/tick of separation. Mechanism (predator.py:37): a predator only CHARGES when the
    # prey is not looking at it (|rel_dir| > pi/2) or is closer than 90; otherwise it PIVOTS 45 deg
    # off-target, which closes radially ~29% slower before the agent's own motion is counted.
    # Why this is worth testing where the reserve/gate arms failed: it does NOT throttle income. It
    # changes the DIRECTION and FACING of an escape that already happens -- movement distance stays
    # at sprint speed, so the agent keeps covering ground while separating.
    # Retreated agent: move relative (bearing + pi) = directly away from the predator, then turn by
    # -bearing so the predator ends up dead ahead (movement is applied BEFORE the turn, so both
    # happen in one tick). Turn cost |turn|/(2pi) <= 0.5 against ~3.5/tick for the sprint itself.
    "evade_mode": 0.0,          # 0 = off; 1 = retreat while facing when a predator is close
    "evade_dist": 140.0,        # engage when the nearest predator is within this
    "evade_disengage": 240.0,   # hysteresis: keep retreating until the predator is beyond this
    "evade_speed_frac": 0.8,    # fraction of sprint_speed (16 units must beat the predator's 15)
    "evade_energy_abs": 130.0,  # only retreat above this energy: the sim forbids sprinting < 100
    # --- PHASE-AWARE boom/famine policy (phase_mode 0 = OFF, i.e. exactly the previous behaviour) ---
    # Why this exists: fruit production decays 0.5^(t/300) at the POPULATION level -- a global property
    # no local observation can see -- while income is ALSO biome-dependent (forest 0.08/s per 100x100,
    # desert 0.05, river 0). So the schedule is TIME-primary with an adaptive food-availability guard.
    # Reaching ~18k ticks needs agents that ARRIVE at the famine FULL: metabolism alone (0.1/tick)
    # drains ~1,200 energy over 12,000 ticks, and max_energy caps the bank at 500 (mutated: up to 1000).
    "phase_mode": 0.0,          # 0 = off; scales the whole phase effect (1 = fully active)
    "famine_tick_lo": 10000.0,  # tick at which consolidation STARTS (must NOT be mid-boom: the
    "famine_tick_hi": 14000.0,  #   first attempt fired at 4,000-7,500 and killed income+relay)
    "famine_vis_guard": 0.0,    # >0: visible food DELAYS the famine (0 = pure time; try 0.4-0.7)
    "famine_move_frac": 0.6,    # movement scale deep in the endgame: do NOT stop dead, movement
                                #   is the INCOME mechanism (stopping was a measured failure)
    "famine_blind_stop": 0.0,   # extra stop when famine AND no fruit visible (0 = don't)
    "famine_pop_target": 3.0,   # legacy gate (superseded by the rescue/invest rule below)
    "famine_min_pop": 2.0,      # RELAY RESCUE: below this population, spawn regardless of banking
                                #   (fixes the ablation's fatal flaw: a fleet that stops breeding
                                #   under age mortality simply dies)
    "famine_bank_frac": 0.9,    # INVEST: otherwise, spawn only from a nearly-full agent...
    "famine_min_cap": 0.0,      # >0: ...whose max_energy >= this (heritable lineage cap; the cap
                                #   is what sets how much energy a survivor can bank for the endgame)
    # --- generational relay (age-aware: survive past max_age by banking heirs) ---
    "relay_age": 0.0,          # sim seconds; spawn an heir once older than this (0 = off)
    "relay_energy_frac": 0.35, # energy gate used by the relay spawn (fraction of max_energy)
    "relay_cooldown": 700,     # ticks between relay spawns
    "repro_energy_abs": 0.0,   # ABSOLUTE energy gate (sim requires >100); 0 = use fractional rf
    # --- age-based roles, fully decentralised (each agent only knows its OWN age) ---
    "role_by_age": 0.0,        # 1 = young agents forage hard, old agents conserve
    "young_age": 25.0,         # sim seconds below which an agent is a "young forager"
    "old_age": 60.0,           # sim seconds above which an agent is "old" (conserve + bank heir)
    "young_speed_frac": 1.0,   # movement scale for young agents
    "old_speed_frac": 0.5,     # movement scale for old agents (they are a dying investment)
    # --- STATE-TRIGGERED THIN-RELAY ENDGAME (thin_relay 0.0 = OFF, i.e. byte-identical) --------
    # WHY THIS SHAPE, and why it is not another "bank late" arm:
    #   MEASURED: the fleet dies at 4,624 / 4,896 ticks with 6,469 / 6,505 energy of fruit STILL
    #   STANDING. Cost of keeping ONE lineage alive (0.1/tick metabolism + one 100-energy spawn per
    #   ~900 ticks = 0.211/tick = 3,800 energy over 18,000 ticks) is 7% of the 55,405/57,072 energy
    #   the world yields -> the endgame is NOT energy-limited, it is ACCESS-limited (agents are blind
    #   76-87% of ticks and see 0.26-0.42 fruits each). Total production over 18k ticks is 55k-57k.
    #   The four earlier 'bank late' arms FAILED because their trigger windows (tick 8,000-15,000)
    #   were NEVER REACHED (fleet dead by ~4,600) and/or a population gate blocked ALL reproduction,
    #   which broke the age-relay and killed the fleet. Both failure SHAPES are avoided here:
    #     (a) the time threshold is LOW (4,000-6,000, not 8,000+) AND gated on a state signal, so the
    #         mode can actually engage before the terminal cascade;
    #     (b) reproduction is never blocked - see the THIN RELAY override in the spawn gate. A spawn
    #         is the ONLY thing that keeps a lineage alive past max_age = 60+U(0,60) s.
    #   The two things that actually end runs (per this session's measurements): age mortality past
    #   one lifetime and a mid-game predation cascade (19 -> 15 -> 6 -> 0 agents inside ~600 ticks).
    #   A TINY fleet is the right answer to BOTH: fewer bodies = fewer -energy/100 predation
    #   liabilities and less 0.1/tick burn, while ONE unbroken heir chain still accrues +0.1/tick.
    "thin_relay": 0.0,          # 0 = off (exactly the old behaviour); >0 enables the block below
    "thin_trig_tick": 6000.0,   # (a) reconstructed-sim-time gate, in TICKS (_SIM_TICK, the existing
                                #     estimator). Deliberately below the 8,000+ windows that were
                                #     measured to be unreachable: the fleet is usually dead by ~4,600.
    "thin_trig_vis": 0.35,      # (b) arm when the fleet EMA of FRUITS-VISIBLE-PER-AGENT falls below
                                #     this. MEASURED collapse values in losing runs: 0.26-0.42 fruits
                                #     visible per agent (vs 3-4x more fruit energy than the fleet owns
                                #     standing in the world) -> the signal is ACCESS, not supply.
    "thin_trig_vis_off": 0.70,  # HYSTERESIS: once armed, stay armed until the EMA recovers above
                                #     this. Without a second threshold the mode would flap on/off
                                #     tick-to-tick and the spawn cadence would never settle.
    "thin_target_pop": 2.0,     # fleet size the relay keeps (1-3 agents). Population is NOT a score
                                #     multiplier (score += dt once per tick regardless), so every extra
                                #     body is pure cost: 0.1/tick metabolism + 100-energy spawn + an
                                #     edible -energy/100 liability during a predation cascade.
    "thin_spawn_energy_abs": 250.0,  # ABSOLUTE parent energy gate (the sim requires >100 to spawn,
                                #     environment.py:621). 250 => the parent keeps >=150, comfortably
                                #     above the sprint lockout at max_energy/5 (environment.py:512
                                #     clamps distance to walking speed below that: 100 for the base
                                #     500 cap) so the heir-maker stays mobile.
    "thin_rescue_energy_abs": 160.0, # EMERGENCY gate when the fleet is down to its LAST agent:
                                #     spawn even though the parent is left at ~60, i.e. sprint-locked.
                                #     Rationale: a lone sprint-locked survivor that banks an heir still
                                #     accrues +0.1/tick; a lone survivor that does NOT bank one scores
                                #     ZERO from its death onward. Survival beats speed when pop <= 1.
    "thin_spawn_cooldown": 200, # ticks between thin-relay spawns (age deaths land at 600-1,200
                                #     ticks, so 200 is fast enough to keep a chain unbroken without
                                #     letting a single rich parent flood the world)
    "thin_move_frac": 0.7,      # move-distance scale while active: CONSERVE but never stop --
                                #     movement is the INCOME mechanism. 'Stop dead' was measured to
                                #     lose thousands of ticks; the floor below is a hard guarantee.
    # --- GENOME-AWARE BREEDER SELECTION (genome_select 0.0 = OFF, i.e. byte-identical) -----------
    # WHY THIS MECHANISM CLASS: a birth copies six heritable traits from the parent and mutates each
    # with p=0.1 by U(0.5,1.5), capped at 2x the defaults (environment.py:296-345). The policy
    # chooses who spawns, so reproduction is a SELECTION CHANNEL the simulator hands over for free.
    # MEASURED on the deployed controller (w_traits.py, seed 100, 7,292 ticks, 119 births): newborns'
    # max_energy reached mean 589.7 / max 996.8 (cap 1,000, default 500) BY ACCIDENT -- richer parents
    # simply spawn more often -- while vision_range stayed FLAT (mean ~200, max 257.8, cap 400),
    # because no part of the controller prefers a better-sighted breeder. An unbiased +/-50% mutation
    # is a random walk; only selection turns it into a ratchet, and selection is a rule, not a
    # training run.
    # THE TRAITS ARE OBSERVABLE, which is what makes this implementable: the request carries the
    # agent's OWN six traits (DTOs.py:14-20) and every observed conspecific carries its `id`
    # (creature.py:173, include_id=True -> {'type','distance','angle','rel_dir','id'}, and NOTHING
    # else). So an id->traits table maintained from our own calls reconstructs the whole fleet's
    # genome with no new input channel -- and the same table answers "who is a good breeder".
    # COST: zero energy. It only decides who may spend the 100-energy spawn (it never blocks the
    # relay at pop <= gs_rescue_pop, the failure that killed four earlier arms).
    "genome_select": 0.0,       # 0 = off; >0 = a spawn requires the parent to rank as a breeder
    "gs_w_vision": 1.0,         # trait weights in the utility U, each normalised by its cap
    "gs_w_cone": 0.6,
    "gs_w_hearing": 0.3,
    "gs_w_energy": 0.4,         # max_energy: the bank that sets how long a survivor outlasts the food
    "gs_w_speed": 0.2,
    "gs_w_sprint": 0.2,
    "gs_topk": 2.0,             # spawn only if own U ranks within the fleet's top-k (1 = best)
    "gs_mode": 1.0,             # 1 = rank gate (top-k only). 2 = IMPROVING gate: any agent whose U
                                #     is above the LIVING FLEET's mean U may breed -- weaker
                                #     selection pressure but ~half the fleet still breeds, so the
                                #     ratchet gets far more draws per 1,000 ticks. Which matters more
                                #     (pressure vs number of draws) is a measurement, not an opinion.
    "gs_min_known": 3.0,        # need at least this many tracked genomes before gating at all
    "gs_rescue_pop": 1.0,       # gpop <= this => never gate (age mortality makes births load-bearing)
    "gs_phase_tick": 6000.0,    # after this sim tick the objective shifts toward the BANK
    "gs_late_energy_mult": 3.0, # multiplier on gs_w_energy past gs_phase_tick (capacity > perception
                                #   once production has halved 4x -- the endgame is a bank race)
    "gs_ttl": 60,               # epochs a tracked genome counts as alive (same TTL as _GC)
}

# module-level memory: per-agent-id last action / flee state / spawn clock.
# Persists across ticks inside one episode (fresh process), keyed by agent_id.
_MEM = {}
_EPOCH = 0
# --- simulated-time reconstruction: `age` is in sim-seconds and every agent is called exactly once
# per tick, so time can be recovered exactly through generations (the state carries NO clock). ---
_SEEN = {}        # agent_id -> [birth_tick, calls_since_birth]
_SIM_TICK = 0     # best estimate of the current simulated tick
_BLIND_EMA = 0.0  # fleet-level EMA of the share of calls that saw NO fruit (adaptive guard)
_FRUIT_EMA = 0.0  # fleet-level EMA of FRUITS VISIBLE PER AGENT (thin-relay state trigger)
_THIN_ARMED = False  # thin-relay hysteresis latch (set/cleared only by the state trigger)
_GC = {}  # agent_id -> last-seen epoch (pruned via TTL -> cheap cooperative population estimate)
_AGE = {}  # agent_id -> last age (monotonic within an episode; a drop => new episode => reset memory)
# --- genome observability (breeder selection): agent_id -> [vision, cone, hearing, max_energy,
# speed, sprint, last_epoch]. Built ONLY from our own calls (each request carries the agent's own
# traits), which is what makes a conspecific's genome readable: the observation gives us its `id`.
_TRAIT_KEYS = ("vision_range", "vision_angle", "hearing_radius", "max_energy", "speed", "sprint_speed")
_TRAIT_CAPS = (400.0, math.pi / 2.0, 100.0, 1000.0, 20.0, 40.0)
_TRAITS = {}
_GS_LAST = None  # telemetry: (own U, own rank, n_known) from the last gated call -- for probes only


def _maybe_new_episode(aid, age):
    """Age is monotonic within one episode, so an age decrease means the world reset.
    Purge all cross-agent memory on reset so consecutive episodes are independent."""
    global _EPOCH
    prev = _AGE.get(aid)
    if prev is not None and age < prev - 10.0:
        _MEM.clear()
        _GC.clear()
        _AGE.clear()
        _EPOCH = 0
        global _THIN_ARMED, _FRUIT_EMA
        _THIN_ARMED = False
        _FRUIT_EMA = 0.0
        _TRAITS.clear()          # genome table is per-episode (ids restart every episode)
    _AGE[aid] = age


def reset_memory():
    """Hard reset of ALL module-level policy state. Call at the START of every episode.

    Without this, evaluation is ORDER-DEPENDENT: `_MEM`/`_GC`/`_AGE`/`_EPOCH` are module globals,
    so an episode inherits the previous episode's per-agent memory and population estimate. The
    age-drop heuristic in `_maybe_new_episode` misses cases, which made the SAME seed produce
    different survival (e.g. 9362 vs 7581 vs 11873 ticks) depending on which seeds ran before it —
    silently biasing every multi-config sweep by config position."""
    global _EPOCH
    _MEM.clear()
    _GC.clear()
    _AGE.clear()
    _SEEN.clear()
    global _SIM_TICK, _BLIND_EMA
    _SIM_TICK = 0
    _BLIND_EMA = 0.0
    global _THIN_ARMED, _FRUIT_EMA
    _THIN_ARMED = False
    _FRUIT_EMA = 0.0
    _TRAITS.clear()
    _EPOCH = 0


def _genome_u(rec, w):
    """Scalar genome utility: traits normalised by their caps, weighted. -1.0 for an unknown agent."""
    if rec is None:
        return -1.0
    u = 0.0
    for i in range(6):
        c = _TRAIT_CAPS[i]
        v = rec[i] / c
        if v > 1.0:
            v = 1.0
        elif v < 0.0:
            v = 0.0
        u += w[i] * v
    return u


def _trait_weights(P):
    """Utility weights, shifted toward the BANK (max_energy) once production has decayed."""
    w = [float(P.get("gs_w_vision", 1.0) or 0.0), float(P.get("gs_w_cone", 0.6) or 0.0),
         float(P.get("gs_w_hearing", 0.3) or 0.0), float(P.get("gs_w_energy", 0.4) or 0.0),
         float(P.get("gs_w_speed", 0.2) or 0.0), float(P.get("gs_w_sprint", 0.2) or 0.0)]
    if _SIM_TICK > float(P.get("gs_phase_tick", 6000.0) or 0.0):
        w[3] = w[3] * float(P.get("gs_late_energy_mult", 1.0) or 1.0)
    return w


def _global_alive(ttl=60):
    dead = [aid for aid, ep in _GC.items() if _EPOCH - ep > ttl]
    for aid in dead:
        _GC.pop(aid, None)
    return len(_GC)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _det_rand(i):
    """Deterministic pseudo-random in [0,1) from an integer (xorshift-ish).
    Replaces random.uniform for exploration: an unseeded RNG made the policy's search path
    non-reproducible run-to-run (same sim seed -> different collapse tick, +/-1.5k ticks)."""
    x = (int(i) * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 29
    return (x & 0xFFFFFFFF) / float(0x100000000)


def _mem(agent_id):
    return _MEM.setdefault(agent_id, {"flee": False, "spawn_clock": 0, "last_steer": 0.0, "clock": 0})


def _facing_turn(P, preds):
    """Radians to rotate so the nearest predator stays inside the facing cone (0.0 = no change).

    See DEFAULT_PARAMS["face_predator"] for the measured mechanic this exploits. Costs
    |turn|/(2pi) energy per tick; once the predator is inside the cone the steady-state correction
    is tiny, because the predator's bearing changes slowly while it pivots instead of charging.
    """
    if P.get("face_predator", 0.0) <= 0.0 or not preds:
        return 0.0
    p = min(preds, key=lambda q: q["distance"])
    d = p["distance"]
    if not (float(P.get("face_min_dist", 90.0)) <= d <= float(P.get("face_dist_max", 320.0))):
        return 0.0                      # too far to matter, or already inside the charge radius
    a = _wrap(p["angle"])               # bearing to the predator, relative to our current facing
    cone = float(P.get("face_cone", 1.05))
    if abs(a) <= cone:
        return 0.0                      # already facing it far enough -> it pivots, not charges
    return math.copysign(abs(a) - cone, a)


def potential_controller(state, P):
    global _EPOCH
    _EPOCH += 1
    o = state.get("observations") or []
    energy = state.get("energy", 0.0)
    max_e = max(state.get("max_energy", 1.0) or 1.0, 1.0)
    ef = energy / max_e
    speed = state.get("speed", 10.0)
    sprint = state.get("sprint_speed", 20.0)
    aid = state.get("agent_id", 0)
    _maybe_new_episode(aid, state.get("age", 0.0))
    _GC[aid] = _EPOCH
    m = _mem(aid)
    m["clock"] += 1
    gpop = _global_alive()

    # ---- GENOME-AWARE BREEDER SELECTION (genome_select > 0) -------------------------------------
    # Every request carries this agent's OWN six heritable traits, and every observed conspecific
    # carries its `id`, so the fleet's genotype is reconstructable from our own call history alone.
    # This block (a) maintains that id->traits table and (b) decides whether THIS agent may spend a
    # 100-energy spawn: only agents whose genome utility ranks in the fleet's top-k may breed, which
    # turns an unbiased +/-50% mutation into a ratchet in whatever direction U points.
    # When genome_select == 0 nothing here runs (no table write, no gate) => behaviour is byte-identical.
    global _TRAITS, _GS_LAST
    gs = float(P.get("genome_select", 0.0) or 0.0)
    gs_ok = True
    if gs > 0.0:
        _TRAITS[aid] = [float(state.get(k, 0.0) or 0.0) for k in _TRAIT_KEYS] + [_EPOCH]
        _ttl = int(P.get("gs_ttl", 60) or 60)
        # prune agents not seen for _ttl epochs: same cooperative-alive proxy as _GC, so the ranking
        # is against the LIVING fleet rather than against every agent that ever existed this episode
        if len(_TRAITS) > 64:
            for _k in [k for k, v in _TRAITS.items() if _EPOCH - v[6] > _ttl]:
                _TRAITS.pop(_k, None)
        _w = _trait_weights(P)
        _alive = [(k, v) for k, v in _TRAITS.items() if _EPOCH - v[6] <= _ttl]
        own_u = _genome_u(_TRAITS.get(aid), _w)
        # rank: 1 = best genome in the fleet. Deterministic tie-break by agent id, so equal genomes
        # (e.g. the 5 founders, all at the defaults) still produce a stable ordering.
        _rank = 1 + sum(1 for k, v in _alive
                        if k != aid and (_genome_u(v, _w) > own_u
                                         or (_genome_u(v, _w) == own_u and k < aid)))
        _GS_LAST = (own_u, _rank, len(_alive))
        _min_known = float(P.get("gs_min_known", 3.0) or 0.0)
        _rescue_pop = float(P.get("gs_rescue_pop", 1.0) or 0.0)
        _mode = float(P.get("gs_mode", 1.0) or 1.0)
        if _mode >= 2.0:
            # IMPROVING gate: breed only from agents above the fleet's own mean genome utility. No
            # population target is implied (roughly half the fleet qualifies at any time), so the
            # ratchet gets many more draws per 1,000 ticks than top-k does.
            _mu = sum(_genome_u(v, _w) for _, v in _alive) / float(len(_alive)) if _alive else 0.0
            _select_ok = own_u > _mu
        else:
            _select_ok = _rank <= float(P.get("gs_topk", 2.0) or 2.0)
        # never gate the relay: below the rescue population the lineage must continue whatever the
        # genome, and with too few known genomes the ranking is not informative yet.
        gs_ok = (len(_alive) < _min_known) or (gpop <= _rescue_pop) or _select_ok

    fruits = [x for x in o if x.get("type") == "Fruit"]
    preds = [x for x in o if x.get("type") == "Predator"]
    agents = [x for x in o if x.get("type") == "Agent"]
    edges = [x for x in o if x.get("type") == "Edge"]

    # ---- reconstruct SIMULATED TIME (the state carries no clock, no score, no episode index) ----
    # A pre-existing agent's `age` IS elapsed sim time (sim-seconds; 1 tick = 0.1 s), and every agent
    # is called exactly once per tick, so a newborn is simply born at the current estimate and each
    # agent's call count advances time by 1. sim_tick = max(birth + calls) over all agents seen.
    global _SIM_TICK, _BLIND_EMA
    _age = float(state.get("age", 0.0) or 0.0)
    _rec = _SEEN.get(aid)
    if _rec is None:
        _rec = [0, int(round(_age * 10.0))] if _age >= 0.5 else [_SIM_TICK, 0]
        _SEEN[aid] = _rec
    else:
        _rec[1] += 1
    if _rec[0] + _rec[1] > _SIM_TICK:
        _SIM_TICK = _rec[0] + _rec[1]

    # ---- PHASE (boom -> famine): TIME-primary, with an optional adaptive food-availability guard.
    # The decay of fruit production is global (invisible to any single agent), but INCOME is also
    # biome-dependent, so a guard term lets visible food delay the switch. ----
    _BLIND_EMA = 0.99 * _BLIND_EMA + 0.01 * (0.0 if fruits else 1.0)
    phase = 0.0
    _pm = float(P.get("phase_mode", 0.0) or 0.0)
    if _pm > 0.0:
        _lo = float(P.get("famine_tick_lo", 4000.0) or 0.0)
        _hi = float(P.get("famine_tick_hi", 7500.0) or 0.0)
        if _hi <= _lo:
            _hi = _lo + 1.0
        _ph = (_SIM_TICK - _lo) / (_hi - _lo)
        _ph = 0.0 if _ph < 0.0 else (1.0 if _ph > 1.0 else _ph)
        _guard = float(P.get("famine_vis_guard", 0.0) or 0.0)
        if _guard > 0.0:
            _ph *= max(0.0, 1.0 - min(1.0, (1.0 - _BLIND_EMA) / _guard))
        phase = _ph * _pm

    # ---- STATE-TRIGGERED THIN-RELAY (thin_relay > 0): a TIME gate AND a food-collapse gate, with
    # a two-threshold latch so the mode cannot flap. Off => nothing below changes any behaviour
    # (the fruit EMA is not even updated, so `_BLIND_EMA` and everything else stay bit-identical).
    global _FRUIT_EMA, _THIN_ARMED
    thin_active = False
    _thin = float(P.get("thin_relay", 0.0) or 0.0)
    if _thin > 0.0:
        # fleet EMA of fruits visible per agent (each call = one agent's view this tick)
        _FRUIT_EMA = 0.985 * _FRUIT_EMA + 0.015 * float(len(fruits))
        _t_lo = float(P.get("thin_trig_tick", 6000.0) or 0.0)
        _v_on = float(P.get("thin_trig_vis", 0.35) or 0.0)
        _v_off = float(P.get("thin_trig_vis_off", 0.70) or 0.0)
        if _THIN_ARMED:
            if _FRUIT_EMA > _v_off:
                _THIN_ARMED = False      # hysteresis release: food genuinely came back
        elif _SIM_TICK > _t_lo and _FRUIT_EMA < _v_on:
            _THIN_ARMED = True           # BOTH gates: sim time past threshold AND visibility collapsed
        thin_active = _THIN_ARMED

    # --- descend spawn clock ---
    if m["spawn_clock"] > 0:
        m["spawn_clock"] -= 1

    # ---- predator repulsion field (local frame vectors) ----
    px = py = 0.0
    threat_dist = float("inf")
    threat_dir = None
    use_pred = P.get("use_predator", True)
    for p in preds:
        if not use_pred:
            break
        d = p["distance"]
        if d > P["danger_dist"]:
            continue
        if d < threat_dist:
            threat_dist, threat_dir = d, p["angle"]
        a = p["angle"]
        fx = -math.cos(a)
        fy = -math.sin(a)
        weight = P["predator_weight"] / (max(d, 5.0) ** 2)
        px += fx * weight
        py += fy * weight

    # ---- RETREAT-WHILE-FACING EVASION (evade_mode > 0) ---------------------------------------
    # Placed BEFORE the flee override so it takes precedence when enabled. It replaces "run away
    # with my back turned" (which triggers a CHARGE) with "back away while keeping the predator in
    # front" (which makes it PIVOT). Measured separation while facing: +0.48 to +1.58 units/tick
    # versus ~0 when turned. Same sprint cost -- this is about geometry, not about spending less.
    if P.get("evade_mode", 0.0) > 0.0 and preds:
        _pn = min(preds, key=lambda p: p["distance"])
        _ed = float(P.get("evade_dist", 140.0) or 0.0)
        _ex = float(P.get("evade_disengage", 240.0) or 0.0)
        _ev = bool(m.get("evade", False))
        if _pn["distance"] < _ed:
            _ev = True
        elif _pn["distance"] > _ex:
            _ev = False
        m["evade"] = _ev
        if _ev and energy > float(P.get("evade_energy_abs", 130.0) or 0.0):
            _a = _wrap(_pn["angle"])                 # predator bearing relative to our facing
            _away = _wrap(_a + math.pi)              # move directly away from it
            return [float(sprint * float(P.get("evade_speed_frac", 0.8) or 0.8)),
                    float(_away),
                    float(-_a),                      # ...then face it (suppresses the charge)
                    0.0]

    # flee override with hysteresis
    facing_threat = False
    for p in preds:
        if not use_pred:
            break
        rdir = math.atan2(math.sin(p.get("rel_dir", 0)), math.cos(p.get("rel_dir", 0)))
        if abs(rdir) < P["concern_cone"] and p["distance"] < P["danger_dist"]:
            facing_threat = True
            break
    in_escape = m["flee"]
    if use_pred and threat_dist < P["flee_dist"]:
        in_escape = True
    elif threat_dist > P["escape_dist"]:
        in_escape = False
    flee_engage = (P.get("use_flee", True)) and (in_escape or (facing_threat and threat_dist < P["danger_dist"] * 0.7))

    if flee_engage:
        # flee directly away from aggregate predator vector; if pv~0 use closest threat
        mag = math.hypot(px, py)
        if mag > 1e-6:
            away = math.atan2(py, px)
        elif threat_dir is not None:
            away = math.atan2(-math.sin(threat_dir), -math.cos(threat_dir))
        else:
            away = 0.0
        m["flee"] = True
        # sprint only if we can afford it
        if ef > P["reserve_frac"]:
            dist = sprint * P["flee_speed_frac"]
        else:
            dist = speed * 0.9
        return [float(dist), float(away), float(_facing_turn(P, preds)), 0.0]

    m["flee"] = False

    # ---- fruit attraction (weighted by safety) ----
    fx = fy = 0.0
    best_fruit = None
    use_fruit = P.get("use_fruit", True)
    for f in fruits:
        if not use_fruit:
            break
        d = f["distance"]
        a = f["angle"]
        # risk: fruits in same direction as a close predator are less attractive
        risk = 0.0
        for p in preds:
            if p["distance"] < P["danger_dist"] * 1.4:
                angdiff = abs(_wrap(p["angle"] - a))
                if angdiff < 1.0:
                    risk += (1.0 - angdiff) * P["fruit_risk_penalty"] * (1 - p["distance"] / (P["danger_dist"] * 1.4))
        # weight: inverse distance, discounted by risk
        w = 1.0 / (d + 8.0) * max(0.0, 1.0 - risk)
        fx += math.cos(a) * w
        fy += math.sin(a) * w
        if best_fruit is None or d < best_fruit["distance"]:
            best_fruit = f

    # ---- tree attraction as weaker fallback (trees host fruit) ----
    for t in [x for x in o if x.get("type") == "Tree"]:
        if not P.get("use_fruit", True):
            break
        d = t["distance"]
        a = t["angle"]
        w = P.get("tree_weight", 0.25) / (d + 20.0)
        fx += math.cos(a) * w
        fy += math.sin(a) * w

    # ---- other-agent dispersion (avoid clustering -> predator multi-kills) ----
    for ag in agents:
        if not P.get("use_disperse", True):
            break
        d = ag["distance"]
        if d < 60.0 and d > 1.0:
            a = ag["angle"]
            w = P["disperse_weight"] / (d + 5.0)
            fx -= math.cos(a) * w
            fy -= math.sin(a) * w

    # ---- wall/edge repulsion ----
    if edges and P.get("use_wall", True):
        pts = [e["coords"][0] for e in edges] + [e["coords"][1] for e in edges]
        pxW = pyW = 0.0
        for (ex, ey) in pts:
            d = math.hypot(ex, ey)
            if d < 90.0 and d > 1.0:
                w = P["wall_weight"] / (d ** 2)
                pxW -= ex / d * w
                pyW -= ey / d * w
        fx += pxW
        fy += pyW

    # ---- wander ---
    rw_ang = m.get("wander_ang")
    if rw_ang is None or m["clock"] % 80 == 0:
        rw_ang = _det_rand(aid * 1000003 + m["clock"]) * 2 * math.pi
    m["wander_ang"] = rw_ang
    if P.get("use_explore", True):
        fx += math.cos(rw_ang) * P["wander_weight"]
        fy += math.sin(rw_ang) * P["wander_weight"]

    # ---- combine; hysteresis on steering angle to reduce oscillation ----
    mag = math.hypot(fx, fy)
    if mag < 1e-6:
        steer = _wrap(rw_ang)
    else:
        steer = math.atan2(fy, fx)

    # ---- direct-forage override: when fruit IS visible, spend the minimum travel distance to reach
    # it (movement costs 0.05/unit, the dominant energy sink). A weighted vector-sum of many fruits
    # can cancel out and produce a heading aimed at NO fruit; going straight at the nearest one
    # maximises energy income per distance travelled.
    direct_forage = False
    if P.get("forage_nearest", 0.0) > 0.0 and fruits:
        tf = min(fruits, key=lambda f: f["distance"])
        steer = _wrap(tf["angle"])
        direct_forage = True

    # sticky: don't flip heading wildly tick-to-tick unless food is very close
    if not direct_forage and (best_fruit is None or best_fruit["distance"] > 25.0):
        diff = _wrap(steer - m["last_steer"])
        steer = m["last_steer"] + diff * P["target_hyst"]
    m["last_steer"] = _wrap(steer)

    # ---- MEMORY-BASED SEARCH (optimal-foraging behaviours; memory_search 0.0 = OFF) -------------
    # The controller is otherwise memoryless, and income is ACCESS-limited, so "where was the food?"
    # is the missing state. See DEFAULT_PARAMS for why, and for the three behaviours implemented.
    committed = False
    if P.get("memory_search", 0.0) > 0.0:
        if fruits:
            m["blind_run"] = 0
            m["mem_steer"] = _wrap(steer)          # remember the heading that led to food
        else:
            m["blind_run"] = int(m.get("blind_run", 0)) + 1
            _fw = float(P.get("follow_agent_weight", 0.0) or 0.0)
            _follow = None
            if _fw > 0.0 and agents:               # SOCIAL FORAGING: follow a possible knower
                _cand = [g for g in agents if g["distance"] < 260.0]
                if _cand:
                    _follow = min(_cand, key=lambda g: g["distance"])
            if _follow is not None:
                steer = _wrap(steer * (1.0 - _fw) + _wrap(_follow["angle"]) * _fw)
            else:
                _cl = int(P.get("commit_len", 250) or 250)
                if m.get("mem_steer") is None or int(m.get("blind_run", 0)) > _cl:
                    # nothing remembered, or the commitment expired -> commit to a fresh straight run
                    m["mem_steer"] = _wrap(rw_ang)
                    m["blind_run"] = 1
                steer = _wrap(m["mem_steer"])      # BALLISTIC RELOCATION: hold the heading and RUN
                committed = True

    # ---- move distance (energy-aware) ----
    use_energy = P.get("use_energy", True)
    if committed:
        dist = speed * float(P.get("commit_speed_frac", 1.0) or 0.0)   # cover ground, don't mill
    elif direct_forage:
        dist = speed * float(P.get("forage_speed", 1.0))
    elif best_fruit is not None and best_fruit["distance"] < 40.0:
        dist = min(speed, best_fruit["distance"])
    elif use_energy and ef < P["low_energy_frac"]:
        dist = speed * 0.4
    else:
        _bf = P.get("blind_explore_frac")
        dist = speed * P["walk_frac"] if (best_fruit is not None) else speed * (P["explore_frac"] if _bf is None else _bf)
    # AREA-RESTRICTED SEARCH: while food is visible, slow down so we stay INSIDE the patch we found
    # instead of walking straight out of it (1.0 = off).
    _ars = float(P.get("ars_speed_frac", 1.0) or 1.0)
    if fruits and _ars < 1.0:
        dist = min(dist, speed * _ars)
    if use_energy and ef < P["reserve_frac"]:
        dist = min(dist, speed * 0.25)

    # ---- DISCOVERY-DERIVED PHASE SPEED RULE (psp_mode, OFF by default) ----
    # From the counterfactual boundary analysis (8,440 branch rows = ~940 states x 9 actions, fixed-tick
    # continuous outcomes, agent-visible features only). The boundary found with fruit in sight:
    #     tick <= 900                 -> WALK   (fruit abundant, conserve energy)      62% consistent
    #     900 < tick <= 2700          -> SPRINT (time is scarce; travel cost dominates) 65% consistent
    # This is a STATE-DEPENDENT rule, not a parameter: no single walk_frac value beats the baseline
    # (the 800-candidate search sampled walk_frac and found nothing), which is precisely why the phase
    # gate is required. Only ever raises the speed, so it cannot slow an agent down.
    if float(P.get("psp_mode", 0.0) or 0.0) and best_fruit is not None:
        _lo = float(P.get("psp_lo", 900.0) or 0.0)
        _hi = float(P.get("psp_hi", 2700.0) or 0.0)
        if _lo <= _SIM_TICK <= _hi:
            dist = max(dist, sprint * float(P.get("psp_mid_frac", 1.0) or 0.0))

    # ---- FREE-ROTATION SCAN AT FRUIT (OFF by default: scan_mode = 0.0) ----
    # Rotation is NOT free but it is CHEAP: turning costs |turn|/(2pi) energy (a 180 deg sweep = 0.5)
    # versus ~5.5/tick ...[truncated]

    _scan_add = 0.0
    if float(P.get("scan_mode", 0.0) or 0.0) > 0.0 and best_fruit is not None:
        _sper = max(2.0, float(P.get("scan_period", 10.0) or 10.0))
        if (_SIM_TICK % _sper) < float(P.get("scan_ticks", 4.0) or 0.0):
            dist = dist * float(P.get("scan_slow", 0.25) or 0.25)
            _scan_add = float(P.get("scan_turn", 0.45) or 0.45)

    # ---- BIOME-AWARE TERRAIN HANDLING (OFF by default: bio_mode=0) ----
    # MEASURED: the world is 27.9% desert + 5.9% river, BOTH producing ZERO fruit, while only 16.9% is
    # grassland at the best rate (0.1 fruit/s/100x100). The observation carries `biome` (see
    # src/utils/DTOs.py: "biome: str") and this controller has NEVER read it. Agents are blind 76-87%
    # of ticks and STARVATION is our largest death category (32-51%), so (a) spawning 12-16 agents per
    # 1k ticks into dead terrain pays 100 energy for a body born where no food is produced, and
    # (b) idling in dead terrain is pure energy loss. Both facts use information already in the payload.
    _bio_mode = float(P.get("bio_mode", 0.0) or 0.0) > 0.0
    _bio_name = str(state.get("biome", "") or "").lower()
    _bio_dead = _bio_mode and (("desert" in _bio_name) or ("river" in _bio_name))
    _bio_ok = not _bio_dead
    # cross zero-production terrain quickly when nothing is visible to eat
    if _bio_dead and best_fruit is None:
        dist = max(dist, sprint * float(P.get("bio_sprint_frac", 0.85) or 0.85))

    # ---- PHASE policy on movement: in famine, travel energy is unaffordable. With food visible we
    # still approach it (at a reduced rate -- finding food is the whole income), but with NOTHING
    # visible we stop dead rather than wander: walking is 0.05/unit per tick against 0.1/tick
    # metabolism, so aimless travel burns the bank 2-5x faster than simply existing. ----
    if phase > 0.0:
        _keep = 1.0 - phase * (1.0 - float(P.get("famine_move_frac", 0.2) or 0.0))
        if not fruits:
            _keep *= max(0.0, 1.0 - phase * float(P.get("famine_blind_stop", 1.0) or 0.0))
        dist = dist * _keep

    # ---- age-based role scaling (audit C2): every agent is called separately and knows only its
    # own age, so roles need no coordination at all. The young are the future -> forage hard; the
    # old are a dying investment (max_age 60-120 s) -> stop burning energy on movement and put it
    # into banking an heir instead.
    if P.get("role_by_age", 0.0) > 0.0:
        _age = state.get("age", 0.0)
        if _age > P.get("old_age", 60.0):
            dist *= P.get("old_speed_frac", 0.5)
        elif _age < P.get("young_age", 25.0):
            dist *= P.get("young_speed_frac", 1.0)

    # ---- THIN-RELAY movement: conserve energy but NEVER stop. Movement is the INCOME mechanism
    # (agents are blind 76-87% of ticks, so standing still earns nothing); 'stop dead' was measured
    # to lose thousands of ticks. The floor is a hard guarantee of nonzero travel.
    if thin_active:
        _tmove = float(P.get("thin_move_frac", 0.7) or 0.0)
        dist = max(speed * 0.2, dist * _tmove)

    # ---- reproduction (investment-gated; cooperative pop cap via shared estimate) ----
    spawn = 0.0
    use_repro = P.get("use_repro", True)
    target = P.get("repro_global_target", 10)
    # ---- CARRYING-CAPACITY PHASE SCALING (OFF by default: cc_k0=0 leaves everything unchanged) ----
    # MEASURED (budget.py, live controller, x86, 1k-tick windows): production falls 9,983 -> 6,963 ->
    # 4,882 energy/1k while spawns stay 12.5-16.0/1k, and the standing stock drains (+1,288 early,
    # then -435, -213, -194). Measured sustainable N: 28.3 -> 10.4 -> 5.9 -> 3.6. Actual N: 10.2 ->
    # 8.0 -> 7.2 -> 6.3 => the fleet is ~75% OVER carrying capacity at 9-12k ticks, which is the modal
    # death window. Carrying capacity tracks production, and production halves every 3,000 ticks, so
    # the reproduction target must halve on the same schedule.
    # This gates POPULATION GROWTH only - not movement, not income - which is why it is NOT the
    # refuted thrift/reserve/repro-floor family: those cut per-agent income and lose in an
    # access-limited world; this stops the fleet from spending its bank on excess bodies.
    if float(P.get("cc_k0", 0.0) or 0.0) > 0.0:
        _hl = max(1.0, float(P.get("cc_halflife", 3000.0) or 3000.0))
        _k = float(P.get("cc_k0", 0.0) or 0.0) * (0.5 ** (_SIM_TICK / _hl))
        target = max(float(P.get("cc_floor", 2.0) or 0.0),
                     _k * float(P.get("cc_margin", 1.0) or 1.0))
    # Population-maintaining gate: the energy bar drops toward repro_frac_min as the alive count
    # falls below target, so a depleted team spawns as soon as it can afford it (sim hard-requires
    # energy > 100 at the moment of spawning, so repro_frac_min is kept safely above that).
    rf = P.get("repro_frac", DEFAULT_PARAMS["repro_frac"])
    urgency = P.get("repro_urgency", 1.0)
    if urgency > 0.0 and target > 0:
        deficit = max(0.0, min(1.0, (target - gpop) / target))
        rf = rf - (rf - P.get("repro_frac_min", 0.24)) * deficit * urgency
    if P.get("repro_unlimited", 0.0) > 0.0:
        pop_ok = True
        crowd_ok = True
    else:
        pop_ok = gpop < target
        crowd_ok = len(agents) <= P.get("repro_popcap", 2)

    # --- generational relay (audit B1/B2, C2): never let the chain break ---
    # agent.py: max_age = 60 + U(0,60) sim-seconds and environment.py applies energy -= 0.01*age
    # per tick past it, so EVERY agent dies of age in 60-120 s no matter how well fed. Survival
    # past ~120 s therefore requires an unbroken line of heirs. An already-old agent is a dying
    # investment and must bank a successor NOW, even at/above the population target: population
    # size is NOT a score multiplier (score += dt fires once per tick regardless of agent count);
    # the metric is the time until the LAST agent dies.
    relay_age = P.get("relay_age", 0.0)  # sim seconds; 0 = disabled
    relay = relay_age > 0.0 and state.get("age", 0.0) > relay_age
    if relay:
        pop_ok = True  # population COUNT is not a score multiplier (score += dt once per tick)
        # ...but keep the LOCAL dispersion gate: a crowd is what gets multi-killed, and each extra
        # body is both a 100-energy cost and an edible -energy/100 liability. A relay needs ONE
        # competent heir, not a swarm.
        rf = min(rf, P.get("relay_energy_frac", 0.35))
        cd = P.get("relay_cooldown", 700)     # slower cadence for relay spawns than the normal gate
    else:
        cd = P.get("spawn_cooldown", 400)
    # --- THIN RELAY override: this is the part the four refuted 'bank late' arms got WRONG. -------
    # A population gate that blocks ALL reproduction breaks the age-relay and the fleet dies (that is
    # the documented failure, measured 4,719 vs 7,816 ticks). Here reproduction is NEVER blocked: the
    # relay admits a spawn whenever the fleet is below the thin target, and always when only ONE agent
    # is left. That single change is what makes an 18k-tick run possible at all -- it is not optional.
    _thin_room = True
    if thin_active:
        _tsz = float(P.get("thin_target_pop", 2.0) or 0.0)
        _rescue = (gpop <= 1)              # LAST agent standing: the chain must continue
        pop_ok = True                      # never disable the relay (see comment above)
        crowd_ok = True
        # room = one spawn's worth below/at the target: at gpop == target exactly ONE agent banks an
        # heir (the fleet settles at target..target+1 = 2-3), and below it any competent adult may.
        # Restricting to gpop < target instead makes the fleet decay silently before the relay is
        # ever allowed to act, which is how the refuted arms lost their lineage.
        _thin_room = _rescue or (gpop <= _tsz)
        cd = int(P.get("thin_spawn_cooldown", 200) or 200)
    # --- reproduction gate, in the RIGHT unit ---
    # environment.py hard-requires energy > 100 to spawn. Our gate was a FRACTION of max_energy
    # (0.35 * 500 = 175), which is both the wrong unit (max_energy varies per agent) and
    # unreachable exactly when food decays and the relay matters most. Measured: ZERO births in
    # the final 2,000 ticks of a losing run, with mean energy 37-120 and fruit_vis 0-1.
    # Minimum viable reproduction: spend down to just above the sim's own requirement.
    abs_gate = P.get("repro_energy_abs", 0.0)
    if thin_active:
        # ABSOLUTE energy gate: 250 normally (parent keeps >=150, clear of the sprint lockout at
        # max_energy/5), dropped to the rescue value when this is the last agent alive -- see
        # DEFAULT_PARAMS["thin_rescue_energy_abs"] for why survival beats keeping the parent fast.
        abs_gate = float(P.get("thin_rescue_energy_abs", 160.0) if gpop <= 1
                         else P.get("thin_spawn_energy_abs", 250.0))
    gate_ok = (energy > abs_gate) if abs_gate > 0.0 else (ef > rf)
    if (use_repro and gate_ok
            and m["spawn_clock"] <= 0
            and pop_ok
            and crowd_ok
            and _bio_ok
            and _thin_room
            and gs_ok):
        safe = all(p["distance"] >= P.get("repro_safe_radius", 330.0) for p in preds)
        if safe:
            spawn = 1.0
            m["spawn_clock"] = cd

    # ---- PHASE policy on reproduction: during the famine, BANK instead of breeding. A spawn costs
    # the parent 100 energy (the child starts at 75) and every extra agent adds 0.1/tick metabolism
    # forever, for zero score benefit (score is +0.1/tick regardless of population). So in famine we
    # only spawn to keep a small insurance population, and only from agents that are nearly full --
    # optionally only from high-capacity lineages, since max_energy is heritable (mutates +/-50%,
    # capped at 1000) and it is what sets how long a survivor can outlast the food. ----
    if spawn and phase > 0.0:
        _minpop = float(P.get("famine_min_pop", 2.0))
        _bank = float(P.get("famine_bank_frac", 0.9))
        _mincap = float(P.get("famine_min_cap", 0.0) or 0.0)
        _own_cap = float(state.get("max_energy", 500.0) or 500.0)
        # RELAY RESCUE first: a fleet that stops breeding under age mortality just dies, which is
        # exactly how the first version of this policy failed (measured 4,719 vs 7,816 ticks).
        _rescue = gpop < _minpop
        # otherwise INVEST: only a nearly-full, high-capacity parent is worth 100 energy
        _invest = (ef >= _bank) and (_own_cap >= _mincap)
        if not (_rescue or _invest):
            spawn = 0.0

    return [float(dist), float(steer), float(_facing_turn(P, preds)) + _scan_add, float(spawn)]


def _load_params():
    import json, os
    P = dict(DEFAULT_PARAMS)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_controller", "params.json")
    try:
        with open(p) as f:
            P.update(json.load(f))  # partial file: merge over defaults so no KeyError
    except Exception:
        pass
    return P


# standalone importable policy function (contract) using saved/evolved params
def best_controller(state):
    return potential_controller(state, _load_params())


def make_policy(params):
    def fn(state):
        return potential_controller(state, params)
    return fn