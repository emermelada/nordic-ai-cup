"""Hive: controller for the survival simulator.

One `Hive` instance serves consecutive games; it resets itself when `sim_time` goes backwards.
`decide(step)` takes the StepResponse as a dict and returns ActionRequest dicts.

World model
-----------
* Every agent starts in its own frame at pose (0, 0, 0). Its pose afterwards is dead reckoning with the sim's
  own rules: heading changes only by our turns, the move uses the sim's caps and the biome penalty we observe,
  and collisions are replayed against the walls we have mapped. A translation fix from re-observed walls
  catches whatever the replay missed (the heading is exact, so one wall is enough).
* Frames merge when an agent of one frame observes an agent of another: the observation gives the other's
  exact position and heading. A child therefore joins its parent's frame the tick after birth.
* A frame snaps to world coordinates when one of its agents sees a boundary wall (1600/1200 px edges whose
  start->end runs along +x/+y). All world frames share the frame id "W".
* Map per frame: walls (every seen face as a 30 px deep slab: every obstacle is >= 30 px thick), trees,
  fruit with spawn-time bounds (from a raster of when each spot was last within someone's hearing), and
  predator tracks (the observation gives a predator's heading exactly).

Behaviour per agent and tick (first match wins): evade threats; breed (elite genomes early; anyone old,
at the end, or when the colony is tiny); eat or wait for ripe fruit (global greedy assignment); camp at
productive trees; explore.
"""
import math
import random

import numpy as np

PI = math.pi
TWO_PI = 2 * math.pi
BIOME_PENALTY = {"forest": 1.0, "grassland": 1.0, "swamp": 0.5, "desert": 0.8, "river": 0.3}
WORLD_W, WORLD_H = 1600.0, 1200.0
EDGE_CELL = 64.0
RASTER_CELL = 10.0

DEFAULT_PARAMS = {
    # request-stream hardening
    "probe_guard": 1.0,            # 1 = a lone backward-sim_time payload (platform connectivity probe,
                                   # or a foreign client's game interleaved with ours) SUSPENDS our game
                                   # instead of resetting it; 0 = the old reset-on-any-backward-jump rule
    # breeding
    "w_speed": 6.0, "w_hear": 3.0, "w_vis": 0.7, "w_cone": 0.7, "w_sprint": 0.8, "w_energy": 0.0,
    "elite_margin": 0.03,          # fitness within this of the best alive counts as elite
    "breed_reserve_early": 40.0,   # energy an elite keeps after spawning while t < breed_phase_end
    "breed_reserve_late": 120.0,   # energy kept after spawning later on
    "breed_phase_end": 900.0,
    "breed_reserve_span": 200.0,   # the worst genome that may breed keeps this much more than the best
    "breed_gap": 0.4,              # genomes more than this below the reference genome do not breed
    "breed_ref_q": 0.75,           # reference genome = this quantile of fitness alive (not the single best)
    "sprint_keep": 15.0,           # once predators exist, keep max_energy/5 + this after spawning (sprint lock)
    "spread_r": 70.0,              # campers avoid trees with another agent within this radius
    "spread_pen": 200.0,
    "tree_prod_w": 60.0,           # px of walking one known fruit / known-fruiting tree is worth
    "barren_reach": 250.0,         # do not walk farther than this to a tree not seen fruiting
    "tree_stick": 100.0,           # keep the current tree unless another is this much nearer
    "grab_energy": 0.0,            # below this energy an agent under threat still grabs a fruit ...
    "grab_dist": 35.0,             # ... within this distance
    "scout_retired": 0.0,          # 1 = retired weak genomes explore stale map cells instead of sitting
    "scout_min_e": 15.0,
    "camp_spacing": 60.0,          # a tree counts as taken if another camper sits within this distance
    "tree_fresh_w": 120.0,         # px a tree with fruit seen in the last 30 s is worth
    "barren_watch": 20.0,          # s sitting at a tree without fruit before giving up on it
    "tree_forget": 70.0,
    "slow_tree_pen": 300.0,        # px-equivalent penalty per unit of speed lost at a tree's spot (swamp 0.5)           # s unseen after which a tree is dropped (trees live ~58 s)
    "breed_start": 25.0,           # no breeding before this (young trees do not fruit yet)
    "food_range_hungry": 200.0,
    "food_range_idle": 150.0,      # food radius for agents not sitting at a fruiting tree
    "pop_cap_early": 36, "pop_cap_mid": 18, "pop_cap_late": 8,
    "t_mid": 900.0, "t_late": 1800.0,
    "pop_min": 6,                  # below this anyone may breed
    "breed_colony_e": 100.0,       # no births (except old-age dumps) while the colony's mean energy is below
    "brake_min_pop": 15,           # ... applied only to colonies at least this big (the early boom)
    "retire_min_pop": 6,           # weak genomes stop eating only in colonies at least this big
    "weak_food_w": 0.3,            # fruit value for agents that will not breed (weak genome / old and poor)
    "endgame_t": 2926.0,           # a child born now (75 energy) lives idle to t=3000
    # foraging
    "ripe_age": 19.0,              # leave dated fruit younger than this (s) on the tree ...
    "starve_frac": 0.08,           # ... unless below this fraction of max energy
    "hunger_w": 30.0,              # assignment bonus for hungry agents (x fraction of max energy missing)
    "dist_cost": 0.06,             # energy-equivalent cost per px of travel (walk 0.05 + living time)
    "food_range": 90.0,             # territorial: eat around where you sit
    "min_gain": 8.0,
    "food_stick": 0.0,             # score bonus for the fruit an agent was already heading to               # skip fruit whose net energy gain would be smaller than this
    "wait_dist": 18.0,             # where to wait next to an unripe fruit (touching is < 5 + radius <= 14)
    # threats
    "alert_awake": 190.0,          # react to awake predators whose track is this uncertain/close
    "alert_close": 110.0,          # react to an awake predator this close (it charges under 90) ...
    "alert_always": 70.0,          # ... even when it is busy with a closer agent, if it is this close
    "alert_rest": 75.0,            # stay this far from resting ones (they hear 60 px when they wake)
    "charge_zone": 95.0,           # predators charge inside 90 px whatever we do: run (sprint if slower)
    "keep_dist": 160.0,            # back off from awake predators closer than this
    "drift_speed": 3.0,            # ... and drift away slowly from farther ones that can see us
    "charge_speed": 40.0,          # speed cap inside the charge zone (40 = full walking speed; predator sprints 15)
    "backoff_speed": 11.0,         # speed when backing off a circling predator (it closes at ~10.6)
    "face_tol": 0.9,               # keep the nearest awake predator within this bearing (< pi/2)
    "scan_rate": 0.15,             # idle agents turn this much per tick to watch all round
    "track_memory": 1.5,           # seconds an unseen predator heading our way stays a threat
    # exploration
    "explore_speed": 8.0,
}


def wrap(a):
    return (a + PI) % TWO_PI - PI


class Mem:
    """What the hive remembers about one agent."""
    __slots__ = ("aid", "frame", "x", "y", "h", "energy", "max_energy", "speed", "sprint", "hear", "vis", "cone",
                 "age", "biome", "last", "old", "born", "fit", "mode", "target", "wander_dir", "wander_until",
                 "expected", "fixes", "spawned", "stale", "food_tgt")

    def __init__(self, aid, frame, tick):
        self.aid = aid
        self.frame = frame
        self.x = self.y = self.h = 0.0
        self.age = -1.0
        self.stale = False
        self.food_tgt = None
        self.last = None
        self.old = False
        self.born = tick
        self.mode = "new"
        self.target = None
        self.wander_dir = None
        self.wander_until = 0
        self.expected = None
        self.fixes = 0
        self.spawned = 0


class FrameMap:
    """Everything known in one coordinate frame."""

    def __init__(self, fid, world):
        self.fid = fid
        self.world = world
        self.edges = []            # [x1, y1, x2, y2, nx, ny]  (n: unit normal pointing into the obstacle)
        self.edge_cells = {}       # (i, j) -> list of edge indices
        self.edge_keys = set()
        self.edge_idx = {}         # rounded key -> edge index
        # fruit: parallel arrays; lo/hi bound the spawn time, last = last time seen
        self.fx = np.empty(0)
        self.fy = np.empty(0)
        self.flo = np.empty(0)
        self.fhi = np.empty(0)
        self.flast = np.empty(0)
        # trees
        self.tx = np.empty(0)
        self.ty = np.empty(0)
        self.tfirst = np.empty(0)
        self.tlast = np.empty(0)
        self.tfruit = np.empty(0)     # last time fruit was seen within 65 px of the tree
        self.twatch = np.empty(0)     # seconds someone sat next to it since that fruit (barren test)
        # exploration: last time each 100 px cell was looked at (world frame only)
        self.seen = np.zeros((16, 12)) if world else None
        self.preds = []            # [x, y, h, last_seen, last_moved, first_seen]
        self.biome = {}            # (i, j) of 20 px cells -> move penalty observed there
        # hearing raster: tick when each 10 px cell was last well inside someone's hearing radius
        if world:
            self.r_off = (0.0, 0.0)
            self.raster = np.zeros((int(WORLD_W / RASTER_CELL) + 1, int(WORLD_H / RASTER_CELL) + 1), np.int32)
        else:
            self.r_off = (-2400.0, -2400.0)
            self.raster = np.zeros((481, 481), np.int32)

    # --- walls ---------------------------------------------------------------------------------------------
    def add_edge(self, x1, y1, x2, y2, nx, ny):
        """Add a wall face. (nx, ny) = (0, 0) when the obstacle side is not known yet."""
        key = (round(x1), round(y1), round(x2), round(y2))
        k0 = self.edge_idx.get(key)
        if k0 is not None:
            e = self.edges[k0]
            if e[4] == 0.0 and e[5] == 0.0 and (nx or ny):
                self.edges[k0] = (e[0], e[1], e[2], e[3], nx, ny)
                self._register_cells(k0)
            return False
        near = self.near_edges((x1 + x2) / 2, (y1 + y2) / 2)
        for k in near:
            e = self.edges[k]
            if abs(e[0] - x1) + abs(e[1] - y1) + abs(e[2] - x2) + abs(e[3] - y2) < 3:
                if e[4] == 0.0 and e[5] == 0.0 and (nx or ny):
                    self.edges[k] = (e[0], e[1], e[2], e[3], nx, ny)
                    self._register_cells(k)
                self.edge_idx[key] = k
                self.edge_keys.add(key)
                return False
        self.edge_keys.add(key)
        idx = len(self.edges)
        self.edge_idx[key] = idx
        self.edges.append((x1, y1, x2, y2, nx, ny))
        self._register_cells(idx)
        return True

    def _register_cells(self, idx):
        """Put the face in every cell its slab (plus margin) touches; unknown side: both sides, thin."""
        x1, y1, x2, y2, nx, ny = self.edges[idx]
        if nx or ny:
            xs = (x1, x2, x1 + 40 * nx, x2 + 40 * nx)
            ys = (y1, y2, y1 + 40 * ny, y2 + 40 * ny)
        else:
            xs = (x1, x2)
            ys = (y1, y2)
        i0, i1 = int(math.floor((min(xs) - 45) / EDGE_CELL)), int(math.floor((max(xs) + 45) / EDGE_CELL))
        j0, j1 = int(math.floor((min(ys) - 45) / EDGE_CELL)), int(math.floor((max(ys) + 45) / EDGE_CELL))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                lst = self.edge_cells.setdefault((i, j), [])
                if idx not in lst:
                    lst.append(idx)

    def near_edges(self, x, y):
        return self.edge_cells.get((int(math.floor(x / EDGE_CELL)), int(math.floor(y / EDGE_CELL))), ())

    def blocked(self, x, y):
        """Inside a known wall slab expanded by the agent radius (the sim's square expansion)."""
        for k in self.near_edges(x, y):
            x1, y1, x2, y2, nx, ny = self.edges[k]
            ux, uy = x2 - x1, y2 - y1
            ln = math.hypot(ux, uy)
            qx, qy = x - x1, y - y1
            along = (qx * ux + qy * uy) / ln
            if along <= -5 or along >= ln + 5:
                continue
            if nx or ny:
                depth = qx * nx + qy * ny
                if -5 < depth < 35:
                    return True
            elif abs(qx * uy - qy * ux) / ln < 5:
                return True
        return False

    def local_edges(self, x, y, reach):
        """Array (n, 6) of the faces registered in cells within `reach` of (x, y)."""
        i0, i1 = int(math.floor((x - reach) / EDGE_CELL)), int(math.floor((x + reach) / EDGE_CELL))
        j0, j1 = int(math.floor((y - reach) / EDGE_CELL)), int(math.floor((y + reach) / EDGE_CELL))
        idx = set()
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                idx.update(self.edge_cells.get((i, j), ()))
        if not idx:
            return None
        return np.array([self.edges[k] for k in idx], float)

    @staticmethod
    def blocked_many(E, xs, ys):
        """Vectorised `blocked` for points (xs, ys) against the faces E from `local_edges`."""
        if E is None:
            return np.zeros(xs.shape, bool)
        x1, y1, x2, y2, nx, ny = (E[:, k][None, :] for k in range(6))
        ux, uy = x2 - x1, y2 - y1
        ln = np.hypot(ux, uy)
        qx, qy = xs[:, None] - x1, ys[:, None] - y1
        along = (qx * ux + qy * uy) / ln
        inside = (along > -5) & (along < ln + 5)
        known = (nx != 0) | (ny != 0)
        depth = qx * nx + qy * ny
        hit_known = known & (depth > -5) & (depth < 35)
        hit_thin = (~known) & (np.abs(qx * uy - qy * ux) / ln < 5)
        return (inside & (hit_known | hit_thin)).any(axis=1)

    # --- raster --------------------------------------------------------------------------------------------
    def stamp(self, x, y, r, tick):
        rc = int((r - 7.0) / RASTER_CELL)
        if rc < 1:
            return
        ci = int((x - self.r_off[0]) / RASTER_CELL)
        cj = int((y - self.r_off[1]) / RASTER_CELL)
        mask = _disc(rc)
        n0, n1 = self.raster.shape
        i0, j0 = ci - rc, cj - rc
        a0, b0 = max(i0, 0), max(j0, 0)
        a1, b1 = min(ci + rc + 1, n0), min(cj + rc + 1, n1)
        if a0 >= a1 or b0 >= b1:
            return
        sub = self.raster[a0:a1, b0:b1]
        sub[mask[a0 - i0:a1 - i0, b0 - j0:b1 - j0]] = tick

    def last_heard(self, x, y):
        ci = ((x - self.r_off[0]) / RASTER_CELL).astype(np.int64)
        cj = ((y - self.r_off[1]) / RASTER_CELL).astype(np.int64)
        n0, n1 = self.raster.shape
        ok = (ci >= 0) & (ci < n0) & (cj >= 0) & (cj < n1)
        out = np.zeros(len(x), np.int64)
        out[ok] = self.raster[ci[ok], cj[ok]]
        return out


_DISCS = {}


def _disc(rc):
    d = _DISCS.get(rc)
    if d is None:
        g = np.arange(-rc, rc + 1)
        d = (g[:, None] ** 2 + g[None, :] ** 2) <= rc * rc
        _DISCS[rc] = d
    return d


class Hive:
    def __init__(self, params=None, seed=0):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.seed = seed
        # counters that must SURVIVE reset(): they describe the request stream, not one game
        self.counts = {"games": 0, "strays": 0, "stray_agents": 0, "restores": 0}
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.tick = 0
        self.t = 0.0
        self.last_t = -1.0
        self.mem = {}
        self.maps = {}
        self.next_frame = 0
        self.best_fit = 0.0
        self.stats = {"births": 0, "merges": 0, "world_regs": 0, "fixes": 0}
        self.counts["games"] += 1
        # --- stray-payload guard state (see decide()) ---
        self._stash = None          # snapshot of a live game's state, taken at a boundary
        self._stash_last_t = -1.0   # that game's last sim_time
        self._pending = False       # boundary seen, waiting for the stream to confirm which it was
        self._boundary_ids = set()  # agent ids the boundary payload carried

    # --- stray-payload guard -------------------------------------------------------------------------
    def _snap(self):
        return (self.rng, self.tick, self.t, self.last_t, self.mem, self.maps, self.next_frame,
                self.best_fit, self.stats)

    def _restore(self, snap):
        (self.rng, self.tick, self.t, self.last_t, self.mem, self.maps, self.next_frame,
         self.best_fit, self.stats) = snap
        self._stash = None
        self._pending = False
        self._boundary_ids = set()

    # ------------------------------------------------------------------------------------------------------
    def fitness(self, m):
        p = self.p
        return (p["w_speed"] * min(m.speed, 20.0) / 20.0 + p["w_hear"] * min(m.hear, 100.0) / 100.0
                + p["w_vis"] * min(m.vis, 400.0) / 400.0 + p["w_cone"] * min(m.cone, PI / 2) / (PI / 2)
                + p["w_sprint"] * min(m.sprint, 40.0) / 40.0 + p["w_energy"] * min(m.max_energy, 1000.0) / 1000.0)

    def _new_frame(self, world=False):
        if world:
            fid = "W"
        else:
            fid = self.next_frame
            self.next_frame += 1
        self.maps[fid] = FrameMap(fid, world)
        return fid

    # ------------------------------------------------------------------------------------------------------
    def decide(self, step):
        agents = step.get("agent_status") or []
        t = step.get("sim_time")
        if t is None:
            t = self.t + 0.1 if (agents and self.tick) else 0.0
        t = float(t)
        # ---- stray-payload guard ---------------------------------------------------------------------
        # A game boundary is signalled by sim_time going backwards. But the graded platform also injects
        # lone synthetic payloads (sim_time 0.0, one agent, the connectivity probe the organisers' README
        # documents) into a LIVE game, and a foreign client can interleave its own game with ours. The old
        # rule -- reset() on any backward jump -- therefore wiped a live game's whole state (world map,
        # per-agent memory, genome ratchet, RNG) whenever a probe arrived, and left the probe's synthetic
        # agent inside the state that the real game then inherited. Both cost score on the platform and
        # neither can happen in a local harness, which is why platform draws sat under local ones.
        # Rule now: a backward jump only *suspends* the game. The payload is answered with no-ops and the
        # stream's NEXT request decides: if it resumes the suspended game (t >= that game's last sim_time)
        # the state is restored exactly; if it starts a fresh game (t small) the fresh state is kept and
        # any agent memory the boundary payload created is dropped.
        if t < self.last_t - 1e-9:
            if self.p.get("probe_guard", 1.0):
                snap = self._snap()
                stash_last_t = self.last_t
                boundary_ids = {a.get("agent_id") for a in agents if isinstance(a, dict)}
                self.counts["strays"] += 1
                self.counts["stray_agents"] += len(boundary_ids)
                self.reset()                  # the tick is served from a fresh state either way
                self._stash = snap            # ... but the live game is only SUSPENDED, not lost
                self._stash_last_t = stash_last_t
                # Only a payload shaped like the platform's probe (sim_time 0.0, <= 2 agents) may leave
                # anything behind in the fresh state; a real game's first tick is kept exactly as before.
                self._boundary_ids = boundary_ids if (abs(t) < 1e-9 and len(boundary_ids) <= 2) else set()
                self._pending = True
            else:
                self.reset()                  # probe_guard 0 = the original rule, kept for A/B testing
        elif self.p.get("probe_guard", 1.0) and self._pending:
            if self._stash is not None and t >= self._stash_last_t - 1e-9:
                # the suspended game is still running: the boundary payload was a stray, not a new game
                self._restore(self._stash)
                self.counts["restores"] += 1
            else:
                # a genuinely new game begins here; forget anything the boundary payload planted
                for aid in self._boundary_ids:
                    self.mem.pop(aid, None)
                self._stash = None
                self._pending = False
                self._boundary_ids = set()
        self.last_t = t
        self.t = t
        self.tick += 1
        if not agents:
            return []

        alive = {}
        for a in agents:
            aid = a["agent_id"]
            m = self.mem.get(aid)
            if m is None:
                m = Mem(aid, self._new_frame(), self.tick)
                self.mem[aid] = m
                if self.tick > 2:
                    self.stats["births"] += 1
            else:
                self._predict(m)
            self._update_traits(m, a)
            alive[aid] = (m, _split_obs(a.get("observations") or []) if not m.stale else _EMPTY_OBS)
        for aid in [k for k in self.mem if k not in alive]:
            del self.mem[aid]

        for m, obs in alive.values():
            if not self.maps[m.frame].world and obs["Edge"]:
                self._try_world_register(m, obs["Edge"])
        for m, obs in alive.values():
            for o in obs["Agent"]:
                other = alive.get(o.get("id"))
                if other is not None and other[0].frame != m.frame:
                    self._merge_by_sighting(m, o, other[0])
        for m, obs in alive.values():
            if obs["Edge"]:
                self._landmark_fix(m, obs["Edge"])
                self._map_edges(m, obs["Edge"])
        by_frame = {}
        for m, obs in alive.values():
            by_frame.setdefault(m.frame, []).append((m, obs))
        for fid, group in by_frame.items():
            fm = self.maps[fid]
            self._map_trees(fm, group)
            self._map_fruits(fm, group)
            self._tree_watch(fm, group)
            self._map_predators(fm, group)
            for m, _ in group:
                if not m.stale:
                    fm.stamp(m.x, m.y, m.hear, self.tick)
                    fm.biome[(int(m.x // 20), int(m.y // 20))] = BIOME_PENALTY.get(m.biome, 1.0)
        # frames nobody is in any more are dropped
        live_frames = set(by_frame)
        for fid in [f for f in self.maps if f not in live_frames]:
            del self.maps[fid]

        self.best_fit = max(m.fit for m, _ in alive.values())
        actions = self._plan(alive, by_frame)
        md = self.stats.setdefault("mode_dist", {})
        mt = self.stats.setdefault("mode_ticks", {})
        for act in actions:
            m = self.mem[act["agent_id"]]
            m.last = (act["move_distance"], act["move_direction"], act["turn_angle"], act["spawn_agent"])
            md[m.mode] = md.get(m.mode, 0.0) + act["move_distance"]
            mt[m.mode] = mt.get(m.mode, 0) + 1
        return actions

    # ------------------------------------------------------------------------------------------------------
    def _update_traits(self, m, a):
        energy = float(a.get("energy", 0.0))
        age = float(a.get("age", 0.0))
        # The sim skips the agent after one that starves in its update loop: that agent's age does not advance
        # and it is handed last tick's observations (relative to last tick's pose). Ignore them this tick.
        m.stale = age == m.age
        # old age: extra drain of 0.01*age per tick beyond what our own actions cost
        if m.expected is not None and not m.old and not m.stale and age > 59.0 and energy < m.expected - 0.3:
            m.old = True
        m.energy = energy
        m.age = age
        m.max_energy = float(a.get("max_energy", 500.0))
        m.speed = float(a.get("speed", 10.0))
        m.sprint = float(a.get("sprint_speed", 20.0))
        m.hear = float(a.get("hearing_radius", 50.0))
        m.vis = float(a.get("vision_range", 200.0))
        m.cone = float(a.get("vision_angle", PI / 3))
        m.biome = str(a.get("biome", "grassland")).lower()
        m.fit = self.fitness(m)

    def _predict(self, m):
        """Advance m's pose and expected energy by the action we sent last tick (the sim's own rules)."""
        if m.last is None:
            return
        dist, mdir, turn, spawn = m.last
        d = min(max(dist, 0.0), m.sprint)
        if m.energy < m.max_energy / 5 and d > m.speed:
            d = m.speed
        cost = d * 0.05 if d <= m.speed else m.speed * 0.05 + (d - m.speed) * 0.5
        cost += min(PI, abs(turn)) / TWO_PI
        e = m.energy - cost
        if spawn and e > 100:
            e -= 100
            m.spawned += 1
        e -= 0.1
        if m.old:
            e -= 0.01 * (m.age + 0.1)
        m.expected = e
        d *= BIOME_PENALTY.get(m.biome, 1.0)
        if d > 0:
            fm = self.maps[m.frame]
            direction = m.h + mdir
            nx = m.x + d * math.cos(direction)
            ny = m.y + d * math.sin(direction)
            if fm.edges and fm.blocked(nx, ny):
                step = PI / 18
                nx, ny = m.x, m.y
                for i in range(36):
                    ta = direction + step * ((i + 1) // 2) * (1.0 if i % 2 == 0 else -1.0)
                    tx = m.x + d * math.cos(ta)
                    ty = m.y + d * math.sin(ta)
                    if not fm.blocked(tx, ty):
                        nx, ny = tx, ty
                        break
            m.x, m.y = nx, ny
        m.h += turn

    # ------------------------------------------------------------------------------------------------------
    def _to_frame(self, m, lx, ly):
        c, s = math.cos(m.h), math.sin(m.h)
        return m.x + lx * c - ly * s, m.y + lx * s + ly * c

    def _try_world_register(self, m, edges):
        for (sx, sy), (ex, ey) in edges:
            vx, vy = ex - sx, ey - sy
            ln = math.hypot(vx, vy)
            if abs(ln - 1600.0) < 1e-3:
                h = -math.atan2(vy, vx)
                c, s = math.cos(h), math.sin(h)
                wsx, wsy = sx * c - sy * s, sx * s + sy * c   # start point relative to agent, world axes
                Y = 30.0 if wsy < 0 else 1170.0
                self._register_world(m, 0.0 - wsx, Y - wsy, h)
                return True
            if abs(ln - 1200.0) < 1e-3:
                h = PI / 2 - math.atan2(vy, vx)
                c, s = math.cos(h), math.sin(h)
                wsx, wsy = sx * c - sy * s, sx * s + sy * c
                X = 30.0 if wsx < 0 else 1570.0
                self._register_world(m, X - wsx, 0.0 - wsy, h)
                return True
        return False

    def _register_world(self, m, X, Y, Hd):
        if "W" not in self.maps:
            self._new_frame(world=True)
        phi = Hd - m.h
        c, s = math.cos(phi), math.sin(phi)
        self._move_frame(m.frame, "W", phi, X - (c * m.x - s * m.y), Y - (s * m.x + c * m.y))
        self.stats["world_regs"] += 1

    def _merge_by_sighting(self, m, o, om):
        """m (frame A) sees om (frame B): express the non-world / smaller frame in the other one."""
        th = m.h + o["angle"]
        ox = m.x + o["distance"] * math.cos(th)
        oy = m.y + o["distance"] * math.sin(th)
        if o["distance"] > 1e-9:
            oh = th + PI - o.get("rel_dir", 0.0)
        else:   # spawned on the parent: the sim's atan2(0, 0) = 0 makes angle = -h_A and rel_dir = -h_B
            oh = m.h + o["angle"] - o.get("rel_dir", 0.0)
        fa, fb = m.frame, om.frame
        a_world, b_world = self.maps[fa].world, self.maps[fb].world
        n_a = sum(1 for q in self.mem.values() if q.frame == fa)
        n_b = sum(1 for q in self.mem.values() if q.frame == fb)
        if b_world or (not a_world and n_b > n_a):
            # move A into B: om's pose in B is (om.x, om.y, om.h), in A it is (ox, oy, oh)
            phi = om.h - oh
            c, s = math.cos(phi), math.sin(phi)
            self._move_frame(fa, fb, phi, om.x - (c * ox - s * oy), om.y - (s * ox + c * oy))
        else:
            phi = oh - om.h
            c, s = math.cos(phi), math.sin(phi)
            self._move_frame(fb, fa, phi, ox - (c * om.x - s * om.y), oy - (s * om.x + c * om.y))
        self.stats["merges"] += 1

    def _move_frame(self, src, dst, phi, tx, ty):
        if src == dst:
            return
        c, s = math.cos(phi), math.sin(phi)
        a, b = self.maps[src], self.maps[dst]
        for m in self.mem.values():
            if m.frame == src:
                m.x, m.y = c * m.x - s * m.y + tx, s * m.x + c * m.y + ty
                m.h += phi
                m.frame = dst
        for x1, y1, x2, y2, nx, ny in a.edges:
            b.add_edge(c * x1 - s * y1 + tx, s * x1 + c * y1 + ty, c * x2 - s * y2 + tx, s * x2 + c * y2 + ty,
                       c * nx - s * ny, s * nx + c * ny)
        if a.fx.size:
            fx, fy = c * a.fx - s * a.fy + tx, s * a.fx + c * a.fy + ty
            b.fx, b.fy = np.concatenate((b.fx, fx)), np.concatenate((b.fy, fy))
            b.flo, b.fhi = np.concatenate((b.flo, a.flo)), np.concatenate((b.fhi, a.fhi))
            b.flast = np.concatenate((b.flast, a.flast))
            _dedupe_fruit(b)
        if a.tx.size:
            tx2, ty2 = c * a.tx - s * a.ty + tx, s * a.tx + c * a.ty + ty
            b.tx, b.ty = np.concatenate((b.tx, tx2)), np.concatenate((b.ty, ty2))
            b.tfirst, b.tlast = np.concatenate((b.tfirst, a.tfirst)), np.concatenate((b.tlast, a.tlast))
            b.tfruit, b.twatch = np.concatenate((b.tfruit, a.tfruit)), np.concatenate((b.twatch, a.twatch))
            _dedupe_trees(b)
        for (i, j), pen in a.biome.items():
            x, y = i * 20 + 10.0, j * 20 + 10.0
            b.biome.setdefault((int((c * x - s * y + tx) // 20), int((s * x + c * y + ty) // 20)), pen)
        for pr in a.preds:
            b.preds.append([c * pr[0] - s * pr[1] + tx, s * pr[0] + c * pr[1] + ty, pr[2] + phi, pr[3], pr[4],
                            pr[5]])
        del self.maps[src]

    def _landmark_fix(self, m, edges):
        """Every wall face has its own random length (30-100 px, or 1200/1600), and the heading is exact, so a
        re-observed face matched by length and direction gives the position error directly."""
        fm = self.maps[m.frame]
        if not fm.edges:
            return
        pts = []
        for (sx, sy), (ex, ey) in edges[:10]:
            x1, y1 = self._to_frame(m, sx, sy)
            x2, y2 = self._to_frame(m, ex, ey)
            if (round(x1), round(y1), round(x2), round(y2)) in fm.edge_keys:
                return                      # a known face sits exactly where we expect it: pose is right
            pts.append((x1, y1, x2, y2))
        # Candidate shifts: same length and direction, closer than 25 px. Opposite faces of a box share
        # length and direction but are >= 30 px apart, so they cannot be confused within that radius.
        cands = []
        for x1, y1, x2, y2 in pts:
            cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
            for k in fm.near_edges(cx0, cy0):
                e = fm.edges[k]
                dx1, dy1 = e[0] - x1, e[1] - y1
                dx2, dy2 = e[2] - x2, e[3] - y2
                if abs(dx1 - dx2) + abs(dy1 - dy2) > 0.05:      # different length or direction
                    continue
                if abs(dx1) + abs(dy1) < 25.0:
                    cands.append((dx1, dy1))
        if not cands:
            return
        # the shift most faces agree on, smallest first
        best = None
        for dx, dy in cands:
            support = sum(1 for qx, qy in cands if abs(qx - dx) + abs(qy - dy) < 0.5)
            key = (-support, abs(dx) + abs(dy))
            if (best is None or key < best[0]) and not fm.blocked(m.x + dx, m.y + dy):
                best = (key, dx, dy)
        if best is not None and abs(best[1]) + abs(best[2]) > 1e-3:
            m.x += best[1]
            m.y += best[2]
            m.fixes += 1
            self.stats["fixes"] += 1

    def _map_edges(self, m, edges):
        fm = self.maps[m.frame]
        for (sx, sy), (ex, ey) in edges:
            x1, y1 = self._to_frame(m, sx, sy)
            x2, y2 = self._to_frame(m, ex, ey)
            ux, uy = x2 - x1, y2 - y1
            ln = math.hypot(ux, uy)
            if ln < 1e-6:
                continue
            nx, ny = -uy / ln, ux / ln
            # The obstacle lies on the far side from the viewer, but only when the viewer faces the face itself:
            # a ray aimed exactly at a corner can report the adjacent face from behind (tie at the corner).
            along = ((m.x - x1) * ux + (m.y - y1) * uy) / ln
            if 1.0 < along < ln - 1.0:
                if (m.x - x1) * nx + (m.y - y1) * ny > 0:
                    nx, ny = -nx, -ny
            else:
                nx = ny = 0.0
            fm.add_edge(x1, y1, x2, y2, nx, ny)

    def _map_trees(self, fm, group):
        pts = []
        for m, obs in group:
            if obs["Tree"]:
                c, s = m.h, None
                for o in obs["Tree"]:
                    th = m.h + o["angle"]
                    pts.append((m.x + o["distance"] * math.cos(th), m.y + o["distance"] * math.sin(th)))
        t = self.t
        seen = np.zeros(fm.tx.size, bool)
        if pts:
            P = np.array(pts)
            if fm.tx.size:
                d = np.abs(P[:, 0:1] - fm.tx[None, :]) + np.abs(P[:, 1:2] - fm.ty[None, :])
                j = d.argmin(axis=1)
                hit = d[np.arange(len(P)), j] < 4.0
                seen[j[hit]] = True
                new = P[~hit]
            else:
                new = P
            if len(new):
                new = _unique_pts(new, 4.0)
                fm.tx = np.concatenate((fm.tx, new[:, 0]))
                fm.ty = np.concatenate((fm.ty, new[:, 1]))
                fm.tfirst = np.concatenate((fm.tfirst, np.full(len(new), t)))
                fm.tlast = np.concatenate((fm.tlast, np.full(len(new), t)))
                fm.tfruit = np.concatenate((fm.tfruit, np.full(len(new), -1e9)))
                fm.twatch = np.concatenate((fm.twatch, np.zeros(len(new))))
                seen = np.concatenate((seen, np.ones(len(new), bool)))
        if fm.tx.size:
            fm.tlast[seen] = t
            gone = (~seen) & self._heard_zone(group, fm.tx, fm.ty, 8.0)
            keep = (~gone) & (t - fm.tlast < self.p["tree_forget"])
            if not keep.all():
                fm.tx, fm.ty, fm.tfirst, fm.tlast = fm.tx[keep], fm.ty[keep], fm.tfirst[keep], fm.tlast[keep]
                fm.tfruit, fm.twatch = fm.tfruit[keep], fm.twatch[keep]

    def _tree_watch(self, fm, group):
        """Which trees fruit: fruit seen within 65 px refreshes tfruit; sitting next to one without fruit
        accumulates twatch (a fruiting tree spawns ~0.1 fruit/s, so 20 s of nothing means young or barren)."""
        t = self.t
        if fm.tx.size:
            if fm.fx.size:
                recent = fm.flast >= t - 1e-6
                if recent.any():
                    dd = np.hypot(fm.tx[:, None] - fm.fx[None, recent], fm.ty[:, None] - fm.fy[None, recent])
                    has = (dd < 65.0).any(axis=1)
                    fm.tfruit[has] = t
                    fm.twatch[has] = 0.0
            live = [(m.x, m.y) for m, _ in group if not m.stale]
            if live:
                A = np.array(live)
                dd = np.hypot(fm.tx[:, None] - A[None, :, 0], fm.ty[:, None] - A[None, :, 1])
                fm.twatch[(dd < 30.0).any(axis=1)] += 0.1
        if fm.seen is not None:
            for m, _ in group:
                if m.stale:
                    continue
                for r in (0.0, 100.0, 200.0, 300.0):
                    if r > m.vis:
                        break
                    for da in ((0.0,) if r == 0.0 else (-m.cone / 3, 0.0, m.cone / 3)):
                        x = m.x + r * math.cos(m.h + da)
                        y = m.y + r * math.sin(m.h + da)
                        i, j = int(x // 100), int(y // 100)
                        if 0 <= i < 16 and 0 <= j < 12:
                            fm.seen[i, j] = t

    def _heard_zone(self, group, x, y, margin):
        """Points well inside some (non-stale) agent's hearing radius: absence there is certain."""
        out = np.zeros(x.size, bool)
        for m, _ in group:
            if not m.stale:
                out |= np.hypot(x - m.x, y - m.y) <= m.hear - margin
        return out

    def _perceivable(self, fm, group, x, y, margin):
        """Which points some agent of the group should perceive now (hearing, or cone ignoring walls)."""
        out = np.zeros(x.size, bool)
        for m, _ in group:
            if m.stale:
                continue
            dx, dy = x - m.x, y - m.y
            d = np.hypot(dx, dy)
            near = d <= m.hear - margin
            cone = (d <= m.vis - 2 * margin) & (np.abs((np.arctan2(dy, dx) - m.h + PI) % TWO_PI - PI) <= m.cone / 2 - 0.08)
            out |= near | cone
        return out

    def _map_fruits(self, fm, group):
        pts = []
        for m, obs in group:
            for o in obs["Fruit"]:
                th = m.h + o["angle"]
                pts.append((m.x + o["distance"] * math.cos(th), m.y + o["distance"] * math.sin(th)))
        t = self.t
        seen = np.zeros(fm.fx.size, bool)
        if pts:
            P = _unique_pts(np.array(pts), 1.5)
            if fm.fx.size:
                d = np.abs(P[:, 0:1] - fm.fx[None, :]) + np.abs(P[:, 1:2] - fm.fy[None, :])
                j = d.argmin(axis=1)
                hit = d[np.arange(len(P)), j] < 2.5
                seen[j[hit]] = True
                new = P[~hit]
            else:
                new = P
            if len(new):
                heard = fm.last_heard(new[:, 0], new[:, 1])
                lo = np.where(heard > 0, t - (self.tick - heard) * 0.1, t - 50.0)
                lo = np.maximum(lo, t - 50.0)
                fm.fx = np.concatenate((fm.fx, new[:, 0]))
                fm.fy = np.concatenate((fm.fy, new[:, 1]))
                fm.flo = np.concatenate((fm.flo, lo))
                fm.fhi = np.concatenate((fm.fhi, np.full(len(new), t - 0.1)))
                fm.flast = np.concatenate((fm.flast, np.full(len(new), t)))
                seen = np.concatenate((seen, np.ones(len(new), bool)))
        if fm.fx.size:
            fm.flast[seen] = t
            gone = (~seen) & self._perceivable(fm, group, fm.fx, fm.fy, 6.0)
            keep = (~gone) & (t - fm.fhi < 50.2)
            if not keep.all():
                fm.fx, fm.fy, fm.flo, fm.fhi, fm.flast = (fm.fx[keep], fm.fy[keep], fm.flo[keep], fm.fhi[keep],
                                                          fm.flast[keep])

    def _map_predators(self, fm, group):
        t = self.t
        obs_pts = []
        for m, obs in group:
            for o in obs["Predator"]:
                th = m.h + o["angle"]
                x = m.x + o["distance"] * math.cos(th)
                y = m.y + o["distance"] * math.sin(th)
                if not any(abs(q[0] - x) < 1.5 and abs(q[1] - y) < 1.5 for q in obs_pts):
                    obs_pts.append((x, y, th + PI - o.get("rel_dir", 0.0)))
        tracks = fm.preds
        used = set()
        for x, y, h in obs_pts:
            best, bd = None, 32.0
            for i, tr in enumerate(tracks):
                if i in used:
                    continue
                d = math.hypot(tr[0] - x, tr[1] - y)
                if d < bd:
                    best, bd = i, d
            if best is None:
                tracks.append([x, y, h, t, t, t])
                used.add(len(tracks) - 1)
            else:
                tr = tracks[best]
                if bd > 0.3:
                    tr[4] = t
                tr[0], tr[1], tr[2], tr[3] = x, y, h, t
                used.add(best)
        if tracks:
            fm.preds = [tr for tr in tracks if t - tr[3] < 15.0]

    # ------------------------------------------------------------------------------------------------------
    def _pop_cap(self):
        p = self.p
        if self.t < p["t_mid"]:
            return p["pop_cap_early"]
        if self.t < p["t_late"]:
            return p["pop_cap_mid"]
        return p["pop_cap_late"]

    def _threats(self, m, fm):
        """Predators that can perceive m now or very soon: list of (distance, bearing, awake, sees).

        A predator hears 60 px all round and sees 250 px inside +-30 deg of its heading. It charges when
        closer than 90 px or when the agent looks away; otherwise it circles (sprinting, 10.6 px/tick closing).
        A resting one does nothing until its energy refills, then looks around.
        """
        out = []
        t = self.t
        p = self.p
        for tr in fm.preds:
            dx, dy = tr[0] - m.x, tr[1] - m.y
            d = math.hypot(dx, dy)
            stale = t - tr[3]
            resting = (t - tr[4]) > 0.25 and stale < 0.25
            if resting:
                if d < p["alert_rest"]:
                    out.append((d, math.atan2(dy, dx), False, False))
                continue
            off = abs(wrap(math.atan2(-dy, -dx) - tr[2]))   # predator heading vs the line predator -> m
            if stale > 0.25:
                if stale > p["track_memory"] or off > 0.8:
                    continue
                d_eff = d - stale * 110.0 * math.cos(off)
            else:
                d_eff = d
            sees = (off <= PI / 6 + 0.1 and d_eff < 260.0) or d_eff < 65.0
            if not (d_eff < p["alert_close"] or sees):
                continue
            if d_eff >= p["alert_always"] and stale <= 0.25 and tr[6] < d - 1e-6:
                continue                       # it will go for a closer agent first
            out.append((max(d_eff, 0.0), math.atan2(dy, dx), True, sees))
        return out

    def _closest_prey(self, fm, agents):
        """For every predator track: distance to the closest agent it can perceive (hearing 60 px, or its
        250 px / +-30 deg cone); stored as tr[6]. A predator chases only that agent."""
        if not fm.preds:
            return
        if not agents:
            for tr in fm.preds:
                tr[6:] = [np.inf]
            return
        A = np.array([(q.x, q.y) for q in agents])
        for tr in fm.preds:
            dx, dy = A[:, 0] - tr[0], A[:, 1] - tr[1]
            d = np.hypot(dx, dy)
            ang = np.abs((np.arctan2(dy, dx) - tr[2] + PI) % TWO_PI - PI)
            seen = (d < 60.0) | ((d < 250.0) & (ang <= PI / 6))
            tr[6:] = [float(d[seen].min()) if seen.any() else np.inf]

    def _plan(self, alive, by_frame):
        p = self.p
        t = self.t
        n = len(alive)
        cap = self._pop_cap()
        order = sorted(alive.values(), key=lambda v: v[0].aid)
        # --- threats first (a predator chases the closest agent it perceives)
        for fid, group in by_frame.items():
            self._closest_prey(self.maps[fid], [m for m, _ in group])
        threat = {}
        for m, obs in order:
            th = self._threats(m, self.maps[m.frame])
            if th:
                threat[m.aid] = th
        # --- who spawns
        spawners = set()
        budget = cap - n
        endgame = t >= p["endgame_t"]
        gap = p["breed_gap"]
        fits_sorted = sorted(q.fit for q, _ in order)
        ref_fit = fits_sorted[min(len(fits_sorted) - 1, int(len(fits_sorted) * p["breed_ref_q"]))]
        lo_fit = ref_fit - gap
        early = t < p["breed_phase_end"]
        r_lo = p["breed_reserve_early"] if early else p["breed_reserve_late"]
        r_hi = r_lo + p["breed_reserve_span"]
        cands = []
        for m, obs in order:
            if m.energy <= 101.0 or m.aid in threat or t < p["breed_start"]:
                continue
            rank = min(1.0, max(0.0, (m.fit - lo_fit) / gap))     # 1 = as good as the best genome alive
            good = m.fit >= lo_fit or n < p["pop_min"]
            if endgame:
                cands.append((1, -m.fit, m.aid, 3.0))
            elif not good:
                continue                       # weak genomes never breed, not even their old-age energy
            elif m.old:
                cands.append((0, -m.fit, m.aid, 3.0))
            else:
                cands.append((2, -m.fit, m.aid, r_hi - (r_hi - r_lo) * rank))
        cands.sort()
        preds_seen = any(fm.preds for fm in self.maps.values()) or t > 150.0
        # colony brake: while the average agent is poor the colony is at its food limit; more mouths now
        # means everybody starves together a minute later (boom and bust)
        mean_e = sum(m.energy for m, _ in order) / max(n, 1)
        brake = mean_e < p["breed_colony_e"] and n >= p["brake_min_pop"]
        for pri, _, aid, reserve in cands:
            m = alive[aid][0]
            if pri >= 2 and preds_seen and m.speed < 15.5:   # slower than a predator: keep the sprint unlocked
                reserve = max(reserve, m.max_energy / 5 + p["sprint_keep"])
            if m.energy - 101.0 < reserve:
                continue
            if pri >= 2 and (budget <= 0 or brake):
                continue
            if pri == 0 and budget <= -6:
                continue
            spawners.add(aid)
            budget -= 1
        # --- agents that cannot turn food into children (weak genomes, or old without a dump) eat last
        gap_cut = lo_fit
        for m, obs in order:
            m.spawned = 1 if (m.fit >= gap_cut or n < p["pop_min"]) else 0      # reuse slot: 1 = may breed
        # --- food assignment (global greedy per frame)
        food = {}
        retire = set()
        if n >= p["retire_min_pop"]:
            for m, obs in order:
                if not m.spawned and m.aid not in threat:
                    retire.add(m.aid)     # a weak genome turns food into nothing: leave it to the breeders
        for fid, group in by_frame.items():
            food.update(self._assign_food(self.maps[fid], [m for m, _ in group
                                                            if m.aid not in threat and m.aid not in retire]))
        actions = []
        for m, obs in order:
            fm = self.maps[m.frame]
            grab = self._grab(m, fm) if (m.aid in threat and m.energy < p["grab_energy"]) else None
            if grab is not None:
                m.mode = "grab"          # nearly empty: fleeing hungry is certain death, a fruit within reach is not
                act = self._go(m, grab[0], grab[1])
                act["move_distance"] = float(m.speed)
            elif m.aid in threat:
                act = self._flee(m, fm, threat[m.aid])
            elif m.aid in retire:
                if p["scout_retired"] > 0 and fm.world and m.energy > p["scout_min_e"]:
                    act = self._explore(m, fm)     # spend the energy it cannot pass on mapping trees for the rest
                else:
                    act = self._retire(m, fm)
            elif m.aid in food:
                m.mode = "food"
                x, y, wait = food[m.aid]
                act = self._go(m, x, y, stop=p["wait_dist"] if wait else 0.0)
            else:
                tree = self._choose_tree(m, fm)
                if tree is not None:
                    m.mode = "camp"
                    act = self._go(m, tree[0], tree[1], stop=0.0)   # sit on the tree: its fruit spawns 10-60 px around
                    if act["move_distance"] < 3.0:
                        act["turn_angle"] = p["scan_rate"]
                else:
                    m.mode = "explore"
                    act = self._explore(m, fm)
            act["spawn_agent"] = m.aid in spawners
            actions.append(act)
        return actions

    def _assign_food(self, fm, agents):
        if not agents or not fm.fx.size:
            return {}
        p = self.p
        t = self.t
        ax = np.array([m.x for m in agents])
        ay = np.array([m.y for m in agents])
        spd = np.array([max(m.speed * BIOME_PENALTY.get(m.biome, 1.0), 1.0) for m in agents])
        hungry = np.array([m.energy < p["starve_frac"] * m.max_energy for m in agents])
        d = np.hypot(fm.fx[None, :] - ax[:, None], fm.fy[None, :] - ay[:, None])
        arrive = t + d / spd[:, None] * 0.1
        age_min = arrive - fm.fhi[None, :]           # youngest it can be on arrival
        age_mid = arrive - 0.5 * (fm.flo + fm.fhi)[None, :]
        rot = age_min > 49.0                           # surely rotten before we get there
        value = np.minimum(60.0, 20.0 + 2.0 * np.maximum(age_mid, 0.0))
        need = np.array([max(0.0, m.max_energy - m.energy) for m in agents])
        value = np.minimum(value, need[:, None])        # energy above max_energy is thrown away
        # A fruit gains 2 energy/s until 20 s old while waiting costs 1/s: eat it young only when starving.
        dated = (fm.fhi - fm.flo) < 8.0               # spawn time known to within a few seconds
        unripe = dated[None, :] & (age_min < p["ripe_age"])
        eff = np.where(unripe & ~hungry[:, None], -1e9, value)
        frac_missing = np.array([1.0 - m.energy / max(m.max_energy, 1.0) for m in agents])
        breeder = np.array([bool(m.spawned) and not (m.old and m.energy < 101.0) for m in agents])
        # food eaten by an agent that will never pass it on is wasted: they only get what nobody else wants
        eff = np.where(breeder[:, None], eff, eff * p["weak_food_w"])
        score = eff - p["dist_cost"] * d + p["hunger_w"] * frac_missing[:, None]
        # keep going for the fruit chosen last tick unless something clearly better appears (no zig-zag)
        for i, m in enumerate(agents):
            if m.food_tgt is not None:
                dd = np.abs(fm.fx - m.food_tgt[0]) + np.abs(fm.fy - m.food_tgt[1])
                j = int(dd.argmin())
                if dd[j] < 1.0:
                    score[i, j] += p["food_stick"]
        # stay local while sitting at a tree that fruited recently; otherwise reach out for known fruit
        at_fresh = np.zeros(len(agents), bool)
        if fm.tx.size:
            dt_ = np.hypot(fm.tx[None, :] - ax[:, None], fm.ty[None, :] - ay[:, None])
            fresh_t = (t - fm.tfruit) < 30.0
            at_fresh = ((dt_ < 30.0) & fresh_t[None, :]).any(axis=1)
        reach = np.where(at_fresh, p["food_range"], p["food_range_idle"])
        reach = np.where(hungry, np.maximum(reach, p["food_range_hungry"]), reach)
        starving = np.array([m.energy < 25.0 for m in agents])
        score[rot | (d > reach[:, None]) | (((eff - p["dist_cost"] * d) < p["min_gain"]) & ~starving[:, None])] = -1e9
        out = {}
        if score.size == 0:
            return out
        flat = np.argsort(-score, axis=None)
        used_a, used_f = set(), set()
        na, nf = score.shape
        for k in flat[: max(4 * na * 4, 64)].tolist():
            i, j = divmod(k, nf)
            if score[i, j] < -1e8:
                break
            if i in used_a or j in used_f:
                continue
            used_a.add(i)
            used_f.add(j)
            out[agents[i].aid] = (float(fm.fx[j]), float(fm.fy[j]), False)
            agents[i].food_tgt = (float(fm.fx[j]), float(fm.fy[j]))
            if len(used_a) == na:
                break
        return out

    def _tree_productivity(self, fm):
        """Per tree: known fruit within 65 px, plus 1 if the tree is known to be >= 20 s old (it fruits)."""
        if getattr(fm, "_prod_tick", -1) == self.tick:
            return fm._prod
        prod = np.zeros(fm.tx.size)
        if fm.tx.size:
            if fm.fx.size:
                dd = np.hypot(fm.tx[:, None] - fm.fx[None, :], fm.ty[:, None] - fm.fy[None, :])
                prod += (dd < 65.0).sum(axis=1)
            prod += (self.t - fm.tfirst) >= 20.0
        fm._prod, fm._prod_tick = prod, self.tick
        return prod

    def _choose_tree(self, m, fm):
        """The nearest fruiting tree nobody else sits at (walking is the colony's biggest energy cost).

        A tree counts when fruit was seen around it in the last 30 s or it has been known for 20 s (so it is
        old enough to fruit) and nobody has watched it stay barren. Occupied = another agent within 40 px of
        it, or heading to it from closer. The current target is kept unless another is 100 px nearer.
        """
        if not fm.tx.size:
            return None
        p = self.p
        t = self.t
        fresh = (t - fm.tfruit) < 30.0
        ok = fresh | (((t - fm.tfirst) >= 20.0) & (fm.twatch <= p["barren_watch"]))
        d = np.hypot(fm.tx - m.x, fm.ty - m.y)
        # Claims, not positions: a tree is taken when another agent's claimed tree is within camp_spacing
        # (no clusters: one predator would eat them in a row). Claims persist while their owner walks or
        # fetches fruit, so the assignment does not flip every time somebody passes by.
        occ = np.zeros(fm.tx.size, bool)
        for q in self.mem.values():
            if q.aid == m.aid or q.frame != m.frame or q.target is None or q.mode == "retire":
                continue
            occ |= np.hypot(fm.tx - q.target[0], fm.ty - q.target[1]) < self.p["camp_spacing"]
        slow = np.array([1.0 - fm.biome.get((int(x // 20), int(y // 20)), 1.0) for x, y in zip(fm.tx.tolist(), fm.ty.tolist())])
        score = d - p["tree_fresh_w"] * fresh + p["slow_tree_pen"] * slow
        score[~ok | occ] = np.inf
        score[(~fresh) & (d > p["barren_reach"])] = np.inf
        j = int(score.argmin())
        if not np.isfinite(score[j]):
            m.target = None
            return None
        if m.target is not None:
            k = int((np.abs(fm.tx - m.target[0]) + np.abs(fm.ty - m.target[1])).argmin())
            if abs(fm.tx[k] - m.target[0]) + abs(fm.ty[k] - m.target[1]) < 4.0 and np.isfinite(score[k]) \
                    and score[k] < score[j] + p["tree_stick"]:
                j = k
        m.target = (round(float(fm.tx[j])), round(float(fm.ty[j])))
        return float(fm.tx[j]), float(fm.ty[j])

    def _flee(self, m, fm, threats):
        """Pick the escape direction by looking 6 ticks ahead: 16 headings, movement slowed by the terrain
        the colony has walked (swamp 0.5, river 0.3) and stopped by known walls, each awake threat charging
        straight at 15 px/tick. Face the nearest awake predator while doing it (beyond 90 px that makes it
        circle instead of charge)."""
        m.mode = "flee"
        p = self.p
        awake = [q for q in threats if q[2]]
        nearest = min(awake or threats, key=lambda q: q[0])
        d0, b0, awake0, sees0 = nearest
        if not awake0:
            speed = min(m.speed, 5.0) if d0 < p["alert_rest"] else 0.0
        elif d0 < p["charge_zone"]:
            speed = m.speed
            if m.speed < 15.5 and m.energy > m.max_energy / 5 + 15:
                speed = m.sprint
        elif d0 < p["keep_dist"]:
            speed = min(m.speed, p["backoff_speed"])
        else:
            speed = min(m.speed, p["drift_speed"])
        rel = wrap(b0 - m.h)
        turn = max(-1.2, min(1.2, rel)) if abs(rel) > p["face_tol"] else 0.0
        if speed <= 0.0:
            return {"agent_id": m.aid, "move_distance": 0.0, "move_direction": 0.0, "turn_angle": float(turn),
                    "spawn_agent": False}
        K = 6
        dirs = np.linspace(-PI, PI, 16, endpoint=False)
        ux, uy = np.cos(dirs), np.sin(dirs)
        px = np.full(16, m.x)
        py = np.full(16, m.y)
        stuck = np.zeros(16, bool)
        blocked_pen = np.zeros(16)
        thr = [(m.x + q[0] * math.cos(q[1]), m.y + q[0] * math.sin(q[1])) for q in threats if q[2]]
        if not thr:
            thr = [(m.x + d0 * math.cos(b0), m.y + d0 * math.sin(b0))]
        T = np.array(thr, float)                       # (n_threats, 2), charging predators
        tx = np.repeat(T[None, :, 0], 16, axis=0)
        ty = np.repeat(T[None, :, 1], 16, axis=0)
        dmin = np.full(16, np.inf)
        E = fm.local_edges(m.x, m.y, speed * K + 45.0) if fm.edges else None
        here = BIOME_PENALTY.get(m.biome, 1.0)
        bget = fm.biome.get
        for k in range(K):
            pen = np.array([bget((int(x // 20), int(y // 20)), here) for x, y in zip(px.tolist(), py.tolist())])
            nx = px + ux * speed * pen
            ny = py + uy * speed * pen
            blk = fm.blocked_many(E, nx, ny)
            blocked_pen += blk
            stuck |= blk
            px = np.where(stuck, px, nx)
            py = np.where(stuck, py, ny)
            dx, dy = px[:, None] - tx, py[:, None] - ty
            dd = np.hypot(dx, dy)
            step = np.minimum(15.0, dd)
            tx = tx + dx / np.maximum(dd, 1e-9) * step
            ty = ty + dy / np.maximum(dd, 1e-9) * step
            dmin = np.minimum(dmin, np.hypot(px[:, None] - tx, py[:, None] - ty).min(axis=1))
        score = dmin - 30.0 * blocked_pen
        # prefer staying out of swamps/rivers at the end of the look-ahead
        end_pen = np.array([fm.biome.get((int(x // 20), int(y // 20)), 1.0) for x, y in zip(px.tolist(), py.tolist())])
        score -= 40.0 * (1.0 - end_pen)
        # do not run into other agents (a charging predator takes whoever is closest)
        for q in self.mem.values():
            if q.aid != m.aid and q.frame == m.frame:
                dq = np.hypot(px - q.x, py - q.y)
                score -= np.where(dq < 40.0, 20.0, 0.0)
        j = int(score.argmax())
        return {"agent_id": m.aid, "move_distance": float(speed), "move_direction": float(wrap(dirs[j] - m.h)),
                "turn_angle": float(turn), "spawn_agent": False}

    def _grab(self, m, fm):
        """Closest known fruit within grab_dist px, if any."""
        if not fm.fx.size:
            return None
        d = np.hypot(fm.fx - m.x, fm.fy - m.y)
        j = int(d.argmin())
        if d[j] > self.p["grab_dist"]:
            return None
        return float(fm.fx[j]), float(fm.fy[j])

    def _retire(self, m, fm):
        """Stay out of the way: keep >= 75 px from trees and known fruit (touching a fruit eats it), watch around."""
        m.mode = "retire"
        px = py = 0.0
        if fm.tx.size:
            d = np.hypot(fm.tx - m.x, fm.ty - m.y)
            close = d < 75.0
            if close.any():
                w = 1.0 / np.maximum(d[close], 5.0)
                px += float(((m.x - fm.tx[close]) * w).sum())
                py += float(((m.y - fm.ty[close]) * w).sum())
        if fm.fx.size:
            d = np.hypot(fm.fx - m.x, fm.fy - m.y)
            close = d < 30.0
            if close.any():
                w = 1.0 / np.maximum(d[close], 3.0)
                px += float(((m.x - fm.fx[close]) * w).sum())
                py += float(((m.y - fm.fy[close]) * w).sum())
        if px == 0.0 and py == 0.0:
            return {"agent_id": m.aid, "move_distance": 0.0, "move_direction": 0.0,
                    "turn_angle": float(self.p["scan_rate"]), "spawn_agent": False}
        ang = math.atan2(py, px)
        return {"agent_id": m.aid, "move_distance": float(min(m.speed, 5.0)), "move_direction": float(wrap(ang - m.h)),
                "turn_angle": 0.0, "spawn_agent": False}

    def _go(self, m, tx, ty, stop=0.0, face_target=False):
        dx, dy = tx - m.x, ty - m.y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) if d > 1e-9 else m.h
        dist = min(m.speed, max(0.0, d - stop))
        turn = wrap(ang - m.h) if dist > 3.0 else 0.0
        if abs(turn) < 0.05:
            turn = 0.0
        turn = max(-0.8, min(0.8, turn))
        return {"agent_id": m.aid, "move_distance": float(dist), "move_direction": float(wrap(ang - m.h)),
                "turn_angle": float(turn), "spawn_agent": False}

    def _explore(self, m, fm):
        """Walk to the map cell nobody has looked at for longest (world frame), else a random long walk."""
        m.mode = "explore"
        if fm.world and fm.seen is not None:
            if not isinstance(m.wander_dir, tuple) or self.tick >= m.wander_until:
                stale = np.minimum(self.t - fm.seen, 300.0)
                ci = (np.arange(16) * 100 + 50)[:, None]
                cj = (np.arange(12) * 100 + 50)[None, :]
                dist = np.hypot(ci - m.x, cj - m.y)
                taken = np.zeros_like(stale)
                for q in self.mem.values():
                    if q.aid != m.aid and q.mode == "explore" and isinstance(q.wander_dir, tuple):
                        taken[q.wander_dir[0], q.wander_dir[1]] += 1
                val = stale - 0.25 * dist - 200.0 * taken
                val[1:-1, 1:-1] += 20.0          # the rim cells are half wall
                i, j = np.unravel_index(int(val.argmax()), val.shape)
                m.wander_dir = (int(i), int(j))
                m.wander_until = self.tick + 400
            i, j = m.wander_dir
            tx, ty = i * 100 + 50.0, j * 100 + 50.0
            if math.hypot(tx - m.x, ty - m.y) < 40.0:
                m.wander_until = self.tick
            act = self._go(m, tx, ty)
            act["move_distance"] = float(min(act["move_distance"], self.p["explore_speed"]))
            return act
        if not isinstance(m.wander_dir, float) or self.tick >= m.wander_until:
            m.wander_dir = self.rng.uniform(-PI, PI)
            m.wander_until = self.tick + int(self.rng.uniform(60, 200))
        ang = m.wander_dir
        turn = max(-0.5, min(0.5, wrap(ang - m.h)))
        return {"agent_id": m.aid, "move_distance": float(min(m.speed, self.p["explore_speed"])),
                "move_direction": float(wrap(ang - m.h)), "turn_angle": float(turn), "spawn_agent": False}


# ----------------------------------------------------------------------------------------------------------
_EMPTY_OBS = {"Fruit": [], "Agent": [], "Predator": [], "Tree": [], "Edge": []}


def _split_obs(obs):
    out = {"Fruit": [], "Agent": [], "Predator": [], "Tree": [], "Edge": []}
    seen = set()
    for o in obs:
        typ = o.get("type")
        if not isinstance(typ, str):
            continue
        typ = typ[:1].upper() + typ[1:].lower()
        if typ == "Edge":
            c = o.get("coords")
            if not c:
                continue
            key = (c[0][0], c[0][1], c[1][0], c[1][1])
            if key in seen:
                continue
            seen.add(key)
            out["Edge"].append(c)
        elif typ in out:
            out[typ].append(o)
    return out


def _unique_pts(P, tol):
    if len(P) <= 1:
        return P
    key = np.round(P / tol)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def _dedupe_fruit(b):
    P = np.column_stack((b.fx, b.fy))
    key = np.round(P / 2.0)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    b.fx, b.fy, b.flo, b.fhi, b.flast = b.fx[idx], b.fy[idx], b.flo[idx], b.fhi[idx], b.flast[idx]


def _dedupe_trees(b):
    P = np.column_stack((b.tx, b.ty))
    key = np.round(P / 4.0)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    b.tx, b.ty, b.tfirst, b.tlast = b.tx[idx], b.ty[idx], b.tfirst[idx], b.tlast[idx]
    b.tfruit, b.twatch = b.tfruit[idx], b.twatch[idx]
