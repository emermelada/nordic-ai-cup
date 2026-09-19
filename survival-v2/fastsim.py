"""Fast drop-in replacement for the official survival simulator.

Import this module instead of building `official.src.core.SimulationCore` directly. It patches the official
Environment so the world is the same one the official code builds from a seed (same RNG calls in the same
order), but a tick costs a fraction of the time:

* no pygame surfaces (the biome render still consumes the RNG exactly like `_render_biome_surface`);
* static wall geometry cached per chunk (same set-build order as the official grid, so the same edge order);
* visibility polygon ray casting compiled with numba, with walls beyond vision range dropped (they cannot
  produce a hit closer than the range, so the polygon and the hit edges are unchanged);
* `shapely.contains_xy` for the vision-cone check (same predicate as `Polygon.contains(Point)`);
* dynamic grids rebuilt lazily (once when queried) instead of after every single change;
* list-with-removal loop quirks of `non_agent_step` kept as they are (they decide which entities are
  skipped in a tick and how much RNG is consumed).

Float results can differ from numpy in the last bit (numba's libm), so long trajectories may drift apart
from the official simulator; statistics and the generated world match. See `verify_fastsim.py`.
"""
import math
import os
import sys
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.join(_HERE, "official")
if _OFFICIAL not in sys.path:
    sys.path.insert(0, _OFFICIAL)

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402


from src.elements import environment as _envmod  # noqa: E402
from src.elements.biome import Map_generator  # noqa: E402
from src.elements.environment import Environment  # noqa: E402
from src.elements.obstacle import Obstacle  # noqa: E402
from src.utils import simulation as _simmod  # noqa: E402

TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------------------------------------------
# Geometry kernels
# ----------------------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _visibility(x, y, direction, cone, radius, ex1, ey1, ex2, ey2, eps):
    """Same algorithm as official `compute_visibility`, on edges already filtered to the vision range.

    Returns (px, py) of the polygon points sorted by relative angle (stable), and the edge index hit by each
    ray in ray order (-1 for no hit).
    """
    n_edges = ex1.shape[0]
    r2 = radius * radius
    half = cone / 2
    rays = np.empty(n_edges * 6 + 5)
    nr = 0
    for i in range(n_edges):
        for k in range(2):
            if k == 0:
                cx = ex1[i]
                cy = ey1[i]
            else:
                cx = ex2[i]
                cy = ey2[i]
            dx = cx - x
            dy = cy - y
            if dx * dx + dy * dy <= r2:
                base = math.atan2(dy, dx)
                for o in range(3):
                    a = base + (o - 1) * eps
                    rel = (a - direction + math.pi) % TWO_PI - math.pi
                    if abs(rel) <= half:
                        rays[nr] = a
                        nr += 1
    rays[nr] = direction - cone / 2
    rays[nr + 1] = direction + cone / 2
    rays[nr + 2] = direction - cone / 4
    rays[nr + 3] = direction + cone / 4
    rays[nr + 4] = direction
    nr += 5
    rays = np.unique(rays[:nr])
    nr = rays.shape[0]
    px = np.empty(nr)
    py = np.empty(nr)
    hit = np.full(nr, -1, np.int64)
    for r in range(nr):
        rdx = math.cos(rays[r])
        rdy = math.sin(rays[r])
        best_t = np.inf
        best_j = -1
        for j in range(n_edges):
            vx = ex2[j] - ex1[j]
            vy = ey2[j] - ey1[j]
            det = -rdx * vy + rdy * vx
            if abs(det) >= 1e-8:
                ox = ex1[j] - x
                oy = ey1[j] - y
                t = (-vy * ox + vx * oy) / det
                u = (-rdy * ox + rdx * oy) / det
                if t >= 0 and u >= 0 and u <= 1:
                    if t < best_t:
                        best_t = t
                        best_j = j
        mt = best_t if best_t < radius else radius
        px[r] = x + rdx * mt
        py[r] = y + rdy * mt
        if best_t < radius:
            hit[r] = best_j
    keys = np.empty(nr)
    for r in range(nr):
        a = math.atan2(py[r] - y, px[r] - x) - direction
        keys[r] = (a + math.pi) % TWO_PI - math.pi
    order = np.argsort(keys, kind="mergesort")
    return px[order], py[order], hit


@nb.njit(cache=True)
def _edges_in_range(x, y, radius, ex1, ey1, ex2, ey2):
    """Indices of edges whose closest point is within `radius` (+1e-6) of (x, y)."""
    n = ex1.shape[0]
    out = np.empty(n, np.int64)
    m = 0
    lim = (radius + 1e-6) * (radius + 1e-6)
    for j in range(n):
        vx = ex2[j] - ex1[j]
        vy = ey2[j] - ey1[j]
        wx = x - ex1[j]
        wy = y - ey1[j]
        ll = vx * vx + vy * vy
        t = 0.0
        if ll > 0:
            t = (wx * vx + wy * vy) / ll
            if t < 0:
                t = 0.0
            elif t > 1:
                t = 1.0
        dx = wx - t * vx
        dy = wy - t * vy
        if dx * dx + dy * dy <= lim:
            out[m] = j
            m += 1
    return out[:m]


@nb.njit(cache=True)
def _point_in_ring(px, py, rx, ry):
    """Even-odd test against the closed ring (rx, ry); same answer as shapely `contains` off the boundary."""
    inside = False
    n = rx.shape[0]
    j = n - 1
    for i in range(n):
        yi = ry[i]
        yj = ry[j]
        if (yi > py) != (yj > py):
            xi = rx[i]
            xj = rx[j]
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


@nb.njit(cache=True)
def _select(cx, cy, direction, hear, vis, half_cone, xs, ys, alive, skip, rx, ry):
    """Official `process_objects` selection over flat arrays: heard (<= hearing) first, then seen
    (<= vision, inside the cone and inside the visibility polygon). Returns (indices, distances, angles).

    Anything within vision (<= 400 = chunk size) lies inside the official 9-chunk neighbourhood, so testing
    every entity is the same as the official chunk gather.
    """
    n = xs.shape[0]
    near = np.empty(n, np.int64)
    seen = np.empty(n, np.int64)
    nn = 0
    ns = 0
    dist = np.empty(n)
    ang = np.empty(n)
    for i in range(n):
        if not alive[i] or i == skip:
            continue
        dx = xs[i] - cx
        dy = ys[i] - cy
        d = math.hypot(dx, dy)
        if d > vis and d > hear:
            continue
        a = math.atan2(dy, dx) - direction
        a = (a + math.pi) % TWO_PI - math.pi
        dist[i] = d
        ang[i] = a
        if d <= hear:
            near[nn] = i
            nn += 1
        elif d <= vis and abs(a) <= half_cone:
            if _point_in_ring(xs[i], ys[i], rx, ry):
                seen[ns] = i
                ns += 1
    out = np.empty(nn + ns, np.int64)
    out[:nn] = near[:nn]
    out[nn:] = seen[:ns]
    return out, dist, ang


@nb.njit(cache=True)
def _in_obstacle(px, py, radius, xs, ys, ws, hs):
    for i in range(xs.shape[0]):
        if (xs[i] - radius < px) and (px < xs[i] + ws[i] + radius) and (ys[i] - radius < py) and (
                py < ys[i] + hs[i] + radius):
            return True
    return False


@nb.njit(cache=True)
def _collision_move(prev_x, prev_y, distance, direction, radius, xs, ys, ws, hs):
    """Official collision handling: try the move, then rotate 10 degree steps alternating sides."""
    nx = prev_x + distance * math.cos(direction)
    ny = prev_y + distance * math.sin(direction)
    if not _in_obstacle(nx, ny, radius, xs, ys, ws, hs):
        return nx, ny
    angle_step = math.pi / 18
    max_attempts = int(2 * math.pi / angle_step)
    for i in range(max_attempts):
        k = (i + 1) // 2
        sign = 1.0 if i % 2 == 0 else -1.0
        test_angle = direction + angle_step * k * sign
        tx = prev_x + distance * math.cos(test_angle)
        ty = prev_y + distance * math.sin(test_angle)
        if not _in_obstacle(tx, ty, radius, xs, ys, ws, hs):
            return tx, ty
    return prev_x, prev_y


# ----------------------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------------------
class _ChunkStatic:
    """Walls around one chunk (its 9-chunk neighbourhood), in the official set-iteration order."""
    __slots__ = ("edges", "ex1", "ey1", "ex2", "ey2", "oxs", "oys", "ows", "ohs", "has_edges", "has_obs")

    def __init__(self, env, cx, cy):
        local_obs = set()
        local_edges = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ch = (cx + dx, cy + dy)
                local_obs.update(env.grid_obstacles.get(ch, []))
                local_edges.update(env.grid_edges.get(ch, []))
        self.edges = list(local_edges)
        arr = np.array(self.edges, dtype=float).reshape(-1, 2, 2)
        self.ex1 = np.ascontiguousarray(arr[:, 0, 0])
        self.ey1 = np.ascontiguousarray(arr[:, 0, 1])
        self.ex2 = np.ascontiguousarray(arr[:, 1, 0])
        self.ey2 = np.ascontiguousarray(arr[:, 1, 1])
        obs = list(local_obs)
        self.oxs = np.array([o.x for o in obs], dtype=float)
        self.oys = np.array([o.y for o in obs], dtype=float)
        self.ows = np.array([o.width for o in obs], dtype=float)
        self.ohs = np.array([o.height for o in obs], dtype=float)
        self.has_edges = len(self.edges) > 0
        self.has_obs = len(obs) > 0


class FastEnvironment(Environment):
    def __init__(self, width, height, chunk_size, rng):
        self.rng = rng
        self.width = width
        self.height = height
        self.agents = []
        self.agents_dict = {}
        self._next_agent_id = 0
        self.fruits = []
        self.fruits_dict = {}
        self._next_fruit_id = 0
        self.trees = []
        self.obstacles = []
        self.edges = []
        self.predators = []
        self.score = 0
        self.time = 0

        map_generator = Map_generator(self.width, self.height, self.rng, num_biomes=10, num_rivers=1)
        self.biome_map = map_generator.generate()
        self._consume_render_rng()
        self._biome_arrays()

        self.chunk_size = chunk_size
        self.grid_agents = defaultdict(set)
        self.grid_fruits = defaultdict(set)
        self.grid_trees = defaultdict(set)
        self.grid_obstacles = defaultdict(set)
        self.grid_edges = defaultdict(set)
        self.grid_predators = defaultdict(set)
        self._dirty_agents = self._dirty_fruits = self._dirty_trees = self._dirty_predators = True
        self._static = {}
        self.agent_observations = {}
        self._create_boundaries(thickness=30)
        self._update_spatial_grid()

    # --- world build -------------------------------------------------------------------------------------
    def _consume_render_rng(self):
        """Consume the RNG exactly like `_render_biome_surface` (one `rng.choice(palette)` per pixel)."""
        lens = np.empty(self.biome_map.shape, dtype=np.int64)
        seen = {}
        flat = self.biome_map.ravel()
        for b in flat:
            if id(b) not in seen:
                seen[id(b)] = b
        for b in seen.values():
            lens[self.biome_map == b] = len(b.color_palette)
        randbelow = self.rng._randbelow
        for n in lens.ravel().tolist():
            randbelow(n)

    def _biome_arrays(self):
        shape = self.biome_map.shape
        self.b_move = np.empty(shape)
        self.b_drain = np.empty(shape)
        self.b_fruit = np.empty(shape)
        self.b_type = np.empty(shape, dtype=object)
        seen = {}
        for b in self.biome_map.ravel():
            seen.setdefault(id(b), b)
        for b in seen.values():
            m = self.biome_map == b
            self.b_move[m] = b.move_penalty
            self.b_drain[m] = b.energy_drain_rate
            self.b_fruit[m] = b.fruit_spawn_rate
            self.b_type[m] = b.type

    def _render_biome_surface(self):  # never called; kept so nothing draws by accident
        raise RuntimeError("FastEnvironment does not render")

    def spawn_obstacle(self, x=None, y=None, width=None, height=None, color=(128, 128, 128)):
        if width is None:
            width = self.rng.uniform(30, 100)
        if height is None:
            height = self.rng.uniform(30, 100)
        if x is None:
            x = self.rng.uniform(0, self.width - width)
        if y is None:
            y = self.rng.uniform(0, self.height - height)
        obs = Obstacle(x, y, width=width, height=height, color=color)
        self.obstacles.append(obs)
        self.edges = set()
        for o in self.obstacles:
            self.edges.update([
                ((o.x, o.y), (o.x + o.width, o.y)),
                ((o.x + o.width, o.y), (o.x + o.width, o.y + o.height)),
                ((o.x, o.y + o.height), (o.x + o.width, o.y + o.height)),
                ((o.x, o.y), (o.x, o.y + o.height)),
            ])
        self._update_obstacle_grid()
        self._update_edge_grid()
        self._static = {}
        return obs

    def draw(self, screen):
        raise RuntimeError("FastEnvironment does not render")

    # --- lazy grids ----------------------------------------------------------------------------------------
    def _update_agent_grid(self):
        self._dirty_agents = True

    def _update_fruit_grid(self):
        self._dirty_fruits = True

    def _update_tree_grid(self):
        self._dirty_trees = True

    def _update_predator_grid(self):
        self._dirty_predators = True

    def _sync(self):
        cs = self.chunk_size
        if self._dirty_agents:
            g = defaultdict(set)
            for a in self.agents:
                g[(int(a.x // cs), int(a.y // cs))].add(a)
            self.grid_agents = g
            self._dirty_agents = False
        if self._dirty_fruits:
            g = defaultdict(set)
            for f in self.fruits:
                g[(int(f.x // cs), int(f.y // cs))].add(f)
            self.grid_fruits = g
            self._dirty_fruits = False
        if self._dirty_trees:
            g = defaultdict(set)
            for t in self.trees:
                g[(int(t.x // cs), int(t.y // cs))].add(t)
            self.grid_trees = g
            self._dirty_trees = False
        if self._dirty_predators:
            g = defaultdict(set)
            for p in self.predators:
                g[(int(p.x // cs), int(p.y // cs))].add(p)
            self.grid_predators = g
            self._dirty_predators = False

    def _chunk_static(self, x, y):
        key = (int(x // self.chunk_size), int(y // self.chunk_size))
        st = self._static.get(key)
        if st is None:
            st = _ChunkStatic(self, key[0], key[1])
            self._static[key] = st
        return st

    def _gather(self, grid, x, y):
        cx, cy = int(x // self.chunk_size), int(y // self.chunk_size)
        out = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                s = grid.get((cx + dx, cy + dy))
                if s:
                    out.update(s)
        return out

    def _get_local_agents(self, creature):
        self._sync()
        s = self._gather(self.grid_agents, creature.x, creature.y)
        s.discard(creature)
        return s

    def _get_local_fruits(self, creature):
        self._sync()
        return self._gather(self.grid_fruits, creature.x, creature.y)

    def _get_local_trees(self, creature):
        self._sync()
        return self._gather(self.grid_trees, creature.x, creature.y)

    def _get_local_predators(self, creature):
        self._sync()
        s = self._gather(self.grid_predators, creature.x, creature.y)
        s.discard(creature)
        return s

    # --- movement ------------------------------------------------------------------------------------------
    def update_entity_position(self, entity, distance, direction=None, local_obstacles=None):
        walking_cost = 0.05
        sprinting_cost = 0.5
        if distance < 0:
            distance = 0
        if distance > entity.sprint_speed:
            distance = entity.sprint_speed
        if entity.energy < entity.max_energy / 5 and distance > entity.speed:
            distance = entity.speed
        if distance <= entity.speed:
            entity.energy -= distance * walking_cost
        else:
            entity.energy -= entity.speed * walking_cost + (distance - entity.speed) * sprinting_cost
        if direction is None:
            direction = entity.direction
        else:
            direction = entity.direction + direction
        x = min(max(int(entity.x), 0), self.width - 1)
        y = min(max(int(entity.y), 0), self.height - 1)
        distance *= self.b_move[x, y]
        st = self._chunk_static(entity.x, entity.y)
        if st.has_obs:
            nx, ny = _collision_move(float(entity.x), float(entity.y), float(distance), float(direction),
                                     float(entity.size), st.oxs, st.oys, st.ows, st.ohs)
        else:
            nx = entity.x + distance * math.cos(direction)
            ny = entity.y + distance * math.sin(direction)
        entity.x, entity.y = nx, ny
        self._keep_agent_in_bounds(entity)

    def agent_step(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent=False):
        agent = self.agents_dict.get(agent_id)
        if agent is None:
            return
        self.update_entity_position(agent, move_distance, move_direction)
        self._dirty_agents = True
        self.update_entity_direction(agent, turn_angle)
        if spawn_agent and agent.energy > 100:
            self.spawn_agent(parent=agent)
            agent.energy -= 100

    # --- perception ----------------------------------------------------------------------------------------
    def _observe(self, c, skip_agent, agents, fruits, trees, predators):
        """Official `Creature.observe` output for creature `c`.

        agents/fruits/trees/predators are (objs, xs, ys, dirs, alive) tuples over every entity of that type
        (or None). `skip_agent` is c's index in the agent arrays (-1 for predators).
        """
        observations = []
        direction = c.direction
        cos_dir, sin_dir = math.cos(-direction), math.sin(-direction)
        half_cone = c.cone_angle / 2.0
        st = self._chunk_static(c.x, c.y)
        cx, cy = float(c.x), float(c.y)
        fdir = float(direction)
        if st.has_edges:
            idx = _edges_in_range(cx, cy, float(c.vision_radius), st.ex1, st.ey1, st.ex2, st.ey2)
            px, py, hit = _visibility(cx, cy, fdir, float(c.cone_angle), float(c.vision_radius),
                                      st.ex1[idx], st.ey1[idx], st.ex2[idx], st.ey2[idx], 1e-3)
            hit_edges = [st.edges[idx[j]] for j in hit if j >= 0]
        else:  # official early return: rays at full range, unsorted, no hits
            cone = c.cone_angle
            rays = np.unique([direction - cone / 2, direction + cone / 2, direction - cone / 4,
                              direction + cone / 4, direction])
            px = cx + np.cos(rays) * c.vision_radius
            py = cy + np.sin(rays) * c.vision_radius
            hit_edges = []
        rx = np.empty(len(px) + 1)
        ry = np.empty(len(px) + 1)
        rx[0], ry[0] = cx, cy
        rx[1:] = px
        ry[1:] = py
        hr, vr = float(c.hearing_radius), float(c.vision_radius)

        def process(group, tag, skip):
            objs, xs, ys, dirs, alive = group
            if not objs:
                return
            sel, dist, ang = _select(cx, cy, fdir, hr, vr, half_cone, xs, ys, alive, skip, rx, ry)
            if not sel.size:
                return
            dl = dist[sel].tolist()
            al = ang[sel].tolist()
            if dirs is not None:
                rel = ((np.arctan2(cy - ys[sel], cx - xs[sel]) - dirs[sel] + np.pi) % TWO_PI) - np.pi
                rl = rel.tolist()
                if tag == "Agent":
                    for k, i in enumerate(sel.tolist()):
                        observations.append({"type": tag, "distance": dl[k], "angle": al[k], "rel_dir": rl[k],
                                             "id": objs[i].agent_id})
                else:
                    for k in range(sel.size):
                        observations.append({"type": tag, "distance": dl[k], "angle": al[k], "rel_dir": rl[k]})
            else:
                for k in range(sel.size):
                    observations.append({"type": tag, "distance": dl[k], "angle": al[k]})

        if fruits is not None:
            process(fruits, "Fruit", -1)
        if agents is not None:
            process(agents, "Agent", skip_agent)
        if predators is not None:
            process(predators, "Predator", -1)
        if trees is not None:
            process(trees, "Tree", -1)
        for (sx, sy), (ex, ey) in hit_edges:
            dxs, dys, dxe, dye = sx - cx, sy - cy, ex - cx, ey - cy
            observations.append({"type": "Edge", "coords": (
                (dxs * cos_dir - dys * sin_dir, dxs * sin_dir + dys * cos_dir),
                (dxe * cos_dir - dye * sin_dir, dxe * sin_dir + dye * cos_dir))})
        return observations

    @staticmethod
    def _pack(objs, with_dir):
        n = len(objs)
        xs = np.fromiter((o.x for o in objs), float, n)
        ys = np.fromiter((o.y for o in objs), float, n)
        dirs = np.fromiter((o.direction for o in objs), float, n) if with_dir else None
        return objs, xs, ys, dirs, np.ones(n, np.bool_)

    # --- tick ----------------------------------------------------------------------------------------------
    def non_agent_step(self, dt):
        W1, H1 = self.width - 1, self.height - 1
        # Snapshot every entity once. Agents and predators do not move during the agent loop; eaten fruit
        # and starved agents are masked out as the official grid rebuilds would drop them.
        A = self._pack(list(self.agents), True)
        a_index = {id(a): i for i, a in enumerate(A[0])}
        F = self._pack(list(self.fruits), False)
        f_rad = np.fromiter((f.radius for f in F[0]), float, len(F[0]))
        T = self._pack(list(self.trees), False)
        P = self._pack(list(self.predators), True)
        a_alive, f_alive = A[4], F[4]
        for agent in self.agents:
            agent.age += dt
            agent.energy -= dt * self.b_drain[min(max(int(agent.x), 0), W1), min(max(int(agent.y), 0), H1)]
            if agent.energy <= 0:
                self.kill_agent(agent)
                a_alive[a_index[id(agent)]] = False
                continue
            if agent.age > agent.max_age:
                agent.energy -= 0.01 * agent.age
            ai = a_index[id(agent)]
            self.agent_observations[agent.agent_id] = self._observe(agent, ai, A, F, T, P)
            if F[0]:
                ax, ay = agent.x, agent.y
                d = np.hypot(F[1] - ax, F[2] - ay)
                for i in np.flatnonzero(f_alive & (d < agent.size + f_rad)).tolist():
                    fruit = F[0][i]
                    agent.energy = min(agent.max_energy, agent.energy + fruit.energy)
                    self.score += fruit.energy / 1000
                    self.remove_fruit(fruit)
                    f_alive[i] = False

        for predator in self.predators:
            if predator.resting:
                if predator.energy > predator.max_energy * 0.5:
                    predator.resting = False
                else:
                    predator.energy += dt * 30
                    continue
            observation = self._observe(predator, -1, A, None, None, None)
            signals = predator.step(observation)
            if "move" in signals:
                self.update_entity_position(predator, signals["move"], signals["direction"])
                self._dirty_predators = True
            if "turn" in signals:
                self.update_entity_direction(predator, signals["turn"])
            if A[0]:
                d = np.hypot(A[1] - predator.x, A[2] - predator.y)
                touching = np.flatnonzero(a_alive & (d < predator.size + 5.0)).tolist()
                for i in reversed(touching):
                    agent = A[0][i]
                    predator.energy = min(predator.max_energy, predator.energy + agent.energy)
                    self.score -= agent.energy / 100
                    self.kill_agent(agent)
                    a_alive[i] = False
            if predator.energy <= 0:
                predator.resting = True
                continue

        for fruit in self.fruits:
            if fruit.age > 100:
                self.remove_fruit(fruit)
                continue
            fruit.grow(amount=2 * dt)

        tree_spawn_chance = 100 / max(1, len(self.trees) / 2) * dt
        tree_spawn_chance *= 0.5 ** (self.time / 300)
        if self.rng.random() < tree_spawn_chance:
            self.spawn_tree()

        rnd = self.rng.random
        for tree in self.trees:
            tree.grow(amount=1 * dt)
            if tree.age > 50 + (100 - 50) * (rnd() ** (1 / 2)):
                self.remove_tree(tree)
                continue
            elif tree.age >= 20:
                if rnd() < dt * self.b_fruit[min(max(int(tree.x), 0), W1), min(max(int(tree.y), 0), H1)]:
                    self.spawn_fruit_around_tree(tree)

        self.time += dt
        self.score += dt
        predator_spawn_chance = (1 / max(1, len(self.predators))) * dt * self.time * 0.0001
        if predator_spawn_chance > self.rng.random():
            self.spawn_predator()

    def get_agent_state(self, agent_id):
        agent = self.agents_dict.get(agent_id)
        if agent is None:
            return None
        ix = min(max(int(agent.x), 0), self.width - 1)
        iy = min(max(int(agent.y), 0), self.height - 1)
        return {
            "agent_id": agent.agent_id,
            "observations": self.agent_observations.get(agent_id, []),
            "energy": agent.energy,
            "biome": self.b_type[ix, iy],
            "age": agent.age,
            "speed": agent.speed,
            "sprint_speed": agent.sprint_speed,
            "hearing_radius": agent.hearing_radius,
            "vision_angle": agent.cone_angle,
            "vision_range": agent.vision_radius,
            "max_energy": agent.max_energy,
        }


# Route the official factory to the fast class. SimulationCore -> create_environment -> Environment.
_simmod.Environment = FastEnvironment

from src.core import SimulationCore  # noqa: E402,F401  (re-exported; builds FastEnvironment now)


class Action:
    __slots__ = ("move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False):
        self.move_distance = move_distance
        self.move_direction = move_direction
        self.turn_angle = turn_angle
        self.spawn_agent = spawn_agent


def to_actions(action_dicts):
    """[{agent_id, move_distance, ...}] -> [(agent_id, Action)] as `step_environment` expects."""
    return [(a["agent_id"], Action(float(a["move_distance"]), float(a["move_direction"]), float(a["turn_angle"]),
                                   bool(a["spawn_agent"]))) for a in action_dicts]
