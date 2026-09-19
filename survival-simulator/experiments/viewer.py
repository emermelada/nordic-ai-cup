#!/usr/bin/env python3
"""viewer.py - replay ANALYSIS DASHBOARD for a recorded episode. Read-only presentation layer.

Reads a replay produced by replay.py and renders it as a compact scientific dashboard: large map,
minimap, compact legend/overlays, selected-agent perception + behaviour panel, an event/death
timeline, and prominent current/final score. Needs no simulator, so what you watch is the recorded
evidence, not a re-simulation that might differ. This file NEVER writes to a replay and never alters
simulator, controller, scoring or recording behaviour.

    ./viewer.py /tmp/rp/seed1508_12007t.json.gz
    ./viewer.py <replay> --dump-frames 40 --outdir /tmp/rp/frames   # PNGs, for a vision model later
    ./viewer.py <replay> --text                                     # no window: ASCII + HUD to stdout

LAYOUT
    top bar     title | playback buttons | speed slider (0.1x-100x, log) | current+final score | frame/tick
                draggable progress slider spans the width under it
    left col    minimap (click/drag to navigate) + legend + overlays + controls
    centre      MAIN MAP (dominant element)
    right col   selected agent: STATUS | PERCEPTION | BEHAVIOUR | RECENT ACTIONS | DEATH
    bottom bar  CURRENT SCORE + score rate | EVENTS / DEATH TIMELINE (clickable) | FINAL SCORE

CONTROLS (keys are unchanged from the previous viewer)
    space / k    play-pause          left / right   step 1 frame (pauses first)
    up / down    speed x2 / /2       1..5           speed presets 1x 2x 5x 20x 100x
    click agent  select + inspect    t              trails
    v            vision cone         h              hearing radius
    d            death markers       n              jump to next event
    b            jump back 500       r              restart          f  fit view
    q / esc      quit
    mouse: click agent = select, drag map = pan, wheel = zoom, click minimap = jump,
           click timeline = seek / jump to event marker, drag the speed and progress sliders
    CLI: --scale Z initial zoom (default: fit), --start N start frame, --select ID preselect an agent,
         --dump-frames N --outdir DIR writes frames as BMP + sips-converted PNG (redrawn after the
             jump, so the tick in the filename is the tick in the image),
         --text prints ASCII + the HUD instead of opening a window (ssh friendly)

TEXT RENDERING. The pygame in this repo's .venv was built without SDL_ttf, so pygame.font does not
exist and a naive dashboard renders NO labels at all. Fonts() therefore walks a ladder:
pygame.font -> pygame.freetype -> PIL (a real system TTF) -> none (map only, HUD printed to stdout).
Check the active backend with: python -c "import viewer; print(viewer.Fonts().backend)"

INTERPRETERS (this bit has bitten us repeatedly).
    ~/.venv-viewer/bin/python   pygame-ce 2.5.6 -> real pygame.font, best text. Needs nothing else.
    .venv/bin/python            pygame 2.6.1 without SDL_ttf -> PIL fonts. Full simulator deps too.
    /usr/bin/python3            no pygame at all -> `--text` still works, a window cannot.
`--text` imports NOTHING from the simulator stack (no replay.py, no shapely, no torch): it renders
ASCII from the recording, so it runs on any interpreter. The window path needs pygame and says so
with the interpreter list above instead of a bare ModuleNotFoundError.

PERFORMANCE. The biome map is per-pixel (1.9M cells) and the previous viewer re-drew all of them each
frame; terrain and walls are now built once and cached per zoom, and redrawn only when the zoom
changes. Draw cost at 1720x1000: ~6 ms/frame (was ~160 ms), i.e. 60 fps everywhere, including with
trails + vision + hearing + 200 death markers on screen.

WHAT TO LOOK AT (this is the point of the exercise - aggregate statistics hid the mechanism)
    * colour of each agent = energy: green healthy, yellow low, red = inside the <20% sprint-lockout
      zone (drawn with a white ring + hollow centre, so colour is not the only cue). Agents that
      cannot sprint cannot outrun a predator; 92-96% of predation deaths measured there.
    * the right panel shows what the SELECTED agent can perceive right now - the vision cone and
      hearing radius are drawn on the map AND in a polar mini-diagram, with counts and the nearest
      threats, so "could it see what killed it?" is answerable in the same screen.
    * 'd' leaves a cross where each agent died, coloured by cause; deaths in the last 200 ticks are
      emphasised and the bottom timeline histogram shows whether deaths cluster in time.
    * clicking a death (map cross or timeline marker) selects that agent and jumps to its death tick,
      opening the recorded death context: cause, energy before death, nearest predator, whether the
      predator was visible, lockout, and the last commanded actions.

DERIVED vs RECORDED. Anything not present in the replay file is labelled "(derived)" in the UI and
computed only from recorded numbers: perception counts come from recorded positions + the recorded
per-agent vision_range/vision_angle (walls occlude, as in the simulator), speed/heading from frame
to frame position deltas, fruit-eat ticks from score deltas (score = int dt + fruit/1000 - eaten/100),
age from the recorded first appearance. Predator heading is derived from frame-to-frame displacement
(predators have no id in the recording); predator detection rings use the simulator's constant
ratios (predator.py: vision 250, hearing 60). Per-agent hearing_radius is NOT recorded, so the
hearing overlay falls back to the simulator default (creature.py: 50) and says so.

KNOWN GAPS (honest limits of the recorded data, not of the viewer): per-agent trailing counters
(wall blocked, oscillation, ...) exist only in death events, so the BEHAVIOUR block fills them at
death and shows "-" (with derived live substitutes) while an agent is alive; predator direction is
inferred from displacement, so a stationary predator shows no heading; the vision polygon is a
ray-march against the recorded wall segments, i.e. an approximation of the simulator's own polygon.
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys

try:
    import pygame  # noqa: E402
except ImportError as exc:      # --text needs NO pygame at all: presentation only
    pygame, PYGAME_OK, PYGAME_ERR = None, False, exc
else:
    PYGAME_OK, PYGAME_ERR = True, None

# DO NOT touch SDL_VIDEODRIVER here. An earlier version did:
#     os.environ.setdefault("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", ""))
# which sets the variable to an EMPTY STRING when it is unset, and SDL cannot choose a video driver from
# an empty value: pygame.display.set_mode() then fails with "windows not available" while a bare pygame
# test in the same venv works. If a headless run is wanted, pass SDL_VIDEODRIVER=dummy explicitly on the
# command line - never default it to empty.

# --------------------------------------------------------------------------- constants / theme
BIOME_COLORS = {"Grassland_biome": (48, 84, 44), "Forest_biome": (28, 62, 36),
                "Desert_biome": (128, 112, 72), "Snow_biome": (168, 174, 182),
                "Water_biome": (34, 60, 104), "Mountain_biome": (88, 84, 82),
                "River_biome": (40, 74, 124), "Swamp_biome": (46, 66, 42)}
DEFAULT_BIOME = (60, 60, 66)
CAUSE_COLORS = {"eaten": (232, 66, 66), "starved": (232, 190, 62), "aged": (168, 176, 190)}
ENERGY_HEALTHY = (86, 200, 122)
ENERGY_LOW = (232, 186, 60)
ENERGY_CRIT = (224, 62, 62)
WALL_CORE = (108, 114, 128)
WALL_CASE = (20, 22, 27)
FRUIT_COL = (244, 216, 96)
PRED_COL = (236, 52, 52)
TRAIL_COL = (96, 132, 208)
DEATH_RECENT_TICKS = 200

C_BG = (14, 15, 19)
C_PANEL = (23, 25, 30)
C_PANEL2 = (30, 33, 39)
C_EDGE = (46, 50, 60)
C_TXT = (228, 232, 238)
C_DIM = (142, 150, 164)
C_DIM2 = (98, 106, 120)
C_ACC = (104, 166, 255)
C_GOOD = (92, 204, 128)
C_WARN = (232, 186, 60)
C_BAD = (224, 72, 72)
C_ORANGE = (238, 142, 62)

DT = 0.1                       # simulator tick length (environment.py: score += dt per tick)
DEFAULT_HEARING = 50.0         # creature.py default hearing_radius - NOT recorded per frame
PRED_VISION, PRED_HEARING = 250.0, 60.0   # predator.py constants (not mutated)
ENERGY_LOW_F = 0.45            # yellow below this fraction
ENERGY_CRIT_F = 0.20           # red below this fraction == the sprint-lockout zone


def energy_color(e, max_e):
    """Energy -> colour. The 20% band is the sprint-lockout zone the whole exercise is about."""
    f = e / max(1.0, max_e)
    if f < ENERGY_CRIT_F:
        return ENERGY_CRIT
    if f < ENERGY_LOW_F:
        return ENERGY_LOW
    return ENERGY_HEALTHY


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def fmt_int(v):
    return f"{int(round(v)):,}"


def load_replay(path):
    return json.load(gzip.open(path, "rt")) if path.endswith(".gz") else json.load(open(path))


# --------------------------------------------------------------------------- geometry helpers
def seg_hit(px, py, dx, dy, ax, ay, bx, by):
    """Ray (p + t*d, t>=0) vs segment a-b. Returns t or None."""
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return None
    t = ((ax - px) * ey - (ay - py) * ex) / den
    u = ((ax - px) * dy - (ay - py) * dx) / den
    if t > 1e-6 and -0.001 <= u <= 1.001:
        return t
    return None


class WallGrid:
    """Small uniform grid over the recorded wall segments so occlusion tests stay cheap.

    The simulator gives an agent a VISION POLYGON and excludes entities behind walls
    (creature.py: update_vision -> ray cast; observe(): `visionshape.contains(point)`), so any honest
    "could it see that?" answer has to respect the recorded walls too.
    """

    def __init__(self, edges, w, h, cell=200.0):
        self.edges = edges
        self.cell = cell
        self.cols = max(1, int(math.ceil(w / cell)))
        self.rows = max(1, int(math.ceil(h / cell)))
        self.grid = {}
        for idx, (a, b) in enumerate(edges):
            (x0, y0), (x1, y1) = a, b
            c0 = (int(clamp(min(x0, x1) // cell, 0, self.cols - 1)),
                  int(clamp(min(y0, y1) // cell, 0, self.rows - 1)))
            c1 = (int(clamp(max(x0, x1) // cell, 0, self.cols - 1)),
                  int(clamp(max(y0, y1) // cell, 0, self.rows - 1)))
            for cx in range(c0[0], c1[0] + 1):
                for cy in range(c0[1], c1[1] + 1):
                    self.grid.setdefault((cx, cy), []).append(idx)

    def near(self, x, y, r):
        out = set()
        c0 = (int(clamp((x - r) // self.cell, 0, self.cols - 1)),
              int(clamp((y - r) // self.cell, 0, self.rows - 1)))
        c1 = (int(clamp((x + r) // self.cell, 0, self.cols - 1)),
              int(clamp((y + r) // self.cell, 0, self.rows - 1)))
        for cx in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                out.update(self.grid.get((cx, cy), ()))
        return out

    def ray(self, x, y, ang, maxd):
        """Distance to the first wall along direction ang (capped at maxd)."""
        if not self.edges:
            return maxd
        dx, dy = math.cos(ang), math.sin(ang)
        best = maxd
        for idx in self.near(x, y, maxd):
            (ax, ay), (bx, by) = self.edges[idx]
            t = seg_hit(x, y, dx, dy, ax, ay, bx, by)
            if t is not None and t < best:
                best = t
        return best

    def blocked(self, x, y, tx, ty, slack=0.98):
        """True when a wall stands between (x,y) and (tx,ty)."""
        if not self.edges:
            return False
        dist = math.hypot(tx - x, ty - y)
        if dist < 1e-6:
            return False
        ang = math.atan2(ty - y, tx - x)
        return self.ray(x, y, ang, dist) < dist * slack


# --------------------------------------------------------------------------- replay index
class ReplayData:
    """Everything the dashboard needs, computed ONCE from the recorded bytes."""

    def __init__(self, rec):
        self.rec = rec
        self.frames = rec["frames"]
        self.events = rec.get("events", [])
        self.every = int(rec.get("every", 1) or 1)
        self.n = len(self.frames)
        self.w = int(rec.get("w", 1600))
        self.h = int(rec.get("h", 1200))
        self.seed = rec.get("seed")
        fin = rec.get("final", {}) or {}
        self.final = fin
        self.final_score = float(fin.get("score", self.frames[-1]["score"] if self.frames else 0.0))
        self.ticks = int(fin.get("ticks") or (self.frames[-1]["t"] + self.every if self.frames else 0))
        self.edges = [(tuple(a), tuple(b)) for a, b in rec.get("edges", [])]
        self.grid = WallGrid(self.edges, self.w, self.h)
        self.trees = [tuple(t) for t in (self.frames[0].get("trees", []) if self.frames else [])]

        # recorded events, indexed
        self.deaths = [e for e in self.events if e.get("k") == "die"]
        self.births = [e for e in self.events if e.get("k") == "born"]
        self.death_by_agent = {}
        for e in self.deaths:
            self.death_by_agent.setdefault(e["a"], []).append(e)
        self.birth_by_agent = {}
        for e in self.births:
            self.birth_by_agent.setdefault(e["a"], []).append(e["t"])
        self.ev_by_tick = {}
        for e in self.events:
            self.ev_by_tick.setdefault(e["t"], []).append(e)
        self.causes = {"eaten": 0, "starved": 0, "aged": 0}
        for e in self.deaths:
            self.causes[e.get("cause", "?")] = self.causes.get(e.get("cause", "?"), 0) + 1

        # per-agent traces (position per recorded frame) + derived life stats
        self.trace = {}
        self.first_seen = {}
        self.last_seen = {}
        self.peak_energy = {}
        self.lock_frames = {}
        self.last_fruit_tick = {}
        prev_e = {}
        for i, fr in enumerate(self.frames):
            t = fr["t"]
            for a in fr["agents"]:
                aid, x, y, e, me = a[0], a[1], a[2], a[3], (a[7] if len(a) > 7 else 500.0)
                self.trace.setdefault(aid, []).append((i, x, y))
                if aid not in self.first_seen:
                    self.first_seen[aid] = t
                self.last_seen[aid] = t
                self.peak_energy[aid] = max(self.peak_energy.get(aid, 0.0), me)
                if e < ENERGY_CRIT_F * max(1.0, me):
                    self.lock_frames[aid] = self.lock_frames.get(aid, 0) + 1
                if e > prev_e.get(aid, -1e9) + 5.0:      # energy jump == fruit absorbed
                    self.last_fruit_tick[aid] = t
                prev_e[aid] = e

        # derived fruit-eat ticks: score = int dt + fruit/1000 - eaten_energy/100
        # (environment.py: score += dt per tick, += fruit.energy/1000 on eating, -= eaten.energy/100)
        pen = {}
        for e in self.deaths:
            if e.get("cause") == "eaten":
                pen[e["t"]] = pen.get(e["t"], 0.0) + float(e.get("e", 0.0)) / 100.0
        self.fruit_ticks = []
        self.delta_per_tick = []                     # per frame: observed score change / ticks
        prev_s, prev_t = None, None
        for fr in self.frames:
            t, s = fr["t"], float(fr["score"])
            if prev_s is not None:
                dtk = max(1, t - prev_t)
                gain = (s - prev_s) - DT * dtk + sum(v for k, v in pen.items() if prev_t < k <= t)
                self.delta_per_tick.append((s - prev_s) / dtk)
                if gain > 0.02:
                    self.fruit_ticks.append(t)
            else:
                self.delta_per_tick.append(0.0)
            prev_s, prev_t = s, t
        self.fruit_set = set(self.fruit_ticks)

        # agent id -> last frame index it appears in (for "select a dead agent" behaviour)
        self.agent_frames = {aid: [p[0] for p in pts] for aid, pts in self.trace.items()}

    # ---------------------------------------------------------------- lookups
    def i_for_tick(self, tick):
        i = int(round(tick / self.every))
        return int(clamp(i, 0, self.n - 1))

    def at(self, aid, i):
        """(x, y) of agent aid at frame index i, nearest recorded frame at or before it."""
        pts = self.trace.get(aid)
        if not pts:
            return None
        lo, hi = 0, len(pts) - 1
        if i <= pts[0][0]:
            return pts[0][1], pts[0][2], pts[0][0]
        if i >= pts[hi][0]:
            return pts[hi][1], pts[hi][2], pts[hi][0]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pts[mid][0] <= i:
                lo = mid
            else:
                hi = mid - 1
        return pts[lo][1], pts[lo][2], pts[lo][0]

    def history(self, aid, i, k=60):
        pts = self.trace.get(aid) or []
        out = []
        for p in pts:
            if p[0] > i:
                break
            out.append(p)
        return out[-k:]

    def heading_deg(self, d):
        return (math.degrees(d) + 360.0) % 360.0

    def derived_speed(self, aid, i):
        """World units per tick, from consecutive recorded positions (labelled derived in the UI)."""
        pts = self.trace.get(aid) or []
        prev = None
        for p in pts:
            if p[0] >= i:
                break
            prev = p
        cur = self.at(aid, i)
        if prev is None or cur is None:
            return None
        # prev[0] / cur[2] are FRAME indices -> convert the delta to ticks
        dticks = max(1, cur[2] - prev[0]) * self.every
        return math.hypot(cur[0] - prev[1], cur[1] - prev[2]) / float(dticks)

    def pred_dirs(self, i):
        """Derived predator headings: nearest-match between consecutive frames (predators have no id)."""
        if i <= 0:
            return [None] * len(self.frames[i].get("preds", []))
        cur = self.frames[i].get("preds", [])
        prv = self.frames[i - 1].get("preds", [])
        out = []
        for p in cur:
            best, bd = None, 1e9
            for q in prv:
                d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                if d < bd:
                    bd, best = d, q
            if best is not None and 1.0 < bd ** 0.5 < 120.0:
                out.append(math.atan2(p[1] - best[1], p[0] - best[0]))
            else:
                out.append(None)
        return out

    # ---------------------------------------------------------------- perception (derived)
    def perception(self, aid, i):
        """What the agent can perceive at frame i, from recorded positions + recorded vision geometry.

        Same rules as creature.py observe(): within hearing_radius everything is sensed; otherwise
        within vision_radius AND inside half the cone angle AND not occluded by a wall.
        Returns dict with counts and the entity lists, each entry (x, y, dist, angle).
        """
        fr = self.frames[i]
        hit = [a for a in fr["agents"] if a[0] == aid]
        if not hit:
            return None
        a = hit[0]
        x, y, d = a[1], a[2], a[4]
        vr = a[5] if len(a) > 5 else 300.0
        va = a[6] if len(a) > 6 else 1.0
        hr = a[8] if len(a) > 8 else DEFAULT_HEARING
        half = va / 2.0
        res = {"vision_range": vr, "vision_angle": va, "hearing": hr, "hearing_recorded": len(a) > 8,
               "agents": [], "preds": [], "fruits": [], "walls": 0}
        groups = (("agents", [(b[1], b[2]) for b in fr["agents"] if b[0] != aid]),
                  ("preds", [(p[0], p[1]) for p in fr.get("preds", [])]),
                  ("fruits", [(f[0], f[1]) for f in fr.get("fruits", [])]))
        for key, pts in groups:
            for (ex, ey) in pts:
                dist = math.hypot(ex - x, ey - y)
                ang = (math.atan2(ey - y, ex - x) - d + math.pi) % (2 * math.pi) - math.pi
                if dist <= hr:
                    res[key].append((ex, ey, dist, ang, True))            # sensed by proximity
                elif dist <= vr and abs(ang) <= half and not self.grid.blocked(x, y, ex, ey):
                    res[key].append((ex, ey, dist, ang, False))           # seen in the cone
        # wall segments whose midpoint falls inside the cone (occlusion is not applied here: a wall
        # INSIDE the cone is itself the occluder, so "in view" == in the cone)
        walls = 0
        for idx in self.grid.near(x, y, vr):
            (ax, ay), (bx, by) = self.edges[idx]
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            dist = math.hypot(mx - x, my - y)
            if dist > vr:
                continue
            ang = (math.atan2(my - y, mx - x) - d + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) <= half or dist <= hr:
                walls += 1
        res["walls"] = walls
        return res

    def vision_poly(self, x, y, d, vr, va, rays=30):
        """Occluded vision polygon (approximation of the simulator's own ray-cast polygon)."""
        pts = [(x, y)]
        for k in range(rays + 1):
            ang = d - va / 2.0 + va * k / rays
            dd = self.grid.ray(x, y, ang, vr)
            pts.append((x + math.cos(ang) * dd, y + math.sin(ang) * dd))
        return pts

    def pred_visible(self, aid, i):
        """Was any predator visible to aid at frame i? (derived)
        The frame at a death tick is recorded AFTER the agent died, so the agent is absent from it;
        step back up to two frames to the last frame it WAS alive, which is the honest "what could it
        see just before dying".
        """
        for j in (i, i - 1, i - 2):
            if 0 <= j < self.n:
                p = self.perception(aid, j)
                if p is not None:
                    return len(p["preds"]) > 0
        return None


# --------------------------------------------------------------------------- fonts / widgets
class Fonts:
    """Text rendering with a dependency ladder, so the dashboard's labels never silently vanish.

    WHY THIS IS NOT JUST pygame.font: the pygame 2.6.1 install in this repo's venv was built for
    Python 3.14 without SDL_ttf, so `pygame.font` (and `pygame.freetype`) do not exist AT ALL and
    every label would render as nothing - the previous viewer degraded to printing the HUD to stdout.
    A dashboard whose labels vanish is worse than no dashboard, so we try, in order:
        pygame.font  ->  pygame.freetype  ->  PIL (a real system TTF)  ->  none
    With "none" the map still draws and the panel is printed to stdout (the old behaviour).
    Rendering is cached by (size, text, colour, bold), so per-frame cost is a dict lookup.
    """

    MONO = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/consola.ttf"]
    MONO_BOLD = ["/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "C:/Windows/Fonts/consolab.ttf"]

    def __init__(self):
        self.tcache = {}
        self.cache = {}
        self.pil = {}
        self.names = ["Menlo", "DejaVu Sans Mono", "Consolas", "Courier New"]
        self.faux = {}                 # must exist BEFORE the probe below calls _pil_font
        self.backend = "none"
        self.ok = False
        if PYGAME_OK:
            try:
                import pygame.font  # noqa: F401
                if not pygame.font.get_init():
                    pygame.font.init()
                self.backend, self.ok = "pygame.font", True
            except Exception:
                try:
                    import pygame.freetype  # noqa: F401
                    if not pygame.freetype.get_init():
                        pygame.freetype.init()
                    self.backend, self.ok = "pygame.freetype", True
                except Exception:
                    self.backend, self.ok = "none", False
        if not self.ok:
            # pygame missing entirely, or its wheel was built without SDL_ttf -> try PIL
            try:
                from PIL import Image, ImageDraw, ImageFont
                self._Image, self._ImageDraw, self._ImageFont = Image, ImageDraw, ImageFont
                self._pil_font(12, False)
                if self.pil[(12, False)] is not None:
                    self.backend, self.ok = "PIL", True
            except Exception:
                self.backend, self.ok = "none", False
            # PIL renders with faux bold via a stroke when no bold face is found
            self.faux = {}

    def get(self, size, bold=False):
        key = (size, bold)
        if key not in self.cache:
            try:
                self.cache[key] = pygame.font.SysFont(self.names, size, bold=bold)
            except Exception:
                self.cache[key] = None
        return self.cache[key]

    def _ft(self, size, bold=False):
        key = (size, bold)
        if key not in self.cache:
            try:
                self.cache[key] = pygame.freetype.SysFont(self.names, size, bold=bold)
            except Exception:
                self.cache[key] = None
        return self.cache[key]

    def _pil_font(self, size, bold=False):
        key = (size, bold)
        if key in self.pil:
            return self.pil[key]
        tries = []
        if bold:
            tries += [(p, 0) for p in self.MONO_BOLD]
            tries += [(p, 1) for p in self.MONO if p.endswith(".ttc")]
        tries += [(p, 0) for p in self.MONO]
        f = None
        for path, idx in tries:
            if not os.path.exists(path):
                continue
            try:
                f = self._ImageFont.truetype(path, size, index=idx)
                break
            except Exception:
                f = None
        self.faux[key] = bool(bold)
        self.pil[key] = f
        return f

    def render(self, size, s, col, bold=False):
        if not self.ok:
            return None
        key = (size, s, col, bold)
        if key in self.tcache:
            return self.tcache[key]
        surf = None
        if self.backend == "pygame.font":
            f = self.get(size, bold)
            if f is not None:
                surf = f.render(s, True, col)
        elif self.backend == "pygame.freetype":
            f = self._ft(size, bold)
            if f is not None:
                try:
                    surf = f.render(s, fgcolor=col, size=size)[0]
                except Exception:
                    surf = None
        elif self.backend == "PIL":
            f = self._pil_font(size, bold)
            if f is not None:
                stroke = 1 if self.faux.get((size, bold)) else 0
                try:
                    w = int(f.getlength(s)) + 8 + 2 * stroke
                    img = self._Image.new("RGBA", (max(1, w), size * 2 + 4), (0, 0, 0, 0))
                    self._ImageDraw.Draw(img).text(
                        (2 + stroke, 2), s, font=f, fill=tuple(col) + (255,),
                        stroke_width=stroke, stroke_fill=tuple(col) + (255,))
                    bb = img.getbbox()
                    if bb:
                        img = img.crop(bb)
                    surf = pygame.image.frombytes(img.tobytes(), img.size, "RGBA")
                except Exception:
                    surf = None
        if len(self.tcache) > 4000:
            self.tcache.clear()
        self.tcache[key] = surf
        return surf

    def tw(self, size, s, bold=False):
        if not self.ok:
            return 6 * len(s)
        if self.backend == "pygame.font":
            f = self.get(size, bold)
            return f.size(s)[0] if f else 6 * len(s)
        if self.backend == "pygame.freetype":
            f = self._ft(size, bold)
            if f:
                try:
                    return int(f.get_rect(s, size=size).width)
                except Exception:
                    return 6 * len(s)
            return 6 * len(s)
        f = self._pil_font(size, bold)
        if f is None:
            return 6 * len(s)
        try:
            return int(self._pil_font(size, bold).getlength(s)) + 2
        except Exception:
            return 6 * len(s)


def player_icon(surf, x, y, r, col, state="ok"):
    """Agent glyph: disc + energy-state cue that is NOT colour-only."""
    pygame.draw.circle(surf, col, (int(x), int(y)), int(r))
    if state == "crit":
        pygame.draw.circle(surf, (255, 255, 255), (int(x), int(y)), int(r) + 2, 1)
        pygame.draw.circle(surf, (24, 20, 20), (int(x), int(y)), max(1, int(r * 0.45)))
    elif state == "low":
        pygame.draw.circle(surf, (36, 30, 16), (int(x), int(y)), max(1, int(r * 0.45)))


def pred_icon(surf, x, y, r, ang=None):
    pts = [(x + r * 1.5, y), (x - r * 0.9, y - r * 1.1), (x - r * 0.45, y), (x - r * 0.9, y + r * 1.1)]
    if ang is not None:
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [(x + (px - x) * ca - (py - y) * sa, y + (px - x) * sa + (py - y) * ca) for px, py in pts]
    pygame.draw.polygon(surf, PRED_COL, pts)
    pygame.draw.polygon(surf, (255, 190, 190), pts, 1)


def fruit_icon(surf, x, y, r):
    pygame.draw.polygon(surf, FRUIT_COL, [(x, y - r), (x + r, y), (x, y + r), (x - r, y)])


def death_cross(surf, x, y, r, col, w=2):
    # dark halo first: an old (dimmed) death cross otherwise reads like a piece of fruit
    pygame.draw.line(surf, (10, 10, 12), (x - r, y - r), (x + r, y + r), w + 2)
    pygame.draw.line(surf, (10, 10, 12), (x - r, y + r), (x + r, y - r), w + 2)
    pygame.draw.line(surf, col, (x - r, y - r), (x + r, y + r), w)
    pygame.draw.line(surf, col, (x - r, y + r), (x + r, y - r), w)


# --------------------------------------------------------------------------- camera
class Cam:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.z = 0.5
        self.cx, self.cy = w / 2.0, h / 2.0

    def fit(self, view):
        z = min(view.w / self.W, view.h / self.H)
        self.z = z
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.clamp(view)

    def clamp(self, view):
        half_w = view.w / (2.0 * self.z)
        half_h = view.h / (2.0 * self.z)
        if half_w * 2 >= self.W:
            self.cx = self.W / 2.0
        else:
            self.cx = clamp(self.cx, half_w, self.W - half_w)
        if half_h * 2 >= self.H:
            self.cy = self.H / 2.0
        else:
            self.cy = clamp(self.cy, half_h, self.H - half_h)

    def zoom_at(self, factor, sx, sy, view):
        """Zoom keeping the world point under the cursor fixed."""
        wx, wy = self.to_world(sx, sy, view)
        self.z = clamp(self.z * factor, 0.12, 1.6)
        self.clamp(view)
        wx2, wy2 = self.to_world(sx, sy, view)
        self.cx += wx - wx2
        self.cy += wy - wy2
        self.clamp(view)

    def to_screen(self, wx, wy, view):
        return (view.left + (wx - self.cx) * self.z + view.w / 2.0,
                view.top + (wy - self.cy) * self.z + view.h / 2.0)

    def to_world(self, sx, sy, view):
        return (self.cx + (sx - view.left - view.w / 2.0) / self.z,
                self.cy + (sy - view.top - view.h / 2.0) / self.z)

    def viewport(self, view):
        return pygame.Rect(int(view.left + (0 - self.cx) * self.z + view.w / 2.0),
                           int(view.top + (0 - self.cy) * self.z + view.h / 2.0),
                           int(self.W * self.z), int(self.H * self.z))


# --------------------------------------------------------------------------- player
class Player:
    def __init__(self, rec, scale=None, dump_dir=None, dump_n=0, text_mode=False):
        self.data = ReplayData(rec)
        self.rec = rec
        self.base_zoom = scale
        self.text_mode = text_mode
        self.font = Fonts()          # always present: draw_* must never hit a None font
        self.paused_for_dump = dump_n > 0
        self.dump_dir, self.dump_n, self.dumped = dump_dir, dump_n, 0

        # playback state (keys/behaviour preserved from the previous viewer)
        self.i = 0
        self.i_f = 0.0
        self.playing = False
        self.speed = 1.0
        self.sel = None
        self.trails = False
        self.vision = False
        self.hearing = False
        self.deaths = True
        self.food = True
        self.obstacles = True
        self.terrain = True
        self.cam = Cam(self.data.w, self.data.h)
        self.drag = None
        self.focus = None            # ("death", agent_id) context after clicking a death
        self.rscroll = 0
        self.content_h = 0
        self.view = pygame.Rect(0, 60, 1000, 600) if PYGAME_OK else None
        self.layers = (None, None, None)   # (zoom_key, scaled terrain, scaled walls)
        self._wl = None                    # full-resolution (terrain, walls), built once
        self._mm = None                    # minimap base, built once
        self.ovl = None                    # reusable alpha overlay
        self.strip = None                  # timeline strip cache
        self.strip_key = None
        self.msg = ""
        self.msg_until = 0.0
        self._death_marks = []

    # ------------------------------------------------------------ derived per-frame facts
    def frame(self):
        return self.data.frames[self.i]

    def set_i(self, i, hard=True):
        self.i = int(clamp(i, 0, self.data.n - 1))
        if hard:
            self.i_f = float(self.i)

    def selected(self):
        if self.sel is None:
            return None
        for a in self.frame()["agents"]:
            if a[0] == self.sel:
                return a
        return None

    def tick(self):
        return self.frame()["t"]

    # ------------------------------------------------------------ cached surfaces
    def world_layers(self):
        """Full-resolution terrain + wall surfaces, built ONCE.

        The biome map is per-pixel (1.9M cells) and indexed [x][y]; the previous viewer transposed it
        and redrew all 1.9M rects on EVERY frame.
        """
        if self._wl is not None:
            return self._wl
        W, H = self.data.w, self.data.h
        terr = None
        biome, legend = self.rec.get("biome"), self.rec.get("biome_legend", [])
        if biome:
            try:
                import numpy as np
                arr = np.asarray(biome, dtype=np.int16)
                rgb = np.empty((W, H, 3), dtype=np.uint8)
                rgb[:, :] = DEFAULT_BIOME
                for idx, name in enumerate(legend):
                    rgb[arr == idx] = BIOME_COLORS.get(name, DEFAULT_BIOME)
                terr = pygame.surfarray.make_surface(rgb)
            except Exception:
                terr = pygame.Surface((W, H))
                terr.fill(DEFAULT_BIOME)
        else:
            terr = pygame.Surface((W, H))
            terr.fill((26, 30, 26))
        # walls layer: a chunky dark obstacle band with a muted top edge. The previous viewer drew
        # bright white 3px lines, which dominated the screen and drowned out the agents; an obstacle
        # has to be immediately identifiable WITHOUT competing with the entities.
        walls = pygame.Surface((W, H), pygame.SRCALPHA)
        for (x0, y0), (x1, y1) in self.data.edges:
            pygame.draw.line(walls, (*WALL_CASE, 225), (x0, y0), (x1, y1), 10)
        for (x0, y0), (x1, y1) in self.data.edges:
            pygame.draw.line(walls, (52, 56, 66, 235), (x0, y0), (x1, y1), 6)
        for (x0, y0), (x1, y1) in self.data.edges:
            pygame.draw.line(walls, (*WALL_CORE, 255), (x0, y0), (x1, y1), 2)
        self._wl = (terr, walls)
        return self._wl

    def terrain_surface(self, z):
        """Scaled layer pair for zoom z, cached and rebuilt ONLY when the zoom changes.

        The cache must not be disturbed by the minimap, which needs a different scale: an earlier
        version of this function rebuilt the 1.9M-cell biome array three times per frame because
        minimap_base() called it with z=1.0 and evicted the campaign-zoom entry (160 ms/frame, 6 fps).
        """
        key, terr, walls = self.layers
        if terr is not None and abs(key - z) < 1e-6:
            return terr, walls
        W, H = self.data.w, self.data.h
        wt, ww = self.world_layers()
        if abs(z - 1.0) < 1e-6:
            terr, walls = wt, ww
        else:
            size = (max(1, int(W * z)), max(1, int(H * z)))
            terr = pygame.transform.smoothscale(wt, size)
            walls = pygame.transform.smoothscale(ww, size)
        self.layers = (z, terr, walls)
        return terr, walls

    def minimap_base(self):
        if self._mm is not None:
            return self._mm
        wt, ww = self.world_layers()
        mm = pygame.transform.smoothscale(wt, (max(1, int(self.data.w * 0.115)),
                                               max(1, int(self.data.h * 0.115))))
        mm.blit(pygame.transform.smoothscale(ww, mm.get_size()), (0, 0))
        self._mm = mm
        return mm

    def timeline_strip(self, w):
        key = (w, self.data.n)
        if self.strip is not None and self.strip_key == key:
            return self.strip
        h = 26
        surf = pygame.Surface((w, h))
        surf.fill((20, 22, 26))
        n = self.data.n
        if n > 1:
            deaths = [0] * w
            marks = []                       # (x, kind) for click targeting
            for t in self.data.fruit_ticks:
                x = int(t / max(1, self.data.ticks - 1) * (w - 1))
                deaths[x] += 0
                marks.append((x, "fruit"))
            for e in self.data.births:
                marks.append((int(e["t"] / max(1, self.data.ticks - 1) * (w - 1)), "born"))
            for e in self.data.deaths:
                x = int(e["t"] / max(1, self.data.ticks - 1) * (w - 1))
                deaths[x] += 1
                marks.append((x, "die"))
            top = h - 10
            peak = max(1, max(deaths))
            for x, c in enumerate(deaths):
                if c:
                    bh = max(1, int(round((c / peak) ** 0.65 * top)))
                    col = (150, 46, 46) if c < peak else (216, 68, 68)
                    pygame.draw.line(surf, col, (x, top - bh), (x, top), 1)
            pygame.draw.line(surf, (52, 56, 64), (0, top), (w, top), 1)
            for x, kind in marks:
                col = {"fruit": (196, 168, 60), "born": (72, 168, 96), "die": (236, 84, 84)}[kind]
                surf.set_at((min(w - 1, max(0, x)), h - 8 if kind == "die" else h - 3), col)
            self._marks = marks
        self.strip, self.strip_key = surf, key
        return surf

    # ------------------------------------------------------------ drawing: map
    def draw_map(self, surf, view):
        d = self.data
        fr = self.frame()
        surf.fill((10, 11, 14))
        clip = surf.get_clip()
        surf.set_clip(view)
        z = self.cam.z
        if self.terrain and self.obstacles:
            terr, walls = self.terrain_surface(z)
            src = (int(self.cam.cx * z - view.w / 2.0), int(self.cam.cy * z - view.h / 2.0), view.w, view.h)
            sx = clamp(src[0], 0, max(0, terr.get_width() - view.w))
            sy = clamp(src[1], 0, max(0, terr.get_height() - view.h))
            surf.blit(terr, (view.left, view.top), (sx, sy, view.w, view.h))
        elif self.terrain:
            terr, _ = self.terrain_surface(z)
            src = (int(self.cam.cx * z - view.w / 2.0), int(self.cam.cy * z - view.h / 2.0), view.w, view.h)
            sx = clamp(src[0], 0, max(0, terr.get_width() - view.w))
            sy = clamp(src[1], 0, max(0, terr.get_height() - view.h))
            surf.blit(terr, (view.left, view.top), (sx, sy, view.w, view.h))
        else:
            pygame.draw.rect(surf, (22, 24, 20), view)
        if self.obstacles:
            _, walls = self.terrain_surface(z)
            src = (int(self.cam.cx * z - view.w / 2.0), int(self.cam.cy * z - view.h / 2.0), view.w, view.h)
            sx = clamp(src[0], 0, max(0, walls.get_width() - view.w))
            sy = clamp(src[1], 0, max(0, walls.get_height() - view.h))
            surf.blit(walls, (view.left, view.top), (sx, sy, view.w, view.h))

        def S(x, y):
            return self.cam.to_screen(x, y, view)

        rr = clamp(5.5 * (z / 0.55), 3.0, 11.0)
        # trees
        for (tx, ty) in d.trees:
            sx, sy = S(tx, ty)
            if view.collidepoint(sx, sy):
                pygame.draw.circle(surf, (26, 54, 32), (int(sx), int(sy)), int(max(2, rr * 0.9)))
        # Predator sense rings (simulator constants: predator.py vision 250, hearing 60). Drawn ONLY
        # for predators near the selected agent: drawing them for every predator floods the map with
        # large circles and hides the thing the viewer is for (where the agents are).
        sel0 = self.selected()
        if self.vision and sel0 is not None:
            for p in fr.get("preds", []):
                if math.hypot(p[0] - sel0[1], p[1] - sel0[2]) > 700:
                    continue
                sx, sy = S(p[0], p[1])
                r = int(PRED_VISION * z)
                if r > 6:
                    pygame.draw.circle(surf, (118, 44, 44), (int(sx), int(sy)), r, 1)
                    r2 = int(PRED_HEARING * z)
                    if r2 > 4:
                        pygame.draw.circle(surf, (150, 60, 60), (int(sx), int(sy)), r2, 1)
        # fruit
        if self.food:
            fr2 = clamp(3.0 * (z / 0.55), 2.0, 6.0)
            for f in fr.get("fruits", []):
                sx, sy = S(f[0], f[1])
                if view.collidepoint(sx, sy):
                    fruit_icon(surf, int(sx), int(sy), fr2)
        # trails
        if self.trails:
            for a in fr["agents"]:
                pts = d.history(a[0], self.i, 46)
                if len(pts) > 1:
                    sp = [S(p[1], p[2]) for p in pts]
                    pygame.draw.lines(surf, TRAIL_COL, False, [(int(x), int(y)) for x, y in sp], 1)
        # deaths
        self._death_marks = []
        if self.deaths:
            t_now = fr["t"]
            for e in d.deaths:
                if e["t"] > t_now:
                    continue
                sx, sy = S(e["x"], e["y"])
                if not (view.left - 20 <= sx <= view.right + 20 and view.top - 20 <= sy <= view.bottom + 20):
                    continue
                recent = (t_now - e["t"]) <= DEATH_RECENT_TICKS
                col = CAUSE_COLORS.get(e.get("cause"), (220, 220, 220))
                if not recent:
                    col = tuple(int(c * 0.55) for c in col)
                death_cross(surf, int(sx), int(sy), 4 if not recent else 6, col, 1 if not recent else 2)
                self._death_marks.append((int(sx), int(sy), e))
        # selected agent overlays under the fleet
        sel = self.selected()
        if sel is not None:
            sid, sxw, syw, se, sd, svr, sva = sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6]
            shr = sel[8] if len(sel) > 8 else DEFAULT_HEARING
            if self.vision:
                poly = d.vision_poly(sxw, syw, sd, svr, sva)
                sp = [S(px, py) for px, py in poly]
                if self.ovl is None or self.ovl.get_size() != (surf.get_width(), surf.get_height()):
                    self.ovl = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                self.ovl.fill((0, 0, 0, 0))
                pygame.draw.polygon(self.ovl, (110, 155, 255, 64), [(int(x), int(y)) for x, y in sp])
                pygame.draw.polygon(self.ovl, (150, 190, 255, 170), [(int(x), int(y)) for x, y in sp], 1)
                surf.blit(self.ovl, (0, 0))
            if self.hearing:
                sx, sy = S(sxw, syw)
                pygame.draw.circle(surf, (110, 110, 190), (int(sx), int(sy)), int(shr * z), 1)
        # predators
        pd = d.pred_dirs(self.i)
        for k, p in enumerate(fr.get("preds", [])):
            sx, sy = S(p[0], p[1])
            pred_icon(surf, sx, sy, clamp(8 * (z / 0.55), 5, 13), pd[k] if k < len(pd) else None)
        # agents
        for a in fr["agents"]:
            aid, x, y, e, dd = a[0], a[1], a[2], a[3], a[4]
            me = a[7] if len(a) > 7 else 500.0
            sx, sy = S(x, y)
            if not (view.left - 30 <= sx <= view.right + 30 and view.top - 30 <= sy <= view.bottom + 30):
                continue
            f = e / max(1.0, me)
            st = "crit" if f < ENERGY_CRIT_F else ("low" if f < ENERGY_LOW_F else "ok")
            col = energy_color(e, me)
            hp = clamp(3.8 * (z / 0.55), 3.5, 9.0)
            # dark halo first: agents have to pop against green terrain and dark walls
            pygame.draw.circle(surf, (12, 13, 16), (int(sx), int(sy)), int(hp) + 1)
            pygame.draw.line(surf, (250, 250, 250), (int(sx), int(sy)),
                             (int(sx + math.cos(dd) * hp * 3.4), int(sy + math.sin(dd) * hp * 3.4)), 2)
            player_icon(surf, sx, sy, hp, col, st)
            # energy bar (only when it is legible at this zoom, or for the selected agent)
            if z >= 0.5 or aid == self.sel:
                bw = 16
                pygame.draw.rect(surf, (18, 18, 20), (int(sx) - bw // 2, int(sy) - hp - 6, bw, 3))
                pygame.draw.rect(surf, col, (int(sx) - bw // 2, int(sy) - hp - 6,
                                             int(bw * clamp(f, 0, 1)), 3))
            if aid == self.sel:
                pygame.draw.circle(surf, (255, 255, 255), (int(sx), int(sy)), int(hp + 5), 2)
                pygame.draw.line(surf, (255, 255, 255), (int(sx), int(sy) - hp - 12),
                                 (int(sx), int(sy) - hp - 20), 2)
            if z >= 0.75 or aid == self.sel:
                lbl = self.font.render(10, f"a{aid}", (255, 255, 255) if aid == self.sel else (206, 212, 220))
                if lbl:
                    surf.blit(lbl, (int(sx) + hp + 3, int(sy) - hp - 4))
        # selected: lines to nearest predator / fruit (the "what is it facing" question)
        if sel is not None:
            x, y = sel[1], sel[2]
            sx, sy = S(x, y)
            best_p, bp = None, 1e9
            for p in fr.get("preds", []):
                dist = math.hypot(p[0] - x, p[1] - y)
                if dist < bp:
                    bp, best_p = dist, p
            if best_p is not None and bp < 420:
                ex, ey = S(best_p[0], best_p[1])
                self._dash(surf, sx, sy, ex, ey, (190, 70, 70), bp)
            best_f, bf = None, 1e9
            for f in fr.get("fruits", []):
                dist = math.hypot(f[0] - x, f[1] - y)
                if dist < bf:
                    bf, best_f = dist, f
            if best_f is not None and bf < 420:
                ex, ey = S(best_f[0], best_f[1])
                self._dash(surf, sx, sy, ex, ey, (150, 140, 70), bf)
        elif self.sel is not None:
            # the selected agent is not in this frame: it is dead (or not yet born). Mark where it
            # died, so clicking a death keeps a visible anchor on the map.
            evs = [e for e in (d.death_by_agent.get(self.sel) or []) if e["t"] <= fr["t"]]
            if evs:
                e0 = evs[-1]
                sx, sy = S(e0["x"], e0["y"])
                pygame.draw.circle(surf, (255, 255, 255), (int(sx), int(sy)), 11, 2)
                lbl = self.font.render(10, f"a{e0['a']} died ({e0.get('cause')})", (255, 235, 235))
                if lbl:
                    surf.blit(lbl, (int(sx) + 14, int(sy) - 6))
        surf.set_clip(clip)
        pygame.draw.rect(surf, C_EDGE, view, 1)
        # scale bar + zoom
        bar = 100.0
        px = int(bar * z)
        if px > 20:
            x0, y0 = view.left + 12, view.bottom - 14
            pygame.draw.line(surf, C_DIM, (x0, y0), (x0 + px, y0), 2)
            pygame.draw.line(surf, C_DIM, (x0, y0 - 4), (x0, y0 + 4), 2)
            pygame.draw.line(surf, C_DIM, (x0 + px, y0 - 4), (x0 + px, y0 + 4), 2)
            lbl = self.font.render(10, f"{int(bar)} u", C_DIM)
            if lbl:
                surf.blit(lbl, (x0 + px + 5, y0 - 7))
        zl = self.font.render(10, f"zoom {z:.2f}x", C_DIM2)
        if zl:
            surf.blit(zl, (view.right - zl.get_width() - 8, view.bottom - 20))

    def _dash(self, surf, x0, y0, x1, y1, col, dist):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 9))
        for k in range(0, n, 2):
            a = k / n
            b = min(1.0, (k + 1) / n)
            pygame.draw.line(surf, col, (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a)),
                             (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b)), 1)
        m = self.font.render(10, f"{dist:.0f}", col)
        if m:
            surf.blit(m, (int((x0 + x1) / 2) + 4, int((y0 + y1) / 2) - 12))

    # ------------------------------------------------------------ drawing: chrome
    def hline(self, surf, x, y, w, col=C_EDGE):
        pygame.draw.line(surf, col, (x, y), (x + w, y), 1)

    def label(self, surf, size, s, x, y, col=C_TXT, bold=False):
        t = self.font.render(size, s, col, bold)
        if t:
            surf.blit(t, (x, y))
        return (t.get_width() if t else 0)

    def btn(self, surf, rect, kind, active=False):
        col = C_PANEL2 if not active else (52, 70, 100)
        pygame.draw.rect(surf, col, rect, border_radius=4)
        pygame.draw.rect(surf, C_EDGE if not active else C_ACC, rect, 1, border_radius=4)
        cx, cy = rect.centerx, rect.centery
        s = 7
        white = (232, 236, 242)
        if kind == "play":
            pygame.draw.polygon(surf, white, [(cx - 5, cy - s), (cx - 5, cy + s), (cx + 7, cy)])
        elif kind == "pause":
            pygame.draw.rect(surf, white, (cx - 6, cy - s, 4, 2 * s))
            pygame.draw.rect(surf, white, (cx + 2, cy - s, 4, 2 * s))
        elif kind == "prev":
            pygame.draw.polygon(surf, white, [(cx + 4, cy - s), (cx + 4, cy + s), (cx - 5, cy)])
        elif kind == "next":
            pygame.draw.polygon(surf, white, [(cx - 4, cy - s), (cx - 4, cy + s), (cx + 5, cy)])
        elif kind == "prevev":
            pygame.draw.polygon(surf, white, [(cx + 3, cy - s), (cx + 3, cy + s), (cx - 3, cy)])
            pygame.draw.rect(surf, white, (cx - 6, cy - s, 2, 2 * s))
        elif kind == "nextev":
            pygame.draw.polygon(surf, white, [(cx - 3, cy - s), (cx - 3, cy + s), (cx + 3, cy)])
            pygame.draw.rect(surf, white, (cx + 4, cy - s, 2, 2 * s))

    def draw_top(self, surf, L):
        bar = L["topbar"]
        pygame.draw.rect(surf, C_PANEL, bar)
        self.hline(surf, 0, bar.bottom - 1, bar.w)
        d = self.data
        self.label(surf, 15, "Survival Simulator - Replay Viewer", 14, bar.top + 6, C_TXT, True)
        # keep the meta line SHORT: the playback buttons start at x=372 on the same visual band
        self.label(surf, 10, f"Seed {d.seed}  ·  {d.n} frames  ·  every {d.every} ticks  ·  "
                             f"{len(d.deaths)} deaths", 14, bar.top + 27, C_DIM)

        # playback buttons
        bx, by = 372, bar.top + 8
        bw, bh = 30, 24
        gap = 6
        keys = [("prevev", "prev_ev"), ("prev", "prev"), ("play", "play"), ("next", "next"), ("nextev", "next_ev")]
        self._btns = {}
        for kind, name in keys:
            r = pygame.Rect(bx, by, bw, bh)
            self._btns[name] = r
            self.btn(surf, r, kind, active=(name == "play" and self.playing))
            bx += bw + gap
        # speed slider (log scale) - offset far enough right that the "Speed: Nx" label cannot
        # collide with the last playback button
        score_x = (bar.right - 14) - 260          # left edge of the score block
        slw = int(clamp(score_x - 24 - (bx + 92), 84, 176))
        sl = pygame.Rect(bx + 92, bar.top + 20, slw, 8)
        self._speed_rect = sl
        self.label(surf, 10, "0.1x", sl.left - 2, sl.bottom + 4, C_DIM2)
        self.label(surf, 10, "100x", sl.right - 22, sl.bottom + 4, C_DIM2)
        pygame.draw.rect(surf, (18, 20, 24), sl, border_radius=4)
        frac = (math.log10(clamp(self.speed, 0.1, 100.0)) + 1.0) / 3.0
        pygame.draw.rect(surf, C_ACC, (sl.left, sl.top, int(sl.w * frac), sl.h), border_radius=4)
        pygame.draw.circle(surf, (245, 248, 252), (sl.left + int(sl.w * frac), sl.centery), 7)
        # "Speed: 5x" sits LEFT of the slider, vertically centred (it used to run off the bar top)
        stxt = f"Speed: {self.speed:g}x"
        self.label(surf, 12, stxt, sl.left - 12 - self.font.tw(12, stxt, True), bar.top + 14, C_TXT, True)

        # right: score block
        fs = fmt_int(d.final_score)
        cur = fmt_int(self.frame()["score"])
        xr = bar.right - 14
        w1 = self.font.tw(15, "FINAL")
        self.label(surf, 10, "CURRENT SCORE", xr - 260, bar.top + 6, C_DIM)
        self.label(surf, 26, cur, xr - 260, bar.top + 18, C_TXT, True)
        self.label(surf, 10, "FINAL", xr - 118, bar.top + 6, C_DIM)
        reached = self.i >= d.n - 1
        fcol = C_TXT if reached else C_DIM2
        self.label(surf, 18, fs, xr - 118, bar.top + 20, fcol, True)
        tail = "(reached)" if reached else "(end of recording)"
        self.label(surf, 9, tail, xr - 118, bar.top + 40, C_DIM2)
        self.label(surf, 10, f"{self.tick() - d.frames[0]['t']:+d} / tick", xr - 260, bar.top + 46,
                   C_GOOD if (d.delta_per_tick[self.i] if d.delta_per_tick else 0) >= 0 else C_BAD)
        # frame / tick / % -- above the progress slider
        pct = 100.0 * self.i / max(1, d.n - 1)
        self._ptext = (f"Frame {self.i} / {d.n - 1}     Tick {self.tick()} / {d.ticks - 1}     {pct:.0f}%")
        # progress slider (full width, at the bottom edge of the top bar)
        pr = pygame.Rect(0, bar.bottom - 9, bar.w, 9)
        self._prog_rect = pr
        pygame.draw.rect(surf, (20, 22, 27), pr)
        pygame.draw.rect(surf, (58, 84, 126), (pr.left, pr.top, int(pr.w * self.i / max(1, d.n - 1)), pr.h))
        pygame.draw.rect(surf, C_ACC, (pr.left + int(pr.w * self.i / max(1, d.n - 1)) - 2, pr.top, 4, pr.h))

    def draw_left(self, surf, L):
        col = L["left"]
        pygame.draw.rect(surf, C_PANEL, col)
        pygame.draw.line(surf, C_EDGE, (col.right - 1, col.top), (col.right - 1, col.bottom - 1), 1)
        clip = surf.get_clip()
        surf.set_clip(col)          # the left column must NEVER spill onto the map
        x = col.left + 10
        y = col.top + 8
        self.label(surf, 11, "MINIMAP", x, y, C_DIM, True)
        y += 16
        mmw = col.w - 20
        mmh = int(mmw * self.data.h / self.data.w)
        base = self.minimap_base()
        mm = pygame.transform.smoothscale(base, (mmw, mmh))
        surf.blit(mm, (x, y))
        fr = self.frame()
        sx = mmw / self.data.w
        sy = mmh / self.data.h
        for p in fr.get("preds", []):
            pygame.draw.circle(surf, PRED_COL, (x + int(p[0] * sx), y + int(p[1] * sy)), 3)
        for a in fr["agents"]:
            c = energy_color(a[3], a[7] if len(a) > 7 else 500.0)
            pygame.draw.circle(surf, c, (x + int(a[1] * sx), y + int(a[2] * sy)), 2)
        if self.deaths:
            t_now = fr["t"]
            for e in self.data.deaths:
                if e["t"] > t_now:
                    continue
                surf.set_at((clamp(x + int(e["x"] * sx), x, x + mmw - 1),
                             clamp(y + int(e["y"] * sy), y, y + mmh - 1)),
                            CAUSE_COLORS.get(e.get("cause"), (200, 200, 200)))
        vp = self.cam.viewport(L["map"])
        vr = pygame.Rect(x + int(vp.left * sx), y + int(vp.top * sy),
                         max(3, int(vp.w * sx)), max(3, int(vp.h * sy)))
        pygame.draw.rect(surf, (255, 255, 255), vr, 1)
        pygame.draw.rect(surf, C_EDGE, (x, y, mmw, mmh), 1)
        self._mm_rect = pygame.Rect(x, y, mmw, mmh)
        y += mmh + 10

        # ---- LEGEND: also the ROW HEIGHT adapts, so a short window loses rows instead of spilling
        small = col.w < 196
        fsz = 9 if small else 10
        rh = 12 if small else 14
        self.label(surf, 10 if small else 11, "LEGEND", x, y, C_DIM, True)
        y += rh + 2
        rows = [("agent", ENERGY_HEALTHY, "Our agent"),
                ("agent", (255, 255, 255), "Selected agent"),
                ("low", ENERGY_LOW, "Low energy <45%"),
                ("crit", ENERGY_CRIT, "Critical <20% (lockout)"),
                ("pred", PRED_COL, "Predator"),
                ("fruit", FRUIT_COL, "Fruit"),
                ("wall", WALL_CORE, "Wall / obstacle"),
                ("vis", (140, 180, 255), "Vision cone"),
                ("hear", (110, 110, 190), "Hearing radius"),
                ("trail", TRAIL_COL, "Trail"),
                ("death", (232, 66, 66), "Death (colour = cause)"),
                ("predring", (150, 60, 60), "Predator sense (near sel.)")]
        for kind, c, txt in rows:
            if y + rh > col.bottom - 4:
                break
            cy = y + rh // 2
            ir = 4 if small else 5
            if kind == "agent":
                player_icon(surf, x + 6, cy, ir, c)
            elif kind in ("low", "crit"):
                player_icon(surf, x + 6, cy, ir, c, kind)
            elif kind == "pred":
                pred_icon(surf, x + 6, cy, ir, 0.0)
            elif kind == "fruit":
                fruit_icon(surf, x + 6, cy, ir - 1)
            elif kind == "wall":
                pygame.draw.line(surf, WALL_CASE, (x + 1, cy), (x + 11, cy), 7)
                pygame.draw.line(surf, WALL_CORE, (x + 1, cy), (x + 11, cy), 2)
            elif kind == "vis":
                pygame.draw.polygon(surf, (110, 155, 255), [(x + 6, cy - 5), (x + 1, cy + 5), (x + 11, cy + 5)])
            elif kind in ("hear", "predring"):
                pygame.draw.circle(surf, c, (x + 6, cy), ir, 1)
            elif kind == "trail":
                pygame.draw.line(surf, c, (x + 1, cy + 2), (x + 11, cy - 2), 1)
            else:
                death_cross(surf, x + 6, cy, ir - 1, c, 2)
            self.label(surf, fsz, txt, x + 17, y, C_DIM)
            y += rh

        y += 6
        self.label(surf, 10 if small else 11, "OVERLAYS", x, y, C_DIM, True)
        y += rh + 1
        self._toggles = []
        for name, key in (("Trails", "trails"), ("Vision", "vision"), ("Hearing", "hearing"),
                          ("Deaths", "deaths"), ("Food", "food"), ("Obstacles", "obstacles"),
                          ("Terrain", "terrain")):
            if y + rh > col.bottom - 4:
                break
            on = getattr(self, key)
            r = pygame.Rect(x, y + 1, 11, 11)
            pygame.draw.rect(surf, (36, 40, 48), r, border_radius=3)
            if on:
                pygame.draw.rect(surf, C_ACC, r, border_radius=3)
                pygame.draw.lines(surf, (12, 16, 24), False,
                                  [(r.x + 3, r.centery), (r.x + 5, r.centery + 3), (r.x + 9, r.centery - 3)], 2)
            pygame.draw.rect(surf, C_EDGE, r, 1, border_radius=3)
            self.label(surf, fsz, name, x + 17, y, C_TXT if on else C_DIM2)
            self._toggles.append((pygame.Rect(x, y - 2, col.w - 20, rh + 1), key))
            y += rh

        # ---- CONTROLS: discoverable, adaptive (1 or 2 columns, or truncated) - never overflow
        y += 6
        rows = [("Space", "Play / Pause"), ("<- ->", "Step frame"), ("1-5", "Speed presets"),
                ("Up/Dn", "Speed up / down"), ("Mouse", "Select agent"), ("Drag", "Pan map"),
                ("Wheel", "Zoom"), ("T", "Trails"), ("V", "Vision"), ("H", "Hearing"),
                ("D", "Deaths"), ("N", "Next event"), ("B", "Back 500"), ("R", "Restart"),
                ("F", "Fit view"), ("Q", "Quit")]
        compact = [("Space", "Play/Pause"), ("<- ->", "Step"), ("1-5", "Speed"), ("Up/Dn", "Speed x2"),
                   ("Mouse", "Select"), ("Drag", "Pan"), ("Wheel", "Zoom"), ("T", "Trails"),
                   ("V", "Vision"), ("H", "Hearing"), ("D", "Deaths"), ("N", "Next evt"),
                   ("B", "Back 500"), ("R", "Restart"), ("F", "Fit"), ("Q", "Quit")]
        avail = col.bottom - y - 4
        if avail >= 24:
            self.label(surf, 10 if small else 11, "CONTROLS", x, y, C_DIM, True)
            y += rh + 1
            avail = col.bottom - y - 4
            rh2 = 13 if not small else 12
            use, cols = rows, 1
            if len(rows) * rh2 > avail:
                if math.ceil(len(rows) / 2.0) * rh2 <= avail:
                    cols, use = 2, compact
                else:
                    rh2 = max(9, avail // max(1, len(compact)))
                    use = compact[: max(1, avail // rh2)]
            per = max(1, int(math.ceil(len(use) / cols)))
            cw = (col.w - 20) // cols
            ksz = 9 if (cols == 2 or small) else 10
            for i, (k, v) in enumerate(use):
                cx = x + (i // per) * cw
                cy = y + (i % per) * rh2
                if cy + rh2 > col.bottom:
                    break
                self.label(surf, ksz, k, cx, cy, C_ACC)
                self.label(surf, ksz, v, cx + (36 if ksz == 9 else 44), cy, C_DIM2)
        surf.set_clip(clip)

    def draw_right(self, surf, L):
        pcol = L["right"]
        pygame.draw.rect(surf, C_PANEL, pcol)
        pygame.draw.line(surf, C_EDGE, (pcol.left, pcol.top), (pcol.left, pcol.bottom - 1), 1)
        clip = surf.get_clip()
        surf.set_clip(pcol)         # the panel scrolls; it must never spill onto the map or the bar
        self.rscroll = int(clamp(self.rscroll, 0, max(0, getattr(self, "content_h", 0) - pcol.h + 20)))
        x = pcol.left + 12
        w = pcol.w - 24
        y = pcol.top + 8 - self.rscroll
        sel = self.selected()
        d = self.data
        fr = self.frame()

        def finish():
            """Remember the content height (so the wheel can scroll next frame) and unclip."""
            self.content_h = (y + self.rscroll) - pcol.top + 8
            surf.set_clip(clip)
            if self.content_h > pcol.h - 12:
                pygame.draw.rect(surf, C_PANEL2, (pcol.right - 5, pcol.top, 5, pcol.h))
                frac = clamp(pcol.h / float(self.content_h), 0.08, 1.0)
                sh = int(pcol.h * frac)
                sy = pcol.top + int((pcol.h - sh) * (self.rscroll / float(max(1, self.content_h - pcol.h + 20))))
                pygame.draw.rect(surf, (86, 96, 112), (pcol.right - 4, sy, 3, sh), border_radius=2)

        def head(txt, yy=None):
            nonlocal y
            if yy is not None:
                y = yy
            self.hline(surf, x, y, w, C_EDGE)
            self.label(surf, 11, txt, x, y + 5, C_DIM, True)
            y += 22

        # A selected agent that is DEAD (or simply not present in this frame, because the frame is
        # recorded after its death) must still show its context: build the status from the recorded
        # trace + the recorded death event instead of silently falling back to the fleet summary.
        dead_sel = d.death_by_agent.get(self.sel) if self.sel is not None else None
        if sel is None and dead_sel is not None:
            at = d.at(self.sel, self.i)
            if at is not None:
                e0 = dead_sel[0]
                sel = [self.sel, at[0], at[1], e0.get("e", 0.0), 0.0, 0.0, 0.0,
                       e0.get("max_e", 500.0)]

        if sel is None:
            self.label(surf, 11, "SELECTED AGENT", x, y, C_DIM, True)
            y += 24
            self.label(surf, 11, "click an agent on the map, a death cross,", x, y, C_DIM2)
            y += 15
            self.label(surf, 11, "or a death marker in the timeline.", x, y, C_DIM2)
            y += 26
            # fleet summary while nothing is selected
            head("FLEET NOW")
            n = fr["n"]
            self.row(surf, x, y, w, "Alive", f"{n}"); y += 15
            self.row(surf, x, y, w, "Mean energy", f"{fr['e_mean']:.0f}"); y += 15
            self.row(surf, x, y, w, "Min energy", f"{fr['e_min']:.0f}",
                     C_BAD if fr["e_min"] < 100 else C_TXT); y += 15
            self.row(surf, x, y, w, "In lockout (<20%)", str(fr["lock"]),
                     C_BAD if fr["lock"] else C_TXT); y += 15
            self.row(surf, x, y, w, "Predators", str(len(fr.get("preds", []))), C_BAD); y += 15
            self.row(surf, x, y, w, "Fruit on map", str(len(fr.get("fruits", [])))); y += 15
            self.row(surf, x, y, w, "Fruit eaten (derived)", str(sum(1 for t in d.fruit_ticks if t <= fr["t"])))
            y += 20
            head("DEATHS SO FAR")
            for cause in ("eaten", "starved", "aged"):
                k = sum(1 for e in d.deaths if e.get("cause") == cause and e["t"] <= fr["t"])
                pygame.draw.circle(surf, CAUSE_COLORS[cause], (x + 5, y + 6), 5)
                self.row(surf, x + 16, y, w - 16, cause, str(k), CAUSE_COLORS[cause])
                y += 15
            y += 8
            # recent events list (compact, clickable)
            head("RECENT EVENTS")
            evs = [e for e in d.events if e["t"] <= fr["t"]][-9:]
            for e in reversed(evs):
                if e["k"] == "die":
                    c = CAUSE_COLORS.get(e.get("cause"), C_TXT)
                    self.label(surf, 10, f"{e['t']:>6}  died a{e['a']}  {e.get('cause','?'):<7}"
                                         f" e={e.get('e',0):.0f}", x, y, c)
                else:
                    self.label(surf, 10, f"{e['t']:>6}  born a{e['a']}", x, y, C_GOOD)
                y += 13
            finish()
            return

        aid = sel[0]
        xw, yw, e, dd, vr, va = sel[1], sel[2], sel[3], sel[4], sel[5], sel[6]
        me = sel[7] if len(sel) > 7 else 500.0
        hr = sel[8] if len(sel) > 8 else DEFAULT_HEARING
        dead = d.death_by_agent.get(aid)
        is_dead = bool(dead and dead[0]["t"] <= fr["t"])
        alive_now = any(a[0] == aid for a in fr["agents"])
        e_dead_tick = dead[0]["t"] if (dead and is_dead) else fr["t"]
        # the frame to read "what was around it" from: for a dead agent the LAST frame it was alive
        # (the frame recorded at its death tick no longer contains it)
        ref_i = self.i if alive_now else max(0, d.i_for_tick(e_dead_tick) - 1)
        ref_fr = d.frames[ref_i]

        # ---- STATUS
        self.label(surf, 13, f"Agent #{aid}", x, y, C_TXT, True)
        stx = x + w - 82
        st = "ALIVE" if alive_now else ("DEAD" if is_dead else "not in frame")
        stcol = C_GOOD if alive_now else C_BAD
        pygame.draw.rect(surf, tuple(int(c * 0.28) for c in stcol), (stx, y - 2, 78, 18), border_radius=4)
        self.label(surf, 11, st, stx + 8, y + 1, stcol, True)
        y += 22
        # energy bar with the 20% lockout threshold marked
        frac = clamp(e / max(1.0, me), 0, 1)
        pygame.draw.rect(surf, (16, 18, 22), (x, y, w, 14), border_radius=3)
        pygame.draw.rect(surf, energy_color(e, me), (x, y, int(w * frac), 14), border_radius=3)
        tx = x + int(w * ENERGY_CRIT_F)
        pygame.draw.line(surf, (255, 255, 255), (tx, y - 3), (tx, y + 17), 2)
        y += 18
        self.label(surf, 10, "0", x, y, C_DIM2)
        self.label(surf, 10, f"{me:.0f} max", x + w - 46, y, C_DIM2)
        self.label(surf, 10, f"energy {e:.0f} / {me:.0f}  ({100*frac:.0f}%)", x + 14, y, C_TXT)
        y += 13
        lock = e < ENERGY_CRIT_F * me
        self.label(surf, 10, f"^ sprint lockout at {ENERGY_CRIT_F*me:.0f}",
                   tx - 62 if tx > x + 70 else x + 14, y,
                   C_BAD if lock else C_DIM2)
        if lock:
            # right-aligned so it can never collide with the threshold caption
            self.label(surf, 10, "SPRINT LOCKED", x + w - self.font.tw(10, "SPRINT LOCKED"),
                       y, C_BAD, True)
        y += 16
        spd = d.derived_speed(aid, self.i)
        if alive_now:
            walk = 10.0
            moving = "SPRINT" if (spd is not None and spd > walk * 1.3) else \
                     ("WALKING" if (spd is not None and spd > 0.3) else "RESTING")
        else:
            moving, spd = "-", None
        self.row(surf, x, y, w, "Position", f"{xw:.0f}, {yw:.0f}"); y += 15
        self.row(surf, x, y, w, "Heading", f"{d.heading_deg(dd):.0f}deg"); y += 15
        self.row(surf, x, y, w, "Speed (derived)", "-" if spd is None else f"{spd:.1f} u/tick"); y += 15
        self.row(surf, x, y, w, "Movement (inferred)", moving); y += 15
        self.row(surf, x, y, w, "Sprint", ("-" if not alive_now else
                                           ("LOCKED" if lock else "AVAILABLE")),
                 C_BAD if (lock and alive_now) else (C_GOOD if alive_now else C_DIM2)); y += 15
        # every "ago"/"for" figure is relative to the agent's OWN last tick: for a dead agent the
        # current playback tick would otherwise report values like "last food 4002 ticks ago" for an
        # agent that died at tick 221.
        ref_t = e_dead_tick if is_dead else fr["t"]
        age = d.last_seen.get(aid, ref_t) - d.first_seen.get(aid, ref_t)
        self.row(surf, x, y, w, "Alive for (derived)", f"{age} ticks"
                 + ("  (dead)" if is_dead else "")); y += 15
        lf = d.lock_frames.get(aid, 0)
        seen = len(d.agent_frames.get(aid, []))
        self.row(surf, x, y, w, "Lockout frames", f"{lf} / {seen} ({100*lf/max(1,seen):.0f}%)",
                 C_BAD if lf else C_TXT); y += 15
        lastfood = d.last_fruit_tick.get(aid)
        self.row(surf, x, y, w, "Last food (derived)",
                 "never" if lastfood is None else f"{ref_t - lastfood} ticks ago",
                 C_DIM if lastfood is None else C_TXT)
        y += 18

        # ---- PERCEPTION (derived: recorded positions + the agent's recorded vision geometry)
        head("PERCEPTION")
        p = d.perception(aid, self.i)
        p_note = ""
        if p is None and not alive_now:
            # the death investigation question: "could it see what killed it?" - so fall back to the
            # LAST frame it was alive instead of showing nothing
            p = d.perception(aid, ref_i)
            if p is not None:
                p_note = f"last frame alive (tick {ref_fr['t']})"
        if p is None:
            self.label(surf, 10, "not alive in this frame (agent is dead)", x, y, C_DIM2)
            y += 16
        else:
            if p_note:
                self.label(surf, 9, p_note, x, y, C_ORANGE)
                y += 13
            # polar mini-diagram: agent at bottom centre facing UP
            dw, dh = w, 104
            dr = pygame.Rect(x, y, dw, dh)
            pygame.draw.rect(surf, (17, 19, 23), dr, border_radius=4)
            pygame.draw.rect(surf, C_EDGE, dr, 1, border_radius=4)
            scale = (dh * 0.86) / max(1.0, p["vision_range"])
            ax, ay = dr.centerx, dr.bottom - 8
            # hearing disc + cone (rotate so the agent faces up)
            hrr = int(p["hearing"] * scale)
            if hrr > 2:
                pygame.draw.circle(surf, (86, 86, 150), (ax, ay), hrr, 1)
            vr2 = int(p["vision_range"] * scale)
            a0 = -math.pi / 2 - p["vision_angle"] / 2.0
            a1 = -math.pi / 2 + p["vision_angle"] / 2.0
            pygame.draw.polygon(surf, (34, 46, 70),
                                [(ax, ay), (ax + int(vr2 * math.cos(a0)), ay + int(vr2 * math.sin(a0))),
                                 (ax + int(vr2 * math.cos(a1)), ay + int(vr2 * math.sin(a1)))])
            for key, col in (("agents", (200, 210, 225)), ("preds", PRED_COL), ("fruits", FRUIT_COL)):
                for (ex, ey, dist, ang, sensed) in p[key]:
                    rel = ang - dd
                    px = ax + int(dist * scale * math.cos(rel - math.pi / 2))
                    py = ay + int(dist * scale * math.sin(rel - math.pi / 2))
                    py = clamp(py, dr.top + 3, dr.bottom - 3)
                    if key == "fruits":
                        fruit_icon(surf, px, py, 3)
                    elif key == "preds":
                        pred_icon(surf, px, py, 4)
                    else:
                        player_icon(surf, px, py, 3, col)
            player_icon(surf, ax, ay, 5, energy_color(e, me), "crit" if e < ENERGY_CRIT_F * me else "ok")
            self.label(surf, 9, "agent faces UP - wedge = vision, disc = hearing",
                       x + 5, dr.bottom - 14, C_DIM2)
            y += dh + 6
            self.row(surf, x, y, w, "Vision range / angle", f"{p['vision_range']:.0f} u / {math.degrees(p['vision_angle']):.0f}deg")
            y += 15
            self.row(surf, x, y, w, "Hearing radius",
                     f"{p['hearing']:.0f} u" + ("" if p["hearing_recorded"] else "  (default)")); y += 15
            self.row(surf, x, y, w, "Visible agents", str(len(p["agents"])))
            y += 15
            self.row(surf, x, y, w, "Visible predators", str(len(p["preds"])),
                     C_BAD if p["preds"] else C_DIM2); y += 15
            self.row(surf, x, y, w, "Visible fruit", str(len(p["fruits"])), C_WARN if p["fruits"] else C_DIM2)
            y += 15
            near = [t for t in p["preds"] if t[2] <= 220]
            self.row(surf, x, y, w, "Predator in range <220", str(len(near)),
                     C_BAD if near else C_DIM2); y += 15
            self.row(surf, x, y, w, "Wall segments in cone", str(p["walls"]))
            y += 18

        # ---- BEHAVIOUR
        head("BEHAVIOUR / DIAGNOSTICS")
        dv = dead[0] if dead else None
        cnt = (dv or {}).get("counters", {}) or {}
        for k, lab, col in (("wall_block", "Wall blocked", C_TXT), ("turned_from_fruit", "Turns away from food", C_WARN),
                            ("oscillate", "Oscillation", C_WARN), ("approach_pred", "Predator-approach ticks", C_TXT)):
            v = cnt.get(k)
            self.row(surf, x, y, w, lab, "-" if v is None else str(v) + ("" if dv else ""), col)
            y += 15
        self.row(surf, x, y, w, "Lockout ticks (recorded)", "-" if not cnt else f"{cnt.get('lockout_ticks', 0)} / {cnt.get('ticks', 0)}",
                 C_BAD if cnt.get("lockout_ticks") else C_TXT)
        y += 15
        # "what was around it" is read at ref_fr, so a dead agent reports the neighbourhood it died in
        # rather than the one the playback cursor happens to be standing in
        atx, aty = d.at(aid, ref_i)[:2] if d.at(aid, ref_i) else (xw, yw)
        nearp, nf = self.nearest(atx, aty, ref_fr)
        suffix = "" if alive_now else "  (at death)"
        self.row(surf, x, y, w, "Nearest predator" + suffix,
                 "-" if nearp is None else f"{nearp:.0f} u",
                 C_BAD if (nearp is not None and nearp < 200) else C_TXT); y += 15
        self.row(surf, x, y, w, "Nearest food" + suffix, "-" if nf is None else f"{nf:.0f} u",
                 C_WARN if nf else C_DIM2)
        y += 15
        if not cnt:
            self.label(surf, 9, "live agent: counters are recorded at death, shown on death.",
                       x, y, C_DIM2)
            y += 13
        y += 6

        # ---- RECENT ACTIONS
        head("RECENT ACTIONS")
        if dv and dv.get("recent"):
            rows = dv["recent"][-6:]
            for r in rows:
                self.label(surf, 10, f"d={r[0]:.1f}  turn {r[1]:+.2f}  b{r[2]:.2f}  "
                                     f"{'SPRINT' if r[3] else 'walk'}", x, y,
                           C_ORANGE if r[3] else C_DIM)
                y += 12
            self.label(surf, 9, "recorded command: dist, turn, blend, sprint", x, y, C_DIM2)
            y += 14
        else:
            pts = d.history(aid, self.i, 7)
            prev = None
            for (fi, px, py) in pts:
                if prev is not None:
                    self.label(surf, 10, f"f{fi:<5d} moved {px - prev[0]:+6.1f} {py - prev[1]:+6.1f}", x, y, C_DIM)
                    y += 12
                prev = (px, py)
            self.label(surf, 9, "derived from recorded positions (commanded actions are only "
                                "recorded at death)", x, y, C_DIM2)
            y += 22
        y += 4

        # ---- DEATH
        if is_dead:
            e0 = dead[0]
            head("DEATH  ·  Agent #%d" % aid)
            cause = e0.get("cause", "?")
            pygame.draw.rect(surf, tuple(int(c * 0.28) for c in CAUSE_COLORS.get(cause, C_TXT)),
                             (x, y - 2, w, 20), border_radius=4)
            self.label(surf, 13, f"cause: {cause}", x + 6, y, CAUSE_COLORS.get(cause, C_TXT), True)
            y += 24
            self.row(surf, x, y, w, "Tick", str(e0["t"])); y += 15
            self.row(surf, x, y, w, "Energy before death", f"{e0.get('e',0):.0f} / {e0.get('max_e',0):.0f}",
                     C_BAD if e0.get("lockout") else C_TXT); y += 15
            self.row(surf, x, y, w, "Sim age at death (recorded)", f"{e0.get('age',0):.0f} ticks"); y += 15
            self.row(surf, x, y, w, "Lockout at death", "YES" if e0.get("lockout") else "no",
                     C_BAD if e0.get("lockout") else C_TXT); y += 15
            pd = e0.get("pred_dist")
            # None means the agent's LAST observation contained no predator at all - NOT "unknown"
            self.row(surf, x, y, w, "Predator seen at death (rec.)",
                     "none in view" if pd is None else f"{pd:.1f} u",
                     C_BAD if pd is not None else C_DIM2); y += 15
            if pd is not None:
                vis = None
                di = d.i_for_tick(e0["t"])
                if di < d.n:
                    vis = d.pred_visible(aid, di)
                self.row(surf, x, y, w, "Predator visible at death", "NO" if vis is False else ("YES" if vis else "-"),
                         C_BAD if vis is False else C_GOOD)
                y += 15
            self.row(surf, x, y, w, "Fruit seen at death (recorded)",
                     str(e0.get("fruit_vis", "-"))); y += 15
            lf = d.last_fruit_tick.get(aid)
            self.row(surf, x, y, w, "Time since last food",
                     "never" if lf is None else f"{e0['t'] - lf} ticks",
                     C_WARN if lf is None else C_TXT)
            y += 18
        finish()

    def row(self, surf, x, y, w, k, v, vcol=C_TXT):
        self.label(surf, 10, k, x, y, C_DIM)
        t = self.font.render(10, str(v), vcol)
        if t:
            surf.blit(t, (x + w - t.get_width(), y))

    def nearest(self, x, y, fr):
        p = min((math.hypot(q[0] - x, q[1] - y) for q in fr.get("preds", [])), default=None)
        f = min((math.hypot(q[0] - x, q[1] - y) for q in fr.get("fruits", [])), default=None)
        return p, f

    def draw_bottom(self, surf, L):
        bar = L["bottom"]
        pygame.draw.rect(surf, C_PANEL, bar)
        self.hline(surf, 0, bar.top, bar.w)
        clip = surf.get_clip()
        surf.set_clip(bar)
        d = self.data
        fr = self.frame()
        # left: current score
        x = 14
        y = bar.top + 8
        self.label(surf, 10, "CURRENT SCORE", x, y, C_DIM, True)
        self.label(surf, 24, fmt_int(fr["score"]), x, y + 14, C_TXT, True)
        dpt = d.delta_per_tick[self.i] if d.delta_per_tick else 0.0
        self.label(surf, 11, f"{dpt:+.2f} / tick", x + 6, y + 42, C_GOOD if dpt >= 0 else C_BAD)
        # right: final score
        rx = bar.right - 186
        reached = self.i >= d.n - 1
        self.label(surf, 10, "FINAL SCORE", rx, y, C_DIM, True)
        self.label(surf, 20, fmt_int(d.final_score), rx, y + 16, C_TXT if reached else C_DIM, True)
        pygame.draw.rect(surf, (18, 20, 24), (rx, y + 42, 168, 7), border_radius=3)
        pf = clamp(fr["score"] / max(1e-6, d.final_score), 0, 1)
        pygame.draw.rect(surf, C_ACC, (rx, y + 42, int(168 * pf), 7), border_radius=3)
        self.label(surf, 9, ("reached end of recording" if reached else
                             f"{100*pf:.0f}% of final score  ·  {'exact' if reached else 'recorded end'}"),
                   rx, y + 52, C_DIM2)
        # centre: event / death timeline
        cx0 = 206
        cw = rx - cx0 - 24
        if cw > 120:
            self.label(surf, 10, "EVENTS / DEATH TIMELINE", cx0, bar.top + 6, C_DIM, True)
            self.label(surf, 9, "click to seek  ·  click a death marker to open that death",
                       cx0 + 180, bar.top + 7, C_DIM2)
            strip = self.timeline_strip(cw)
            sr = pygame.Rect(cx0, bar.top + 22, cw, strip.get_height())
            surf.blit(strip, (cx0, bar.top + 22))
            pygame.draw.rect(surf, C_EDGE, sr, 1)
            self._tl_rect = sr
            px = cx0 + int(cw * self.i / max(1, d.n - 1))
            pygame.draw.line(surf, (255, 255, 255), (px, sr.top - 4), (px, sr.bottom + 4), 2)
            # counts legend
            cts = d.causes
            lx = cx0
            for cause in ("eaten", "starved", "aged"):
                pygame.draw.circle(surf, CAUSE_COLORS[cause], (lx + 4, sr.bottom + 9), 4)
                txt = f"{cause} {cts.get(cause, 0)}"
                lx += 12 + self.label(surf, 10, txt, lx + 8, sr.bottom + 3, C_DIM)
                lx += 6
            lx += 14
            pygame.draw.circle(surf, FRUIT_COL, (lx + 4, sr.bottom + 9), 4)
            lx += 10 + self.label(surf, 10, f"fruit {len(d.fruit_ticks)}", lx + 8, sr.bottom + 3, C_DIM)
            pygame.draw.circle(surf, C_GOOD, (lx + 14, sr.bottom + 9), 4)
            self.label(surf, 10, f"births {len(d.births)}", lx + 20, sr.bottom + 3, C_DIM)
            self.label(surf, 10, self._ptext, bar.right - 20 - self.font.tw(10, self._ptext),
                       sr.bottom + 3, C_TXT)
        else:
            self._tl_rect = None
        surf.set_clip(clip)

    def draw(self, surf):
        L = self.layout()
        self.view = L["map"]
        surf.fill(C_BG)
        if self.cam.z <= 0 or self._fit_key != (L["map"].size, self.data.w):
            self.cam.fit(L["map"])
            self._fit_key = (L["map"].size, self.data.w)
            if self.base_zoom:            # --scale: explicit initial zoom, applied once
                self.cam.z = clamp(self.base_zoom, 0.12, 1.6)
                self.cam.clamp(L["map"])
                self.base_zoom = None
        self.draw_map(surf, L["map"])
        self.draw_left(surf, L)
        self.draw_right(surf, L)
        self.draw_top(surf, L)
        self.draw_bottom(surf, L)

    def layout(self):
        W, H = self.win[0], self.win[1]
        top = 62 if H >= 760 else 56
        bot = min(106, max(88, H // 10))
        left = 208 if W >= 1460 else (188 if W >= 1300 else 168)
        right = 344 if W >= 1460 else (306 if W >= 1300 else 276)
        mapr = pygame.Rect(left, top, W - left - right, H - top - bot)
        return {"topbar": pygame.Rect(0, 0, W, top), "map": mapr,
                "left": pygame.Rect(0, top, left, H - top - bot),
                "right": pygame.Rect(W - right, top, right, H - top - bot),
                "bottom": pygame.Rect(0, H - bot, W, bot)}

    # ------------------------------------------------------------ interaction
    def toast(self, msg):
        self.msg, self.msg_until = msg, pygame.time.get_ticks() + 2200

    def jump_event(self, direction=1):
        t = self.tick()
        if direction > 0:
            nxt = [e["t"] for e in self.data.events if e["t"] > t]
            if nxt:
                self.set_i(self.data.i_for_tick(min(nxt)))
        else:
            prv = [e["t"] for e in self.data.events if e["t"] < t]
            if prv:
                self.set_i(self.data.i_for_tick(max(prv)))
        self.playing = False

    def open_death(self, e):
        self.sel = e["a"]
        self.set_i(self.data.i_for_tick(e["t"]))
        self.playing = False
        self.focus = ("death", e["a"])
        self.toast(f"death of agent #{e['a']} at tick {e['t']}: {e.get('cause')}")

    def seek_from_x(self, x, rect):
        f = clamp((x - rect.left) / max(1, rect.w), 0.0, 1.0)
        self.playing = False
        self.set_i(round(f * (self.data.n - 1)))

    def pick_agent(self, mx, my):
        fr = self.frame()
        best, bd = None, 1e9
        for a in fr["agents"]:
            sx, sy = self.cam.to_screen(a[1], a[2], self.view)
            dd = (sx - mx) ** 2 + (sy - my) ** 2
            if dd < bd:
                bd, best = dd, a[0]
        return best if bd < 18 ** 2 else None

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.VIDEORESIZE:
            self.win = (ev.w, ev.h)
            self._fit_key = None
            return True
        if ev.type == pygame.MOUSEWHEEL:
            # MOUSEWHEEL carries no position in pygame 2, so use the live cursor (tests may inject pos)
            mx, my = getattr(ev, "pos", None) or pygame.mouse.get_pos()
            if self.layout()["right"].collidepoint(mx, my):
                self.rscroll = max(0, self.rscroll - ev.y * 14)
            else:
                self.cam.zoom_at(1.15 ** ev.y, mx, my, self.view)
            return True
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if k in (pygame.K_SPACE, pygame.K_k):
                self.playing = not self.playing
            elif k == pygame.K_RIGHT:
                self.playing = False
                self.set_i(self.i + 1)
            elif k == pygame.K_LEFT:
                self.playing = False
                self.set_i(self.i - 1)
            elif k == pygame.K_UP:
                self.speed = clamp(self.speed * 2, 0.1, 100.0)
            elif k == pygame.K_DOWN:
                self.speed = clamp(self.speed / 2, 0.1, 100.0)
            elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                self.speed = float([1, 2, 5, 20, 100][k - pygame.K_1])
            elif k == pygame.K_t:
                self.trails = not self.trails
            elif k == pygame.K_v:
                self.vision = not self.vision
            elif k == pygame.K_h:
                self.hearing = not self.hearing
            elif k == pygame.K_d:
                self.deaths = not self.deaths
            elif k == pygame.K_r:
                self.set_i(0)
                self.playing = False
            elif k == pygame.K_b:
                self.set_i(max(0, self.i - 500))
                self.playing = False
            elif k == pygame.K_n:
                self.jump_event(1)
            elif k == pygame.K_p:
                self.jump_event(-1)
            elif k == pygame.K_f:
                self._fit_key = None
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            L = self.layout()
            if getattr(self, "_prog_rect", None) and self._prog_rect.collidepoint(ev.pos):
                self.drag = ("prog",)
                self.seek_from_x(ev.pos[0], self._prog_rect)
                return True
            if getattr(self, "_speed_rect", None) and self._speed_rect.inflate(14, 16).collidepoint(ev.pos):
                self.drag = ("speed",)
                self.set_speed_from_x(ev.pos[0])
                return True
            if getattr(self, "_tl_rect", None) and self._tl_rect.collidepoint(ev.pos):
                evt = self.event_at_x(ev.pos[0])
                if evt is not None:
                    if evt["k"] == "die":
                        self.open_death(evt)
                    else:
                        self.set_i(self.data.i_for_tick(evt["t"]))
                        self.playing = False
                else:
                    self.seek_from_x(ev.pos[0], self._tl_rect)
                return True
            for r, name in getattr(self, "_toggles", []):
                if r.collidepoint(ev.pos):
                    setattr(self, name, not getattr(self, name))
                    return True
            for name, r in getattr(self, "_btns", {}).items():
                if r.collidepoint(ev.pos):
                    if name == "play":
                        self.playing = not self.playing
                    elif name == "prev":
                        self.playing = False; self.set_i(self.i - 1)
                    elif name == "next":
                        self.playing = False; self.set_i(self.i + 1)
                    elif name == "prev_ev":
                        self.jump_event(-1)
                    elif name == "next_ev":
                        self.jump_event(1)
                    return True
            if getattr(self, "_mm_rect", None) and self._mm_rect.collidepoint(ev.pos):
                self.drag = ("mm",)
                self.mm_jump(ev.pos)
                return True
            if L["map"].collidepoint(ev.pos):
                # death cross under the cursor?
                for (sx, sy, e) in self._death_marks:
                    if (sx - ev.pos[0]) ** 2 + (sy - ev.pos[1]) ** 2 < 12 ** 2:
                        self.open_death(e)
                        return True
                self.drag = ("map", ev.pos, self.cam.cx, self.cam.cy, False)
                return True
        if ev.type == pygame.MOUSEMOTION:
            if self.drag is None:
                return True
            if self.drag[0] == "speed":
                self.set_speed_from_x(ev.pos[0])
            elif self.drag[0] == "prog":
                self.seek_from_x(ev.pos[0], self._prog_rect)
            elif self.drag[0] == "mm":
                self.mm_jump(ev.pos)
            elif self.drag[0] == "map":
                _, o, cx0, cy0, moved = self.drag
                dx, dy = ev.pos[0] - o[0], ev.pos[1] - o[1]
                self.cam.cx = cx0 - dx / self.cam.z
                self.cam.cy = cy0 - dy / self.cam.z
                self.cam.clamp(self.view)
                self.drag = ("map", o, cx0, cy0, moved or abs(dx) + abs(dy) > 3)
            return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.drag and self.drag[0] == "map":
                _, o, cx0, cy0, moved = self.drag
                if not moved:
                    hit = self.pick_agent(*ev.pos)
                    self.sel = hit
                    if hit is None:
                        self.focus = None
            self.drag = None
            return True
        return True

    def set_speed_from_x(self, x):
        f = clamp((x - self._speed_rect.left) / max(1, self._speed_rect.w), 0.0, 1.0)
        self.speed = clamp(10 ** (-1 + 3 * f), 0.1, 100.0)

    def mm_jump(self, pos):
        r = self._mm_rect
        wx = clamp((pos[0] - r.left) / max(1, r.w), 0, 1) * self.data.w
        wy = clamp((pos[1] - r.top) / max(1, r.h), 0, 1) * self.data.h
        self.cam.cx, self.cam.cy = wx, wy
        self.cam.clamp(self.view)

    def event_at_x(self, x):
        """Nearest recorded event marker within a few pixels of a timeline click."""
        r = self._tl_rect
        if r is None:
            return None
        tick = (x - r.left) / max(1, r.w) * (self.data.ticks - 1)
        tol_ticks = max(40, self.data.ticks * 12 / max(1, r.w))
        best, bd = None, 1e18
        for e in self.data.events:
            dd = abs(e["t"] - tick)
            if dd < bd:
                bd, best = dd, e
        return best if (best is not None and bd <= tol_ticks) else None

    # ------------------------------------------------------------ loop
    def _init_window(self):
        pygame.init()
        info = pygame.display.Info()
        sw, sh = max(1024, info.current_w or 1440), max(700, info.current_h or 900)
        W = int(clamp(sw * 0.94, 1180, 1860))
        H = int(clamp(sh * 0.92, 720, 1160))
        self.win = (W, H)
        return pygame.display.set_mode((W, H), pygame.RESIZABLE)

    def run(self):
        try:
            surf = self._init_window()
        except Exception as exc:
            # Fail SOFT, with the cause spelled out: a display problem must never look like a broken
            # recording. The most common cause by far is SDL_VIDEODRIVER being set to "" or to a driver
            # this machine does not have.
            print(f"[viewer] cannot create a window: {exc}")
            print("[viewer] SDL_VIDEODRIVER is currently "
                  f"{os.environ.get('SDL_VIDEODRIVER', '<unset>')!r}")
            print("[viewer] falling back to text playback (same data, no window). "
                  "For the window, ensure SDL_VIDEODRIVER is UNSET (not empty) in your shell.")
            self.text_playback()
            return
        d = self.data
        pygame.display.set_caption(f"Survival Simulator - Replay Viewer - seed {d.seed} "
                                  f"({d.n} frames, {d.ticks} ticks, final {d.final_score:.1f})")
        self._fit_key = None
        self.win = surf.get_size()
        clock = pygame.time.Clock()
        self.sel = self._auto_pick()
        running = True
        while running:
            for ev in pygame.event.get():
                if not self.handle_event(ev):
                    running = False
            if self.playing:
                self.i_f = min(float(d.n - 1), self.i_f + self.speed)
                self.i = int(self.i_f)
                if self.i >= d.n - 1:
                    self.playing = False
            self.draw(surf)
            if self.msg_until > pygame.time.get_ticks():
                t = self.font.render(12, self.msg, (250, 250, 250), True)
                if t:
                    r = pygame.Rect(self.view.centerx - t.get_width() // 2 - 10, self.view.top + 10,
                                    t.get_width() + 20, 24)
                    pygame.draw.rect(surf, (40, 46, 60), r, border_radius=6)
                    pygame.draw.rect(surf, C_ACC, r, 1, border_radius=6)
                    surf.blit(t, (r.x + 10, r.y + 5))
            pygame.display.flip()
            if self.paused_for_dump and self.dumped < self.dump_n:
                stride = max(1, d.n // max(1, self.dump_n))
                if self.dumped:                      # advance BETWEEN dumps, otherwise every save is t=0
                    self.set_i(min(d.n - 1, self.i + stride))
                    # REDRAW after the jump: without this the PNG named t=<new tick> still contained the
                    # PREVIOUS frame's pixels (the surface was drawn before the advance), so dumped
                    # frames and their filenames disagreed - exactly what a vision model would be fed.
                    self.draw(surf)
                os.makedirs(self.dump_dir or "/tmp/rp/frames", exist_ok=True)
                bmp = os.path.join(self.dump_dir or "/tmp/rp/frames",
                                   f"frame_{self.dumped:04d}_t{d.frames[self.i]['t']}.bmp")
                pygame.image.save(surf, bmp)
                png = bmp[:-4] + ".png"
                if not os.path.exists(png):
                    os.system(f'sips -s format png "{bmp}" --out "{png}" >/dev/null 2>&1 || '
                              f'convert "{bmp}" "{png}" >/dev/null 2>&1 || true')
                self.dumped += 1
                if self.dumped >= self.dump_n:
                    pygame.image.save(surf, os.path.join(self.dump_dir or "/tmp/rp/frames",
                                                         "final_view.bmp"))
                    pygame.quit()
                    print(f"dumped {self.dumped} frames to {self.dump_dir}")
                    return
            clock.tick(60)
        pygame.quit()

    def _auto_pick(self):
        """Start with the agent that is currently worst off - the one worth explaining."""
        fr = self.data.frames[0]
        if not fr["agents"]:
            return None
        return min(fr["agents"], key=lambda a: a[3] / max(1.0, a[7] if len(a) > 7 else 500.0))[0]

    # ------------------------------------------------------------ text mode (unchanged contract)
    def text_panel_lines(self, fr):
        d = self.data
        tally = {}
        for e in d.deaths:
            if e["t"] <= fr["t"]:
                tally[e.get("cause")] = tally.get(e.get("cause"), 0) + 1
        out = [f"CURRENT SCORE {fr['score']:.1f}   FINAL {d.final_score:.1f}   "
               f"{'PLAY' if self.playing else 'PAUSE'} {self.speed:g}x",
               f"frame {self.i}/{d.n-1}  tick {fr['t']}/{d.ticks-1}  ({100*self.i/max(1,d.n-1):.0f}%)",
               f"agents={fr['n']}  preds={len(fr.get('preds', []))}  fruit={len(fr.get('fruits', []))}  "
               f"e_mean={fr['e_mean']}  e_min={fr['e_min']}  in <20% lockout: {fr['lock']}",
               f"deaths so far: {tally or '{}'}   walls drawn: {len(d.edges)}",
               f"fruit eaten (derived): {sum(1 for t in d.fruit_ticks if t <= fr['t'])}"]
        for e in [e for e in d.events if e["t"] <= fr["t"]][-6:]:
            if e["k"] == "die":
                c = e.get("counters", {})
                out.append(f" t{e['t']} DEATH a{e['a']} {e['cause']} e={e['e']}/{e['max_e']:.0f}"
                           f"{' LOCKOUT' if e.get('lockout') else ''} pd={e.get('pred_dist')}"
                           f" wall_block={c.get('wall_block')} osc={c.get('oscillate')}")
            else:
                out.append(f" t{e['t']} born a{e['a']}")
        if self.sel is not None:
            hit = [a for a in fr["agents"] if a[0] == self.sel]
            out.append(f"SELECTED AGENT {self.sel}")
            if hit:
                a = hit[0]
                p = d.perception(self.sel, self.i)
                out.append(f" pos ({a[1]:.0f},{a[2]:.0f}) energy {a[3]:.1f}/{a[7]:.0f} "
                           f"facing {d.heading_deg(a[4]):.0f}deg speed {d.derived_speed(self.sel, self.i) or 0:.1f}")
                if p:
                    out.append(f" sees: agents {len(p['agents'])} preds {len(p['preds'])} "
                               f"fruit {len(p['fruits'])} | vision {p['vision_range']:.0f}/"
                               f"{math.degrees(p['vision_angle']):.0f}deg hearing {p['hearing']:.0f}"
                               f"{'' if p['hearing_recorded'] else ' (default)'}")
            for ev in [e for e in d.deaths if e.get("a") == self.sel]:
                c = ev.get("counters", {})
                out += [f" death t={ev['t']} cause={ev['cause']} energy={ev['e']}/{ev['max_e']:.0f}"
                        f" lockout={int(bool(ev['lockout']))} pred_dist={ev['pred_dist']}"
                        f" fruit_visible={ev['fruit_vis']}",
                        f"  wall_blocked {c.get('wall_block', 0)} turned_from_fruit "
                        f"{c.get('turned_from_fruit', 0)} oscillated {c.get('oscillate', 0)} "
                        f"lockout_ticks {c.get('lockout_ticks', 0)}/{c.get('ticks', 0)}",
                        f"  recent actions {ev.get('recent', [])[-4:]}"]
        return out

    # backwards-compatible alias (the old HUD had this name)
    def panel_lines(self, fr):
        return self.text_panel_lines(fr)

    def ascii_frame(self, cols=96, rows=34):
        """ASCII view of the CURRENT recorded frame. Rendered from the recording alone.

        WHY THIS IS NOT replay.ascii_frame: this file is the presentation layer, and importing
        replay.py pulls in best_controller + src.core (shapely, torch, ...) - i.e. the whole
        simulator - just to draw characters. That made `--text` fail in a venv with only pygame
        installed (ModuleNotFoundError: No module named 'shapely') even though text playback
        reads nothing but the replay. The rendering below is identical to replay.ascii_frame.
        """
        fr, w, h = self.frame(), self.data.w, self.data.h
        grid = [[" "] * cols for _ in range(rows)]
        sx, sy = cols / w, rows / h

        def put(x, y, ch):
            c, r = int(x * sx), int(y * sy)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = ch

        for e in fr.get("edges", []):
            (x0, y0), (x1, y1) = e
            n = max(2, int(math.hypot((x1 - x0) * sx, (y1 - y0) * sy)))
            for k in range(n + 1):
                put(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, "#")
        for t in fr.get("trees", []):
            put(t[0], t[1], "T")
        for f in fr.get("fruits", []):
            put(f[0], f[1], ".")
        for p in fr.get("preds", []):
            put(p[0], p[1], "P")
        for a in fr.get("agents", []):
            ch = "@" if a[3] > 0.2 * (a[7] or 500.0) else ("o" if a[3] > 100 else "!")
            put(a[1], a[2], ch)
        lines = ["+" + "-" * cols + "+"]
        lines += ["|" + "".join(r) + "|" for r in grid]
        lines += ["+" + "-" * cols + "+"]
        lines.append(f"t={fr['t']} score={fr['score']} agents={fr['n']} "
                     f"e_mean={fr['e_mean']} e_min={fr['e_min']} in_lockout={fr['lock']} "
                     f"fruit_on_map={len(fr.get('fruits', []))} predators={len(fr.get('preds', []))}")
        return "\n".join(lines)

    def text_playback(self, step=250):
        last = -10 ** 9
        for i, fr in enumerate(self.data.frames):
            if fr["t"] - last < step and i != 0:
                continue
            last = fr["t"]
            self.i = i
            print(self.ascii_frame())
            for ln in self.text_panel_lines(fr):
                print("   " + ln)
            print()


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--scale", type=float, default=None,
                    help="initial map zoom in px per world unit; default = fit the world to the map pane")
    ap.add_argument("--dump-frames", type=int, default=0, dest="dump_frames")
    ap.add_argument("--outdir", default="/tmp/rp/frames")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--text", action="store_true",
                    help="no window: print ASCII frames + the HUD to stdout (works over ssh, no font needed)")
    ap.add_argument("--step", type=int, default=250, help="text mode: print a frame every N ticks")
    ap.add_argument("--select", type=int, default=None, help="agent id to select at start")
    args = ap.parse_args()

    if not args.text and not PYGAME_OK:
        # A window needs pygame. Rather than making the user remember which interpreter has it
        # (this has cost several round-trips: system python3 has none, .venv has pygame without
        # SDL_ttf, ~/.venv-viewer has pygame-ce with real fonts), re-run ourselves on the first
        # sibling interpreter that can actually open a window. VIEWER_NO_REEXEC=1 disables this.
        if not os.environ.get("VIEWER_NO_REEXEC"):
            me = os.path.abspath(__file__)
            marked = dict(os.environ, VIEWER_NO_REEXEC="1")
            for cand in (os.path.expanduser("~/.venv-viewer/bin/python"),
                         os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(me))),
                                      ".venv", "bin", "python")):
                if not os.path.exists(cand):
                    continue
                try:
                    # compare ENVIRONMENTS, not interpreter binaries: ~/.venv-viewer and the Xcode
                    # python3 share one binary, so samefile() would wrongly call them the same.
                    pre = subprocess.run([cand, "-c", "import sys;print(sys.prefix)"],
                                         capture_output=True, text=True)
                    if pre.returncode != 0 or pre.stdout.strip() == sys.prefix:
                        continue
                except OSError:
                    continue
                try:
                    probe = subprocess.run([cand, "-c", "import pygame"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:
                    continue
                if probe.returncode == 0:
                    print(f"[viewer] no pygame in {sys.executable} -> re-running with {cand}", flush=True)
                    return subprocess.call([cand, me] + sys.argv[1:], env=marked)
        print(f"[viewer] no pygame in this interpreter ({PYGAME_ERR}) - a window needs it.", file=sys.stderr)
        print("[viewer] interpreters that have pygame in this repo:", file=sys.stderr)
        print(f"[viewer]   {os.path.expanduser('~/.venv-viewer/bin/python')}   (pygame-ce, real fonts)", file=sys.stderr)
        print("[viewer]   .venv/bin/python                      (pygame without SDL_ttf, PIL fonts)", file=sys.stderr)
        print("[viewer] or read the recording with no window at all:  <python> "
              f"{os.path.basename(__file__)} <replay> --text", file=sys.stderr)
        return 2

    rec = load_replay(args.path)
    p = Player(rec, scale=args.scale, dump_dir=args.outdir, dump_n=args.dump_frames,
               text_mode=args.text)
    d = p.data
    p.set_i(min(args.start, d.n - 1))
    if args.select is not None:
        p.sel = args.select
    print(f"replay: seed {d.seed}, {d.n} frames (every {d.every} ticks, {d.ticks} ticks total), "
          f"{len(d.events)} events, {len(d.edges)} wall segments, final score {d.final_score:.2f}",
          flush=True)
    if args.text:
        p.text_playback(step=args.step)
        return
    print("controls: space play/pause | <-/-> step | 1-5 speed presets | up/down speed x2 | click select "
          "| drag pan | wheel zoom | t trails v vision h hearing d deaths n next-event b back r restart "
          "f fit q quit")
    p.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
