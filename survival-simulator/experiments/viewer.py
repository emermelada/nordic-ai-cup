#!/usr/bin/env python3
"""viewer.py - interactive player for a recorded episode (Phase 1: human visualization).

Reads a replay produced by replay.py. Needs no simulator, so what you watch is the recorded evidence,
not a re-simulation that might differ. Read-only: this cannot change or deploy anything.

    ./viewer.py /tmp/rp/seed1500_9523t.json.gz
    ./viewer.py <replay> --dump-frames 40 --outdir /tmp/rp/frames   # PNGs, for a vision model later

CONTROLS
    space / k    play-pause          left / right   step 1 frame (pauses first)
    up / down    speed x2 / /2       1..5           speeds 1x 2x 5x 20x 100x
    click agent  select + inspect    t              trails
    v            vision cone         h              hearing radius
    d            death markers       n              jump to next event
    b            jump back 500       r              restart
    q / esc      quit

WHAT TO LOOK AT (this is the point of the exercise - aggregate statistics hid the mechanism)
    * colour of each agent = energy: green healthy, yellow, red = inside the <20% sprint-lockout zone.
      Agents that cannot sprint cannot outrun a predator; 92-96% of predation deaths measured there.
    * the selected agent's panel shows its recent commanded actions and the derived counters
      (wall-blocked, turned-from-visible-fruit, oscillation, lockout ticks), so a visual impression
      can be checked against a number in the same screen.
    * 'd' leaves a cross where each agent died, coloured by cause, so you can see WHERE the fleet dies.
"""
import argparse
import gzip
import json
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", ""))

import pygame  # noqa: E402

BIOME_COLORS = {"Grassland_biome": (54, 92, 48), "Forest_biome": (34, 74, 40),
                "Desert_biome": (150, 132, 84), "Snow_biome": (196, 200, 205),
                "Water_biome": (40, 70, 120), "Mountain_biome": (104, 100, 96)}
DEFAULT_BIOME = (70, 70, 74)
CAUSE_COLORS = {"eaten": (230, 60, 60), "starved": (230, 200, 60), "aged": (170, 170, 170)}


def energy_color(e, max_e):
    f = e / max(1.0, max_e)
    if f < 0.20:                     # inside the sprint-lockout zone
        return (220, 40, 40)
    if f < 0.45:
        return (230, 180, 40)
    return (70, 190, 90)


class Player:
    def __init__(self, rec, scale=0.72, dump_dir=None, dump_n=0, text_mode=False):
        self.rec = rec
        self.scale = scale
        self.W = int(rec.get("w", 1600) * scale)
        self.H = int(rec.get("h", 1200) * scale)
        self.panel = 300
        self.frames = rec["frames"]
        self.events = rec["events"]
        self.text_mode = text_mode
        # pygame.font may be absent in a minimal build (SDL_ttf missing). The HUD must not be the
        # reason the viewer cannot run, so fall back to printing the panel to stdout instead.
        self.font = None
        if not text_mode:
            try:
                import pygame.font  # noqa: F401
                self.font = pygame.font.SysFont("Menlo", 13)
            except Exception as exc:      # pragma: no cover - environment dependent
                print(f"[viewer] no pygame.font ({exc}); panel text goes to stdout instead")
        self.by_tick = {}
        for e in self.events:
            self.by_tick.setdefault(e["t"], []).append(e)
        self.i = 0
        self.playing = False
        self.speed = 1
        self.trails = False
        self.vision = False
        self.hearing = False
        self.deaths = True
        self.sel = None
        self.paused_for_dump = dump_n > 0
        self.dump_dir, self.dump_n, self.dumped = dump_dir, dump_n, 0

    # ---------------------------------------------------------------- drawing
    def draw(self, surf):
        rec = self.rec
        fr = self.frames[self.i]
        surf.fill((18, 18, 22))
        # biome background (coarse blocks)
        biome = rec.get("biome")
        if biome:
            legend = rec.get("biome_legend", [])
            rows, cols = len(biome), len(biome[0])
            bw = rec.get("w", 1600) / cols * self.scale
            bh = rec.get("h", 1200) / rows * self.scale
            for r in range(rows):
                for c in range(cols):
                    name = legend[biome[r][c]] if biome[r][c] < len(legend) else ""
                    col = BIOME_COLORS.get(name, DEFAULT_BIOME)
                    pygame.draw.rect(surf, col, (c * bw, r * bh, bw + 1, bh + 1))

        def P(x, y):
            return (int(x * self.scale), int(y * self.scale))

        # walls
        for (x0, y0), (x1, y1) in rec.get("edges", []):
            pygame.draw.line(surf, (235, 235, 235), P(x0, y0), P(x1, y1), 3)
        # trees
        for tr in fr.get("trees", []):
            pygame.draw.circle(surf, (26, 60, 34), P(tr[0], tr[1]), 7)
        # fruit
        for f in fr.get("fruits", []):
            pygame.draw.circle(surf, (240, 230, 110), P(f[0], f[1]), 3)
        # trails
        if self.trails and self.i > 8:
            for aid in {a[0] for a in fr["agents"]}:
                pts = []
                for j in range(max(0, self.i - 40), self.i + 1):
                    hit = [a for a in self.frames[j]["agents"] if a[0] == aid]
                    if hit:
                        pts.append(P(hit[0][1], hit[0][2]))
                if len(pts) > 1:
                    pygame.draw.lines(surf, (90, 120, 190), False, pts, 1)
        # deaths
        if self.deaths:
            for e in self.events:
                if e["k"] == "die" and e["t"] <= fr["t"]:
                    pygame.draw.line(surf, CAUSE_COLORS.get(e["cause"], (255, 255, 255)),
                                     P(e["x"] - 12, e["y"] - 12), P(e["x"] + 12, e["y"] + 12), 3)
                    pygame.draw.line(surf, CAUSE_COLORS.get(e["cause"], (255, 255, 255)),
                                     P(e["x"] - 12, e["y"] + 12), P(e["x"] + 12, e["y"] - 12), 3)
        # predators
        for pr in fr.get("preds", []):
            x, y = P(pr[0], pr[1])
            pygame.draw.circle(surf, (255, 60, 60, 60), (x, y), 34, 1)
            pygame.draw.polygon(surf, (235, 40, 40), [(x, y - 9), (x - 8, y + 7), (x + 8, y + 7)])
        # agents
        for a in fr["agents"]:
            aid, x, y, e, d, vr, va, me = a
            if self.vision and (self.sel is None or self.sel == aid):
                pts = [P(x, y)]
                for k in range(7):
                    ang = d - va / 2 + va * k / 6
                    pts.append(P(x + math.cos(ang) * vr, y + math.sin(ang) * vr))
                s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
                pygame.draw.polygon(s, (120, 160, 255, 40), pts)
                surf.blit(s, (0, 0))
            if self.hearing and (self.sel is None or self.sel == aid):
                pygame.draw.circle(surf, (90, 90, 140), P(x, y), int(60 * self.scale), 1)
            col = energy_color(e, me)
            pygame.draw.circle(surf, col, P(x, y), 7)
            if e < 0.2 * me:
                pygame.draw.circle(surf, (255, 255, 255), P(x, y), 9, 1)
            pygame.draw.line(surf, (250, 250, 250), P(x, y),
                             P(x + math.cos(d) * 22, y + math.sin(d) * 22), 2)
            if aid == self.sel:
                pygame.draw.circle(surf, (255, 255, 255), P(x, y), 12, 1)
            # energy bar
            bw = 18
            pygame.draw.rect(surf, (30, 30, 30), (P(x, y)[0] - bw // 2, P(x, y)[1] - 14, bw, 3))
            pygame.draw.rect(surf, col, (P(x, y)[0] - bw // 2, P(x, y)[1] - 14, int(bw * min(1, e / max(1, me))), 3))
        self.draw_panel(surf, fr)

    def panel_lines(self, fr):
        """The HUD as text: drawn with a font when available, printed otherwise."""
        tally = {}
        for e in self.events:
            if e["k"] == "die" and e["t"] <= fr["t"]:
                tally[e["cause"]] = tally.get(e["cause"], 0) + 1
        out = [f"t={fr['t']}  score={fr['score']}  {'PLAY' if self.playing else 'PAUSE'}  {self.speed}x",
               f"agents={fr['n']}  preds={len(fr.get('preds', []))}  fruit={len(fr.get('fruits', []))}",
               f"energy mean={fr['e_mean']} min={fr['e_min']}", f"in <20% lockout: {fr['lock']}",
               f"deaths: {tally or '{}'}",
               f"walls drawn: {len(self.rec.get('edges', []))}"]
        for e in [e for e in self.events if e["t"] <= fr["t"]][-7:]:
            if e["k"] == "die":
                out.append(f" t{e['t']} DEATH a{e['a']} {e['cause']} e={e['e']}/{e['max_e']:.0f}"
                           f"{' LOCKOUT' if e.get('lockout') else ''} pd={e.get('pred_dist')}")
            else:
                out.append(f" t{e['t']} {e['k']} a{e.get('a')}")
        if self.sel is not None:
            hit = [a for a in fr["agents"] if a[0] == self.sel]
            out.append(f"SELECTED agent {self.sel}")
            if hit:
                aid, x, y2, e, d, vr, va, me = hit[0]
                out.append(f" pos ({x:.0f},{y2:.0f}) energy {e:.1f}/{me:.0f} ({100*e/me:.0f}%)"
                           f" facing {d:.2f}rad vision {vr:.0f}/{va:.2f}rad")
            for ev in [e for e in self.events if e.get("a") == self.sel and e["k"] == "die"]:
                c = ev.get("counters", {})
                out += [" death context:",
                        f"  wall_blocked {c.get('wall_block', 0)}  turned_from_fruit {c.get('turned_from_fruit', 0)}",
                        f"  oscillated {c.get('oscillate', 0)}  lockout {c.get('lockout_ticks', 0)}/{c.get('ticks', 0)}",
                        f"  fruit visible at death {ev.get('fruit_vis')}  pred_dist {ev.get('pred_dist')}",
                        f"  recent actions {ev.get('recent', [])[-4:]}"]
        return out

    def draw_panel(self, surf, fr):
        x0 = self.W
        pygame.draw.rect(surf, (24, 24, 28), (x0, 0, self.panel, self.H))
        lines = self.panel_lines(fr)
        if self.font is None:
            # no font available: draw compact colour bars so the visual still carries the key signals
            bar_h = max(2, self.H // max(1, len(lines) * 3))
            for i, txt in enumerate(lines[: len(lines)]):
                col = (220, 220, 220)
                if "DEATH" in txt:
                    col = (240, 90, 90) if "LOCKOUT" in txt else (220, 180, 90)
                elif "lockout" in txt:
                    col = (240, 120, 120)
                pygame.draw.rect(surf, col, (x0 + 8, 8 + i * bar_h, min(self.panel - 16, 6 + 2 * len(txt)), 3))
            return
        y = 8
        for txt in lines:
            col = (220, 220, 220)
            if txt.startswith("SELECTED") or txt.startswith(" death"):
                col = (255, 255, 255)
            elif "DEATH" in txt:
                col = CAUSE_COLORS.get(txt.split()[3] if len(txt.split()) > 3 else "", (255, 200, 200))
            elif "lockout" in txt:
                col = (240, 120, 120)
            surf.blit(self.font.render(txt, True, col), (x0 + 10, y))
            y += 16

    # ---------------------------------------------------------------- loop
    def run(self):
        pygame.init()
        surf = pygame.display.set_mode((self.W + self.panel, self.H))
        pygame.display.set_caption(f"replay seed {self.rec['seed']} - {len(self.frames)} frames")
        clock = pygame.time.Clock()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return
                if ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); return
                    if k in (pygame.K_SPACE, pygame.K_k):
                        self.playing = not self.playing
                    elif k == pygame.K_RIGHT:
                        self.playing = False; self.i = min(len(self.frames) - 1, self.i + 1)
                    elif k == pygame.K_LEFT:
                        self.playing = False; self.i = max(0, self.i - 1)
                    elif k == pygame.K_UP:
                        self.speed = min(100, self.speed * 2)
                    elif k == pygame.K_DOWN:
                        self.speed = max(1, self.speed // 2)
                    elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        self.speed = [1, 2, 5, 20, 100][k - pygame.K_1]
                    elif k == pygame.K_t:
                        self.trails = not self.trails
                    elif k == pygame.K_v:
                        self.vision = not self.vision
                    elif k == pygame.K_h:
                        self.hearing = not self.hearing
                    elif k == pygame.K_d:
                        self.deaths = not self.deaths
                    elif k == pygame.K_r:
                        self.i = 0
                    elif k == pygame.K_b:
                        self.i = max(0, self.i - 500)
                    elif k == pygame.K_n:
                        t = self.frames[self.i]["t"]
                        nxt = [e["t"] for e in self.events if e["t"] > t]
                        if nxt:
                            tgt = min(nxt)
                            self.i = min(range(len(self.frames)),
                                         key=lambda j: abs(self.frames[j]["t"] - tgt))
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[0] < self.W:
                    mx, my = ev.pos[0] / self.scale, ev.pos[1] / self.scale
                    fr = self.frames[self.i]
                    if fr["agents"]:
                        self.sel = min(fr["agents"], key=lambda a: (a[1] - mx) ** 2 + (a[2] - my) ** 2)[0]
            if self.playing:
                self.i = min(len(self.frames) - 1, self.i + self.speed)
                if self.i >= len(self.frames) - 1:
                    self.playing = False
            self.draw(surf)
            pygame.display.flip()
            if self.paused_for_dump and self.dumped < self.dump_n:
                stride = max(1, len(self.frames) // max(1, self.dump_n))
                if self.dumped:                      # advance BETWEEN dumps, otherwise every save is t=0
                    self.i = min(len(self.frames) - 1, self.i + stride)
                os.makedirs(self.dump_dir or "/tmp/rp/frames", exist_ok=True)
                # NOTE: this pygame build cannot write PNG ("extended format is not available"), so
                # frames are written as BMP and converted with sips on macOS (or ImageMagick elsewhere).
                bmp = os.path.join(self.dump_dir or "/tmp/rp/frames",
                                   f"frame_{self.dumped:04d}_t{self.frames[self.i]['t']}.bmp")
                pygame.image.save(surf, bmp)
                png = bmp[:-4] + ".png"
                if not os.path.exists(png):
                    os.system(f'sips -s format png "{bmp}" --out "{png}" >/dev/null 2>&1 || '
                              f'convert "{bmp}" "{png}" >/dev/null 2>&1 || true')
                self.dumped += 1
                if self.dumped >= self.dump_n:
                    pygame.quit(); print(f"dumped {self.dumped} frames to {self.dump_dir}"); return
            clock.tick(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--scale", type=float, default=0.72)
    ap.add_argument("--dump-frames", type=int, default=0, dest="dump_frames")
    ap.add_argument("--outdir", default="/tmp/rp/frames")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--text", action="store_true",
                    help="no window: print ASCII frames + the HUD to stdout (works over ssh, no font needed)")
    ap.add_argument("--step", type=int, default=250, help="text mode: print a frame every N ticks")
    args = ap.parse_args()

    rec = json.load(gzip.open(args.path, "rt")) if args.path.endswith(".gz") else json.load(open(args.path))
    p = Player(rec, scale=args.scale, dump_dir=args.outdir, dump_n=args.dump_frames,
               text_mode=args.text)
    p.i = min(args.start, len(p.frames) - 1)
    print(f"replay: seed {rec['seed']}, {len(p.frames)} frames (every {rec.get('every')} ticks), "
          f"{len(p.events)} events, {len(rec.get('edges', []))} wall segments", flush=True)
    if args.text:
        import replay as _rp
        last = -10 ** 9
        for i, fr in enumerate(p.frames):
            if fr["t"] - last < args.step and i != 0:
                continue
            last = fr["t"]
            p.i = i
            print(_rp.ascii_frame(fr, w=rec.get("w", 1600), h=rec.get("h", 1200)))
            for ln in p.panel_lines(fr):
                print("   " + ln)
            print()
        return
    print("controls: space play/pause | <-/-> step | 1-5 speed | click select | t trails v vision "
          "h hearing d deaths n next-event b back r restart q quit")
    p.run()


if __name__ == "__main__":
    main()
