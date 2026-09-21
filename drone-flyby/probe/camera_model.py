"""A physical camera model of the flyby, fitted on the official Helsinki truth.

The official boxes are the projected 3D bounding box of each object. This fits
the camera (focal length, depression angle, principal point) and every object's
3D box (ground position, length, width, height, yaw) so that projecting the box
through the camera reproduces the official 2D boxes, clipped to the frame.

With it, one sighting of an object places it on the ground, and its official
box can then be predicted in every frame -- including how its shape changes as
it crosses the frame, which no 2D motion model gets right for tall objects.

    python probe/camera_model.py            # fit on Helsinki, report residuals
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

W, H = 3840, 2160
ROOT = Path(__file__).resolve().parent.parent


def load_helsinki():
    frames, poses = defaultdict(dict), {}
    for f in sorted(glob.glob(str(ROOT / 'src/helsinki/annotations/*.json'))):
        d = json.loads(Path(f).read_text())
        poses[d['frame']] = d['pose']
        for a in d['annotations']:
            frames[a['object_id']][d['frame']] = a['bbox']
    return frames, poses


class Camera:
    """Pinhole camera flying level, pitched forward, yawed along the flight.

    NED world (x north, y east, z down), as AirSim poses are. The camera sits
    at (px, py, -alt) and looks along the heading, depressed by `dep` radians
    below the horizon (pi/2 = straight down).
    """

    def __init__(self, f, dep, cu=W / 2, cv=H / 2, heading=0.0, alt=600.0):
        self.f, self.dep, self.cu, self.cv = f, dep, cu, cv
        self.heading, self.alt = heading, alt

    def axes(self):
        ch, sh = np.cos(self.heading), np.sin(self.heading)
        fwd = np.array([ch, sh, 0.0])
        right = np.array([-sh, ch, 0.0])
        down = np.array([0.0, 0.0, 1.0])
        zc = np.cos(self.dep) * fwd + np.sin(self.dep) * down
        xc = right
        yc = np.cross(zc, xc)
        return xc, yc, zc

    def project(self, pts, cam_xy):
        """World points (N, 3) -> image (N, 2), camera at (cam_x, cam_y, -alt)."""
        xc, yc, zc = self.axes()
        rel = pts - np.array([cam_xy[0], cam_xy[1], -self.alt])
        x, y, z = rel @ xc, rel @ yc, rel @ zc
        return np.stack([self.cu + self.f * x / z, self.cv + self.f * y / z], axis=1)

    def ground_point(self, u, v, cam_xy, z_world=0.0):
        """Image point -> world point on the plane z = z_world (NED)."""
        xc, yc, zc = self.axes()
        ray = xc * (u - self.cu) / self.f + yc * (v - self.cv) / self.f + zc
        origin = np.array([cam_xy[0], cam_xy[1], -self.alt])
        t = (z_world - origin[2]) / ray[2]
        return origin + t * ray


def box_corners(x, y, length, width, height, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for dl in (-0.5, 0.5):
        for dw in (-0.5, 0.5):
            px = x + c * dl * length - s * dw * width
            py = y + s * dl * length + c * dw * width
            for dz in (0.0, -height):
                pts.append((px, py, dz))
    return np.array(pts)


def projected_box(cam, cam_xy, obj):
    uv = cam.project(box_corners(*obj), cam_xy)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    return np.array([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])


def fit_helsinki(verbose=True):
    tracks, poses = load_helsinki()
    names = sorted(tracks)
    frames = sorted(poses)
    cam_xy = {fr: (poses[fr]['x'], poses[fr]['y']) for fr in frames}
    d = np.array(cam_xy[frames[-1]]) - np.array(cam_xy[frames[0]])
    heading = float(np.arctan2(d[1], d[0]))

    # Initial guesses: f from the ~0.2 m/px ground sample distance at 600 m.
    # Motion field: 68.4 px/frame at the centre, ~54 top / ~83 bottom
    # -> depression ~71-72 deg, f ~3300 px (a ~60 deg horizontal FOV).
    f0, dep0 = 3300.0, np.radians(71)
    cam0 = Camera(f0, dep0, heading=heading)
    obj0 = []
    for n in names:
        fr = sorted(tracks[n])[len(tracks[n]) // 2]
        b = tracks[n][fr]
        g = cam0.ground_point((b[0] + b[2]) / 2, b[3], cam_xy[fr])
        size = max(3.0, (b[2] - b[0]) * 0.17)
        obj0 += [g[0], g[1], size, size, 2.0, 0.3]
    x0 = np.array([f0, dep0, W / 2, H / 2, 0.0] + obj0)

    def unpack(p):
        cam = Camera(p[0], p[1], p[2], p[3], heading=heading + p[4])
        objs = {n: p[5 + 6 * i: 11 + 6 * i] for i, n in enumerate(names)}
        return cam, objs

    def residuals(p):
        cam, objs = unpack(p)
        out = []
        for n in names:
            o = objs[n].copy()
            o[2:5] = np.abs(o[2:5])
            for fr, b in tracks[n].items():
                out.extend(projected_box(cam, cam_xy[fr], o) - np.array(b, float))
        return np.array(out)

    res = least_squares(residuals, x0, loss='soft_l1', f_scale=3.0, max_nfev=4000)
    cam, objs = unpack(res.x)
    r = residuals(res.x).reshape(-1, 4)
    if verbose:
        print(f'camera: f={cam.f:.1f}px  depression={np.degrees(cam.dep):.2f} deg  '
              f'cu={cam.cu:.1f} cv={cam.cv:.1f}  heading={np.degrees(heading):.2f} deg, camera yaw offset {np.degrees(cam.heading - heading):.2f} deg')
        print(f'box-coordinate residual: median |r| {np.median(np.abs(r)):.2f} px, '
              f'p90 {np.quantile(np.abs(r), 0.9):.2f} px, max {np.abs(r).max():.1f}')
        for n in names:
            o = objs[n]
            print(f'  {n:16s} L={abs(o[2]):5.1f}m W={abs(o[3]):5.1f}m H={abs(o[4]):5.1f}m '
                  f'yaw={np.degrees(o[5]) % 180:6.1f}')
    return cam, objs, cam_xy, tracks


if __name__ == '__main__':
    cam, objs, cam_xy, tracks = fit_helsinki()
    # IoU of the model's box against every official box.
    ious = []
    for n, t in tracks.items():
        o = objs[n].copy(); o[2:5] = np.abs(o[2:5])
        for fr, b in t.items():
            p = projected_box(cam, cam_xy[fr], o)
            ix = max(0, min(p[2], b[2]) - max(p[0], b[0])); iy = max(0, min(p[3], b[3]) - max(p[1], b[1]))
            inter = ix * iy
            union = (p[2] - p[0]) * (p[3] - p[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            ious.append(inter / union if union > 0 else 0)
    ious = np.array(ious)
    print(f'IoU model vs official: median {np.median(ious):.3f}, p10 {np.quantile(ious, 0.1):.3f}, '
          f'min {ious.min():.3f}, share >= 0.5: {(ious >= 0.5).mean():.3f}')


def refit_objects(cam, cam_xy, tracks, restarts=((0.0, 2.0), (0.8, 2.0), (0.0, 12.0), (0.8, 12.0), (0.4, 25.0), (1.2, 6.0))):
    """Camera fixed: fit each object's 3D box alone, from several starts."""
    fitted = {}
    for name, t in tracks.items():
        frs = sorted(t)
        mid = frs[len(frs) // 2]
        b = t[mid]
        g = cam.ground_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, cam_xy[mid])

        def res(p):
            o = p.copy(); o[2:5] = np.abs(o[2:5])
            return np.concatenate([projected_box(cam, cam_xy[fr], o) - np.array(t[fr], float) for fr in frs])

        best = None
        for yaw0, h0 in restarts:
            size = max(2.0, (b[2] - b[0]) * 0.15)
            x0 = np.array([g[0], g[1], size, size * 0.8, h0, yaw0])
            r = least_squares(res, x0, loss='soft_l1', f_scale=2.0, max_nfev=3000)
            if best is None or r.cost < best.cost:
                best = r
        o = best.x.copy(); o[2:5] = np.abs(o[2:5])
        fitted[name] = o
    return fitted


def box_iou(p, b):
    ix = max(0, min(p[2], b[2]) - max(p[0], b[0])); iy = max(0, min(p[3], b[3]) - max(p[1], b[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
