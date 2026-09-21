"""Flight geometry shared by the probe tools and the 3D tracker.

The camera was fitted on the official Helsinki truth (probe/camera_model.py):
official boxes are the projected, oriented 3D box of each object, and the fit
reproduces all 259 of them at median IoU 0.935. Here the world is expressed in
the FLIGHT frame, which is all the tracker ever needs:

    x  forward along the flight, metres, 0 at the camera of frame 0
    y  right of the flight, metres
    z  down (NED), metres; the camera flies at z = -ALT

The camera moves STEP metres forward per frame. Only the ratio STEP/altitude
reaches the image, so the ground elevation of each object is a free parameter
(z_ground > 0 means below the 600 m datum).
"""

import numpy as np

W, H = 3840, 2160
F = 3286.0
DEP = np.radians(71.26)
CU, CV = 1920.5, 1079.0
ALT = 600.0
STEP = 13.8888889

# Class 3D dimensions (length, width, height) in metres, fitted on Helsinki.
# One instance per class there; the renderer is assumed to reuse the model.
CLASS_DIMS = {
    'condor': (28.1, 30.3, 7.4), 'hangar': (33.4, 17.8, 14.3),
    'helicopter': (17.8, 21.4, 4.1), 'jammer': (2.9, 6.9, 4.9),
    'jet_plane': (16.9, 15.3, 4.2), 'large_launcher': (24.8, 15.8, 10.0),
    'large_tower': (6.0, 6.0, 14.6), 'medium_launcher': (13.1, 3.0, 3.8),
    'medium_plane': (9.5, 9.7, 4.0), 'mine_roller': (9.4, 10.1, 7.0),
    'small_launcher': (4.2, 2.5, 4.3), 'small_plane': (8.9, 7.9, 2.0),
    'small_tower': (9.5, 8.4, 9.3), 'spacecraft': (8.4, 7.7, 6.0),
    'ta-ta': (1.7, 5.4, 4.4), 'tank': (9.6, 8.6, 2.7),
}

_ZC = np.array([np.cos(DEP), 0.0, np.sin(DEP)])      # optical axis
_XC = np.array([0.0, 1.0, 0.0])                      # image right
_YC = np.cross(_ZC, _XC)                             # image down


def cam_pos(frame: float) -> np.ndarray:
    return np.array([STEP * frame, 0.0, -ALT])


def project(points: np.ndarray, frame: float) -> np.ndarray:
    rel = points - cam_pos(frame)
    x, y, z = rel @ _XC, rel @ _YC, rel @ _ZC
    return np.stack([CU + F * x / z, CV + F * y / z], axis=-1)


def ground_point(u: float, v: float, frame: float, z_ground: float = 0.0) -> np.ndarray:
    ray = _XC * (u - CU) / F + _YC * (v - CV) / F + _ZC
    origin = cam_pos(frame)
    t = (z_ground - origin[2]) / ray[2]
    return origin + t * ray


def corners(x, y, z_ground, length, width, height, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for dl in (-0.5, 0.5):
        for dw in (-0.5, 0.5):
            px = x + c * dl * length - s * dw * width
            py = y + s * dl * length + c * dw * width
            out.append((px, py, z_ground))
            out.append((px, py, z_ground - height))
    return np.array(out)


def box_at(obj, frame: float, clip=True):
    """Official-convention box of obj = (x, y, z_ground, L, W, H, yaw) at frame."""
    uv = project(corners(*obj), frame)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    if not clip:
        return np.array([x1, y1, x2, y2])
    return np.array([max(0.0, x1), max(0.0, y1), min(W, x2), min(H, y2)])


def visible(box) -> bool:
    return box[2] - box[0] > 0.5 and box[3] - box[1] > 0.5


def shrink(box, k):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * k / 2, (box[3] - box[1]) * k / 2
    return np.array([cx - w, cy - h, cx + w, cy + h])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre_speed(z_ground: float = 0.0) -> float:
    """Image speed (px/frame) of a ground point at the frame centre."""
    g = ground_point(CU, CV, 0.0, z_ground)
    return float(project(g[None], 1.0)[0, 1] - CV)
