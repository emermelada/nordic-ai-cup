"""Can two views of the same ground be aligned well enough to stack them?

Multi-frame super-resolution only works if the alignment is sub-pixel. The
camera model gives a closed-form homography between any two frames (both are
projections of the same ground plane), so this measures what that homography
leaves behind, and what a phase-correlation refinement recovers on top.

    python probe/align_test.py --run <id>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402


def ground_homography(frame_from, frame_to, z_ground=0.0):
    """Homography mapping source pixels at frame_from to source pixels at frame_to.

    Both frames see the same ground plane, so four ground points determine it.
    """
    src, dst = [], []
    for u, v in ((600, 300), (3200, 300), (600, 1900), (3200, 1900)):
        p = G.ground_point(u, v, frame_from, z_ground)
        src.append([u, v])
        dst.append(G.project(p[None], frame_to)[0])
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def view_to_source(view, region, out_size=(3840, 2160)):
    """Place a 960x540 view back into a full-size source-frame canvas."""
    x1, y1, x2, y2 = region
    canvas = np.zeros((out_size[1], out_size[0], 3), np.uint8)
    mask = np.zeros((out_size[1], out_size[0]), np.uint8)
    resized = cv2.resize(view, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    canvas[y1:y2, x1:x2] = resized
    mask[y1:y2, x1:x2] = 255
    return canvas, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--pairs', type=int, default=12)
    a = ap.parse_args()

    metas = []
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        png = meta_path.with_suffix('.png')
        if png.exists():
            metas.append((meta, png))

    residuals = []
    for i, (meta, png) in enumerate(metas):
        # Find a later view whose region overlaps this one's ground after motion.
        for meta2, png2 in metas[i + 1:i + 10]:
            gap = meta2['frame'] - meta['frame']
            if gap < 1 or gap > 8:
                continue
            if meta['view']['resolution_level'] != 1 or meta2['view']['resolution_level'] != 1:
                continue
            r1 = meta['view']['source_region_xyxy']
            r2 = meta2['view']['source_region_xyxy']
            img1 = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(png2), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                continue
            # Warp view 1 into frame 2's source coordinates through the ground.
            H = ground_homography(meta['frame'], meta2['frame'])
            # view1 pixels -> source1 pixels
            s1 = np.array([[(r1[2] - r1[0]) / 960.0, 0, r1[0]], [0, (r1[3] - r1[1]) / 540.0, r1[1]], [0, 0, 1]])
            # source2 pixels -> view2 pixels
            s2inv = np.array([[960.0 / (r2[2] - r2[0]), 0, 0], [0, 540.0 / (r2[3] - r2[1]), 0], [0, 0, 1]])
            s2inv = s2inv @ np.array([[1, 0, -r2[0]], [0, 1, -r2[1]], [0, 0, 1]])
            M = s2inv @ H @ s1
            warped = cv2.warpPerspective(img1, M, (960, 540))
            overlap = (warped > 0)
            if overlap.mean() < 0.25:
                continue
            # Compare on the overlapping part only.
            ys, xs = np.where(overlap)
            y0, y1_, x0, x1_ = ys.min(), ys.max(), xs.min(), xs.max()
            if y1_ - y0 < 200 or x1_ - x0 < 200:
                continue
            patch_a = warped[y0:y1_, x0:x1_].astype(np.float32)
            patch_b = img2[y0:y1_, x0:x1_].astype(np.float32)
            valid = (patch_a > 0)
            if valid.mean() < 0.9:
                continue
            win = cv2.createHanningWindow((patch_a.shape[1], patch_a.shape[0]), cv2.CV_32F)
            shift, response = cv2.phaseCorrelate(patch_a, patch_b, win)
            residuals.append((abs(shift[0]), abs(shift[1]), response, gap))
            break

        if len(residuals) >= a.pairs:
            break

    if not residuals:
        print('no comparable pairs found')
        return
    arr = np.array([[r[0], r[1], r[2]] for r in residuals])
    print(f'{len(residuals)} overlapping pairs')
    print(f'model-only residual: dx median {np.median(arr[:,0]):.2f} px, '
          f'dy median {np.median(arr[:,1]):.2f} px, p90 {np.quantile(np.hypot(arr[:,0],arr[:,1]),0.9):.2f} px')
    print(f'phase-correlation confidence: median {np.median(arr[:,2]):.3f}')
    print('per pair (dx, dy, response, gap):')
    for r in residuals:
        print(f'   {r[0]:6.2f} {r[1]:6.2f}  {r[2]:.3f}  gap {r[3]}')


if __name__ == '__main__':
    main()
