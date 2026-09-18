"""Rebuild the recorded validation flight as complete 4K frames.

The validation flight is deterministic: every attempt renders the same scene, so
a view recorded in one run is byte-identical to the same view in another. All the
views we have recorded therefore mosaic into one full frame per flight frame.

    python tools/build_scene.py                    # every frame -> data/scene/
    python tools/build_scene.py --frames 60 120    # just these
    python tools/build_scene.py --report           # coverage only, write nothing

Each pixel is taken from the highest resolution level that ever saw it, so a
region surveyed at L2 keeps its full detail.
"""

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDINGS = ROOT / 'data' / 'recordings'
SCENE = ROOT / 'data' / 'scene'

WIDTH, HEIGHT = 3840, 2160


def index_views():
    """frame -> [(level, region, png path)], lowest level first."""
    views = collections.defaultdict(list)
    seen = set()
    for meta_path in sorted(RECORDINGS.glob('*/*.json')):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        view = meta.get('view') or {}
        region = view.get('source_region_xyxy')
        if not region:
            continue
        png = meta_path.with_suffix('.png')
        if not png.exists():
            continue
        key = (meta['frame'], view['resolution_level'], tuple(region))
        if key in seen:      # the same view recorded again in another run
            continue
        seen.add(key)
        views[meta['frame']].append((view['resolution_level'], tuple(int(v) for v in region), png))
    for frame in views:
        views[frame].sort(key=lambda item: item[0])
    return views


def build_frame(views_for_frame):
    """Mosaic one frame. Returns (image, level map) with -1 where nothing was seen."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    levels = np.full((HEIGHT, WIDTH), -1, np.int8)
    for level, (x1, y1, x2, y2), png in views_for_frame:
        image = cv2.imread(str(png))
        if image is None:
            continue
        width, height = x2 - x1, y2 - y1
        if (image.shape[1], image.shape[0]) != (width, height):
            # L0/L1 views are downsampled; upscale back to source pixels.
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        canvas[y1:y2, x1:x2] = image
        levels[y1:y2, x1:x2] = level
    return canvas, levels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', nargs='*', type=int, default=None)
    parser.add_argument('--report', action='store_true', help='Print coverage, write nothing.')
    parser.add_argument('--out', type=Path, default=SCENE)
    args = parser.parse_args()

    views = index_views()
    frames = sorted(args.frames if args.frames else views)
    if not args.report:
        args.out.mkdir(parents=True, exist_ok=True)

    coverage = {}
    for frame in frames:
        entries = views.get(frame, [])
        if not entries:
            print(f'frame {frame:3d}: nothing recorded')
            continue
        image, levels = build_frame(entries)
        seen = levels >= 0
        coverage[frame] = {
            'views': len(entries),
            'any': round(float(seen.mean()), 4),
            'l1_plus': round(float((levels >= 1).mean()), 4),
            'l2': round(float((levels >= 2).mean()), 4),
        }
        if not args.report:
            cv2.imwrite(str(args.out / f'frame_{frame:06d}.png'), image)
            # level map as a tiny paletted PNG: 0 = unseen, 1..3 = level + 1
            cv2.imwrite(str(args.out / f'levels_{frame:06d}.png'), (levels + 1).astype(np.uint8))
        print(f'frame {frame:3d}: {len(entries):2d} views  seen {seen.mean():6.1%}  '
              f'L1+ {(levels >= 1).mean():6.1%}  L2 {(levels >= 2).mean():5.1%}')

    if coverage and not args.report:
        (args.out / 'coverage.json').write_text(json.dumps(coverage, indent=1))
    if coverage:
        any_ = np.array([c['any'] for c in coverage.values()])
        l1 = np.array([c['l1_plus'] for c in coverage.values()])
        print(f'\n{len(coverage)} frames: mean coverage {any_.mean():.1%} (L1+ {l1.mean():.1%}), '
              f'{(any_ > 0.999).sum()} frames complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
