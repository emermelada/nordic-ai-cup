"""Build a synthetic YOLO dataset from the supplied frames and object patches.

    python training/extract_patches.py                    # once, makes data/patches
    python training/make_dataset.py --scenes 20 --preview # quick look
    python training/make_dataset.py --scenes 1200         # the real thing

For every synthetic scene:

1. take a background: one of the 4K Helsinki frames (randomly flipped, real
   objects and labels kept), or, with --backgrounds, a random 3840x2160 cut
   from any large aerial photo (no objects of ours in it, so no labels);
2. paste extra objects from data/patches at free spots, classes drawn evenly,
   each randomly rotated, scaled and recoloured a little;
3. cut camera views out of it exactly the way the evaluator does (the crop for
   the level, then INTER_AREA down to 960x540) and write YOLO labels.

Output: data/yolo/{images,labels}/{train,val}/ and data/yolo/data.yaml.
"""

import argparse
import math
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

# GeoTIFF photos carry map tags OpenCV does not know; the warnings are noise.
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import (  # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    center_bounds_for_level,
    frame_numbers,
    load_sample,
    source_region_for_view,
)

VIEW_SIZE = (960, 540)
# How many views to cut from each synthetic scene, per resolution level.
VIEWS_PER_LEVEL = {0: 1, 1: 2, 2: 3}
PASTES_PER_SCENE = (8, 30)
UNTOUCHED_SCENE_SHARE = 0.15   # scenes left exactly as supplied
SCALE_RANGE = (0.85, 1.15)     # cut-outs taken from the validation flight
# Helsinki cut-outs. PLATEAU_IDEAS.md measured validation objects at 0.55-0.85x
# their Helsinki box diagonal and v5 pasted them that small, but v5 then scored
# 0.140 against v4's 0.143 and reported 1.2-1.5x oversized boxes. The measurement
# was made with a rotation-biased box-ratio method, and the real check points the
# other way: BOX_SCALE=0.8 at inference collapsed a validation run from 0.143 to
# 0.017, so the true boxes are not small. Back to the range v4 trained on.
HELSINKI_SCALE_RANGE = (0.85, 1.15)
# Hue rotation in degrees: whole backgrounds, and pasted objects (whose colour
# is mostly their own, so less).
BACKGROUND_HUE = 20
PATCH_HUE = 8
MIN_VISIBLE = 0.5              # keep a box cut by the view edge if this much shows
MIN_BOX_VIEW_PIXELS = 2
L2_ON_OBJECT_SHARE = 0.7       # zoomed views mostly look at something
FILLED_MASK = 0.7              # mask this full is the ellipse fallback

CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}

# With --backgrounds, this share of scenes still uses the Helsinki frames.
HELSINKI_SHARE = 0.2
BACKGROUND_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
# Label rasters live next to the photos in segmentation datasets.
BACKGROUND_SKIP_WORDS = ('mask', '/gt/', 'drone-flyby-code')
BACKGROUND_MIN_SIDE = 2000   # smaller photos would need blurry upscaling
BACKGROUND_SCALE = (0.8, 1.25)
# Share of background scenes taken from the recorded flight rather than the
# stock aerial photo sets, when --real-backgrounds is given.
REAL_BACKGROUND_SHARE = 0.6

# Patches cut from recorded validation views (--extra-patches) are used for
# this share of pastes of their class; they carry that scene's lighting.
EXTRA_PATCH_SHARE = 0.4
# Photometric and shadow variation per pasted object, so a model does not learn
# one scene's light: validation objects are darker and cast hard shadows.
GRADE_SHARE = 0.6
BLUR_SHARE = 0.3
SHADOW_SHARE = 0.5
NOISE_SHARE = 0.25
# Share of well-masked patches that are also Poisson-blended (the badly masked
# ones always are), so a model never learns one pasting style.
BLEND_SHARE = 0.25
BLEND_MIN_TEXTURE = 8.0    # grey-level std range of the ground under a blended patch
BLEND_MAX_TEXTURE = 22.0

# Filled in per worker by _init_worker.
_patches = {}
_extra_patches = {}
_class_weights = None
_frames = []
_backgrounds = []
_real_backgrounds = []


def load_patches(folder: Path, required: bool = True):
    patches = {}
    for name in OBJECT_CLASSES:
        files = sorted((folder / name).glob('*.png'))
        if not files:
            if required:
                raise SystemExit(f'No patches for {name} in {folder}: run extract_patches.py first')
            continue
        patches[name] = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in files]
    return patches


def pick_patch(name: str, rng: random.Random):
    """A patch of this class and the scale range that suits its source."""
    extra = _extra_patches.get(name)
    if extra and rng.random() < EXTRA_PATCH_SHARE:
        return rng.choice(extra), SCALE_RANGE
    return rng.choice(_patches[name]), HELSINKI_SCALE_RANGE


def rotate_hue(bgr: np.ndarray, degrees: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + int(round(degrees / 2))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def find_backgrounds(folders) -> List[Path]:
    found = []
    for folder in folders:
        for path in Path(folder).rglob('*'):
            text = str(path).lower()
            if path.suffix.lower() in BACKGROUND_SUFFIXES and not any(w in text for w in BACKGROUND_SKIP_WORDS):
                found.append(path)
    return sorted(found)


def _init_worker(patch_folder: Path, backgrounds, extra_folders=(), class_weights=None, helsinki_share=None,
                 real_backgrounds=(), real_share=None):
    global _patches, _frames, _backgrounds, _extra_patches, _class_weights, HELSINKI_SHARE
    global _real_backgrounds, REAL_BACKGROUND_SHARE
    if real_share is not None:
        REAL_BACKGROUND_SHARE = real_share
    _real_backgrounds = list(real_backgrounds)
    if helsinki_share is not None:
        HELSINKI_SHARE = helsinki_share
    _class_weights = [class_weights.get(name, 1.0) for name in OBJECT_CLASSES] if class_weights else None
    _patches = load_patches(patch_folder)
    _extra_patches = {}
    for folder in extra_folders:
        for name, images in load_patches(folder, required=False).items():
            _extra_patches.setdefault(name, []).extend(images)
    _frames = frame_numbers()
    _backgrounds = backgrounds


def load_background(rng: random.Random) -> Optional[np.ndarray]:
    """A random 3840x2160 cut from one of the aerial photos, or None."""
    for _attempt in range(5):
        # Terrain from the flight itself is the closest background we have to the
        # scene being scored, so it is drawn far more often than the stock photos.
        pool = (_real_backgrounds if _real_backgrounds and rng.random() < REAL_BACKGROUND_SHARE
                else _backgrounds or _real_backgrounds)
        path = rng.choice(pool)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < BACKGROUND_MIN_SIDE:
            continue
        scale = rng.uniform(*BACKGROUND_SCALE)
        # Never smaller than the frame.
        scale = max(scale, IMAGE_WIDTH / image.shape[1], IMAGE_HEIGHT / image.shape[0])
        if abs(scale - 1) > 0.01:
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        y = rng.randint(0, image.shape[0] - IMAGE_HEIGHT)
        x = rng.randint(0, image.shape[1] - IMAGE_WIDTH)
        image = image[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH]
        if rng.random() < 0.5:
            image = image[::-1, ::-1]
        # Nudge the colours a little so photos from one city do not all look alike.
        image = image.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
        image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
        return rotate_hue(image, rng.uniform(-BACKGROUND_HUE, BACKGROUND_HUE))
    return None


def transform_patch(patch: np.ndarray, rng: random.Random, scale_range=SCALE_RANGE) -> np.ndarray:
    """Rotate, scale, flip and recolour one BGRA patch; crop to its mask."""
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    filled = (patch[:, :, 3] > 0).mean() > FILLED_MASK
    # An ellipse mask carries ground in its corners; only right angles keep
    # its box honest.
    angle = rng.choice([0, 90, 180, 270]) if filled else rng.uniform(0, 360)
    scale = rng.uniform(*scale_range)

    height, width = patch.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(math.ceil(height * sin + width * cos)) + 2
    new_height = int(math.ceil(height * cos + width * sin)) + 2
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    patch = cv2.warpAffine(
        np.ascontiguousarray(patch), matrix, (new_width, new_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    ys, xs = np.nonzero(patch[:, :, 3] > 127)
    if len(xs) == 0:
        return None
    patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    bgr = patch[:, :, :3].astype(np.float32)
    bgr = bgr * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    bgr *= np.array([rng.uniform(0.95, 1.05) for _ in range(3)], np.float32)
    if rng.random() < GRADE_SHARE:
        # Another scene's light: gamma (mostly darker), flatter, greyer.
        bgr = 255.0 * (np.clip(bgr, 0, 255) / 255.0) ** rng.uniform(0.8, 1.8)
        mean = bgr.mean()
        bgr = mean + (bgr - mean) * rng.uniform(0.6, 1.1)
        grey = bgr.mean(axis=2, keepdims=True)
        bgr = grey + (bgr - grey) * rng.uniform(0.4, 1.2)
    patch[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    if PATCH_HUE:
        patch[:, :, :3] = rotate_hue(np.ascontiguousarray(patch[:, :, :3]), rng.uniform(-PATCH_HUE, PATCH_HUE))
    if rng.random() < BLUR_SHARE:
        patch[:, :, :3] = cv2.GaussianBlur(patch[:, :, :3], (0, 0), rng.uniform(0.4, 1.2))
    return patch


def overlaps(box, boxes, gap: int = 8) -> bool:
    x1, y1, x2, y2 = box
    return any(
        x1 - gap < bx2 and bx1 < x2 + gap and y1 - gap < by2 and by1 < y2 + gap
        for bx1, by1, bx2, by2 in boxes
    )


def cast_shadow(scene: np.ndarray, patch: np.ndarray, x: int, y: int, rng: random.Random) -> None:
    """Darken the ground where the object's silhouette would throw a shadow."""
    height, width = patch.shape[:2]
    length = rng.uniform(0.15, 0.6) * max(height, width)
    angle = rng.uniform(0, 2 * math.pi)
    dx, dy = int(length * math.cos(angle)), int(length * math.sin(angle))
    sx1, sy1 = max(0, x + dx), max(0, y + dy)
    sx2, sy2 = min(scene.shape[1], x + dx + width), min(scene.shape[0], y + dy + height)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    mask = patch[sy1 - y - dy:sy2 - y - dy, sx1 - x - dx:sx2 - x - dx, 3].astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.8, 2.5))[:, :, None]
    darkness = rng.uniform(0.3, 0.65)
    region = scene[sy1:sy2, sx1:sx2].astype(np.float32)
    scene[sy1:sy2, sx1:sx2] = (region * (1 - darkness * mask)).astype(np.uint8)


def paste_blended(scene: np.ndarray, patch: np.ndarray, x: int, y: int) -> bool:
    """Poisson-blend a patch whose mask is a disc or rectangle of its old ground.

    Helicopters, towers and launchers are thin and camouflaged, so their cut-outs
    keep a disc of Helsinki grass. Pasted as is, that disc is what a model learns.
    Mixed cloning keeps the strongest gradients of either image: the object's
    edges survive, the old ground takes the new ground's colour and texture.
    """
    height, width = patch.shape[:2]
    mask = (patch[:, :, 3] > 0).astype(np.uint8) * 255
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0
    if (x < 1 or y < 1 or x + width >= scene.shape[1] - 1 or y + height >= scene.shape[0] - 1
            or height < 5 or width < 5 or not mask.any()):
        return False
    try:
        # Flatten the old ground's fine texture; strong edges (rotors, lattice) stay.
        source = cv2.edgePreservingFilter(np.ascontiguousarray(patch[:, :, :3]),
                                          flags=cv2.RECURS_FILTER, sigma_s=20, sigma_r=0.25)
        blended = cv2.seamlessClone(source, scene, mask,
                                    (x + width // 2, y + height // 2), cv2.MIXED_CLONE)
    except cv2.error:
        return False
    scene[:] = blended
    return True


def paste(scene: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    height, width = patch.shape[:2]
    alpha = patch[:, :, 3].astype(np.float32) / 255.0
    # A one-pixel feather hides the cut line without smearing the object.
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[:, :, None]
    region = scene[y:y + height, x:x + width].astype(np.float32)
    blended = region * (1 - alpha) + patch[:, :, :3].astype(np.float32) * alpha
    scene[y:y + height, x:x + width] = blended.astype(np.uint8)


def build_scene(rng: random.Random):
    """One 4K image plus its labels as [(class, x1, y1, x2, y2)]."""
    image = None
    if (_backgrounds or _real_backgrounds) and rng.random() >= HELSINKI_SHARE:
        image = load_background(rng)
    if image is not None:
        labels = []
    else:
        image, annotations = load_sample(rng.choice(_frames))
        labels = [(a['object_id'], *a['bbox']) for a in annotations]

    if rng.random() < 0.5:
        image = image[:, ::-1]
        labels = [(n, IMAGE_WIDTH - x2, y1, IMAGE_WIDTH - x1, y2) for n, x1, y1, x2, y2 in labels]
    if rng.random() < 0.5:
        image = image[::-1]
        labels = [(n, x1, IMAGE_HEIGHT - y2, x2, IMAGE_HEIGHT - y1) for n, x1, y1, x2, y2 in labels]
    image = np.ascontiguousarray(image)

    if rng.random() < UNTOUCHED_SCENE_SHARE:
        return image, labels

    occupied = [box[1:] for box in labels]
    for _ in range(rng.randint(*PASTES_PER_SCENE)):
        name = rng.choices(OBJECT_CLASSES, weights=_class_weights)[0]
        source, scale_range = pick_patch(name, rng)
        patch = transform_patch(source, rng, scale_range)
        if patch is None:
            continue
        height, width = patch.shape[:2]
        filled = (patch[:, :, 3] > 0).mean() >= FILLED_MASK
        for _attempt in range(20):
            x = rng.randint(0, IMAGE_WIDTH - width)
            y = rng.randint(0, IMAGE_HEIGHT - height)
            box = (x, y, x + width, y + height)
            if filled and _attempt < 15:
                # Blending loses a thin object in busy texture (forest) and
                # leaves the old ground visible on flat ground (water): look
                # for moderately textured ground first.
                region = image[y:y + height:2, x:x + width:2]
                texture = region.mean(axis=2).std() if region.size else 0.0
                if not BLEND_MIN_TEXTURE <= texture <= BLEND_MAX_TEXTURE:
                    continue
            if not overlaps(box, occupied):
                # Only cut-outs with a real silhouette get a shadow: a
                # rectangle's shadow would be a giveaway.
                if rng.random() < SHADOW_SHARE and (patch[:, :, 3] > 0).mean() < FILLED_MASK:
                    cast_shadow(image, patch, x, y, rng)
                if not ((filled or rng.random() < BLEND_SHARE) and paste_blended(image, patch, x, y)):
                    paste(image, patch, x, y)
                occupied.append(box)
                labels.append((name, *box))
                break
    return image, labels


def pick_centre(level: int, labels, rng: random.Random):
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    if level == 0:
        return IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
    if level == 2 and labels and rng.random() < L2_ON_OBJECT_SHARE:
        _, x1, y1, x2, y2 = rng.choice(labels)
        # Anywhere that still shows the object, not always dead centre.
        cx = (x1 + x2) / 2 + rng.uniform(-400, 400)
        cy = (y1 + y2) / 2 + rng.uniform(-220, 220)
    else:
        cx, cy = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)
    return int(min(max(cx, min_x), max_x)), int(min(max(cy, min_y), max_y))


def render_view(image, labels, level: int, cx: int, cy: int):
    """Crop and shrink like the evaluator; return (view, yolo label lines)."""
    rx1, ry1, rx2, ry2 = source_region_for_view(level, cx, cy)
    view = image[ry1:ry2, rx1:rx2]
    if view.shape[1] != VIEW_SIZE[0]:
        view = cv2.resize(view, VIEW_SIZE, interpolation=cv2.INTER_AREA)
    if NOISE_SHARE and random.random() < NOISE_SHARE:
        # Sensor-like grain, so the model does not rely on perfectly clean pixels.
        view = np.clip(view + np.random.normal(0, random.uniform(1.5, 5), view.shape), 0, 255).astype(np.uint8)
    region_width, region_height = rx2 - rx1, ry2 - ry1
    scale = VIEW_SIZE[0] / region_width

    lines = []
    for name, x1, y1, x2, y2 in labels:
        ix1, iy1, ix2, iy2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
            continue
        if min(ix2 - ix1, iy2 - iy1) * scale < MIN_BOX_VIEW_PIXELS:
            continue
        xc = ((ix1 + ix2) / 2 - rx1) / region_width
        yc = ((iy1 + iy2) / 2 - ry1) / region_height
        w, h = (ix2 - ix1) / region_width, (iy2 - iy1) / region_height
        lines.append(f'{CLASS_INDEX[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
    return view, lines


def make_scene(job):
    index, split, seed, out, extension = job
    rng = random.Random(seed)
    # render_view's noise uses the module generators: keep scenes reproducible.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    image, labels = build_scene(rng)
    written = 0
    for level, count in VIEWS_PER_LEVEL.items():
        for view_number in range(count if level else 1):
            cx, cy = pick_centre(level, labels, rng)
            view, lines = render_view(image, labels, level, cx, cy)
            stem = f's{index:05d}_L{level}_{view_number}'
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == 'png' else [cv2.IMWRITE_JPEG_QUALITY, 95]
            cv2.imwrite(str(out / 'images' / split / f'{stem}.{extension}'), view, params)
            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            written += 1
    return written


def write_preview(out: Path, split: str, count: int = 12):
    """Draw the labels onto a few views so they can be checked by eye."""
    tiles = []
    for image_path in sorted((out / 'images' / split).iterdir())[:count]:
        view = cv2.imread(str(image_path))
        label_path = out / 'labels' / split / f'{image_path.stem}.txt'
        for line in label_path.read_text().split('\n'):
            if not line:
                continue
            cls, xc, yc, w, h = line.split()
            xc, yc, w, h = float(xc) * 960, float(yc) * 540, float(w) * 960, float(h) * 540
            p1 = (int(xc - w / 2) - 2, int(yc - h / 2) - 2)
            p2 = (int(xc + w / 2) + 2, int(yc + h / 2) + 2)
            cv2.rectangle(view, p1, p2, (0, 0, 255), 1)
            cv2.putText(view, OBJECT_CLASSES[int(cls)], (p1[0], p1[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(view, image_path.stem, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(view)
    columns = 3
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    path = out / f'preview_{split}.jpg'
    cv2.imwrite(str(path), sheet)
    print(f'preview: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenes', type=int, default=1200, help='Synthetic 4K scenes (6 views each).')
    parser.add_argument('--val-share', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patches', type=Path, default=HERE.parent / 'data' / 'patches')
    parser.add_argument('--out', type=Path, default=HERE.parent / 'data' / 'yolo')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png',
                        help='png matches what the evaluator sends; jpg is ~6x smaller.')
    parser.add_argument('--real-backgrounds', type=Path, nargs='*', default=[],
                        help='Folders of backgrounds cut from the recorded flight '
                             '(training/make_real_backgrounds.py); drawn REAL_BACKGROUND_SHARE of the time.')
    parser.add_argument('--real-share', type=float, default=None,
                        help=f'Override that share (default {REAL_BACKGROUND_SHARE}).')
    parser.add_argument('--backgrounds', type=Path, nargs='*', default=[],
                        help='Folders searched for large aerial photos to paste onto.')
    parser.add_argument('--extra-patches', type=Path, nargs='*', default=[],
                        help='More patch folders (e.g. data/patches_val), used for part of the pastes.')
    parser.add_argument('--helsinki-share', type=float, default=HELSINKI_SHARE,
                        help='Share of scenes on Helsinki frames when --backgrounds is given.')
    parser.add_argument('--class-weights', default='',
                        help='name=weight,... pasted more (or less) often, e.g. small_launcher=2,ta-ta=2')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--preview', action='store_true', help='Also draw labels onto a few views.')
    args = parser.parse_args()

    for kind in ('images', 'labels'):
        for split in ('train', 'val'):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)

    real_backgrounds = find_backgrounds(args.real_backgrounds)
    if args.real_backgrounds:
        print(f'{len(real_backgrounds)} backgrounds from the recorded flight', flush=True)
        if not real_backgrounds:
            raise SystemExit('No backgrounds found in ' + ', '.join(map(str, args.real_backgrounds)))
    backgrounds = find_backgrounds(args.backgrounds)
    if args.backgrounds:
        print(f'{len(backgrounds)} background photos found', flush=True)
        for path in backgrounds[:5]:
            print('  e.g.', path)
        if not backgrounds:
            raise SystemExit('No background photos found in ' + ', '.join(map(str, args.backgrounds)))

    weights = {}
    for item in filter(None, args.class_weights.split(',')):
        name, value = item.split('=')
        if name not in OBJECT_CLASSES:
            raise SystemExit(f'unknown class {name}')
        weights[name] = float(value)

    val_count = max(1, int(args.scenes * args.val_share))
    jobs = [
        (i, 'val' if i < val_count else 'train', args.seed * 1_000_003 + i, args.out, args.format)
        for i in range(args.scenes)
    ]
    with Pool(args.workers, initializer=_init_worker, initargs=(args.patches, backgrounds, [p for p in args.extra_patches if p.exists()], weights,
                                                                   args.helsinki_share, real_backgrounds,
                                                                   args.real_share)) as pool:
        total = 0
        for done, written in enumerate(pool.imap_unordered(make_scene, jobs), 1):
            total += written
            if done % 50 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} scenes, {total} views', flush=True)

    names = '\n'.join(f'  {i}: {name}' for i, name in enumerate(OBJECT_CLASSES))
    (args.out / 'data.yaml').write_text(
        f'path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n'
    )
    print(f'dataset: {args.out / "data.yaml"}')

    if args.preview:
        write_preview(args.out, 'train')
    return 0


if __name__ == '__main__':
    sys.exit(main())
