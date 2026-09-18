"""Cut every labelled object out of the supplied frames, with a mask.

    python training/extract_patches.py            # -> data/patches/<class>/fNNNNNN.png

Each patch is a 4-channel PNG (BGRA): the object at native 4K scale, with an
alpha channel from GrabCut so it can be pasted onto new ground without
dragging a rectangle of the old ground along. Boxes cut by the frame edge are
skipped: they only show part of the object.

A contact sheet of all masks is written to data/patches/_sheet.png so you
can eyeball the cut-outs before generating a dataset from them.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from utils import IMAGE_HEIGHT, IMAGE_WIDTH, frame_numbers, load_sample  # noqa: E402

EDGE_GAP = 2        # a box this close to the frame edge counts as cut off
GRABCUT_ITERATIONS = 5
# Context pixels GrabCut may label as background. Six was not enough: GrabCut
# fits a five-component background GMM, and a six-pixel ring around a 45x47 box
# is a few hundred samples. Half the longest side gives it a real sample of the
# ground the object is standing on.
CONTEXT_SHARE = 0.5
MIN_CONTEXT = 10
# A mask this small is a failed cut; this large is a slab of ground.
MIN_FOREGROUND = 0.04
MAX_FOREGROUND = 0.70
# GrabCut is run on an upscaled crop as well as the native one. These objects
# are 20-170 px across at 4K, and at native scale the graph cut has too few
# pixels to separate a camouflaged object from grass: it returned nothing at all
# for helicopter, large_tower and medium_launcher, which then fell back to the
# ellipse and carried ~82% ground. Upscaling is what rescues them.
SUPERSAMPLE_TARGET = 340   # upscale the longest side of the box towards this
MAX_SUPERSAMPLE = 4
# Straight dark struts - rotor blades, launcher rails - are one or two pixels
# wide and the graph cut's smoothness term deletes them. They are recovered
# separately, so the cut-out keeps the object's real span.
STRUT_LENGTH = 9
STRUT_SIGMA = 2.0
# A strut is thin: erode it by one pixel and little should be left.
STRUT_MAX_CORE = 0.25
# make_dataset.py treats a mask this full as the ellipse fallback; keep in step.
FILLED_MASK = 0.7


def _grabcut(crop: np.ndarray, box, scale: int):
    """GrabCut the labelled box out of ``crop``, optionally upscaled first."""
    x1, y1, x2, y2 = box
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        x1, y1, x2, y2 = (value * scale for value in (x1, y1, x2, y2))
    mask = np.zeros(crop.shape[:2], np.uint8)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if scale > 1:
        mask = cv2.resize(mask, (crop.shape[1] // scale, crop.shape[0] // scale),
                          interpolation=cv2.INTER_AREA)
        mask = (mask > 110).astype(np.uint8) * 255
    return mask


def _struts(crop: np.ndarray, box, base: np.ndarray) -> np.ndarray:
    """Straight dark lines that touch ``base``: the object's thin appendages."""
    x1, y1, x2, y2 = box
    light = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    smooth = cv2.GaussianBlur(light, (0, 0), 1.0)
    # How much darker than its surroundings each pixel is.
    dark = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) - smooth

    # Open with a line at each angle: grass texture survives none of them.
    line = np.zeros((STRUT_LENGTH, STRUT_LENGTH), np.uint8)
    line[STRUT_LENGTH // 2, :] = 255
    centre = (STRUT_LENGTH / 2 - 0.5, STRUT_LENGTH / 2 - 0.5)
    strongest = np.zeros_like(dark)
    for angle in range(0, 180, 15):
        rotated = cv2.warpAffine(line, cv2.getRotationMatrix2D(centre, angle, 1.0),
                                 (STRUT_LENGTH, STRUT_LENGTH))
        kernel = (rotated > 90).astype(np.uint8)
        if kernel.sum() < 3:
            continue
        strongest = np.maximum(strongest, cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel))

    inside = strongest[y1:y2, x1:x2]
    found = np.zeros(crop.shape[:2], np.uint8)
    found[y1:y2, x1:x2] = (inside > inside.mean() + STRUT_SIGMA * inside.std()) * np.uint8(255)

    # Keep a component only if it is attached to the object we already found and
    # is actually thin. Without the thinness test a dark lake touching a wing
    # reads as one enormous strut and comes along with the aircraft.
    attached = cv2.dilate(base, np.ones((5, 5), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats(found)
    keep = []
    for index in range(1, count):
        component = (labels == index).astype(np.uint8)
        if not (attached[labels == index] > 0).any():
            continue
        core = cv2.erode(component, np.ones((3, 3), np.uint8))
        if core.sum() > component.sum() * STRUT_MAX_CORE:
            continue
        keep.append(index)
    return np.isin(labels, keep).astype(np.uint8) * 255


def _tidy(mask: np.ndarray, box, grow: bool = False) -> np.ndarray:
    """Clip to the box, close pinholes, drop specks.

    ``grow`` dilates by a pixel so the object's own outline survives the paste.
    Only the finished mask is grown: dilating an intermediate one too leaves a
    two-pixel halo of ground, which on a 17x32 cut-out is most of the object.
    """
    x1, y1, x2, y2 = box
    mask = mask.copy()
    mask[:y1] = 0
    mask[y2:] = 0
    mask[:, :x1] = 0
    mask[:, x2:] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [1 + i for i, area in enumerate(areas) if area >= areas.max() * 0.10]
        mask = np.isin(labels, keep).astype(np.uint8) * 255
    return cv2.dilate(mask, kernel) if grow else mask


def mask_fill(mask: np.ndarray, box) -> float:
    """Share of the labelled box the mask covers."""
    x1, y1, x2, y2 = box
    return float((mask[y1:y2, x1:x2] > 0).mean())


def object_mask(crop: np.ndarray, box_in_crop) -> np.ndarray:
    """Foreground mask (0/255) for the object whose box sits inside ``crop``."""
    x1, y1, x2, y2 = box_in_crop
    longest = max(x2 - x1, y2 - y1)
    top = int(max(1, min(MAX_SUPERSAMPLE, round(SUPERSAMPLE_TARGET / max(longest, 1)))))

    # Every scale that produced a plausible cut; the tightest one wins, because
    # the failure that costs us is a mask that swallows ground, not one that
    # clips an edge. The struts pass puts the thin parts back afterwards.
    candidates = []
    for scale in range(1, top + 1):
        mask = _grabcut(crop, box_in_crop, scale)
        if mask is None:
            continue
        mask = _tidy(mask, box_in_crop)
        if MIN_FOREGROUND <= mask_fill(mask, box_in_crop) <= MAX_FOREGROUND:
            candidates.append(mask)

    if not candidates:
        # Nothing separated the object from the ground. An ellipse keeps the box
        # honest at the price of ground in the corners; make_dataset.py spots a
        # mask this full and only rotates it by right angles.
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                    ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
        return mask

    base = min(candidates, key=lambda m: mask_fill(m, box_in_crop))
    return _tidy(cv2.bitwise_or(base, _struts(crop, box_in_crop, base)),
                 box_in_crop, grow=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE.parent / 'data' / 'patches')
    args = parser.parse_args()

    tiles = []
    quality = {}
    for frame in frame_numbers():
        image, annotations = load_sample(frame)
        for annotation in annotations:
            x1, y1, x2, y2 = annotation['bbox']
            if (x1 <= EDGE_GAP or y1 <= EDGE_GAP
                    or x2 >= IMAGE_WIDTH - EDGE_GAP or y2 >= IMAGE_HEIGHT - EDGE_GAP):
                continue
            margin = int(max(MIN_CONTEXT, CONTEXT_SHARE * max(x2 - x1, y2 - y1)))
            cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
            cx2, cy2 = min(IMAGE_WIDTH, x2 + margin), min(IMAGE_HEIGHT, y2 + margin)
            crop = image[cy1:cy2, cx1:cx2]
            mask = object_mask(crop, (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1))

            # Store exactly the labelled box: the pasted box is then the label.
            bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
            patch = np.dstack([crop[by1:by2, bx1:bx2], mask[by1:by2, bx1:bx2]])

            folder = args.out / annotation['object_id']
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f'f{frame:06d}.png'), patch)

            # Two numbers decide whether a cut-out is usable. Fill is how much
            # of the box survived: near 1.0 means a slab of ground came along.
            # Extent is the mask's own bounding box against the labelled box,
            # and it matters because make_dataset.py crops each patch to its
            # alpha before pasting, so the mask's bounds become the training
            # label. Well under 1.0 means we would teach an undersized box.
            alpha = patch[:, :, 3]
            ys, xs = np.nonzero(alpha > 127)
            extent = 0.0 if len(xs) == 0 else (
                (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
                / float((x2 - x1) * (y2 - y1)))
            quality.setdefault(annotation['object_id'], []).append(
                ((alpha > 127).mean(), extent))

            preview = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_NEAREST)
            cut = cv2.resize(cv2.bitwise_and(crop, crop, mask=mask), (96, 96), interpolation=cv2.INTER_NEAREST)
            tiles.append(np.hstack([preview, cut]))

    written = sorted(args.out.glob('*/*.png'))
    per_class = {}
    for path in written:
        per_class[path.parent.name] = per_class.get(path.parent.name, 0) + 1
    print(f'{"class":16s} {"n":>3s} {"fill":>6s} {"extent":>7s}')
    for name, count in sorted(per_class.items()):
        fills = [f for f, _ in quality.get(name, [])]
        extents = [e for _, e in quality.get(name, [])]
        fill = float(np.median(fills)) if fills else float('nan')
        extent = float(np.median(extents)) if extents else float('nan')
        flag = '  <- ground in the cut-out' if fill > FILLED_MASK else (
               '  <- mask misses part of the object' if extent < 0.45 else '')
        print(f'{name:16s} {count:3d} {fill:6.2f} {extent:7.2f}{flag}')
    print(f'{len(written)} patches in {args.out}')

    columns = 10
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    cv2.imwrite(str(args.out / '_sheet.png'), sheet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
