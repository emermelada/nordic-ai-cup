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

MARGIN = 6          # context pixels GrabCut may label as background
EDGE_GAP = 2        # a box this close to the frame edge counts as cut off
GRABCUT_ITERATIONS = 5
MIN_FOREGROUND = 0.05  # below this share of the box, GrabCut failed: use an ellipse


def object_mask(crop: np.ndarray, box_in_crop) -> np.ndarray:
    """Foreground mask (0/255) for the object whose box sits inside ``crop``."""
    x1, y1, x2, y2 = box_in_crop
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        mask[:] = 0

    box_area = (x2 - x1) * (y2 - y1)
    if mask[y1:y2, x1:x2].sum() / 255 < MIN_FOREGROUND * box_area:
        mask[:] = 0
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.ellipse(mask, centre, ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)

    # Keep the largest blob, close small holes, grow by a pixel so the object's
    # own outline survives the paste.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(mask, kernel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE.parent / 'data' / 'patches')
    args = parser.parse_args()

    tiles = []
    for frame in frame_numbers():
        image, annotations = load_sample(frame)
        for annotation in annotations:
            x1, y1, x2, y2 = annotation['bbox']
            if (x1 <= EDGE_GAP or y1 <= EDGE_GAP
                    or x2 >= IMAGE_WIDTH - EDGE_GAP or y2 >= IMAGE_HEIGHT - EDGE_GAP):
                continue
            cx1, cy1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
            cx2, cy2 = min(IMAGE_WIDTH, x2 + MARGIN), min(IMAGE_HEIGHT, y2 + MARGIN)
            crop = image[cy1:cy2, cx1:cx2]
            mask = object_mask(crop, (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1))

            # Store exactly the labelled box: the pasted box is then the label.
            bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
            patch = np.dstack([crop[by1:by2, bx1:bx2], mask[by1:by2, bx1:bx2]])

            folder = args.out / annotation['object_id']
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f'f{frame:06d}.png'), patch)

            preview = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_NEAREST)
            cut = cv2.resize(cv2.bitwise_and(crop, crop, mask=mask), (96, 96), interpolation=cv2.INTER_NEAREST)
            tiles.append(np.hstack([preview, cut]))

    written = sorted(args.out.glob('*/*.png'))
    per_class = {}
    for path in written:
        per_class[path.parent.name] = per_class.get(path.parent.name, 0) + 1
    for name, count in sorted(per_class.items()):
        print(f'{name:16s} {count:3d} patches')
    print(f'{len(written)} patches in {args.out}')

    columns = 10
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    cv2.imwrite(str(args.out / '_sheet.png'), sheet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
