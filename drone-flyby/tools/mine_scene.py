"""Mine a complete truth file from the rebuilt 4K scene, by temporal consistency.

    python tools/mine_scene.py --device cuda --out training/scene_objects.json

The existing truth (training/validation_objects.json, 32 objects) is badly
incomplete -- it was mined by v2 at conf >= 0.5 from Level-1 views -- and that
makes every offline PRECISION and AP number measurement error rather than score
(HANDOVER, 17:45). This builds a better one, two ways at once:

1. **Detect at native resolution.** The scene frames are full 3840x2160, so a
   34 px object is tiled at 1920x1080 and run at imgsz 1920 -- no downsampling
   at all, against the 2x of a Level-1 view. This is the best detection the
   stack can produce on this flight.
2. **Require temporal consistency.** A rendered asset is static ground: it
   appears in ~33 consecutive frames, drifting exactly with the ground motion.
   A spurious detection does not. Linking detections frame to frame through the
   fitted motion and keeping only chains of >= MIN_FRAMES sightings is an
   independent filter on precision -- it does not share a failure mode with the
   confidence threshold or with cross-model agreement, which is what earlier
   mining rounds used.

Output matches validation_objects.json so every existing tool reads it.
"""
import argparse, collections, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

TILES = [(0, 0), (1920, 0), (0, 1080), (1920, 1080), (960, 540)]   # 4 quadrants + centre
TILE_W, TILE_H = 1920, 1080


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--imgsz', type=int, default=1920)
    ap.add_argument('--conf', type=float, default=0.25, help='per-detection floor')
    ap.add_argument('--min-frames', type=int, default=6, help='sightings before an object is believed')
    ap.add_argument('--out', default='training/scene_objects.json')
    ap.add_argument('--frames', type=int, default=249)
    args = ap.parse_args()

    os.environ['DRONE_MODEL'] = str(ROOT / 'models/drone-yolo11m-v8.pt')
    os.environ['DRONE_MODEL_ALT'] = str(ROOT / 'models/drone-yolo11s-v6.pt')
    os.environ['DRONE_IMGSZ'] = f'{args.imgsz},{args.imgsz}'
    os.environ['DRONE_DEVICE'] = args.device
    os.environ['DRONE_DET_CONF'] = str(min(0.01, args.conf))

    import cv2, numpy as np, flyby
    import score_offline as so
    from dtos import OBJECT_CLASSES
    flyby.load_model()

    truth = json.loads((ROOT / 'training' / 'validation_objects.json').read_text())
    motion, _ = so.fit_truth_motion(truth['objects'])

    def step(box):
        a, b, c, d, e, f = motion
        x1, y1, x2, y2 = box
        nx1 = x1 + a + b*x1 + c*y1; ny1 = y1 + d + e*nx1 + f*y1
        nx2 = x2 + a + b*x2 + c*y2; ny2 = y2 + d + e*nx2 + f*y2
        return np.array([nx1, ny1, nx2, ny2])

    coverage = json.loads((ROOT / 'data' / 'scene' / 'coverage.json').read_text())
    chains = []                       # {'class','obs':[(frame,box)],'box','frame','misses'}
    done = []
    for frame in range(1, args.frames + 1):
        if coverage.get(str(frame), {}).get('any', 0) < 0.999:
            continue
        path = ROOT / 'data' / 'scene' / f'frame_{frame:06d}.png'
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        found = []
        for ox, oy in TILES:
            tile = image[oy:oy+TILE_H, ox:ox+TILE_W]
            for which in (0, 1):
                xyxy, prob = flyby.raw_detections(tile, which)
                for b, p in zip(xyxy, prob):
                    if p.max() < args.conf:
                        continue
                    found.append((OBJECT_CLASSES[int(p.argmax())], float(p.max()),
                                  np.array([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy])))
        # de-duplicate tiles/models: keep the most confident box per cluster
        found.sort(key=lambda d: -d[1])
        keep = []
        for name, cf, box in found:
            if all(iou(box, k[2]) < 0.5 for k in keep):
                keep.append((name, cf, box))
        # link into chains through the ground motion
        for ch in chains:
            ch['box'] = step(ch['box']); ch['misses'] += 1
        for name, cf, box in keep:
            best, score = None, 0.3
            for ch in chains:
                v = iou(ch['box'], box)
                if v > score:
                    best, score = ch, v
            if best is None:
                chains.append({'class': collections.Counter({name: cf}), 'obs': [(frame, box)],
                               'box': box.copy(), 'misses': 0})
            else:
                best['class'][name] += cf
                best['obs'].append((frame, box))
                best['box'] = box.copy(); best['misses'] = 0
        alive = []
        for ch in chains:
            if ch['misses'] > 4 or ch['box'][3] < 0 or ch['box'][1] > 2160:
                done.append(ch)
            else:
                alive.append(ch)
        chains = alive
        if frame % 25 == 0:
            print(f'  frame {frame}: {len(chains)} live chains, {len(done)} closed', flush=True)
    done.extend(chains)

    objects = []
    for i, ch in enumerate(done, 1):
        if len(ch['obs']) < args.min_frames:
            continue
        name = ch['class'].most_common(1)[0][0]
        objects.append({'id': 10000 + i, 'class': name,
                        'observations': [{'frame': f, 'box': [round(float(v), 1) for v in b]}
                                         for f, b in ch['obs']]})
    by_class = collections.Counter(o['class'] for o in objects)
    out = {'note': (f'Mined from data/scene at native resolution (tiles of {TILE_W}x{TILE_H} at '
                    f'imgsz {args.imgsz}, v8+v6), kept only chains of >= {args.min_frames} '
                    f'consecutive sightings linked by the fitted ground motion. '
                    f'Temporal consistency is the precision filter, not confidence.'),
           'objects': objects}
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print(f'\n{len(objects)} objects kept of {len(done)} chains -> {args.out}')
    for c, n in by_class.most_common():
        print(f'  {c:<16} {n}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
