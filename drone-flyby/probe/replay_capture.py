"""POST a captured flight back through a live service and keep the answers.

The captures under data/captures are the real transmitted views of a real
validation flight, so this replays the genuine article against any config
without spending a validation. Camera commands are ignored: the frames are
fixed, which is the point -- two configs see byte-identical input.
"""
import argparse, base64, json, sys, time
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture')
    p.add_argument('--url', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()

    records = sorted(Path(a.capture).glob('*.json'))
    if a.limit:
        records = records[:a.limit]
    seq = f'replay{int(time.time()*1000)%10**9}'
    answers, t0 = {}, time.time()
    for i, rec in enumerate(records):
        meta = json.loads(rec.read_text())
        image = rec.with_suffix('.png').read_bytes()
        view = dict(meta['view'])
        view['image'] = base64.b64encode(image).decode()
        body = {
            'sequence_id': seq,
            'frame': meta['frame'],
            'frame_index': meta['frame_index'],
            'request_id': f"{seq}:{meta['frame_index']}",
            'frame_interval_ms': meta['frame_interval_ms'],
            'response_timeout_ms': meta['response_timeout_ms'],
            'original_width': meta['original_width'],
            'original_height': meta['original_height'],
            'view': view,
            'camera_constraints': meta['camera_constraints'],
            'camera_command_feedback': meta['camera_command_feedback'],
        }
        req = urllib.request.Request(
            a.url, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        answers[str(meta['frame'])] = out.get('annotations', [])
        if i % 50 == 0:
            print(f'  {i}/{len(records)}', flush=True)
    Path(a.out).write_text(json.dumps(answers))
    n = [len(v) for v in answers.values()]
    print(f'{a.out}: {len(answers)} frames, {sum(n)} boxes, '
          f'mean {sum(n)/len(n):.1f}/frame, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    sys.exit(main())
