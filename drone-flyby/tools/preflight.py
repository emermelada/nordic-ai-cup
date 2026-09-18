"""Check a live drone-flyby service the way the evaluator will, before an attempt.

    python tools/preflight.py                                   # localhost:9053
    python tools/preflight.py --url https://xyz.trycloudflare.com/predict

Answers the three questions that have actually cost us runs:

* is it serving the configuration we think it is (which weights, how many)?
* does it answer a real recorded frame, with real boxes?
* does it answer inside the 333 ms frame budget, over the public URL?

A 68/249 run and a 248/249 run looked identical from the outside; run this
against the submitted URL, not localhost, or it measures the wrong link.
"""

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDINGS = ROOT / 'data' / 'recordings'
BUDGET_MS = 333


def post(url: str, body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://localhost:9053/predict',
                        help='The /predict URL exactly as it will be submitted.')
    parser.add_argument('--frames', type=int, default=40, help='Recorded frames to replay.')
    parser.add_argument('--expect-models', type=int, default=2,
                        help='How many sets of weights should be loaded (2 = the pair).')
    args = parser.parse_args()

    failures = []
    status_url = args.url.rsplit('/predict', 1)[0] + '/api'
    try:
        with urllib.request.urlopen(status_url, timeout=15) as response:
            status = json.loads(response.read())
    except (urllib.error.URLError, OSError) as exc:
        print(f'FAIL  cannot reach {status_url}: {exc}')
        return 1

    print(f'service   {status.get("model")}')
    print(f'          alt={status.get("model_alt")} device={status.get("device")} '
          f'camera={status.get("camera")} recording={status.get("recording")}')
    loaded = status.get('models_loaded')
    if loaded is None:
        failures.append('/api has no models_loaded: the service predates this check, rebuild it')
    elif loaded != args.expect_models:
        failures.append(f'models_loaded is {loaded}, expected {args.expect_models}')
    print(f'          models_loaded={loaded} (expected {args.expect_models})')

    # The longest recording, not whichever sorts last: most runs here are short
    # probes, and replaying one frame measures nothing but the cold start.
    runs = [p for p in RECORDINGS.iterdir() if p.is_dir()] if RECORDINGS.is_dir() else []
    runs.sort(key=lambda p: len(list(p.glob('*.json'))))
    metas = sorted(runs[-1].glob('*.json'))[:args.frames] if runs else []
    if len(metas) < 2:
        print('\nno recording long enough to replay; checked the status endpoint only')
        return 1 if failures else 0

    print(f'\nreplaying {len(metas)} frames from {runs[-1].name} against {args.url}')
    latencies, empty = [], 0
    tag = f'preflight-{int(time.time())}'
    for index, meta_path in enumerate(metas):
        meta = json.loads(meta_path.read_text())
        meta.pop('response', None)
        # A throwaway sequence_id: this must not disturb the state of a real run.
        meta['sequence_id'] = tag
        meta['request_id'] = f'{tag}:{index}'
        meta['view']['view_id'] = f'{tag}:{index}'
        meta['view']['image'] = base64.b64encode(meta_path.with_suffix('.png').read_bytes()).decode()
        try:
            payload, elapsed = post(args.url, json.dumps(meta).encode(), timeout=30)
        except (urllib.error.URLError, OSError) as exc:
            failures.append(f'frame {meta["frame"]} failed: {exc}')
            break
        latencies.append(elapsed)
        if not payload.get('annotations'):
            empty += 1

    # The first request pays for a cold path (533 ms against a 25 ms median when
    # this was measured). That is one frame at the start of a run, so it is
    # reported and then left out of the budget check.
    if len(latencies) > 1:
        print(f'first request {latencies[0]:.0f} ms (cold start, excluded below)')
        latencies = latencies[1:]
    if latencies:
        ordered = sorted(latencies)
        p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        over = sum(1 for value in latencies if value > BUDGET_MS)
        print(f'round trip mean {statistics.mean(latencies):.0f} ms / '
              f'median {statistics.median(latencies):.0f} / p90 {p90:.0f} / max {max(latencies):.0f}')
        print(f'over {BUDGET_MS} ms budget: {over}/{len(latencies)}   frames with no boxes: {empty}')
        # The first request after an idle service pays for a cold path; that is
        # one frame at the start of a run, not a reason to fail the check.
        if statistics.median(latencies) > BUDGET_MS:
            failures.append(f'median round trip {statistics.median(latencies):.0f} ms exceeds the budget')
        if over > len(latencies) * 0.1:
            failures.append(f'{over} of {len(latencies)} frames over budget')
        if empty == len(latencies):
            failures.append('every frame came back with no boxes')

    print()
    for failure in failures:
        print(f'FAIL  {failure}')
    if not failures:
        print('PASS  serving the expected models, answering inside the budget')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
