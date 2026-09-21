import json, sys, time, glob, os
from pathlib import Path
import mlx_whisper
AUDIO = str(Path(__file__).resolve().parents[2] / 'data' / 'audio')
OUT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'mlx-community/whisper-large-v3-turbo'
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f'{AUDIO}/*.mp3'), key=lambda p: int(p.rsplit('_',1)[1].split('.')[0]))
timings = {}
for i, f in enumerate(files):
    name = os.path.basename(f).replace('.mp3','')
    dst = f'{OUT}/{name}.json'
    if os.path.exists(dst):
        continue
    t0 = time.time()
    r = mlx_whisper.transcribe(f, path_or_hf_repo=MODEL, language='en', word_timestamps=True,
                               condition_on_previous_text=False, temperature=0.0)
    dt = time.time() - t0
    segs = [{'start': s['start'], 'end': s['end'], 'text': s['text'],
             'words': [{'word': w['word'], 'start': w['start'], 'end': w['end'], 'p': w.get('probability')} for w in s.get('words', [])]}
            for s in r['segments']]
    json.dump({'file': name, 'model': MODEL, 'seconds': dt, 'segments': segs, 'text': r['text']}, open(dst, 'w'), indent=1)
    timings[name] = dt
    print(f'{i+1}/{len(files)} {name}: {dt:.1f}s, {len(segs)} segments', flush=True)
json.dump(timings, open(f'{OUT}/_timings.json', 'w'), indent=1)
print('DONE')
