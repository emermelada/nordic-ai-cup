import json, sys, time, glob, os
from pathlib import Path
from omi_stt.mlx_runtime import _load_model, _generate_centered, _clear_mlx_cache
AUDIO = str(Path(__file__).resolve().parents[2] / 'data' / 'audio')
OUT = sys.argv[1]; MODEL = 'omi-health/omi-med-stt-v1-mlx-q8'
os.makedirs(OUT, exist_ok=True)
t0=time.time(); model=_load_model(MODEL); print(f'load {time.time()-t0:.1f}s', flush=True)
files = sorted(glob.glob(f'{AUDIO}/*.mp3'), key=lambda p: int(p.rsplit('_',1)[1].split('.')[0]))
timings={}
for i,f in enumerate(files):
    name=os.path.basename(f).replace('.mp3','')
    t0=time.time()
    try:
        r=_generate_centered(model, Path(f))
    finally:
        _clear_mlx_cache()
    dt=time.time()-t0
    segs=[]
    for s in r.sentences:
        words=[]
        for tok in s.tokens:
            txt=tok.text
            if txt.startswith(' ') or not words:
                words.append({'word': txt, 'start': tok.start, 'end': tok.end, 'p': getattr(tok,'confidence',None)})
            else:
                words[-1]['word']+=txt; words[-1]['end']=tok.end
        segs.append({'start': s.start, 'end': s.end, 'text': s.text, 'words': words})
    json.dump({'file':name,'model':MODEL,'seconds':dt,'segments':segs,'text':r.text}, open(f'{OUT}/{name}.json','w'), indent=1)
    timings[name]=dt
    print(f'{i+1}/{len(files)} {name}: {dt:.1f}s, {len(segs)} sentences', flush=True)
json.dump(timings, open(f'{OUT}/_timings.json','w'), indent=1)
print('DONE')
