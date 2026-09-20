"""Does emitting fewer, better boxes raise AP under the correct truth?

Recall is good (0.7-0.88) but AP is 0.33-0.6, which means false positives are
outranking true ones. We emit ~64 boxes a frame against ~4 real objects.
Sweeps a per-class-per-frame cap and a confidence floor. Both are pure
post-processing on what we already emit.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
def setup(which):
    o=json.loads((ROOT/'training'/which).read_text())['objects']
    bf=S.box_factors()
    if bf: o=S.loosen(o,bf)
    return o,S.fit_truth_motion(o)[0],S.ignore_regions(('confirmed',))
frames=list(range(1,250)); runs=sys.argv[1:]
P={r:S.from_recording(r) for r in runs}
for which in ('scene_objects.json','validation_objects.json'):
    objects,motion,regions=setup(which)
    print(f'\n== {which}')
    print(f'{"top-K per class/frame":>24s}  ' + '  '.join(f'{k:>6}' for k in ('all',1,2,3,5,8)))
    row=[]
    for K in (None,1,2,3,5,8):
        tot=[]
        for r in runs:
            Q=collections.defaultdict(list)
            for fr,v in P[r].items():
                if K is None: Q[fr]=list(v)
                else:
                    by=collections.defaultdict(list)
                    for n,c,b in v: by[n].append((n,c,b))
                    for n,lst in by.items():
                        lst.sort(key=lambda p:-p[1]); Q[fr].extend(lst[:K])
            o,_,_=S.score(Q,frames,objects,motion,regions); tot.append(o)
        row.append(np.mean(tot))
    print(f'{"macro":>24s}  ' + '  '.join(f'{x:6.3f}' for x in row))
    print(f'{"conf floor":>24s}  ' + '  '.join(f'{k:>6}' for k in (0.0,0.05,0.10,0.20,0.35,0.50)))
    row=[]
    for F in (0.0,0.05,0.10,0.20,0.35,0.50):
        tot=[]
        for r in runs:
            Q=collections.defaultdict(list)
            for fr,v in P[r].items():
                Q[fr]=[p for p in v if p[1]>=F]
            o,_,_=S.score(Q,frames,objects,motion,regions); tot.append(o)
        row.append(np.mean(tot))
    print(f'{"macro":>24s}  ' + '  '.join(f'{x:6.3f}' for x in row))
