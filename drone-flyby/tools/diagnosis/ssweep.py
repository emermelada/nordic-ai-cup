"""Per-class box factor against AP, using scene_objects.json (the +0.894 truth).

COCO AP for a category depends only on that category's detections, so scaling
every class by s and reading each class's own AP gives all twelve sweeps in one
pass per scale.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
objects=json.loads((ROOT/'training'/'scene_objects.json').read_text())['objects']
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf)
motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
SC=[0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30]
runs=sys.argv[1:]
acc=collections.defaultdict(lambda: collections.defaultdict(list))
for run in runs:
    P=S.from_recording(run)
    for s in SC:
        Q=collections.defaultdict(list)
        for fr,v in P.items():
            for n,c,b in v: Q[fr].append((n,c,rs(np.asarray(b,float),s,s)))
        _,c2,_=S.score(Q,frames,objects,motion,regions)
        for k,v in c2.items(): acc[k][s].append(v)
print(f'{"class":16s} ' + ' '.join(f'{s:6.2f}' for s in SC) + '   best   now   gain')
gain=0; now=0
best={}
for k in sorted(acc):
    m=[np.mean(acc[k][s]) for s in SC]
    nb=m[SC.index(1.00)]
    b=SC[int(np.argmax(m))]
    if max(m) < nb+0.004: b,bm=1.00,nb
    else: bm=max(m)
    best[k]=b; gain+=bm-nb; now+=nb
    print(f'{k:16s} ' + ' '.join(f'{x:6.3f}' for x in m) + f'   {b:.2f}  {nb:.3f}  {bm-nb:+.3f}')
n=len(acc)
print(f'\nmacro {now/n:.3f} -> {(now+gain)/n:.3f}  ({gain/n:+.3f})')
print('iso:', {k:round(1.3*v,2) for k,v in best.items() if v!=1.0})
