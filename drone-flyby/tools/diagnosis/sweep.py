"""Per-class box factor chosen against AP itself, averaged over several runs.

COCO AP for a category depends only on that category's detections, so scaling
EVERY class by s and reading each class's own AP gives the whole grid in one
pass per scale -- 12 independent sweeps for the price of one.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT = Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import score_offline as S
objects = json.loads(S.OBJECTS.read_text())['objects']
motion = S.fit_truth_motion(objects)[0]
bf = S.box_factors()
if bf: objects = S.loosen(objects, bf); motion = S.fit_truth_motion(objects)[0]
regions = S.ignore_regions(('confirmed',)); frames = list(range(1,250))
def rescale(b,s):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*s/2,(b[3]-b[1])*s/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
SC=[0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10]
runs=sys.argv[1:]
acc=collections.defaultdict(lambda: collections.defaultdict(list))
base=collections.defaultdict(list)
for run in runs:
    P=S.from_recording(run)
    _,bc,_=S.score(P,frames,objects,motion,regions)
    for k,v in bc.items(): base[k].append(v)
    for s in SC:
        Q=collections.defaultdict(list)
        for fr,v in P.items():
            for name,conf,box in v:
                Q[fr].append((name,conf,rescale(np.asarray(box,float),s)))
        _,c,_=S.score(Q,frames,objects,motion,regions)
        for k,v in c.items(): acc[k][s].append(v)
print(f'{"class":16s} ' + ' '.join(f'{s:5.2f}' for s in SC) + '   best   now    gain')
best={}; gain=0; nowsum=0
for k in sorted(acc):
    m=[np.mean(acc[k][s]) for s in SC]
    b=SC[int(np.argmax(m))]; nb=np.mean(base[k])
    # only move off 1.0 if it really beats it
    if max(m) < nb + 0.004: b, bm = 1.00, nb
    else: bm = max(m)
    best[k]=b; gain += bm-nb; nowsum += nb
    print(f'{k:16s} ' + ' '.join(f'{x:5.2f}' for x in m) + f'   {b:.2f}  {nb:.3f}  {bm-nb:+.3f}')
print(f'\nmacro now {nowsum/len(acc):.3f} -> {(nowsum+gain)/len(acc):.3f}   ({gain/len(acc):+.3f})')
print('\nDRONE_BOX_GROW=' + ','.join(f'{k}={1.3*v:.2f}' for k,v in sorted(best.items())))
