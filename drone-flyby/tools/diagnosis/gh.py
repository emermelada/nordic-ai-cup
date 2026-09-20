"""One global height factor on top of the per-class isotropic best.

Nearly every class wanted h>w and most sat at the grid edge, which points at a
single systematic height bias rather than twelve independent quirks. One
parameter fitted over three runs is far harder to overfit than twenty-four.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
ISO={'hangar':1.00,'helicopter':0.90,'jammer':0.60,'jet_plane':1.05,
     'large_launcher':0.90,'large_tower':0.95,'mine_roller':0.85,
     'small_launcher':1.00,'small_plane':0.80,'small_tower':0.80,
     'spacecraft':1.00,'tank':1.00}
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
H=[1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.80,2.00,2.20]
runs=sys.argv[1:]
tot=collections.defaultdict(list); per=collections.defaultdict(lambda: collections.defaultdict(list))
for run in runs:
    P=S.from_recording(run)
    B=collections.defaultdict(list)
    for fr,v in P.items():
        for n,c,b in v: B[fr].append((n,c,rs(np.asarray(b,float),ISO.get(n,1.0),ISO.get(n,1.0))))
    for h in H:
        Q=collections.defaultdict(list)
        for fr,v in B.items():
            for n,c,b in v: Q[fr].append((n,c,rs(b,1.0,h)))
        o,c2,_=S.score(Q,frames,objects,motion,regions)
        tot[h].append(o)
        for k,v in c2.items(): per[k][h].append(v)
print(f'{"global h":>9s} ' + ' '.join(f'{h:6.2f}' for h in H))
print(f'{"macro":>9s} ' + ' '.join(f'{np.mean(tot[h]):6.3f}' for h in H))
print()
for k in sorted(per):
    print(f'{k:16s} ' + ' '.join(f'{np.mean(per[k][h]):6.3f}' for h in H))
b=max(H,key=lambda h:np.mean(tot[h]))
print(f'\nbest global height {b:.2f}  macro {np.mean(tot[b]):.3f}  (h=1.00 -> {np.mean(tot[1.00]):.3f})')
