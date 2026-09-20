"""One global weight factor on top of the per-class isotropic best.

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
# (isotropic, height multiplier) -- the config already validated held-out
F={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
   'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
   'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
   'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
H=[0.70,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.20,1.30]
runs=sys.argv[1:]
tot=collections.defaultdict(list); per=collections.defaultdict(lambda: collections.defaultdict(list))
for run in runs:
    P=S.from_recording(run)
    B=collections.defaultdict(list)
    for fr,v in P.items():
        for n,c,b in v:
            i,h=F.get(n,(1.0,1.0)); B[fr].append((n,c,rs(np.asarray(b,float),i,i*h)))
    for h in H:
        Q=collections.defaultdict(list)
        for fr,v in B.items():
            for n,c,b in v: Q[fr].append((n,c,rs(b,h,1.0)))
        o,c2,_=S.score(Q,frames,objects,motion,regions)
        tot[h].append(o)
        for k,v in c2.items(): per[k][h].append(v)
print(f'{"global w":>9s} ' + ' '.join(f'{h:6.2f}' for h in H))
print(f'{"macro":>9s} ' + ' '.join(f'{np.mean(tot[h]):6.3f}' for h in H))
print()
for k in sorted(per):
    print(f'{k:16s} ' + ' '.join(f'{np.mean(per[k][h]):6.3f}' for h in H))
b=max(H,key=lambda h:np.mean(tot[h]))
print(f'\nbest global weight {b:.2f}  macro {np.mean(tot[b]):.3f}  (h=1.00 -> {np.mean(tot[1.00]):.3f})')
