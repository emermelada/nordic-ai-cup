"""Re-score the geometry fix against scene_objects.json.

score_offline defaults to validation_objects.json (32 objects), which
calibrate_truth measured at Pearson -0.039 / Spearman -0.287 against real
scores. scene_objects.json (66 objects) measured +0.894 / +0.811. Everything
fitted so far used the anti-correlated one, so this re-checks the conclusion
against the truth that actually predicts reality.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
F={'hangar':(1.00,1.00),'helicopter':(0.90,1.60),'jammer':(0.60,1.40),
   'jet_plane':(1.05,1.00),'large_launcher':(0.90,1.20),'large_tower':(0.95,1.00),
   'mine_roller':(0.85,1.50),'small_launcher':(1.00,1.00),'small_plane':(0.80,1.20),
   'small_tower':(0.80,1.10),'spacecraft':(1.00,1.00),'tank':(1.00,1.10)}
which=sys.argv[1]; runs=sys.argv[2:]
raw=json.loads((ROOT/'training'/which).read_text())['objects']
frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
for ign in (True,False):
    objects=list(raw)
    bf=S.box_factors()
    if bf: objects=S.loosen(objects,bf)
    motion=S.fit_truth_motion(objects)[0]
    regions=S.ignore_regions(('confirmed',)) if ign else ()
    A=[];B=[];pa=collections.defaultdict(list);pb=collections.defaultdict(list)
    for run in runs:
        P=S.from_recording(run)
        o0,c0,inst=S.score(P,frames,objects,motion,regions)
        Q=collections.defaultdict(list)
        for fr,v in P.items():
            for n,c,b in v:
                i,h=F.get(n,(1.0,1.0)); Q[fr].append((n,c,rs(np.asarray(b,float),i,i*h)))
        o1,c1,_=S.score(Q,frames,objects,motion,regions)
        A.append(o0);B.append(o1)
        for k in set(c0)|set(c1): pa[k].append(c0.get(k,0)); pb[k].append(c1.get(k,0))
    print(f'\n{which}  ignore={"on" if ign else "off"}  ({inst} object-frames)')
    print(f'   current {np.mean(A):.3f}  ->  fixed {np.mean(B):.3f}   ({np.mean(B)-np.mean(A):+.3f})')
    if ign:
        for k in sorted(pb,key=lambda k:-(np.mean(pb[k])-np.mean(pa[k]))):
            print(f'      {k:16s} {np.mean(pa[k]):.3f} -> {np.mean(pb[k]):.3f}  {np.mean(pb[k])-np.mean(pa[k]):+.3f}')
