"""Separate width and height factors, on top of the per-class isotropic best.

An isotropic factor cannot fix aspect ratio: a wing-span class and a mast class
need opposite corrections. flyby supports this already via DRONE_BOX_GROW_WH=1.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT = Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import score_offline as S
ISO={'hangar':1.00,'helicopter':0.90,'jammer':0.60,'jet_plane':1.05,
     'large_launcher':0.90,'large_tower':0.95,'mine_roller':0.85,
     'small_launcher':1.00,'small_plane':0.80,'small_tower':0.80,
     'spacecraft':1.00,'tank':1.00}
objects = json.loads(S.OBJECTS.read_text())['objects']
motion = S.fit_truth_motion(objects)[0]
bf = S.box_factors()
if bf: objects = S.loosen(objects, bf); motion = S.fit_truth_motion(objects)[0]
regions = S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
G=[0.80,0.90,1.00,1.10,1.20]
runs=sys.argv[1:]
acc=collections.defaultdict(lambda: collections.defaultdict(list)); base=collections.defaultdict(list)
for run in runs:
    P=S.from_recording(run)
    B=collections.defaultdict(list)
    for fr,v in P.items():
        for n,c,b in v: B[fr].append((n,c,rs(np.asarray(b,float),ISO.get(n,1.0),ISO.get(n,1.0))))
    _,bc,_=S.score(B,frames,objects,motion,regions)
    for k,v in bc.items(): base[k].append(v)
    for sw in G:
        for sh in G:
            Q=collections.defaultdict(list)
            for fr,v in B.items():
                for n,c,b in v: Q[fr].append((n,c,rs(b,sw,sh)))
            _,c2,_=S.score(Q,frames,objects,motion,regions)
            for k,v in c2.items(): acc[k][(sw,sh)].append(v)
print(f'{"class":16s} {"now":>6s} {"best":>6s} {"gain":>7s}   w    h')
tot_n=tot_b=0
out={}
for k in sorted(acc):
    nb=np.mean(base[k]); m={p:np.mean(v) for p,v in acc[k].items()}
    p=max(m,key=m.get)
    if m[p] < nb+0.004: p,bm=(1.0,1.0),nb
    else: bm=m[p]
    out[k]=(ISO.get(k,1.0)*p[0], ISO.get(k,1.0)*p[1])
    tot_n+=nb; tot_b+=bm
    print(f'{k:16s} {nb:6.3f} {bm:6.3f} {bm-nb:+7.3f}   {p[0]:.2f} {p[1]:.2f}')
print(f'\nmacro {tot_n/len(acc):.3f} -> {tot_b/len(acc):.3f}  ({(tot_b-tot_n)/len(acc):+.3f}) on top of iso')
print('\nDRONE_BOX_GROW_WH=1 DRONE_BOX_GROW_CAP=1.6 DRONE_BOX_GROW=' +
      ','.join(f'{k}={1.3*w:.2f}x{1.3*h:.2f}' for k,(w,h) in sorted(out.items())))
