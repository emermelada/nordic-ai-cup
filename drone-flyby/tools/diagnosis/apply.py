"""Rescale stored answers per class and rescore, against the same run unchanged.

The stored boxes already carry BOX_GROW=1.3, so a factor here of f means
shipping DRONE_BOX_GROW=<class>=1.3*f.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT = Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import score_offline as S
BEST = {'hangar':0.70,'helicopter':0.90,'jammer':0.65,'jet_plane':1.00,
        'large_launcher':0.88,'large_tower':0.88,'mine_roller':0.82,
        'small_launcher':1.00,'small_plane':0.80,'small_tower':0.82,
        'spacecraft':0.97,'tank':1.00}
objects = json.loads(S.OBJECTS.read_text())['objects']
motion = S.fit_truth_motion(objects)[0]
f = S.box_factors()
if f: objects = S.loosen(objects, f); motion = S.fit_truth_motion(objects)[0]
regions = S.ignore_regions(('confirmed',))
frames = list(range(1,250))
def rescale(b,s):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*s/2,(b[3]-b[1])*s/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
for run in sys.argv[1:]:
    P = S.from_recording(run)
    base,bc,_ = S.score(P, frames, objects, motion, regions)
    Q = collections.defaultdict(list)
    for fr,v in P.items():
        for name,conf,box in v:
            Q[fr].append((name, conf, rescale(np.asarray(box,float), BEST.get(name,1.0))))
    new,nc,_ = S.score(Q, frames, objects, motion, regions)
    print(f'\n== {run[:12]}   {base:.3f} -> {new:.3f}   ({new-base:+.3f})')
    for k in sorted(nc, key=lambda k: -(nc[k]-bc.get(k,0))):
        d = nc[k]-bc.get(k,0)
        if abs(d) > 0.005: print(f'   {k:16s} {bc.get(k,0):.3f} -> {nc[k]:.3f}  {d:+.3f}')
