"""Correct the measured per-class centre offset, by one global fraction.

Offsets were measured on 7e44/76be/b554. One parameter (how much of the
measured offset to undo) keeps this honest; 0.0 is the current behaviour.
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
OFF={'spacecraft':(0.046,-0.215),'mine_roller':(0.026,0.100),'large_launcher':(0.009,-0.094),
     'small_launcher':(-0.003,-0.064),'helicopter':(-0.042,-0.063),'jammer':(0.093,0.059),
     'small_plane':(-0.030,0.031),'tank':(-0.036,-0.025),'hangar':(-0.011,0.015),
     'small_tower':(0.006,0.010),'large_tower':(-0.001,-0.006),'jet_plane':(-0.015,-0.001)}
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def build(b,n,k):
    i,hh=F.get(n,(1.0,1.0)); sw,sh=i,i*hh
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw,(b[3]-b[1])*sh
    dx,dy=OFF.get(n,(0.0,0.0))
    cx-=k*dx*w; cy-=k*dy*h
    return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
K=[0.0,0.5,1.0,1.5,2.0]
tot=collections.defaultdict(list); per=collections.defaultdict(lambda: collections.defaultdict(list))
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for k in K:
        Q=collections.defaultdict(list)
        for fr,v in P.items():
            for n,c,b in v: Q[fr].append((n,c,build(np.asarray(b,float),n,k)))
        o,c2,_=S.score(Q,frames,objects,motion,regions)
        tot[k].append(o)
        for cls,val in c2.items(): per[cls][k].append(val)
print(f'{"shift k":>9s} ' + ' '.join(f'{k:6.1f}' for k in K))
print(f'{"macro":>9s} ' + ' '.join(f'{np.mean(tot[k]):6.3f}' for k in K))
print()
for cls in sorted(per):
    print(f'{cls:16s} ' + ' '.join(f'{np.mean(per[cls][k]):6.3f}' for k in K))
