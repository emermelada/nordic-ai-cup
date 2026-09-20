"""Does the geometry gain survive the truth assumptions it was fitted under?

The factors were fitted against 'loose' truth (the official convention) with
unlabelled-object regions ignored. If the gain only exists under those two
choices it is an artefact of them. Re-score under all four combinations.
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
raw=json.loads(S.OBJECTS.read_text())['objects']
frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
runs=sys.argv[1:]
print(f'{"truth":>10s} {"ignore":>8s}   {"current":>8s} {"fixed":>8s} {"gain":>8s}')
for loose in (True, False):
    for ign in (True, False):
        objects=list(raw); motion=S.fit_truth_motion(objects)[0]
        if loose:
            bf=S.box_factors()
            if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
        regions=S.ignore_regions(('confirmed',)) if ign else ()
        a=[];b=[]
        for run in runs:
            P=S.from_recording(run)
            o0,_,_=S.score(P,frames,objects,motion,regions)
            Q=collections.defaultdict(list)
            for fr,v in P.items():
                for n,c,bx in v:
                    i,h=F.get(n,(1.0,1.0))
                    Q[fr].append((n,c,rs(np.asarray(bx,float),i,i*h)))
            o1,_,_=S.score(Q,frames,objects,motion,regions)
            a.append(o0); b.append(o1)
        print(f'{"loose" if loose else "tight":>10s} {"on" if ign else "off":>8s}   '
              f'{np.mean(a):8.3f} {np.mean(b):8.3f} {np.mean(b)-np.mean(a):+8.3f}')
