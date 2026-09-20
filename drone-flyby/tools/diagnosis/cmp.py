"""Four box-geometry policies, on runs none of them were fitted to.

Stored answers already carry BOX_GROW=1.3 capped at 1.3, so a policy wanting
factor X is simulated by rescaling the stored box by X/1.3 per axis.

  A current    flat 1.3, cap 1.3          (what we ship today)
  B helsinki-iso   official iso, cap 1.3  (arm.sh's own default)
  C helsinki-WH    official w/h, cap 2.2  (real labels, nothing fitted)
  D fitted-WH      mine, cap 1.9
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
conv=json.loads(Path('training/box_convention.json').read_text())
FIT={'hangar':(1.30,1.30),'helicopter':(1.17,1.87),'jammer':(0.78,1.09),
 'jet_plane':(1.37,1.37),'large_launcher':(1.17,1.40),'large_tower':(1.23,1.23),
 'mine_roller':(1.10,1.66),'small_launcher':(1.30,1.30),'small_plane':(1.04,1.25),
 'small_tower':(1.04,1.14),'spacecraft':(1.30,1.30),'tank':(1.30,1.43)}
def pol(name):
    if name=='E': return {k:(1.0,1.0) for k in conv}, 1.0
    if name=='F': return {k:(1.15,1.15) for k in conv}, 1.15
    if name=='A': return {}, 1.3
    if name=='B': return {k:(v['iso'],v['iso']) for k,v in conv.items()}, 1.3
    if name=='C': return {k:(v['w'],v['h']) for k,v in conv.items()}, 2.2
    return FIT, 1.9
objects=json.loads(S.OBJECTS.read_text())['objects']
motion=S.fit_truth_motion(objects)[0]
bf=S.box_factors()
if bf: objects=S.loosen(objects,bf); motion=S.fit_truth_motion(objects)[0]
regions=S.ignore_regions(('confirmed',)); frames=list(range(1,250))
def rs(b,sw,sh):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*sw/2,(b[3]-b[1])*sh/2
    return np.array([cx-w,cy-h,cx+w,cy+h])
res=collections.defaultdict(list); per=collections.defaultdict(lambda: collections.defaultdict(list))
for run in sys.argv[1:]:
    P=S.from_recording(run)
    for name in 'AEFBCD':
        F,cap=pol(name)
        Q=collections.defaultdict(list)
        for fr,v in P.items():
            for n,c,b in v:
                w,h=F.get(n,(1.3,1.3)); w=min(w,cap); h=min(h,cap)
                Q[fr].append((n,c,rs(np.asarray(b,float),w/1.3,h/1.3)))
        o,c2,_=S.score(Q,frames,objects,motion,regions)
        res[name].append(o)
        for k,v in c2.items(): per[k][name].append(v)
print(f'{"policy":>22s}  ' + '  '.join(f'{n:>6s}' for n in 'ABCD'))
print(f'{"":>22s}  ' + '  '.join(f'{x:>6s}' for x in ['cur','hel-i','hel-WH','fit']))
print(f'{"MACRO":>22s}  ' + '  '.join(f'{np.mean(res[n]):6.3f}' for n in 'ABCD'))
print()
for k in sorted(per):
    print(f'{k:>22s}  ' + '  '.join(f'{np.mean(per[k][n]):6.3f}' for n in 'AEFBCD'))
