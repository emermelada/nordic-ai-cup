"""Emit each box under its confused sibling class too, at reduced confidence.

AP is computed per class, so a duplicate emitted as the sibling is one more
low-ranked false positive in a class that already has thousands -- nearly free
-- while the copy that lands in the RIGHT class is a new true positive. This is
the "additions work, replacements fail" pattern that has paid five times here.

Confusion pairs are taken from the measured misnaming under scene truth:
large_launcher <-> mine_roller / tank, small_plane <-> small_tower,
ta-ta <-> helicopter.
"""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
# Built from the measured confusion under scene truth: for each truth class,
# the labels we wrongly give it. Key = what we emit, value = also emit as.
PAIRS={
 'hangar':['jet_plane'],
 'mine_roller':['large_launcher','medium_launcher'],
 'tank':['large_launcher','mine_roller','large_tower'],
 'medium_plane':['large_launcher','small_plane','small_tower'],
 'small_plane':['large_tower','small_tower'],
 'small_tower':['small_plane'],
 'spacecraft':['small_plane'],
 'helicopter':['ta-ta','tank','hangar'],
 'small_launcher':['ta-ta','medium_launcher'],
 'large_tower':['tank'],
 'large_launcher':['mine_roller'],
}
def setup(which):
    o=json.loads((ROOT/'training'/which).read_text())['objects']
    bf=S.box_factors()
    if bf: o=S.loosen(o,bf)
    return o,S.fit_truth_motion(o)[0],S.ignore_regions(('confirmed',))
frames=list(range(1,250)); runs=sys.argv[1:]
P={r:S.from_recording(r) for r in runs}
for which in ('scene_objects.json','validation_objects.json'):
    objects,motion,regions=setup(which)
    print(f'\n== {which}')
    base=None
    for A in (0.0,0.10,0.20,0.30,0.45,0.60):
        tot=[];per=collections.defaultdict(list)
        for r in runs:
            Q=collections.defaultdict(list)
            for fr,v in P[r].items():
                out=list(v)
                if A>0:
                    for n,c,b in v:
                        for alt in PAIRS.get(n,()):
                            out.append((alt, c*A, b))
                Q[fr]=out
            o,c2,_=S.score(Q,frames,objects,motion,regions); tot.append(o)
            for k,val in c2.items(): per[k].append(val)
        m=np.mean(tot)
        if A==0.0: base=m; pb={k:np.mean(v) for k,v in per.items()}
        print(f'   alias conf x{A:<5}  macro {m:.3f}  ({m-base:+.3f})')
        if A in (0.30,):
            for k in sorted(per,key=lambda k:-(np.mean(per[k])-pb.get(k,0))):
                d=np.mean(per[k])-pb.get(k,0)
                if abs(d)>0.008: print(f'        {k:16s} {pb.get(k,0):.3f} -> {np.mean(per[k]):.3f}  {d:+.3f}')
