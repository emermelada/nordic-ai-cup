"""Keep only the alias edges that do not hurt under EITHER truth file."""
import sys, json, collections, os
from pathlib import Path
import numpy as np
ROOT=Path('/home/emer/Proyectos/nordic-ai-cup/drone-flyby')
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import score_offline as S
EDGES=[('hangar','jet_plane'),('mine_roller','large_launcher'),('mine_roller','medium_launcher'),
 ('tank','large_launcher'),('tank','mine_roller'),('tank','large_tower'),
 ('medium_plane','large_launcher'),('medium_plane','small_plane'),('medium_plane','small_tower'),
 ('small_plane','large_tower'),('small_plane','small_tower'),('small_tower','small_plane'),
 ('spacecraft','small_plane'),('helicopter','ta-ta'),('helicopter','tank'),('helicopter','hangar'),
 ('small_launcher','ta-ta'),('small_launcher','medium_launcher'),
 ('large_tower','tank'),('large_launcher','mine_roller')]
A=0.30
def setup(which):
    o=json.loads((ROOT/'training'/which).read_text())['objects']
    bf=S.box_factors()
    if bf: o=S.loosen(o,bf)
    return o,S.fit_truth_motion(o)[0],S.ignore_regions(('confirmed',))
frames=list(range(1,250)); runs=sys.argv[1:]
P={r:S.from_recording(r) for r in runs}
CTX={w:setup(w) for w in ('scene_objects.json','validation_objects.json')}
def run(edges, which):
    objects,motion,regions=CTX[which]
    m=collections.defaultdict(list)
    for a,b in edges: m[a].append(b)
    tot=[]
    for r in runs:
        Q=collections.defaultdict(list)
        for fr,v in P[r].items():
            out=list(v)
            for n,c,bx in v:
                for alt in m.get(n,()): out.append((alt,c*A,bx))
            Q[fr]=out
        o,_,_=S.score(Q,frames,objects,motion,regions); tot.append(o)
    return np.mean(tot)
b_s, b_v = run([],'scene_objects.json'), run([],'validation_objects.json')
print(f'baseline   scene {b_s:.3f}   validation {b_v:.3f}\n')
print(f'{"edge":34s} {"scene":>16s} {"validation":>16s}  keep?')
keep=[]
for e in EDGES:
    ds=run([e],'scene_objects.json')-b_s
    dv=run([e],'validation_objects.json')-b_v
    ok = ds>=-0.001 and dv>=-0.001 and (ds>0.001 or dv>0.001)
    if ok: keep.append(e)
    print(f'{e[0]+" -> "+e[1]:34s} {ds:+16.4f} {dv:+16.4f}  {"KEEP" if ok else "drop"}')
print(f'\nkept {len(keep)} edges: {keep}')
ks,kv=run(keep,'scene_objects.json'),run(keep,'validation_objects.json')
print(f'combined   scene {b_s:.3f} -> {ks:.3f} ({ks-b_s:+.3f})   validation {b_v:.3f} -> {kv:.3f} ({kv-b_v:+.3f})')
