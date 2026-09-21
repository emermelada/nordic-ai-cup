"""Calibrate an offline truth file against runs whose REAL score is known.

    python tools/calibrate_truth.py training/scene_objects.json

Scores each recorded run in data/runs_20260919 whose real validation score is
known, and reports MAE, bias and rank correlation. What matters for choosing
between configurations is CORRELATION, not absolute error: a truth file with a
constant offset still ranks correctly, one with no correlation is useless
however close its numbers look.
"""

import sys, json, pathlib, collections
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import numpy as np
import score_offline as so
from score_offline import fit_truth_motion, known_boxes, loosen, ignore_regions, ignore_boxes
from faster_coco_eval import COCO, COCOeval_faster
from dtos import OBJECT_CLASSES

TRUTH = sys.argv[1]
RUNS = pathlib.Path('data/runs_20260919')
# run prefix -> real validation score, 249/249 complete runs only
KNOWN = {
 'e7e81c83':0.5065,'0cc4a253':0.4983,'3833e905':0.5052,'e265a637':0.5017,
 '9fe1efd9':0.5187,'e51e8e30':0.5234,'0743b57e':0.4994,'58c35cec':0.5278,
 '50dcacf1':0.5076,'b5544ad3':0.5339,'f4314c5b':0.5320,'ade2dfb0':0.5118,
}
objects = json.loads(open(TRUTH).read())['objects']
objects = loosen(objects, {c:{'w':1.3,'h':1.3} for c in {o['class'] for o in objects}})
motion,_ = fit_truth_motion(objects)
regions = ignore_regions()
cats={n:i+1 for i,n in enumerate(OBJECT_CLASSES)}
frames=list(range(1,250)); imgs={f:i+1 for i,f in enumerate(frames)}

def score_run(d):
    preds=collections.defaultdict(list)
    for p in sorted(d.glob('*.json')):
        m=json.loads(p.read_text())
        for a in (m.get('response') or {}).get('annotations') or []:
            b=np.array(a['bbox'],float)*[3840,2160,3840,2160]
            preds[m['frame']].append((a['object_id'],a['confidence'],b))
    anns=[];aid=1;present=collections.Counter()
    for f in frames:
        for cls,b in known_boxes(objects,f,motion):
            anns.append({'id':aid,'image_id':imgs[f],'category_id':cats[cls],
                'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'area':(b[2]-b[0])*(b[3]-b[1]),'iscrowd':0});aid+=1
            present[cls]+=1
        for b in ignore_boxes(regions,f,motion):
            for cid in set(cats.values()):
                anns.append({'id':aid,'image_id':imgs[f],'category_id':cid,
                    'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'area':(b[2]-b[0])*(b[3]-b[1]),'iscrowd':1});aid+=1
    dts=[{'image_id':imgs[f],'category_id':cats[n],'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':c}
         for f in frames for n,c,b in preds.get(f,[]) if b[2]>b[0] and b[3]>b[1]]
    if not dts: return 0.0
    gt=COCO({'info':{},'licenses':[],'images':[{'id':v,'file_name':f'{k}.png','width':3840,'height':2160} for k,v in imgs.items()],
        'categories':[{'id':v,'name':k,'supercategory':'o'} for k,v in cats.items()],'annotations':anns})
    dt=gt.loadRes(dts);ev=COCOeval_faster(gt,dt,'bbox')
    ev.params.imgIds=list(imgs.values())
    kl=[c for c in OBJECT_CLASSES if c in present]
    ev.params.catIds=[cats[c] for c in kl];ev.params.iouThrs=np.array([0.5])
    ev.evaluate();ev.accumulate()
    P=ev.eval['precision'];tot=0
    for i,c in enumerate(kl):
        cp=P[0,:,i,0,-1];v=cp[cp>-1];tot+=float(np.mean(v)) if v.size else 0.0
    return tot/len(kl)

rows=[]
for pre,real in KNOWN.items():
    ds=[d for d in RUNS.iterdir() if d.is_dir() and d.name.startswith(pre)]
    if not ds: continue
    off=score_run(ds[0])
    rows.append((pre,real,off))
rows.sort(key=lambda r:-r[1])
print(f'truth: {TRUTH}  ({len(objects)} objects)')
print(f"{'run':10s} {'real':>7s} {'offline':>8s} {'err':>7s}")
for pre,real,off in rows: print(f'{pre:10s} {real:7.4f} {off:8.4f} {off-real:+7.4f}')
r=np.array([x[1] for x in rows]); o=np.array([x[2] for x in rows])
def rank(x): return np.argsort(np.argsort(x)).astype(float)
print()
print(f'n={len(rows)}  MAE {np.mean(np.abs(o-r)):.4f}  bias {np.mean(o-r):+.4f}')
print(f'Pearson  {np.corrcoef(r,o)[0,1]:+.3f}')
print(f'Spearman {np.corrcoef(rank(r),rank(o))[0,1]:+.3f}')
