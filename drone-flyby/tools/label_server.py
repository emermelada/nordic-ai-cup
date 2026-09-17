"""A small local page for labelling objects in recorded validation frames.

    python tools/label_server.py                 # then open http://localhost:8765
    python tools/label_server.py --every 3       # label every 3rd recorded frame

Draw a box by dragging, then press the class key shown in the reference list
(or click the class). Every frame must be marked done, even with no objects,
so its unlabelled ground counts as "nothing here" for training. Labels are
saved as you go to data/labels/<sequence>/<frame>.json, in view pixels and in
source pixels.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402

RECORDINGS = ROOT / 'data' / 'recordings'
LABELS = ROOT / 'data' / 'labels'
PATCHES = ROOT / 'data' / 'patches'
KEYS = '1234567890qwerty'

app = FastAPI()
frames = []


def reference_images():
    refs = {}
    for name in OBJECT_CLASSES:
        files = sorted((PATCHES / name).glob('*.png'))
        if not files:
            continue
        # The biggest cut-out of each class is the clearest example.
        image = max((cv2.imread(str(f)) for f in files), key=lambda im: im.shape[0] * im.shape[1])
        scale = 72 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        refs[name] = base64.b64encode(cv2.imencode('.png', image)[1]).decode()
    return refs


@app.get('/', response_class=HTMLResponse)
def page():
    return PAGE.replace('__CLASSES__', json.dumps(list(OBJECT_CLASSES))) \
               .replace('__KEYS__', json.dumps(KEYS)) \
               .replace('__REFS__', json.dumps(reference_images()))


@app.get('/api/frames')
def list_frames():
    out = []
    for sequence, stem in frames:
        label = LABELS / sequence / f'{stem}.json'
        meta = json.loads((RECORDINGS / sequence / f'{stem}.json').read_text())
        out.append({
            'sequence': sequence, 'stem': stem, 'level': meta['view']['resolution_level'],
            'done': label.exists(),
            'boxes': json.loads(label.read_text())['boxes'] if label.exists() else [],
        })
    return out


@app.get('/img/{sequence}/{stem}.png')
def image(sequence: str, stem: str):
    path = RECORDINGS / sequence / f'{stem}.png'
    if not path.is_file() or path.parent.parent != RECORDINGS:
        raise HTTPException(404)
    return FileResponse(path)


@app.post('/api/labels/{sequence}/{stem}')
def save(sequence: str, stem: str, body: dict):
    meta_path = RECORDINGS / sequence / f'{stem}.json'
    if not meta_path.is_file() or meta_path.parent.parent != RECORDINGS:
        raise HTTPException(404)
    meta = json.loads(meta_path.read_text())
    rx1, ry1, rx2, ry2 = meta['view']['source_region_xyxy']
    scale = (rx2 - rx1) / 960
    boxes = []
    for box in body.get('boxes', []):
        if box['cls'] not in OBJECT_CLASSES:
            raise HTTPException(400, f'unknown class {box["cls"]}')
        x1, y1, x2, y2 = (float(v) for v in box['view'])
        boxes.append({
            'cls': box['cls'],
            'view': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            'source': [round(rx1 + x1 * scale, 1), round(ry1 + y1 * scale, 1),
                       round(rx1 + x2 * scale, 1), round(ry1 + y2 * scale, 1)],
        })
    folder = LABELS / sequence
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{stem}.json').write_text(json.dumps({
        'sequence': sequence, 'stem': stem, 'frame': meta['frame'],
        'level': meta['view']['resolution_level'],
        'source_region_xyxy': meta['view']['source_region_xyxy'], 'boxes': boxes,
    }, indent=1))
    return {'saved': len(boxes)}


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Drone labels</title>
<style>
 body{margin:0;font:14px system-ui,sans-serif;background:#1d1f24;color:#e8e8e8;display:flex;height:100vh}
 #main{flex:1;display:flex;flex-direction:column;align-items:center;padding:8px;min-width:0}
 #bar{display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap}
 button{background:#3a3f4b;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
 button.primary{background:#2f7d4f} button:hover{filter:brightness(1.2)}
 #wrap{position:relative;max-width:100%}
 canvas{max-width:100%;cursor:crosshair;display:block}
 #side{width:250px;overflow:auto;background:#262930;padding:8px}
 .ref{display:flex;align-items:center;gap:8px;padding:3px;border-radius:6px;cursor:pointer}
 .ref:hover{background:#353945} .ref.sel{background:#2f5d8a}
 .ref img{width:48px;height:48px;object-fit:contain;background:#111;border-radius:4px}
 kbd{background:#444;border-radius:4px;padding:1px 5px;font-size:12px}
 #boxes div{display:flex;justify-content:space-between;padding:2px 0}
 .hint{color:#aaa;font-size:12px;margin-top:6px;max-width:960px}
</style></head><body>
<div id="main">
 <div id="bar">
  <button onclick="go(-1)">◀ Prev <kbd>A</kbd></button>
  <span id="pos"></span>
  <button class="primary" onclick="finish()">Done, next ▶ <kbd>Space</kbd></button>
  <button onclick="undo()">Undo box <kbd>Z</kbd></button>
  <button onclick="zoom()">Zoom <kbd>X</kbd></button>
  <span id="progress"></span>
 </div>
 <div id="wrap"><canvas id="c" width="960" height="540"></canvas></div>
 <div class="hint">Drag to draw a tight box around each object, then press its key (or click it on the right).
 Mark every frame <b>Done</b>, also when there is nothing in it. Unsure what it is? Pick the closest class; skip only
 what you cannot tell from the ground.</div>
 <div id="boxes"></div>
</div>
<div id="side"><b>Classes</b><div id="refs"></div></div>
<script>
const CLASSES=__CLASSES__, KEYS=__KEYS__, REFS=__REFS__;
let frames=[], i=0, boxes=[], draft=null, pending=null, img=new Image(), scale=1;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
function refs(){
 const el=document.getElementById('refs'); el.innerHTML='';
 CLASSES.forEach((c,k)=>{const d=document.createElement('div'); d.className='ref'; d.id='ref_'+c;
  d.innerHTML=(REFS[c]?`<img src="data:image/png;base64,${REFS[c]}">`:'<span style="width:48px"></span>')+`<span><kbd>${KEYS[k]}</kbd> ${c}</span>`;
  d.onclick=()=>assign(c); el.appendChild(d);});
}
async function load(){ frames=await (await fetch('/api/frames')).json(); i=Math.max(0,frames.findIndex(f=>!f.done)); show(); }
function show(){
 const f=frames[i]; boxes=f.boxes.map(b=>({cls:b.cls,view:b.view})); pending=null; draft=null;
 img.onload=draw; img.src=`/img/${f.sequence}/${f.stem}.png`;
 document.getElementById('pos').textContent=`${i+1}/${frames.length}  ${f.stem}  L${f.level} ${f.done?'✓':''}`;
 document.getElementById('progress').textContent=`${frames.filter(x=>x.done).length} done`;
}
function draw(){
 ctx.drawImage(img,0,0,960,540); ctx.font='12px sans-serif';
 for(const b of boxes){ const [x1,y1,x2,y2]=b.view; ctx.strokeStyle='#ff3b3b'; ctx.lineWidth=1; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.fillStyle='#ff3b3b'; ctx.fillText(b.cls,x1,y1-3);}
 const d=draft||pending; if(d){ctx.strokeStyle='#ffd400'; ctx.setLineDash([4,3]); ctx.strokeRect(d[0],d[1],d[2]-d[0],d[3]-d[1]); ctx.setLineDash([]);}
 document.getElementById('boxes').innerHTML=boxes.map((b,k)=>`<div><span>${b.cls}</span><button onclick="del(${k})">✕</button></div>`).join('')
  + (pending?'<div style="color:#ffd400">new box: press a class key</div>':'');
}
function pt(e){const r=cv.getBoundingClientRect(); return [(e.clientX-r.left)*960/r.width,(e.clientY-r.top)*540/r.height];}
let start=null;
cv.onmousedown=e=>{start=pt(e); pending=null;};
cv.onmousemove=e=>{ if(!start) return; const p=pt(e); draft=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])]; draw(); };
cv.onmouseup=e=>{ if(draft && draft[2]-draft[0]>=2 && draft[3]-draft[1]>=2) pending=draft; start=null; draft=null; draw(); };
function assign(c){ if(!pending) return; boxes.push({cls:c,view:pending.map(v=>Math.round(v*10)/10)}); pending=null; save(false); draw(); }
function del(k){ boxes.splice(k,1); save(false); draw(); }
function undo(){ if(pending){pending=null;} else boxes.pop(); save(false); draw(); }
async function save(done){
 const f=frames[i]; if(!done && !f.done && boxes.length===0) return;
 await fetch(`/api/labels/${f.sequence}/${f.stem}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({boxes})});
 f.boxes=boxes.map(b=>({cls:b.cls,view:b.view})); f.done=true;
}
async function finish(){ await save(true); go(1); }
function go(d){ i=Math.min(frames.length-1,Math.max(0,i+d)); show(); }
function zoom(){ const w=document.getElementById('wrap'); w.style.width = w.style.width ? '' : '1800px'; }
document.onkeydown=e=>{
 if(e.key===' '){e.preventDefault(); finish(); return;}
 if(e.key==='a'||e.key==='A'){go(-1); return;}
 if(e.key==='z'||e.key==='Z'){undo(); return;}
 if(e.key==='x'||e.key==='X'){zoom(); return;}
 const k=KEYS.indexOf(e.key.toLowerCase()); if(k>=0 && k<CLASSES.length) assign(CLASSES[k]);
};
refs(); load();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--every', type=int, default=4, help='Label every Nth recorded frame.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('sequences', nargs='*', help='Recording folders to label (default: all with 20+ frames).')
    args = parser.parse_args()

    sequences = args.sequences or sorted(
        p.name for p in RECORDINGS.iterdir() if p.is_dir() and len(list(p.glob('*.png'))) >= 20
    )
    for sequence in sequences:
        stems = sorted(p.stem for p in (RECORDINGS / sequence).glob('*.json'))
        frames.extend((sequence, stem) for stem in stems[::args.every])
    print(f'{len(frames)} frames to label from {len(sequences)} recordings: open http://localhost:{args.port}')
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
    return 0


if __name__ == '__main__':
    sys.exit(main())
