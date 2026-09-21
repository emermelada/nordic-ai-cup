import json, re, sys, statistics as st
sys.argv=[sys.argv[0], sys.argv[1] if len(sys.argv)>1 else 'llm_dump.json']
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from align_experiments import dump, iou, clean, parse, fuzzy_span, unit_span
STOP=set('the a an is was were be been are did does do has have had of to in on for at with by and or that this it its as from about any there their patient doctor will should would can could right correct didn t wasn isn also still'.split())
def qterms(q): return [t for t in clean(q) if t not in STOP]
def sentences(quote): return [s for s in re.split(r'(?<=[.?!])\s+', quote.strip()) if s]
ratios=[]; longer=shorter=0; rows=[]
for tid,d in dump.items():
    res=parse(d['raw']); units=d['units']; allwords=[w for u in units for w in u['words']]
    for i,q in enumerate(d['questions']):
        if q['label']!='1': continue
        x=res.get(i+1,{}); ans=str(x.get('answer','')).lower().startswith('y'); g=(float(q['gs']),float(q['ge']))
        quote=x.get('quote') or ''; ids=[u for u in (x.get('units') or []) if isinstance(u,int) and 0<=u<len(units)]
        base=fuzzy_span(allwords, quote) if ans else None
        rows.append((tid,q,x,ans,g,quote,ids,base,units,allwords))
        if base and iou(g,base)>0:
            r=(base[1]-base[0])/(g[1]-g[0]); ratios.append(r); longer+= r>1.15; shorter+= r<0.85
print(f'overlapping cases: {len(ratios)}  pred/gold length ratio median {st.median(ratios):.2f}; too long (>1.15x): {longer}, too short (<0.85x): {shorter}')
def score(fn):
    t=[]; z=0
    for (tid,q,x,ans,g,quote,ids,base,units,allwords) in rows:
        sp=fn(q,x,ans,quote,ids,base,units,allwords) if ans else None
        v=iou(g,sp) if sp else 0.0; t.append(v); z+=(v==0)
    return st.mean(t), z
def rule_base(q,x,ans,quote,ids,base,units,allwords): return base
def rule_pad(q,x,ans,quote,ids,base,units,allwords): return (base[0]+0.1, base[1]) if base else None
def rule_trim(q,x,ans,quote,ids,base,units,allwords):
    ss=sentences(quote)
    if len(ss)>1:
        qt=set(qterms(q['q'])); best=max(ss, key=lambda s: (len(qt & set(clean(s))), -ss.index(s)))
        sp=fuzzy_span(allwords, best)
        if sp: return sp
    return base
def rule_short_extend(q,x,ans,quote,ids,base,units,allwords):
    if base and len(clean(quote))<4 and ids:
        lo=max(0,min(ids)-1); return (units[lo]['start'], base[1])
    return base
def rule_all(q,x,ans,quote,ids,base,units,allwords):
    sp=rule_trim(q,x,ans,quote,ids,base,units,allwords)
    if sp and len(clean(quote))<4 and ids:
        lo=max(0,min(ids)-1); sp=(units[lo]['start'], sp[1])
    return (sp[0]+0.1, sp[1]) if sp else None
for name,fn in [('fuzzy_global',rule_base),('+pad start 0.1',rule_pad),('+trim multi-sentence',rule_trim),('+extend short reply',rule_short_extend),('all three',rule_all)]:
    m,z=score(fn); print(f'{name:<24} mean tIoU {m:.3f}  zero {z}')
nz=[iou(g,b) for (tid,q,x,ans,g,quote,ids,b,units,allwords) in rows if b and iou(g,b)>0]
print(f'mean tIoU over the {len(nz)} overlapping cases only: {st.mean(nz):.3f}')
