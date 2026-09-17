"""Offline: turn the dumped LLM outputs into spans with different alignment strategies and score tIoU."""
from pathlib import Path
import json, re, sys, statistics as st
from rapidfuzz import fuzz
DUMP=sys.argv[1]
dump=json.load(open(DUMP))
def iou(a,b):
    i=max(0,min(a[1],b[1])-max(a[0],b[0])); u=max(a[1],b[1])-min(a[0],b[0]); return i/u if u>0 else 0
def clean(t): return re.sub(r"[^a-z0-9 ]", "", t.lower().replace("'", "")).split()
def parse(raw):
    m=re.search(r'\{.*\}', raw, flags=re.S)
    try: return {int(x['q']):x for x in json.loads(m.group(0))['results'] if isinstance(x,dict)}
    except Exception: return {}
def unit_span(units, ids): return (units[min(ids)]['start'], units[max(ids)]['end'])
def fuzzy_span(words, quote, max_extra=3):
    """Best window of consecutive words matching the quote (token-level fuzzy)."""
    q=clean(quote); 
    if not q: return None
    toks=[clean(w['word']) for w in words]; flat=[]; idx=[]
    for wi,ts in enumerate(toks):
        for t in ts: flat.append(t); idx.append(wi)
    n=len(flat); L=len(q); best=(-1,None)
    for i in range(n):
        for L2 in range(max(1,L-max_extra), L+max_extra+1):
            j=i+L2
            if j>n: break
            s=fuzz.ratio(' '.join(flat[i:j]), ' '.join(q))
            if s>best[0]: best=(s,(idx[i],idx[j-1]))
    if best[1] is None or best[0]<70: return None
    a,b=best[1]; return (words[a]['start'], words[b]['end'])
def run(strategy, pad=(0.0,0.0)):
    tious=[]; zeros=0; nyes=0
    for tid,d in dump.items():
        res=parse(d['raw']); units=d['units']
        allwords=[w for u in units for w in u['words']]
        for i,q in enumerate(d['questions']):
            if q['label']!='1': continue
            g=(float(q['gs']),float(q['ge'])); x=res.get(i+1,{}); ans=str(x.get('answer','')).lower().startswith('y')
            span=None
            if ans:
                nyes+=1
                ids=[u for u in (x.get('units') or []) if isinstance(u,int) and 0<=u<len(units)]
                quote=x.get('quote') or ''
                if strategy=='unit' and ids: span=unit_span(units,ids)
                elif strategy=='fuzzy_units':
                    if ids:
                        lo,hi=max(0,min(ids)-1),min(len(units)-1,max(ids)+1)
                        span=fuzzy_span([w for u in units[lo:hi+1] for w in u['words']], quote) or unit_span(units,ids)
                    else: span=fuzzy_span(allwords, quote)
                elif strategy=='fuzzy_global':
                    span=fuzzy_span(allwords, quote) or (unit_span(units,ids) if ids else None)
                if span: span=(span[0]+pad[0], span[1]+pad[1])
            v=iou(g,span) if span else 0.0; tious.append(v); zeros+= (v==0)
    return st.mean(tious), zeros, nyes, tious
for strat in ['unit','fuzzy_units','fuzzy_global']:
    m,z,ny,t=run(strat); print(f'{strat:<14} mean tIoU {m:.3f}   zero-IoU {z}/195   answered-yes {ny}   quartiles {[round(x,2) for x in st.quantiles(t,n=4)]}')
print('--- padding sweep on fuzzy_global ---')
best=None
for a in [-0.2,-0.1,0.0,0.1]:
    for b in [-0.1,0.0,0.1,0.2,0.3]:
        m,_,_,_=run('fuzzy_global',(a,b))
        if best is None or m>best[0]: best=(m,a,b)
print(f'best pad start {best[1]:+.1f} end {best[2]:+.1f} -> {best[0]:.3f}')
# quote quality: how long are quotes vs gold, and does the quote text overlap gold words?
ql=[]; 
for tid,d in dump.items():
    res=parse(d['raw'])
    for i,q in enumerate(d['questions']):
        x=res.get(i+1,{}); 
        if q['label']=='1' and x.get('quote'): ql.append(len(clean(x['quote'])))
print(f'quote length words: mean {st.mean(ql):.1f} median {st.median(ql)}')
if '--worst' in sys.argv:
    print('--- worst cases (fuzzy_global) ---')
    for tid,d in dump.items():
        res=parse(d['raw']); units=d['units']; allwords=[w for u in units for w in u['words']]
        for i,q in enumerate(d['questions']):
            if q['label']!='1': continue
            x=res.get(i+1,{}); g=(float(q['gs']),float(q['ge']))
            span=fuzzy_span(allwords, x.get('quote') or '')
            v=iou(g,span) if span else 0
            if v<0.3: print(f"{tid} tIoU={v:.2f} gold={g} pred={span} ans={x.get('answer')} Q: {q['q']} | QUOTE: {x.get('quote')}")
