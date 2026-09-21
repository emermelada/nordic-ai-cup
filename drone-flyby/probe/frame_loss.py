"""How many frames did each run actually answer?

The grader sends 250 frames at 3 fps whatever we do, so a run that logs fewer
lost them on the wire. Each lost frame is worth about 0.009 of score, and the
evaluation is a single draw, so the loss RATE matters as much as the mean.
"""
import re, sys
runs, cur = [], None
pat = re.compile(r'^(\S+ \S+) INFO (?:\S+ )?frame (\d+) \(index (\d+)\)')
since = sys.argv[1] if len(sys.argv) > 1 else '2026-09-20 08:5'
for line in open('data/serve.log', errors='ignore'):
    m = pat.match(line)
    if not m:
        continue
    ts, fr, idx = m.group(1), int(m.group(2)), int(m.group(3))
    if idx == 0:
        cur = {'start': ts, 'n': 0, 'last': 0}
        runs.append(cur)
    if cur is not None:
        cur['n'] += 1
        cur['last'] = max(cur['last'], fr)
runs = [r for r in runs if r['start'] >= since and r['n'] > 5]
print('%-22s %7s %5s %5s' % ('run start', 'frames', 'last', 'lost'))
bad = 0
for r in runs:
    lost = r['last'] - r['n']
    if lost > 5:
        bad += 1
    print('%-22s %7d %5d %5d%s' % (r['start'], r['n'], r['last'], lost,
                                   '   <-- LOST FRAMES' if lost > 5 else ''))
print('\n%d of %d runs lost frames' % (bad, len(runs)))
