"""Evaluate pre-specified span-selection rules over out-of-fold ranker scores.

Reproduces every number in RESULTS.md's vote section from artifacts on disk:

    python -m tools.evidence_training.rules runs/rank-20260920/dump-s17.json

Each dump comes from `ranker.py --mode dump`, which scores each test conversation with the
fold checkpoint that never saw it. Passing several dumps averages their scores into one
voter, which is measurably the wrong ensemble here -- see RESULTS.md.

Every input is out of fold: the control is frozen, the extractor spans come from its own
fold CV, and each ranker score comes from the checkpoint whose training excluded that
conversation. The rule family is also chosen out of fold at the end, which is the only
number worth quoting when several families are on the table.
"""
import argparse, json, random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]


def tiou(gold, pred):
    if not gold or not pred or pred[1] <= pred[0]:
        return 0.0
    inter = max(0.0, min(gold[1], pred[1]) - max(gold[0], pred[0]))
    union = max(gold[1], pred[1]) - min(gold[0], pred[0])
    return inter / union if union > 0 else 0.0


def bootstrap(triples, draws=4000, seed=17):
    groups = {}
    for conversation, new, old in triples:
        groups.setdefault(conversation, []).append((new, old))
    names = sorted(groups)
    rng = random.Random(seed)
    gains = []
    for _ in range(draws):
        flat = [x for _ in names for x in groups[names[rng.randrange(len(names))]]]
        gains.append(.6 * (mean(a for a, _ in flat) - mean(b for _, b in flat)))
    gains.sort()
    return gains[int(.025 * draws)], gains[int(.975 * draws) - 1], sum(g > 0 for g in gains) / draws


def medoid(spans):
    spans = [s for s in spans if s]
    if len(spans) < 3:
        return spans[0] if spans else None
    return max(spans, key=lambda s: sum(tiou(s, other) for other in spans if other is not s))


def load(dumps):
    rows = {}
    for path in dumps:
        for qid, row in json.loads(Path(path).read_text())['rows'].items():
            slot = rows.setdefault(qid, {'conversation': row['conversation'], 'gold': row['gold'],
                                         'control': row['baseline_span'], 'scores': []})
            slot['scores'].append({(c[0], c[1]): (c[2], [c[3], c[4]]) for c in row['candidates']})
    return rows


def ensemble_scores(seed_maps):
    """Mean score per candidate; a candidate missing from a seed takes that seed's floor."""
    keys = set().union(*[set(m) for m in seed_maps])
    floors = [min((v[0] for v in m.values()), default=0.0) for m in seed_maps]
    out = {}
    for key in keys:
        total, span = 0.0, None
        for m, floor in zip(seed_maps, floors):
            if key in m:
                total += m[key][0]
                span = m[key][1]
            else:
                total += floor
        out[key] = (total / len(seed_maps), span)
    return out


def pick(scores, control, rule):
    pool = list(scores.values())
    if not pool:
        return control
    if rule == 'argmax':
        return max(pool)[1]
    if rule == 'overlap':
        local = [p for p in pool if p[1][0] < control[1] and control[0] < p[1][1]]
        return max(local)[1] if local else control
    if rule.startswith('window'):
        pad = float(rule[6:])
        local = [p for p in pool if p[1][0] >= control[0] - pad and p[1][1] <= control[1] + pad]
        return max(local)[1] if local else control
    raise ValueError(rule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dumps', nargs='+')
    parser.add_argument('--variant', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--extractor', default=str(
        ROOT / 'runs/evidence-external-20260919/gpu-artifacts/extractor-spans.json'))
    args = parser.parse_args()
    rows = load(args.dumps)
    extractor = {r['question_id']: (r.get('candidate') if r.get('answer') else None)
                 for r in json.loads(Path(args.extractor).read_text())['rows']}
    variants = {}
    for entry in args.variant:
        name, _, path = entry.partition('=')
        variants[name] = {r['question_id']: (r.get('candidate') if r.get('answer') else None)
                          for r in json.loads(Path(path).read_text())['rows']}
    order = sorted(rows)
    control = [rows[q]['control'] for q in order]
    gold = [rows[q]['gold'] for q in order]
    base = mean(tiou(g, c) for g, c in zip(gold, control))
    print('questions %d   seeds %d' % (len(order), len(rows[order[0]]['scores'])))
    print('control            tIoU %.6f  raw %.6f' % (base, .4 + .6 * base))
    ex = [extractor.get(q) for q in order]
    value = mean(tiou(g, e or c) for g, e, c in zip(gold, ex, control))
    print('extractor          tIoU %.6f  raw %.6f  %+.6f' % (value, .4 + .6 * value, .6 * (value - base)))

    combined = {q: ensemble_scores(rows[q]['scores']) for q in order}
    pooled = mean(max((tiou(rows[q]['gold'], v[1]) for v in combined[q].values()), default=0.0)
                  for q in order)
    print('dumped oracle      tIoU %.6f  raw %.6f' % (pooled, .4 + .6 * pooled))

    results = {}
    families = ['argmax', 'overlap', 'window5.0', 'medoid', 'gated'] + [
        'medoid+' + name for name in variants] + (
        ['medoid+' + '+'.join(variants)] if len(variants) > 1 else [])
    for seeds, tag in [(range(len(rows[order[0]]['scores'])), 'ensemble')]:
        for rule in families:
            if rule.startswith('medoid+'):
                names = rule[len('medoid+'):].split('+')
                values = [tiou(rows[q]['gold'],
                               medoid([rows[q]['control'], extractor.get(q),
                                       pick(combined[q], rows[q]['control'], 'argmax')]
                                      + [variants[n].get(q) for n in names]))
                          for q in order]
            elif rule == 'medoid':
                values = [tiou(rows[q]['gold'],
                               medoid([rows[q]['control'], extractor.get(q),
                                       pick(combined[q], rows[q]['control'], 'argmax')]))
                          for q in order]
            elif rule == 'gated':
                values = None  # filled in below, needs an out-of-fold threshold
            else:
                values = [tiou(rows[q]['gold'], pick(combined[q], rows[q]['control'], rule))
                          for q in order]
            if values is not None:
                results[rule] = values

    # gated: the ranker's argmax only when its score clears a threshold picked elsewhere
    grid = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 8]
    conversations = sorted({rows[q]['conversation'] for q in order})
    best_span = {q: max(combined[q].values()) if combined[q] else (0.0, rows[q]['control'])
                 for q in order}
    gated = {}
    for held in conversations:
        inner = [q for q in order if rows[q]['conversation'] != held]
        t = max(grid, key=lambda t: mean(
            tiou(rows[q]['gold'], best_span[q][1] if best_span[q][0] >= t else rows[q]['control'])
            for q in inner))
        for q in order:
            if rows[q]['conversation'] == held:
                gated[q] = (t, best_span[q][1] if best_span[q][0] >= t else rows[q]['control'])
    results['gated'] = [tiou(rows[q]['gold'], gated[q][1]) for q in order]

    for rule in families:
        values = results[rule]
        low, high, above = bootstrap([(rows[q]['conversation'], v, tiou(rows[q]['gold'], rows[q]['control']))
                                      for q, v in zip(order, values)])
        extra = ''
        if rule == 'gated':
            extra = '  thresholds %s' % sorted({t for t, _ in gated.values()})
        print('%-11s tIoU %.6f  raw %.6f  %+.6f  [%+.6f, %+.6f]  >0 %3.0f%%  zero %2d  high %3d%s'
              % (rule, mean(values), .4 + .6 * mean(values), .6 * (mean(values) - base), low, high,
                 100 * above, sum(1 for v in values if v == 0),
                 sum(1 for v in values if v >= .9), extra))

    nested, chosen = [], set()
    for held in conversations:
        inner = [i for i, q in enumerate(order) if rows[q]['conversation'] != held]
        best = max(families, key=lambda rule: mean(results[rule][i] for i in inner))
        chosen.add(best)
        nested += [(held, results[best][i], tiou(rows[q]['gold'], rows[q]['control']))
                   for i, q in enumerate(order) if rows[q]['conversation'] == held]
    value = mean(v for _, v, _ in nested)
    low, high, above = bootstrap(nested)
    print('\nnested family choice %s' % sorted(chosen))
    print('  tIoU %.6f  raw %.6f  %+.6f  [%+.6f, %+.6f]  >0 %.0f%%'
          % (value, .4 + .6 * value, .6 * (value - base), low, high, 100 * above))


if __name__ == '__main__':
    main()
