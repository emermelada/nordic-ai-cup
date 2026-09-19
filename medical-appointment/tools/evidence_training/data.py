"""Prepare auditable word-span targets and conversation-disjoint folds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / 'tools/gpu_eval/inputs_base.json'
DEFAULT_CONTROL = ROOT / 'runs/resume-20260919-0325/faithful-results/stageb-on-own-resume-control.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tiou(gold, prediction):
    if not gold or not prediction:
        return 0.0
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in prediction):
        return 0.0
    if prediction[1] <= prediction[0]:
        return 0.0
    intersection = max(0.0, min(gold[1], prediction[1]) - max(gold[0], prediction[0]))
    union = max(gold[1], prediction[1]) - min(gold[0], prediction[0])
    return intersection / union if union > 0 else 0.0


def render_words(words):
    """Return deterministic text and character ranges retaining occurrence identity."""
    chunks, ranges, position = [], [], 0
    for word in words:
        text = word['word'].strip()
        if not text:
            raise ValueError('Empty ASR word cannot be mapped to a text span')
        if chunks:
            position += 1
        ranges.append([position, position + len(text)])
        chunks.append(text)
        position += len(text)
    return ' '.join(chunks), ranges


def best_word_span(words, gold):
    """Maximize actual temporal IoU over contiguous, monotonic ASR word intervals."""
    best, best_score = None, -1.0
    for start, word in enumerate(words):
        if word['start'] >= gold[1]:
            break
        for end in range(start, len(words)):
            if words[end]['end'] <= gold[0]:
                continue
            score = tiou(gold, [word['start'], words[end]['end']])
            if score > best_score:
                best, best_score = [start, end], score
            # Later ends only enlarge the union for this fixed start.
            if words[end]['end'] >= gold[1]:
                break
    if best is None or best_score <= 0:
        raise ValueError(f'Gold evidence has no overlap with ASR words: {gold}')
    return best, best_score


def validate_records(records):
    seen, contexts = set(), {}
    for row in records:
        qid = row['id']
        if qid in seen:
            raise ValueError(f'Duplicate question ID: {qid}')
        seen.add(qid)
        if not isinstance(row['label'], bool) or not isinstance(row['baseline_answer'], bool):
            raise ValueError(f'Labels and baseline answers must be booleans: {qid}')
        text, ranges = render_words(row['words'])
        if text != row['context'] or ranges != row['word_chars']:
            raise ValueError(f'Character/word mapping mismatch: {qid}')
        for key in ('start', 'end'):
            times = [w[key] for w in row['words']]
            if not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in times):
                raise ValueError(f'Invalid ASR timestamps: {qid}')
            if times != sorted(times):
                raise ValueError(f'Nonmonotonic ASR timestamps: {qid}')
        if any(w['end'] < w['start'] for w in row['words']):
            raise ValueError(f'Inverted ASR word: {qid}')
        previous = contexts.setdefault(row['conversation'], row['context'])
        if previous != row['context']:
            raise ValueError(f'Inconsistent context for conversation: {qid}')
        if row['label']:
            start, end = row['gold_words']
            if not 0 <= start <= end < len(ranges):
                raise ValueError(f'Invalid gold word span: {qid}')
            if row['gold_chars'] != [ranges[start][0], ranges[end][1]]:
                raise ValueError(f'Invalid gold character span: {qid}')
            if not row['gold'] or tiou(row['gold'], row['gold']) != 1:
                raise ValueError(f'Invalid gold times: {qid}')
        elif any(row[k] is not None for k in ('gold', 'gold_words', 'gold_chars')):
            raise ValueError(f'Negative question has gold evidence: {qid}')
    # Copies of a context must never become separate training/test conversations.
    if len(set(contexts.values())) != len(contexts):
        raise ValueError('Duplicate transcripts across conversation IDs; group them before training')


def fold_plan(records, folds=3, seed=17, dev_conversations=2):
    ids = sorted({r['conversation'] for r in records},
                 key=lambda c: hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest())
    if folds < 2 or len(ids) < folds * 2 or dev_conversations < 1:
        raise ValueError('Need at least two folds and an independent inner development set')
    assignment = {cid: i % folds for i, cid in enumerate(ids)}
    plans = []
    for fold in range(folds):
        test = [cid for cid in ids if assignment[cid] == fold]
        remaining = [cid for cid in ids if assignment[cid] != fold]
        if len(remaining) <= dev_conversations:
            raise ValueError('Not enough training conversations after reserving inner development')
        dev = remaining[:dev_conversations]
        train = remaining[dev_conversations:]
        plans.append({'fold': fold, 'train': train, 'dev': dev, 'test': test})
    return plans


def validate_bundle(bundle):
    if bundle.get('schema_version') != 1:
        raise ValueError('Unsupported training-data schema')
    validate_records(bundle['records'])
    ids = {r['conversation'] for r in bundle['records']}
    tested, fold_ids = [], []
    for plan in bundle['folds']:
        train, dev, test = (set(plan[k]) for k in ('train', 'dev', 'test'))
        if not train or not dev or not test or train & dev or train & test or dev & test:
            raise ValueError('Training, development and test conversations must be nonempty and disjoint')
        if train | dev | test != ids:
            raise ValueError('Fold coverage differs from the dataset')
        for key in ('train', 'dev', 'test'):
            if len(set(plan[key])) != len(plan[key]):
                raise ValueError('Duplicate conversation within a split')
        tested.extend(plan['test'])
        fold_ids.append(plan['fold'])
    if set(tested) != ids or len(tested) != len(ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError('Each conversation must appear in exactly one outer test fold')


def prepare(inputs=DEFAULT_INPUTS, control=DEFAULT_CONTROL, folds=3, seed=17):
    items = json.loads(Path(inputs).read_text())
    baseline_rows = json.loads(Path(control).read_text())['rows']
    baseline = {r['question_id']: r for r in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError('Duplicate baseline question IDs')
    records = []
    for item in items:
        words = item['words']
        context, word_chars = render_words(words)
        # Fail before constructing labels if the timestamp grid is malformed.
        for key in ('start', 'end'):
            values = [w[key] for w in words]
            if values != sorted(values) or not all(math.isfinite(x) for x in values):
                raise ValueError(f'Invalid {key} grid: {item["id"]}')
        for question in item['rows']:
            qid, label = question['question_id'], question['label'] == '1'
            original = baseline[qid]
            gold = [float(question['evidence_start']), float(question['evidence_end'])] if label else None
            if original['conversation'] != item['id'] or bool(original['label']) != label:
                raise ValueError(f'Baseline identity mismatch: {qid}')
            if gold != original['gold']:
                raise ValueError(f'Baseline gold mismatch: {qid}')
            endpoints, oracle = best_word_span(words, gold) if label else (None, None)
            records.append({
                'id': qid, 'conversation': item['id'], 'question': question['question'],
                'context': context, 'words': words, 'word_chars': word_chars, 'label': label,
                'gold': gold, 'gold_words': endpoints,
                'gold_chars': [word_chars[endpoints[0]][0], word_chars[endpoints[1]][1]] if label else None,
                'mapping_tiou': oracle, 'baseline_answer': original['answer'],
                'baseline_span': original['candidate'],
            })
    validate_records(records)
    if set(baseline) != {r['id'] for r in records}:
        raise ValueError('Baseline and dataset question coverage differ')
    positives = [r for r in records if r['label']]
    return {
        'schema_version': 1, 'scope': 'Exploratory public-training folds; not hidden-platform validation',
        'sources': [{'path': str(Path(p).resolve()), 'sha256': digest(p)} for p in (inputs, control)],
        'seed': seed, 'folds': fold_plan(records, folds, seed), 'records': records,
        'summary': {'conversations': len(items), 'questions': len(records), 'positives': len(positives),
                    'mean_mapping_tiou': mean(r['mapping_tiou'] for r in positives),
                    'min_mapping_tiou': min(r['mapping_tiou'] for r in positives),
                    'max_gold_words': max(r['gold_words'][1] - r['gold_words'][0] + 1 for r in positives)},
    }


def score(records, predictions):
    if set(predictions) != {r['id'] for r in records}:
        raise ValueError('Prediction coverage must exactly match evaluation records')
    rows = []
    for record in records:
        span = predictions[record['id']]
        answer = record['baseline_answer']
        rows.append({'id': record['id'], 'conversation': record['conversation'], 'label': record['label'],
                     'answer': answer, 'gold': record['gold'], 'candidate': span,
                     'tiou': tiou(record['gold'], span) if answer and record['label'] else 0.0,
                     'baseline_tiou': tiou(record['gold'], record['baseline_span'])
                     if answer and record['label'] else 0.0})
    positives = [r for r in rows if r['label']]
    if not positives:
        raise ValueError('Evaluation requires at least one positive')
    accuracy = mean(r['answer'] == r['label'] for r in rows)
    overlap, baseline = mean(r['tiou'] for r in positives), mean(r['baseline_tiou'] for r in positives)
    conversation_gains = {
        cid: mean(r['tiou'] - r['baseline_tiou'] for r in positives if r['conversation'] == cid)
        for cid in sorted({r['conversation'] for r in positives})}
    return {'questions': len(rows), 'positives': len(positives), 'accuracy': accuracy,
            'mean_tiou': overlap, 'raw': .4 * accuracy + .6 * overlap,
            'baseline_tiou': baseline, 'baseline_raw': .4 * accuracy + .6 * baseline,
            'raw_gain': .6 * (overlap - baseline), 'zero_overlap': sum(r['tiou'] == 0 for r in positives),
            'conversation_tiou_gains': conversation_gains, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--control', type=Path, default=DEFAULT_CONTROL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve earlier artifacts')
    data = prepare(args.inputs, args.control, args.folds, args.seed)
    write_json(args.output, data)
    print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
