"""Reproduce research diagnostics from saved public-training predictions; no inference."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONTROL = ROOT / 'runs/resume-20260919-0325/faithful-results/stageb-on-own-resume-control.json'
INPUTS = ROOT / 'tools/gpu_eval/inputs_base.json'


def tiou(gold, prediction):
    if not prediction or prediction[0] is None or prediction[1] <= prediction[0]:
        return 0.0
    intersection = max(0.0, min(gold[1], prediction[1]) - max(gold[0], prediction[0]))
    return intersection / (max(gold[1], prediction[1]) - min(gold[0], prediction[0]))


def words_in_span(words, span):
    if not span:
        return ''
    return ' '.join(w['word'].strip() for w in words
                    if w['end'] > span[0] + 1e-6 and w['start'] < span[1] - 1e-6)


def main():
    result = json.loads(CONTROL.read_text())
    inputs = {item['id']: item for item in json.loads(INPUTS.read_text())}
    rows = result['rows']
    assert len(rows) == 390 and len({r['question_id'] for r in rows}) == 390
    assert {r['conversation'] for r in rows} == set(inputs)
    positives = [r for r in rows if r['label']]
    assert len(positives) == 195
    accuracy = sum(r['answer'] == bool(r['label']) for r in rows) / len(rows)
    scores = [tiou(r['gold'], r['candidate']) if r['answer'] else 0.0 for r in positives]
    raw = 0.4 * accuracy + 0.6 * mean(scores)
    bins = [("zero", lambda x: x == 0), ("0_to_0.5", lambda x: 0 < x < 0.5),
            ("0.5_to_0.9", lambda x: 0.5 <= x < 0.9), ("at_least_0.9", lambda x: x >= 0.9)]
    summary = {
        'scope': 'Saved public-training control; not the latest hidden-platform validation or new inference.',
        'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (CONTROL, INPUTS)],
        'conversations': len(inputs), 'questions': len(rows), 'positives': len(positives),
        'accuracy': accuracy, 'mean_tiou': mean(scores), 'raw': raw,
        'span_bins': {name: {'count': sum(test(v) for v in scores),
                            'remaining_raw_loss': 0.6 * sum(1-v for v in scores if test(v)) / len(scores)}
                      for name, test in bins},
        'hypothetical_perfect_zero_repairs_only': raw + 0.6 * sum(v == 0 for v in scores) / len(scores),
        'required_tiou': {str(target): {str(acc): (target - 0.4 * acc) / 0.6 for acc in (1.0, 0.99)}
                          for target in (0.85, 0.88, 0.90, 0.95)},
    }
    (OUT / 'saved_control_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    detailed = []
    for row, score in zip(positives, scores):
        item = inputs[row['conversation']]
        question = next(r['question'] for r in item['rows'] if r['question_id'] == row['question_id'])
        detailed.append({'conversation': row['conversation'], 'question_id': row['question_id'],
                         'question': question, 'tiou': score, 'gold_start': row['gold'][0],
                         'gold_end': row['gold'][1],
                         'predicted_start': row['candidate'][0] if row['candidate'] else None,
                         'predicted_end': row['candidate'][1] if row['candidate'] else None,
                         'gold_text': words_in_span(item['words'], row['gold']),
                         'predicted_text': words_in_span(item['words'], row['candidate'])})
    detailed.sort(key=lambda row: (row['tiou'], row['conversation'], row['question_id']))
    with (OUT / 'positive_span_audit.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
