"""Score medoid votes over several evidence producers against the current build.

The vote is parameter-free: among the candidate spans for a question, keep the one
with the greatest total temporal overlap with the others. Rules must be named on
the command line, because choosing the voter subset by its own score overfits 39
conversations (measured: +0.016 selected freely, -0.001 under nested validation).
"""

import argparse
import json
from pathlib import Path
import random
from statistics import mean

from .data import tiou


def load_rows(path):
    payload = json.loads(Path(path).read_text())
    rows = payload['rows'] if isinstance(payload, dict) else payload
    return {r['question_id']: r for r in rows}


def medoid(spans):
    return max(spans, key=lambda s: sum(tiou(s, other) for other in spans if other is not s))


def vote(row, names):
    spans = [row['build']] + [row[n] for n in names if row.get(n)]
    return medoid(spans) if len(spans) >= 3 else row['build']


def bootstrap(data, names, draws=4000, seed=17):
    by_conversation = {}
    for row in data:
        by_conversation.setdefault(row['conversation'], []).append(
            (tiou(row['gold'], vote(row, names)), tiou(row['gold'], row['build'])))
    conversations = sorted(by_conversation)
    rng = random.Random(seed)
    gains = []
    for _ in range(draws):
        flat = [x for _ in conversations
                for x in by_conversation[conversations[rng.randrange(len(conversations))]]]
        gains.append(.6 * (mean(a for a, _ in flat) - mean(b for _, b in flat)))
    gains.sort()
    return gains[int(.025 * draws)], gains[int(.975 * draws) - 1], sum(g > 0 for g in gains) / draws


def build_rows(build, voters):
    """One row per question the build answered yes, with every voter's span."""
    data = []
    for qid, row in build.items():
        if row.get('label') != 1 or not row.get('answer') or not row.get('candidate'):
            continue
        item = {'id': qid, 'conversation': row['conversation'], 'gold': row['gold'],
                'build': row['candidate']}
        for name, source in voters.items():
            other = source.get(qid)
            item[name] = (other.get('candidate') if other and other.get('answer') else None)
        data.append(item)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, help='Current build stage-B result')
    parser.add_argument('--voter', action='append', default=[], metavar='NAME=PATH',
                        help='Additional evidence producer; repeat for each voter')
    parser.add_argument('--rule', action='append', default=[], metavar='NAME[,NAME...]',
                        help='Voter subset to score; repeat to compare several named rules')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    build = load_rows(args.build)
    voters = {}
    for entry in args.voter:
        name, _, path = entry.partition('=')
        if not path:
            parser.error(f'Expected NAME=PATH, got {entry!r}')
        voters[name] = load_rows(path)
    data = build_rows(build, voters)
    if not data:
        parser.error('No answered positives in the build result')
    base = mean(tiou(r['gold'], r['build']) for r in data)
    report = {'questions': len(data), 'build_mean_tiou': base, 'build_raw': .4 + .6 * base, 'rules': []}
    print('build: tIoU %.6f  raw %.6f  (n=%d)\n' % (base, .4 + .6 * base, len(data)))
    for rule in args.rule or [','.join(voters)]:
        names = [n for n in rule.split(',') if n]
        missing = [n for n in names if n not in voters]
        if missing:
            parser.error(f'Unknown voter(s): {missing}')
        value = mean(tiou(r['gold'], vote(r, names)) for r in data)
        low, high, above = bootstrap(data, names)
        entry = {'rule': names, 'mean_tiou': value, 'raw': .4 + .6 * value,
                 'raw_gain': .6 * (value - base), 'bootstrap_ci95': [low, high],
                 'resamples_above_zero': above,
                 'voted_questions': sum(1 for r in data if len([n for n in names if r.get(n)]) >= 2)}
        report['rules'].append(entry)
        print('%-44s raw %.6f  %+.6f  [%+.6f, %+.6f]  >0 in %.0f%%'
              % ('+'.join(names), entry['raw'], entry['raw_gain'], low, high, 100 * above))
    if args.output:
        from .data import write_json
        write_json(args.output, report)
    return report


if __name__ == '__main__':
    main()
