"""Text-only rationale data; never manufacture audio timestamps for external text."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .data import digest


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def context_record(text):
    words = [m.group() for m in re.finditer(r'\S+', text)]
    context = ' '.join(words)
    return {'context': context, 'word_chars': [[m.start(), m.end()] for m in re.finditer(r'\S+', context)]}


def word_span(text, start, end, expected=None):
    if not 0 <= start < end <= len(text):
        raise ValueError('invalid_character_span')
    if expected is not None and text[start:end] != expected:
        raise ValueError('source_rationale_mismatch')
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        raise ValueError('empty_rationale')
    matches = list(re.finditer(r'\S+', text))
    first = bisect_right([m.end() for m in matches], start)
    last = bisect_left([m.start() for m in matches], end) - 1
    if not 0 <= first <= last < len(matches):
        raise ValueError('unmappable_rationale')
    return [first, last], start != matches[first].start() or end != matches[last].end()


class OverlapIndex:
    """Conservative lexical quarantine, not a guarantee against semantic overlap."""
    def __init__(self, documents):
        self.postings = {5: defaultdict(set), 8: defaultdict(set)}
        self.sizes = {}
        for cid, text in documents.items():
            tokens = normalize(text).split()
            for n in self.postings:
                shingles = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
                self.sizes[cid, n] = len(shingles)
                for shingle in shingles:
                    self.postings[n][shingle].add(cid)

    def matches(self, text, near=False):
        tokens, matches = normalize(text).split(), set()
        for n, index in self.postings.items():
            shingles = {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
            counts = Counter(cid for s in shingles for cid in index.get(s, ()))
            fraction, minimum = ((.20, 10) if n == 5 else (.08, 6)) if near else (.60, 10)
            for cid, count in counts.items():
                denominator = min(len(shingles), self.sizes[cid, n])
                if denominator and count >= minimum and count / denominator >= fraction:
                    matches.add(cid)
        return sorted(matches)


def coqa_documents(raw, split):
    data = json.loads((raw / f'coqa-{split}-v1.0.json').read_text())['data']
    for doc in data:
        # Keep a single, clearly attributed license; do not mix RACE or MCTest terms.
        if doc['source'] not in ('wikipedia', 'gutenberg'):
            continue
        examples, previous = [], []
        if len(doc['questions']) != len(doc['answers']):
            raise ValueError('Unpaired CoQA questions and answers')
        for q, answer in zip(doc['questions'], doc['answers']):
            if q['turn_id'] != answer['turn_id']:
                raise ValueError('CoQA turn IDs differ')
            history = ' '.join(f'Previous question: {question} Previous answer: {a}'
                               for question, a in previous[-2:])
            question = (history + ' Current question: ' + q['input_text']).strip() if history else q['input_text']
            previous.append((q['input_text'], answer['input_text']))
            examples.append({'id': f'coqa:{doc["id"]}:{q["turn_id"]}', 'question': question,
                'current_question': q['input_text'], 'label': answer['input_text'].strip().lower() != 'unknown',
                'raw_span': [answer['span_start'], answer['span_end']], 'span_text': answer['span_text'],
                'answer_type': answer['input_text'].strip().lower() if answer['input_text'].strip().lower()
                               in ('yes', 'no', 'unknown') else 'span',
                'annotation': 'human_rationale', 'turn_id': q['turn_id']})
        yield {'id': f'coqa:{doc["id"]}', 'group': f'coqa:{doc["source"]}:{doc["filename"]}',
               'text': doc['story'], 'examples': examples, 'split': split, 'source': 'coqa',
               'domain': doc['source'], 'license': 'CC-BY-SA-4.0',
               'source_document': doc['filename']}


def simord_documents(raw, split, sentence_tokenize):
    """Mirror the organizer's sentence numbering, without running downloaded code."""
    import textgrid

    aci = {}
    for path in sorted(raw.glob('aci-*.json')):
        basename = path.stem.removeprefix('aci-')
        for row in json.loads(path.read_text()).get('data', []):
            name = '_'.join(row['file'].split('-')[:2])
            aci[f'acibench_{name}_{basename}'] = row['src']
    for doc in json.loads((raw / f'simord-{split}.json').read_text()):
        cid = doc['id']
        if cid.startswith('acibench_'):
            transcript = aci[cid]
            group = ':'.join(cid.split('_')[:2])
        else:
            _, day, number = cid.split('_')
            utterances = []
            for speaker in ('doctor', 'patient'):
                grid = textgrid.TextGrid.fromFile(str(raw / f'day{day}_consultation{int(number):02d}_{speaker}.TextGrid'))
                for tier in grid.tiers:
                    for interval in tier.intervals:
                        if interval.mark:
                            text = interval.mark
                            for tag in ('<UNSURE>', '</UNSURE>', '<UNIN/>', '<INAUDIBLE_SPEECH/>'):
                                text = text.replace(tag, '')
                            utterances.append((interval.minTime, f'[{speaker}] ' + ' '.join(text.split())))
            utterances.sort(key=lambda x: x[0])  # stable doctor/patient tie order, as in the source converter
            transcript = '\n'.join(text for _, text in utterances)
            group = cid
        sentences = []
        for line in transcript.split('\n'):
            body = line.strip().split(']', 1)[-1].strip()
            sentences.extend(sentence_tokenize(body))
        text, offsets = '', []
        for sentence in sentences:
            if text:
                text += ' '
            offsets.append([len(text), len(text) + len(sentence)])
            text += sentence
        examples = []
        for number, order in enumerate(doc['expected_orders']):
            provenance = sorted(set(order['provenance']))
            valid = (provenance and provenance[0] >= 1 and provenance[-1] <= len(sentences)
                     and provenance == list(range(provenance[0], provenance[-1] + 1)))
            span = [offsets[provenance[0] - 1][0], offsets[provenance[-1] - 1][1]] if valid else None
            question = f'Is this recommendation made during the consultation: {order["description"]}?'
            examples.append({'id': f'simord:{cid}:{number}', 'question': question,
                             'current_question': question, 'label': True, 'raw_span': span,
                             'span_text': text[span[0]:span[1]] if span else None,
                             'answer_type': 'medical_order', 'annotation': 'clinician_order_sentence_provenance',
                             'provenance_sentence_ids': provenance,
                             'reject_reason': None if valid else 'noncontiguous_or_invalid_medical_provenance'})
        yield {'id': f'simord:{cid}', 'group': f'simord:{group}', 'text': text, 'examples': examples,
               'split': split, 'source': 'simord', 'domain': 'medical_dialogue',
               'license': 'CDLA-Permissive-2.0 annotations; CC-BY-4.0 transcripts', 'source_document': cid}


def mashqa_documents(raw, split):
    filename = 'val' if split == 'dev' else split
    with zipfile.ZipFile(raw / 'mashqa_data.zip') as archive:
        data = json.loads(archive.read(f'mashqa_data/{filename}_webmd_squad_v2_consec.json'))['data']
    for doc in data:
        for index, paragraph in enumerate(doc['paragraphs']):
            text, examples = paragraph['context'], []
            # Published test labels are not imported or used for selection.
            for item in paragraph['qas'] if split != 'test' else []:
                answers = item['answers']
                valid = len(answers) == 1 and not item.get('is_impossible', False)
                answer = answers[0] if valid else {}
                sentence_ids = answer.get('answer_span', [])
                valid = valid and bool(sentence_ids) and sentence_ids == list(range(sentence_ids[0], sentence_ids[-1] + 1))
                span = [answer['answer_start'], answer['answer_start'] + len(answer['text'])] if valid else None
                examples.append({'id': 'mashqa:' + item['id'], 'question': item['question'],
                                 'current_question': item['question'], 'label': True, 'raw_span': span,
                                 'span_text': answer.get('text'), 'answer_type': 'medical_article',
                                 'annotation': 'published_webmd_qa_contiguous_evidence',
                                 'reject_reason': None if valid else 'noncontiguous_or_invalid_mashqa_answer'})
            group = 'mashqa:' + doc['title'].rstrip('/').split('#')[0]
            yield {'id': group + f':{index}', 'group': group, 'text': text, 'examples': examples,
                   'source': 'mashqa', 'split': split, 'domain': 'medical_article',
                   'source_document': doc['title'],
                   'license': 'Author-released research data; data terms not separately stated; retain WebMD attribution'}


def hydrate(bundle, split):
    result = []
    for example in bundle['examples']:
        if example['split'] == split:
            context = bundle['contexts'][example['conversation']]
            result.append({**context, **example})
    return result


def validate_external(bundle, target_path=None):
    if bundle.get('schema_version') != 'external-rationales-v1' or bundle.get('metric') != 'word_span_iou_not_temporal':
        raise ValueError('Not a supported text-only rationale bundle')
    if target_path and digest(target_path) != bundle['excluded_target_sha256']:
        raise ValueError('External data was not decontaminated against this target dataset')
    groups, signatures, ids = defaultdict(set), defaultdict(set), set()
    for cid, context in bundle['contexts'].items():
        expected = context_record(context['context'])
        if context['word_chars'] != expected['word_chars']:
            raise ValueError(f'Invalid word mapping: {cid}')
        if any(k in context for k in ('words', 'gold', 'baseline_answer')):
            raise ValueError('External text must not contain fabricated timestamps or frozen answers')
    for row in bundle['examples']:
        if row['id'] in ids:
            raise ValueError('Duplicate external ID')
        ids.add(row['id'])
        if row['split'] not in ('train', 'dev', 'diagnostic') or not isinstance(row['label'], bool):
            raise ValueError('Invalid external split/label')
        context = bundle['contexts'][row['conversation']]
        if row['label']:
            a, b = row['gold_words']
            if not 0 <= a <= b < len(context['word_chars']):
                raise ValueError('Invalid external evidence endpoints')
        elif row['gold_words'] is not None:
            raise ValueError('Unanswerable example has evidence')
        groups[context['group']].add(row['split'])
        signatures[normalize(context['context'])].add(row['split'])
    if any(len(s) != 1 for s in [*groups.values(), *signatures.values()]):
        raise ValueError('External train/dev context leakage')
    if not {'train', 'dev'} <= {row['split'] for row in bundle['examples']}:
        raise ValueError('External data requires training and development examples')


def load_external(path, target_path=None):
    bundle = json.loads(Path(path).read_text())
    validate_external(bundle, target_path)
    return bundle


def source_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()
