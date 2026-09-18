"""Optional CPU ranking of word boundaries around the primary evidence passage."""

import json
import logging
import math
from pathlib import Path
import time

import numpy as np
from rank_bm25 import BM25Okapi

from pipeline.core import STOP, _norm, _numbers, split_units
from pipeline.evidence import _speech_onset
from utils import evidence_interval, temporal_iou

logger = logging.getLogger(__name__)
RADIUS_SECONDS = 2.0
RETRIEVAL_PASSAGES = 3
BUDGET_SECONDS = 0.25
MAX_WORDS = 2000
MAX_TRANSCRIPT_CHARACTERS = 32768
MAX_QUESTION_CHARACTERS = 4096
MAX_CANDIDATES = 8192
MAX_EXPANSIONS = 50000
NEGATION = set('no not never neither without denies denied absent'.split())
FEATURE_NAMES = (
    'log_seconds', 'log_words', 'relative_start', 'query_coverage', 'weighted_coverage',
    'query_precision', 'missing_numbers', 'extra_numbers', 'negation', 'negation_in_both',
    'sentence_start', 'clause_start', 'sentence_end', 'clause_end', 'question_end',
    'short_reply_start', 'leading_gap', 'trailing_gap',
) + tuple(name + '_' + suffix for name in ('primary', 'secondary', 'referee')
          for suffix in ('iou', 'exact', 'covered', 'coverage', 'start_delta', 'end_delta'))
NODE_FIELDS = ['feature', 'threshold', 'left', 'right', 'value']


def load_model(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Boundary model exceeds size limit')
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1 or data.get('feature_names') != list(FEATURE_NAMES)
            or data.get('node_fields') != NODE_FIELDS):
        raise ValueError('Unsupported boundary model schema')
    baseline, trees = data['baseline'], data['trees']
    if type(baseline) not in (int, float) or not math.isfinite(baseline):
        raise ValueError('Invalid boundary model baseline')
    if not isinstance(trees, list) or not 1 <= len(trees) <= 256:
        raise ValueError('Invalid boundary tree count')
    compiled = []
    for nodes in trees:
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 15:
            raise ValueError('Invalid boundary node count')
        depths = {0: 0}
        for i, node in enumerate(nodes):
            if (not isinstance(node, list) or len(node) != len(NODE_FIELDS)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in node)):
                raise ValueError('Invalid boundary node')
            feature, threshold, left, right, value = node
            if (i not in depths or type(feature) is not int or not -1 <= feature < len(FEATURE_NAMES)
                    or type(left) is not int or type(right) is not int):
                raise ValueError('Invalid boundary tree indices')
            if feature != -1:
                if depths[i] >= 3:
                    raise ValueError('Boundary tree exceeds depth limit')
                for child in (left, right):
                    if not i < child < len(nodes) or child in depths:
                        raise ValueError('Invalid boundary tree child')
                    depths[child] = depths[i] + 1
        columns = tuple(np.asarray([node[i] for node in nodes], dtype=dtype)
                        for i, dtype in enumerate((np.intp, np.float64, np.intp, np.intp, np.float64)))
        for column in columns:
            column.flags.writeable = False
        compiled.append(columns)
    return float(baseline), tuple(compiled)


try:
    MODEL = load_model(Path(__file__).with_name('boundary_model.json'))
except Exception:
    logger.exception('Boundary model unavailable; keeping primary evidence')
    MODEL = None


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Boundary adjustment deadline')


def predict_scores(matrix, model, deadline=None):
    if (matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES)
            or not np.isfinite(matrix).all()):
        raise ValueError('Invalid boundary features')
    baseline, trees = model
    scores = np.full(len(matrix), baseline, dtype=np.float64)
    for feature, threshold, left, right, value in trees:
        _check_deadline(deadline)
        indices = np.zeros(len(matrix), dtype=np.intp)
        for _ in range(3):
            active = np.flatnonzero(feature[indices] >= 0)
            nodes = indices[active]
            indices[active] = np.where(matrix[active, feature[nodes]] <= threshold[nodes],
                                       left[nodes], right[nodes])
        scores += value[indices]
    if not np.isfinite(scores).all():
        raise ValueError('Invalid boundary scores')
    return scores


def interval(response, index):
    if response is None:
        return None
    return evidence_interval(response.evidence_start[index], response.evidence_end[index])


def prepare_context(words, duration, envelope, deadline=None):
    _check_deadline(deadline)
    if not words or len(words) > MAX_WORDS or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Unsupported boundary transcript size or duration')
    if any(not math.isfinite(word[key]) for word in words for key in ('start', 'end')):
        raise ValueError('Invalid boundary word timestamps')
    if sum(len(word['word']) for word in words) > MAX_TRANSCRIPT_CHARACTERS:
        raise ValueError('Boundary transcript text limit')
    units = split_units(words)
    texts = [''.join(word['word'] for word in unit) for unit in units]
    tokens = [[token for token in _norm(text) if token not in STOP] for text in texts]
    starts = []
    for word in words:
        _check_deadline(deadline)
        starts.append(_speech_onset(envelope or (), word['start'], word['end']))
    return {
        'words': words, 'duration': duration, 'units': units, 'tokens': tokens,
        'bm25': BM25Okapi(tokens) if any(tokens) else None,
        'starts': starts, 'ends': [word['end'] for word in words], 'text_cache': {},
    }


def candidate_spans(context, question, proposals, deadline=None):
    _check_deadline(deadline)
    anchor = proposals[0]
    if anchor is None:
        return [], []
    words, starts, ends = context['words'], context['starts'], context['ends']
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ValueError('Boundary question text limit')
    query = [token for token in _norm(question) if token not in STOP and not token[0].isdigit()]
    scores = context['bm25'].get_scores(query) if context['bm25'] is not None else []
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    retrieval = [(starts[sum(len(u) for u in context['units'][:i])], context['units'][i][-1]['end'])
                 for i in ranked[:RETRIEVAL_PASSAGES] if scores[i] > 0]
    seeds = [span for span in proposals if span is not None] + retrieval
    candidates = {}
    expansions = 0
    for start, end in seeds:
        _check_deadline(deadline)
        inside = [i for i, word in enumerate(words)
                  if word['start'] < end and (word['end'] > start or word['start'] == start)]
        if not inside or end <= start:
            continue
        candidates[(start, end)] = (inside[0], inside[-1])
        left = [i for i, value in enumerate(starts) if abs(value - start) <= RADIUS_SECONDS]
        right = [j for j, value in enumerate(ends) if abs(value - end) <= RADIUS_SECONDS]
        expansions += len(left) * len(right)
        if expansions > MAX_EXPANSIONS:
            raise ValueError('Boundary expansion limit')
        for i in left:
            _check_deadline(deadline)
            for j in right:
                if j >= i and starts[i] < ends[j]:
                    candidates[(starts[i], ends[j])] = (i, j)
                if len(candidates) > MAX_CANDIDATES:
                    raise ValueError('Boundary candidate limit')
    selected = [(span, boundary) for span, boundary in candidates.items()
                if temporal_iou(anchor, span) > 0
                and abs(span[0] - anchor[0]) <= RADIUS_SECONDS
                and abs(span[1] - anchor[1]) <= RADIUS_SECONDS]
    return [span for span, _ in selected], [boundary for _, boundary in selected]


def span_features(context, question, proposals, spans, boundaries, deadline=None):
    words, duration = context['words'], context['duration']
    query = set(_norm(question)) - STOP
    query_numbers = _numbers(query)
    query_terms = query - {token for token in query if token[0].isdigit()}
    idf = {}
    for term in query_terms:
        _check_deadline(deadline)
        idf[term] = math.log((1 + len(context['tokens'])) /
                             (1 + sum(term in unit for unit in context['tokens']))) + 1
    matrix = []
    for span, (left, right) in zip(spans, boundaries):
        _check_deadline(deadline)
        start, end = span
        key = left, right
        if key not in context['text_cache']:
            text = ''.join(word['word'] for word in words[left:right + 1])
            context['text_cache'][key] = set(_norm(text)) - STOP
        tokens = context['text_cache'][key]
        numbers = _numbers(tokens)
        common = query_terms & tokens
        length = end - start
        first, last = words[left]['word'].strip(), words[right]['word'].strip()
        previous = words[left - 1]['word'].strip() if left else ''
        features = {
            'log_seconds': math.log1p(length), 'log_words': math.log1p(right - left + 1),
            'relative_start': start / duration,
            'query_coverage': len(common) / max(1, len(query_terms)),
            'weighted_coverage': sum(idf[term] for term in common) / max(1, sum(idf.values())),
            'query_precision': len(common) / max(1, len(tokens)),
            'missing_numbers': len(query_numbers - numbers),
            'extra_numbers': len(numbers - query_numbers),
            'negation': bool(tokens & NEGATION),
            'negation_in_both': bool(query & NEGATION) and bool(tokens & NEGATION),
            'sentence_start': left == 0 or previous.endswith(('.', '?', '!')),
            'clause_start': left == 0 or previous.endswith(('.', '?', '!', ',', ';', ':')),
            'sentence_end': last.endswith(('.', '?', '!')),
            'clause_end': last.endswith(('.', '?', '!', ',', ';', ':')),
            'question_end': last.endswith('?'),
            'short_reply_start': first.lower().rstrip(',!.?') in ('yes', 'no', 'yeah', 'nope', 'sure'),
            'leading_gap': min(2., max(0., words[left]['start'] - words[left - 1]['end'])) if left else 0.,
            'trailing_gap': min(2., max(0., words[right + 1]['start'] - words[right]['end'])) if right + 1 < len(words) else 0.,
        }
        for name, proposal in zip(('primary', 'secondary', 'referee'), proposals):
            intersection = (max(0., min(end, proposal[1]) - max(start, proposal[0]))
                            if proposal is not None else 0.)
            features.update({
                name + '_iou': temporal_iou(span, proposal),
                name + '_exact': span == proposal,
                name + '_covered': intersection / max(length, 0.01),
                name + '_coverage': intersection / max(proposal[1] - proposal[0], 0.01) if proposal else 0.,
                name + '_start_delta': max(-4., min(4., start - proposal[0])) if proposal else 0.,
                name + '_end_delta': max(-4., min(4., end - proposal[1])) if proposal else 0.,
            })
        matrix.append([features[name] for name in FEATURE_NAMES])
    return np.asarray(matrix, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))


def adjust_boundaries(response, words, questions, duration, envelope, primary, secondary, referee,
                      *, deadline=None, trace=None):
    started = time.monotonic()
    deadline = min(deadline, started + BUDGET_SECONDS) if deadline is not None else started + BUDGET_SECONDS
    info = {'status': 'skipped', 'changed_spans': 0}
    try:
        _check_deadline(deadline)
        if MODEL is None:
            raise ValueError('Boundary model unavailable')
        eligible = [i for i, answer in enumerate(response.answers) if answer and interval(primary, i) is not None]
        if not eligible:
            info['status'] = 'not-needed'
            return response
        context = prepare_context(words, duration, envelope, deadline)
        result = response.model_copy(deep=True)
        changed = 0
        for i in eligible:
            _check_deadline(deadline)
            proposals = [interval(proposal, i) for proposal in (primary, secondary, referee)]
            spans, boundaries = candidate_spans(context, questions[i], proposals, deadline)
            if not spans:
                continue
            matrix = span_features(context, questions[i], proposals, spans, boundaries, deadline)
            scores = predict_scores(matrix, MODEL, deadline)
            span = spans[int(np.argmax(scores))]
            if not 0 <= span[0] < span[1] <= duration:
                raise ValueError('Invalid selected boundary span')
            changed += span != interval(result, i)
            result.evidence_start[i], result.evidence_end[i] = span
        _check_deadline(deadline)
        info.update(status='completed', changed_spans=changed)
        return result
    except Exception as exc:
        info['reason'] = f'{type(exc).__name__}: {exc}'
        logger.warning('Keeping primary evidence: %s', info['reason'])
        return response
    finally:
        if trace is not None:
            trace['boundary'] = {**info, 'seconds': time.monotonic() - started}
