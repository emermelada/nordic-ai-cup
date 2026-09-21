"""Locate a passage globally, then select its exact original word boundaries."""

import json

from pipeline.core import _bounded_span, _expired, _question_id, sanitize_response
from pipeline.evidence import refine_evidence

CONTEXT_SENTENCES = 1
LOCATE_SYSTEM = '''Locate the passage that supports an already-YES question in a doctor-patient transcript. Sentences have zero-based IDs in square brackets.
Select one contiguous interval of sentences containing the fact and the context needed to interpret its subject, polarity, timing and details. If the same fact appears more than once, choose the occurrence that matches this particular question, not merely another mention of the topic. Include the question or antecedent when a short reply depends on it.
This pass locates the passage; it does not trim individual words. Return only a JSON object with integer fields first_sentence and last_sentence. Both endpoints are inclusive and must be sentence IDs shown in the transcript. Do not return an answer, quotation or explanation.'''


def build_locate_messages(transcript, question):
    return [{'role': 'system', 'content': LOCATE_SYSTEM},
            {'role': 'user', 'content': f'TRANSCRIPT\n{transcript}\n\nQUESTION (answered yes): {question}\n\nLOCATION:'}]


def build_bounds_messages(words, question, window):
    from pipeline.stage_b import CONVENTIONS

    conventions = CONVENTIONS.split('Conventions of the annotation:\n', 1)[1].split('\nExample transcript:', 1)[0]
    system = ('Select the minimal self-contained evidence for an already-YES question from this consultation passage. '
              'Each [wN] label identifies one original ASR word; IDs are absolute, zero-based and must not be renumbered. '
              'Neighboring sentences are context, not a requirement to include them.\n\n'
              'Conventions for the selected passage:\n' + conventions + '\n\n'
              'Return only a JSON object with integer fields start_word and end_word, the first and last word IDs '
              'of one contiguous passage. Both endpoints are inclusive and must occur in the supplied passage. '
              'Do not copy a quotation, change the answer, or explain the selection.')
    passage = ' '.join(f'[w{i}] {words[i]["word"].strip()}' for i in range(window[0], window[1] + 1))
    user = f'QUESTION (answered yes): {question}\n\nPASSAGE\n{passage}\n\nWORD BOUNDARIES:'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def _range(first, last, count):
    if type(first) is int and type(last) is int and 0 <= first <= last < count:
        return first, last
    return None


def parse_range(raw, first_key, last_key, count):
    if not isinstance(raw, str):
        return None
    text = raw.rsplit('</think>', 1)[-1].strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return None
    # The model reliably answers with a bare pair, "[31, 34]", rather than the named object
    # the prompt asks for; both carry the same inclusive endpoints, so both are accepted.
    if isinstance(value, list):
        return _range(*value, count) if len(value) == 2 else None
    if not isinstance(value, dict):
        return None
    return _range(value.get(first_key), value.get(last_key), count)


def question_outputs(outputs, count):
    result, seen = {}, set()
    if not isinstance(outputs, dict):
        return result
    for key, value in outputs.items():
        qid = _question_id(key, count)
        if qid is not None:
            result[qid] = None if qid in seen else value
            seen.add(qid)
    return result


def locate_evidence(words, questions, answers, generate, deadline=None):
    """generate(phase, prompts) shares one request deadline across both model batches."""
    from pipeline.stage_b import render_sentences, sentence_ranges

    frame = {'mode': 'locate', 'locations': {}, 'windows': {}, 'outputs': {}}
    if not words or not any(answers):
        return frame
    if _expired(deadline):
        return {**frame, 'skipped': 'budget'}
    sentences = sentence_ranges(words)
    transcript = render_sentences(words, sentences)
    prompts = [(i + 1, build_locate_messages(transcript, question))
               for i, question in enumerate(questions) if answers[i]]
    frame['locations'] = question_outputs(generate('locate', prompts), len(questions))
    if _expired(deadline):
        return {**frame, 'skipped': 'budget'}
    prompts = []
    for i, question in enumerate(questions):
        if not answers[i]:
            continue
        located = parse_range(frame['locations'].get(i + 1), 'first_sentence', 'last_sentence', len(sentences))
        if located is None:
            continue
        first, last = located
        window = (sentences[max(0, first - CONTEXT_SENTENCES)][0],
                  sentences[min(len(sentences) - 1, last + CONTEXT_SENTENCES)][1])
        frame['windows'][i + 1] = list(window)
        prompts.append((i + 1, build_bounds_messages(words, question, window)))
    if prompts:
        if _expired(deadline):
            return {**frame, 'skipped': 'budget'}
        frame['outputs'] = question_outputs(generate('bounds', prompts), len(questions))
        if _expired(deadline):
            frame['skipped'] = 'budget'
    return frame


def word_spans(evidence, words, count, duration=None, deadline=None):
    windows = question_outputs(evidence.get('windows'), count)
    spans = {}
    for qid, raw in question_outputs(evidence.get('outputs'), count).items():
        if _expired(deadline):
            break
        spans[qid] = None
        window = windows.get(qid)
        if not isinstance(window, (list, tuple)) or len(window) != 2 or _range(*window, len(words)) is None:
            continue
        span = parse_range(raw, 'start_word', 'end_word', len(words))
        if span is None or span[0] < window[0] or span[1] > window[1]:
            continue
        spans[qid] = _bounded_span(words[span[0]]['start'], words[span[1]]['end'], duration)
    return spans


def apply_located(evidence, response, words, duration, envelope=None, deadline=None, extend_replies=True):
    result = response.model_copy(deep=True)
    count = len(result.answers)
    proposal = result.model_copy(deep=True)
    proposal.answers = [False] * count
    proposal.evidence_start = [None] * count
    proposal.evidence_end = [None] * count
    replaced = []
    for qid, span in word_spans(evidence, words, count, duration, deadline).items():
        index = qid - 1
        if span is None or not result.answers[index]:
            continue
        proposal.answers[index] = True
        proposal.evidence_start[index], proposal.evidence_end[index] = span
        replaced.append(index)
    proposal = sanitize_response(refine_evidence(proposal, words, envelope, extend_replies), count, duration)
    for index in replaced:
        if proposal.evidence_start[index] is not None:
            result.evidence_start[index] = proposal.evidence_start[index]
            result.evidence_end[index] = proposal.evidence_end[index]
    return result
