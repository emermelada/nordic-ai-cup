"""Evidence-focused prompting and word-level localization."""

import re
from decimal import Decimal

from rapidfuzz import fuzz

from pipeline.core import SYSTEM, _expired, build_messages


EVIDENCE_RULES = '''- For every "yes", locate the passage that most directly and explicitly states the queried fact. Read the whole transcript before choosing: prefer the clinical finding, assessment or agreed plan itself over a vague confirmation or a later conversational recap. Match the question's specific subject, action, value and polarity, not just its topic.
- Copy an EXACT contiguous quote, preserving the transcript's wording. Select only the relevant clause, not the whole numbered unit. Include the subject and the fact it establishes; exclude greetings, filler, unrelated findings, explanations and subsequent advice. There is no fixed word limit: a single clause usually suffices, but retain necessary context.
- For a question about a specific item in a list, quote that item's clause rather than the entire list. For a yes/no reply whose subject is only in the preceding question, include the question and the reply together. Never quote an unanswered question as evidence.
- Give the numbered unit id(s) containing the quote. Different questions may need different parts of the same sentence. Do not automatically reuse an entire quote for related questions.'''

FOCUSED_SYSTEM = SYSTEM[:SYSTEM.index('- For every "yes"')] + EVIDENCE_RULES + '''
Respond with JSON only: {"results":[{"q":1,"answer":"yes","units":[12],"quote":"exact supporting clause"},{"q":2,"answer":"no","units":[],"quote":""}, ...]}'''


def build_focused_messages(words, questions):
    messages = build_messages(words, questions)
    messages[0] = {'role': 'system', 'content': FOCUSED_SYSTEM}
    return messages


# The legacy rules with one-line output: a third of the tokens, so larger models fit the budget.
COMPACT_SYSTEM = SYSTEM[:SYSTEM.index('Respond with JSON only')] + '''Respond with minified JSON on a single line and nothing else: {"results":[{"q":1,"answer":"yes","units":[12],"quote":"..."},{"q":2,"answer":"no"},...]}'''


def build_compact_messages(words, questions):
    messages = build_messages(words, questions)
    messages[0] = {'role': 'system', 'content': COMPACT_SYSTEM}
    return messages


# Qwen3.5 sometimes enumerated every unit id on "no" answers until the token limit.
MINIMAL_SYSTEM = COMPACT_SYSTEM.replace(
    'Respond with minified JSON',
    '- For every "no", give only the question number and the answer: no unit ids and no quote.\n'
    'Respond with minified JSON',
)


def build_minimal_messages(words, questions):
    messages = build_messages(words, questions)
    messages[0] = {'role': 'system', 'content': MINIMAL_SYSTEM}
    return messages


TOKEN = re.compile(r'\d+(?:\.\d+)?|[a-z]+')


def _text(text):
    return text.lower().replace("'", '').replace('’', '')


def _numbers(tokens):
    return {Decimal(token) for token in tokens if token[0].isdigit()}


def align_quote(words, quote, deadline=None):
    """Align contiguous text without conflating decimals with different doses."""
    if not isinstance(quote, str) or _expired(deadline):
        return None
    query = TOKEN.findall(_text(quote))
    if not query:
        return None
    parts, characters = [], []
    for index, word in enumerate(words):
        if _expired(deadline):
            return None
        part = _text(word['word'])
        parts.append(part)
        characters.extend([index] * len(part))
    # Whisper can split a single decimal into adjacent words, e.g. " 7" and ".0".
    matches = list(TOKEN.finditer(''.join(parts)))
    tokens = [match.group() for match in matches]
    query_numbers = _numbers(query)
    length = len(query)
    query_text = ' '.join(query)
    best_score, best = -1, None
    for start in range(len(tokens)):
        for size in range(max(1, length - 3), min(length + 3, len(tokens) - start) + 1):
            if _expired(deadline):
                return None
            window = tokens[start:start + size]
            if _numbers(window) != query_numbers:
                continue
            score = fuzz.ratio(' '.join(window), query_text)
            if score >= 70 and score > best_score:
                best_score = score
                best = (characters[matches[start].start()], characters[matches[start + size - 1].end() - 1])
    if best is None:
        return None
    return words[best[0]]['start'], words[best[1]]['end']


FRAME_SAMPLES = 160  # 10 ms at 16 kHz
ONSET_THRESHOLD_DB = -45.0
REPLY_MAX_WORDS = 6
REPLY_MAX_GAP_SECONDS = 2.0


def energy_envelope(samples):
    """Frame loudness in dBFS, one value per 10 ms."""
    import numpy as np

    count = len(samples) // FRAME_SAMPLES
    frames = np.asarray(samples[:count * FRAME_SAMPLES], dtype=np.float64).reshape(count, FRAME_SAMPLES)
    return np.round(20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9), 1).tolist()


def _speech_onset(envelope, start, end):
    # Whisper often starts a word inside the pause before it; annotations start at speech.
    first = max(0, int(round(start * 100)))
    for frame in range(first, min(len(envelope), first + 100)):
        if envelope[frame] > ONSET_THRESHOLD_DB:
            onset = max(start, frame / 100)
            return onset if onset < end else start
    return start


def _reply_end(words, start, end):
    inside = [i for i, word in enumerate(words) if word['start'] >= start - 1e-6 and word['end'] <= end + 1e-6]
    if not inside or not words[inside[-1]]['word'].strip().endswith('?'):
        return end
    reply = []
    previous_end = end
    for word in words[inside[-1] + 1:]:
        if word['start'] - previous_end > REPLY_MAX_GAP_SECONDS:
            return end
        previous_end = word['end']
        reply.append(word)
        if len(reply) > REPLY_MAX_WORDS or word['word'].strip().endswith('?'):
            return end
        if word['word'].strip().endswith(('.', '!')):
            return reply[-1]['end']
    return end


def refine_evidence(response, words, envelope=None):
    """Extend a quoted question to its short reply, then start the span at audible speech."""
    result = response.model_copy(deep=True)
    for i, (start, end) in enumerate(zip(result.evidence_start, result.evidence_end)):
        if start is None or end is None:
            continue
        end = _reply_end(words, start, end)
        if envelope:
            onset_end = next((min(end, word['end']) for word in words
                              if word['start'] < end
                              and (word['end'] > start or word['start'] == start)), start)
            start = _speech_onset(envelope, start, onset_end)
        result.evidence_start[i], result.evidence_end[i] = start, end
    return result


def _span_overlap(a, b):
    if a[0] is None or b[0] is None:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / union if union > 0 else 0.0


def consensus_response(primary, secondary, referee):
    """Keep the primary answers; take whichever model span the retrieval span agrees with.

    Retrieval overlap selects the passage; equal scores prefer the earlier occurrence.
    Missing retrieval evidence gives both model spans a score of zero.
    """
    result = primary.model_copy(deep=True)
    for i, answer in enumerate(result.answers):
        chosen = (result.evidence_start[i], result.evidence_end[i])
        other = (secondary.evidence_start[i], secondary.evidence_end[i])
        judge = (referee.evidence_start[i], referee.evidence_end[i])
        if not answer or chosen[0] is None or other[0] is None:
            continue
        chosen_score = _span_overlap(chosen, judge)
        other_score = _span_overlap(other, judge)
        if (other_score > chosen_score
                or (other_score == chosen_score and other[0] < chosen[0])):
            result.evidence_start[i], result.evidence_end[i] = other
    return result
