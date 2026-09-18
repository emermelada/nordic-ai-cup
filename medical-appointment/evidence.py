"""From a cited passage to the timestamps that score.

An answering model can say where its answer came from in two ways: the numbers
of the sentences it read it from, and a verbatim quote. The quote is the precise
one — the annotators' spans are quotes too, cut on word boundaries — so it is
tried first, and the sentence numbers only narrow where to look and serve as the
fallback. What each fallback is worth once the right passage is found, measured
against the 195 annotated spans:

    the annotator's exact quote        tIoU 1.000
    whole sentence(s) around it        tIoU 0.873
    best word-overlap sentence         tIoU 0.51   (no model at all)

Matching compares letters and digits only. The ASR and the model disagree about
case, punctuation, hyphens and spacing ("follow -up." against "follow-up",
"Ibu Medin" against "Ibu-Medin"), and none of that moves a word boundary.
"""

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from asr import Span, Transcript

# How many sentences either side of the cited ones a quote is still looked for
# in. Models misnumber by one far more often than by more.
EXACT_SEARCH_MARGIN = 1
FUZZY_SEARCH_MARGIN = 2

# A fuzzy match has to account for this share of the quote's characters...
FUZZY_MIN_COVERAGE = 0.6
# ...without being smeared over much more transcript than the quote is long...
FUZZY_MAX_STRETCH = 1.6
# ...and counting only runs long enough not to be coincidence.
FUZZY_MIN_BLOCK = 3

_STOPWORDS = frozenset("""
a an the is are was were be been being to of in on at for and or with that this
it its did does do has have had will would should could can not no yes right
okay ok as by from than then so if you your i me my we our he she they them
his her their patient doctor any there about what which who whom also just
some into up out over after before during correct true
""".split())


@dataclass
class Evidence:
    """A located span, and how it was found — worth logging, not just using."""

    span: Optional[Span]
    method: str  # 'exact', 'fuzzy', 'lines', 'lexical' or 'none'
    first_word: Optional[int] = None
    last_word: Optional[int] = None


def _key(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


class _CharIndex:
    """The transcript as one string of letters and digits, mapped back to words."""

    def __init__(self, transcript: Transcript):
        keys = [_key(word.text) for word in transcript.words]
        self.offsets = []
        self.owner = []
        position = 0

        for index, key in enumerate(keys):
            self.offsets.append(position)
            self.owner.extend([index] * len(key))
            position += len(key)

        self.text = ''.join(keys)
        self.lengths = [len(key) for key in keys]

    def word_range_to_chars(self, first_word: int, last_word: int) -> Tuple[int, int]:
        return self.offsets[first_word], self.offsets[last_word] + self.lengths[last_word]


def _clamp_lines(
    transcript: Transcript, lines: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    if not lines or not transcript.sentences:
        return None

    last = len(transcript.sentences) - 1
    first, second = sorted(int(line) for line in lines)
    if second < 0 or first > last:
        return None
    return max(first, 0), min(second, last)


def _window(
    transcript: Transcript, lines: Optional[Tuple[int, int]], margin: int
) -> Tuple[int, int]:
    """First and last word worth searching, around the cited sentences."""
    if lines is None:
        return 0, len(transcript.words) - 1

    last = len(transcript.sentences) - 1
    first_sentence = transcript.sentences[max(lines[0] - margin, 0)]
    last_sentence = transcript.sentences[min(lines[1] + margin, last)]
    return first_sentence.first_word, last_sentence.last_word


def _exact(
    index: _CharIndex, quote_key: str, window: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Every exact occurrence, preferring the one nearest the cited sentences."""
    best = None
    start = index.text.find(quote_key)

    while start != -1:
        first = index.owner[start]
        last = index.owner[start + len(quote_key) - 1]
        distance = max(window[0] - last, first - window[1], 0)

        if best is None or distance < best[0]:
            best = (distance, first, last)

        start = index.text.find(quote_key, start + 1)

    return (best[1], best[2]) if best else None


def _fuzzy(
    index: _CharIndex, quote_key: str, window: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """The stretch of the window that best lines up with the quote, if any does."""
    char_start, char_end = index.word_range_to_chars(*window)
    haystack = index.text[char_start:char_end]

    matcher = difflib.SequenceMatcher(None, quote_key, haystack, autojunk=False)
    blocks = [
        block for block in matcher.get_matching_blocks()
        if block.size >= FUZZY_MIN_BLOCK
    ]
    matched = sum(block.size for block in blocks)

    if not blocks or matched < FUZZY_MIN_COVERAGE * len(quote_key):
        return None

    first_char = blocks[0].b
    last_char = blocks[-1].b + blocks[-1].size - 1
    if last_char - first_char + 1 > FUZZY_MAX_STRETCH * len(quote_key):
        return None

    return index.owner[char_start + first_char], index.owner[char_start + last_char]


def content_words(text: str) -> set:
    return {
        word for word in re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", text.lower())
        if word not in _STOPWORDS and len(word) > 1
    }


def best_sentence(transcript: Transcript, question: str) -> Tuple[Optional[int], int]:
    """The sentence sharing the most content words with the question, and how many."""
    wanted = content_words(question)
    best, best_overlap = None, 0

    for index, sentence in enumerate(transcript.sentences):
        overlap = len(wanted & content_words(sentence.text))
        if overlap > best_overlap:
            best, best_overlap = index, overlap

    return best, best_overlap


def locate(
    transcript: Transcript,
    quote: Optional[str] = None,
    lines: Optional[Tuple[int, int]] = None,
    question: Optional[str] = None,
) -> Evidence:
    """Turn whatever the answering model cited into a span, as precisely as it allows.

    Tries, in order: the quote exactly (near the cited sentences, else anywhere),
    the quote fuzzily around the cited sentences, the cited sentences whole, and
    — given the question — the best word-overlap sentence. Never raises.
    """
    if not transcript.words:
        return Evidence(span=None, method='none')

    lines = _clamp_lines(transcript, lines)
    quote_key = _key(quote or '')

    if len(quote_key) >= 2:
        index = _CharIndex(transcript)

        found = _exact(index, quote_key, _window(transcript, lines, EXACT_SEARCH_MARGIN))
        if found:
            return Evidence(transcript.span(*found), 'exact', *found)

        found = _fuzzy(index, quote_key, _window(transcript, lines, FUZZY_SEARCH_MARGIN))
        if found:
            return Evidence(transcript.span(*found), 'fuzzy', *found)

    if lines is not None:
        first_word = transcript.sentences[lines[0]].first_word
        last_word = transcript.sentences[lines[1]].last_word
        return Evidence(transcript.span(first_word, last_word), 'lines', first_word, last_word)

    if question:
        sentence, overlap = best_sentence(transcript, question)
        if sentence is not None and overlap > 0:
            chosen = transcript.sentences[sentence]
            return Evidence(
                transcript.span(chosen.first_word, chosen.last_word),
                'lexical', chosen.first_word, chosen.last_word,
            )

    return Evidence(span=None, method='none')


def widen_to_sentences(transcript: Transcript, evidence: Evidence) -> Evidence:
    """The same passage, widened to the whole sentences it touches.

    69% of the annotated quotes are exactly whole sentences, and with the right
    sentences this scores tIoU 0.873 against them — a floor for a model whose
    own trimming is worse than that.
    """
    if evidence.first_word is None or evidence.last_word is None:
        return evidence

    first = transcript.sentences[transcript.sentence_of(evidence.first_word)].first_word
    last = transcript.sentences[transcript.sentence_of(evidence.last_word)].last_word
    return Evidence(transcript.span(first, last), evidence.method, first, last)


def _related(a: str, b: str) -> bool:
    """Same word up to its ending: scar/scarring, sinus/sinuses, treat/treatment."""
    short, long_ = sorted((a, b), key=len)
    return (len(short) >= 4 and long_.startswith(short)) or (
        len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]
    )


def _hits(question_words: set, text: str) -> set:
    """The question's content words that this text mentions."""
    words = content_words(text)
    return {q for q in question_words if any(_related(q, w) for w in words)}


def _text(transcript: Transcript, first: int, last: int) -> str:
    return ''.join(word.text for word in transcript.words[first:last + 1]).strip()


_CONNECTIVE = re.compile(r'^(and|but|so|which|while|because|although)$', re.IGNORECASE)


def trim_to_statement(transcript: Transcript, evidence: Evidence, question: str) -> Evidence:
    """Cut a located quote down to the way the annotators quoted.

    Models quote the fact and then the confirmations and repetitions that follow
    it; the annotators quoted where the fact is first stated, and only as far as
    the fact goes. On the 9B model's quotes (156 positives, box timings) this took
    tIoU from 0.637 to 0.688, the same gain on both halves of the conversations:

    1. Keep the earliest run of sentences that covers every question word the
       quote covers, plus the next sentence if that run ends on a question —
       the answer to it.
    2. Stop at ", and ..." / ", but ..." after the last word the question needs.
    3. Drop a leading "And" from a statement (not from a question).
    """
    if evidence.first_word is None or evidence.last_word is None:
        return evidence

    wanted_words = content_words(question)
    first, last = evidence.first_word, evidence.last_word

    # 1. First statement.
    pieces = [
        (max(s.first_word, first), min(s.last_word, last))
        for s in transcript.sentences
        if max(s.first_word, first) <= min(s.last_word, last)
    ]
    wanted = _hits(wanted_words, _text(transcript, first, last))
    if wanted and len(pieces) > 1:
        start = next(
            (n for n, piece in enumerate(pieces) if _hits(wanted_words, _text(transcript, *piece))), 0
        )
        end, covered = len(pieces) - 1, set()
        for n in range(start, len(pieces)):
            covered |= _hits(wanted_words, _text(transcript, *pieces[n]))
            if covered >= wanted:
                end = n
                break
        if _text(transcript, *pieces[end]).endswith('?') and end + 1 < len(pieces):
            end += 1
        first, last = pieces[start][0], pieces[end][1]

    # 2. Clause end.
    last_hit = None
    for index in range(first, last + 1):
        if _hits(wanted_words, transcript.words[index].text):
            last_hit = index
    if last_hit is not None:
        for index in range(last_hit, last):
            if (transcript.words[index].text.strip().endswith(',')
                    and _CONNECTIVE.match(transcript.words[index + 1].text.strip())):
                last = index
                break

    # 3. Leading "And".
    if (first < last and transcript.words[first].text.strip().lower() == 'and'
            and not _text(transcript, first, last).endswith('?')):
        first += 1

    return Evidence(transcript.span(first, last), evidence.method, first, last)


def words_in_span(transcript: Transcript, span: Span) -> List[int]:
    """Indices of the words whose midpoint falls inside ``span``."""
    start, end = span
    return [
        index for index, word in enumerate(transcript.words)
        if start - 1e-6 <= (word.start + word.end) / 2 <= end + 1e-6
    ]


def sentences_overlapping(transcript: Transcript, span: Span) -> Optional[Tuple[int, int]]:
    """First and last sentence that overlap ``span`` in time."""
    start, end = span
    hit = [
        index for index, sentence in enumerate(transcript.sentences)
        if sentence.end > start and sentence.start < end
    ]
    return (hit[0], hit[-1]) if hit else None
