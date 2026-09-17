"""CPU-only transcript, answer, and evidence processing."""

import json
import math
import re
import time
from decimal import Decimal

from json_repair import loads as repair_json
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from dtos import ASRQuestionResponseDto


SYSTEM = """You are checking claims against the transcript of a doctor-patient consultation. The transcript is split into numbered units. Answer each yes/no question ONLY from what the transcript says.
Rules:
- Answer "yes" only if the transcript explicitly states or clearly implies it. A near-miss is "no": same drug but a different dose, same test but a different value, same symptom but a different body part, a plan that was only mentioned as a possibility, or the opposite polarity (stable vs unstable, renewed vs stopped).
- Things never discussed are "no".
- Questions phrased as an absence ("free of fever", "no signs of X", "unchanged") are "yes" when the transcript states that absence or unchanged status.
- Tag questions ("..., right?") are ordinary yes/no questions.
- Abbreviations in questions may be spoken in full in the transcript (HbA1c = haemoglobin A1c / long-term sugar; mmol/mol = millimoles per mole; ECG = electrocardiogram; BMI = body mass index; BP = blood pressure).
- For every "yes", give the unit id(s) (at most 2, adjacent) and copy the EXACT words from those units that establish the answer: a contiguous substring, as short as possible while still containing the fact (usually 4-15 words).
Respond with JSON only: {"results":[{"q":1,"answer":"yes","confidence":90,"units":[12],"quote":"..."},{"q":2,"answer":"no","confidence":80,"units":[],"quote":""}, ...]}"""

SYN = {
    "hba1c": "hemoglobin a1c long-term sugar", "ecg": "electrocardiogram",
    "bmi": "body mass index", "tsh": "thyroid stimulating hormone",
    "bp": "blood pressure", "mmol/mol": "millimoles per mole",
    "mmol/l": "millimoles per liter", "mg": "mg milligrams",
    "iu": "international units", "nsaid": "anti-inflammatory", "gp": "doctor",
}
STOP = set(
    "the a an is was were be been are did does do has have had of to in on for at "
    "with by and or that this it its as from about any there their patient doctor "
    "be will should would can could right correct didn t wasn isn".split()
)


def floor_response(count: int) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(
        answers=[True] * count,
        evidence_start=[None] * count,
        evidence_end=[None] * count,
    )


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _normalize_words(words):
    normalized = []
    if not isinstance(words, list):
        return normalized
    for word in words:
        if not isinstance(word, dict):
            continue
        text = word.get("word")
        start = _finite_number(word.get("start"))
        end = _finite_number(word.get("end"))
        if not isinstance(text, str) or not text.strip():
            continue
        if start is None or end is None or end < start:
            continue
        normalized.append({
            "word": text, "start": max(0.0, start), "end": max(0.0, end),
            "p": _finite_number(word.get("p", word.get("probability"))),
        })
    return normalized


def words_from_transcript(transcript: dict) -> list[dict]:
    if not isinstance(transcript, dict):
        return []
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return []
    return [
        word
        for segment in segments if isinstance(segment, dict)
        for word in _normalize_words(segment.get("words"))
    ]


def split_units(words: list[dict]) -> list[list[dict]]:
    units = []
    cur = []
    for i, word in enumerate(words):
        if cur:
            prev = words[i - 1]
            text = prev["word"].strip()
            gap = word["start"] - prev["end"]
            if (text.endswith((".", "?", "!"))
                    or (text.endswith((",", ";", ":")) and len(cur) >= 6)
                    or gap >= 0.5 or len(cur) >= 18):
                units.append(cur)
                cur = []
        cur.append(word)
    if cur:
        units.append(cur)
    merged = []
    for unit in units:
        if merged and len(merged[-1]) < 3 and unit[0]["start"] - merged[-1][-1]["end"] < 1.0:
            merged[-1] = merged[-1] + unit
        else:
            merged.append(unit)
    return merged


def build_messages(words: list[dict], questions: list[str]) -> list[dict]:
    transcript = "\n".join(
        f'[{i}] {"".join(word["word"] for word in unit).strip()}'
        for i, unit in enumerate(split_units(words))
    )
    qtext = "\n".join(f"{i + 1}. {question}" for i, question in enumerate(questions))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TRANSCRIPT\n{transcript}\n\nQUESTIONS\n{qtext}\n\nJSON:"},
    ]


def _bounded_span(start, end, duration=None):
    start, end = _finite_number(start), _finite_number(end)
    if start is None or end is None:
        return None
    start, end = max(0.0, start), max(0.0, end)
    if duration is not None:
        duration = _finite_number(duration)
        if duration is None or duration < 0:
            return None
        start, end = min(start, duration), min(end, duration)
    return (start, end) if start < end else None


def sanitize_response(response, count: int, duration: float | None = None) -> ASRQuestionResponseDto:
    if isinstance(response, ASRQuestionResponseDto):
        response = response.model_dump()
    if not isinstance(response, dict):
        return floor_response(count)
    answers = response.get("answers")
    starts = response.get("evidence_start")
    ends = response.get("evidence_end")
    answers = answers if isinstance(answers, list) else []
    starts = starts if isinstance(starts, list) else []
    ends = ends if isinstance(ends, list) else []
    result = floor_response(count)
    for i in range(count):
        if i >= len(answers) or type(answers[i]) is not bool:
            continue
        result.answers[i] = answers[i]
        if answers[i] and i < len(starts) and i < len(ends):
            span = _bounded_span(starts[i], ends[i], duration)
            if span:
                result.evidence_start[i], result.evidence_end[i] = span
    return result


def _norm(text):
    text = text.lower()
    for key, value in SYN.items():
        text = re.sub(r"\b" + re.escape(key) + r"\b", value, text)
    return re.findall(r"[a-z][a-z0-9]*|\d+(?:\.\d+)?", text)


def _numbers(tokens):
    return {Decimal(token) for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?", token)}


def retrieval_response(words, questions, duration: float) -> ASRQuestionResponseDto:
    words = _normalize_words(words)
    if not words or not questions:
        return floor_response(len(questions))
    units = split_units(words)
    tokens = [[token for token in _norm("".join(w["word"] for w in unit)) if token not in STOP]
              for unit in units]
    if not any(tokens):
        return floor_response(len(questions))
    bm25 = BM25Okapi(tokens)
    result = floor_response(len(questions))
    for i, question in enumerate(questions):
        query = [token for token in _norm(question) if token not in STOP]
        lexical = [token for token in query if not re.fullmatch(r"\d+(?:\.\d+)?", token)]
        result.answers[i] = False
        if not lexical:
            continue
        scores = bm25.get_scores(lexical)
        query_terms = set(lexical)
        coverage = [len(query_terms.intersection(unit)) / len(query_terms) for unit in tokens]
        # BM25's IDF is nonpositive with only one or two documents.
        if len(units) < 3:
            best = max(range(len(units)), key=lambda index: (coverage[index], scores[index]))
            answer = (coverage[best] >= 0.6
                      and len(query_terms.intersection(tokens[best])) >= min(2, len(query_terms)))
        else:
            best = max(range(len(units)), key=lambda index: scores[index])
            answer = bool(scores[best] > 2)
        # An unrelated unit containing the requested number must not rescue a near-miss.
        answer = answer and _numbers(query) <= _numbers(tokens[best])
        result.answers[i] = bool(answer)
        if answer:
            span = _bounded_span(units[best][0]["start"], units[best][-1]["end"], duration)
            if span:
                result.evidence_start[i], result.evidence_end[i] = span
    return result


def _clean(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("'", "")).split()


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def fuzzy_span(words, quote, max_extra=3, deadline=None):
    """Use the probe's global window search, keeping the first tied maximum."""
    if _expired(deadline) or not isinstance(quote, str):
        return None
    query = _clean(quote)
    if not query:
        return None
    flat = []
    indices = []
    for i, word in enumerate(words):
        if _expired(deadline):
            return None
        for token in _clean(word["word"]):
            flat.append(token)
            indices.append(i)
    length = len(query)
    query_text = " ".join(query)
    best_score, best = -1, None
    for i in range(len(flat)):
        for size in range(max(1, length - max_extra), length + max_extra + 1):
            if _expired(deadline):
                return None
            end = i + size
            if end > len(flat):
                break
            score = fuzz.ratio(" ".join(flat[i:end]), query_text)
            if score > best_score:
                best_score, best = score, (indices[i], indices[end - 1])
    if best is None or best_score < 70:
        return None
    start, end = best
    return words[start]["start"], words[end]["end"]


def _results(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        try:
            parsed = repair_json(raw)
        except (ValueError, TypeError, RecursionError):
            return []
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        return []
    return parsed["results"]


def _question_id(value, count):
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        try:
            value = int(value)
        except ValueError:
            return None
    return value if type(value) is int and 1 <= value <= count else None


def _answer(value):
    if type(value) is bool:
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("yes", "no"):
            return value == "yes"
    return None


def _unit_span(units, ids):
    if not isinstance(ids, list) or not 1 <= len(ids) <= 2:
        return None
    if any(type(i) is not int or not 0 <= i < len(units) for i in ids):
        return None
    if len(ids) == 2 and abs(ids[0] - ids[1]) != 1:
        return None
    return units[min(ids)][0]["start"], units[max(ids)][-1]["end"]


def answer_response(
    raw: str, words, questions, duration: float,
    fallback: ASRQuestionResponseDto | None = None, start_offset: float = 0.0,
    deadline: float | None = None, alignment: str = 'legacy',
) -> ASRQuestionResponseDto:
    """Keep valid answers; start_offset shifts only model-aligned/cited evidence."""
    if alignment not in ('legacy', 'numeric'):
        raise ValueError(f'Unknown quote alignment: {alignment}')
    words = _normalize_words(words)
    count = len(questions)
    if not words or not count:
        return floor_response(count)
    if fallback is None:
        fallback = retrieval_response(words, questions, duration)
    fallback = sanitize_response(fallback, count, duration)
    if _expired(deadline):
        return fallback
    by_question = {}
    for entry in _results(raw):
        if not isinstance(entry, dict):
            continue
        qid = _question_id(entry.get("q"), count)
        if qid is not None:
            by_question[qid] = None if qid in by_question else entry
    result = fallback.model_copy(deep=True)
    units = split_units(words)
    offset = _finite_number(start_offset)
    offset = offset if offset is not None else 0.0
    for i in range(count):
        entry = by_question.get(i + 1)
        answer = _answer(entry.get("answer")) if entry is not None else None
        if answer is None:
            continue
        result.answers[i] = answer
        result.evidence_start[i] = result.evidence_end[i] = None
        if not answer:
            continue
        if alignment == 'numeric':
            from pipeline.evidence import align_quote

            span = align_quote(words, entry.get('quote'), deadline)
        else:
            span = fuzzy_span(words, entry.get("quote"), deadline=deadline)
        if span:
            span = _bounded_span(span[0] + offset, span[1], duration)
        if span is None and not _expired(deadline):
            span = _unit_span(units, entry.get("units"))
            if span:
                span = _bounded_span(span[0] + offset, span[1], duration)
        if span is None:
            span = _bounded_span(fallback.evidence_start[i], fallback.evidence_end[i], duration)
        if span:
            result.evidence_start[i], result.evidence_end[i] = span
    return result
