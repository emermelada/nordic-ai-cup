"""The endpoint's model: exact-coordinate ASR, a pluggable answerer, and span cutting.

    transcribe (asr.py) -> answer (answering.py) -> decide + locate (here)

The decision and the span cutting live here, not in the answerer, so that
``eval_offline.py`` scores exactly what the endpoint would send. Two scoring
rules are baked in (PLAN.md has the arithmetic):

* **Yes is the cheaper mistake.** A yes on a true positive earns the answer and
  the evidence; a no on a negative earns only the answer. So the call is yes
  when P(yes) >= ``YES_THRESHOLD``, about 0.22 rather than 0.5.
* **A yes always carries a span.** A yes with no span scores nothing on the
  larger half, so if the model's citation cannot be placed, the best
  word-overlap sentence is sent instead.

``predict`` never raises and always answers every question: a request that
fails loses all ten questions, while a guess keeps half of them on average.
"""

import concurrent.futures
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

import asr
from answering import Answer, Answerer, load, lexical
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from evidence import Evidence, locate
from utils import Span, decode_audio

logger = logging.getLogger(__name__)

# tau = 0.4 / (0.8 + 1.2 t), where t is the tIoU a correct yes usually earns.
# 0.22 assumes t of about 0.85; tune it with eval_offline.py's threshold sweep.
YES_THRESHOLD = float(os.environ.get('YES_THRESHOLD', '0.22'))

# The service allows 60 s per conversation on average across the attempt, so
# stop waiting for the answerer well before that and send what we have.
ANSWER_DEADLINE_SECONDS = float(os.environ.get('ANSWER_DEADLINE_SECONDS', '50'))

# The LLM by default, so a forgotten environment variable cannot ship the
# no-model floor. With no LLM server running it degrades to lexical per question.
ANSWERER_SPEC = os.environ.get('ANSWERER', 'llm:answer')

_answerer = load(ANSWERER_SPEC)
# More than one worker, so an answerer still stuck past its deadline on one
# conversation cannot hold up the next.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def evidence_for(transcript: asr.Transcript, answer: Answer, question: str) -> Evidence:
    """Where an answer points, as a span — quote first, then its sentences."""
    return locate(transcript, answer.quote, answer.lines, question)


def finalize(
    transcript: asr.Transcript,
    questions: List[str],
    answers: List[Optional[Answer]],
    threshold: float = YES_THRESHOLD,
) -> Tuple[List[bool], List[Optional[Span]], List[str]]:
    """Answers -> (yes/no, span, how the span was found) per question."""
    backups = lexical(transcript, questions, deadline=float('inf'))
    decisions, spans, methods = [], [], []

    for question, answer, backup in zip(questions, answers, backups):
        answer = answer or backup

        if answer.p_yes < threshold:
            decisions.append(False)
            spans.append(None)
            methods.append('no')
            continue

        evidence = evidence_for(transcript, answer, question)
        if evidence.span is None:
            evidence = evidence_for(transcript, backup, question)

        decisions.append(True)
        spans.append(evidence.span)
        methods.append(evidence.method)

    return decisions, spans, methods


def answer_before(
    transcript: asr.Transcript,
    questions: List[str],
    deadline: float,
    answerer: Optional[Answerer] = None,
) -> List[Optional[Answer]]:
    """Run the answerer, but never past the deadline and never raising."""
    future = _executor.submit(answerer or _answerer, transcript, questions, deadline)

    try:
        answers = future.result(timeout=max(deadline - time.monotonic(), 0.1))
    except concurrent.futures.TimeoutError:
        logger.warning('Answerer missed the deadline; falling back for all questions.')
        return [None] * len(questions)
    except Exception:
        logger.exception('Answerer failed; falling back for all questions.')
        return [None] * len(questions)

    answers = list(answers or [])[:len(questions)]
    return answers + [None] * (len(questions) - len(answers))


def _response(decisions: List[bool], spans: List[Optional[Span]]) -> ASRQuestionResponseDto:
    return ASRQuestionResponseDto(
        answers=decisions,
        evidence_start=[span[0] if span else None for span in spans],
        evidence_end=[span[1] if span else None for span in spans],
    )


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation."""
    started = time.monotonic()
    deadline = started + ANSWER_DEADLINE_SECONDS
    questions = request.questions

    try:
        transcript = asr.transcribe(decode_audio(request.audio_base64))
    except Exception:
        # Nothing to point at. Without evidence yes and no are worth the same,
        # so keep the baseline's all-yes rather than invent anything.
        logger.exception('Transcription failed for %s', request.audio_filename)
        return _response([True] * len(questions), [None] * len(questions))

    transcribed = time.monotonic()
    answers = answer_before(transcript, questions, deadline)
    decisions, spans, methods = finalize(transcript, questions, answers)

    logger.info(
        '%s: %d words, %d sentences, asr %.1fs, answer %.1fs, yes %d/%d, evidence %s',
        request.audio_filename,
        len(transcript.words),
        len(transcript.sentences),
        transcribed - started,
        time.monotonic() - transcribed,
        sum(decisions),
        len(decisions),
        ','.join(methods),
    )
    return _response(decisions, spans)


def _warm_up() -> None:
    """Load the model and run it once, so the first scored request is not the slowest."""
    model = asr.load_model()
    segments, _ = model.transcribe(
        np.zeros(16000, dtype=np.float32), language='en', word_timestamps=True
    )
    list(segments)
    logger.info('ASR warm (%s, %s, %d threads); answerer %s',
                asr.MODEL_NAME, asr.COMPUTE_TYPE, asr.CPU_THREADS, ANSWERER_SPEC)


_warm_up()
