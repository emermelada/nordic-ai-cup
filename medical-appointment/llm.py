"""The answerer: a local LLM server, asked one question at a time.

Talks to any OpenAI-compatible chat server on this machine — vLLM or
llama.cpp's ``llama-server`` — so nothing leaves the box, as the rules require.
Select it with ``ANSWERER=llm:answer``. PLAN.md has the serving commands.

    LLM_BASE_URL    http://127.0.0.1:8080/v1
    LLM_MODEL       the name the server knows the model by (vLLM checks it)
    LLM_THINKING    0 (default): answer directly; 1: let a hybrid model reason
                    first; omit: send no template switch at all
    LLM_PARALLEL    10: questions in flight at once

Every question gets its own request with the same prefix — instructions, worked
examples, then the transcript — so the server's prompt cache pays for that once
per conversation and the ten requests run side by side.

The model replies in four labelled lines: the sentences it read, the verbatim
passage that settles the question (whole sentences, the way the annotators
quoted), a one-line check, and yes or no.
The quote becomes the span (``evidence.locate``); the probability the model put
on "yes" at the answer token becomes ``p_yes``, so the pipeline's threshold
works on a real confidence rather than a coin that only ever lands 0 or 1.

The worked examples are real training excerpts, verbatim from the ``base``
transcript. The conversations they came from are listed in
``FEW_SHOT_SOURCES``; ``eval_offline.py`` leaves those out of the score.
"""

import bisect
import concurrent.futures
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from answering import Answer
from asr import Transcript
from evidence import locate
from utils import temporal_iou

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get('LLM_BASE_URL', 'http://127.0.0.1:8080/v1').rstrip('/')
MODEL = os.environ.get('LLM_MODEL', 'local')
THINKING = os.environ.get('LLM_THINKING', '0')
PARALLEL = int(os.environ.get('LLM_PARALLEL', '10'))
MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '2048' if THINKING == '1' else '300'))

# Consensus spans, opt-in: besides the greedy reply, draw this many sampled
# replies per likely-yes question and cite the one whose span overlaps the
# others most — under an IoU score, the most typical span is the best single
# bet. The yes/no call stays the greedy one. 0 is the 0.802 behaviour.
SAMPLES = int(os.environ.get('LLM_SAMPLES', '0'))
SAMPLE_TEMPERATURE = float(os.environ.get('LLM_SAMPLE_TEMPERATURE', '0.7'))
# Below this P(yes) the span will not be sent, so it is not worth sampling for.
SAMPLE_FROM_P_YES = 0.1

# Conversations the worked examples below are taken from.
FEW_SHOT_SOURCES = (
    'sample_4', 'sample_5', 'sample_17', 'sample_18', 'sample_19', 'sample_20', 'sample_71',
)

SYSTEM_PROMPT = """\
You check yes/no questions against the transcript of a recorded consultation \
between a GP and a patient.

The transcript is automatic speech recognition: one numbered sentence per line, \
both speakers mixed together. Numbers and ordinary words are reliable. Drug and \
brand names are often misspelled by sound — "Ibu Medin" is Ibumetin, "pan top \
resolve" is pantoprazole, "Active L" is Activelle, "Aromere" is Airomir — so \
match names by how they sound.

For each question:
1. Find the line or lines that bear on it.
2. Compare every detail of the question with what was actually said: drug, dose, \
unit, frequency, duration, body site, timing, test value, who said it, and \
whether it was actually done or agreed rather than only mentioned or asked \
about. The same topic with any detail different is NO. A subject that never \
comes up is NO. Statement-style questions ("..., right?", "..., didn't it?") \
are checked the same way as any other.
3. Answer yes only if the transcript establishes it.

Reply in exactly this format, four lines, nothing else:
LINES: <first>-<last>, or none
QUOTE: <the verbatim passage that establishes the answer, or none>
CHECK: <one short sentence comparing the question with the transcript>
ANSWER: <yes or no>

{quote_rule}"""

# How to cut the QUOTE. 'whole-sentences' is the rule that scored 0.802 on
# validation (tag medical-v0.802). 'first-statement' also tells the model to
# leave out the confirmations that follow a fact; it went with the 0.777
# validation, so it is opt-in: LLM_QUOTE_RULE=first-statement.
QUOTE_RULES = {
    'whole-sentences': """\
The QUOTE is copied character for character from the transcript, misspellings \
included. Quote the whole sentence that states the fact, from its first word — \
keep openers such as "So," or "Yes," — to its end. When the fact takes several \
sentences — a question and its answer, a list and its confirmation, an \
examination and its finding — quote all of them. Only when a single sentence \
packs several separate facts, quote just the part about the fact asked: from \
where that part starts to its last word.""",
    'first-statement': """\
The QUOTE is copied character for character from the transcript, misspellings \
included. Quote where the fact is first stated: the whole sentence that states \
it, from its first word — keep openers such as "So," or "Yes," — to its end. \
Leave out what follows it: confirmations, repetitions and summaries such as \
"Correct.", "No complications." or "Right, a scar and an infection risk." are \
not part of the quote. When the fact takes several sentences — a question and \
its answer, a list and its confirmation — quote all of them. When a sentence \
goes on to a different fact the question does not ask about, stop before that \
part.""",
}
QUOTE_RULE = os.environ.get('LLM_QUOTE_RULE', 'whole-sentences')
SYSTEM_PROMPT = SYSTEM_PROMPT.replace('{quote_rule}', QUOTE_RULES[QUOTE_RULE])

# (excerpt, question, reply). Verbatim base-model transcript lines, with the
# line numbers they have in their own conversation, and the annotators' own
# quote as the QUOTE. The mix follows the annotations: most quotes are whole
# sentences, a third are cut to one fact of a sentence that states several.
_EXAMPLES = [
    (
        '[18] Your chest and heart both sound normal.\n'
        '[19] Nothing abnormal to report.\n'
        '[20] Good.',
        'Is the heart examination without abnormal findings?',
        'LINES: 19-19\n'
        'QUOTE: Nothing abnormal to report.\n'
        'CHECK: The doctor reports nothing abnormal.\n'
        'ANSWER: yes',
    ),
    (
        '[27] I was worried it might have drifted since last time.\n'
        '[28] Your LDL cholesterol is 2.2 millimoles per liter.\n'
        '[29] Right.',
        'Is the LDL cholesterol 4.2 mmol/L?',
        'LINES: 28-28\n'
        'QUOTE: Your LDL cholesterol is 2.2 millimoles per liter.\n'
        'CHECK: The LDL is 2.2 mmol/L, not 4.2.\n'
        'ANSWER: no',
    ),
    (
        '[8] It tells me we are running on time for once.\n'
        '[9] So, this is your annual follow-up.\n'
        '[10] It is, for the asthma.\n'
        '[11] Exactly that.',
        'Did the patient attend for an annual asthma follow-up?',
        'LINES: 9-10\n'
        'QUOTE: So, this is your annual follow-up. It is, for the asthma.\n'
        'CHECK: The visit is the annual follow-up, and it is for asthma.\n'
        'ANSWER: yes',
    ),
    (
        '[34] I will take that gladly.\n'
        '[35] So my assessment is that your diabetes is stable, and there are no signs of complications.\n'
        '[36] No complications.',
        'Are there no signs of complications?',
        'LINES: 35-35\n'
        'QUOTE: there are no signs of complications.\n'
        'CHECK: The sentence states two facts; the part about complications says there are none.\n'
        'ANSWER: yes',
    ),
    (
        '[26] And you will still renew the ibumet in today?\n'
        '[27] Yes, the prescription is created.\n'
        '[28] Does that plan sound workable to you?',
        'Has the Ibumetin prescription been issued?',
        'LINES: 27-27\n'
        'QUOTE: Yes, the prescription is created.\n'
        'CHECK: Asked about renewing Ibumetin ("ibumet in"), the doctor confirms the prescription is created.\n'
        'ANSWER: yes',
    ),
    (
        '[13] I have almost run out of my painkillers and I was hoping to have the prescriptions renewed.\n'
        '[14] Which painkillers are we talking about?\n'
        '[15] I would like to hear you name them yourself.\n'
        '[16] Pamol and Ibu Metin.',
        'Did the patient ask for morphine to be renewed?',
        'LINES: 13-16\n'
        'QUOTE: Pamol and Ibu Metin.\n'
        'CHECK: The patient asked for Pamol and Ibumetin; morphine is never mentioned.\n'
        'ANSWER: no',
    ),
    (
        '[35] Your gate and standing are normal.\n'
        '[36] I can feel muscle tension in your neck and shoulders.\n'
        '[37] That tension is exactly where it always sits, right across there.',
        'Was muscle tension found in the neck and shoulders?',
        'LINES: 36-36\n'
        'QUOTE: I can feel muscle tension in your neck and shoulders.\n'
        'CHECK: The doctor finds muscle tension in the neck and shoulders.\n'
        'ANSWER: yes',
    ),
    (
        '[18] What does the treatment involve?\n'
        '[19] Sporanox, 100 milligrams daily for two weeks.\n'
        '[20] You take it after a meal.',
        'Should the daily dose be 100 mg?',
        'LINES: 19-19\n'
        'QUOTE: 100 milligrams daily\n'
        'CHECK: The sentence gives drug, dose and duration; the dose part is 100 milligrams daily.\n'
        'ANSWER: yes',
    ),
    (
        '[21] I have been curious about them all week.\n'
        '[22] Your hemoglobin A1c is 42 millimoles per mole.\n'
        '[23] And that is a good one?',
        'Was the HbA1c 42 mmol/mol?',
        'LINES: 22-22\n'
        'QUOTE: Your hemoglobin A1c is 42 millimoles per mole.\n'
        'CHECK: The HbA1c is 42 mmol/mol.\n'
        'ANSWER: yes',
    ),
    (
        '[31] The one you always check for me?\n'
        '[32] Your creatinine is normal.\n'
        '[33] Well, that is all three of them behaving themselves.',
        'Did the creatinine turn out to be elevated?',
        'LINES: 32-32\n'
        'QUOTE: Your creatinine is normal.\n'
        'CHECK: The creatinine is normal, not elevated.\n'
        'ANSWER: no',
    ),
    (
        '[10] All right.\n'
        '[11] And you are still taking Pantoprazole alongside it?\n'
        '[12] Yes, every day with it.\n'
        '[13] Good.',
        'Is the patient also taking Pantoprazole?',
        'LINES: 11-12\n'
        'QUOTE: And you are still taking Pantoprazole alongside it? Yes, every day with it.\n'
        'CHECK: Asked whether they still take Pantoprazole, the patient confirms.\n'
        'ANSWER: yes',
    ),
    (
        '[55] Yes.\n'
        '[56] I am creating prescriptions for both Pamol and Ibu Medin now, and they will be waiting for you at the pharmacy.\n'
        '[57] That is a weight off my mind.',
        'Is Pamol one of the medicines requested?',
        'LINES: 56-56\n'
        'QUOTE: I am creating prescriptions for both Pamol\n'
        'CHECK: The sentence covers two medicines and the pharmacy; the part about Pamol shows it is prescribed.\n'
        'ANSWER: yes',
    ),
    (
        '[35] Which prescriptions will I get?\n'
        '[36] Active L, Aromere and Esomeprizol.\n'
        '[37] All three renewed.\n'
        '[38] The Esomeprizol is the one for the reflux?',
        'Is Airomir among the renewed medicines?',
        'LINES: 36-37\n'
        'QUOTE: Active L, Aromere and Esomeprizol. All three renewed.\n'
        'CHECK: "Aromere" is Airomir, and all three were renewed.\n'
        'ANSWER: yes',
    ),
    (
        '[27] I was worried it might have drifted since last time.\n'
        '[28] Your LDL cholesterol is 2.2 millimoles per liter.\n'
        '[29] Right.',
        'Was the patient discussing their pet?',
        'LINES: none\n'
        'QUOTE: none\n'
        'CHECK: Pets are never discussed.\n'
        'ANSWER: no',
    ),
]

_LINES = re.compile(r'^\s*\**LINES\**\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)
_QUOTE = re.compile(r'^\s*\**QUOTE\**\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)
_ANSWER = re.compile(r'ANSWER\**\s*:\s*\**\s*(yes|no)\b', re.IGNORECASE)
_THINK = re.compile(r'<think>.*?</think>', re.DOTALL)

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL)


def _user_message(transcript_text: str, question: str, excerpt: bool = False) -> str:
    label = 'TRANSCRIPT (excerpt)' if excerpt else 'TRANSCRIPT'
    return f'{label}:\n{transcript_text}\n\nQUESTION: {question}'


def _messages(transcript_text: str, question: str) -> List[Dict[str, str]]:
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for excerpt, example_question, reply in _EXAMPLES:
        messages.append({'role': 'user', 'content': _user_message(excerpt, example_question, excerpt=True)})
        messages.append({'role': 'assistant', 'content': reply})
    messages.append({'role': 'user', 'content': _user_message(transcript_text, question)})
    return messages


def _payload(messages: List[Dict[str, str]], samples: int = 0) -> dict:
    payload = {
        'model': MODEL,
        'messages': messages,
        'temperature': 0.0,
        'max_tokens': MAX_TOKENS,
        'logprobs': True,
        'top_logprobs': 5,
    }
    if samples:
        payload.update(temperature=SAMPLE_TEMPERATURE, n=samples, logprobs=False)
        del payload['top_logprobs']
    if THINKING in ('0', '1'):
        payload['chat_template_kwargs'] = {'enable_thinking': THINKING == '1'}
    return payload


def _parse(content: str) -> Optional[Tuple[Optional[Tuple[int, int]], Optional[str], bool]]:
    """(lines, quote, said_yes) from a reply, or None if it has no answer."""
    content = _THINK.sub('', content)
    answers = _ANSWER.findall(content)
    if not answers:
        return None

    lines = None
    match = _LINES.search(content)
    if match:
        numbers = [int(n) for n in re.findall(r'\d+', match.group(1))]
        if numbers:
            lines = (min(numbers), max(numbers))

    quote = None
    match = _QUOTE.search(content)
    if match:
        quote = match.group(1).strip().strip('"“”\'').strip()
        if not quote or quote.lower() in ('none', 'n/a', '-'):
            quote = None

    return lines, quote, answers[-1].lower() == 'yes'


def _p_yes(choice: dict) -> Optional[float]:
    """Probability of "yes" at the final answer token, from the returned logprobs.

    Rebuilds the text from the tokens themselves rather than trusting
    ``content`` offsets, so it also works when a server splits reasoning out.
    """
    tokens = ((choice.get('logprobs') or {}).get('content')) or []
    if not tokens:
        return None

    text, ends = '', []
    for token in tokens:
        text += token.get('token') or ''
        ends.append(len(text))

    matches = list(_ANSWER.finditer(text))
    if not matches:
        return None

    position = bisect.bisect_right(ends, matches[-1].start(1))
    if position >= len(tokens):
        return None

    yes = no = 0.0
    for alternative in tokens[position].get('top_logprobs') or []:
        word = (alternative.get('token') or '').strip().strip('"*').lower()
        probability = math.exp(alternative.get('logprob', -math.inf))
        if word.startswith('y'):
            yes += probability
        elif word.startswith('n'):
            no += probability

    return yes / (yes + no) if yes + no > 0 else None


def _consensus(
    transcript: Transcript,
    question: str,
    greedy: Answer,
    transcript_text: str,
    timeout: float,
) -> Answer:
    """The greedy answer, citing whichever yes-reply's span overlaps the rest most."""
    try:
        response = requests.post(
            f'{BASE_URL}/chat/completions',
            json=_payload(_messages(transcript_text, question), samples=SAMPLES),
            timeout=timeout,
        )
        response.raise_for_status()
        choices = response.json()['choices']
    except Exception as exc:
        logger.warning('Sampling failed for %r, keeping the greedy quote: %s', question, exc)
        return greedy

    # The greedy citation first, so it wins ties.
    candidates = [(greedy.lines, greedy.quote)]
    for choice in choices:
        parsed = _parse(choice.get('message', {}).get('content') or '')
        if parsed and parsed[2] and (parsed[0] or parsed[1]):
            candidates.append((parsed[0], parsed[1]))

    spans = [locate(transcript, quote, lines).span for lines, quote in candidates]
    usable = [n for n, span in enumerate(spans) if span]
    if len(usable) < 3:
        return greedy

    best = max(usable, key=lambda n: sum(temporal_iou(spans[n], spans[m]) for m in usable))
    lines, quote = candidates[best]
    return Answer(p_yes=greedy.p_yes, lines=lines, quote=quote)


def _answer_one(
    transcript_text: str,
    question: str,
    timeout: float,
    transcript: Optional[Transcript] = None,
) -> Optional[Answer]:
    started = time.monotonic()
    try:
        response = requests.post(
            f'{BASE_URL}/chat/completions',
            json=_payload(_messages(transcript_text, question)),
            timeout=timeout,
        )
        response.raise_for_status()
        choice = response.json()['choices'][0]
    except Exception as exc:
        logger.warning('LLM request failed for %r: %s', question, exc)
        return None

    parsed = _parse(choice.get('message', {}).get('content') or '')
    if parsed is None:
        logger.warning('No answer in LLM reply for %r', question)
        return None

    lines, quote, said_yes = parsed
    p_yes = _p_yes(choice)
    greedy = Answer(
        p_yes=float(said_yes) if p_yes is None else p_yes,
        lines=lines,
        quote=quote,
    )

    remaining = timeout - (time.monotonic() - started)
    if SAMPLES and transcript is not None and greedy.p_yes >= SAMPLE_FROM_P_YES and remaining > 1.0:
        return _consensus(transcript, question, greedy, transcript_text, remaining)
    return greedy


def answer(transcript: Transcript, questions: List[str], deadline: float) -> List[Optional[Answer]]:
    """All questions in parallel; whatever is not back by the deadline is None."""
    transcript_text = transcript.render()
    timeout = max(deadline - time.monotonic() - 0.5, 1.0)

    futures = [_pool.submit(_answer_one, transcript_text, q, timeout, transcript) for q in questions]
    done, _ = concurrent.futures.wait(futures, timeout=max(deadline - time.monotonic(), 0.0))
    return [future.result() if future in done else None for future in futures]


def _warm_up() -> None:
    """One real request at import, so the first scored conversation is not the slowest."""
    try:
        requests.get(f'{BASE_URL}/models', timeout=3).raise_for_status()
    except Exception as exc:
        logger.warning('No LLM server at %s (%s): answers fall back to lexical until there is.',
                       BASE_URL, exc)
        return

    excerpt, question, _ = _EXAMPLES[0]
    started = time.monotonic()
    result = _answer_one(excerpt, question, timeout=120)
    logger.info('LLM warm (%s at %s) in %.1fs: %s', MODEL, BASE_URL,
                time.monotonic() - started, result)


_warm_up()
