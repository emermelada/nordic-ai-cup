"""Second-stage evidence selection over a sentence table.

The answering pass decides yes/no and gives a draft quote per yes. This stage re-reads the
transcript as numbered sentences and copies the passage the annotation convention marks:
the minimal self-contained clause, a question plus its bare reply, or the mention whose
wording the question paraphrases. Its output only ever replaces the span of a retained yes.
"""

import os
import re

from rank_bm25 import BM25Okapi

from pipeline.core import STOP, _expired, _norm, _question_id, _results, sanitize_response
from pipeline.evidence import align_quote, refine_evidence

SENTENCE_GAP_SECONDS = 1.0
ANCHOR_CONTEXT = 2
LEXICAL_HITS = 3
BRIDGE_GAP_SECONDS = 6.0
MAX_QUOTE_TOKENS = 60

CONVENTIONS = '''You extract the evidence passage for questions whose answer is already known to be YES, from the transcript of a doctor-patient consultation. The transcript is split into numbered sentences. For each question, copy the EXACT contiguous words that establish the fact, as the annotator marked them.

Conventions of the annotation:
1. Self-contained. The quote names its subject and states the fact about it. A sentence such as "It came back normal.", "That was normal as well." or "Blocked, aching, the whole business." is never quoted alone: start at the preceding sentence that names the subject ("Your glucose, which is your blood sugar. That was normal as well.") and quote through the fact.
2. One clause when a sentence joins two facts. For "Your blood pressure is normal, and your foot status is normal." quote only the clause the question asks about ("your foot status is normal."). Keep both clauses when the question covers both ("Your tonsils are red and there is coating on them." for a question about coated tonsils).
3. Drop a separate acknowledgement before the fact ("Thank you.", "Good.", "Correct.", "Of course.", "Exactly.", "Nothing has."), but keep the fact sentence's own opening words ("So, my assessment is that", "Overall,", "Honestly,").
4. Nothing after the fact: no following explanation, one-word echo ("Stable."), advice, or next sentence.
5. Question plus bare answer. When the reply is only a bare confirmation of one or two words ("Yes.", "None.", "Observation.", "Yes, every day."), quote from the first word of the question to the last word of the reply, and never quote the question alone. When the reply is a full clause that states the fact itself ("No, none that I know of.", "No, nothing new.", "Well, actually, I feel fine.", "No fever."), quote only the reply.
6. The mention the question was written from. When the fact appears more than once, choose the mention whose wording is closest to the question (same terms, numbers and polarity), preferring an explicit statement of the finding, decision or plan over a passing remark, a paraphrase or a later recap. Never choose a passage whose details contradict the question.

Example transcript:
[3] How have you been feeling? [4] Honestly, fine. [5] Good. Your blood pressure is normal, and your feet look fine. [6] Any side effects from the tablets? [7] None. [8] Then the kidney test. [9] I had wondered about that one. [10] It came back normal. [11] Overall, the diabetes is stable, so we make no changes today. [12] I will renew the metformin, 500 mg twice daily. [13] Twice daily. [14] And your cholesterol is up a little, so keep an eye on the diet. [15] Do you smoke? [16] No, I have never smoked.
Example evidence:
Does the patient feel fine? -> "How have you been feeling? Honestly, fine."
Was the blood pressure normal? -> "Your blood pressure is normal,"
Is the patient free of side effects? -> "Any side effects from the tablets? None."
Was the kidney test normal? -> "Then the kidney test. I had wondered about that one. It came back normal."
Is the diabetes stable? -> "Overall, the diabetes is stable,"
Will the treatment continue unchanged? -> "so we make no changes today."
Is metformin prescribed at 500 mg twice a day? -> "I will renew the metformin, 500 mg twice daily."
Is the cholesterol slightly raised? -> "your cholesterol is up a little,"
Is the patient a non-smoker? -> "No, I have never smoked."
'''

REFINE_SYSTEM_V3 = CONVENTIONS + '''
For each question you are given a draft quote. Keep it when it already follows the conventions; otherwise trim it, extend it, or replace it with the right mention. Quote transcript words exactly, never paraphrase.
Respond with minified JSON only: {"results":[{"q":1,"quote":"..."},...]}'''

# The annotators' own convention, measured on their quotes by the team's other branch
# (76% one sentence, 86% start at a sentence start, 81% end at a sentence end), with their
# real quotes as examples in the base-model transcript's wording.
ANNOT_EXAMPLE_SOURCES = ('sample_4', 'sample_5', 'sample_17', 'sample_18', 'sample_19', 'sample_20', 'sample_71')
REFINE_SYSTEM_ANNOT = '''You mark the evidence passage for questions whose answer is already known to be YES, in the transcript of a recorded GP consultation. The transcript is automatic speech recognition, one numbered sentence per line, both speakers mixed together. Numbers and ordinary words are reliable; drug and brand names are often misspelled by sound ("Ibu Medin" is Ibumetin, "pan top resolve" is pantoprazole, "Active L" is Activelle, "Aromere" is Airomir), so match names by how they sound.

For each question, copy character for character the passage that establishes the fact, misspellings included, the way the annotators quoted:
- Quote the whole sentence that states the fact, from its first word (keep openers such as "So," or "Yes,") to its end.
- When the fact takes several sentences (a question and its answer, a list and its confirmation, an examination and its finding), quote all of them together, in order.
- Only when a single sentence packs several separate facts, quote just the part about the fact asked: from where that part starts to its last word.
- When the fact is stated more than once, quote the statement itself (the finding, decision or plan as it is given), not a passing mention or a later recap.
Each question comes with a draft quote from a first pass: keep it when it already follows these rules, otherwise fix it.

Examples, from other consultations (excerpt, then question -> quote):
[18] Your chest and heart both sound normal. [19] Nothing abnormal to report. [20] Good.
Is the heart examination without abnormal findings? -> "Nothing abnormal to report."
[8] It tells me we are running on time for once. [9] So, this is your annual follow-up. [10] It is, for the asthma. [11] Exactly that.
Did the patient attend for an annual asthma follow-up? -> "So, this is your annual follow-up. It is, for the asthma."
[34] I will take that gladly. [35] So my assessment is that your diabetes is stable, and there are no signs of complications. [36] No complications.
Are there no signs of complications? -> "there are no signs of complications."
[26] And you will still renew the ibumet in today? [27] Yes, the prescription is created. [28] Does that plan sound workable to you?
Has the Ibumetin prescription been issued? -> "Yes, the prescription is created."
[35] Your gate and standing are normal. [36] I can feel muscle tension in your neck and shoulders. [37] That tension is exactly where it always sits, right across there.
Was muscle tension found in the neck and shoulders? -> "I can feel muscle tension in your neck and shoulders."
[18] What does the treatment involve? [19] Sporanox, 100 milligrams daily for two weeks. [20] You take it after a meal.
Should the daily dose be 100 mg? -> "100 milligrams daily"
[21] I have been curious about them all week. [22] Your hemoglobin A1c is 42 millimoles per mole. [23] And that is a good one?
Was the HbA1c 42 mmol/mol? -> "Your hemoglobin A1c is 42 millimoles per mole."
[10] All right. [11] And you are still taking Pantoprazole alongside it? [12] Yes, every day with it. [13] Good.
Is the patient also taking Pantoprazole? -> "And you are still taking Pantoprazole alongside it? Yes, every day with it."
[55] Yes. [56] I am creating prescriptions for both Pamol and Ibu Medin now, and they will be waiting for you at the pharmacy. [57] That is a weight off my mind.
Is Pamol one of the medicines requested? -> "I am creating prescriptions for both Pamol"
[35] Which prescriptions will I get? [36] Active L, Aromere and Esomeprizol. [37] All three renewed. [38] The Esomeprizol is the one for the reflux?
Is Airomir among the renewed medicines? -> "Active L, Aromere and Esomeprizol. All three renewed."
Respond with minified JSON only: {"results":[{"q":1,"quote":"..."},...]}'''

# Every fully-missed span in the v3 measurement was a *different valid mention* of the same
# fact: the annotator wrote each question while reading one passage, so the gold is the
# mention whose wording the question echoes, not the most clinical or the earliest one.
# This prompt makes that the first rule and drops v3's "prefer the finding over a passing
# remark", which pushed the other way.
REFINE_SYSTEM_MATCH = '''You mark the evidence passage for questions whose answer is already known to be YES, in the transcript of a recorded GP consultation. The transcript is speech recognition, one numbered sentence per line, both speakers mixed together. Drug, brand and condition names are written by sound — "Aromere" is Airomir, "Ibu Medin" is Ibumetin, "Active L" is Activelle, "Isomeprazole" is Esomeprazole, "Malaskam Contegiosum" is molluscum contagiosum — so match names by how they sound.

Each question was written by someone who was reading ONE passage of this transcript and turned that passage into a question. Your job is to find that passage again and copy it out.

1. Choose the passage by its wording. When the same fact is stated more than once — by both speakers, or as a statement and again as a confirmation, a recap or a plan — pick the one whose words the question echoes most closely, including their form:
   "Is the heart examination without abnormal findings?" -> "Nothing abnormal to report." (not "Your chest and heart both sound normal.")
   "Did the patient get a tick bite?" -> "A tick bite." (not "I found a tick attached to my leg.")
   "Is this vaccination something the patient receives annually?" -> "You are here for your yearly influenza vaccination. Is that correct? Yes, the annual one." (not "the seasonal flu vaccine that we offer once a year")
   Neither the earliest mention nor the doctor's clinical phrasing wins by default. The closest wording does.
2. Quote whole sentences. Copy the sentence that states the fact from its first word — keep openers such as "So," or "Yes," — to its end. When the fact takes several sentences, such as a question and its answer or a list and its confirmation, quote all of them in order.
3. Cut inside a sentence only when one sentence states several separate facts and the question asks about one of them: quote from where that part starts to its last word. "Your blood pressure is normal, and your foot status is normal." -> "your foot status is normal."
4. Stop at the fact: no following explanation, no one-word echo of it, no unrelated next sentence.
5. Copy character for character, misspellings included.

Each question comes with a draft quote from a first pass. Keep it when it already follows these rules; otherwise fix it.
Respond with minified JSON only: {"results":[{"q":1,"quote":"..."},...]}'''

# The v3 measurement's two biggest buckets were spans that stop before the fact's own
# confirmation (35) and spans that cite a different mention than the question was written
# from (20). This prompt keeps v3's cut-inside-a-sentence rule but replaces "nothing after
# the fact" with the exchange rule, and makes wording match decide between mentions. Every
# example is from the designated example-source conversations, so the held-out 320-question
# figure the harness also prints stays free of them.
REFINE_SYSTEM_EXCHANGE = '''You mark the evidence passage for questions whose answer is already known to be YES, in the transcript of a recorded GP consultation. The transcript is speech recognition, one numbered sentence per line, both speakers mixed together. Drug, brand and condition names are written by sound — "Aromere" is Airomir, "Ibu Medin" is Ibumetin, "Active L" is Activelle, "Isomeprazole" is Esomeprazole, "Malaskam Contegiosum" is molluscum contagiosum — so match names by how they sound.

Each question was written by someone reading ONE passage of this transcript and turning that passage into a question. Find that passage again and copy it out, by these rules:

1. Wording decides which mention. When the same fact is stated more than once, pick the passage whose words the question echoes most closely, including their form. Neither the earliest mention nor the doctor's clinical phrasing wins by default.
   "Should the daily dose be 100 mg?" -> "100 milligrams daily" (the dose as the question states it)
   "Will the treatment last two weeks?" -> "After a meal every day for two weeks." (not the earlier "daily for two weeks")
2. Take the whole exchange that settles the fact, not the shortest sentence in it. When the neighbouring sentences are about that same fact — the question that prompted it, the other speaker's confirmation, an echo of it, an immediate recap — include them, in order, as one contiguous passage.
   "Is the patient also taking Pantoprazole?" -> "And you are still taking Pantoprazole alongside it? Yes, every day with it."
   "Is Airomir among the renewed medicines?" -> "Active L, Aromere and Esomeprizol. All three renewed."
   "Were the lungs found to be normal on auscultation?" -> "Your chest and heart both sound normal. Nothing abnormal to report."
   "Was the patient listened to with a stethoscope?" -> "Let me listen to your chest and your heart, and then we will talk. Go ahead. Breathe normally for me. And again. Is it all right? Your chest and heart both sound normal."
   Stop where the topic changes. A sentence about a different finding, a further explanation of why, or advice that follows is not part of the passage.
3. Quote whole sentences: from the first word of the first sentence, openers such as "So," or "Yes," included, to the end of the last.
4. Cut inside a sentence only when one sentence states several separate facts and the question asks about one of them: quote from where that part starts to its last word.
   "Is Pamol one of the medicines requested?" -> "I am creating prescriptions for both Pamol"
   "Are there no signs of complications?" -> "there are no signs of complications."
5. Copy character for character, misspellings included.

Each question comes with a draft quote from a first pass. Keep it when it already follows these rules; otherwise fix it.
Respond with minified JSON only: {"results":[{"q":1,"quote":"..."},...]}'''

REFINE_PROMPTS = {'v3': REFINE_SYSTEM_V3, 'annot': REFINE_SYSTEM_ANNOT,
                  'match': REFINE_SYSTEM_MATCH, 'exchange': REFINE_SYSTEM_EXCHANGE}


def refine_system():
    return REFINE_PROMPTS[os.environ.get('MEDICAL_EVIDENCE_PROMPT', 'v3')]


REFINE_SYSTEM = REFINE_SYSTEM_V3

WINDOW_SYSTEM = ('You extract the evidence passage for a question whose answer is YES, from the transcript of a '
                 'doctor-patient consultation split into numbered sentences. Reply with the exact contiguous transcript '
                 'words that establish the fact, copied verbatim, and nothing else.')


def _text(word):
    return word['word'].strip()


def sentence_ranges(words):
    """Inclusive word-index ranges split at sentence punctuation or a long pause.

    In the annotators' coordinate system (MEDICAL_ASR_MODE=base) the split is on end
    punctuation only, which is the unit their quotes follow.
    """
    if os.environ.get('MEDICAL_ASR_MODE') == 'base':
        from pipeline.base_asr import sentence_ranges as base_ranges

        return base_ranges(words)
    ranges, start = [], 0
    for index, word in enumerate(words):
        last = index + 1 == len(words)
        gap = 0.0 if last else words[index + 1]['start'] - word['end']
        if last or _text(word).endswith(('.', '?', '!')) or gap >= SENTENCE_GAP_SECONDS:
            ranges.append((start, index))
            start = index + 1
    return ranges


def sentence_text(words, span):
    return ' '.join(_text(word) for word in words[span[0]:span[1] + 1])


def _content(text):
    return [token for token in _norm(text) if token not in STOP]


def draft_quotes(raw, count):
    quotes = {}
    for entry in _results(raw):
        if not isinstance(entry, dict):
            continue
        qid = _question_id(entry.get('q'), count)
        if qid is not None and isinstance(entry.get('quote'), str):
            quotes[qid] = ' '.join(entry['quote'].split())
    return quotes


def lexical_span(words, question, sentences=None):
    """The sentence sharing the most content words with the question, as a time span.

    A yes with no span scores nothing on the larger half of the mark, while any span scores
    at least nothing, so this is the floor for a yes the evidence stage could not place.
    """
    sentences = sentences if sentences is not None else sentence_ranges(words)
    wanted = set(_content(question or ''))
    if not sentences or not wanted:
        return None
    best, best_overlap = None, 0
    for span in sentences:
        overlap = len(wanted & set(_content(sentence_text(words, span))))
        if overlap > best_overlap:
            best, best_overlap = span, overlap
    if best is None:
        return None
    return words[best[0]]['start'], words[best[1]]['end']


def render_sentences(words, sentences=None):
    """The transcript as the numbered sentences an evidence prompt reads."""
    sentences = sentences if sentences is not None else sentence_ranges(words)
    return '\n'.join(f'[{k}] {sentence_text(words, span)}' for k, span in enumerate(sentences))


def perq_system():
    """The selected conventions, asking for one bare quote instead of a JSON batch."""
    conventions = refine_system().rsplit('Respond with minified JSON only', 1)[0].rstrip()
    return (conventions + '\n'
            'Reply with that one passage, copied exactly from the transcript, and nothing else: '
            'no quotation marks, no line number, no explanation.')


def build_perq_messages(transcript_text, question, draft='', examples=None):
    """One question against the whole transcript; the reply is the quote itself."""
    if examples is None:
        system = perq_system()
    else:
        from pipeline.span_examples import retrieved_system

        system = retrieved_system(examples)
    draft_line = f'\ndraft: "{draft}"' if draft else ''
    user = f'TRANSCRIPT\n{transcript_text}\n\nQUESTION (answered yes): {question}{draft_line}\n\nEVIDENCE:'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def build_refine_messages(words, questions, answers, drafts):
    """Whole transcript as numbered sentences, every yes question with its draft quote."""
    sentences = sentence_ranges(words)
    transcript = '\n'.join(f'[{k}] {sentence_text(words, span)}' for k, span in enumerate(sentences))
    listed = [f'{i + 1}. {question}\n   draft: "{drafts.get(i + 1, "")}"'
              for i, question in enumerate(questions) if answers[i]]
    user = f'TRANSCRIPT\n{transcript}\n\nQUESTIONS (all answered yes)\n' + '\n'.join(listed) + '\n\nJSON:'
    return [{'role': 'system', 'content': refine_system()}, {'role': 'user', 'content': user}]


def focus_sentences(words, sentences, anchor, question, ranker=None):
    """Sentence indices around the draft span plus the lexically closest sentences."""
    focus = set()
    if anchor is not None and anchor[0] is not None:
        for k, (a, b) in enumerate(sentences):
            if words[b]['end'] > anchor[0] + 1e-6 and words[a]['start'] < anchor[1] - 1e-6:
                focus.update(range(k - ANCHOR_CONTEXT, k + ANCHOR_CONTEXT + 1))
    if ranker is None:
        ranker = BM25Okapi([_content(sentence_text(words, span)) for span in sentences])
    scores = ranker.get_scores(_content(question))
    for k in sorted(range(len(sentences)), key=lambda k: -scores[k])[:LEXICAL_HITS]:
        if scores[k] > 0:
            focus.update((k - 1, k, k + 1))
    return sorted(k for k in focus if 0 <= k < len(sentences))


def build_window_messages(words, question, anchor, sentences=None, ranker=None):
    """One question with the local sentence window; the reply is the bare quote."""
    sentences = sentences or sentence_ranges(words)
    lines, previous = [], None
    for k in focus_sentences(words, sentences, anchor, question, ranker):
        if previous is not None and k != previous + 1:
            lines.append('...')
        lines.append(f'[{k}] {sentence_text(words, sentences[k])}')
        previous = k
    user = f'QUESTION: {question}\n' + '\n'.join(lines) + '\nEVIDENCE:'
    return [{'role': 'system', 'content': WINDOW_SYSTEM}, {'role': 'user', 'content': user}]


def align_bridged(words, quote, deadline=None):
    """Align a multi-sentence quote piecewise so a skipped sentence does not truncate the span."""
    single = align_quote(words, quote, deadline)
    if not isinstance(quote, str) or _expired(deadline):
        return single
    parts = [part for part in re.split(r'(?<=[.?!])\s+', quote.strip())
             if len(re.findall(r'[A-Za-z0-9]+', part)) >= 2]
    if len(parts) < 2:
        return single
    spans = [align_quote(words, part, deadline) for part in parts]
    if any(span is None for span in spans):
        return single
    for first, second in zip(spans, spans[1:]):
        if second[0] < first[1] - 1e-6 or second[0] - first[1] > BRIDGE_GAP_SECONDS:
            return single
    union = (spans[0][0], spans[-1][1])
    if single is None or (single[1] - single[0]) < 0.9 * (union[1] - union[0]):
        return union
    return single


def parse_refine_output(raw, count):
    """Quotes keyed by question number; duplicated question numbers are dropped."""
    quotes, seen = {}, set()
    for entry in _results(raw):
        if not isinstance(entry, dict):
            continue
        qid = _question_id(entry.get('q'), count)
        if qid is None:
            continue
        if qid in seen:
            quotes.pop(qid, None)
            continue
        seen.add(qid)
        if isinstance(entry.get('quote'), str):
            quotes[qid] = entry['quote']
    return quotes


def clean_window_output(text):
    if not isinstance(text, str):
        return ''
    text = text.split('</think>')[-1].strip().strip('"').strip()
    return ' '.join(text.split()[:MAX_QUOTE_TOKENS])


def apply_quotes(quotes, response, words, duration, envelope=None, deadline=None, extend_replies=True):
    """Replace the span of each retained yes whose new quote aligns; answers never change."""
    result = response.model_copy(deep=True)
    count = len(result.answers)
    proposal = result.model_copy(deep=True)
    proposal.answers = [False] * count
    proposal.evidence_start = [None] * count
    proposal.evidence_end = [None] * count
    replaced = []
    for qid, quote in quotes.items():
        if _expired(deadline):
            break
        index = qid - 1
        if not 0 <= index < count or not result.answers[index]:
            continue
        span = align_bridged(words, quote, deadline)
        if span is None:
            continue
        proposal.answers[index] = True
        proposal.evidence_start[index], proposal.evidence_end[index] = span
        replaced.append(qid)
    proposal = sanitize_response(refine_evidence(proposal, words, envelope, extend_replies), count, duration)
    for qid in replaced:
        index = qid - 1
        if proposal.evidence_start[index] is not None:
            result.evidence_start[index] = proposal.evidence_start[index]
            result.evidence_end[index] = proposal.evidence_end[index]
    return result


def apply_evidence(evidence, response, words, duration, envelope=None, deadline=None, extend_replies=True):
    """Apply a worker evidence frame; anything malformed leaves the response unchanged."""
    if not isinstance(evidence, dict):
        return response
    count = len(response.answers)
    mode = evidence.get('mode')
    if mode == 'locate':
        from pipeline.locate import apply_located

        return apply_located(evidence, response, words, duration, envelope, deadline, extend_replies)
    if mode == 'refine':
        quotes = parse_refine_output(evidence.get('raw'), count)
    elif mode in ('window', 'perq'):
        outputs = evidence.get('outputs')
        quotes = {}
        if isinstance(outputs, dict):
            for key, text in outputs.items():
                qid = _question_id(key, count)
                if qid is not None:
                    quotes[qid] = clean_window_output(text)
    else:
        return response
    return apply_quotes(quotes, response, words, duration, envelope, deadline, extend_replies)
