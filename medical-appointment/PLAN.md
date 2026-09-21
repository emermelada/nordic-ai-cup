# Medical Appointment: Implementation Plan

> Historical implementation plan. See [UPGRADE_PLAN.md](UPGRADE_PLAN.md) for the current baseline, a reassessment of earlier experiments, and the roadmap toward 0.90 raw.

Written 2026-09-17 (competition day 1 of 4; deadline **Sunday 2026-09-20 16:00 CEST**).
Audience: whoever implements this next, human or AI. Everything here is grounded in the
files in this folder (`README.md`, `local_evaluator.py`, `utils.py`, `dtos.py`,
`data/question_train.csv`) plus the measurements in section 2. Where a decision is
still open it says so and names the experiment that settles it.

---

## 0. TL;DR

1. **The score is 0.4 × accuracy + 0.6 × mean tIoU over the gold-yes questions.** Evidence
   localisation is the bigger half and, as measured below, the only hard half.
2. **Answering is solved.** On the 39 training conversations, whisper-large-v3-turbo
   transcripts + a single JSON call to a local Qwen3-8B (4-bit, MLX) answer **386 / 390**
   questions correctly (hard negatives 140 / 142, off-topic 53 / 53), with no JSON
   failures, in **~12 s per conversation** end to end on this Mac. Do not spend the
   competition on bigger LLMs unless the platform's validation accuracy comes back
   clearly below 0.95.
3. **Localisation is where every remaining point is.** Gold spans are short clauses
   (median 2.9 s) on a 20 ms grid, i.e. whisper word timestamps. The best possible word
   span scores 0.93; whisper's own segments cap at 0.62; the LLM's quoted evidence,
   aligned back to words, currently scores **0.57** (score **0.72**). The loss is split
   between *citing a different mention of the fact than the annotator* (30 / 195 score
   zero) and *quoting a sentence too much or a clause too little* (56 too long, 44 too
   short). Fixing that is the plan for days 2–3 (section 8).
4. **Pipeline:** MP3 → whisper word timestamps (+ medical Parakeet as a 1 s second
   opinion) → clause units → one LLM JSON call (answers + cited units + exact quotes,
   with candidate passages) → passage re-ranking → quote-to-word alignment → response.
   Every stage has a fallback (retrieval-only path scores 0.51 with no LLM at all).
5. **Iterate offline, deploy early.** Cache transcripts, iterate on prompts/spans with the
   offline evaluator (minutes per run), use `local_evaluator.py` only for timing. Get a
   reachable endpoint up today and validate on the platform; keep a second host ready.

---

## 1. The task, precisely

### 1.1 Protocol (from `dtos.py`, `api.py`, `README.md`)

- One `POST /predict` per conversation. Body: `audio_base64` (plain base64 of an MP3,
  no `data:` prefix), `audio_filename`, `questions` (ten English yes/no questions).
- Reply: `answers: list[bool]`, `evidence_start: list[float|null]`,
  `evidence_end: list[float|null]`, all three of length ten, matched by position.
  `null`/`null` for every `false`; a span in seconds for every `true`.
- `utils.validate_response` is called in `api.py` after `predict`; if it raises, the
  request fails and all ten questions are scored wrong. So `predict` must never raise
  and must always produce a well-formed reply.
- Budget: 60 s per request **and** 60 s × N for the whole attempt. Five consecutive
  timeouts abort the attempt. There is no warm-up request, so models must be loaded
  and exercised at import time.
- Validation: 19 conversations, unlimited attempts. Evaluation: 38 conversations,
  **one attempt**. Both are exactly balanced yes/no.
- Rules: no cloud API in the request path. Anything goes during development
  (hosted LLMs for labelling, synthetic data, etc.).

### 1.2 Scoring (from `utils.py`, `local_evaluator.py`)

```
score    = 0.4 * accuracy + 0.6 * mean_tIoU
accuracy = correct / 390          (every question, all three types, equal weight)
mean_tIoU = mean over the 195 gold-yes questions of IoU(gold_span, predicted_span)
            (a gold-yes answered "no" or with a null span contributes 0;
             spans returned with a "no" are ignored)
```

Consequences that shape the design:

- A **missed positive costs twice** (accuracy and tIoU). A false positive costs once.
  When the answering model is unsure between yes and no on a question that looks
  like it *could* be supported, lean **yes with a span** rather than no. The
  break-even is at roughly P(yes) > 0.4 when the span quality is decent (see 3.2).
- **Span width is punished symmetrically.** A 3 s gold clause returned inside an 8 s
  segment scores 0.375; returned as a 1.5 s fragment it scores 0.5. Target: return the
  gold clause itself, or slightly wider (an extra ~0.3 s on each side of a 3 s span
  costs ~17 %, the same as being 0.5 s off at both ends).
- Baseline (all-yes, no spans) = 0.200. Perfect accuracy with no spans = 0.400.
  Perfect spans with 50 % accuracy = 0.800. **tIoU is the lever.**

### 1.3 Data facts (measured on `data/question_train.csv` and `data/audio/`)

| Fact | Value |
| --- | --- |
| Conversations / questions | 39 / 390 (ten per conversation, all types mixed in order) |
| Question types | positive 195, hard_negative 142, off_topic 53 |
| Audio | MP3, 128 kbps, 44.1 kHz, mono; 74–232 s, mean 122 s, 79 min total |
| Gold span length | mean 3.21 s, **median 2.88 s**, deciles 1.3 / 1.7 / 2.0 / 2.4 / 2.9 / 3.3 / 3.8 / 4.4 / 5.6 s, max 14.2 s |
| Gold timestamps | **195 / 195 on an exact 0.02 s grid** (whisper word-timestamp resolution) |
| Spans shared by several questions | 7 spans serve 2–3 questions each (e.g. sample_19: three prescriptions renewed) |
| Overlapping but different spans | 21 pairs, i.e. neighbouring questions often point at overlapping clauses |
| Degenerate spans | sample_63 `0.00–0.26`, sample_64 `0.00–0.16` (annotation artefacts; unrecoverable, ~1 % of tIoU) |
| Questions with a number | 18 positives, 10 hard negatives (doses, lab values, BP, ages, counts) |
| Positives phrased as an absence | 27 ("free of fever", "no signs of complications", "rule out", "unchanged") |
| Tag questions | 21 ("..., right?", "..., didn't it?") |
| Brand / drug names | Danish brand names appear: Ibumetin, Panodil, Pamol, Pantoprazole, Esomeprazole, Airomir, Activelle, Brentan, Fluconazole; plus HbA1c, TSH, LDL, creatinine, mmol/mol, mmol/L |

Hard negatives are built by perturbing a true statement in one slot: number
(100 mg → 200 mg, 135/88 → 155/98, BMI 28 vs "severely obese"), entity (Pantoprazole
vs Esomeprazole; pneumococcal vs flu vaccine; physiotherapist vs further procedures),
location (thigh vs back; lower back "alone"), polarity (stable vs unstable; renewed vs
stopped; with vs without imaging), or plan (referral / insulin / surgery that was never
agreed). Off-topic questions are about hobbies, pets, jobs, weather, and a few jokes
(the Force, Doctor Strange).

---

## 2. What the measurements say (done on this machine, 2026-09-17)

Setup: Apple M4 Pro, 24 GB, `mlx-whisper 0.4.3`, model `mlx-community/whisper-large-v3-turbo`,
`language="en"`, `word_timestamps=True`, `temperature=0`, `condition_on_previous_text=False`.
All 39 conversations transcribed; scripts live in the scratchpad and are reproduced in
`tools/` by the work plan.

### 2.1 Speed

| | Value |
| --- | --- |
| whisper-large-v3-turbo, MLX, per conversation | 3–9 s (≈ 25× realtime; 232 s file → 9.1 s) |
| Whole training set (79 min audio) | ≈ 3 min |

ASR is cheap on this hardware. The 60 s budget is almost entirely available to the
answering model.

### 2.2 Transcript quality

- Speech is clean, studio-quality, one speaker at a time, two voices; transcripts read
  as near-verbatim. Numbers come out as digits and units are spoken in full:
  "100 mg daily for 2 weeks", "47 millimoles per mole", "7.0 millimoles per liter",
  "135 over 88", "one million international units, four times daily for seven days".
- **Abbreviations in questions are spoken in full**: HbA1c → "hemoglobin A1c" /
  "long-term sugar value"; ECG → "electrocardiogram"; BMI → "body mass index";
  TSH → "thyroid-stimulating hormone"; mmol/mol → "millimoles per mole". The LLM
  handles this; any lexical matcher needs a synonym table.
- **Danish brand names are spelled phonetically**: Panodil → "panadil",
  Activelle → "Activel", Airomir → "Aromir", Esomeprazole → "Isomeprazole";
  Ibumetin, Pamol, Pantoprazole, Fluconazole are right. Fix with an
  `initial_prompt` vocabulary and, in matching code, fuzzy comparison (rapidfuzz ≥ 85).
- Drug names of hard negatives that were never said (Brentan, erythromycin, metformin,
  morphine, pneumococcal) are absent from the transcripts, as they should be.

### 2.3 Localisation ceilings (what the best possible span selection could score)

| Span source | mean tIoU vs gold | median | gold spans that cannot reach 0.5 |
| --- | --- | --- | --- |
| Best single whisper segment | 0.617 | 0.545 | 78 / 195 |
| Best run of 1–3 whisper segments | 0.671 | 0.753 | |
| Best single clause unit (rule in §6) | 0.674 | | |
| Best 1–2 adjacent clause units | 0.744–0.757 | | |
| **Best contiguous word span** | **0.926** | **0.940** | **0** |

Whisper's own segments are already sentence-sized here (median 2.5 s), and still cap
the score at ~0.62 because gold spans often start or end mid-sentence
("**what happens with my treatment?** No changes. You carry on exactly as you are.",
"penicillin. One million international units, four times daily for seven days.").
The only granularity that matches the annotation is the word.

### 2.4 Boundary alignment of whisper word timestamps to the gold

| | mean | median | p10 | p90 |
| --- | --- | --- | --- | --- |
| gold_start − first gold word's start | −0.05 s | −0.10 s | −0.24 | +0.20 |
| gold_end − last gold word's end | 0.00 s | −0.02 s | −0.08 | +0.06 |

Best fixed padding on training data: start −0.10 s, end 0 s → 0.930 vs 0.925 unpadded.
Whisper word timings are already inside the annotation's tolerance; calibration is a
half-point, not a lever. (The 20 ms grid and this near-zero offset both say the gold
was produced from whisper-family word timestamps.)

### 2.5 What the gold spans look like in text

From reading the 195 gold spans against the transcript:

- A span is the **shortest clause or clause pair that states the fact**, typically one
  sentence, sometimes a question plus its one-word answer
  ("So, there is cardiovascular disease in your family? Yes."), sometimes two
  consecutive sentences when the fact needs both ("Your chest and heart both sound
  normal. Nothing abnormal to report.").
- Spans for several questions about the same sentence coincide or nest (the three
  sample_19 renewals; "Fluconazole 50 mg" nested inside the "fungal infection in the
  mouth" span).
- A few spans are long procedural stretches (14.2 s "listened with a stethoscope"
  covering the whole examination exchange; 9.8 s "healthy and living a healthy life").
  These are rare; optimise for the 3 s case.
- Two spans are degenerate (`0.00–0.26`, `0.00–0.16`): annotation errors. Ignore.

### 2.6 Implication for the design

1. Word-level timestamps from the ASR are mandatory; the answering stage must return
   an **exact quote**, and the quote is aligned back to words.
2. The LLM should see clause-sized units with ids for orientation, but the final span
   comes from the quote, not the unit (unit-only spans cap at ~0.75).
3. There is time to spare: ASR takes < 10 s and the 8 B answering call ~12 s, so a
   second evidence pass, or a 14 B model, fits in the budget on this hardware.

---

## 3. Where the points are

### 3.1 Targets, given the measurements

| Component | Measured now (Qwen3-8B, whisper turbo) | Day-3 target | Ceiling |
| --- | --- | --- | --- |
| Accuracy: positive | 0.99 | 0.98 | 1.00 |
| Accuracy: hard_negative | 0.99 | 0.95 | 1.00 |
| Accuracy: off_topic | 1.00 | 1.00 | 1.00 |
| Overall accuracy | 0.99 | 0.97 | 1.00 |
| Passages cited in the right place | 165 / 195 | 180 / 195 | 195 |
| tIoU when the passage is right | 0.675 | 0.78 | 0.93 |
| mean tIoU (all 195 gold-yes) | 0.57 | 0.72 | 0.93 |
| **Score** | **0.72** | **0.82** | **0.96** |

The validation and evaluation questions are presumably harder or at least different
from the training ones; expect accuracy to drop a few points on the platform and treat
the platform's validation score as the real number.

### 3.2 When to say yes

Let p be the model's belief that the answer is yes and q the expected tIoU if we return
our best span. Expected gain of answering yes vs no on one question, in score units per
question: `yes: 0.4p + 0.6pq`, `no: 0.4(1-p)`. Say yes when `p > 0.4 / (0.8 + 0.6q)`;
with q ≈ 0.6 that is **p > 0.34**. In practice: threshold the LLM's yes-probability at
about 0.35–0.45 rather than 0.5, and tune it on the training set. Only do this once
spans are decent; with q = 0 the threshold is 0.5.

---

## 4. Architecture

```
request ──► decode MP3 ──► ASR (word timestamps) ──► words[] ──► units[] (clauses, timed)
                                                                      │
questions[10] ────────────────────────────────────────────────────────┤
                                                                      ▼
                                                   answering LLM (one JSON call, all 10)
                                                     answer, unit ids, exact quote
                                                                      │
                                                                      ▼
                                             quote → word alignment → span → calibration
                                                                      │
                                                                      ▼
                                             validate, fill defaults, ASRQuestionResponseDto
```

Every stage has a fallback so the request always returns:
ASR failure → all `true` with `null` spans (0.2 floor) or a small-model retry;
LLM failure/timeout → retrieval-only answers (BM25 + number match) with unit spans;
alignment failure → the cited unit's span; nothing at all → `true`, `null`.

### 4.1 Module layout

```
medical-appointment/
├── api.py                 # unchanged transport
├── example.py             # thin: builds Pipeline at import, predict() delegates, catches everything
├── pipeline/
│   ├── __init__.py
│   ├── config.py          # one dataclass: model names, thresholds, offsets, timeouts, device
│   ├── audio.py           # base64 → 16 kHz float32 mono (ffmpeg subprocess or PyAV); duration
│   ├── asr/
│   │   ├── base.py        # Word(text,start,end,p), Segment; ASRBackend.transcribe(pcm) -> list[Word]
│   │   ├── mlx_whisper_backend.py
│   │   ├── faster_whisper_backend.py
│   │   └── parakeet_backend.py
│   ├── units.py           # words → Unit(id, start, end, text, words[]) clauses; speaker turn guess
│   ├── retrieval.py       # BM25 + number/entity matching; per-question top-k units (fallback + LLM focus)
│   ├── llm/
│   │   ├── base.py        # LLMBackend.complete_json(prompt, schema) -> dict
│   │   ├── mlx_lm_backend.py
│   │   └── openai_compat_backend.py   # llama.cpp server / vLLM / Ollama on localhost
│   ├── answer.py          # prompt building, JSON parsing, yes-threshold, per-question fallbacks
│   ├── evidence.py        # quote → word alignment, span calibration
│   └── watchdog.py        # deadline handling: soft deadline at 45 s, hard at 55 s
├── tools/
│   ├── transcribe_all.py  # cache transcripts: transcripts/<backend>/<sample>.json
│   ├── eval_offline.py    # answering+evidence on cached transcripts; per-type accuracy, tIoU, ablations
│   ├── span_upper_bound.py# best-achievable tIoU per ASR backend (segment / merged / word)
│   └── calibrate_spans.py # grid-search boundary offsets on training data
├── transcripts/           # gitignored cache
└── models/                # gitignored weights (optional, else HF cache)
```

### 4.2 Core data types

```python
@dataclass
class Word:
    text: str          # as transcribed, with leading space stripped, punctuation kept
    start: float
    end: float
    p: float | None    # ASR confidence if available

@dataclass
class Unit:            # one clause; the candidate evidence granularity
    id: int
    start: float
    end: float
    text: str
    words: list[Word]
    speaker: str | None   # "D"/"P" guess, optional

@dataclass
class QuestionResult:
    answer: bool
    p_yes: float
    unit_ids: list[int]
    quote: str | None
    span: tuple[float, float] | None
```

---

---

## 5. Stage A: ASR

### 5.1 Measured candidates

| Model / runtime | Time per conversation (M4 Pro) | Transcript quality | Word-span ceiling (mean tIoU) | Boundary bias vs gold |
| --- | --- | --- | --- | --- |
| `mlx-community/whisper-large-v3-turbo` via `mlx-whisper` | 3–9 s | Near-verbatim; digits for numbers; units in words ("50 milligrams"); brand names phonetic | **0.926** | start −0.05 s, end 0.00 s |
| `omi-health/omi-med-stt-v1-mlx-q8` (Parakeet TDT 0.6B v2, medical fine-tune) via `omi-med-stt[mlx]` | **0.9–2 s** | Same words, same phonetic brand names; normalises units the way the questions do ("50 mg", "47 mmol/mol", "7.0 mmol/l"); emits `<unk>` for apostrophes/dashes | 0.862 (≈ 0.90 after +0.2 s end padding) | start 0.00 s, **end −0.20 s** (TDT frame granularity) |

Both transcribe the numbers and the drug names identically on every case checked
(100 mg / two weeks, 1 million IU × 4 daily × 7 days, fluconazole 50 mg, 47 mmol/mol,
7.0 mmol/L, BMI 29, 138/83, LDL 3.9, 135/86). Neither was helped or hurt by a vocabulary
prompt yet (not tested; see 5.3).

Not measured, worth one experiment each on day 2 if there is time:

| Candidate | Why it might win | Cost |
| --- | --- | --- |
| `mlx-community/whisper-large-v3-mlx` (full large-v3) | Slightly better rare-word accuracy than turbo | ~2–3× slower (still < 25 s); same timestamp machinery |
| `nyralabs/CrisperWhisper2.0_turbo` (CTranslate2, Linux/CUDA or CPU) | Claims 30–40 ms word-boundary error, verbatim mode, no chunk seams | Non-commercial licence (fine for a student competition), no MLX build → Linux host only |
| `Qwen/Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B` | Strong ASR; the aligner timestamps *arbitrary text units*, i.e. it could time the LLM's quote directly | Two models, transformers/vLLM stack, CUDA preferred |
| `mlx-community/parakeet-tdt-0.6b-v2` (stock) | Same speed as the medical one without `<unk>` artefacts | Same TDT end bias |
| `faster-whisper large-v3-turbo` (CTranslate2) | The Linux/CUDA equivalent of the MLX choice | Needed anyway for the VM fallback |

### 5.2 Decision

**Primary: whisper-large-v3-turbo with word timestamps** (MLX on the Mac, faster-whisper
on Linux). It has the best agreement with the annotation's timing, the ceiling is 0.93,
and it costs < 10 s. **Secondary: the medical Parakeet**, run in parallel or after
(≈ 1 s), used for (a) a normalised second transcript for number/unit checking,
(b) disagreement detection on the cited unit (if the two transcripts disagree on a
number or drug inside the cited unit, lower the confidence and let the second pass or
the number rule decide), and (c) an ASR fallback if whisper throws.

Keep the ASR abstraction (§4.1) so swapping to full large-v3 or CrisperWhisper on the
Linux host is a config change.

### 5.3 Configuration details

- Decode MP3 to 16 kHz mono float32 once (`ffmpeg -i - -f f32le -ac 1 -ar 16000 -`
  via subprocess, or PyAV); feed the array to both models. Do not write temp files
  per request unless the backend insists.
- `condition_on_previous_text=False` (avoids drift/repetition loops across 30 s windows),
  `temperature=0` with the default fallback ladder disabled (deterministic timing),
  `hallucination_silence_threshold=2.0`, `no_speech_threshold=0.6`.
- Test `initial_prompt` with the brand-name vocabulary (Appendix B) on the training set:
  accept it only if the entity spellings improve **and** the word-span ceiling does
  not drop.
- Post-process words: strip leading spaces; keep punctuation attached (the LLM sees
  it, the aligner ignores it); drop zero-length words; enforce monotonic times.
- Parakeet: replace `<unk>` with `'` where it sits inside a word (I<unk>m → I'm) and
  with `—` between words; add +0.2 s to token ends (calibrated) before use.

---

## 6. Stage B: from words to units (clauses)

The gold spans are clauses, not whisper segments. Build the candidate granularity
from word timestamps, independent of how the ASR chunked the audio.

**Algorithm (`units.py`):**

1. Take the flat word list. Normalise: strip leading spaces, keep trailing punctuation
   on the word, keep the original casing for the LLM prompt, lowercase copy for matching.
2. Start a new unit at word *i* when any of these holds:
   - the previous word ends with `.`, `?` or `!`;
   - the previous word ends with `,`, `;` or `:` **and** the current unit already has ≥ 6 words;
   - the pause `words[i].start − words[i-1].end` ≥ **0.5 s** (speaker turn or breath);
   - the current unit is ≥ **18 words** long (hard cap; split at the nearest comma/pause).
3. Units shorter than 3 words are merged into the following unit unless a ≥ 1 s pause
   separates them.
4. Each unit gets `start = words[0].start`, `end = words[-1].end`. Do **not** pad here;
   padding is a calibrated step in stage D.
5. Speaker guess (optional, cheap): a new unit after a pause ≥ 0.7 s that starts with a
   question word or "Yes/No/Okay/Right" is a turn change; alternate D/P labels from
   the first turn (the doctor almost always opens). Only used as prompt context.
6. Sanity: the tuned thresholds (0.5 s, 6 words, 18 words) are settled by
   `tools/span_upper_bound.py`: maximise the mean best-unit tIoU against the gold
   spans, and also report the best "1–2 adjacent units" tIoU, because the LLM is
   allowed to cite two neighbouring units.

Measured outcome (§2.3): best single unit 0.674, best two adjacent units ~0.75, and the
thresholds barely matter (0.665–0.676 across the grid), so do not tune them further;
the word-level quote alignment is what reaches the 0.93 ceiling.

**Number normalisation.** Keep the transcript verbatim for the LLM, but build a
parallel normalised string per unit for retrieval and matching:
`one hundred milligrams → 100 mg`, `forty-seven → 47`, `one million IU`,
`hundred and thirty-five over eighty-eight → 135/88`, `mmol per mol → mmol/mol`.
Use a small rule table plus `text2num`/`word2number`. Hard negatives are mostly
number/entity swaps, so this normalised text is what the retrieval fallback and the
yes/no cross-check (7.4) compare against.

---

## 7. Stage C: answering the questions

### 7.0 Measured: Qwen3-8B, zero-shot, one JSON call per conversation

Run on this machine over all 39 training conversations, whisper-turbo transcripts,
clause units as in §6, the prompt of §7.1 (thinking off, temperature 0, `max_tokens=900`),
`mlx-lm 0.31.3`, `mlx-community/Qwen3-8B-4bit`:

| Metric | Value |
| --- | --- |
| Accuracy | **0.990** (386 / 390) |
| positive | 193 / 195 |
| hard_negative | 140 / 142 |
| off_topic | 53 / 53 |
| JSON parse failures | 0 / 39 |
| Latency per conversation (10 questions) | mean 11.6 s, max 18.6 s |
| Prompt size | ≈ 1050 tokens |
| mean tIoU with a crude first/last-token quote alignment | 0.536 |
| Score with that alignment | 0.717 |

So the answering half is essentially solved by an 8 B model on these transcripts, at a
fifth of the time budget, without retrieval, second passes, or fine-tuning. The whole
remaining gap to the ~0.93 ceiling is **evidence localisation** (section 8), and the
larger models in 7.2 are only worth trying if validation-set accuracy turns out lower
than this (the validation/evaluation questions may be harder than the training ones).
Do not spend day 2 on bigger LLMs; spend it on spans.

### 7.1 Chosen approach: one structured LLM call per conversation

The transcript is short (2 min ≈ 300–450 words ≈ 600–900 tokens with unit ids). Ten
questions share one prompt, so prompt processing is paid once and the model sees the
questions together, which helps it notice that "100 mg" and "200 mg daily" cannot
both be true. Output is strict JSON, one object per question, in order.

**Prompt skeleton (system + user):**

```
You are checking claims against the transcript of a doctor–patient consultation.
The transcript is split into numbered units. Answer each yes/no question ONLY from
what the transcript says. Rules:
- Answer "yes" only if the transcript explicitly states or clearly implies it.
  A near-miss is "no": same drug but a different dose, same test but a different
  value, same symptom but a different body part, a plan that was mentioned as a
  possibility but not agreed, the opposite polarity (stable vs unstable).
- Questions about things never discussed are "no".
- Questions phrased as an absence ("free of fever", "no signs of X", "unchanged")
  are "yes" when the transcript states the absence or the unchanged status.
- Tag questions ("..., right?") are ordinary yes/no questions.
- For every "yes", give the unit id(s) (at most 2, adjacent) and copy the EXACT
  words from those units that establish the answer; the quote must be a
  contiguous substring of the unit text, as short as possible while still
  containing the fact (typically 4–15 words).

TRANSCRIPT
[0] (D) Hello, welcome. What brings you in today?
[1] (P) I've had a sore throat for about three days now.
...

QUESTIONS
1. Should the daily dose be 100 mg?
2. ...

Respond with JSON only:
{"results":[{"q":1,"answer":"yes"|"no","confidence":0-100,"units":[..],"quote":"..."}, ...]}
```

Implementation notes:

- **Constrained decoding** when the backend supports it (llama.cpp GBNF / vLLM guided
  JSON / `outlines`). With mlx-lm, parse leniently (`json_repair`) and re-ask once on
  failure with a shorter "fix the JSON" prompt; on second failure fall back per
  question to retrieval (7.3).
- **Thinking mode off** for Qwen3-family (`enable_thinking=False`, or `/no_think`);
  it is too slow for the budget. If a thinking model turns out clearly better on hard
  negatives, cap reasoning tokens and only use it for the per-question retry (7.4).
- Temperature 0, deterministic. Keep `max_tokens` ≈ 60 per question (≈ 700 total).
- Confidence is used with the threshold from 3.2, tuned on training data.
- Log every prompt/response to `logs/` during development; it is the main debugging tool.

### 7.2 Candidate answering models (local, 24 GB Apple Silicon or a 16–24 GB GPU)

All exist on Hugging Face as of 2026-09-17 (checked via the HF API), MLX 4-bit unless noted.

| Model | Size on disk | Why | Risk |
| --- | --- | --- | --- |
| **`mlx-community/Qwen3-8B-4bit`** (measured, §7.0) | ~5 GB | 386/390 on training data, 0 JSON failures, ~12 s per conversation | Missed "47 mmol/mol" vs "millimoles per mole" and "molluscs" vs "molluscum": add these to the synonym list in the prompt |
| `mlx-community/Qwen3.5-9B-4bit` (or `-MLX-4bit`) | ~5.5 GB | Newer generation, same footprint | Less tested in mlx-lm; check JSON reliability |
| `mlx-community/Qwen3-14B-4bit` | ~8.5 GB | Noticeably better reading comprehension | ~2× slower than 8B |
| `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | ~17 GB | MoE: 3 B active → 8B-class speed with 30B-class quality | Memory: 17 GB + ASR ≈ 19 GB on a 24 GB box; test headroom |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | ~19 GB | Newer MoE, same idea | Memory very tight with ASR loaded |
| `mlx-community/gemma-3-12b-it-4bit` / `-qat-4bit` | ~7.5 GB | Good at grounded QA, different failure modes from Qwen (useful for ensembling) | Slower prompt processing |
| `mlx-community/Qwen3.8-27B-4bit` | ~16 GB | Likely the strongest dense model that fits | Dense 27B is slow: maybe 12–15 tok/s → 700 tokens ≈ 50 s. Too slow unless output is trimmed to ids only |
| `google/medgemma-4b-it` | ~3 GB (4-bit) | Medical vocabulary | Small; general models read better |

**Decision procedure:** run `tools/eval_offline.py` with the same cached transcripts for
each model; report per-type accuracy, JSON failure rate, and p50/p95 latency for the
whole ten-question call. Pick the best score that keeps p95 latency of
`ASR + LLM` under **40 s** on the deployment host. Measured: Qwen3-8B already
answers 386/390, so it is the default; try 14B or 30B-A3B only if the platform's
validation accuracy is clearly below 0.95. On a Linux GPU box
with vLLM, the same models in AWQ/GPTQ, or `Qwen3.8-27B` if a 24 GB+ GPU is available.

### 7.3 Retrieval fallback and cross-check (no LLM)

Cheap, always available, and also a useful signal:

1. BM25 over normalised unit text, query = normalised question; plus a bonus for
   units containing the same number tokens / drug names as the question.
2. Score `s = max unit score`. Off-topic questions get near-zero scores; positives and
   hard negatives both score high (same topic), so BM25 alone cannot separate them.
3. **Number/entity check:** if the question contains a number (or a drug name) and the
   top units contain a *different* number of the same kind (mg, mmol, BP, weeks) and
   not the question's number, force **no**. This alone kills a large share of hard
   negatives and is a safety net when the LLM is unavailable.
4. Fallback answer: yes if `s > τ` and the number check passes, span = top unit.
   Measured on the training set (BM25 over clause units, threshold 2–4): accuracy
   0.67–0.72, all 53 off-topic right, mean tIoU 0.30–0.40, score **0.47–0.51**.
   That is the floor the watchdog falls back to, far above 0.20.

### 7.4 Optional second pass on low-confidence questions

If time remains inside the request (watchdog says > 15 s left), re-ask the LLM one
question at a time for the questions with confidence in [30, 70], giving only the
top-3 retrieved units plus their neighbours and asking for a short justification
before the answer. This targets exactly the hard negatives. Measure whether it helps
before shipping it; it costs ~2 s per question.

### 7.5 Alternatives considered

- **NLI cross-encoder** (DeBERTa-v3 MNLI/ANLI, `bge-reranker-v2-m3`): fast (ms per
  pair), but pairs each question with one unit; it misses answers spread over two
  units and is weak on numeric near-misses. Keep as an ensemble feature, not as the
  main answerer.
- **Extractive QA** (RoBERTa/DeBERTa SQuAD2): gives a span, but the questions are
  yes/no, so it would need a rewrite step. Not worth it given the LLM handles both.
- **Audio-LLM end to end** (Qwen-Audio class): no reliable timestamps, and the
  20 ms grid says the gold was built from a text transcript. No.
- **Fine-tuned small LLM** (LoRA on Qwen3-4B/8B with synthetic questions): the
  strongest stretch idea, see section 13.

---

## 8. Stage D: evidence localisation

This is where the score is decided, so this section is written from the measured
failure modes (section 7.0 dump; scripts in `tools/probes/`) rather than from first
principles.

### 8.1 What the measurement says

With the §7.1 prompt, Qwen3-8B answers yes on 193 of the 195 gold-yes questions and
returns a quote for each. Aligning that quote back to whisper words:

| Span strategy | mean tIoU (195) | zero-overlap cases |
| --- | --- | --- |
| Span of the cited unit id(s) | 0.515 | 31 |
| Fuzzy match of the quote inside the cited units ± 1 | 0.560 | 33 |
| **Fuzzy match of the quote over the whole transcript** | **0.571** | 30 |
| + start padding +0.1 s | 0.579 | 30 |
| + trim multi-sentence quotes to the best sentence | 0.578 | 29 |
| + prepend the previous unit to a < 4-word reply | 0.558 | 30 |

Over the 165 cases where the passage is right, mean tIoU is 0.675; the quote is
> 1.15× the gold length in 56 cases and < 0.85× in 44. So post-hoc rules on the quote
are worth ≤ 0.01; the two real problems are **which passage** and **how much of it**,
both of which are decided by the LLM.

The 30 zero cases are mostly *another true mention of the same fact*: the model quotes
"I think so." where the annotation is the doctor's "this is a viral infection"; the
patient's early "I have almost run out of my painkillers" where the annotation is the
later "Pamol and Ibumetin" at prescription time; "None." where the annotation is
"there are no signs of complications". The annotator (evidently an LLM writing
questions *from* a specific passage) prefers the **explicit, self-contained statement
that contains the question's own terms**, and when the fact is a short reply it
includes the question that reply answers ("So, there is cardiovascular disease in your
family? Yes.").

### 8.2 The design

1. **Better evidence instruction in the main call (day 2, first thing).** Replace the
   quote rule of §7.1 with:
   - quote the *single* shortest sentence or clause that states the fact explicitly;
     prefer a statement that contains the question's key words (drug, value, finding)
     over a bare "yes / none / I think so";
   - if the only explicit evidence is a short reply, quote the question and the reply
     together;
   - never quote more than one sentence unless the fact needs both halves;
   - give up to **three candidate quotes** per yes, best first (`"quotes": [...]`).
   Measure with the offline harness. Expected: zero cases 30 → ~20, right-passage
   tIoU 0.675 → ~0.72.
2. **Candidate re-ranking (day 2).** For each yes, candidates = the LLM's quotes +
   the top-3 BM25 units for the question. Score each candidate with a few features:
   content-word overlap with the question (with the synonym table and rapidfuzz for
   brand names), whether it contains the question's number/entity, length in words
   (prior peaked at 6–12), speaker guess (doctor for findings/plans, patient for
   symptoms/history), LLM rank. Start with hand-set weights, then fit a logistic
   regression on the 195 training examples with leave-one-conversation-out; keep it
   only if LOO tIoU improves. This directly targets the 30 zero cases.
3. **Quote → words.** Fuzzy alignment of the chosen quote against the whole word
   sequence (`rapidfuzz.fuzz.ratio` over windows of |quote| ± 3 tokens, accept ≥ 70);
   fall back to the cited unit span, then to the top BM25 unit. Pad start by +0.1 s
   (measured optimum). Never return `null` with a yes.
4. **Per-question evidence pass for the unsure ones (day 3).** When the re-ranker's top
   two candidates are close, ask the LLM once more for that question only: show the
   candidates with ids and ask which one *most directly states* the fact. ~1.5 s each,
   only for the ambiguous ones, inside the watchdog budget.
5. **Length calibration.** With candidates chosen, grid-search a trim/extend rule on
   training data: e.g. drop a trailing clause after a comma when it contains none of
   the question's terms; extend a one-clause quote to the full sentence when the gold
   statistics say so for that length. Keep only rules that improve LOO tIoU by ≥ 0.01.

Metrics to watch in `eval_offline.py`: zero-overlap count, mean tIoU over right-passage
cases, mean tIoU over all gold-yes, and the diagnostic "tIoU when answered yes".

---

## 9. Robustness and timing

- **Import-time warm-up:** load ASR and LLM, run one 10 s transcription and one tiny
  JSON completion before uvicorn starts serving. `api.py` imports `example.py`, so the
  warm-up sits in `example.py`'s module body (or `Pipeline.__init__`).
- **Watchdog:** record `t0` at the start of `predict`. Soft deadline **45 s**: skip the
  optional second pass. Hard deadline **52 s**: abandon the LLM (run it in a worker
  thread; on timeout ignore its result) and answer from retrieval. Never exceed 55 s.
- **Per-question try/except** and a final `validate_response` call inside `predict`
  itself, with a last-resort all-`true`/`null` reply if validation fails.
- **Memory:** keep both models resident; measure peak RSS. On the 24 GB Mac,
  ASR (~1.6 GB fp16 turbo) + Qwen3-14B-4bit (~8.5 GB) is comfortable;
  30B-A3B (~17 GB) needs checking under load with a 4 min file.
- **Concurrency:** the evaluator is strictly sequential, so run uvicorn with one
  worker and a lock around the pipeline to protect the GPU.
- **Long files:** the longest training file is 232 s; validation/evaluation may go to
  ~3.5–4 min. Test a 5 min synthetic file for both time and memory.
- **Logging:** one line per request with filename, duration, ASR seconds, LLM seconds,
  answers, and spans. Keep the last N transcripts on disk for post-mortems.

---

## 10. Evaluation harness (build this before tuning anything)

`tools/transcribe_all.py --backend mlx_whisper --model mlx-community/whisper-large-v3-turbo`
writes `transcripts/<backend>__<model>/<sample>.json` with words and (optionally)
units. Run once per ASR candidate.

`tools/eval_offline.py --transcripts <dir> --llm <name> [--questions data/question_train.csv]`
runs stages B–D on the cache, imports `temporal_iou`/`gold_evidence` from `utils.py`,
and prints exactly the blocks `local_evaluator.py` prints (accuracy by type, mean tIoU,
no span returned, tIoU when answered yes) plus:

- confusion by type and by "question contains a number";
- per-conversation worst cases (lowest tIoU, most wrong) to read by hand;
- JSON parse failures and LLM latency percentiles;
- an `--oracle-units` mode that answers with gold labels but our spans (isolates
  localisation quality) and an `--oracle-answers` mode that uses our answers with the
  best possible span (isolates answering quality).

`tools/span_upper_bound.py --transcripts <dir>` prints the ceilings from section 2 for
any ASR cache, so ASR candidates can be compared on the metric that matters without
running the LLM.

Only after the offline score is good: `python api.py` + `python local_evaluator.py
--verbose` for end-to-end timing, then a validation attempt on the platform.

**Overfitting guard:** 39 conversations is small. Tune thresholds/offsets with
leave-one-conversation-out or a fixed 30/9 split, and prefer few, coarse knobs.
Watch the validation-set score on the platform; it is the only held-out signal.

---

## 11. Deployment

Two viable hosts; set up the primary on day 1 and the fallback by day 2.

### 11.1 Primary: this Mac (M4 Pro, 24 GB) behind a tunnel

- Stack: `mlx-whisper` (or `parakeet-mlx`) + `mlx-lm`, Python 3.11–3.13 (the repo's
  `.venv` is Python 3.14: recreate it with 3.11/3.13 for wheel availability).
- Expose: `brew install cloudflared` then `cloudflared tunnel --url http://localhost:9054`
  (quick tunnel; URL changes on restart) or, better, a named tunnel on a domain you
  control so the submitted URL never changes. ngrok with a static domain is the
  alternative. Submit `https://<host>/predict`.
- Cloudflare's free-tier request-body limit is 100 MB; bodies here are ≤ 5 MB. Check the
  tunnel is not adding latency (>1 s) by running `local_evaluator.py --url https://...`.
- Keep the machine awake (`caffeinate -dimsu`), on power, on a wired or reliable
  network; disable automatic updates for the weekend; run the server under a
  restart loop (`while true; do python api.py; done`) or `launchd`.
- This is a tunnel, not a model API: it does not violate the "no cloud API" rule.

### 11.2 Fallback: a Linux GPU VM

- Azure for Students rarely grants GPU quota; UCloud (SDU/DeiC) gives Danish
  students GPU hours; otherwise a rented GPU box (RunPod/Lambda/Vast, a 24 GB card is
  plenty). Open port 9054 in the security group.
- Stack: `faster-whisper` (`large-v3-turbo`, `float16`, `word_timestamps=True`) +
  `vllm` or `llama-cpp-python` server on localhost for the LLM, same pipeline code
  via the backend abstraction. Provide a CUDA `Dockerfile` (`nvidia/cuda:12.x` base,
  pre-download weights into the image so start-up is offline and fast).
- CPU-only VMs are **not** viable for large-v3-class ASR inside 60 s on 3.5 min files;
  if forced, use `distil-large-v3`/`turbo` int8 with 8+ vCPUs and measure.

### 11.3 Go-live checklist

1. `python local_evaluator.py --oracle` prints 1.000.
2. Local run: 39/39 conversations answered, worst latency < 45 s, no timeouts.
3. Remote run through the public URL: same numbers within +2 s.
4. "Verify" on cases.nordicaicup.com passes; queue a validation attempt; read the
   per-type breakdown it reports (if any) against local numbers.
5. Fallback host reachable and validated too.
6. Before the single evaluation attempt: freeze code, restart the server, re-run
   `local_evaluator.py` once, confirm free memory/disk, then submit with ≥ 3 hours of
   margin before 16:00 CEST on Sunday.

---

## 12. Work plan and milestones

**Day 1 (Wed 17 Sep):** Python 3.11 venv; `pipeline/` skeleton with the ASR, units,
answer and evidence modules reproducing the probe exactly (whisper turbo + Qwen3-8B +
fuzzy quote alignment); `tools/transcribe_all.py` and `tools/eval_offline.py`;
`example.py` wired with warm-up and watchdog; `python local_evaluator.py` green on
39/39 with worst latency < 25 s; tunnel up; first platform validation attempt.
Acceptance: local score ≥ 0.70, platform validation accuracy reported.

**Day 2 (Thu 18 Sep):** Evidence. New quote instruction + candidate quotes (8.2.1),
BM25 candidates + re-ranker (8.2.2), unit-normalised transcript from Parakeet fed to
the number check and to the synonym table (fixes the two "mmol/mol" / "molluscum"
misses). Offline target: zero cases ≤ 20, mean tIoU ≥ 0.66, score ≥ 0.78. Fallback
host set up. Validation attempt in the evening.

**Day 3 (Fri 19 Sep):** Per-question evidence pass for ambiguous cases (8.2.4), length
calibration (8.2.5), yes-threshold tuning (3.2), robustness (5 min file, memory, restart
loop, five-timeouts test), second validation attempts. Target score ≥ 0.82. If time
remains: the LoRA stretch (13.1) on synthetic questions, or Qwen3-14B if the platform
accuracy is below 0.95.

**Day 4 (Sat 20 Sep, deadline 16:00):** Freeze by 10:00, restart both hosts, one last
`local_evaluator.py` run, evaluation attempt by ~12:00 with margin. Keep the top-5
hand-in ready (code + list of weights).

---

## 13. Stretch ideas, in order of expected value

1. **Learned passage re-ranker (extends 8.2.2).** Train a small model (logistic
   regression or a tiny cross-encoder fine-tune) on (question, candidate passage) pairs
   from the 195 annotated positives, augmented with synthetic questions written from
   the training transcripts by a hosted LLM in the annotator's style. The annotator's
   preference for explicit self-contained statements is learnable; this attacks the
   largest measured loss.
2. **Synthetic supervision + LoRA on the answering/evidence model.** During development
   cloud LLMs are allowed. Generate hundreds of extra positive / hard-negative / off-topic
   questions with quoted evidence from the transcripts, fine-tune Qwen3-8B (or 4B) with
   `mlx_lm.lora` to emit the §7.1 JSON with annotator-style quotes. Validate only on
   held-out real conversations.
3. **Two-model vote on hard negatives** (Qwen + Gemma-3-12b): only if platform accuracy
   is below 0.95; locally there are 4 errors to fix, not 40.
4. **ASR ensemble for numbers.** Parakeet's unit-normalised text ("47 mmol/mol") beside
   whisper's; flag questions where the two transcripts disagree on a number or drug in
   the cited passage.
5. **Diarisation** (pyannote, local) for the speaker feature in the re-ranker; the pause
   heuristic in §6 is probably enough.

---

## 14. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Request over 60 s on a long file | −10 marks each, abort after 5 | Watchdog with retrieval fallback; test 5 min file; p95 < 45 s |
| Host dies mid-attempt (sleep, Wi-Fi, OOM) | Loses the tail of the attempt | caffeinate, restart loop, fallback host validated |
| ASR mishears brand names / numbers | Hard negatives flip | `initial_prompt` vocabulary, ASR comparison on entity recall, number normalisation, ASR ensemble |
| LLM misses unit/name variants ("mmol/mol" vs "millimoles per mole", "molluscs" vs "molluscum") | 2–4 wrong answers per 390 | Synonym/unit table in the prompt, Parakeet's normalised text, fuzzy entity match |
| Platform questions harder than training (accuracy < 0.95 there) | −0.02 to −0.05 score | Second pass on low-confidence questions, 14B model, threshold tuning |
| Spans too wide (segment-level) | tIoU stuck ~0.35 | Word-level units, quote alignment, calibration |
| JSON parse failure | Whole conversation falls back | Constrained decoding, repair, per-question retry, retrieval fallback |
| Overfitting 39 conversations | Validation ≠ evaluation | Few knobs, LOO tuning, watch platform validation |
| Python 3.14 venv lacks wheels | Blocked install | Recreate venv with 3.11 or 3.13 |

---

## Appendix A. Environment and commands

```bash
# Fresh venv (the checked-in .venv is Python 3.14; ASR/LLM wheels target 3.11–3.13)
/usr/local/bin/python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -r requirements.txt
# Mac (Apple Silicon)
pip install mlx-whisper mlx-lm parakeet-mlx rank-bm25 rapidfuzz json-repair text2num
# Linux GPU alternative
pip install faster-whisper vllm rank-bm25 rapidfuzz json-repair text2num

# Cache transcripts for one ASR backend (run once per candidate)
python tools/transcribe_all.py --backend mlx_whisper --model mlx-community/whisper-large-v3-turbo
python tools/transcribe_all.py --backend parakeet   --model omi-health/omi-med-stt-v1-mlx-q8

# Localisation ceiling per ASR cache
python tools/span_upper_bound.py --transcripts transcripts/mlx_whisper__whisper-large-v3-turbo

# Offline answering + evidence evaluation
python tools/eval_offline.py --transcripts transcripts/mlx_whisper__whisper-large-v3-turbo \
    --llm mlx-community/Qwen3-14B-4bit --verbose

# End to end
python api.py                     # terminal 1
python local_evaluator.py --verbose   # terminal 2
cloudflared tunnel --url http://localhost:9054   # terminal 3, then submit https://<id>.trycloudflare.com/predict
```

## Appendix B. ASR backend configuration

```python
# mlx-whisper
mlx_whisper.transcribe(
    pcm_16k, path_or_hf_repo=MODEL, language="en", word_timestamps=True,
    temperature=0.0, condition_on_previous_text=False,    # no cross-window drift
    initial_prompt=VOCAB_PROMPT,                          # brand names + units, see below
    hallucination_silence_threshold=2.0, no_speech_threshold=0.6,
)
# faster-whisper
WhisperModel("large-v3-turbo", device="cuda", compute_type="float16").transcribe(
    pcm_16k, language="en", word_timestamps=True, beam_size=5,
    condition_on_previous_text=False, initial_prompt=VOCAB_PROMPT, vad_filter=False)

VOCAB_PROMPT = ("Consultation transcript. Ibumetin, Panodil, Pamol, Pantoprazole, "
                "Esomeprazole, Airomir, Activelle, Brentan, Fluconazole, penicillin, "
                "erythromycin, metformin, HbA1c 47 mmol/mol, LDL 2.2 mmol/L, "
                "creatinine, TSH, blood pressure 135/88, BMI 28, 100 mg, 50 mg.")
```

The initial prompt biases spelling of the Danish brand names and number formats;
verify on the training transcripts that it does not increase hallucination
(compare entity recall and best-span ceiling with and without it).

## Appendix C. `example.py` contract

```python
# example.py
from pipeline import Pipeline, Config
PIPELINE = Pipeline(Config())     # loads + warms up at import; api.py imports this module

def predict(request):
    try:
        return PIPELINE.predict(request)         # always returns a valid DTO
    except Exception:
        logger.exception("pipeline failed; returning floor answer")
        n = len(request.questions)
        return ASRQuestionResponseDto(answers=[True]*n, evidence_start=[None]*n, evidence_end=[None]*n)
```

`Pipeline.predict` itself: decode → ASR (fallback: smaller model, then floor) → units →
retrieval → LLM within deadline (fallback: retrieval answers) → evidence → validate.

## Appendix D. Offline evaluator output format

Mirror `local_evaluator.py`'s report exactly (same labels, same three decimals) so
numbers are comparable, and add a `--dump results.csv` with one row per question:
`question_id, type, label, pred, p_yes, gold_start, gold_end, pred_start, pred_end, tiou, quote, units`.
Reading this CSV sorted by `tiou` is how the span calibration and prompt fixes are found.
