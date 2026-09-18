# Medical Appointment — plan to 0.8+ on submission (2026-09-18)

Written so it can be acted on directly, by us or by Claude Code. Competition ends
**Sun 20 Sep 16:00 CEST**. Validation = 19 conversations, as often as we like.
Evaluation = 38 *other* conversations, **one** attempt. Score =
`0.4 × accuracy + 0.6 × mean tIoU`, where tIoU is averaged over every question
whose true answer is yes.

## The finding

The gold evidence spans are not hand-drawn. They are **faster-whisper `base`
word timestamps, int8, on CPU**: `language='en'`, `word_timestamps=True`, all
other settings default (beam 5, no VAD, no rounding). Each gold span runs from
the start of the first word to the end of the last word of a verbatim quote.

Proof: `python check_timestamps.py` transcribes the 39 training conversations
that way. All **390/390** annotated timestamps land exactly on a word boundary,
including the two float artifacts `28.520000000000003` (sample_4, "Let") and
`80.46000000000001` (sample_70, "Then"), reproduced bit for bit.

| ASR used for timestamps | spans exact at both ends | both ends within 20 ms | tIoU if the right words are picked |
|---|---|---|---|
| small.en, no VAD (our served code also adds VAD, which is worse) | 2% | 14% | 0.95 |
| large-v3 (sample_4, int8 and float32) | 0% | 0% | 0.89 |
| small, base.en, medium.en (2 conversations) | 0% | 0–8% | 0.91–0.94 |
| base, float32 | 41% | 83% | 0.99 |
| **base, int8** | **100%** | **100%** | **1.000** |

With `base` int8 words, this is what each kind of span is worth once the right
passage has been found:

| Span we return | Mean tIoU against gold (195 positives) |
|---|---|
| The exact quote the annotator used | **1.000** |
| The whole sentence(s) containing it | 0.873 |
| The whole Whisper segment(s) containing it | 0.746 |
| No LLM: best word-overlap sentence | 0.51 |

The 0.746 row is the leaderboard's 0.70–0.77 cluster. Segment-level spans cap
tIoU there even with perfect answers. At ~0.93 accuracy that is
`0.37 + 0.6 × ~0.63 ≈ 0.75`, which is where we are. The 0.99–1.00 teams must
be returning word-exact spans, and possibly also tuning against the fixed
validation set, which will not carry over to the 38 unseen evaluation
conversations.

## Status on this branch (`medical-appointment-exact-evidence`)

Steps 2–4 are built. The team's `example.py` pipeline is untouched; on this
branch `api.py` serves `pipeline.py` instead.

| File | What it does |
|---|---|
| `asr.py` | Transcribes exactly the annotators' way (`base`, int8, CPU, pinned snapshot), then words → numbered sentences |
| `evidence.py` | Maps a quote and/or sentence numbers → span: exact match, then fuzzy, then whole sentences, then word overlap |
| `answering.py` | The answerer interface (`Answer(p_yes, lines, quote)`) plus the no-model `lexical` answerer |
| `pipeline.py` | `predict()`: ASR → answerer (50 s deadline) → yes if `p_yes ≥ 0.22` → span. Never raises |
| `eval_offline.py` | Offline harness on cached transcripts, scored by the official `Statistics` class, with a threshold sweep |
| `check_timestamps.py` | The 390/390 gate for any serving machine |

Harness results (390 questions; the `gold*` rows read the labels to measure ceilings):

| Answerer | Accuracy | tIoU | Score |
|---|---|---|---|
| `gold`: exact quotes | 1.000 | 1.000 (195/195 exact) | 1.000 |
| `gold-noisy`: quotes with typos, no case or punctuation | 1.000 | 0.999 (193/195 exact) | 0.999 |
| `gold-lines`: whole sentences only | 1.000 | 0.873 | 0.924 |
| `lexical`: no model, the fallback | 0.656 | 0.508 | 0.567 |

```
python eval_offline.py cache            # once, ~5 min on a laptop CPU
python eval_offline.py score gold       # must print 1.000
python eval_offline.py score llm:answer --verbose     # step 5 onward
ANSWERER=llm:answer python api.py       # serve it
```

**Step 5 is written, but no real model has been run through it yet**
(`llm.py`). It talks to a local OpenAI-compatible server (vLLM or
llama.cpp):
- One request per question, all ten in parallel, with the instructions,
  worked examples and transcript as a shared prefix.
- The reply is four labelled lines: `LINES`, `QUOTE`, `CHECK`, `ANSWER`.
- `p_yes` comes from the logprobs at the answer token.
- The 11 worked examples are verbatim training excerpts. Their 7 source
  conversations are listed in `FEW_SHOT_SOURCES`, and the harness leaves them
  out automatically (320 questions scored).
- Tested against a fake server:
  - parsing and logprob extraction work;
  - 10 requests run in parallel;
  - a missed deadline falls back per question;
  - "no server" logs a warning and falls back without crashing.

## Rented 5090 runbook (vast.ai)

1. Rent the instance:
   - on-demand, not interruptible
   - plain CUDA ≥ 12.8 image with Python
   - disk ≥ 100 GB
   - expose port 9054
2. Get the code:
   `git clone https://github.com/emermelada/nordic-ai-cup && cd nordic-ai-cup && git checkout medical-appointment-exact-evidence && cd medical-appointment`
3. Run `bash gpu_setup.sh`. It:
   - installs two venvs
   - fetches the training data
   - runs the timestamp gate (expect 390/390)
   - starts vLLM with `nvidia/Qwen3.6-35B-A3B-NVFP4`
   - scores offline, writing to `data/logs/offline_score.txt`
4. Read the offline score and the threshold sweep. If the best τ differs from
   0.22, set `YES_THRESHOLD`. Offline runs cost no attempts, so iterate here.
5. Run `tmux new -s med 'bash gpu_setup.sh serve'`, then submit
   `http://<public ip>:<port mapped to 9054>/predict` and validate.
6. Stop the instance. On Sunday, repeat steps 2–5 on whatever box you get, and
   do one validation before the evaluation.

If vLLM will not run the NVFP4 build on this GPU, there are two alternatives:
- `LLM_HF_MODEL=openai/gpt-oss-20b LLM_THINKING=omit LLM_MAX_TOKENS=2048 bash gpu_setup.sh`
- llama.cpp with `unsloth/Qwen3.6-35B-A3B-GGUF` (UD-Q4_K_XL, 22.4 GB):
  `llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL -ngl 999 -c 65536 -np 10 --port 8080 --jinja --alias local`

`llm.py` works with either.

## Scoring levers besides the timestamps

1. **Lean towards yes.** A yes on a true positive earns accuracy and tIoU; a no
   on a negative earns accuracy only. Answer yes when
   `P(yes) ≥ τ = 0.4 / (0.8 + 1.2·t)`, where `t` is the tIoU a correct yes
   usually earns. At t ≈ 0.85, τ ≈ 0.22, not 0.5. A missed positive costs
   about 3.5× what a false yes costs. Tune τ on the offline harness.
2. **Never return `null` on a yes.** It scores 0 on the larger half.
3. **Fail towards yes.** If the LLM fails or runs out of time, answer yes with
   the best word-overlap sentence as the span. The exception is a question
   sharing no content word with the transcript (off-topic), which gets no. Our
   current code falls back to no, which is the worse bet.

## Target pipeline

```
POST /predict
 ├─ base64 → temp .mp3 (path-based decode, same as the annotators)
 ├─ ASR-T  WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=<all cores>)
 │         .transcribe(path, language='en', word_timestamps=True)     4–15 s on a laptop CPU at 12 threads
 │                                                                     (up to ~25 s at the library's default)
 │         → words (start, end, text); keep the raw floats, never round
 │         → sentences: split after words ending in . ? !  (numbered, timestamped)
 ├─ LLM    local server; one request per question, 10 in parallel, transcript as shared prefix
 │         → {"lines":[a,b], "quote":"...", "check":"...", "answer":"yes|no"} + P(yes)
 ├─ decide yes if P(yes) ≥ τ
 ├─ span   quote → exact token match in base words (lines a..b ±1) → fuzzy match → whole lines a..b
 └─ guard  50 s wall clock; anything unanswered gets the lexical fallback above
```

`base` int8 stays on the **CPU** even on a GPU box. It is cheap, and only the CPU
int8 path has been shown to reproduce the labels. CUDA is untested and its
kernels likely round differently.

## Steps, in order, each with a gate

**1. Serving box + timestamp gate (1 h).** Pick the machine (see Hardware),
install the pinned versions (`faster-whisper==1.2.1`, `ctranslate2==4.8.2`,
`av==18.1.0`), and run `python check_timestamps.py` there.
*Gate: 390/390.* If it reports fewer, it is still usable: float32-level drift
costs about 1% tIoU. Note it and move on.

**2. ASR swap — done (`asr.py`).** The exact call above. It returns words and
sentences, with no VAD and no rounding. `eval_offline.py cache` stores the
words of the 39 training conversations as JSON under `data/transcripts/`.

**3. Offline harness — done (`eval_offline.py`).** Take the cached words and
a prompt/model config, run all 390 questions against the LLM server, and report:
- accuracy by type
- mean tIoU (the official formula in `utils.py`)
- exact-quote rate
- right-sentence rate
- P(yes) calibration and the best τ
- p50/p95 time per conversation

Few-shot examples must come from conversations outside the scored fold.
*Gate: the harness reproduces 1.000 when fed gold quotes.*

**4. Quote → span mapper — done (`evidence.py`).** Compares letters and digits
only, so case, punctuation, hyphens and spacing don't matter. Then:
1. Look for an exact match anywhere, preferring the one nearest lines a..b ±1.
2. Otherwise run a character-level `difflib.SequenceMatcher` over lines a..b ±2.
   Accept if ≥60% of the quote's characters match without smearing past 1.6× its
   length.
3. Otherwise use the whole of lines a..b.
4. Otherwise use the best lexical sentence.

Return `(words[i].start, words[j].end)` as raw floats.

**5. M1: sentence-level evidence (3 h), first validation attempt.** The LLM
returns line ranges only, and the span is the whole lines. Projected: accuracy
≈0.92, right lines ≈90% → tIoU ≈ 0.78 over positives → **≈0.83**.
*Gate: offline ≥ 0.82 before spending a validation attempt.*

**6. M2: quote-level evidence (Sat).** Add the quote field and the style rules
and few-shots below. The mapper does the rest. Projected **0.85–0.90**.
*Gate: offline exact-quote rate ≥ 50% and tIoU up on M1.*

**7. τ, model bake-off, latency (Sat).** Tune τ on the harness. Try 2–3 LLMs.
Keep p95 per conversation ≤ 40 s on the serving box.

**8. Freeze (Sun ≤ 11:00).** One clean validation run on the exact final setup,
then the evaluation, leaving at least 3 h of buffer.

## Prompt (per question, transcript as a cached prefix)

System: you check claims against the transcript of a recorded GP consultation.
The transcript is speech recognition. Numbers are reliable. **Drug names are
often misspelled phonetically**: "Ibu Medin" = Ibumetin, "pan top resolve" =
pantoprazole, "active L4" / "Active L" = Activelle, "Aromere" = Airomir.

User: the numbered transcript (`[12] 41.02–43.58 Your hemoglobin A1c is 42
millimoles per mole.`), then the question, then:

1. Find the passage that bears on the question and quote it verbatim.
2. Compare every detail against it: drug, dose, unit, frequency, duration, body
   site, timing, result value, who said it, and whether it was done/agreed or
   only mentioned. The same topic with a different detail is **no**. A subject
   never discussed is **no**.
3. Answer. JSON: `{"lines":[a,b],"quote":"...","check":"one sentence","answer":"yes|no"}`

Read P(yes) from the answer token's logprobs (llama.cpp and vLLM both return
them). The fallback is 5 samples at T=0.7 and a vote.

**Quote rules, measured on the 195 gold quotes:**
- 76% are one sentence, 15% two, the rest 3+ (typically a question-and-answer exchange).
- 86% start at a sentence start. Otherwise a lead-in is dropped: "And",
  "So,", "Then", "Yes,", "That is right,", "It means", "Putting it together,",
  "For the sinuses,", "From what you describe,".
- 81% end at a sentence end. Otherwise the quote stops right after the last
  word the fact needs.

**Few-shots (verbatim `base` text; the `[[ ]]` part is the gold quote):**

| Question | Transcript, gold quote in `[[ ]]` |
|---|---|
| Is Pamol one of the medicines requested? | Yes. [[I am creating prescriptions for both Pamol]] and Ibu Medin now, … |
| Were both prescriptions issued? | Yes. [[I am creating prescriptions for both]] Pamol and Ibu Medin now, … |
| Should the daily dose be 100 mg? | Sporanox, [[100 milligrams daily]] for two weeks. |
| Did the patient undergo surgery for a cervical disc prolapse in 2017? | That is right, [[the cervical disc prolapse operated in 2017.]] |
| Is the patient still being treated for asthma? | [[I am still being treated for asthma]] and for reflux. |
| Is the patient also taking Pantoprazole? | All right. [[And you are still taking Pantoprazole alongside it? Yes, every day with it.]] Good. |
| Is Airomir among the renewed medicines? | [[Active L, Aromere and Esomeprizol. All three renewed.]] |
| Will the current treatment continue unchanged? | So, [[what happens with my treatment? No changes. You carry on exactly as you are.]] |

Add one or two hard-negative examples, e.g. "Is the LDL cholesterol 4.2
mmol/L?" against "Your LDL cholesterol is 2.2 millimoles per liter." → no.

## Hardware and models

| Serving box | ASR-T (`base` int8, CPU) | LLM | Notes |
|---|---|---|---|
| Rented x86 GPU, RTX 4090 24 GB / 5090 32 GB | its CPU; run the check | Qwen3.6-35B-A3B or gemma-4-26B-A4B-it at 4-bit, via llama.cpp server or vLLM | strongest and simplest; public IP, no tunnel |
| RTX 3060 desktop, x86, 12 GB (free if Drone serving moves to the 5090) | its CPU; run the check | gpt-oss-20b, Qwen3.6-35B-A3B with experts on CPU (`--n-cpu-moe`), or an 8–14B dense model at Q4 | latency to be measured |
| M4 Mac | CTranslate2 on ARM: int8 exactness unverified, run the check | MLX, sized by RAM | fine if the check is ≥ float32-level |

Choose the LLM by offline score first and p95 latency second. The LLM runs as a
separate local server, so the FastAPI container no longer needs torch or
transformers. The request path stays fully local, as the rules require.

## What to drop from the current `example.py`

- `small.en` with `vad_filter=True` measures in the wrong coordinate system. VAD
  re-times every word.
- Segment-level spans cap tIoU at 0.75.
- Qwen2.5-0.5B is far too weak for near-miss negatives.
- Top-3 word-overlap retrieval finds the right sentence only about half the
  time (the lexical floor above).
- Answering no on errors; answer yes instead (see Scoring levers).

## Risks

- **Serving CPU rounds int8 differently.** Run the check; the worst case is
  about −1% tIoU.
- **Wrong passage is the largest remaining loss.** Mitigations: numbered
  transcript, quote-first reasoning, few-shots, harness numbers before any
  validation attempt.
- **Quote style mismatch.** The mapper falls back to whole lines, 0.87 at worst
  when the passage is right.
- **Drug-name errors in `base` text.** Prompt hint first. If the harness still
  shows drug-name hard negatives wrong, add a larger ASR (e.g. large-v3-turbo on
  the GPU) as *text-only* reference context, never for timestamps.
- **Timeouts.** 50 s guard with fallback answers. Five timeouts in a row ends
  the attempt.
- **Overfitting the 39 conversations.** Use cross-fold harness numbers. The 19
  validation conversations are the honest check. Don't tune per question against
  the validation leaderboard; the evaluation set is different.
- **Version drift.** Pin the three packages above and keep the model snapshot
  `Systran/faster-whisper-base@ebe41f7`.

Log faster-whisper `base`, the chosen LLM and Claude Code in `AI_TOOLS_LOG.md`.
