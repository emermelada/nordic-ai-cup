# Medical appointment: roadmap toward 0.90 raw

Research snapshot: 2026-09-17, commit `5dc2b86`. This is an improvement plan, not a report of implemented gains. No inference models, server processes, or competition submissions were changed during this investigation.

## Recommendation

Keep the validated 9B answerer as the control, but replace the single-quote bottleneck with **multiple occurrence-aware evidence candidates, word-level boundary selection, and question-conditioned ranking**. Revisit examples, fine-tuning, larger models, reasoning, and ASR as contributors to that system—not only as wholesale replacements of the current pipeline.

Earlier negative runs are evidence about particular configurations. They are not permanent exclusions of those approaches.

## 1. Baseline and the size of the gap

The best documented platform validation is **0.743**, using Qwen3.5-9B-4bit, compact prompt, fp16 Whisper turbo, numeric alignment, short-reply extension, and speech-onset refinement. Current source defaults match that model/prompt/ASR combination (`pipeline/mlx_backend.py:10–22`). The loaded server process was not queried. Parts of `RUNNING.md` still describe the preceding REAP/8-bit configuration; `HANDOFF.md` and much of `PLAN.md` describe an earlier implementation stage.

Replaying the surviving 9B outputs reproduced all 390 saved per-question rows:

| Metric | Local public-training baseline |
| --- | ---: |
| Correct answers | 384/390 = 0.984615 |
| Correctly detected positives | 193/195 |
| Correct hard negatives / off-topic questions | 139/142 / 52/53 |
| Mean tIoU over all 195 positives | 0.594369 |
| Raw score | 0.750468 |
| Zero-overlap positives | 31: 29 present spans, 2 missing |
| Positives with some overlap | 164 |
| Mean tIoU conditional on overlap | 0.706720 |

These are **local**, not hidden-validation component metrics. No platform accuracy/tIoU breakdown or repeated-identical-build series was recovered; an asserted ±0.02 evaluator-noise range is not established.

With `raw = 0.4 × accuracy + 0.6 × mean_tIoU`:

| Raw target | Required tIoU at baseline accuracy | Required tIoU at perfect accuracy |
| --- | ---: | ---: |
| 0.90 | 0.84359 | 0.83333 |
| 0.95 | 0.92692 | 0.91667 |
| 1.00 | Impossible without improving accuracy | 1.00000 |

Perfecting classification alone, with spans unchanged, adds only **0.00615 raw**. Both wrong-passage selection and the extent of correctly located passages need improvement. Even granting perfect spans to every currently zero-overlap case only reaches about 0.848 after also correcting the two missed-positive answers, with other predictions unchanged.

### New oracle diagnostics

An oracle uses gold annotations to choose candidates during analysis. It is not a deployable selector, and its score is not a forecast.

| Available span representation/pool | Oracle mean tIoU | Raw upper bound with perfect answers |
| --- | ---: | ---: |
| Best single current clause unit | 0.70638 | 0.82383 |
| Best one or two adjacent units | 0.78327 | 0.86996 |
| Fixed outputs from four models: 8B, 9B, 27B, REAP | 0.77252 | 0.86351 |
| Fixed outputs from 16 retained model/prompt/ASR configurations | 0.81622 | 0.88973 |
| Four-model pool plus word-boundary alternatives within ±2 seconds of each endpoint | 0.86175 | 0.91705 |
| Same four-model pool with ±4-second endpoint alternatives | 0.88892 | 0.93335 |
| Best contiguous fp16-Whisper word interval, with onset refinement | 0.94498 | 0.96699 |

Consequences:

- **Whole-unit selection alone cannot reach 0.90 on this data.** Final boundaries must be word-level or finer.
- **The existing fixed-span ensemble pool is insufficient even with perfect selection.** Ensembling remains valid, but needs new passages or new boundaries, not just a vote over these intervals.
- Diverse passages **plus** local word-boundary alternatives have enough geometric coverage to cross 0.90 in an oracle. At fixed baseline accuracy, the four-model ±2-second oracle is 0.91089, leaving only 0.01089 raw of selection headroom. Better candidate coverage is still desirable; this is not evidence that a practical ranker already reaches the target.
- The four-model bank is an offline diversity diagnostic, not a feasible four-large-model serving proposal for this Mac. Reproduce useful diversity with retrieval, a small specialist, targeted proposals, or offline distillation.
- These alternatives are contiguous word spans with each endpoint near an existing prediction, not a rule that pads every response by two seconds. The generator does not inspect gold; only the oracle selection does. Small ±0.5-second adjustments to the 9B prediction alone raise its oracle raw score only from approximately 0.7505 to 0.7580.
- Twelve positives have no overlap in any of the 16 saved configurations. Diagnose these separately from boundary errors.
- The word oracle measures timestamp representability, not semantic understanding. It can select a greeting merely because the annotation points there. It does not prove that ASR wording is perfect or that 0.967 is a universal task ceiling.

## 2. What earlier experiments actually establish

| Family | Retained evidence | Meaningful next experiment |
| --- | --- | --- |
| Examples | One Qwen3-8B demonstration improved matched 38-conversation raw score **0.74623 → 0.75835**; accuracy fell, tIoU rose. | Cross-fitted, diverse examples for evidence selection on 9B; preserve the baseline answer decision while comparing evidence proposals. |
| Separate evidence pass | On 30 development conversations, **0.7258 legacy → 0.7453 second pass**; focused one-pass scored 0.7550. | Verify only ambiguous cases against explicit alternatives, rather than regenerate every positive quote from the same full transcript. |
| Per-question inference | Only three conversations were completed; that unconditional version lost. | Batched/selective passage and boundary verification, with a fixed time allowance and baseline fallback. |
| LoRA | One completed Qwen3-8B fold: 26 train/13 test; matched test raw **0.82326 → 0.74608**, although accuracy improved. Rank 8, 80 steps, LR 1e-4; monitoring validation reused two training examples. | Evidence-specific targets, real inner validation, less aggressive training, more varied supervision, and completed conversation-level folds. Test a discriminative span/ranking model before another full-JSON LoRA. |
| Prompt wording | Several 9B instructions were flat/worse; first-mention wording was slightly better locally. | Change the information/representation: candidate alternatives, unique positions, contrasting boundary examples, or short structured verification—not only adjectives in the instruction. |
| Larger models/reasoning | Several swaps improved local raw score without improving platform validation; 27B also stressed memory. Medium reasoning lost on a 12-conversation gpt-oss comparison. | Evaluate evidence quality and complementary proposals; use a stronger model as an offline teacher or a time-bounded specialist. Short pointer outputs and current ASR unloading change the resource trade-off. |
| ASR/alignment | Turbo 8-bit slightly improved local score; full large-v3 worsened it. Timing oracles remain much higher than actual selection. | Target drug/number disagreements and low-confidence crops; separately test text quality and timestamp quality. Try forced alignment on an already selected passage. |
| Reranking/boundaries | Citation-priority matching and simple sentence snapping disappointed. No completed learned candidate-ranker experiment was identified. | Learn question-conditioned passage preference and start/end boundaries; do not equate lexical matching or blanket padding with this experiment. |

The public training set and its earlier 30/9 split have already been inspected and tuned against. A new split improves discipline but does not make the same data genuinely unseen.

## 3. Ordered implementation experiments

### E0 — Preserve the control and make failures attributable

Use the exact documented best recipe as the control; compare 8-bit ASR separately rather than silently changing both components.

- Preserve the recovered baseline, later generation caches, scripts, and original scores in an unused durable experiment directory before temporary-session cleanup removes them.
- Extend existing evaluation tools with a small immutable manifest: code revision/diff, model and adapter snapshot, complete prompt/template settings, ASR/cache identity, split membership, decoding settings, and refinement version.
- Trace each question's raw quote/citations, candidate word positions, pre/post-refinement span, fallback source, and stage timings. Do not put labels or gold timestamps into model inputs.
- Report accuracy by type, zero overlap, IoU conditional on overlap, candidate oracle mean IoU, candidate coverage at IoU ≥0.5/0.8, selected-vs-oracle gap, and start/end errors.
- Break the 31 baseline zeros into missing classification, wrong repeated mention, incomplete/ambiguous quote, ASR mismatch, and possible annotation anomaly. Separately inspect under- and over-extended overlapping spans.

Existing entry points: `tools/eval_offline.py`, `tools/compare_evidence.py`, `tools/transcribe_all.py`. Replay currently applies the current `refine_evidence` implementation, so saved raw output alone does not fully version a historical result.

**Exit:** one reproducible baseline and an error table that distinguishes candidate absence, candidate selection, and boundary quality.

### E1 — Correct narrow grounding/refinement defects

These behaviors were reproduced synthetically. Their prevalence and score contribution on real conversations remain to be measured; they are not a promised 0.15-point improvement.

1. **Refine each span once.** Runtime refines retrieval fallback when ASR arrives, then can refine inherited fallback spans again after generation. Question extension is not idempotent (`pipeline/runtime.py:299–308`). Keep an unrefined internal fallback and refine at the response boundary, or preserve explicit refinement provenance.
2. **Guard question/reply linking.** The current rule can attach an unrelated next question or a reply after a long pause. Require plausible temporal/turn adjacency and a supporting response; retain support for necessary question-plus-answer evidence, not just short standalone replies (`pipeline/evidence.py:122–145`).
3. **Do not confuse repaired JSON with complete evidence.** A truncated quote such as “The dose is” can be accepted. Preserve completed boolean decisions, but route incomplete grounding through validated alternatives (`pipeline/core.py:256–271,331–355`).
4. **Preserve occurrence identity.** Global alignment breaks exact-repeat ties by first occurrence and ignores cited units until matching fails. Retain alternative occurrences or use validated word pointers; do not blindly restore citation priority, which previously regressed (`pipeline/evidence.py:61–95`).
5. **Guard semantic corruption in alignment/fallback.** Numeric matching protects decimals, but sets lose order/multiplicity; fuzzy text can match normal/abnormal, and cited-unit fallback can bypass a rejected numeric quote. Validate candidate support in context without automatically flipping correct boolean answers.
6. **Constrain onset refinement.** Its one-second forward scan can skip a quiet first word such as “No.” Evaluate a first-selected-word limit or acoustic alignment rather than replacing the beneficial existing rule with another global offset.

Implement/test each independently, then score the combination on the same replay. Preserve the existing decimal tests, process isolation, deadline/recovery behavior, sanitization, and response protocol. Core/evidence CPU tests passed in this investigation; synthetic correctness cases should become regressions when implemented.

**Exit:** defects corrected generally, with any measured score trade-offs explicit. No unrelated pipeline refactor.

### E2 — Build candidates that can actually support a 0.90 score

This is the primary high-upside change.

- Keep the current 9B quote/span as a candidate.
- Add distinct supporting passages from semantic/lexical retrieval and, when useful, a second evidence proposal. Retain repeated mentions with their local context instead of merging identical text.
- Within shortlisted passages, generate **word-level** start/end alternatives: relevant subclauses, complete propositions, question/reply pairs, and supported multi-sentence spans. Include small learned/candidate boundary changes; avoid a fixed three-second or one-sentence rule.
- Give candidates stable global word indices, or unit plus local word offsets. Compare model-generated pointers against exact quotes; validate range, order, occurrence, and support before converting to timestamps. Pointer validity alone does not guarantee correct evidence.
- Preserve full-conversation context for distinguishing current versus historical medication, proposals versus agreements, and repeated symptoms/findings. Speaker/turn information is a feature, not a rigid “always choose the doctor” rule.

First measure the candidate oracle, separately for each source and their union. **Aim for at least 0.90 oracle mean tIoU on held-out conversations** so a real selector has headroom above the required ~0.844. This is an engineering target, not a measured outcome. If the pool is deficient, expand its support/boundaries before investing in a complex selector.

**Exit:** high candidate coverage with a practical candidate count and stable positions. Merely returning larger spans is not success.

### E3 — Learn which supported span matches the annotation

Start with a small supervised ranker as a diagnostic. Train on `(question, candidate, surrounding context)` with gold temporal IoU as a soft quality target. Include misleading but topically similar passages, alternate true mentions, incomplete clauses, and overlong spans.

Useful signals: subject/action/value/polarity coverage, relation of numbers to units, local discourse context, span length, exact-copy quality, baseline preference, and ASR confidence. Do not label an alternative true mention “clinically false” merely because its annotated IoU is zero; entailment and annotation preference are different objectives.

If a simple ranker cannot exploit a good candidate pool, compare a small local cross-encoder or **extractive start/end model**. Yes/no questions are compatible with predicting their supporting evidence; the target is a passage, not the literal word “yes.” This directly differs from an off-the-shelf SQuAD answer extractor and from generative full-response LoRA.

For higher-capacity training:

- Map gold intervals to candidate word boundaries and audit ambiguous mappings; avoid silently training on malformed or semantically unrelated evidence.
- Use conversation-level outer folds and a genuinely disjoint inner early-stopping split. Demonstrations, synthetic derivatives, and all questions from one conversation stay in its training fold.
- Generate additional independent consultations/questions with known source passages. Teacher-generated labels must be checked for exact grounding and annotation-style mismatch. Many variants of the same 39 conversations are not independent new data.
- Local stronger models can label offline without the inference deadline. Hosted development assistance is permitted by competition rules, but spending money or uploading data needs separate approval. No hosted service enters `/predict`.
- Revisit 9B LoRA with evidence-only/pointer targets, regularization and checkpoint selection if it earns its complexity; preserve or independently verify the strong boolean branch.

**Exit:** out-of-fold selected IoU closes a meaningful part of the oracle gap, with classification stable and enough improvement to merit platform validation. Do not promote training fit alone.

### E4 — Spend extra inference only on unresolved cases

Use candidate disagreement, near-tied ranks, incomplete support, repeated occurrences, or entity/number disagreement to trigger a short verifier. Ask it to choose/refine explicit alternatives, not just repeat the first prompt. Larger models, bounded reasoning and cross-ASR comparison remain eligible here.

Keep a completed baseline response available before optional work starts. If refinement times out or fails, return that response—not a weaker retrieval fallback. Batch ambiguous questions where possible; do not assume ten independent model calls fit.

Benchmark on this 24 GB M4 Pro with the answering model and actual transient allocations. ASR unloading already exists, so test the current memory schedule rather than treating an older out-of-memory result as a permanent model-size ban. Nevertheless, swapping or tunnel instability disqualifies that serving configuration.

**Exit:** gain on difficult cases without losing the first-pass result or the deadline margin.

### E5 — Close the remaining gap toward 0.95+

Once passage selection is strong, isolate timing by fixing the selected words and comparing current timestamps, local forced alignment, and speech-boundary refinement. Measure text corrections separately from timing changes; a new ASR may improve one while harming the other.

At baseline accuracy, 0.95 requires tIoU about **0.927**, close to the measured **0.945** word-grid oracle. Reaching beyond that representation's local ceiling requires better timestamp representation as well as better answers and selection.

Cached transcripts indicate that two training annotations, `sample_63_yes_q02` and `sample_64_yes_q02`, point to opening greetings rather than the later supporting content; this investigation did not audio-audit them. Audit and flag them; do not special-case their IDs/filenames in inference, alter official gold, or remove them from headline scoring. Exactly 1.0 also demands resolving annotation ambiguity and reproducing exact endpoints; it is an aspiration, not a justified promise from current evidence.

## 4. Validation and delivery gates

- Compare candidates on the same conversations, transcript version, and scoring/refinement version. Reconstruct full denominators from generation caches when a scratch CSV contains only a subset.
- Use paired conversation-level comparisons and bootstrap intervals; 390 correlated questions are not 390 independent consultations. Freeze folds and avoid endlessly choosing winners on the previously inspected nine-conversation split.
- Evaluate positive recall and false positives separately; never force five yes answers per request merely because the aggregate dataset is balanced.
- Submit only selected candidates to platform validation, recording build identity and every available component metric. Unlimited validation is not immunity to hidden-set overfitting. Platform improvement matters more than a reused-training-set gain.
- Run the unchanged official scoring oracle, relevant tests, all 39 fresh audio-to-response requests, and the longest/stress cases before promotion. Target p95 <40 seconds and worst observed <45 seconds, with zero failed requests; retain the existing 52-second inference watchdog and transport margin.
- Do not run competing GPU experiments during platform requests. Keep the tunnel stable; serving-process replacement is a separately authorized action.
- Preserve the best validated build even if the stretch target is not reached. No experiment spends the single final evaluation attempt.

### Suggested order before the deadline

1. **First work block:** archive/replay the best run, add attribution traces, and implement E1 fixes with isolated comparisons.
2. **Next block:** E2 candidate/pointer prototype and oracle coverage; begin the most promising evidence-only few-shot comparison in parallel.
3. **Following block:** E3 ranker/extractor and real held-out training; add E4 only if ambiguity diagnostics and timing justify it. Validate the best complete candidates, not each tiny tweak.
4. **2026-09-20:** freeze the selected build by approximately 10:00 CEST; complete final checks and leave time for an explicitly authorized final attempt around noon, ahead of the **16:00 CEST** deadline.

The first implementation should therefore be **traceable baseline + narrow grounding fixes + a candidate/word-boundary oracle**, followed by the selector that the measurements justify. This offers a credible experiment path to 0.90 without pretending a combination of unmeasured gains is guaranteed.

## Evidence locations

Repository references: `RUNNING.md:55–95,192–220`; `pipeline/core.py:16–24,256–356`; `pipeline/evidence.py:11–47,50–146`; `pipeline/runtime.py:291–310`; `tools/eval_offline.py:132–268`.

Recovered experiment root (ephemeral; archive before relying on it long-term):

```text
/tmp/claude-501/-Users-chinese-AICUP-Nordic-AI-Cup-2026-medical-appointment/a82ae223-4597-4326-aa17-c22cf4543823/scratchpad/
```

Within that root:

- `runs/q35-9b-compact-units/{gen/,per_question.csv}`: matched 9B baseline.
- `runs/`: other model/prompt/ASR generation caches and per-question outputs; some CSVs were overwritten by subset rescoring, so check denominators.
- `runs_fewshot.log:39–45`: directly matched few-shot comparison.
- `lora/adapters/q3-8b-lora-units-fold0/adapter_config.json`, `lora_data.py:46–62`, `runs_lora_q3.log`: actual LoRA recipe/split/completion evidence.
- `exp.py:109–172`, `prompts.py`, `ceiling.py`, `energy/`: replay generation/scoring, tested prompts, and timing-oracle inputs.

New union-oracle numbers above were computed read-only from saved predictions using the official temporal-IoU function and the full 195-positive denominator. Perfect-label raw upper bounds are `0.4 + 0.6 × oracle mean tIoU`; no trained selector or fresh inference produced those scores.
