**Research for a substantial improvement in Medical Appointment — 19 September 2026**

The strongest next investment is **task-specific evidence learning: train on substantially more varied, precisely labeled evidence spans, and teach the model the difference between the annotated passage and plausible competing passages**. Keep the successful answer system as the control. Compare a contextual start/end extractor with an evidence-only generative model on the same training data. This changes the training signal and the output representation, rather than asking the current model to reconsider the same passage again.

This is a research recommendation, not a measured improvement. A score above 0.90 remains a stretch target. None of the papers below establishes that its gains will transfer to this competition.

I searched academic literature, publisher pages, authors’ repositories, shared-task reports, GitHub discussions, and Reddit. Direct Google Scholar search was inaccessible through the browser tool; the academic sources were retrieved through ACL Anthology, arXiv, IJCAI, JMIR, university sites, and authors’ code instead. Searches for this exact competition did not uncover an independently verifiable public solution write-up. The competition is still in progress according to its [official website](https://nordicaicup.com/). Similar-task evidence is therefore more useful than an alleged winning recipe for this exact dataset.

**Your current evidence narrows the problem considerably.**

The first section of [RUNNING.md](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/RUNNING.md:3) documents a best platform validation of **0.8167711209** for the locked build: Qwen3.8-27B, base-model timestamp coordinates, per-question evidence extraction, and answer rescue. That is a documented result; I did not reconnect to the GPU or independently rerun platform validation. Later sections preserve older states and sometimes conflict with the opening section, so they should not be treated as the current configuration.

I independently recomputed the saved, runtime-faithful public-training control. It is an earlier control, not a replay of the final rescue build:

| Quantity | Recomputed result |
| --- | ---: |
| Conversations / questions / positives | 39 / 390 / 195 |
| Answer accuracy | 390/390 |
| Mean temporal IoU | 0.716181 |
| Combined raw score | 0.829709 |
| Evidence with zero overlap | 21 |
| Evidence with overlap below 0.5, excluding zeros | 32 |
| Evidence with overlap from 0.5 to below 0.9 | 37 |
| Evidence with overlap at least 0.9 | 105 |

The [reproducible audit](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/research/evidence_research_20260919/saved_control_audit.json) records input hashes. The [195-row evidence comparison](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/research/evidence_research_20260919/positive_span_audit.csv) includes each question, reference passage, predicted passage, and overlap score.

The score equation gives a useful target:

| Desired raw score | Required mean tIoU at 100% accuracy | Required mean tIoU at 99% accuracy |
| --- | ---: | ---: |
| 0.85 | 0.7500 | 0.7567 |
| 0.88 | 0.8000 | 0.8067 |
| 0.90 | 0.8333 | 0.8400 |
| 0.95 | 0.9167 | 0.9233 |

Even replacing all 21 zero-overlap predictions with perfect spans raises this saved control only to **0.894324**. The 32 low-overlap predictions collectively lose another **0.064913 raw**, almost exactly as much as the 21 complete misses. A major improvement must handle both occurrence selection and passage boundaries.

For example, the question “Will the treatment last two weeks?” has an annotated passage “After a meal every day for two weeks.” The prediction “Sporanox, 100 milligrams daily for two weeks.” is elsewhere and scores zero. Both convey duration. Similarly, “Nothing abnormal to report.” is the gold passage for one heart-examination question, while “Your chest and heart both sound normal.” is the zero-scoring prediction. These examples explain why a generic entailment verifier can be right about meaning yet wrong for the scoring rule.

Two reference intervals map to the word “Good” near the opening of their transcripts: `sample_63_yes_q02` and `sample_64_yes_q02`. These are suspicious labels, not confirmed corrections; I inspected cached text, not the underlying audio. Keep them in official scoring, and audit them before using them to teach a model. Do not create filename or question-ID exceptions.

The documented base-coordinate word-span oracle is about **0.9859 tIoU**. That is a gold-informed representational ceiling, not an achievable model result. The documented candidate-pool oracle of **0.8619** similarly says the pool contains useful answers; it does not tell us how to select them. These figures support work on evidence selection, but cannot justify promising 0.90.

**The most relevant other teams and research groups found several different ways forward.**

| Work | What they actually did and measured | What transfers to this project |
| --- | --- | --- |
| **RadQA-DPO — Nahian and Kavuluru, BioNLP 2025** | Clinical extractive QA with supervised T5-family models, followed by preference learning. Test F1: best earlier BERT baseline **63.55**; supervised Flan-T5-3B **76.38**; its model-negative DPO version **77.48**. T5-3B improved **72.29 → 75.18** with combined preference data. The headline 12–15-point gain is not the isolated DPO gain. [Paper, Table 2](https://aclanthology.org/2025.bionlp-1.10.pdf), [code](https://github.com/sultanalnahian/RadQA-DPO). | Learn from actual extraction mistakes, including incomplete and oversized spans. Use temporal overlap and occurrence identity for our preferences. This is different from asking an unfine-tuned model to rank candidates at inference. |
| **SUnsET — Wright and colleagues, Copenhagen/Michigan, EMNLP 2025** | Trained models to cite arbitrary-length evidence using synthetic supervision. Their staged generation process creates evidence and surrounding documents, refines them, and filters poor examples. Released **2,352 documents / 11,309 tuples**; evaluated five models on four datasets. [Paper](https://arxiv.org/abs/2502.14409), [generation and training code](https://github.com/dwright37/unstructured-evidence-sunset). | Generate evidence with known locations first, then plausible consultations around it. Train exact evidence extraction. Their summary-quality and citation findings are not a temporal-IoU benchmark result. |
| **KR Labs — ArchEHR-QA 2025** | Generated **58,000 sentence-level training examples** for Clinical ModernBERT. Development extraction F1 was **0.61**, versus **0.63** for Gemma-3-27B; final pipeline scores were **41.85** and **42.01** respectively. The submitted LLM pipeline beat the organizer’s **30.7** baseline. The small extractor matched rather than surpassed the LLM. [Paper, Tables 2–3](https://aclanthology.org/2025.bionlp-share.8.pdf), [authors’ project](https://www.krlabs.eu/research/papers/verbatim-rag-attribution-2025/). | A compact, trained extractor is a credible alternative. Sentence labels alone are too coarse here; add word-level boundaries and repeated-occurrence supervision. |
| **Lanz and colleagues — JMIR, August 2026** | Clinical evidence extraction and normalized answering were separated. Across five languages and more than 1,500 discharge summaries, reported evidence scores were **85% F1 / 79% exact match**, versus **95% answer accuracy given gold evidence**. End-to-end accuracy was 88%. [Study](https://www.jmir.org/2026/1/e96347/). | Closely mirrors our bottleneck: strong answer prediction does not solve evidence extraction. Their reporting-style dependence supports training to this task’s annotation conventions. Their notes are not consultation audio. |
| **CliniQG4QA — Yue and colleagues, 2020/2021** | Generated diverse clinical questions on new contexts for domain adaptation; reported up to about **8 percentage points exact-match improvement**. Diversity of question forms mattered. [Paper](https://arxiv.org/abs/2010.16021), [code](https://github.com/sunlab-osu/CliniQG4QA/). | Generate diverse formulations, including absence questions, tag questions, shortened references, and conversational confirmations. Rephrasing the same few questions is insufficient context diversity. |
| **Roundtrip synthetic QA — Alberti and colleagues, ACL 2019** | Generated QA data and retained examples whose independently recovered answers were consistent with the intended answers; improved SQuAD2 and Natural Questions. [Primary source](https://research.google/pubs/synthetic-qa-corpora-generation-with-roundtrip-consistency/). | Filter synthetic questions by recovering their intended evidence. For us, check the actual location as well as matching text; repeated strings can otherwise pass a bad roundtrip check. |
| **Neural1.5 — ArchEHR-QA 2026** | Used DSPy MIPROv2 to optimize instructions and demonstrations, with stochastic voting. Reported first place in evidence identification at **63.7 strict micro F1**. The report does not isolate all gains from the optimizer itself. [Paper](https://arxiv.org/abs/2605.10877), [optimizer documentation](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md). | A bounded search using our actual tIoU objective is a legitimate separate experiment. It is not equivalent to another hand-written prompt. However, our small, heavily reused dataset makes overfitting a serious practical concern. |

A valuable counterexample is the **sebis team at ArchEHR-QA 2026**, which worked entirely locally on a notebook. Their synthetic-data BioClinicalBERT evidence model scored **44.43** on test versus **51.61** for embedding retrieval; their fine-tuned alignment model scored **8.33**, versus **74.84** for a prompted Qwen system. These are different subtasks and metrics, but they show that smaller supervised models and synthetic data can fail badly. [Authors’ paper and result tables](https://arxiv.org/html/2603.13962v1). Our recommendation needs a properly separated evaluation, not faith in fine-tuning.

**There is also a genuinely different audio-model route, but its evidence is mixed.**

| Research direction | What changed | Relevance and decision |
| --- | --- | --- |
| Spoken SQuAD, 2018 → MRD-Net, IJCAI 2021 | Early work diagnosed ASR errors as a major source of spoken-QA failure. MRD-Net trained a noisy-transcript student using a clean-transcript teacher and an audio assistant. [Spoken SQuAD](https://arxiv.org/abs/1804.00320), [MRD-Net](https://www.ijcai.org/proceedings/2021/549). | Train robustness to sound-alike errors and transcript disagreement. Our global turbo/base hybrid already lost, so this supports supervised or selective use, not blindly replacing base text. |
| DUAL, Interspeech 2022 | Predicts answer intervals from discrete speech units without an ASR transcript. [Paper](https://arxiv.org/abs/2203.04911), [official code and NMSQA data](https://github.com/DanielLin94144/DUAL-textless-SQA). | A true architecture change with released resources. Medical-domain adaptation and timestamp-convention mismatch make it a longer-horizon project, not the first deadline experiment. |
| Listen First, Then Answer, March 2026 preprint | First teaches timestamp alignment, then grounded speech reasoning. Its reported alignment experiment improved Qwen2.5-Omni IoU **0.2324 → 0.7189**. [Method and Table 1](https://arxiv.org/html/2603.19468v1). | A large within-study improvement from explicit grounding supervision. It is not an improvement over our 0.716 control on the same data, and the task differs. Test only after a speech-grounding checkpoint and applicable training data are available. |
| TAG-Bench, September 2026 preprint | Evaluates temporal audio grounding over varied recordings. The best of 21 systems reached **31.2 mIoU**; nine were below 5. [Benchmark](https://arxiv.org/abs/2609.01542). | A reason to demand a local comparison before switching to a general audio LLM. It does not prove that any particular model will fail on our much narrower task. |
| Prosody study, NAACL 2025 | Found useful speech cues beyond text, while models largely favored lexical information when it was available. [Authors’ publication](https://machinelearning.apple.com/research/role-of-prosody). | Speaker turns and pauses could supplement boundary selection, but a full prosody model has weaker near-term justification than fixing the training objective. |
| VSLNet, 2020/2021 | Reframed temporal video localization as start/end span prediction, with query-guided highlighting. [Paper](https://arxiv.org/abs/2102.13558). | Borrow the supervised span-localization formulation. Transferring the video architecture and training data directly would be poorly matched to speech. |

One superficially perfect paper title was misleading for this request: **Attention-guided Evidence Grounding for Spoken Question Answering** handles **spoken questions over textual contexts**. Our questions are text and the evidence is in recorded consultations. Its attention-supervision idea is interesting, but its reported speedup does not establish a solution to our localization task. [Task definition](https://arxiv.org/html/2603.16292v1).

**Forums support some engineering choices, but they do not supply credible score forecasts.**

A [LocalLLaMA discussion about extractive models](https://www.reddit.com/r/LocalLLaMA/comments/1afdo7e) proposes token selection and teacher-to-small-model training. This is a useful implementation lead, not a controlled result. A [first-person report on Flan-T5 extraction](https://www.reddit.com/r/LocalLLaMA/comments/18qi4u0/best_models_out_there_for_improving_article/) similarly describes practical use without a comparable evaluation. I grounded the recommendations in the papers above rather than treating either thread as proof.

The [Whisper discussion on word-timestamp behavior](https://github.com/openai/whisper/discussions/2535) documents reproducibility questions when changing timestamp options. [WhisperX](https://arxiv.org/abs/2303.00747) supplies a researched forced-alignment approach. Neither establishes that more acoustically accurate timestamps would score better against reference timestamps produced by a different recognizer. The local project already found that distinction important.

**The next major experiment should combine three changes that have not been established by the existing pilots.**

First, teach evidence selection from enough genuinely different contexts. The completed small LoRA pilots in the notes used roughly 130–137 positive examples per training fold. Sampling 20 completions for each still supplies roughly the same number of independent situations. A rank-8 adapter’s failure on that setting does not refute supervised evidence learning, and increasing the adapter rank alone does not solve the data problem.

Second, teach the distinction the evaluator actually rewards. A passage can be true, topical, and sufficient while having zero overlap with the annotated occurrence. Include two kinds of labels: whether a passage supports the proposition, and whether it matches the annotation’s selected occurrence and extent. Do not label another valid mention “medically false” just because its temporal overlap is zero. Train the second task explicitly.

Third, make exact locations native to the output. A neural start/end head scores positions in the transcript directly. This differs from asking an instruction model to print word numbers: that earlier experiment mixed evidence reasoning with arbitrary-number generation and parsing. An encoder’s position scores require no numerical generation. For a generative comparison, output an occurrence anchor plus an exact quote; a quote alone cannot distinguish identical repeated text.

The most useful architecture comparison is:

| Candidate | Input and output | Reason to test | Main risk |
| --- | --- | --- | --- |
| **Contextual evidence extractor** | Question plus full transcript; start/end probabilities over original words, optionally an evidence-token mask | Every selected span is located unambiguously; can learn from all token positions in a pass | Only 39 authentic consultations; transferred QA models often learn shorter entity answers rather than evidence clauses |
| **Evidence-only generative specialist** | Same question/context; occurrence anchor plus copied passage | Most directly comparable to the working evidence stage; supports supervised then preference training | Copy errors, length preferences, and repeated-text ambiguity unless anchors are validated |
| **Current 27B evidence system** | Existing per-question evidence stage | Strong unchanged control | Already documented occurrence and boundary errors |

A reasonable encoder starting point is ModernBERT, whose official documentation supports up to **8,192 tokens**. That context capacity is useful, but does not imply it will beat a QA-pretrained alternative. Use a QA-adapted model or intermediate evidence supervision before expecting useful start/end predictions. [Official architecture documentation](https://huggingface.co/docs/transformers/main/model_doc/modernbert). CoQA is relevant intermediate supervision because it includes evidence rationales alongside answers, including conversational phenomena; its answer F1 should not be mistaken for evidence-span accuracy. [CoQA paper](https://aclanthology.org/Q19-1016/).

For the generative candidate, compare an evidence-only adapter on a locally usable instruction model with a small encoder-decoder such as Flan-T5. Keep transcript length within the selected checkpoint’s supported configuration; use overlapping context windows if necessary, and choose among windows globally. Model choice should be decided by the matched experiment, not parameter count.

The training-data proposal is specific to this project:

1. **Split by consultation before creating anything.** Group near-duplicate scenarios as well where feasible. All questions, examples, perturbations, and synthetic derivatives of a source remain in its training fold. Existing public data have already been inspected, so cross-validation remains exploratory rather than pristine testing.
2. **Build clean reference targets.** Map annotated intervals to original base-ASR word indices. Record mapping quality. Preserve original word order, punctuation, times, and occurrence IDs. Manually review suspicious targets rather than silently creating misleading training labels.
3. **Start with a quality pilot of new consultations.** A proposed first batch is about 100–200 novel conversations, covering short confirmations, pronouns, joined facts, repeated plans, corrections, drug names, units, and negative findings. Construct intended evidence before the surrounding dialogue so its exact offsets are known. These counts are an experiment design, not a literature-established requirement.
4. **Filter by support and exact recovery.** Verify factual consistency, copied-text validity, question/answer agreement, and reference-location recovery. For intentionally repeated valid facts, review the intended annotation choice; semantic consistency alone cannot certify the occurrence. Inspect a sample manually before scaling.
5. **Scale only if the pilot is clean.** An initial larger target is 1,000–2,000 distinct short consultations and roughly 5,000–10,000 positive evidence pairs, plus hard negatives. Vary discourse structure, not just names and doses. A small matched subset can be rendered as speech and re-transcribed to learn realistic ASR corruption; generating audio for the entire corpus is not the first dependency.
6. **Create hard competing spans.** Use actual control mistakes, the other true occurrence, overlong/shortened passages, and facts belonging to another question. Attach temporal-IoU preferences on real audio. Synthetic text without timestamps supplies token-boundary supervision; do not pretend token overlap equals time overlap.
7. **Train the exact-span task first.** Then compare plain supervised learning with preference/ranking learning. Balance by consultation/question so hundreds of variants from one example do not dominate. On real annotations, test a soft temporal-overlap objective in addition to exact endpoints; this is our proposed adaptation, not a published result on this challenge.
8. **Evaluate both models against the frozen control.** Keep the answer decisions fixed initially. Report raw score, mean tIoU, zero-overlap count, IoU above 0.9, performance by discourse pattern, and latency. Only then evaluate answer rescue and other interactions end to end.

A second, smaller experiment can apply MIPROv2 to the evidence stage using exact temporal IoU as its score, with conversation-level inner and outer splits. Allow it to optimize instructions and demonstrations; give it a fixed search budget. Preserve the existing alignment and fallback behavior during that comparison. The previously failed retrieved-demo and prompt variants make this less compelling than the data-and-training program, but an optimized objective is a materially different mechanism. Do not spend another large block on uncapped manual prompt variations.

**Some conclusions in the existing notes should be narrowed.**

“Every route to better spans is exhausted” and “selection is a closed question” go beyond the evidence. Several inference-time selectors failed. Linear and shallow nonlinear feature rankers produced small local gains on an older build. Small generative extraction pilots also failed. I did not find a completed evaluation of a pretrained contextual neural span model trained with substantial task-matched data and hard occurrence/boundary negatives. That is the remaining hypothesis worth testing, not a guarantee.

The notes also infer that a hidden-set shortfall in the number of predicted yes answers explains the entire local/platform score gap. With 95 gold positives and 92 predicted positives, all we know is **false negatives minus false positives = 3**. There could be 3 misses and no false positives, or 6 misses and 3 false positives, and localization can also change. The count supports testing rescue; it cannot prove evidence generalization is solved.

A global five-yes-per-conversation rule would also be unjustified. Balance is guaranteed over the complete set, not necessarily within every request. Keep any decision-threshold argument tied to the actual scoring rule and calibrated probabilities. The existing 0.24 threshold follows a utility calculation under an assumed expected span quality; it is not a universally valid confidence cutoff.

**A substantial-gain gate should decide whether this line of work deserves more compute.**

For planning, define substantial as at least **+0.03 raw** on a matched exploratory comparison, supported across conversation folds, followed by an actual platform improvement. With answers fixed, +0.03 raw means **+0.05 mean tIoU**. From the saved control, that is roughly 0.716 → 0.766. A 0.90 target needs approximately 0.833 tIoU at perfect accuracy, closing about 41% of its current evidence-loss gap. These are targets and arithmetic, not forecasts.

Use an inner split for hyperparameters and early stopping, then report pooled outer-fold predictions once. Bootstrap by consultation rather than by question. Keep the original official labels and denominator in headline results. For a candidate that clears the evidence gate, run fresh audio-to-response evaluation and verify request latency before any platform validation. Model and ASR revision, prompt, decoding, timing policy, and rescue settings must match the tested build. Avoid concurrent experiments during scored requests; this project already measured the consequences.

Within the logged September 20 deadline, the practical first commitment is the data-quality pilot and one matched supervised comparison. A full multimodal retraining effort is a longer program. Training time and cost must be measured on the available hardware; neither the papers nor the current logs justify promising completion in a particular number of GPU hours.

Research and saved-output analysis are complete. No recommended model has been trained or validated in this work. The deliverable is a source-backed experiment direction, a reproducible diagnosis, and an implementation brief that can be used with the existing Claude Code workflow.
