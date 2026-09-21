**Evidence learning experiment — implementation brief**

Build an offline experiment that tests whether supervised, question-conditioned evidence extraction can substantially improve the current Medical Appointment system. Read the accompanying [research report](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/research/evidence_research_20260919/RESEARCH.md) for the evidence and limitations. The design below is proposed work, not implemented serving behavior.

The main hypothesis is that current inference-time generation and verification do not learn the dataset’s preference among several semantically valid spans. We need supervision on exact occurrence and boundaries, plus enough diverse contexts for that supervision to generalize.

**Freeze a comparable control before training.**

The documented final platform score is 0.8167711209, while the locally available runtime-faithful evidence control scores 0.8297086456 on 390 public-training questions. Treat these as different evaluations. Use the latter as a reproducible offline control until a matching replay of the locked build is available. Preserve all inputs and output hashes.

Relevant existing files:

| File | Reuse |
| --- | --- |
| [inputs_base.json](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/tools/gpu_eval/inputs_base.json) | Public-training questions, annotations, base-ASR words and times |
| [saved control](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/runs/resume-20260919-0325/faithful-results/stageb-on-own-resume-control.json) | Fixed answer decisions and baseline evidence |
| [eval.py](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/tools/gpu_eval/eval.py) | Runtime-faithful scoring, cache fingerprints, conversation grouping |
| [stage_b.py](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/pipeline/stage_b.py) | Existing evidence-stage interfaces and sentence rendering |
| [locate.py](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/pipeline/locate.py) | Validated word-index-to-time conversion ideas; its LLM pointer experiment already lost |
| [evidence.py](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/pipeline/evidence.py) | Existing alignment/refinement policies for baseline parity |
| [positive_span_audit.csv](/Users/chinese/AICUP/Nordic-AI-Cup-2026/medical-appointment/research/evidence_research_20260919/positive_span_audit.csv) | Error inspection and training-fold hard-negative sources |

Keep the experiment in its own directory. Preserve current serving defaults. The project records a user-requested deployment/SSH pause; this research has not restarted the instance or changed that operational state.

**Produce three explicit datasets.**

1. Original public-training evidence records, with question, full transcript, stable word indices, gold times, mapped gold endpoints, mapping quality, conversation ID, and fold ID. Use IDs only for bookkeeping and splitting, never as predictive features.
2. Novel synthetic consultations with question/evidence pairs, exact character and word offsets, scenario ID, generation provenance, intended occurrence, and quality checks. Every derivative of a source stays in the same fold. Vary discourse patterns, not only entities. Begin with 100–200 conversations and inspect quality before scaling toward 1,000–2,000.
3. Preference/quality records derived exclusively from training-fold examples: question, context, preferred span, competing span, source of competitor, and separate semantic-support and annotation-overlap labels. Include actual model mistakes and difficult boundary alternatives. Weight examples so one source with many variants cannot dominate training.

Audit suspicious gold mappings before using them as training targets. Keep all official examples in evaluation. For synthetic text without generated audio, use exact token targets and token-overlap labels, clearly distinguished from measured temporal IoU.

For synthetic construction, define the intended evidence and question first, build a coherent dialogue around them, and recover the span with exact matching plus an independent extraction check. Preserve local dialogue blocks when varying position. Randomly shuffling individual turns can destroy coreference and confirmation meaning. Repeated valid statements need explicit annotation-preference review; an entailment checker cannot establish which one the competition would mark.

**Compare two trainable evidence mechanisms.**

The primary alternative is a pretrained bidirectional encoder with a question-conditioned start/end head. Render the transcript deterministically and retain a character-to-original-word map. Use subword offset mappings to supervise original word endpoints. Mask invalid positions, enforce start ≤ end, and return an existing contiguous word interval. Score the context globally; if windows are needed, include negative windows and compare their confidence on a common scale. An auxiliary token mask can teach the model which words belong to evidence.

For this encoder, use cross-entropy on endpoint positions as the first objective. Compare a second version that also learns relative quality among competing spans, using gold temporal overlap for real examples. Joint span scoring or an end head conditioned on the start can be an ablation if independent endpoints are inconsistent. Do not begin with many architectural changes at once.

The comparison model is an evidence-only generative specialist: supervised training first, followed by a separately measured preference-training phase. Its output must preserve occurrence identity through an anchor plus quote, or validated positions. Constrain copying where practical. A generative preference loss over identical quote strings cannot distinguish repeated occurrences, so the occurrence anchor is necessary for those pairs.

Train only the evidence task initially. Keep the known-good boolean decisions fixed. The earlier single-fold small LoRA result is a baseline to beat, not a justification for rerunning the same experiment with a different rank.

**Use this bounded experiment sequence.**

| Run | Question it answers |
| --- | --- |
| Control replay | Can the experiment reproduce existing answers, spans, and score exactly? |
| Encoder on real training-fold examples | Does position prediction help before synthetic data is added? |
| Encoder with matched synthetic data | Does context diversity improve unseen-consultation span selection? |
| Same encoder plus span-quality supervision | Does learning annotation preference improve over plain endpoint training? |
| Generative specialist with supervised training | Is a sequence model stronger on exactly the same data? |
| Same generative specialist plus preference training | Does preference learning add a measured gain beyond supervised training? |

Do not treat this as permission to launch all runs regardless of results. Stop expanding a branch if target construction is unreliable, or if the first completed comparison gives no plausible benefit. Use the same fold assignments and example budget for architecture comparisons.

Keep 3 outer conversation folds for tractability, with a disjoint inner development subset inside each training fold. Derive augmentation prompts, demonstrations, early stopping, and mixing weights using only the corresponding training/inner split. Include all scored questions from every outer fold in the final report. Public-training folds have been explored before; describe them honestly as exploratory out-of-fold evaluation.

**Measure gains at the resolution of the actual failure.**

Report combined raw score, accuracy, mean tIoU, zero-overlap positives, positives above 0.9 tIoU, each conversation’s change, and each fold’s change. Show errors for repeated mentions, short replies, pronoun-dependent statements, joined clauses, numbers/drug names, and under/over-extension. Distinguish manually assigned error categories from automatic metrics.

A proposed substantial-gain gate is +0.03 raw at fixed answers, equivalent to +0.05 tIoU. Require improvements to be reasonably distributed across conversations and report a paired conversation-bootstrap interval. This is an experiment-selection rule, not a promised effect. Then evaluate fresh ASR and actual end-to-end serving before any promotion.

Candidate inference should fit within the existing request budget and preserve a completed baseline response as fallback. A compact extractor might be fast, but measure its latency with the real transcript lengths and concurrent model memory usage. Do not infer end-to-end speed from parameter count.

A 0.90 score at 99% accuracy requires 0.84 mean tIoU. Correcting only the 21 zero-overlap examples in the saved control cannot reach it. The experiment must improve both where the evidence is and how much of it is selected.

**Optional lower-cost comparison:** run a capped MIPROv2 search over evidence instructions and demonstrations with exact temporal IoU as the optimization metric. Use separate inner and outer folds, preserve the same runtime interface, and score all selected prompts on the same control. This is secondary to the supervised evidence program, given the project’s extensive prior prompt experiments.

The final output of the experiment should be one report with immutable predictions, source/model revisions, dataset manifests, all fold metrics, latency, and a promotion recommendation. A negative result is useful if it rules out this specific data/objective/architecture combination. It should not be expanded into a claim that all supervised extraction or all evidence selection has been exhausted.
