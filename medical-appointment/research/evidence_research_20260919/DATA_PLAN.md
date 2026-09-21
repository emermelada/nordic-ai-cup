# External data and a small-data pilot

Research checked 2026-09-19. This is an experiment proposal; no new extractor has been trained and no external corpus has been imported by this addendum.

## Decision

The available 39 conversations, 390 questions, and 195 positive evidence spans can support an exploratory pilot. They cannot reliably establish whether a model will reach 0.90 on unseen competition data. Questions from one conversation are correlated: the independent context count is 39, not 390. A negative small-data result would not rule out a version trained with additional relevant data.

For a pilot using only the original examples, prioritize `deepset/deberta-v3-large-squad2`, which already has a trained extractive question-answering head. This refines the earlier ModernBERT recommendation for the specific small-data condition. ModernBERT-large remains a reasonable candidate once extra span supervision is available; its base checkpoint requires teaching a new extraction head. The DeBERTa checkpoint's documented training used 512-token windows, so the pilot needs overlapping windows, negative-window handling, and globally comparable selection scores. Neither checkpoint is already trained for this competition's yes/no evidence boundaries.

Model source: https://huggingface.co/deepset/deberta-v3-large-squad2

## External sources

| Source | Verified contents | Role and remaining work |
| --- | --- | --- |
| [SIMORD / MEDIQA-OE](https://huggingface.co/datasets/microsoft/SIMORD) | Clinician-annotated medical orders linked to supporting transcript material. The [organizer repository](https://github.com/jpcorb20/mediqa-oe) lists 63 training encounters / 143 orders and 100 development encounters / 255 orders. | Closest ready-made evidence attribution supervision found. Inspect actual provenance granularity, turn mapping, and contiguous-span suitability; convert orders into faithful questions. It does not supply our competition's exact audio boundaries. |
| [PriMock57](https://github.com/babylonhealth/primock57) | 57 simulated primary-care consultations, audio, manual utterance transcripts, and notes. | Strong domain match and useful for later ASR robustness checks. Construct question/evidence pairs and verify exact occurrence and boundaries. |
| [ACI-Bench](https://github.com/wyim/aci-bench) | 207 full consultations with notes: 67 train, 20 validation, three 40-example test sets. | Diverse conversational contexts; needs evidence-span labels. Some speaker tags are known to be swapped. Start from the training split. |
| [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | 1,701 short dialogue/summary pairs, including 1,201 training examples. | Additional contexts for supervised question/evidence construction. Short contexts may make localization artificially easy; preserve realistic surrounding dialogue. |

SIMORD reuses ACI-Bench and PriMock57: these counts must not be added as independent conversations. Deduplicate by source encounter, text, and close paraphrases before splitting. Retain the external corpora's own held-out splits for diagnostic evaluation when practical, and never use a split for both training and reported external evaluation.

ACI-Bench, PriMock57, and MTS-Dialog are published under CC BY 4.0. SIMORD annotations are published under CDLA-Permissive-2.0; retain source transcript attribution. The competition's root README explicitly permits collecting additional training data and using cloud APIs during development, while inference must be self-contained.

[RadQA](https://physionet.org/content/radqa/1.0.0/) has useful clinical QA spans, but access requires PhysioNet credentialing, training, and a data-use agreement. It is a secondary option if access already exists, not an assumed instant download.

## Minimal experiment

1. Reproduce the saved runtime-faithful control. Freeze the answer decisions and the base-ASR word/timestamp grid.
2. Establish three outer folds grouped by whole conversation. Keep all derivative examples in their source fold. Use only inner training/development examples for checkpoint selection or label-construction examples. These are exploratory folds: the public dataset has already been inspected and used for development.
3. Measure the QA-pretrained extractor without task fine-tuning, then with only training-fold competition examples. Include a tiny training-set overfit check to distinguish a broken label/alignment pipeline from a generalization problem.
4. Build a first batch of 200–500 checked external question/span pairs from multiple source conversations. Questions must be supported by the exact selected passage. Review cases with repeated statements, corrections, yes/no confirmation, and pronouns; semantic support alone cannot resolve annotation preference.
5. Train on these external pairs, then adapt to the original training-fold examples. Compare against the original-only run on the same held-out conversations. Expand toward 1,000–3,000 useful pairs only if this comparison gives a plausible gain.

Text-only data can teach passage selection without generating audio. Exact character/word offsets are required. The original competition audio remains necessary to measure temporal overlap and runtime behavior; a text-only overlap score is not measured tIoU.

Budget the initial small-data pilot at roughly 1–2 hours after trainer setup, including evaluation. This is a time box, not a hardware benchmark. The earlier 30–120-minute training estimate concerned thousands of examples and should not be mistaken for a required duration on 195 spans.

## Interpreting the result

- Improvement spread across folds/conversations is stronger evidence than one aggregate score driven by a handful of cases. Report paired conversation-bootstrap uncertainty, not question-level independence.
- A proposed substantial-gain gate is +0.03 combined score at fixed answers, equivalent to +0.05 mean tIoU. This is a selection threshold, not a predicted effect or a statistical significance threshold.
- Improvement confined to training conversations indicates overfitting or leakage. A near-flat original-only result remains inconclusive about an augmented-data model.
- If original-only and the checked augmented pilot both fail under correct alignment and a sound training setup, stop allocating the night to this branch.
- Platform validation remains necessary: these public conversations are reused development data. No score from this pilot establishes 0.90 on hidden evaluation.

The project already records a negative small generative LoRA experiment. This pilot tests a QA-pretrained position extractor and additional matched supervision; it should not be represented as the first fine-tuning attempt.
