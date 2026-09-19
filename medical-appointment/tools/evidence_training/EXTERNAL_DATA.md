# External rationale data

Supervision for the span extractor from three public corpora with human evidence
annotations. It is **text only**: no audio, no fabricated timestamps, no generated
labels, no paid annotation calls. The competition's boolean answers and ASR
timestamps are untouched, and the 39 competition conversations are excluded.

Prepared bundle: `runs/evidence-external-20260919/prepared-v3/external.json`
(`sha256 f6cdff0e…`, 219 MB), summary and quarantine beside it.

## Sources and terms

| Source | What it gives | Licence |
| --- | --- | --- |
| [CoQA](https://stanfordnlp.github.io/coqa/) (Wikipedia and Gutenberg only) | Conversational questions with human-highlighted rationales | CC BY-SA 4.0 |
| [MASH-QA](https://github.com/mingzhu0527/MASHQA) | Medical web reading comprehension over WebMD articles | Repository code Apache-2.0; the data's own terms are not separately stated. WebMD URLs are retained per document. Not Apache-licensed content. |
| [SIMORD](https://huggingface.co/datasets/microsoft/SIMORD) over ACI-Bench and PriMock57 transcripts | Clinician-annotated orders with sentence provenance | Annotations CDLA-Permissive-2.0; transcripts CC BY 4.0 |

RACE and MCTest passages are excluded from CoQA so one licence covers the export.
emrQA and RadQA need access agreements and were not downloaded. PriMock57 and
ACI-Bench enter only as SIMORD's source transcripts; they are not counted twice.

## Contents

47,603 checked question/evidence pairs over 6,469 contexts.

| Split | Source | Questions | Contexts | 512-token windows |
| --- | --- | ---: | ---: | ---: |
| train | CoQA | 35,699 | 3,041 | 36,141 |
| train | MASH-QA | 8,113 | 2,819 | 30,138 |
| train | SIMORD | 16 | 16 | 79 |
| dev | CoQA | 2,599 | 200 | 2,644 |
| dev | MASH-QA (`val`) | 1,129 | 359 | 5,268 |
| diagnostic | SIMORD (`test1`) | 47 | 34 | 248 |

Training answer types: 27,012 spans, 4,625 yes, 3,480 no, 582 unknown,
8,113 medical article answers, 16 medical orders. A CoQA **no** keeps its
rationale and stays a positive example; only **unknown** is unanswerable.

Pretraining consumes CoQA + MASH-QA: **43,812 questions, 66,279 windows**.
SIMORD's 16 training questions are the only doctor–patient dialogue supervision,
so they are spent adapting each competition fold instead of pretraining.
SIMORD `test1` is diagnostic: it never trains and never selects a checkpoint.

## Filters

Competition gold evidence is 1–33 words (mean 9.1, median 8). MASH-QA answers are
seven times longer (median 55), so a global **3–48 word** evidence filter keeps the
external spans within reach of the target length. After filtering, external
evidence is a mean of 14.7 and a median of 11 words.

26,603 items were quarantined with reasons recorded in `quarantine.json`:

| Reason | Count |
| --- | ---: |
| Evidence outside 3–48 words | 20,914 (13,296 MASH-QA, 7,608 CoQA, 10 SIMORD) |
| CoQA question under three words | 5,126 |
| SIMORD noncontiguous or invalid provenance | 199 |
| SIMORD evidence over 64 words | 126 |
| Duplicate context | 178 |
| Overlap with an external held-out source or context | 57 |
| Question history over the 128-token budget | 2 |
| Duplicate context and question | 1 |

**No competition-context overlap was found** against all 39 conversations, using
exact normalized text and 5-/8-word shingle matching. Every accepted example is
tokenized during preparation and its gold span must be reachable inside a real
window, so training cannot silently drop a target. 18,333 rationales were widened
to whole-word boundaries, which is recorded per example.

## Preparation

Dependencies stay out of the application environment (`nltk`, `TextGrid` in
`requirements-data.txt`, installed into the experiment directory):

```bash
PYTHONPATH=runs/evidence-external-20260919/prep-deps \
HF_HOME=runs/evidence-external-20260919/hf-cache \
TOKENIZERS_PARALLELISM=false .venv311/bin/python -m tools.evidence_training.prepare_external \
  --raw runs/evidence-external-20260919/raw \
  --target runs/evidence-training-20260919/dataset.json \
  --output runs/evidence-external-20260919/prepared-v3 \
  --medical-window-cap 45000 --max-evidence-words 48
```

Raw downloads (95 files with URLs and SHA-256 in `raw/download-manifest.json`) come
from `fetch_external.py`; preparation verifies every hash and needs no network.
Always choose a fresh output directory. Preparation takes about two minutes and
uses no GPU.

## Measured cost

RTX PRO 6000 Blackwell Max-Q, BF16 autocast with FP32 weights, effective batch 16,
measured on this mixture (`runs/bench-mix-b*/benchmark.json` on the GPU host):

| Micro-batch × accumulation | Seconds per update | Peak CUDA | Minutes per epoch |
| --- | ---: | ---: | ---: |
| 4 × 4 | 0.421 | 13.9 GiB | 29.0 |
| **8 × 2** | **0.351** | 20.6 GiB | **24.3** |
| 16 × 1 | 0.370 | 32.3 GiB | 25.5 |

Two epochs at 8 × 2 is about 49 minutes of optimizer updates; evaluation,
feature building and checkpoint writes add roughly ten minutes. At $1.52/hour
that is near one dollar of GPU time.

## Limitations

- Lexical overlap checks do not prove the absence of semantic overlap.
- CoQA is general reading comprehension with two preceding turns of history, not
  clinical dialogue. MASH-QA is medical prose, not a patient conversation.
- The length filter keeps only the shortest answers of MASH-QA and SIMORD, which
  is a biased subset of those corpora.
- SIMORD provenance is sentence-level, so its boundaries are coarser than the
  competition's minimal spans, and 16 training questions cannot carry a domain.
- External development IoU is word-span IoU. It is neither temporal IoU nor the
  competition score, and improvement there does not imply improvement here.
