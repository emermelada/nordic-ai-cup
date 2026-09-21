# Running and reproducing the evaluated solution

Evaluation score **0.8222490889515863** (single attempt, 19 hidden conversations, zero errors).
The same build validated **0.8307887829198248** four times identically.

Three things are needed to serve it: a 27B language model, a CPU speech recogniser, and two
trained span models of about 1.7 GB each. **The trained weights are not in this repository** —
they are 6.5 GB of binaries. Section 4 rebuilds them from scratch; section 5 checks the reported
numbers with no GPU and no weights at all.

---

## 1. What the pipeline is

One HTTP request carries one consultation's audio and its ten yes/no questions. The pipeline:

1. **Transcribes** the audio on the CPU, deliberately with a small model — see section 2.
2. **Answers** all ten questions in one 27B call over the numbered transcript.
3. **Re-asks** each `no` alone and flips it above a probability threshold (`rescue`).
4. **Selects an evidence span** for every `yes` from three independent producers, and returns the
   **medoid** — the span with the greatest total temporal overlap with the other two:
   - the 27B's own quote, re-read over a numbered sentence table (`stage B`)
   - a fine-tuned DeBERTa-v3-large **span extractor**
   - a fine-tuned DeBERTa-v3-large **listwise span ranker** over an enumerated candidate pool

The score is `0.4 x answer accuracy + 0.6 x mean temporal IoU`. Hidden-set accuracy is 1.0000, so
everything above the 0.4 is span localisation.

## 2. Exact model and hardware configuration

| Component | Exact setting | Why |
| --- | --- | --- |
| Language model | **`Qwen/Qwen3.8-27B`**, bf16, served by vLLM as `qwen` on `127.0.0.1:18000` | answers and stage B |
| vLLM arguments | `--max-model-len 12288 --max-num-seqs 4 --reasoning-parser qwen3 --gpu-memory-utilization 0.64 --dtype bfloat16` | 0.64 leaves room for the two span models |
| GPU | **RTX PRO 6000 Blackwell Max-Q, 96 GB** (Vast.ai). An A100 80 GB also works | the 27B is about 61 GiB in bf16 |
| Speech recognition | **faster-whisper `base`, `compute_type='int8'`, `device='cpu'`**, `language='en'`, `word_timestamps=True`, everything else default, path-based decode | not a quality choice — the gold spans *are* this model's word timestamps, so any better ASR caps temporal IoU at about 0.93 |
| Span extractor | `deepset/deberta-v3-large-squad2` fine-tuned, ~435M | second producer |
| Span ranker | same base, listwise head over the candidate pool | third producer |
| Answer prompt | `MEDICAL_ANSWER_PROMPT=named` | |
| Evidence stage | `MEDICAL_EVIDENCE_MODE=perq`, `MEDICAL_EVIDENCE_PROMPT=v3` | |
| Rescue | `MEDICAL_RESCUE_PHONETIC=1 MEDICAL_RESCUE_READING=turbo MEDICAL_RESCUE_THRESHOLD=0.24` | 0.24, not 0.5, because a missed positive costs 3.2x a false one |

Host setup from a bare Vast.ai instance, including driver requirements and the CUDA 12 libraries
CTranslate2 needs, is in [GPU_SETUP.md](GPU_SETUP.md).

## 3. Serving it

With both checkpoints on disk and vLLM already answering on 18000:

```bash
tools/serving/serve_ranker.sh 0.24 /path/to/rank-fit-001/model /path/to/fit-extractor-002/model
tools/serving/preflight.sh      # must print PREFLIGHT PASS before anything is submitted
```

`preflight.sh` checks what an HTTP 200 cannot: that **both** producers actually loaded, that a real
conversation's vote used both, and that the server has not silently restarted. This matters — a
deleted extractor checkpoint once left the endpoint answering normally while the vote quietly fell
back to stage B spans, worth about 0.02. `MEDICAL_REQUIRE_PRODUCERS=1`, which the serve script
sets, now makes that fail at startup instead.

Worst observed round trip is 19.6 s against the 60 s budget.

## 4. Rebuilding the two trained checkpoints

Both descend from one intermediate model. Total about 1h15m on one RTX PRO 6000. Exact arguments,
taken from each run's own `manifest.json`:

**Step 1 — external rationale data** (no GPU). Downloads and prepares CoQA and MASH-QA into
`external.json`, decontaminated against the competition data; counts, licences and filters are in
[tools/evidence_training/EXTERNAL_DATA.md](tools/evidence_training/EXTERNAL_DATA.md).

**Step 2 — the dataset export** (no GPU):

```bash
python -m tools.evidence_training.data --output dataset.json
```

390 questions, 39 conversations, 195 positive spans; mean attainable temporal IoU 0.985899.

**Step 3 — `pretrain-mix-001`**, extraction pretraining on 43,812 external pairs, ~57 min:

```bash
python -m tools.evidence_training.train --data dataset.json --external-data external.json \
  --output runs/pretrain-mix-001 --mode pretrain --epochs 2 --evals-per-epoch 2 --patience 3 \
  --batch-size 8 --accumulation 2 --eval-batch-size 32 --max-seconds 6000 --min-free-gb 60
```

**Step 4 — `fit-extractor-002`**, the deployed extractor, 16 seconds:

```bash
python -m tools.evidence_training.train --data dataset.json --output runs/fit-extractor-002 \
  --mode fit --model runs/pretrain-mix-001/model --device cuda --epochs 1 --batch-size 4 \
  --accumulation 4 --eval-batch-size 8 --evals-per-epoch 1 --patience 2 --learning-rate 1e-5 \
  --max-length 512 --stride 192 --seed 17 --max-seconds 1800 --min-free-gb 20
```

**Step 5 — `rank-pretrain-001`**, listwise ranking pretraining on 30,000 external questions, ~10 min:

```bash
python -m tools.evidence_training.ranker --data dataset.json --external-data external.json \
  --output runs/rank-pretrain-001 --mode pretrain --model runs/pretrain-mix-001/model \
  --device cuda --epochs 1 --evals-per-epoch 4 --batch-size 8 --accumulation 2 \
  --eval-batch-size 24 --limit 30000 --dev-limit 400 --max-candidates 192 \
  --max-seconds 2400 --min-free-gb 20
```

**Step 6 — `rank-fit-001`**, the deployed ranker, ~30 seconds:

```bash
python -m tools.evidence_training.ranker --data dataset.json --model runs/pretrain-mix-001/model \
  --ranker-state runs/rank-pretrain-001/model/ranker.pt --device cuda --mode fit --select final \
  --epochs 8 --evals-per-epoch 1 --seed 17 --batch-size 8 --accumulation 2 --eval-batch-size 24 \
  --max-candidates 192 --min-free-gb 20 --max-seconds 1200 --output runs/rank-fit-001
```

Candidate pool: sentence runs of one to four sentences plus comma-delimited clause spans, capped at
40 words. `--select final` is deliberate: the fold plan reserves only two conversations for
checkpoint selection, they saturate immediately, and selecting on them picked an untrained epoch 1
in two of three folds and faked a −0.0153 result.

**A rebuild is not bit-identical.** Rebuilding the extractor gave epoch loss 0.7100546 against the
original 0.7103700 from CUDA nondeterminism. Spans were unchanged in practice — a rebuilt extractor
reproduced 0.8307887829198248 exactly — but the validation is what establishes that, not the loss.

## 5. Checking the reported numbers without a GPU

The out-of-fold spans of all three producers are committed, so the central claim is verifiable
directly:

```bash
python -m tools.evidence_training.rules runs/rank-20260920/dump-s17.json
```

Expected: control 0.716181 mean temporal IoU, the medoid **0.750307 (+0.020476 raw)**, a
conversation bootstrap interval of [+0.007924, +0.033806], and `nested family choice ['medoid']` —
the rule family chosen leave-one-conversation-out over six candidates, on every one of the 39
splits.

```bash
python -m unittest discover -s tests -t .
```

313 tests, no GPU required. One of them re-derives the frozen control score from the dataset export
and asserts it to ten decimal places.

Every measurement, including the ones that failed, is in
[tools/evidence_training/RESULTS.md](tools/evidence_training/RESULTS.md). The negative results are
the more useful half: a better third voter lost on the platform, a more independent one lost out of
fold, metric-aware decoding lost, and four separate attempts at a span selector failed for the same
reason — the system cannot tell when it is already right.

## 6. What is not in this repository

| Absent | Size | How to get it |
| --- | --- | --- |
| `rank-fit-001`, `fit-extractor-002` | 1.7 GB each | section 4, or from the author's local copy |
| `pretrain-mix-001`, `rank-pretrain-001` | 1.7 GB each | section 4 steps 3 and 5 |
| `Qwen/Qwen3.8-27B` | ~61 GB | Hugging Face |
| `external.json` | 209 MB | section 4 step 1 |
| Consultation audio | — | the competition's own data |
