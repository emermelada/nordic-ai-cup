"""Prepare and fully tokenize external rationale supervision with a quarantine audit."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import zipfile

from .data import digest, write_json
from .external import (OverlapIndex, coqa_documents, context_record, mashqa_documents, normalize,
                       simord_documents, source_hash, validate_external, word_span)
from .features import make_features


def sentence_tokenizer(raw, output):
    import nltk
    directory = output / 'nltk_data'
    with zipfile.ZipFile(raw / 'punkt_tab.zip') as archive:
        for name in archive.namelist():
            if name.startswith('punkt_tab/english/') and not name.endswith('/'):
                path = directory / 'tokenizers/punkt_tab/english' / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
    nltk.data.path.insert(0, str(directory.resolve()))
    return nltk.sent_tokenize


def prepare(raw, target, output, tokenizer, model_revision, max_question_tokens=128,
            medical_window_cap=45000, max_evidence_words=48):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((raw / 'download-manifest.json').read_text())
    for row in manifest:
        if digest(raw / row['file']) != row['sha256']:
            raise ValueError(f'Source hash changed: {row["file"]}')
    documents = list(coqa_documents(raw, 'dev')) + list(coqa_documents(raw, 'train'))
    for split in ('dev', 'train', 'test'):
        documents.extend(mashqa_documents(raw, split))
    sent_tokenize = sentence_tokenizer(raw, output)
    for split in ('dev', 'train'):
        for document in simord_documents(raw, split, sent_tokenize):
            if split == 'dev':
                document['split'] = 'diagnostic'  # SIMORD dev is now the published test1 set.
            documents.append(document)
    protected = {r['conversation']: r['context'] for r in json.loads(target.read_text())['records']}
    target_index = OverlapIndex(protected)
    target_exact = {normalize(text) for text in protected.values()}
    heldout = [d for d in documents if d['split'] != 'train']
    heldout_groups = {d['group'] for d in heldout}
    heldout_exact = {normalize(d['text']) for d in heldout}
    heldout_index = OverlapIndex({d['id']: d['text'] for d in heldout})
    rejected, contexts, examples, signatures = [], {}, [], set()
    duplicate_examples = set()
    window_counts, positive_windows = Counter(), Counter()
    tokens_by_split, lengths = Counter(), []

    def reject(document, reason, example=None, **extra):
        rejected.append({'document': document['id'], 'example': example['id'] if example else None,
                         'source': document['source'], 'split': document['split'], 'reason': reason, **extra})

    # Held-out contexts take precedence if a source accidentally duplicates a passage.
    documents.sort(key=lambda d: (d['split'] == 'train', d['source'], source_hash(d['id'])))
    for count, document in enumerate(documents):
        if count % 250 == 0:
            print(json.dumps({'documents_processed': count, 'examples_kept': len(examples)}), flush=True)
        text, split = document['text'], document['split']
        if split == 'test':
            continue  # only source identity/context participates in exclusion, never its labels
        if normalize(text) in target_exact or (matches := target_index.matches(text, near=True)):
            reject(document, 'competition_context_overlap', matches=matches if normalize(text) not in target_exact else ['exact'])
            continue
        if split == 'train':
            if document['group'] in heldout_groups or normalize(text) in heldout_exact:
                reject(document, 'heldout_source_or_exact_context_overlap')
                continue
            if matches := heldout_index.matches(text):
                reject(document, 'heldout_near_context_overlap', matches=matches)
                continue
        signature = source_hash(text)
        if signature in signatures:
            reject(document, 'duplicate_context')
            continue
        signatures.add(signature)
        context = {**context_record(text), **{k: document[k] for k in
                   ('group', 'source', 'domain', 'license', 'source_document')}, 'source_context_sha256': signature}
        for item in document['examples']:
            try:
                if item.get('reject_reason'):
                    raise ValueError(item['reject_reason'])
                if document['source'] == 'coqa' and len(item['current_question'].split()) < 3:
                    raise ValueError('underspecified_short_question')
                backend = tokenizer.backend_tokenizer
                backend.no_truncation()
                backend.no_padding()
                if len(backend.encode(item['question'], add_special_tokens=False).ids) > max_question_tokens:
                    raise ValueError('question_history_over_token_budget')
                gold_words, expanded = None, False
                if item['label']:
                    gold_words, expanded = word_span(text, *item['raw_span'], expected=item['span_text'])
                    size = gold_words[1] - gold_words[0] + 1
                    if document['source'] == 'simord' and size > 64:
                        raise ValueError('medical_dialogue_evidence_too_broad')
                    if not 3 <= size <= max_evidence_words:
                        raise ValueError(f'evidence_word_length_outside_3_to_{max_evidence_words}')
                row = {k: item[k] for k in ('id', 'question', 'label', 'answer_type', 'annotation')}
                row.update(conversation=document['id'], split=split, source=document['source'],
                           gold_words=gold_words, raw_span=item['raw_span'],
                           word_boundary_expanded=expanded)
                if 'provenance_sentence_ids' in item:
                    row['provenance_sentence_ids'] = item['provenance_sentence_ids']
                key = (signature, normalize(item['question']))
                if key in duplicate_examples:
                    raise ValueError('duplicate_context_question')
                features = make_features([{**context, **row}], tokenizer)
                if split == 'train' and document['source'] == 'mashqa' and (
                        window_counts['train:mashqa'] + len(features) > medical_window_cap):
                    raise ValueError('medical_article_window_budget')
            except ValueError as error:
                reject(document, str(error), example=item)
                continue
            duplicate_examples.add(key)
            contexts[document['id']] = context
            examples.append(row)
            window_counts[split + ':' + document['source']] += len(features)
            positive_windows[split] += sum(f['start_position'] != f['cls'] for f in features)
            tokens_by_split[split] += sum(len(f['inputs']['input_ids']) for f in features)
            if row['label']:
                lengths.append(gold_words[1] - gold_words[0] + 1)
    bundle = {'schema_version': 'external-rationales-v1', 'metric': 'word_span_iou_not_temporal',
              'excluded_target_sha256': digest(target), 'contexts': contexts, 'examples': examples,
              'window_counts': dict(window_counts),
              'sources': manifest, 'tokenizer': {'model': 'deepset/deberta-v3-large-squad2',
                  'revision': model_revision, 'max_length': 512, 'stride': 192,
                  'max_question_tokens': max_question_tokens},
              'filters': {'min_evidence_words': 3, 'max_evidence_words': max_evidence_words,
                  'medical_window_cap': medical_window_cap,
                  'rationale': 'Competition gold evidence is 1-33 words (median 8); longer external '
                               'rationales teach a span length the target task never rewards.'},
              'limitations': ['Lexical overlap checks do not prove absence of semantic overlap.',
                  'CoQA is general reading comprehension with two preceding QA turns, not a clinical transcript.',
                  'MASH-QA is medical web reading comprehension, not a patient-specific clinical transcript.',
                  'SIMORD questions are templates over clinician orders; evidence boundaries are sentence-level.',
                  'MASH-QA and SIMORD rationales are far longer than competition evidence; the length filter '
                  'keeps only their shortest answers, which is a biased subset of those corpora.',
                  'No synthetic audio, time labels, or paid teacher annotations are included.']}
    validate_external(bundle, target)
    pretrain_windows = window_counts['train:coqa'] + window_counts['train:mashqa']
    seconds_per_update, batch = .46498765, 16
    # Include a range because the earlier microbenchmark was short and used another length distribution.
    estimates = {str(epochs): {'window_passes': epochs * pretrain_windows,
                   'minutes_at_pilot_throughput': round(epochs * pretrain_windows / batch * seconds_per_update / 60, 1)}
                 for epochs in (1, 2, 3)}
    summary = {'created_at': datetime.now(timezone.utc).isoformat(),
               'contexts': len(contexts), 'examples': len(examples),
               'examples_by_split_source': dict(Counter(r['split'] + ':' + r['source'] for r in examples)),
               'contexts_by_split_source': dict(Counter(split + ':' + contexts[cid]['source'] for split, cid in
                   {(r['split'], r['conversation']) for r in examples})),
               'answers_by_split_type': dict(Counter(r['split'] + ':' + r['answer_type'] for r in examples)),
               'training_windows_by_split_source': dict(window_counts),
               'positive_windows_by_split': dict(positive_windows), 'tokens_by_split': dict(tokens_by_split),
               'boundary_expansions': sum(r['word_boundary_expanded'] for r in examples),
               'evidence_words': {'max_allowed': max_evidence_words,
                   'mean': round(sum(lengths) / len(lengths), 2),
                   'median': sorted(lengths)[len(lengths) // 2]},
               'max_positive_words': max(lengths), 'rejected_items': len(rejected),
               'rejection_counts': dict(Counter(r['reason'] for r in rejected)),
               'pretrain_timing_estimate': estimates,
               'timing_caveat': 'Not a new GPU measurement. Excludes evaluation, setup and checkpoint I/O; benchmark this mixture first.',
               'versions': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'nltk', 'TextGrid')},
               'gpu_started': False, 'paid_annotation_calls': 0}
    write_json(output / 'external.json', bundle)
    write_json(output / 'summary.json', summary)
    write_json(output / 'quarantine.json', rejected)
    tokenizer.save_pretrained(output / 'tokenizer')
    print(json.dumps(summary, indent=2), flush=True)
    return bundle, summary


def main():
    from transformers import AutoConfig, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--medical-window-cap', type=int, default=45000)
    parser.add_argument('--max-evidence-words', type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh preparation directory')
    model = 'deepset/deberta-v3-large-squad2'
    revision = AutoConfig.from_pretrained(model, trust_remote_code=False)._commit_hash
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if args.medical_window_cap <= 0 or args.max_evidence_words < 3:
        parser.error('Window cap must be positive and the evidence cap at least three words')
    prepare(args.raw, args.target, args.output, tokenizer, revision,
            medical_window_cap=args.medical_window_cap, max_evidence_words=args.max_evidence_words)


if __name__ == '__main__':
    main()
