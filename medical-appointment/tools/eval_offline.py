"""Score cached ASR with fresh local Qwen, a raw-output replay, or retrieval only."""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_evaluator import Statistics
from pipeline.core import (
    answer_response, retrieval_response, sanitize_response, split_units,
    words_from_transcript,
)
from pipeline.evidence import energy_envelope, refine_evidence
from pipeline.mlx_backend import DEFAULT_PROMPT, QWEN_MODEL, MLXBackend, decode_audio
from tools.transcribe_all import (
    CACHE_PRODUCER, atomic_write_json, atomic_write_text, positive_limit,
    validate_transcript,
)
from utils import (
    audio_duration_seconds, evidence_interval, gold_evidence,
    group_questions_by_conversation, load_sample_audio, temporal_iou,
    validate_response,
)


def split_conversations(conversations, subset):
    if subset not in ('all', 'dev', 'holdout'):
        raise ValueError(f'Unknown dataset split: {subset}')
    ids = [rows[0]['transcript_id'] for _, rows in conversations]
    holdout = set(sorted(ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:9])
    return [(filename, rows) for filename, rows in conversations
            if subset == 'all' or (rows[0]['transcript_id'] in holdout) == (subset == 'holdout')]


def _finite_seconds(value):
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0):
        return float(value)
    return None


def _word_key(words):
    return [(word['word'], word['start'], word['end']) for word in words]


def prepare_inputs(conversations, transcripts: Path | None, replay: dict | None):
    """Check the entire selected denominator before loading a model or scoring."""
    prepared = []
    for filename, rows in conversations:
        sample_id = rows[0]['transcript_id']
        entry = None
        replay_units = None
        if replay is not None:
            if sample_id not in replay:
                raise FileNotFoundError(f'Missing replay conversation: {sample_id}')
            entry = replay[sample_id]
            if not isinstance(entry, dict) or not isinstance(entry.get('raw'), str):
                raise ValueError(f'{sample_id}: replay must contain a raw string')
            dumped_questions = entry.get('questions')
            if (not isinstance(dumped_questions, list) or len(dumped_questions) != len(rows)
                    or any(not isinstance(question, dict)
                           or question.get('q') != row['question']
                           or question.get('id') != row['question_id']
                           for question, row in zip(dumped_questions, rows))):
                raise ValueError(f'{sample_id}: replay question text/order differs from official CSV')
            if not isinstance(entry.get('units'), list):
                raise ValueError(f'{sample_id}: replay units are missing')
            replay_units = []
            for unit in entry['units']:
                if not isinstance(unit, dict) or not isinstance(unit.get('words'), list):
                    raise ValueError(f'{sample_id}: invalid replay unit')
                unit_words = words_from_transcript({'segments': [unit]})
                if not unit_words or len(unit_words) != len(unit['words']):
                    raise ValueError(f'{sample_id}: invalid replay words')
                replay_units.append(unit_words)
        transcript = None
        if transcripts is not None:
            source = transcripts / f'{Path(filename).stem}.json'
            if not source.is_file():
                raise FileNotFoundError(f'Missing transcript conversation: {source}')
            transcript = json.loads(source.read_text())
            validate_transcript(transcript)
            metadata = transcript.get('_cache', {})
            if isinstance(metadata, dict) and metadata.get('producer') == CACHE_PRODUCER:
                digest = hashlib.sha256(load_sample_audio(filename)).hexdigest()
                if metadata.get('complete') is not True or metadata.get('source_sha256') != digest:
                    raise ValueError(f'{source}: incomplete cache or source audio hash mismatch')
            words = words_from_transcript(transcript)
        elif replay_units is not None:
            words = [word for unit in replay_units for word in unit]
        else:
            raise ValueError('Transcripts are required without an LLM replay')
        units = split_units(words)
        if replay_units is not None:
            if [_word_key(unit) for unit in units] != [_word_key(unit) for unit in replay_units]:
                raise ValueError(f'{sample_id}: transcript/splitter differs from replay unit table')
        duration = (transcript or entry or {}).get('duration')
        duration_source = 'decoded_transcript' if transcript else 'replay_dump'
        if duration is None:
            duration = audio_duration_seconds(load_sample_audio(filename))
            duration_source = 'mp3_header_estimate_legacy_cache'
        if _finite_seconds(duration) is None or duration <= 0:
            raise ValueError(f'{sample_id}: cannot determine a positive audio duration')
        prepared.append({
            'id': sample_id, 'filename': filename, 'rows': rows, 'words': words,
            'units': units, 'duration': float(duration), 'duration_source': duration_source,
            'envelope': energy_envelope(decode_audio(load_sample_audio(filename))),
            'entry': entry,
            'cached_asr_seconds': _finite_seconds(transcript.get('seconds')) if transcript else None,
        })
    return prepared


def _timing_summary(values):
    values = [value for value in values if value is not None]
    return {
        'count': len(values), 'total': sum(values),
        'mean': sum(values) / len(values) if values else None,
        'max': max(values) if values else None,
    }


def evaluate(prepared, *, replay: bool, retrieval_only: bool, start_offset: float,
             output: Path, subset: str = 'all', prompt: str | None = None,
             alignment: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / name for name in ('questions.csv', 'summary.json', 'raw_outputs.json')]
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Refusing to overwrite an existing output: {destination}')
    mode = 'retrieval_only' if retrieval_only else ('replay' if replay else 'fresh_llm')
    backend = (MLXBackend(prompt=prompt) if prompt else MLXBackend()) if mode == 'fresh_llm' else None
    statistics = Statistics()
    question_results = []
    raw_outputs = {}
    timings = []
    right_passage_tious = []
    zero_overlap_present = 0
    started_run = time.monotonic()
    for index, item in enumerate(prepared, 1):
        questions = [row['question'] for row in item['rows']]
        words, duration = item['words'], item['duration']
        started = time.monotonic()
        raw = ''
        fresh_seconds = call_seconds = replay_seconds = None
        if backend is not None:
            started_call = time.monotonic()
            raw = backend.complete(words, questions)
            call_seconds = time.monotonic() - started_call
            fresh_seconds = backend.last_generation_seconds
        elif replay and not retrieval_only:
            raw = item['entry']['raw']
            replay_seconds = _finite_seconds(item['entry'].get('seconds'))
        quote_alignment = alignment or (
            item['entry'].get('alignment', 'legacy') if replay else 'numeric'
        )
        prompt_name = (prompt or DEFAULT_PROMPT) if backend else (item['entry'] or {}).get('prompt', 'legacy')
        started_cpu = time.process_time()
        fallback = retrieval_response(words, questions, duration)
        if retrieval_only:
            response = fallback
        else:
            response = answer_response(
                raw, words, questions, duration, fallback=fallback,
                start_offset=start_offset, alignment=quote_alignment,
            )
        response = refine_evidence(response, words, item.get('envelope'))
        response = sanitize_response(response, len(questions), duration)
        validate_response(response, len(questions))
        postprocess_cpu_seconds = time.process_time() - started_cpu
        processing_seconds = time.monotonic() - started
        # Offline processing is not HTTP latency; do not populate Statistics.latencies_ms.
        statistics.record_request(len(questions), None, failed=False)
        for position, row in enumerate(item['rows']):
            prediction = int(response.answers[position])
            predicted = evidence_interval(
                response.evidence_start[position], response.evidence_end[position]
            )
            gold = gold_evidence(row)
            label = int(row['label'])
            iou = temporal_iou(gold, predicted) if label == 1 and gold is not None else None
            statistics.record(row['question_type'], label, prediction, gold, predicted)
            if iou is not None:
                if iou > 0:
                    right_passage_tious.append(iou)
                elif predicted is not None:
                    zero_overlap_present += 1
            question_results.append({
                'transcript_id': item['id'], 'question_id': row['question_id'],
                'question': row['question'], 'question_type': row['question_type'],
                'label': label, 'prediction': prediction, 'correct': prediction == label,
                'gold_start': gold[0] if gold else None, 'gold_end': gold[1] if gold else None,
                'evidence_start': predicted[0] if predicted else None,
                'evidence_end': predicted[1] if predicted else None, 'temporal_iou': iou,
            })
        timing = {
            'transcript_id': item['id'], 'cached_asr_seconds': item['cached_asr_seconds'],
            'historical_generation_seconds': replay_seconds,
            'fresh_generation_seconds': fresh_seconds,
            'fresh_llm_call_seconds_including_load': call_seconds,
            'postprocess_cpu_seconds': postprocess_cpu_seconds,
            'offline_processing_wall_seconds': processing_seconds,
        }
        timings.append(timing)
        if not retrieval_only:
            raw_outputs[item['id']] = {
                'raw': raw, 'seconds': fresh_seconds if backend else replay_seconds,
                'prompt': prompt_name, 'alignment': quote_alignment,
                'model': QWEN_MODEL if backend else item['entry'].get('model'),
                'generation_source': 'fresh' if backend else 'historical_replay',
                'duration': duration, 'duration_source': item['duration_source'],
                'units': [{'words': unit} for unit in item['units']],
                'questions': [
                    {'id': row['question_id'], 'q': row['question'],
                     'type': row['question_type'], 'label': row['label'],
                     'gs': row['evidence_start'], 'ge': row['evidence_end']}
                    for row in item['rows']
                ],
                'timing': timing,
            }
        print(f'{index}/{len(prepared)} {item["id"]}: {processing_seconds:.3f}s offline processing', flush=True)
    summary = {
        'mode': mode, 'start_offset': start_offset, 'subset': subset,
        'prompt': (prompt or DEFAULT_PROMPT) if backend else sorted({entry['prompt'] for entry in raw_outputs.values()}),
        'alignment': sorted({entry['alignment'] for entry in raw_outputs.values()}),
        'conversations': statistics.conversations, 'questions': statistics.total,
        'correct': statistics.correct, 'failed_conversations': statistics.failed_conversations,
        'accuracy': statistics.accuracy, 'mean_tiou': statistics.mean_tiou,
        'score': statistics.final_score,
        'by_type': {
            kind: {'correct': correct, 'total': total, 'accuracy': correct / total}
            for kind, (correct, total) in statistics.by_type.items()
        },
        'diagnostics_not_scored': {
            'annotated_yes_count': len(statistics.tious),
            'right_passage_definition': 'strictly positive temporal overlap',
            'right_passage_count': len(right_passage_tious),
            'mean_tiou_right_passage': (sum(right_passage_tious) / len(right_passage_tious)
                                        if right_passage_tious else None),
            'zero_overlap_count': sum(iou == 0 for iou in statistics.tious),
            'zero_overlap_with_present_span': zero_overlap_present,
            'missing_spans': statistics.missing_spans,
            'mean_tiou_when_answered_yes': statistics.mean_tiou_answered_yes,
        },
        'timing_seconds': {
            name: _timing_summary([timing[name] for timing in timings])
            for name in (
                'cached_asr_seconds', 'historical_generation_seconds',
                'fresh_generation_seconds', 'fresh_llm_call_seconds_including_load',
                'postprocess_cpu_seconds', 'offline_processing_wall_seconds',
            )
        },
        'http_latency_seconds': None,
        'timing_note': 'Cached ASR and historical generation were not measured in this run. '
                       'Fresh generation excludes load/template time. Offline CPU/wall time '
                       'is not an HTTP or end-to-end audio request benchmark.',
        'run_wall_seconds': time.monotonic() - started_run,
        'per_conversation': timings,
    }
    csv_buffer = io.StringIO(newline='')
    if question_results:
        writer = csv.DictWriter(csv_buffer, fieldnames=list(question_results[0]))
        writer.writeheader()
        writer.writerows(question_results)
    atomic_write_text(destinations[0], csv_buffer.getvalue())
    atomic_write_json(destinations[2], raw_outputs)
    atomic_write_json(destinations[1], summary)
    print(statistics.report())
    diagnostics = summary['diagnostics_not_scored']
    print(f'\nDiagnostic only: {diagnostics["zero_overlap_count"]} zero-overlap gold-yes '
          f'cases, including {statistics.missing_spans} missing spans; '
          f'right-passage mean tIoU {diagnostics["mean_tiou_right_passage"]}.')
    print(summary['timing_note'])
    print(f'Results: {output}')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path)
    parser.add_argument('--replay-llm', type=Path)
    parser.add_argument('--limit', type=positive_limit)
    parser.add_argument('--subset', choices=['all', 'dev', 'holdout'], default='all')
    parser.add_argument('--alignment', choices=['legacy', 'numeric'],
                        help='Quote alignment; defaults to serving mode or recorded replay mode.')
    parser.add_argument('--prompt', choices=['legacy', 'focused', 'compact'],
                        help='Prompt for fresh generation; default is the serving prompt.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start-offset', type=float, default=0.0,
                        help='Shift aligned/cited LLM evidence starts in seconds.')
    parser.add_argument('--retrieval-only', action='store_true')
    args = parser.parse_args(argv)
    if args.transcripts is None and args.replay_llm is None:
        parser.error('--transcripts or --replay-llm is required')
    if not math.isfinite(args.start_offset):
        parser.error('--start-offset must be finite')
    if args.retrieval_only and args.start_offset != 0:
        parser.error('--start-offset applies to LLM evidence, not --retrieval-only')
    replay = None
    if args.replay_llm is not None:
        replay = json.loads(args.replay_llm.read_text())
        if not isinstance(replay, dict):
            parser.error('--replay-llm must contain a conversation-keyed JSON object')
    if args.prompt and (args.replay_llm or args.retrieval_only):
        parser.error('--prompt applies only to fresh generation')
    conversations = split_conversations(group_questions_by_conversation(), args.subset)[:args.limit]
    prepared = prepare_inputs(conversations, args.transcripts, replay)
    output = args.output or (
        Path(__file__).resolve().parents[1] / 'runs' / 'offline'
        / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    )
    evaluate(prepared, replay=replay is not None, retrieval_only=args.retrieval_only,
             start_offset=args.start_offset, output=output, subset=args.subset, prompt=args.prompt,
             alignment=args.alignment)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
