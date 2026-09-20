"""Bounded, standalone span-extractor training on CUDA, MPS, or CPU.

No hosted inference, serving changes, or paid dataset-generation calls are made.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from .data import digest, score, validate_bundle, validate_records, write_json
from .features import collate, decode_window, make_features, prediction_spans
from .external import hydrate, load_external


class BudgetExpired(RuntimeError):
    pass


class Runner:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.max_seconds
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=False)
        self.events = self.out / 'events.jsonl'
        self.device = args.device
        if self.device == 'auto':
            self.device = ('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        self.use_bf16 = self.device == 'cuda' and args.precision != 'fp32' and torch.cuda.is_bf16_supported()
        if args.precision == 'bf16' and not self.use_bf16:
            raise ValueError('This runner enables BF16 only on supported CUDA devices; use fp32')
        self.bundle = json.loads(args.data.read_text())
        validate_bundle(self.bundle)
        self.records = self.bundle['records']
        self.generated = []
        if args.extra_data:
            extra = json.loads(args.extra_data.read_text())['records']
            validate_records(extra + self.records)
            known = {r['conversation'] for r in self.records}
            unknown = sorted({r['conversation'] for r in extra} - known)
            if unknown:
                raise ValueError(f'Generated rows reference unknown conversations: {unknown}')
            self.generated = extra
        self.external = load_external(args.external_data, args.data) if args.external_data else None
        self.medical_records = ([r for r in hydrate(self.external, 'train') if r['source'] == 'simord']
                                if self.external else [])
        if max(r['gold_words'][1] - r['gold_words'][0] + 1 for r in self.records if r['label']) > args.max_span_words:
            raise ValueError('max-span-words would make some gold spans unreachable')
        self.metadata = {
            'status': 'initializing', 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'data_sha256': digest(args.data), 'source_manifest': self.bundle['sources'],
            'scope': self.bundle['scope'], 'device': self.device,
            'precision': 'bf16_autocast_float32_weights' if self.use_bf16 else 'fp32',
            'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy')},
            'answers_frozen': True, 'folds': self.bundle['folds'],
            'code_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
            'external_data_sha256': digest(args.external_data) if args.external_data else None,
            'external_medical_training_questions': len(self.medical_records),
            'extra_data_sha256': digest(args.extra_data) if args.extra_data else None,
            'generated_training_questions': len(self.generated),
        }
        if self.device == 'cuda':
            free, total = torch.cuda.mem_get_info()
            self.metadata['gpu'] = {'name': torch.cuda.get_device_name(), 'free_bytes': free, 'total_bytes': total}
            if free < args.min_free_gb * 1024**3:
                raise RuntimeError(f'Only {free / 1024**3:.1f} GiB GPU memory free. Need {args.min_free_gb} GiB; '
                                   'this runner never stops existing model servers automatically.')
        self.save_status('initializing')
        config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        self.revision = getattr(config, '_commit_hash', None) or args.revision
        self.metadata['resolved_model_revision'] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, revision=self.revision,
                                                      use_fast=True, trust_remote_code=False)
        if not self.tokenizer.is_fast or self.tokenizer.cls_token_id is None:
            raise ValueError('A fast tokenizer with CLS token is required')
        self.tokenizer.padding_side = 'right'
        self.check_budget()
        self.save_status('ready')

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired('Wall-time budget reached; no further training or evaluation will start')

    def emit(self, event, **values):
        row = {'event': event, 'elapsed_seconds': round(time.monotonic() - self.started, 3), **values}
        with self.events.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def save_status(self, status, **extra):
        self.metadata.update(status=status, elapsed_seconds=time.monotonic() - self.started, **extra)
        write_json(self.out / 'manifest.json', self.metadata)

    def autocast(self):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_bf16 else nullcontext()

    def sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()

    def new_model(self, path=None, seed=None):
        self.check_budget()
        seed = self.args.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        source = path or self.args.model
        model = AutoModelForQuestionAnswering.from_pretrained(
            source, revision=None if path else self.revision, trust_remote_code=False)
        model.to(self.device)
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.check_budget()
        return model

    def features(self, records):
        self.check_budget()
        return make_features(records, self.tokenizer, self.args.max_length, self.args.stride)

    def infer(self, model, records, features):
        model.eval()
        choices = {}
        selected = [f for f in features if records[f['record_index']]['baseline_answer']]
        with torch.inference_mode():
            for start in range(0, len(selected), self.args.eval_batch_size):
                self.check_budget()
                current = selected[start:start + self.args.eval_batch_size]
                batch = collate(current, self.tokenizer, self.device)
                with self.autocast():
                    outputs = model(**batch['inputs'])
                starts = outputs.start_logits.float().cpu().numpy()
                ends = outputs.end_logits.float().cpu().numpy()
                for feature, a, b in zip(current, starts, ends):
                    candidate = decode_window(feature, a, b, self.args.max_span_words)
                    ri = feature['record_index']
                    if candidate and (ri not in choices or candidate['margin'] > choices[ri]['margin']):
                        choices[ri] = candidate
        return prediction_spans(records, choices)

    def update(self, model, optimizer, batches):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for features in batches:
            self.check_budget()
            batch = collate(features, self.tokenizer, self.device)
            with self.autocast():
                output = model(**batch['inputs'])
            start = output.start_logits.float().masked_fill(~batch['start_mask'], -10000)
            end = output.end_logits.float().masked_fill(~batch['end_mask'], -10000)
            a = torch.nn.functional.cross_entropy(start, batch['start_position'], reduction='none')
            b = torch.nn.functional.cross_entropy(end, batch['end_position'], reduction='none')
            loss = (((a + b) / 2) * batch['weight']).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            (loss / len(batches)).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(np.mean(losses)), float(norm)

    def benchmark(self):
        epoch_windows = None
        if self.external:
            from .pretrain import PRETRAIN_SOURCES
            all_records = [r for r in hydrate(self.external, 'train') if r['source'] in PRETRAIN_SOURCES]
            if not all_records:
                raise ValueError('No external pretraining examples')
            # Sampling questions proportionally reproduces the real window mixture in expectation.
            rng = random.Random(self.args.seed)
            records = rng.sample(all_records, min(len(all_records), 512))
            counts = self.external.get('window_counts', {})
            epoch_windows = sum(counts.get(f'train:{source}', 0) for source in PRETRAIN_SOURCES) or None
        else:
            ids = set(self.bundle['folds'][0]['train'])
            records = [r for r in self.records if r['conversation'] in ids]
        features = self.features(records)
        model = self.new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=.01)
        self.sync()
        durations = []
        for step in range(self.args.benchmark_steps):
            start = time.monotonic()
            group = []
            for acc in range(self.args.accumulation):
                offset = ((step * self.args.accumulation + acc) * self.args.batch_size) % len(features)
                group.append(features[offset:offset + self.args.batch_size])
            loss, norm = self.update(model, optimizer, group)
            self.sync()
            durations.append(time.monotonic() - start)
            self.emit('benchmark_step', step=step + 1, loss=loss, gradient_norm=norm,
                      seconds=durations[-1])
        stable = durations[1:] or durations
        steps_per_epoch = math.ceil(math.ceil(len(features) / self.args.batch_size) / self.args.accumulation)
        per_window = float(np.mean(stable)) / (self.args.batch_size * self.args.accumulation)
        report = {'optimizer_steps': len(durations), 'training_records': len(records),
                  'scope': 'External pretraining mixture sample' if self.external else 'Original training fold',
                  'sampled_windows_by_source': dict(Counter(records[f['record_index']].get('source', 'original')
                                                            for f in features)),
                  'training_windows': len(features), 'step_seconds': durations,
                  'mean_step_seconds_after_warmup': float(np.mean(stable)),
                  'seconds_per_window': per_window,
                  'external_epoch_windows': epoch_windows,
                  'estimated_minutes_per_external_epoch': round(epoch_windows * per_window / 60, 1)
                  if epoch_windows else None,
                  'estimated_training_seconds_per_fold': float(np.mean(stable)) * steps_per_epoch * self.args.epochs,
                  'estimate_excludes': 'downloads, preprocessing, evaluation, checkpoint writes, other folds',
                  'max_cuda_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3 if self.device == 'cuda' else None}
        write_json(self.out / 'benchmark.json', report)
        self.save_status('complete', benchmark=report)

    def train_fold(self, plan):
        fold = plan['fold']
        directory = self.out / f'fold_{fold}'
        directory.mkdir()
        groups = {k: [r for r in self.records if r['conversation'] in set(plan[k])]
                  for k in ('train', 'dev', 'test')}
        groups['train'].extend(self.medical_records)
        # Generated rows only ever join the fold that already trains on their conversation.
        training_conversations = set(plan['train'])
        generated = [r for r in self.generated if r['conversation'] in training_conversations]
        groups['train'].extend(generated)
        features = {k: self.features(v) for k, v in groups.items()}
        write_json(directory / 'split.json', {**plan,
                                              'external_medical_ids': [r['id'] for r in self.medical_records],
                                              'generated_questions': len(generated),
                                              'questions': {k: len(v) for k, v in groups.items()},
                                              'windows': {k: len(v) for k, v in features.items()}})
        model = self.new_model(seed=self.args.seed + fold)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=.01)
        initial = score(groups['dev'], self.infer(model, groups['dev'], features['dev']))
        best_score, best_epoch = initial['mean_tiou'], 0
        history = [{'epoch': 0, 'dev_tiou': best_score}]
        best_path = directory / 'best'
        model.save_pretrained(best_path, safe_serialization=True)
        self.tokenizer.save_pretrained(best_path)
        self.emit('fold_start', fold=fold, initial_dev_tiou=best_score)
        rng = random.Random(self.args.seed + fold)
        updates, stale = 0, 0
        for epoch in range(1, self.args.epochs + 1):
            shuffled = list(features['train'])
            rng.shuffle(shuffled)
            batches = [shuffled[i:i + self.args.batch_size] for i in range(0, len(shuffled), self.args.batch_size)]
            losses = []
            for position in range(0, len(batches), self.args.accumulation):
                updates += 1
                loss, norm = self.update(model, optimizer, batches[position:position + self.args.accumulation])
                losses.append(loss)
                if updates == 1 or updates % 10 == 0:
                    self.emit('train_step', fold=fold, epoch=epoch, step=updates, loss=loss, gradient_norm=norm)
            development = score(groups['dev'], self.infer(model, groups['dev'], features['dev']))
            metric = development['mean_tiou']
            history.append({'epoch': epoch, 'dev_tiou': metric, 'training_loss': float(np.mean(losses))})
            write_json(directory / 'history.json', history)
            self.emit('epoch_end', fold=fold, **history[-1])
            if metric > best_score:
                best_score, best_epoch, stale = metric, epoch, 0
                model.save_pretrained(best_path, safe_serialization=True)
            else:
                stale += 1
            if stale >= self.args.patience:
                break
        del optimizer, model
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        model = self.new_model(best_path)
        predictions = self.infer(model, groups['test'], features['test'])
        report = score(groups['test'], predictions)
        report.update(fold=fold, selected_epoch=best_epoch, selection='inner-dev temporal IoU only',
                      checkpoint=str(best_path), history=history)
        write_json(directory / 'report.json', report)
        self.emit('fold_complete', fold=fold, raw=report['raw'], raw_gain=report['raw_gain'], best_epoch=best_epoch)
        del model
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return report

    def cross_validate(self):
        plans = [p for p in self.bundle['folds'] if self.args.fold == 'all' or p['fold'] == int(self.args.fold)]
        if not plans:
            raise ValueError('Requested fold does not exist')
        reports = []
        for plan in plans:
            reports.append(self.train_fold(plan))
        ids = {r['id'] for report in reports for r in report['rows']}
        records = [r for r in self.records if r['id'] in ids]
        predictions = {r['id']: r['candidate'] for report in reports for r in report['rows']}
        combined = score(records, predictions)
        positives = [r for r in combined['rows'] if r['label']]
        conversations = sorted({r['conversation'] for r in positives})
        sums = np.array([sum(r['tiou'] - r['baseline_tiou'] for r in positives if r['conversation'] == c)
                         for c in conversations])
        counts = np.array([sum(r['conversation'] == c for r in positives) for c in conversations])
        draws = np.random.default_rng(self.args.seed).integers(0, len(conversations), (2000, len(conversations)))
        gains = .6 * sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        combined.update(scope=self.bundle['scope'], all_outer_folds_completed=len(plans) == len(self.bundle['folds']),
                        folds=[{'fold': r['fold'], 'raw': r['raw'], 'raw_gain': r['raw_gain'],
                                'selected_epoch': r['selected_epoch']} for r in reports],
                        raw_gain_conversation_bootstrap_ci95=np.quantile(gains, [.025, .975]).tolist(),
                        promotion='Requires separate end-to-end and platform validation; serving is unchanged')
        write_json(self.out / 'report.json', combined)
        self.save_status('complete', raw=combined['raw'], raw_gain=combined['raw_gain'])

    def fit(self):
        """Fit a fixed number of epochs on all public data, without claiming held-out performance."""
        features = self.features(self.records + self.medical_records + self.generated)
        model = self.new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=.01)
        rng = random.Random(self.args.seed)
        for epoch in range(1, self.args.epochs + 1):
            rng.shuffle(features)
            batches = [features[i:i + self.args.batch_size] for i in range(0, len(features), self.args.batch_size)]
            losses = []
            for position in range(0, len(batches), self.args.accumulation):
                loss, _ = self.update(model, optimizer, batches[position:position + self.args.accumulation])
                losses.append(loss)
            self.emit('fit_epoch', epoch=epoch, loss=float(np.mean(losses)))
        self.check_budget()
        model.save_pretrained(self.out / 'model', safe_serialization=True)
        self.tokenizer.save_pretrained(self.out / 'model')
        self.save_status('complete', scope='All-data fit; no held-out score and no automatic deployment')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='deepset/deberta-v3-large-squad2')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--mode', choices=('benchmark', 'pretrain', 'cv', 'fit'), default='cv')
    parser.add_argument('--extra-data', type=Path,
                        help='Generated in-domain rows; added only to folds that already train '
                             'on their conversation')
    parser.add_argument('--external-data', type=Path,
                        help='Checked text-only bundle: CoQA and MASH-QA pretrain; only SIMORD train adapts CV/fit')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32'), default='auto')
    parser.add_argument('--fold', default='all')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--evals-per-epoch', type=int, default=1,
                        help='Development evaluations inside each pretraining epoch')
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=192)
    parser.add_argument('--max-span-words', type=int, default=128)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--benchmark-steps', type=int, default=10)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--min-free-gb', type=float, default=16)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'pretrain' and not args.external_data:
        parser.error('--mode pretrain requires --external-data')
    for name in ('epochs', 'evals_per_epoch', 'patience', 'batch_size', 'eval_batch_size', 'accumulation',
                 'benchmark_steps', 'max_seconds'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.learning_rate <= 0 or args.min_free_gb < 0 or args.max_span_words <= 0:
        parser.error('Learning rate/span length must be positive and min-free-gb nonnegative')
    if args.output.exists():
        parser.error('Output already exists. Choose a new run directory; old results are never removed.')
    runner = None
    try:
        runner = Runner(args)
        runner.save_status('running')
        if args.mode == 'benchmark':
            runner.benchmark()
        elif args.mode == 'pretrain':
            from .pretrain import pretrain
            pretrain(runner)
        elif args.mode == 'fit':
            runner.fit()
        else:
            runner.cross_validate()
    except (BudgetExpired, KeyboardInterrupt) as error:
        if runner:
            runner.save_status('budget_exceeded' if isinstance(error, BudgetExpired) else 'interrupted', error=str(error))
        elif args.output.exists():
            write_json(args.output / 'failure.json', {'status': 'initialization_interrupted', 'error': str(error)})
        raise SystemExit(124 if isinstance(error, BudgetExpired) else 130)
    except Exception as error:
        if runner:
            runner.save_status('failed', error=f'{type(error).__name__}: {error}')
        elif args.output.exists():
            write_json(args.output / 'failure.json', {'status': 'initialization_failed',
                                                    'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
