import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from tools.evidence_training.data import (
    best_word_span, fold_plan, prepare, render_words, score, tiou, validate_bundle, write_json,
)
from tools.evidence_training.features import decode_window, make_features, prediction_spans


def record(cid, positive=True, text=None):
    tokens = (text or f'Visit {cid} take the tablet daily now').split()
    words = [{'word': ' ' + token, 'start': i * .5, 'end': i * .5 + .4} for i, token in enumerate(tokens)]
    context, ranges = render_words(words)
    return {'id': f'{cid}_{positive}', 'conversation': cid, 'question': 'Take the tablet daily?',
            'label': positive, 'context': context, 'word_chars': ranges, 'words': words,
            'gold_words': [4, 5] if positive else None,
            'gold_chars': [ranges[4][0], ranges[5][1]] if positive else None,
            'gold': [words[4]['start'], words[5]['end']] if positive else None,
            'mapping_tiou': 1.0 if positive else None, 'baseline_answer': positive,
            'baseline_span': [words[4]['start'], words[5]['end']] if positive else None}


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {v: i for i, v in enumerate(['[PAD]', '[CLS]', '[SEP]', '[UNK]', 'Visit', 'Take', 'take',
                                        'the', 'tablet', 'daily', 'daily?', 'now', 'filler'])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    raw.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', cls_token='[CLS]',
                                  sep_token='[SEP]', unk_token='[UNK]', model_max_length=128)


class DataTest(unittest.TestCase):
    def test_gold_mapper_matches_brute_force(self):
        rng = random.Random(14)
        words = record('x')['words']
        for _ in range(50):
            a = rng.uniform(0, 2.5)
            gold = [a, a + rng.uniform(.1, 1)]
            endpoints, quality = best_word_span(words, gold)
            oracle = max(tiou(gold, [words[i]['start'], words[j]['end']])
                         for i in range(len(words)) for j in range(i, len(words)))
            self.assertAlmostEqual(quality, oracle)
            self.assertGreaterEqual(endpoints[1], endpoints[0])

    def test_real_corpus_coverage_and_frozen_control(self):
        bundle = prepare()
        validate_bundle(bundle)
        self.assertEqual(bundle['summary']['questions'], 390)
        self.assertEqual(bundle['summary']['positives'], 195)
        report = score(bundle['records'], {r['id']: r['baseline_span'] for r in bundle['records']})
        self.assertAlmostEqual(report['raw'], .8297086456044358)
        self.assertEqual(report['raw_gain'], 0)
        self.assertGreater(bundle['summary']['mean_mapping_tiou'], .98)

    def test_fold_leakage_and_duplicate_context_rejected(self):
        records = [record(f'c{i}', positive) for i in range(6) for positive in (True, False)]
        bundle = {'schema_version': 1, 'records': records, 'folds': fold_plan(records)}
        validate_bundle(bundle)
        bad = copy.deepcopy(bundle)
        bad['folds'][0]['train'].append(bad['folds'][0]['test'][0])
        with self.assertRaises(ValueError):
            validate_bundle(bad)
        bad = copy.deepcopy(bundle)
        for row in bad['records']:
            if row['conversation'] == 'c1':
                row['context'] = records[0]['context']
                row['words'] = records[0]['words']
        with self.assertRaisesRegex(ValueError, 'Duplicate transcripts'):
            validate_bundle(bad)

    def test_missing_prediction_and_false_answer_do_not_escape_denominator(self):
        rows = [record('a')]
        with self.assertRaises(ValueError):
            score(rows, {})
        rows[0]['baseline_answer'] = False
        report = score(rows, {rows[0]['id']: rows[0]['gold']})
        self.assertEqual(report['mean_tiou'], 0)
        self.assertEqual(report['accuracy'], 0)


class FeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tiny_tokenizer()

    def test_repeated_text_preserves_later_occurrence(self):
        r = record('a', text='Visit a tablet daily tablet daily now')
        feature = make_features([r], self.tokenizer, max_length=96, stride=16)[0]
        a = np.zeros(len(feature['first_word']))
        b = np.zeros(len(a))
        a[feature['start_position']] = 9
        b[feature['end_position']] = 9
        choice = decode_window(feature, a, b)
        self.assertEqual(choice['words'], [4, 5])
        spans = prediction_spans([r], {0: choice})
        self.assertEqual(spans[r['id']], r['gold'])

    def test_overflow_targets_question_mask_and_negative_windows(self):
        r = record('a', text='Visit a take the tablet daily ' + ' '.join(['filler'] * 130))
        features = make_features([r], self.tokenizer, max_length=96, stride=16)
        self.assertGreater(len(features), 1)
        self.assertTrue(any(f['start_position'] != f['cls'] for f in features))
        self.assertTrue(any(f['start_position'] == f['cls'] for f in features))
        self.assertAlmostEqual(sum(f['weight'] for f in features), 1)
        for f in features:
            # Question tokens must not compete with evidence from the transcript.
            self.assertFalse(f['start_mask'][1])
            self.assertFalse(f['end_mask'][1])
        negative = record('b', positive=False)
        for f in make_features([negative], self.tokenizer, max_length=96, stride=16):
            self.assertEqual((f['start_position'], f['end_position']), (f['cls'], f['cls']))

    def test_invalid_order_and_overlong_pairs_are_excluded(self):
        f = make_features([record('a')], self.tokenizer, max_length=96, stride=16)[0]
        a = np.zeros(len(f['first_word']))
        b = np.zeros(len(a))
        a[-2], b[7] = 20, 20  # individually attractive, invalid start/end order
        choice = decode_window(f, a, b, max_words=2)
        start, end = choice['words']
        self.assertLessEqual(start, end)
        self.assertLessEqual(end - start + 1, 2)
        a[0] = float('nan')
        with self.assertRaises(ValueError):
            decode_window(f, a, b)

    def test_long_context_tail_is_reachable_without_using_gold_to_choose_windows(self):
        r = record('tail', text='Visit tail ' + ' '.join(['filler'] * 400) + ' tablet daily')
        a, b = len(r['words']) - 2, len(r['words']) - 1
        r.update(gold_words=[a, b], gold_chars=[r['word_chars'][a][0], r['word_chars'][b][1]],
                 gold=[r['words'][a]['start'], r['words'][b]['end']])
        features = make_features([r], self.tokenizer, max_length=96, stride=16)
        self.assertGreater(len(features), 4)
        self.assertTrue(any(f['first_word'][f['start_position']] == a for f in features))
        self.assertEqual(max(max(f['last_word']) for f in features), b)


class EndToEndTest(unittest.TestCase):
    def test_expired_budget_prevents_new_work(self):
        import time
        from tools.evidence_training.train import BudgetExpired, Runner
        runner = object.__new__(Runner)
        runner.deadline = time.monotonic() - 1
        with self.assertRaises(BudgetExpired):
            runner.check_budget()

    def test_existing_run_directory_is_preserved(self):
        from tools.evidence_training.train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sentinel').write_text('keep')
            with self.assertRaises(SystemExit) as caught:
                main(['--data', str(root / 'missing-data.json'), '--output', str(root)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual((root / 'sentinel').read_text(), 'keep')

    def test_cpu_training_checkpoint_and_complete_out_of_fold_scoring(self):
        import torch
        from transformers import BertConfig, BertForQuestionAnswering
        from tools.evidence_training.train import main

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = tiny_tokenizer()
            model = BertForQuestionAnswering(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
            model.save_pretrained(root / 'tiny')
            tokenizer.save_pretrained(root / 'tiny')
            records = [record(f'c{i}', positive) for i in range(6) for positive in (True, False)]
            bundle = {'schema_version': 1, 'records': records, 'folds': fold_plan(records),
                      'sources': [], 'scope': 'Synthetic mechanics test, not an accuracy benchmark'}
            write_json(root / 'data.json', bundle)
            main(['--data', str(root / 'data.json'), '--output', str(root / 'result'),
                  '--model', str(root / 'tiny'), '--device', 'cpu', '--mode', 'cv', '--epochs', '1',
                  '--batch-size', '2', '--accumulation', '1', '--max-length', '96', '--stride', '16',
                  '--max-seconds', '60'])
            report = json.loads((root / 'result/report.json').read_text())
            self.assertTrue(report['all_outer_folds_completed'])
            self.assertEqual(report['questions'], 12)
            self.assertEqual(report['positives'], 6)
            self.assertEqual(report['accuracy'], 1)
            self.assertTrue((root / 'result/fold_0/best/model.safetensors').exists())
            self.assertEqual(json.loads((root / 'result/manifest.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
