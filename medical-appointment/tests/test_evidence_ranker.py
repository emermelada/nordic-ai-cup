import json
from pathlib import Path
import random
import tempfile
import unittest

from tools.evidence_training.data import fold_plan, prepare, write_json
from tools.evidence_training.ranker import (
    candidates, make_features, predict, score_records, sentence_ranges, soft_targets, span_time,
    temporal_iou, word_iou, word_texts,
)
from tests.test_evidence_training import record, tiny_tokenizer


def sentence_record(cid, text, gold_words=(4, 5)):
    words = [{'word': ' ' + token, 'start': i * .5, 'end': i * .5 + .4}
             for i, token in enumerate(text.split())]
    from tools.evidence_training.data import render_words
    context, ranges = render_words(words)
    a, b = gold_words
    return {'id': cid, 'conversation': cid, 'question': 'Take the tablet daily?', 'label': True,
            'context': context, 'word_chars': ranges, 'words': words, 'gold_words': [a, b],
            'gold_chars': [ranges[a][0], ranges[b][1]], 'mapping_tiou': 1.0,
            'gold': [words[a]['start'], words[b]['end']], 'baseline_answer': True,
            'baseline_span': [words[0]['start'], words[0]['end']]}


FILLER = ' '.join(['filler filler filler filler,'] * 60) + ' End.'


class PoolTest(unittest.TestCase):
    def test_sentences_split_on_end_punctuation_only(self):
        texts = [' Take', ' the', ' tablet,', ' daily.', ' Then', ' rest.']
        self.assertEqual(sentence_ranges(texts), [(0, 3), (4, 5)])

    def test_pool_contains_runs_and_clauses_and_respects_the_word_cap(self):
        texts = [' Take', ' the', ' tablet,', ' daily.', ' Then', ' rest.']
        pool = candidates(texts, max_sentences=2, max_words=6)
        self.assertIn((0, 3), pool)
        self.assertIn((4, 5), pool)
        self.assertIn((0, 5), pool)
        self.assertIn((0, 2), pool)
        self.assertIn((3, 3), pool)
        self.assertTrue(all(b - a + 1 <= 6 for a, b in pool))
        self.assertEqual(len(pool), len(set(pool)))
        capped = candidates(texts, max_sentences=2, max_words=3)
        self.assertNotIn((0, 5), capped)

    def test_word_and_temporal_overlap(self):
        self.assertEqual(word_iou((2, 4), (2, 4)), 1)
        self.assertAlmostEqual(word_iou((0, 3), (2, 5)), 2 / 6)
        self.assertEqual(temporal_iou([1.0, 2.0], [3.0, 4.0]), 0)
        self.assertEqual(temporal_iou([1.0, 2.0], [2.0, 1.0]), 0)
        self.assertAlmostEqual(temporal_iou([0.0, 2.0], [1.0, 3.0]), 1 / 3)

    def test_padded_candidate_slots_do_not_poison_the_loss(self):
        import torch
        from transformers import BertConfig, BertModel
        from tools.evidence_training.ranker import SpanRanker, collate

        tokenizer = tiny_tokenizer()
        model = SpanRanker(BertModel(BertConfig(
            vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=128)), 16)
        wide = sentence_record('a', 'Visit the tablet, daily now. Take the tablet daily now.')
        narrow = sentence_record('b', 'Take the tablet daily now.', gold_words=(0, 4))
        batch = (make_features([wide], tokenizer, max_length=96, stride=16)
                 + make_features([narrow], tokenizer, max_length=96, stride=16))
        self.assertNotEqual(len(batch[0]['candidate_index']), len(batch[1]['candidate_index']))
        tensors = collate(batch, tokenizer, 'cpu')
        logits = model(tensors['inputs'], tensors['cls'], tensors['start'], tensors['end'],
                       tensors['mask'])
        loss = -(tensors['target'] * torch.log_softmax(logits, dim=1)).sum(1)
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_soft_targets_favour_the_closest_candidate(self):
        weights = soft_targets([0.2, 1.0, 0.5])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(max(range(3), key=lambda i: weights[i]), 1)
        self.assertGreater(weights[1], 10 * weights[2])

    def test_public_corpus_pool_reaches_the_documented_oracle(self):
        bundle = prepare()
        positives = [r for r in bundle['records'] if r['label']]
        values = []
        for row in positives:
            pool = candidates(word_texts(row))
            values.append(max(temporal_iou(row['gold'], span_time(row['words'], a, b))
                              for a, b in pool))
        self.assertEqual(len(values), 195)
        self.assertGreater(sum(values) / len(values), 0.92)


class FeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tiny_tokenizer()

    def test_every_candidate_maps_back_to_its_own_words(self):
        row = sentence_record('a', 'Visit the tablet daily. Take the tablet daily now.')
        pool = candidates(word_texts(row))
        features = make_features([row], self.tokenizer, max_length=96, stride=16)
        self.assertTrue(features)
        seen = set()
        for feature in features:
            self.assertEqual(len(feature['candidate_index']), len(feature['start']))
            for key, start, end in zip(feature['candidate_index'], feature['start'], feature['end']):
                self.assertLessEqual(start, end)
                seen.add(pool[key])
        self.assertEqual(seen, set(pool))

    def test_repeated_text_keeps_both_occurrences_apart(self):
        row = sentence_record('a', 'Take the tablet daily. Take the tablet daily.', gold_words=(4, 7))
        pool = candidates(word_texts(row))
        self.assertIn((0, 3), pool)
        self.assertIn((4, 7), pool)
        targets = [temporal_iou(row['gold'], span_time(row['words'], a, b)) for a, b in pool]
        best = pool[max(range(len(pool)), key=lambda i: targets[i])]
        self.assertEqual(best, (4, 7))

    def test_windows_without_gold_are_trained_as_null(self):
        row = sentence_record('a', 'Take the tablet daily now. ' + FILLER, gold_words=(0, 4))
        features = make_features([row], self.tokenizer, max_length=96, stride=16)
        self.assertGreater(len(features), 1)
        self.assertTrue(any(f['positive'] for f in features))
        self.assertTrue(any(not f['positive'] for f in features))
        self.assertAlmostEqual(sum(f['weight'] for f in features), 1)

    def test_serving_records_without_gold_still_produce_every_window(self):
        row = sentence_record('a', 'Take the tablet daily now. Visit the tablet now.')
        serving = {'id': '3', 'question': row['question'], 'context': row['context'],
                   'word_chars': row['word_chars'], 'words': row['words'], 'label': False,
                   'gold_words': None}
        features = make_features([serving], self.tokenizer, max_length=96, stride=16)
        self.assertTrue(features)
        self.assertTrue(all(not f['positive'] for f in features))
        self.assertTrue(all(all(t == 0 for t in f['target']) for f in features))

    def test_null_windows_are_dropped_when_sampling_is_off(self):
        row = sentence_record('a', 'Take the tablet daily now. ' + FILLER, gold_words=(0, 4))
        features = make_features([row], self.tokenizer, max_length=96, stride=16,
                                 rng=random.Random(0), keep_null=0.0)
        self.assertTrue(features)
        self.assertTrue(all(f['positive'] for f in features))


class TrainingTest(unittest.TestCase):
    def test_cpu_training_learns_the_pool_and_scores_out_of_fold(self):
        import torch
        from transformers import BertConfig, BertModel
        from tools.evidence_training.ranker import SpanRanker, main

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = tiny_tokenizer()
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=128)
            BertModel(config).save_pretrained(root / 'tiny')
            tokenizer.save_pretrained(root / 'tiny')
            records = []
            for i in range(6):
                positive = sentence_record(
                    f'c{i}', f'Visit the tablet now, take the tablet daily, and rest well c{i}.',
                    gold_words=(4, 7))
                negative = dict(positive, id=f'c{i}_no', label=False, gold_words=None, gold=None,
                                gold_chars=None, mapping_tiou=None, baseline_answer=False,
                                baseline_span=None)
                records += [positive, negative]
            for row in records:
                row['conversation'] = row['id'].split('_')[0]
            bundle = {'schema_version': 1, 'records': records, 'folds': fold_plan(records),
                      'sources': [], 'scope': 'Synthetic mechanics test, not an accuracy benchmark'}
            write_json(root / 'data.json', bundle)
            report = main(['--data', str(root / 'data.json'), '--output', str(root / 'result'),
                           '--model', str(root / 'tiny'), '--device', 'cpu', '--mode', 'cv',
                           '--epochs', '2', '--batch-size', '2', '--accumulation', '1',
                           '--max-length', '96', '--stride', '16', '--max-seconds', '120',
                           '--min-free-gb', '0'])
            self.assertEqual(len(report['folds']), 3)
            self.assertEqual(report['pooled']['questions'], 6)
            self.assertIn('rows', report)
            for row in report['rows'].values():
                self.assertLessEqual(row['words'][0], row['words'][1])
                self.assertAlmostEqual(row['span'][0], row['span'][0])
            saved = json.loads((root / 'result/manifest.json').read_text())
            self.assertEqual(saved['status'], 'complete')
            self.assertTrue(saved['answers_frozen'])

    def test_a_trained_ranker_beats_an_untrained_one_on_its_own_data(self):
        # Gold is exactly one enumerated sentence, so a working ranker reaches 1.0.
        import torch
        from transformers import BertConfig, BertModel
        from tools.evidence_training.ranker import SpanRanker, batches, collate

        torch.set_num_threads(1)
        tokenizer = tiny_tokenizer()
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
        model = SpanRanker(BertModel(config), 32)
        rows = [sentence_record(
            f'c{i}', 'Visit the tablet now. Take the tablet daily. Visit the tablet. Take now.',
            gold_words=(4, 7)) for i in range(4)]
        features = make_features(rows, tokenizer, max_length=96, stride=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        history = []
        for _ in range(40):
            for batch in batches(features, 4, random.Random(0)):
                tensors = collate(batch, tokenizer, 'cpu')
                logits = model(tensors['inputs'], tensors['cls'], tensors['start'], tensors['end'],
                               tensors['mask'])
                loss = -(tensors['target'] * torch.log_softmax(logits, dim=1)).sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(float(loss))
        self.assertLess(history[-1], 0.5 * history[0])
        after = score_records(rows, predict(model, rows, features, tokenizer, 'cpu'))
        self.assertGreater(after, 0.99)


if __name__ == '__main__':
    unittest.main()


class ServingTest(unittest.TestCase):
    def test_checkpoint_round_trips_through_the_serving_module(self):
        import importlib
        import os
        import torch
        from transformers import BertConfig, BertModel
        from tools.evidence_training.ranker import SpanRanker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'model'
            root.mkdir()
            tokenizer = tiny_tokenizer()
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=128)
            model = SpanRanker(BertModel(config), 16)
            torch.save(model.state_dict(), root / 'ranker.pt')
            config.save_pretrained(root)
            tokenizer.save_pretrained(root)

            os.environ['MEDICAL_RANKER_MODEL'] = str(root)
            os.environ['MEDICAL_RANKER_MAX_LENGTH'] = '96'
            os.environ['MEDICAL_RANKER_STRIDE'] = '16'
            try:
                import pipeline.span_ranker as serving
                serving = importlib.reload(serving)
                words = [{'word': ' ' + token, 'start': i * .5, 'end': i * .5 + .4}
                         for i, token in enumerate('Take the tablet daily now. '
                                                   'Visit the tablet now.'.split())]
                spans = serving.predict_spans(words, ['Take the tablet daily?', 'Another?'],
                                              [True, False])
                self.assertEqual(sorted(spans), [1])
                start, end = spans[1]
                self.assertLess(start, end)
                self.assertGreaterEqual(start, words[0]['start'])
                self.assertLessEqual(end, words[-1]['end'])
                loaded = serving._state['model'].state_dict()
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(loaded[key].cpu(), value), key)
                self.assertEqual(serving.predict_spans(words, ['q', 'q'], [False, False]), {})
            finally:
                for key in ('MEDICAL_RANKER_MODEL', 'MEDICAL_RANKER_MAX_LENGTH',
                            'MEDICAL_RANKER_STRIDE'):
                    os.environ.pop(key, None)
                importlib.reload(importlib.import_module('pipeline.span_ranker'))
