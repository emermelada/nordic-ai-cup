"""Checks for external rationale ingestion: offsets, leakage, and mixed-source training."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.evidence_training.data import digest, fold_plan, write_json
from tools.evidence_training.external import (
    OverlapIndex, coqa_documents, context_record, mashqa_documents, simord_documents,
    source_hash, validate_external, word_span,
)
from tests.test_evidence_training import record, tiny_tokenizer

RAW = Path(__file__).resolve().parents[1] / 'runs/evidence-external-20260919/raw'


def coqa_story(raw, story, answers):
    """Write one minimal CoQA file; answers are (input_text, span_start, span_end) triples."""
    data = {'data': [{'id': 'story1', 'source': 'wikipedia', 'filename': 'story1.txt', 'story': story,
                      'questions': [{'turn_id': i + 1, 'input_text': f'What happened in step {i + 1}?'}
                                    for i in range(len(answers))],
                      'answers': [{'turn_id': i + 1, 'input_text': text, 'span_start': start,
                                   'span_end': end, 'span_text': story[start:end]}
                                  for i, (text, start, end) in enumerate(answers)]},
                     {'id': 'story2', 'source': 'race', 'filename': 'race1.txt', 'story': story,
                      'questions': [{'turn_id': 1, 'input_text': 'Excluded licence?'}],
                      'answers': [{'turn_id': 1, 'input_text': 'yes', 'span_start': 0, 'span_end': 5,
                                   'span_text': story[:5]}]}]}
    (raw / 'coqa-train-v1.0.json').write_text(json.dumps(data))


def external_bundle(target_path, rows):
    """Build a validated bundle from (id, source, split, text, question, gold_words) rows."""
    contexts, examples = {}, []
    for index, (source, split, text, question, gold) in enumerate(rows):
        cid = f'{source}:{index}'
        contexts[cid] = {**context_record(text), 'group': f'{source}:group{index}', 'source': source,
                         'domain': 'test', 'license': 'test', 'source_document': cid,
                         'source_context_sha256': source_hash(text)}
        examples.append({'id': f'{cid}:q', 'question': question, 'label': gold is not None,
                         'answer_type': 'span', 'annotation': 'test', 'conversation': cid, 'split': split,
                         'source': source, 'gold_words': gold, 'raw_span': None,
                         'word_boundary_expanded': False})
    return {'schema_version': 'external-rationales-v1', 'metric': 'word_span_iou_not_temporal',
            'excluded_target_sha256': digest(target_path), 'contexts': contexts, 'examples': examples,
            'sources': [], 'window_counts': {}}


class SourceConversionTest(unittest.TestCase):
    def test_coqa_history_labels_and_licence_filter(self):
        story = 'Visit one. Take the tablet daily now. Then rest until later today.'
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            coqa_story(raw, story, [('the tablet', 16, 26), ('no', 11, 37), ('unknown', -1, -1),
                                    ('yes', 38, 50)])
            documents = list(coqa_documents(raw, 'train'))
        self.assertEqual([d['id'] for d in documents], ['coqa:story1'])  # RACE licence excluded
        examples = documents[0]['examples']
        self.assertEqual([e['answer_type'] for e in examples], ['span', 'no', 'unknown', 'yes'])
        # A CoQA "no" keeps its rationale; only "unknown" is unanswerable.
        self.assertEqual([e['label'] for e in examples], [True, True, False, True])
        self.assertNotIn('Previous', examples[0]['question'])
        self.assertIn('Previous answer: the tablet', examples[1]['question'])
        # Two turns of history at most, and never the current turn's answer.
        self.assertEqual(examples[3]['question'].count('Previous question:'), 2)
        for example in examples:
            self.assertTrue(example['question'].endswith(example['current_question']))
            if example['label']:
                self.assertNotIn(example['span_text'], example['question'])

    def test_character_spans_expand_to_whole_words_and_reject_mismatch(self):
        text = 'Take the tablet daily now'
        self.assertEqual(word_span(text, 5, 15, expected='the tablet'), ([1, 2], False))
        self.assertEqual(word_span(text, 6, 14, expected='he table'), ([1, 2], True))  # partial words widen
        self.assertEqual(word_span(text, 4, 15, expected=' the tablet'), ([1, 2], False))  # surrounding space
        with self.assertRaisesRegex(ValueError, 'source_rationale_mismatch'):
            word_span(text, 5, 15, expected='tablet')
        with self.assertRaisesRegex(ValueError, 'invalid_character_span'):
            word_span(text, 5, 500, expected=None)
        with self.assertRaisesRegex(ValueError, 'empty_rationale'):
            word_span(text, 4, 5, expected=' ')

    @unittest.skipUnless((RAW / 'mashqa_data.zip').exists(), 'MASH-QA archive not downloaded')
    def test_mashqa_offsets_match_the_published_archive(self):
        documents = list(mashqa_documents(RAW, 'train'))
        self.assertGreater(len(documents), 4000)
        checked = 0
        for document in documents[:200]:
            for example in document['examples']:
                if example['reject_reason']:
                    continue
                start, end = example['raw_span']
                self.assertEqual(document['text'][start:end], example['span_text'])
                words, _ = word_span(document['text'], start, end, expected=example['span_text'])
                self.assertLessEqual(words[0], words[1])
                checked += 1
        self.assertGreater(checked, 100)

    @unittest.skipUnless((RAW / 'mashqa_data.zip').exists(), 'MASH-QA archive not downloaded')
    def test_mashqa_test_labels_are_never_imported(self):
        documents = list(mashqa_documents(RAW, 'test'))
        self.assertGreater(len(documents), 100)
        self.assertEqual(sum(len(d['examples']) for d in documents), 0)
        self.assertTrue(all(d['text'] for d in documents))  # contexts still exclude overlapping training text

    def test_mashqa_noncontiguous_and_impossible_answers_are_rejected(self):
        context = 'Take the tablet daily now and then rest until later today'
        qas = [{'id': 'a', 'question': 'When?', 'is_impossible': False,
                'answers': [{'text': 'the tablet', 'answer_start': 5, 'answer_span': [0, 1]}]},
               {'id': 'b', 'question': 'Why?', 'is_impossible': False,
                'answers': [{'text': 'the tablet', 'answer_start': 5, 'answer_span': [0, 3]}]},
               {'id': 'c', 'question': 'Who?', 'is_impossible': True, 'answers': []}]
        payload = {'data': [{'title': 'https://example.test/page#fragment',
                             'paragraphs': [{'context': context, 'qas': qas}]}]}
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with zipfile.ZipFile(raw / 'mashqa_data.zip', 'w') as archive:
                archive.writestr('mashqa_data/train_webmd_squad_v2_consec.json', json.dumps(payload))
            documents = list(mashqa_documents(raw, 'train'))
        reasons = [e['reject_reason'] for e in documents[0]['examples']]
        self.assertEqual(reasons, [None, 'noncontiguous_or_invalid_mashqa_answer',
                                   'noncontiguous_or_invalid_mashqa_answer'])
        self.assertEqual(documents[0]['group'], 'mashqa:https://example.test/page')

    @unittest.skipUnless(importlib.util.find_spec('textgrid'), 'TextGrid is a preparation-only dependency')
    def test_simord_rejects_noncontiguous_provenance_and_keeps_sentence_offsets(self):
        transcript = 'Take the tablet daily. Then rest until later. Come back next month.'
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / 'aci-train.json').write_text(json.dumps(
                {'data': [{'file': 'D2N068-virtassist-gpt4', 'src': transcript}]}))
            (raw / 'simord-train.json').write_text(json.dumps(
                [{'id': 'acibench_D2N068_virtassist_train',
                  'expected_orders': [{'description': 'take the tablet', 'provenance': [1, 2]},
                                      {'description': 'return later', 'provenance': [1, 3]},
                                      {'description': 'unsupported', 'provenance': []}]}]))
            documents = list(simord_documents(raw, 'train', lambda t: [s + '.' for s in t.split('. ') if s][:3]))
        document = documents[0]
        first, second, third = document['examples']
        self.assertIsNone(first['reject_reason'])
        self.assertEqual(document['text'][first['raw_span'][0]:first['raw_span'][1]],
                         'Take the tablet daily. Then rest until later.')
        self.assertEqual(second['reject_reason'], 'noncontiguous_or_invalid_medical_provenance')
        self.assertEqual(third['reject_reason'], 'noncontiguous_or_invalid_medical_provenance')
        self.assertIsNone(second['raw_span'])
        self.assertIn('take the tablet', first['question'])


class QuarantineTest(unittest.TestCase):
    def test_lexical_index_finds_reused_passages(self):
        protected = ('the patient takes one tablet every morning before breakfast and rests afterwards '
                     'because the clinic advised a slower routine until the follow up appointment next month')
        index = OverlapIndex({'target': protected})
        self.assertEqual(index.matches(protected), ['target'])
        self.assertEqual(index.matches('completely different words about unrelated equipment repairs today '
                                       'that never mention any clinic advice or tablets at all in this text'), [])
        partial = 'unrelated opening line. ' + protected + '. unrelated closing line here'
        self.assertEqual(index.matches(partial, near=True), ['target'])

    def test_bundle_rejects_leakage_fabricated_timestamps_and_foreign_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'target.json', {'records': []})
            bundle = external_bundle(root / 'target.json', [
                ('coqa', 'train', 'Take the tablet daily now', 'When?', [1, 2]),
                ('coqa', 'dev', 'Visit the clinic again later', 'Where?', [1, 2]),
                ('mashqa', 'train', 'Rest until later today please', 'How?', [0, 1]),
                ('mashqa', 'dev', 'Drink water with every tablet', 'What?', [0, 1])])
            validate_external(bundle, root / 'target.json')

            write_json(root / 'other.json', {'records': [1]})
            with self.assertRaisesRegex(ValueError, 'decontaminated'):
                validate_external(bundle, root / 'other.json')

            shared = copy.deepcopy(bundle)
            shared['contexts']['coqa:1']['group'] = shared['contexts']['coqa:0']['group']
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_external(shared, root / 'target.json')

            duplicated = copy.deepcopy(bundle)
            duplicated['contexts']['coqa:1'] = copy.deepcopy(duplicated['contexts']['coqa:0'])
            duplicated['contexts']['coqa:1']['group'] = 'coqa:group1'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_external(duplicated, root / 'target.json')

            timestamps = copy.deepcopy(bundle)
            timestamps['contexts']['coqa:0']['words'] = [{'word': 'Take', 'start': 0, 'end': 1}]
            with self.assertRaisesRegex(ValueError, 'fabricated timestamps'):
                validate_external(timestamps, root / 'target.json')

            unanswerable = copy.deepcopy(bundle)
            unanswerable['examples'][0]['label'] = False
            with self.assertRaisesRegex(ValueError, 'Unanswerable'):
                validate_external(unanswerable, root / 'target.json')

            train_only = copy.deepcopy(bundle)
            train_only['examples'] = [r for r in train_only['examples'] if r['split'] == 'train']
            with self.assertRaisesRegex(ValueError, 'training and development'):
                validate_external(train_only, root / 'target.json')


class MixedSourceTrainingTest(unittest.TestCase):
    """Real CPU training over a synthetic mixture; mechanics only, never an accuracy claim."""

    def setUp(self):
        import torch
        from transformers import BertConfig, BertForQuestionAnswering
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        tokenizer = tiny_tokenizer()
        model = BertForQuestionAnswering(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
        model.save_pretrained(self.root / 'tiny')
        tokenizer.save_pretrained(self.root / 'tiny')
        records = [record(f'c{i}', positive) for i in range(6) for positive in (True, False)]
        write_json(self.root / 'data.json', {'schema_version': 1, 'records': records,
                   'folds': fold_plan(records), 'sources': [],
                   'scope': 'Synthetic mechanics test, not an accuracy benchmark'})
        rows = []
        for index in range(6):
            text = f'Visit {index} take the tablet daily now ' + ' '.join(['filler'] * 20)
            rows.append(('coqa', 'train' if index else 'dev', text, 'Take the tablet daily?', [3, 4]))
            other = f'Take the tablet daily now visit {index} ' + ' '.join(['filler'] * 20)
            rows.append(('mashqa', 'train' if index else 'dev', other, 'Take the tablet daily?', [1, 2]))
            medical = f'Take the tablet now filler visit {index} ' + ' '.join(['filler'] * 20)
            rows.append(('simord', 'train', medical, 'Take the tablet daily?', [1, 2]))
        self.bundle = external_bundle(self.root / 'data.json', rows)
        write_json(self.root / 'external.json', self.bundle)

    def run_training(self, name, *extra):
        from tools.evidence_training.train import main
        main(['--data', str(self.root / 'data.json'), '--external-data', str(self.root / 'external.json'),
              '--output', str(self.root / name), '--model', str(self.root / 'tiny'), '--device', 'cpu',
              '--batch-size', '2', '--accumulation', '1', '--max-length', '96', '--stride', '16',
              '--max-seconds', '300', *extra])
        return self.root / name

    def test_pretraining_uses_both_general_and_medical_article_sources(self):
        out = self.run_training('pretrain', '--mode', 'pretrain', '--epochs', '1', '--evals-per-epoch', '2')
        split = json.loads((out / 'split.json').read_text())
        self.assertEqual(split['sources'], {'train': ['coqa', 'mashqa'], 'dev': ['coqa', 'mashqa']})
        self.assertNotIn('simord', str(split['training_ids']))  # dialogue data is saved for the folds
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual(sorted(report['by_source']), ['coqa', 'mashqa'])
        self.assertAlmostEqual(report['macro_word_iou'],
                               sum(v['positive_word_iou'] for v in report['by_source'].values()) / 2)
        history = json.loads((out / 'history.json').read_text())
        self.assertEqual([h['epoch'] for h in history], [0, 1, 1])  # two evaluations inside the epoch
        self.assertTrue(all(sorted(h['dev_word_iou_by_source']) == ['coqa', 'mashqa'] for h in history))
        self.assertEqual(max(h['dev_macro_word_iou'] for h in history),
                         json.loads((out / 'manifest.json').read_text())['dev_macro_word_iou'])
        self.assertTrue((out / 'model/model.safetensors').exists())

    def test_folds_add_only_medical_dialogue_and_reload_a_pretrained_checkpoint(self):
        checkpoint = self.run_training('pretrain2', '--mode', 'pretrain', '--epochs', '1') / 'model'
        out = self.run_training('cv', '--mode', 'cv', '--epochs', '1', '--fold', '0',
                                '--model', str(checkpoint))
        split = json.loads((out / 'fold_0/split.json').read_text())
        self.assertEqual(len(split['external_medical_ids']), 6)
        self.assertTrue(all(i.startswith('simord') for i in split['external_medical_ids']))
        self.assertEqual(split['questions']['train'], len(split['train']) * 2 + 6)
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual(report['questions'], len(split['test']) * 2)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['external_medical_training_questions'], 6)

    def test_training_data_must_match_the_declared_competition_dataset(self):
        write_json(self.root / 'moved.json', json.loads((self.root / 'data.json').read_text()) | {'sources': [{}]})
        with self.assertRaisesRegex(ValueError, 'decontaminated'):
            from tools.evidence_training.train import main
            main(['--data', str(self.root / 'moved.json'), '--external-data', str(self.root / 'external.json'),
                  '--output', str(self.root / 'guard'), '--model', str(self.root / 'tiny'), '--device', 'cpu',
                  '--mode', 'benchmark', '--max-seconds', '120'])


if __name__ == '__main__':
    unittest.main()
