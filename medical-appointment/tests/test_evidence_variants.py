import copy
import json
import os
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch

from rank_bm25 import BM25Okapi

from pipeline.stage_b import (
    CONVENTIONS,
    _content,
    build_perq_messages,
    render_sentences,
    sentence_ranges,
    sentence_text,
)
from tests.test_stage_b import make_words
from tools.gpu_eval.evidence_variants import (
    DRAFT_INSTRUCTION,
    MAX_DEMO_WORDS,
    EvidenceVariants,
)


def make_item(index):
    conversation = f'c{index:02d}'
    words = make_words(
        f'Earlier {conversation}. Background {conversation}. Before {conversation}. '
        'Your blood pressure is normal, and your pulse is regular. '
        f'Following {conversation}. Later {conversation}. Last {conversation}.'
    )
    first = next(i for i, word in enumerate(words) if word['word'].strip() == 'Your')
    last = first + 4
    row = {'question_id': f'{conversation}_q01', 'transcript_id': conversation,
           'question': 'Was the blood pressure normal?', 'label': '1', 'answer': 'yes',
           'evidence_start': str(words[first]['start']), 'evidence_end': str(words[last]['end'])}
    negative = dict(row, question_id=f'{conversation}_q02', label='0', answer='no',
                    question='Was there a fever?', evidence_start='', evidence_end='')
    return {'id': conversation, 'words': words, 'rows': [row, negative]}


def row_map(items):
    return {row['question_id']: (item, row) for item in items for row in item['rows']}


class EvidenceVariantsTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MEDICAL_ASR_MODE': 'base', 'MEDICAL_EVIDENCE_PROMPT': 'v3'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.items = [make_item(i) for i in range(9)]
        self.question = 'Was the blood pressure normal?'
        self.draft = 'raw DRAFT\n  deliberately  untrimmed'

    def test_control_exactly_matches_existing_builder(self):
        for prompt in ('v3', 'annot'):
            for draft in ('', self.draft):
                with self.subTest(prompt=prompt, draft=draft), patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': prompt}):
                    variant = EvidenceVariants(self.items)
                    messages, donors = variant.messages(self.items[0], self.question, draft)
                    self.assertEqual(messages, build_perq_messages(
                        render_sentences(self.items[0]['words']), self.question, draft,
                    ))
                    self.assertEqual(donors, [])
                    self.assertEqual(variant.manifest['curation_provenance']['prepared_folds'], {})

    def test_no_draft_removes_both_draft_and_keep_fix_instruction(self):
        variant = EvidenceVariants(self.items, 'no-draft')
        messages, donors = variant.messages(self.items[0], self.question, self.draft)
        control = build_perq_messages(render_sentences(self.items[0]['words']), self.question, '')
        self.assertEqual(messages[1], control[1])
        self.assertEqual(messages[0]['content'], control[0]['content'].replace(DRAFT_INSTRUCTION, '', 1))
        self.assertTrue(messages[0]['content'].startswith(CONVENTIONS))
        self.assertIn('Quote transcript words exactly, never paraphrase.', messages[0]['content'])
        self.assertNotIn('draft', json.dumps(messages).lower())
        self.assertEqual(donors, [])
        self.assertEqual(messages, variant.messages(self.items[0], self.question, 'different draft')[0])

    def test_ablations_keep_v3_rules_even_with_another_prompt_environment(self):
        with patch.dict(os.environ, {'MEDICAL_EVIDENCE_PROMPT': 'exchange'}):
            for name in ('no-draft', 'retrieved'):
                with self.subTest(variant=name):
                    variant = EvidenceVariants(self.items, name)
                    messages, _ = variant.messages(self.items[0], self.question, self.draft)
                    self.assertTrue(messages[0]['content'].startswith(CONVENTIONS.split('\nExample transcript:', 1)[0]))
                    self.assertEqual(variant.manifest['config']['base_prompt'], 'v3')

    def test_balanced_folds_depend_only_on_conversation_ids(self):
        items = [make_item(i) for i in range(39)]
        original = EvidenceVariants(items)
        changed = copy.deepcopy(items[::-1])
        for item in changed:
            item['rows'].reverse()
            for row in item['rows']:
                row.update(label='unreadable', evidence_start={'not': 'a timestamp'}, evidence_end=None)
        altered = EvidenceVariants(changed)
        self.assertEqual(original.folds, altered.folds)
        self.assertEqual(Counter(original.folds.values()), {0: 13, 1: 13, 2: 13})
        self.assertNotEqual(original.manifest['data_sha256'], altered.manifest['data_sha256'])

    def test_donor_construction_is_lazy_and_nonretrieval_never_uses_it(self):
        with patch('tools.gpu_eval.evidence_variants._demonstration', side_effect=AssertionError('eager donor access')):
            EvidenceVariants(self.items, 'retrieved')
            for name in ('control', 'no-draft'):
                EvidenceVariants(self.items, name).messages(self.items[0], self.question, self.draft)

    def test_retrieved_excludes_whole_fold_and_never_duplicates_conversations(self):
        variant = EvidenceVariants(self.items, 'retrieved')
        lookup = row_map(self.items)
        for target in self.items:
            with self.subTest(target=target['id']):
                _, donors = variant.messages(target, self.question, self.draft)
                conversations = [lookup[donor][0]['id'] for donor in donors]
                self.assertEqual(len(donors), 2)
                self.assertEqual(len(set(conversations)), 2)
                self.assertTrue(all(variant.folds[c] != variant.folds[target['id']] for c in conversations))
                decisions = variant.manifest['curation_provenance']['prepared_folds'][str(variant.folds[target['id']])]
                self.assertTrue(all(variant.folds[d['conversation']] != variant.folds[target['id']] for d in decisions))

    def test_all_target_fold_labels_answers_and_spans_are_prompt_invariant(self):
        for name in ('control', 'no-draft', 'retrieved'):
            with self.subTest(variant=name):
                original = EvidenceVariants(self.items, name)
                expected = original.messages(self.items[0], self.question, self.draft)
                changed = copy.deepcopy(self.items)
                target_fold = original.folds[self.items[0]['id']]
                for item in changed:
                    if original.folds[item['id']] == target_fold:
                        for row in item['rows']:
                            row.update(label='UNREADABLE_TARGET_LABEL', answer='SECRET_TARGET_ANSWER',
                                       evidence_start={'secret': 'TARGET_GOLD'}, evidence_end=['invalid'])
                altered = EvidenceVariants(changed, name)
                self.assertEqual(altered.messages(changed[0], self.question, self.draft), expected)
                self.assertEqual(altered.folds, original.folds)

    def test_other_target_fold_items_are_excluded_before_word_or_row_processing(self):
        original = EvidenceVariants(self.items, 'retrieved')
        expected = original.messages(self.items[0], self.question, self.draft)
        changed = copy.deepcopy(self.items)
        for item in changed[1:]:
            if original.folds[item['id']] == original.folds[changed[0]['id']]:
                item['words'] = None
                item['rows'] = []
        self.assertEqual(EvidenceVariants(changed, 'retrieved').messages(changed[0], self.question, self.draft), expected)

    def test_retrieved_replaces_fixed_examples_and_keeps_raw_draft(self):
        variant = EvidenceVariants(self.items, 'retrieved')
        messages, donors = variant.messages(self.items[0], self.question, self.draft)
        control = build_perq_messages(render_sentences(self.items[0]['words']), self.question, self.draft)
        self.assertEqual(messages[1], control[1])
        system = messages[0]['content']
        rules, fabricated = CONVENTIONS.split('\nExample transcript:\n', 1)
        self.assertTrue(system.startswith(rules))
        self.assertIn(DRAFT_INSTRUCTION, system)
        self.assertNotIn('Example transcript:', system)
        self.assertNotIn(fabricated.strip(), system)
        self.assertEqual(sum(' -> "' in line for line in system.splitlines()), 2)
        self.assertEqual(len(donors), 2)

    def test_demo_quotes_use_donor_word_boundaries_with_neighboring_context(self):
        variant = EvidenceVariants(self.items, 'retrieved')
        messages, donors = variant.messages(self.items[0], self.question, self.draft)
        decisions = {d['question_id']: d for d in variant.manifest['curation_provenance']['prepared_folds']['0']}
        lookup = row_map(self.items)
        for donor in donors:
            item, row = lookup[donor]
            record = decisions[donor]
            words = item['words']
            quote = sentence_text(words, record['quote_word_range'])
            self.assertEqual(quote, 'Your blood pressure is normal,')
            self.assertEqual(record['quote_span'], [float(row['evidence_start']), float(row['evidence_end'])])
            self.assertIn(f'{row["question"]} -> {json.dumps(quote)}', messages[0]['content'])
            self.assertEqual(record['support_sentence_range'], [3, 3])
            self.assertEqual(record['excerpt_sentence_range'], [1, 5])
            excerpt = render_sentences(words, sentence_ranges(words)[1:6])
            self.assertIn(excerpt, messages[0]['content'])
            self.assertNotIn(f'Earlier {item["id"]}', messages[0]['content'])
            self.assertNotIn(f'Last {item["id"]}', messages[0]['content'])

    def test_bm25_indexes_and_queries_only_question_text(self):
        captures = []
        queries = []

        class RecordingBM25(BM25Okapi):
            def __init__(self, corpus, **kwargs):
                captures.append(corpus)
                super().__init__(corpus, **kwargs)

            def get_scores(self, query):
                queries.append(query)
                return super().get_scores(query)

        with patch('tools.gpu_eval.evidence_variants.BM25Okapi', RecordingBM25):
            variant = EvidenceVariants(self.items, 'retrieved')
            variant.messages(self.items[0], 'Does the patient have migraine?', self.draft)
        self.assertEqual(queries, [_content('Does the patient have migraine?')])
        self.assertEqual(captures, [[_content(item['rows'][0]['question']) for item in self.items
                                     if variant.folds[item['id']] != 0]])

    def test_question_similarity_and_distinct_conversation_selection(self):
        items = copy.deepcopy(self.items)
        items[1]['rows'][0]['question'] = 'Is migraine migraine migraine diagnosed?'
        duplicate = dict(items[1]['rows'][0], question_id='c01_q03')
        items[1]['rows'].append(duplicate)
        items[2]['rows'][0]['question'] = 'Does the patient report migraine?'
        variant = EvidenceVariants(items, 'retrieved')
        _, donors = variant.messages(items[0], 'Is migraine diagnosed?', self.draft)
        self.assertEqual(donors, ['c01_q01', 'c02_q01'])

    def test_ties_are_stable_across_input_and_row_order(self):
        self.items[1]['rows'].append(dict(self.items[1]['rows'][0], question_id='c01_q00'))
        reversed_items = copy.deepcopy(self.items[::-1])
        for item in reversed_items:
            item['rows'].reverse()
        first = EvidenceVariants(self.items, 'retrieved')
        second = EvidenceVariants(reversed_items, 'retrieved')
        expected = first.messages(self.items[0], 'Unmatchedtoken?', self.draft)
        self.assertEqual(expected[1], ['c01_q00', 'c02_q01'])
        self.assertEqual(expected, second.messages(self.items[0], 'Unmatchedtoken?', self.draft))
        self.assertEqual(first.manifest, second.manifest)

    def test_generic_greeting_curation_keeps_rows_and_supports_later_donors(self):
        for greeting in ('Good', 'Hello.', 'Good morning.'):
            with self.subTest(greeting=greeting):
                items = copy.deepcopy(self.items)
                item = items[1]
                item['words'] = make_words(greeting + ' Welcome to this appointment.')
                last = len(greeting.split()) - 1
                item['rows'][0].update(evidence_start='0.0', evidence_end=str(item['words'][last]['end']))
                before = copy.deepcopy(items)
                variant = EvidenceVariants(items, 'retrieved')
                _, donors = variant.messages(items[0], self.question, self.draft)
                decisions = {d['question_id']: d for d in variant.manifest['curation_provenance']['prepared_folds']['0']}
                self.assertNotIn(item['rows'][0]['question_id'], donors)
                self.assertEqual(decisions[item['rows'][0]['question_id']]['reason'], 'greeting_only_quote')
                self.assertEqual(items, before)

    def test_demo_context_is_bounded_and_oversized_support_is_not_truncated(self):
        items = copy.deepcopy(self.items)
        for index in (1, 2):
            item = items[index]
            words = make_words('Context ' * 200 + 'ends. Your pressure is normal. Afterward.')
            item['words'] = words
            first = 0 if index == 1 else 201
            last = 200 if index == 1 else 204
            item['rows'][0].update(evidence_start=str(words[first]['start']), evidence_end=str(words[last]['end']))
        variant = EvidenceVariants(items, 'retrieved')
        variant.messages(items[0], self.question, self.draft)
        decisions = {d['question_id']: d for d in variant.manifest['curation_provenance']['prepared_folds']['0']}
        self.assertEqual(decisions['c01_q01']['reason'], 'support_sentences_exceed_word_limit')
        self.assertEqual(decisions['c02_q01']['reason'], 'usable')
        self.assertEqual(decisions['c02_q01']['excerpt_sentence_range'], [1, 2])
        for record in decisions.values():
            if record['reason'] == 'usable':
                lo, hi = record['excerpt_word_range']
                self.assertLessEqual(hi - lo + 1, MAX_DEMO_WORDS)
                a, b = record['quote_word_range']
                self.assertLessEqual(lo, a)
                self.assertGreaterEqual(hi, b)

    def test_retrieved_fails_explicitly_without_two_usable_other_conversations(self):
        for count in (1, 2):
            with self.subTest(conversations=count):
                items = copy.deepcopy(self.items[:count])
                items[-1]['rows'].append(dict(items[-1]['rows'][0], question_id='another_positive'))
                variant = EvidenceVariants(items, 'retrieved')
                with self.assertRaisesRegex(ValueError, 'two usable donor conversations outside scored fold'):
                    variant.messages(items[0], self.question, self.draft)

    def test_manifest_is_deterministic_and_json_serializable(self):
        first = EvidenceVariants(self.items, 'retrieved')
        second = EvidenceVariants(copy.deepcopy(self.items), 'retrieved')
        self.assertEqual(first.manifest, json.loads(json.dumps(second.manifest)))
        for item in self.items:
            first.messages(item, self.question, self.draft)
        for item in self.items[::-1]:
            second.messages(item, self.question, self.draft)
        self.assertEqual(json.dumps(first.manifest), json.dumps(second.manifest))
        self.assertEqual(set(first.manifest['curation_provenance']['prepared_folds']), {'0', '1', '2'})
        self.assertEqual(first.manifest['folds'], first.folds)
        self.assertEqual(first.manifest['config']['max_demo_words'], MAX_DEMO_WORDS)

    def test_invalid_variant_and_duplicate_conversation_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'unknown evidence variant'):
            EvidenceVariants(self.items, 'unapproved')
        with self.assertRaisesRegex(ValueError, 'conversation IDs must be unique'):
            EvidenceVariants([self.items[0], self.items[0]])

    def test_real_cached_dataset_has_balanced_folds_and_curates_only_bogus_greetings(self):
        path = Path(__file__).resolve().parents[1] / 'tools' / 'gpu_eval' / 'inputs_base.json'
        items = json.loads(path.read_text())
        before = copy.deepcopy(items)
        variant = EvidenceVariants(items, 'retrieved')
        self.assertEqual(Counter(variant.folds.values()), {0: 13, 1: 13, 2: 13})
        lookup = row_map(items)
        for item in items:
            for row in item['rows']:
                _, donors = variant.messages(item, row['question'], '')
                conversations = [lookup[donor][0]['id'] for donor in donors]
                self.assertEqual(len(set(conversations)), 2)
                self.assertTrue(all(variant.folds[c] != variant.folds[item['id']] for c in conversations))
        decisions = [record for fold in variant.manifest['curation_provenance']['prepared_folds'].values() for record in fold]
        greetings = {record['question_id'] for record in decisions if record['reason'] == 'greeting_only_quote'}
        self.assertEqual(greetings, {'sample_63_yes_q02', 'sample_64_yes_q02'})
        self.assertEqual({record['reason'] for record in decisions}, {'not_positive', 'usable', 'greeting_only_quote'})
        self.assertEqual(sum(len(item['rows']) for item in items), 390)
        self.assertEqual(items, before)


if __name__ == '__main__':
    unittest.main()
