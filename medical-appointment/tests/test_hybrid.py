import unittest

from pipeline.hybrid import carry_span, span_indices, word_map
from tests.test_stage_b import make_words


def retimed(words, offset=0.0, scale=1.0):
    return [{'word': w['word'], 'start': w['start'] * scale + offset,
             'end': w['end'] * scale + offset} for w in words]


class WordMapTests(unittest.TestCase):
    def test_identical_transcripts_map_one_to_one(self):
        words = make_words('The dose is 100 mg daily.')
        mapping = word_map(words, retimed(words, 5.0))
        self.assertEqual(mapping, {i: i for i in range(len(words))})

    def test_misspellings_still_align_their_neighbours(self):
        source = make_words('She takes Airomir and Esomeprazole every day.')
        target = make_words('She takes Aromere and Isomeprazol every day.')
        mapping = word_map(source, target)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[len(source) - 1], len(target) - 1)
        self.assertTrue(all(0 <= v < len(target) for v in mapping.values()))

    def test_insertions_do_not_collapse_positions(self):
        source = make_words('one two three four five')
        target = make_words('one two extra words here three four five')
        mapping = word_map(source, target)
        self.assertEqual(mapping[0], 0)
        self.assertLess(mapping[2], mapping[4])
        self.assertEqual(mapping[4], len(target) - 1)

    def test_empty_input(self):
        self.assertEqual(word_map([], make_words('a b')), {})
        self.assertEqual(word_map(make_words('a b'), []), {})


class CarryTests(unittest.TestCase):
    def test_a_span_lands_on_the_matching_words_of_the_other_transcript(self):
        source = make_words('Morning. The dose is 100 mg daily. Take it after a meal.')
        target = retimed(make_words('Morning. The dose is 100 mg daily. Take it after a meal.'), offset=3.0)
        span = (source[2]['start'], source[6]['end'])
        carried = carry_span(span, source, target)
        self.assertAlmostEqual(carried[0], target[2]['start'])
        self.assertAlmostEqual(carried[1], target[6]['end'])

    def test_missing_or_empty_spans(self):
        source = make_words('a b c')
        target = make_words('a b c')
        self.assertIsNone(carry_span(None, source, target))
        self.assertIsNone(carry_span((None, None), source, target))
        self.assertIsNone(carry_span((900.0, 901.0), source, target))

    def test_span_indices_cover_only_overlapping_words(self):
        words = make_words('a b c d')
        self.assertEqual(span_indices(words, (words[1]['start'], words[2]['end'])), [1, 2])
        self.assertEqual(span_indices(words, None), [])


if __name__ == '__main__':
    unittest.main()
