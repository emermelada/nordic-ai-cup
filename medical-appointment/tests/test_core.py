import json
import math
import unittest
from unittest.mock import patch

from dtos import ASRQuestionResponseDto
from pipeline.core import (
    SYSTEM, answer_response, build_messages, floor_response, fuzzy_span,
    retrieval_response, sanitize_response, split_units, words_from_transcript,
)


def make_words(text, start=1.0, step=0.25):
    return [
        {"word": " " + token, "start": start + i * step,
         "end": start + (i + 1) * step, "p": 0.9}
        for i, token in enumerate(text.split())
    ]


def response(answers, starts=None, ends=None):
    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=starts if starts is not None else [None] * len(answers),
        evidence_end=ends if ends is not None else [None] * len(answers),
    )


def raw_results(*entries):
    return json.dumps({"results": list(entries)})


class TranscriptTests(unittest.TestCase):
    def test_normalize_legacy_words_without_losing_spacing_or_zero_lengths(self):
        transcript = {"segments": [
            {"words": [
                {"word": " Hello", "start": "0", "end": "0", "p": 0.8},
                {"word": "  world.", "start": -0.1, "end": 0.3, "probability": 0.7},
                {"word": " bad", "start": 2, "end": 1},
                {"word": " nan", "start": float("nan"), "end": 2},
                {"word": " infinity", "start": 1, "end": float("inf")},
                {"word": " bool", "start": True, "end": 2},
                {"word": " missing", "start": 1},
                {"word": 12, "start": 1, "end": 2},
                {"word": " ", "start": 1, "end": 2}, None,
            ]},
            {"words": None}, None,
        ]}
        words = words_from_transcript(transcript)
        self.assertEqual(words, [
            {"word": " Hello", "start": 0.0, "end": 0.0, "p": 0.8},
            {"word": "  world.", "start": 0.0, "end": 0.3, "p": 0.7},
        ])
        for invalid in (None, {}, {"segments": None}, {"segments": [42]}):
            self.assertEqual(words_from_transcript(invalid), [])

    def test_splitter_short_merge_and_exact_prompt_spacing(self):
        words = make_words("Yes. Take this medicine daily. Okay? See you next week.")
        words[1]["word"] = "  Take"
        units = split_units(words)
        self.assertEqual([len(unit) for unit in units], [5, 5])
        messages = build_messages(words, ["Take medicine?", "Come back?"])
        self.assertEqual(messages, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                "TRANSCRIPT\n[0] Yes.  Take this medicine daily.\n"
                "[1] Okay? See you next week.\n\n"
                "QUESTIONS\n1. Take medicine?\n2. Come back?\n\nJSON:"
            )},
        ])
        self.assertIn('A near-miss is "no"', SYSTEM)
        self.assertIn('"q":1,"answer":"yes","confidence":90', SYSTEM)
        self.assertIs(units[0][0], words[0])

    def test_splitter_pause_comma_and_maximum_boundaries(self):
        comma = make_words("one two three four five six, seven eight nine")
        self.assertEqual([len(unit) for unit in split_units(comma)], [6, 3])
        short_comma = make_words("one two, three four five six")
        self.assertEqual([len(unit) for unit in split_units(short_comma)], [6])
        long = make_words(" ".join(["word"] * 20))
        self.assertEqual([len(unit) for unit in split_units(long)], [18, 2])
        paused = make_words("one two three four five six")
        for word in paused[3:]:
            word["start"] += 0.5
            word["end"] += 0.5
        self.assertEqual([len(unit) for unit in split_units(paused)], [3, 3])
        short = make_words("Yes. Continue treatment today.")
        for word in short[1:]:
            word["start"] += 1
            word["end"] += 1
        self.assertEqual([len(unit) for unit in split_units(short)], [1, 3])
        self.assertEqual(split_units([]), [])


class AlignmentTests(unittest.TestCase):
    def test_global_quote_punctuation_and_earliest_tie(self):
        words = make_words("We prescribed metformin, 2.5 mg. We prescribed metformin 2.5 mg.")
        self.assertEqual(
            fuzzy_span(words, '"prescribed metformin 2.5 mg!"'),
            (words[1]["start"], words[4]["end"]),
        )
        raw = raw_results({"q": 1, "answer": "yes", "quote": "prescribed metformin 2.5 mg", "units": [1]})
        actual = answer_response(raw, words, ["Treatment?"], 20, fallback=response([False]))
        self.assertEqual(actual.evidence_start, [words[1]["start"]])
        self.assertEqual(actual.evidence_end, [words[4]["end"]])

    def test_multiple_tokens_in_one_word_and_apostrophes(self):
        words = make_words("Your long-term sugar isn't elevated.")
        words[1]["word"] = " long term"
        self.assertEqual(
            fuzzy_span(words, "long term sugar isn't elevated"),
            (words[1]["start"], words[-1]["end"]),
        )
        self.assertIsNone(fuzzy_span(words, "!!!"))
        self.assertIsNone(fuzzy_span([], "treatment"))

    def test_threshold_and_ties_keep_shortest_first_window(self):
        words = make_words("one two three four five six seven eight")
        with patch("pipeline.core.fuzz.ratio", return_value=70) as ratio:
            self.assertEqual(fuzzy_span(words, "one two three four"), (words[0]["start"], words[0]["end"]))
        first_windows = [call.args[0] for call in ratio.call_args_list[:7]]
        self.assertEqual(first_windows, [" ".join("one two three four five six seven".split()[:i]) for i in range(1, 8)])
        with patch("pipeline.core.fuzz.ratio", return_value=69.99):
            self.assertIsNone(fuzzy_span(words, "one two three four"))

    def test_expiring_deadline_discards_partial_fuzzy_match(self):
        words = make_words(" ".join(["word"] * 20))
        with patch("pipeline.core.time.monotonic", side_effect=[0] * 26 + [2]):
            with patch("pipeline.core.fuzz.ratio", return_value=80) as ratio:
                self.assertIsNone(fuzzy_span(words, "one two three four", deadline=1))
        self.assertEqual(ratio.call_count, 5)


class RetrievalTests(unittest.TestCase):
    def test_decimal_numbers_are_contextual_not_pooled_across_units(self):
        words = make_words(
            "Metformin was continued at 2.5 milligrams daily. "
            "Bisoprolol was continued at 5 milligrams daily."
        )
        result = retrieval_response(words, [
            "Was metformin continued at 2.50 milligrams daily?",
            "Was metformin continued at 5 milligrams daily?",
            "Was metformin continued at 25 milligrams daily?",
        ], 20)
        self.assertEqual(result.answers, [True, False, False])
        self.assertEqual(result.evidence_start, [words[0]["start"], None, None])
        self.assertEqual(result.evidence_end, [words[6]["end"], None, None])

    def test_trailing_punctuation_and_small_corpus(self):
        words = make_words("The fasting glucose level was 6.2.")
        result = retrieval_response(words, [
            "Was the fasting glucose level 6.2?", "Was the fasting glucose level 62?",
            "Was cataract surgery scheduled?",
        ], 20)
        self.assertEqual(result.answers, [True, False, False])

    def test_empty_corpus_and_no_questions(self):
        for words in ([], make_words("..."), make_words("the and of")):
            with self.subTest(words=words):
                self.assertEqual(retrieval_response(words, ["Anything?"], 1), floor_response(1))
        self.assertEqual(retrieval_response(make_words("Hello"), [], 1), floor_response(0))

    def test_bm25_branch_and_duration_clamp(self):
        words = make_words(
            "Metformin daily medication was prescribed. "
            "The patient reports sleeping soundly. "
            "Exercise includes swimming every weekend."
        )
        with patch("pipeline.core.BM25Okapi") as bm25:
            bm25.return_value.get_scores.return_value = [3.0, 0.0, 0.0]
            result = retrieval_response(words, ["Metformin daily medication?"], 1.5)
        self.assertEqual(result.answers, [True])
        self.assertEqual(result.evidence_start, [1.0])
        self.assertEqual(result.evidence_end, [1.5])


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.words = make_words(
            "Continue the treatment today. Return for review tomorrow. Keep your regular appointments."
        )

    def test_valid_answers_preserve_order_and_ignore_confidence(self):
        raw = raw_results(
            {"q": 3, "answer": "yes", "confidence": 0, "quote": "Return for review tomorrow"},
            {"q": "1", "answer": " NO ", "confidence": 100, "units": [0]},
            {"q": 2, "answer": True, "confidence": "nonsense", "units": [0]},
        )
        result = answer_response(raw, self.words, ["a", "b", "c"], 20, response([True, False, False]))
        self.assertEqual(result.answers, [False, True, True])
        self.assertEqual(result.evidence_start, [None, self.words[0]["start"], self.words[4]["start"]])
        self.assertEqual(result.evidence_end, [None, self.words[3]["end"], self.words[7]["end"]])

    def test_duplicates_missing_and_invalid_entries_fall_back_independently(self):
        fallback = response([False, True, False, True, False, True], [None, 1, None, 2, None, 3], [None, 2, None, 3, None, 4])
        raw = raw_results(
            {"q": 1, "answer": "yes"}, {"q": "1", "answer": "no"}, {"q": 1, "answer": "yes"},
            {"q": 2, "answer": "yes, definitely"},
            {"q": 4, "answer": "no", "quote": "Continue the treatment today"},
            {"q": 5, "answer": "yes", "units": "invalid", "quote": {}},
            {"q": 6, "answer": 1},
            {"q": 0, "answer": "no"}, {"q": True, "answer": "no"},
            {"q": 3.0, "answer": "yes"}, {"q": "-1", "answer": "no"},
            {"q": 99, "answer": "no"}, {"answer": "no"}, None, 42, [],
        )
        result = answer_response(raw, self.words, list("abcdef"), 20, fallback)
        self.assertEqual(result.answers, [False, True, False, False, True, True])
        self.assertEqual(result.evidence_start, [None, 1, None, None, None, 3])
        self.assertEqual(result.evidence_end, [None, 2, None, None, None, 4])
        self.assertEqual(fallback.answers, [False, True, False, True, False, True])

    def test_repair_fenced_malformed_json_retains_valid_entries(self):
        raw = '```json\n{"results":[{"q":1,"answer":"no"},{"q":2,"answer":"yes",},null,]}\n```'
        result = answer_response(raw, self.words, ["a", "b"], 20, response([True, False]))
        self.assertEqual(result.answers, [False, True])
        self.assertEqual(result.evidence_start, [None, None])

    def test_reasoning_channel_output_is_parsed(self):
        raw = ('<|channel|>analysis<|message|>Weighing the wording.<|end|>'
               '<|start|>assistant<|channel|>final<|message|>{"results":[{"q":1,"answer":"no"},'
               '{"q":2,"answer":"yes"}]}')
        result = answer_response(raw, self.words, ["a", "b"], 20, response([True, False]))
        self.assertEqual(result.answers, [False, True])

    def test_invalid_whole_outputs_use_retrieval(self):
        fallback = response([False, True], [None, 2], [None, 3])
        for raw in ("", "not json", "null", "[]", '{"results":{}}', '{"results":[null,42]}'):
            with self.subTest(raw=raw):
                self.assertEqual(answer_response(raw, self.words, ["a", "b"], 20, fallback), fallback)
        with patch("pipeline.core.retrieval_response", return_value=fallback) as retrieval:
            self.assertEqual(answer_response("", self.words, ["a", "b"], 20), fallback)
        retrieval.assert_called_once()

    def test_invalid_evidence_never_replaces_valid_yes_or_no(self):
        fallback = response([True, False, True], [2, None, 3], [3, None, 4])
        raw = raw_results(
            {"q": 1, "answer": "yes", "quote": [], "units": [False]},
            {"q": 2, "answer": "yes", "quote": None, "units": [999]},
            {"q": 3, "answer": False, "quote": "Continue the treatment today", "units": [0]},
        )
        result = answer_response(raw, self.words, ["a", "b", "c"], 20, fallback)
        self.assertEqual(result.answers, [True, True, False])
        self.assertEqual(result.evidence_start, [2, None, None])
        self.assertEqual(result.evidence_end, [3, None, None])

    def test_cited_units_then_retrieval_evidence(self):
        for ids in ([True], [0.0], ["0"], [0, 2], [0, 0], [0, 1, 2], [-1], None):
            with self.subTest(ids=ids):
                result = answer_response(
                    raw_results({"q": 1, "answer": "yes", "units": ids}),
                    self.words, ["a"], 20, response([True], [2], [3]),
                )
                self.assertEqual(result.evidence_start, [2])
                self.assertEqual(result.evidence_end, [3])
        result = answer_response(
            raw_results({"q": 1, "answer": "yes", "units": [1, 0]}),
            self.words, ["a"], 20, response([False]),
        )
        self.assertEqual(result.evidence_start, [self.words[0]["start"]])
        self.assertEqual(result.evidence_end, [self.words[7]["end"]])

    def test_no_default_offset_and_explicit_offset(self):
        raw = raw_results({"q": 1, "answer": "yes", "quote": "Continue the treatment today"})
        baseline = answer_response(raw, self.words, ["a"], 1.9, response([False]))
        shifted = answer_response(raw, self.words, ["a"], 1.9, response([False]), start_offset=0.1)
        self.assertEqual(baseline.evidence_start, [1.0])
        self.assertEqual(shifted.evidence_start, [1.1])
        self.assertEqual(baseline.evidence_end, [1.9])
        self.assertEqual(shifted.evidence_end, [1.9])

    def test_zero_length_quote_uses_valid_unit(self):
        self.words[1]["end"] = self.words[1]["start"]
        raw = raw_results({"q": 1, "answer": "yes", "quote": "the", "units": [0]})
        result = answer_response(raw, self.words, ["a"], 20, response([False]))
        self.assertEqual(result.evidence_start, [self.words[0]["start"]])
        self.assertEqual(result.evidence_end, [self.words[3]["end"]])

    def test_deadline_expired_and_expiry_during_alignment(self):
        fallback = response([False])
        raw = raw_results({"q": 1, "answer": "yes", "units": [0]})
        with patch("pipeline.core.fuzzy_span") as align:
            self.assertEqual(answer_response(raw, self.words, ["a"], 20, fallback, deadline=0), fallback)
        align.assert_not_called()
        with patch("pipeline.core._expired", side_effect=[False, True]):
            with patch("pipeline.core.fuzzy_span", return_value=None):
                result = answer_response(raw, self.words, ["a"], 20, fallback, deadline=1)
        self.assertEqual(result, response([True]))

    def test_empty_asr_always_returns_floor(self):
        raw = raw_results({"q": 1, "answer": "no"})
        self.assertEqual(answer_response(raw, [], ["a"], 20, response([False])), floor_response(1))
        self.assertEqual(answer_response(raw, self.words, [], 20), floor_response(0))


class SanitizationTests(unittest.TestCase):
    def test_strict_booleans_paired_spans_padding_and_bounds(self):
        actual = sanitize_response({
            "answers": [True, False, "no", True, True, True, True, True, True],
            "evidence_start": [-1, 1, 1, float("nan"), 4, 3, 10, 1, True],
            "evidence_end": [20, 2, 2, 2, 3, 3, 11, float("inf"), 2],
        }, 10, 5)
        self.assertEqual(actual.answers, [True, False, True, True, True, True, True, True, True, True])
        self.assertEqual(actual.evidence_start, [0.0] + [None] * 9)
        self.assertEqual(actual.evidence_end, [5.0] + [None] * 9)
        self.assertEqual(sanitize_response(actual, 1, 5), response([True], [0], [5]))
        self.assertEqual(sanitize_response(None, 2), floor_response(2))

    def test_duration_and_unpaired_spans(self):
        original = response([True], [-1], [2])
        self.assertEqual(sanitize_response(original, 1), response([True], [0], [2]))
        for duration in (0, -1, float("inf"), float("nan")):
            with self.subTest(duration=duration):
                self.assertEqual(sanitize_response(original, 1, duration), floor_response(1))
        for start, end in (([], [2]), ([1], []), ([None], [2]), ([1], [None])):
            actual = sanitize_response({"answers": [True], "evidence_start": start, "evidence_end": end}, 1, 3)
            self.assertEqual(actual, floor_response(1))
        self.assertTrue(all(value is None or math.isfinite(value) for value in sanitize_response(original, 1).evidence_end))


if __name__ == "__main__":
    unittest.main()
