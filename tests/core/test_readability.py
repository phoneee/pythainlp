# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0

import unittest

from pythainlp.readability import (
    ReadabilityResult,
    automated_readability_index,
    average_sentence_length,
    average_syllables_per_word,
    average_word_length,
    coleman_liau_index,
    consonant_cluster_ratio,
    dead_syllable_ratio,
    karan_density,
    lix,
    rare_word_ratio,
    readability_score,
    tone_mark_density,
    type_token_ratio,
    word_frequency_score,
)

try:
    import pycrfsuite  # noqa: F401

    _HAS_CRFSUITE = True
except ImportError:
    _HAS_CRFSUITE = False

_SKIP_MSG = "python-crfsuite not installed (needed by crfcut/han_solo)"

# Pre-tokenized fixtures (no crfsuite needed)
_EASY_WORDS = ["แมว", "กิน", "ปลา", "แมว", "นอน", "หลับ"]
_EASY_SENTS = ["แมวกินปลา", "แมวนอนหลับ"]
_EASY_SYLLS = ["แมว", "กิน", "ปลา", "แมว", "นอน", "หลับ"]

_HARD_WORDS = [
    "ปรัชญา", "เศรษฐกิจ", "พอเพียง", "เป็น", "แนวคิด", "ที่",
    "พระบาท", "สมเด็จ", "พระเจ้าอยู่หัว", "ทรง", "พระราชดำริ",
]
_HARD_SENTS = [
    "ปรัชญาเศรษฐกิจพอเพียงเป็นแนวคิดที่พระบาทสมเด็จพระเจ้าอยู่หัวทรงพระราชดำริ"
]
_HARD_SYLLS = [
    "ปรัช", "ญา", "เศรษ", "ฐ", "กิจ", "พอ", "เพียง", "เป็น",
    "แนว", "คิด", "ที่", "พระ", "บาท", "สม", "เด็จ", "พระ",
    "เจ้า", "อยู่", "หัว", "ทรง", "พระ", "ราช", "ดำ", "ริ",
]

# Raw text fixtures (need crfsuite for sent/syllable tokenization)
_EASY_TEXT = "แมวกินปลา แมวนอนหลับ"
_MEDIUM_TEXT = "นักเรียนทุกคนต้องทำการบ้านให้เสร็จก่อนเวลาที่กำหนด"
_HARD_TEXT = (
    "ปรัชญาเศรษฐกิจพอเพียงเป็นแนวคิดที่พระบาทสมเด็จพระเจ้าอยู่หัว"
    "ทรงพระราชดำริเพื่อชี้แนะแนวทางการดำเนินชีวิตแก่ประชาชน"
)


class TestTier1Statistics(unittest.TestCase):
    """Tier 1: Core statistical metrics."""

    def test_average_sentence_length_pre_tokenized(self):
        score = average_sentence_length(
            words=["แมว", "กิน", "ปลา"],
            sentences=["แมวกินปลา"],
        )
        self.assertAlmostEqual(score, 3.0)

    def test_average_sentence_length_empty(self):
        self.assertEqual(average_sentence_length(""), 0.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_average_sentence_length_raw(self):
        score = average_sentence_length(_EASY_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_average_word_length(self):
        score = average_word_length(_EASY_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_average_word_length_empty(self):
        self.assertEqual(average_word_length(""), 0.0)

    def test_average_syllables_per_word_pre_tokenized(self):
        score = average_syllables_per_word(
            words=_EASY_WORDS, syllables=_EASY_SYLLS
        )
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_average_syllables_per_word_empty(self):
        self.assertEqual(average_syllables_per_word(""), 0.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_average_syllables_per_word_raw(self):
        score = average_syllables_per_word(_MEDIUM_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_type_token_ratio(self):
        score = type_token_ratio(_EASY_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_type_token_ratio_all_unique(self):
        score = type_token_ratio(words=["ก", "ข", "ค"])
        self.assertAlmostEqual(score, 1.0)

    def test_type_token_ratio_empty(self):
        self.assertEqual(type_token_ratio(""), 0.0)

    def test_word_frequency_score(self):
        score = word_frequency_score(_EASY_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_word_frequency_score_ttc(self):
        score = word_frequency_score(_EASY_TEXT, freq_corpus="ttc")
        self.assertIsInstance(score, float)

    def test_word_frequency_score_empty(self):
        self.assertEqual(word_frequency_score(""), 0.0)

    def test_rare_word_ratio(self):
        score = rare_word_ratio(_EASY_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_rare_word_ratio_empty(self):
        self.assertEqual(rare_word_ratio(""), 0.0)

    def test_dead_syllable_ratio_pre_tokenized(self):
        score = dead_syllable_ratio(syllables=_EASY_SYLLS)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_dead_syllable_ratio_empty(self):
        self.assertEqual(dead_syllable_ratio(""), 0.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_dead_syllable_ratio_raw(self):
        score = dead_syllable_ratio(_EASY_TEXT)
        self.assertIsInstance(score, float)


class TestTier2Character(unittest.TestCase):
    """Tier 2: Thai character complexity metrics."""

    def test_tone_mark_density(self):
        score = tone_mark_density("น้ำตกสวยมาก")
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_tone_mark_density_no_marks(self):
        score = tone_mark_density("กขคง")
        self.assertEqual(score, 0.0)

    def test_tone_mark_density_empty(self):
        self.assertEqual(tone_mark_density(""), 0.0)

    def test_consonant_cluster_ratio_pre_tokenized(self):
        score = consonant_cluster_ratio(words=["กรุงเทพ", "เป็น", "เมือง"])
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_consonant_cluster_ratio_empty(self):
        self.assertEqual(consonant_cluster_ratio(""), 0.0)

    def test_karan_density_pre_tokenized(self):
        score = karan_density(words=["ภูมิศาสตร์", "และ", "วิทยาศาสตร์"])
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_karan_density_no_karan(self):
        score = karan_density(words=["แมว", "กิน", "ปลา"])
        self.assertEqual(score, 0.0)

    def test_karan_density_empty(self):
        self.assertEqual(karan_density(""), 0.0)


class TestTier3Formulas(unittest.TestCase):
    """Tier 3: Adapted international readability formulas."""

    def test_ari_pre_tokenized(self):
        score = automated_readability_index(
            words=["ฉัน", "ไป", "โรงเรียน"],
            sentences=["ฉันไปโรงเรียน"],
        )
        self.assertIsInstance(score, float)

    def test_ari_empty(self):
        self.assertEqual(automated_readability_index(""), 0.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_ari_raw(self):
        score = automated_readability_index(_MEDIUM_TEXT)
        self.assertIsInstance(score, float)

    def test_coleman_liau_pre_tokenized(self):
        score = coleman_liau_index(
            words=["แมว", "กิน", "ปลา"],
            sentences=["แมวกินปลา"],
        )
        self.assertIsInstance(score, float)

    def test_coleman_liau_empty(self):
        self.assertEqual(coleman_liau_index(""), 0.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_coleman_liau_raw(self):
        score = coleman_liau_index(_MEDIUM_TEXT)
        self.assertIsInstance(score, float)

    def test_lix_pre_tokenized(self):
        score = lix(
            words=["นักเรียน", "ทุกคน", "ต้อง", "ทำ"],
            sentences=["นักเรียนทุกคนต้องทำ"],
        )
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_lix_empty(self):
        self.assertEqual(lix(""), 0.0)

    def test_lix_custom_threshold(self):
        score = lix(
            words=["นักเรียน", "ทุก", "คน"],
            sentences=["นักเรียนทุกคน"],
            long_word_threshold=3,
        )
        self.assertIsInstance(score, float)


class TestCompositeReadabilityScore(unittest.TestCase):
    """Test the composite readability_score function."""

    def test_basic_result_structure(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        self.assertIn("score", result)
        self.assertIn("level", result)
        self.assertIn("level_disclaimer", result)
        self.assertIn("stats", result)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["level"], str)
        self.assertIsInstance(result["stats"], dict)

    def test_score_range(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_easy_vs_hard(self):
        easy = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        hard = readability_score(
            _HARD_TEXT,
            words=_HARD_WORDS,
            sentences=_HARD_SENTS,
            syllables=_HARD_SYLLS,
        )
        self.assertLess(easy["score"], hard["score"])

    def test_empty_input(self):
        result = readability_score("")
        self.assertEqual(result["score"], 0.0)

    def test_grade_levels_exist(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        valid_levels = {"ป.1-3", "ป.4-6", "ม.1-3", "ม.4-6", "อุดมศึกษา"}
        self.assertIn(result["level"], valid_levels)

    def test_custom_weights(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
            weights={
                "dead_syllable_ratio": 0.5,
                "rare_word_ratio": 0.5,
            },
        )
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_custom_grade_thresholds(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
            grade_thresholds=[
                (50.0, "Easy"),
                (100.0, "Hard"),
            ],
        )
        self.assertIn(result["level"], {"Easy", "Hard"})

    def test_stats_contains_all_metrics(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        expected_keys = {
            "avg_sentence_length",
            "avg_word_length",
            "avg_syllables_per_word",
            "type_token_ratio",
            "word_frequency_score",
            "rare_word_ratio",
            "dead_syllable_ratio",
            "tone_mark_density",
            "consonant_cluster_ratio",
            "karan_density",
            "ari",
            "coleman_liau",
            "lix",
        }
        self.assertEqual(set(result["stats"].keys()), expected_keys)

    def test_disclaimer_present(self):
        result = readability_score(
            _EASY_TEXT,
            words=_EASY_WORDS,
            sentences=_EASY_SENTS,
            syllables=_EASY_SYLLS,
        )
        self.assertTrue(len(result["level_disclaimer"]) > 0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_readability_score_raw_text(self):
        result = readability_score(_MEDIUM_TEXT)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    @unittest.skipUnless(_HAS_CRFSUITE, _SKIP_MSG)
    def test_easy_vs_hard_raw(self):
        easy = readability_score(_EASY_TEXT)
        hard = readability_score(_HARD_TEXT)
        self.assertLess(easy["score"], hard["score"])
