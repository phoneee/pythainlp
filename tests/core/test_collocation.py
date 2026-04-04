# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from pythainlp.collocation import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)


class TestBigramAssocMeasures(unittest.TestCase):
    """Test all 8 association measures with known values."""

    def test_raw_freq(self):
        self.assertAlmostEqual(
            BigramAssocMeasures.raw_freq(10, 100, 200, 1000), 0.01
        )
        self.assertEqual(BigramAssocMeasures.raw_freq(0, 0, 0, 0), 0.0)

    def test_pmi(self):
        # P(w1,w2) = 10/1000 = 0.01
        # P(w1) = 100/1000 = 0.1, P(w2) = 200/1000 = 0.2
        # PMI = log2(0.01 / (0.1 * 0.2)) = log2(0.5) ≈ -1.0
        score = BigramAssocMeasures.pmi(10, 100, 200, 1000)
        self.assertAlmostEqual(score, math.log2(0.5), places=5)

        # Perfect co-occurrence: n_ii=n_ix=n_xi=n_xx → PMI = log2(1) = 0
        score = BigramAssocMeasures.pmi(100, 100, 100, 100)
        self.assertAlmostEqual(score, 0.0, places=5)

        # Zero count returns -inf
        self.assertEqual(
            BigramAssocMeasures.pmi(0, 100, 200, 1000), -float("inf")
        )

    def test_npmi(self):
        # Range should be [-1, +1]
        score = BigramAssocMeasures.npmi(10, 100, 200, 1000)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

        # Zero count returns -1
        self.assertEqual(
            BigramAssocMeasures.npmi(0, 100, 200, 1000), -1.0
        )

    def test_likelihood_ratio(self):
        # Should be non-negative
        score = BigramAssocMeasures.likelihood_ratio(10, 100, 200, 1000)
        self.assertGreaterEqual(score, 0.0)

        # Zero corpus
        self.assertEqual(
            BigramAssocMeasures.likelihood_ratio(0, 0, 0, 0), 0.0
        )

    def test_chi_sq(self):
        score = BigramAssocMeasures.chi_sq(10, 100, 200, 1000)
        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(BigramAssocMeasures.chi_sq(0, 0, 0, 0), 0.0)

    def test_t_score(self):
        score = BigramAssocMeasures.t_score(10, 100, 200, 1000)
        # Expected = 100*200/1000 = 20, observed = 10
        # t = (10 - 20) / sqrt(10) ≈ -3.162
        expected = (10 - 20) / math.sqrt(10)
        self.assertAlmostEqual(score, expected, places=3)

        self.assertEqual(BigramAssocMeasures.t_score(0, 100, 200, 1000), 0.0)

    def test_dice(self):
        # Dice = 2 * 10 / (100 + 200) = 20/300 ≈ 0.0667
        score = BigramAssocMeasures.dice(10, 100, 200, 1000)
        self.assertAlmostEqual(score, 20 / 300, places=5)
        self.assertEqual(BigramAssocMeasures.dice(0, 0, 0, 0), 0.0)

    def test_log_dice(self):
        score = BigramAssocMeasures.log_dice(10, 100, 200, 1000)
        expected = 14 + math.log2(20 / 300)
        self.assertAlmostEqual(score, expected, places=5)

        # Zero returns -inf
        self.assertEqual(
            BigramAssocMeasures.log_dice(0, 100, 200, 1000), -float("inf")
        )


class TestTrigramAssocMeasures(unittest.TestCase):
    """Trigram measures inherit from bigram — same signatures."""

    def test_inherits_all(self):
        self.assertTrue(hasattr(TrigramAssocMeasures, "pmi"))
        self.assertTrue(hasattr(TrigramAssocMeasures, "likelihood_ratio"))
        self.assertTrue(hasattr(TrigramAssocMeasures, "log_dice"))

    def test_pmi(self):
        score = TrigramAssocMeasures.pmi(5, 50, 100, 500)
        self.assertIsInstance(score, float)


class TestBigramCollocationFinder(unittest.TestCase):
    """Test BigramCollocationFinder construction, filtering, and scoring."""

    def setUp(self):
        self.words = ["ฉัน", "กิน", "ข้าว", "ฉัน", "กิน", "ปลา"]
        self.finder = BigramCollocationFinder.from_words(self.words)

    def test_from_words_counts(self):
        # "ฉัน" appears 2 times, "กิน" appears 2 times
        self.assertEqual(self.finder.word_fd["ฉัน"], 2)
        self.assertEqual(self.finder.word_fd["กิน"], 2)
        # Bigram ("ฉัน", "กิน") appears 2 times
        self.assertEqual(self.finder.ngram_fd[("ฉัน", "กิน")], 2)
        self.assertGreater(self.finder.N, 0)

    def test_from_words_empty(self):
        finder = BigramCollocationFinder.from_words([])
        self.assertEqual(finder.N, 0)
        self.assertEqual(finder.score_ngrams(BigramAssocMeasures.pmi), [])

    def test_from_corpus(self):
        docs = [
            ["ฉัน", "กิน", "ข้าว"],
            ["แมว", "กิน", "ปลา"],
        ]
        finder = BigramCollocationFinder.from_corpus(docs)
        self.assertEqual(finder.word_fd["กิน"], 2)
        self.assertIn(("แมว", "กิน"), finder.ngram_fd)

    def test_apply_freq_filter(self):
        self.finder.apply_freq_filter(2)
        for ng, count in self.finder.ngram_fd.items():
            self.assertGreaterEqual(count, 2)

    def test_apply_word_filter(self):
        self.finder.apply_word_filter(lambda w: w == "ฉัน")
        for w1, w2 in self.finder.ngram_fd:
            self.assertNotEqual(w1, "ฉัน")
            self.assertNotEqual(w2, "ฉัน")

    def test_apply_ngram_filter(self):
        self.finder.apply_ngram_filter(
            lambda w1, w2: w1 == "ฉัน" and w2 == "กิน"
        )
        self.assertNotIn(("ฉัน", "กิน"), self.finder.ngram_fd)

    def test_score_ngrams(self):
        scored = self.finder.score_ngrams(BigramAssocMeasures.raw_freq)
        self.assertIsInstance(scored, list)
        if scored:
            ngram, score = scored[0]
            self.assertIsInstance(ngram, tuple)
            self.assertIsInstance(score, float)
            # Should be sorted descending
            scores = [s for _, s in scored]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_nbest(self):
        top = self.finder.nbest(BigramAssocMeasures.pmi, 3)
        self.assertIsInstance(top, list)
        self.assertLessEqual(len(top), 3)

    def test_above_score(self):
        result = self.finder.above_score(BigramAssocMeasures.raw_freq, 0.0)
        self.assertIsInstance(result, list)

    def test_window_size(self):
        # window_size=3 should find skip-bigrams
        finder = BigramCollocationFinder.from_words(
            ["a", "b", "c", "d"], window_size=3
        )
        self.assertIn(("a", "c"), finder.ngram_fd)


class TestTrigramCollocationFinder(unittest.TestCase):
    def setUp(self):
        self.words = ["ฉัน", "กิน", "ข้าว", "ผัด", "กุ้ง"]
        self.finder = TrigramCollocationFinder.from_words(self.words)

    def test_from_words_counts(self):
        self.assertIn(("ฉัน", "กิน", "ข้าว"), self.finder.ngram_fd)
        self.assertIn(("ข้าว", "ผัด", "กุ้ง"), self.finder.ngram_fd)
        self.assertGreater(self.finder.N, 0)

    def test_from_words_empty(self):
        finder = TrigramCollocationFinder.from_words([])
        self.assertEqual(finder.N, 0)

    def test_from_corpus(self):
        docs = [
            ["ก", "ข", "ค"],
            ["ง", "จ", "ฉ"],
        ]
        finder = TrigramCollocationFinder.from_corpus(docs)
        self.assertIn(("ก", "ข", "ค"), finder.ngram_fd)

    def test_apply_freq_filter(self):
        self.finder.apply_freq_filter(2)
        for ng, count in self.finder.ngram_fd.items():
            self.assertGreaterEqual(count, 2)

    def test_score_ngrams(self):
        scored = self.finder.score_ngrams(TrigramAssocMeasures.raw_freq)
        self.assertIsInstance(scored, list)
        if scored:
            ngram, score = scored[0]
            self.assertEqual(len(ngram), 3)

    def test_nbest(self):
        top = self.finder.nbest(TrigramAssocMeasures.likelihood_ratio, 2)
        self.assertIsInstance(top, list)
        self.assertLessEqual(len(top), 2)
