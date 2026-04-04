# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Collocation finders for Thai text.

Provides BigramCollocationFinder and TrigramCollocationFinder
that follow the Build-Filter-Score pattern.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

__all__: list[str] = [
    "BigramCollocationFinder",
    "TrigramCollocationFinder",
]


class BigramCollocationFinder:
    """Find and rank bigram collocations from Thai text.

    Build a finder, apply filters, then score with an association measure.

    :param dict word_fd: unigram frequency distribution
    :param dict ngram_fd: bigram frequency distribution

    :Example:
    ::

        from pythainlp.tokenize import word_tokenize
        from pythainlp.collocation import (
            BigramCollocationFinder,
            BigramAssocMeasures,
        )

        words = word_tokenize("ฉันกินข้าวที่โรงเรียนทุกวัน")
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    """

    def __init__(
        self,
        word_fd: dict[str, int],
        ngram_fd: dict[tuple[str, str], int],
    ) -> None:
        self.word_fd: dict[str, int] = dict(word_fd)
        self.ngram_fd: dict[tuple[str, str], int] = dict(ngram_fd)
        self._recompute_n()

    def _recompute_n(self) -> None:
        self.N: int = sum(self.ngram_fd.values())

    @classmethod
    def from_words(
        cls,
        words: list[str],
        window_size: int = 2,
    ) -> BigramCollocationFinder:
        """Construct from a pre-tokenized word list.

        :param list[str] words: pre-tokenized Thai words
        :param int window_size: window for co-occurrence
            (2 = adjacent only, >2 = skip-bigrams)
        :return: BigramCollocationFinder
        :rtype: BigramCollocationFinder

        :Example:
        ::

            from pythainlp.tokenize import word_tokenize
            from pythainlp.collocation import BigramCollocationFinder

            words = word_tokenize("ฉันกินข้าวที่โรงเรียน")
            finder = BigramCollocationFinder.from_words(words)
        """
        if window_size < 2:
            window_size = 2

        word_fd: dict[str, int] = defaultdict(int)
        ngram_fd: dict[tuple[str, str], int] = defaultdict(int)

        for i, w in enumerate(words):
            word_fd[w] += 1
            for j in range(i + 1, min(i + window_size, len(words))):
                ngram_fd[(w, words[j])] += 1

        return cls(word_fd, ngram_fd)

    @classmethod
    def from_corpus(
        cls,
        documents: Iterable[list[str]],
        window_size: int = 2,
    ) -> BigramCollocationFinder:
        """Construct from multiple documents (streaming).

        :param documents: iterable of pre-tokenized word lists
        :param int window_size: window for co-occurrence
        :return: BigramCollocationFinder
        :rtype: BigramCollocationFinder
        """
        if window_size < 2:
            window_size = 2

        word_fd: dict[str, int] = defaultdict(int)
        ngram_fd: dict[tuple[str, str], int] = defaultdict(int)

        for words in documents:
            for i, w in enumerate(words):
                word_fd[w] += 1
                for j in range(i + 1, min(i + window_size, len(words))):
                    ngram_fd[(w, words[j])] += 1

        return cls(word_fd, ngram_fd)

    @classmethod
    def from_tnc(cls) -> BigramCollocationFinder:
        """Construct from Thai National Corpus bigram frequencies.

        Uses pre-computed TNC unigram and bigram frequency data.
        The TNC corpus must be available (downloaded automatically
        or via ``pythainlp.corpus.download()``).

        :return: BigramCollocationFinder populated with TNC data
        :rtype: BigramCollocationFinder

        :Example:
        ::

            from pythainlp.collocation import (
                BigramCollocationFinder,
                BigramAssocMeasures,
            )

            finder = BigramCollocationFinder.from_tnc()
            finder.apply_freq_filter(50)
            top = finder.nbest(BigramAssocMeasures.log_dice, 20)
        """
        from pythainlp.corpus.tnc import (
            bigram_word_freqs,
            unigram_word_freqs,
        )

        return cls(unigram_word_freqs(), bigram_word_freqs())

    # ── Filtering (mutating) ──

    def apply_freq_filter(self, min_freq: int) -> None:
        """Remove bigrams with frequency below *min_freq*.

        :param int min_freq: minimum frequency threshold
        """
        self.ngram_fd = {
            ng: c for ng, c in self.ngram_fd.items() if c >= min_freq
        }
        self._recompute_n()

    def apply_word_filter(self, fn: Callable[[str], bool]) -> None:
        """Remove bigrams containing any word where *fn(word)* is True.

        :param fn: filter function returning True for words to remove

        :Example:
        ::

            from pythainlp.corpus import thai_stopwords

            finder.apply_word_filter(lambda w: w in thai_stopwords())
        """
        self.ngram_fd = {
            ng: c
            for ng, c in self.ngram_fd.items()
            if not any(fn(w) for w in ng)
        }
        # Also clean word_fd
        self.word_fd = {w: c for w, c in self.word_fd.items() if not fn(w)}
        self._recompute_n()

    def apply_ngram_filter(
        self, fn: Callable[[str, str], bool]
    ) -> None:
        """Remove bigrams where *fn(w1, w2)* is True.

        :param fn: filter function taking two words, returns True to remove
        """
        self.ngram_fd = {
            ng: c for ng, c in self.ngram_fd.items() if not fn(*ng)
        }
        self._recompute_n()

    def apply_pos_filter(
        self,
        allowed_patterns: list[tuple[str, str]] | None = None,
        tagged_words: list[tuple[str, str]] | None = None,
    ) -> None:
        """Filter bigrams by POS tag patterns.

        If *tagged_words* is not provided, words are auto-tagged using
        ``pythainlp.tag.pos_tag()``.

        :param allowed_patterns: list of (POS1, POS2) tuples to keep.
            Defaults to common compound patterns:
            ``[("NCMN","NCMN"), ("NCMN","VACT"), ("VACT","NCMN")]``
        :param tagged_words: pre-tagged words as ``[(word, tag), ...]``.
            If None, auto-tags using perceptron tagger with orchid corpus.

        :Example:
        ::

            finder.apply_pos_filter(
                allowed_patterns=[("NCMN", "NCMN"), ("VACT", "NCMN")]
            )
        """
        if allowed_patterns is None:
            allowed_patterns = [
                ("NCMN", "NCMN"),
                ("NCMN", "VACT"),
                ("VACT", "NCMN"),
            ]

        # Build word → POS mapping
        if tagged_words is not None:
            word_pos = {w: t for w, t in tagged_words}
        else:
            from pythainlp.tag import pos_tag

            unique_words = list(
                {w for ng in self.ngram_fd for w in ng}
            )
            tags = pos_tag(unique_words)
            word_pos = {w: t for w, t in tags}

        allowed_set = set(allowed_patterns)
        self.ngram_fd = {
            (w1, w2): c
            for (w1, w2), c in self.ngram_fd.items()
            if (word_pos.get(w1, ""), word_pos.get(w2, ""))
            in allowed_set
        }
        self._recompute_n()

    # ── Scoring ──

    def score_ngrams(
        self,
        measure: Callable[[int, int, int, int], float],
    ) -> list[tuple[tuple[str, str], float]]:
        """Score all bigrams with the given association measure.

        :param measure: a static method from BigramAssocMeasures
        :return: list of ``((w1, w2), score)`` sorted by score descending
        :rtype: list[tuple[tuple[str, str], float]]
        """
        scored = []
        for (w1, w2), n_ii in self.ngram_fd.items():
            n_ix = self.word_fd.get(w1, 0)
            n_xi = self.word_fd.get(w2, 0)
            score = measure(n_ii, n_ix, n_xi, self.N)
            scored.append(((w1, w2), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def nbest(
        self,
        measure: Callable[[int, int, int, int], float],
        n: int = 10,
    ) -> list[tuple[str, str]]:
        """Return the top *n* bigrams by the given measure.

        :param measure: a static method from BigramAssocMeasures
        :param int n: number of top results
        :return: list of ``(w1, w2)`` tuples
        :rtype: list[tuple[str, str]]
        """
        return [ng for ng, _ in self.score_ngrams(measure)[:n]]

    def above_score(
        self,
        measure: Callable[[int, int, int, int], float],
        min_score: float,
    ) -> list[tuple[str, str]]:
        """Return bigrams scoring at or above *min_score*.

        :param measure: a static method from BigramAssocMeasures
        :param float min_score: minimum score threshold
        :return: list of ``(w1, w2)`` tuples
        :rtype: list[tuple[str, str]]
        """
        return [
            ng
            for ng, score in self.score_ngrams(measure)
            if score >= min_score
        ]


class TrigramCollocationFinder:
    """Find and rank trigram collocations from Thai text.

    Same Build-Filter-Score pattern as BigramCollocationFinder.
    Trigrams are scored by decomposing ``(w1, w2, w3)`` into
    leading bigram ``(w1, w2)`` and trailing word ``w3``.

    :param dict word_fd: unigram frequency distribution
    :param dict bigram_fd: bigram frequency distribution
    :param dict ngram_fd: trigram frequency distribution

    :Example:
    ::

        from pythainlp.tokenize import word_tokenize
        from pythainlp.collocation import (
            TrigramCollocationFinder,
            TrigramAssocMeasures,
        )

        words = word_tokenize("ฉันกินข้าวผัดกุ้งที่ร้าน")
        finder = TrigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(TrigramAssocMeasures.likelihood_ratio)
    """

    def __init__(
        self,
        word_fd: dict[str, int],
        bigram_fd: dict[tuple[str, str], int],
        ngram_fd: dict[tuple[str, str, str], int],
    ) -> None:
        self.word_fd: dict[str, int] = dict(word_fd)
        self.bigram_fd: dict[tuple[str, str], int] = dict(bigram_fd)
        self.ngram_fd: dict[tuple[str, str, str], int] = dict(ngram_fd)
        self._recompute_n()

    def _recompute_n(self) -> None:
        self.N: int = sum(self.ngram_fd.values())

    @classmethod
    def from_words(
        cls,
        words: list[str],
        window_size: int = 2,
    ) -> TrigramCollocationFinder:
        """Construct from a pre-tokenized word list.

        :param list[str] words: pre-tokenized Thai words
        :param int window_size: not used for trigrams (always adjacent)
        :return: TrigramCollocationFinder
        :rtype: TrigramCollocationFinder
        """
        word_fd: dict[str, int] = defaultdict(int)
        bigram_fd: dict[tuple[str, str], int] = defaultdict(int)
        ngram_fd: dict[tuple[str, str, str], int] = defaultdict(int)

        for i, w in enumerate(words):
            word_fd[w] += 1
            if i + 1 < len(words):
                bigram_fd[(w, words[i + 1])] += 1
            if i + 2 < len(words):
                ngram_fd[(w, words[i + 1], words[i + 2])] += 1

        return cls(word_fd, bigram_fd, ngram_fd)

    @classmethod
    def from_corpus(
        cls,
        documents: Iterable[list[str]],
        window_size: int = 2,
    ) -> TrigramCollocationFinder:
        """Construct from multiple documents (streaming).

        :param documents: iterable of pre-tokenized word lists
        :param int window_size: not used for trigrams
        :return: TrigramCollocationFinder
        :rtype: TrigramCollocationFinder
        """
        word_fd: dict[str, int] = defaultdict(int)
        bigram_fd: dict[tuple[str, str], int] = defaultdict(int)
        ngram_fd: dict[tuple[str, str, str], int] = defaultdict(int)

        for words in documents:
            for i, w in enumerate(words):
                word_fd[w] += 1
                if i + 1 < len(words):
                    bigram_fd[(w, words[i + 1])] += 1
                if i + 2 < len(words):
                    ngram_fd[(w, words[i + 1], words[i + 2])] += 1

        return cls(word_fd, bigram_fd, ngram_fd)

    @classmethod
    def from_tnc(cls) -> TrigramCollocationFinder:
        """Construct from Thai National Corpus trigram frequencies.

        Uses pre-computed TNC unigram, bigram, and trigram data.

        :return: TrigramCollocationFinder populated with TNC data
        :rtype: TrigramCollocationFinder
        """
        from pythainlp.corpus.tnc import (
            bigram_word_freqs,
            trigram_word_freqs,
            unigram_word_freqs,
        )

        return cls(
            unigram_word_freqs(),
            bigram_word_freqs(),
            trigram_word_freqs(),
        )

    # ── Filtering (mutating) ──

    def apply_freq_filter(self, min_freq: int) -> None:
        """Remove trigrams with frequency below *min_freq*."""
        self.ngram_fd = {
            ng: c for ng, c in self.ngram_fd.items() if c >= min_freq
        }
        self._recompute_n()

    def apply_word_filter(self, fn: Callable[[str], bool]) -> None:
        """Remove trigrams containing any word where *fn(word)* is True."""
        self.ngram_fd = {
            ng: c
            for ng, c in self.ngram_fd.items()
            if not any(fn(w) for w in ng)
        }
        self.bigram_fd = {
            ng: c
            for ng, c in self.bigram_fd.items()
            if not any(fn(w) for w in ng)
        }
        self.word_fd = {w: c for w, c in self.word_fd.items() if not fn(w)}
        self._recompute_n()

    def apply_ngram_filter(
        self, fn: Callable[[str, str, str], bool]
    ) -> None:
        """Remove trigrams where *fn(w1, w2, w3)* is True."""
        self.ngram_fd = {
            ng: c for ng, c in self.ngram_fd.items() if not fn(*ng)
        }
        self._recompute_n()

    def apply_pos_filter(
        self,
        allowed_patterns: list[tuple[str, str, str]] | None = None,
        tagged_words: list[tuple[str, str]] | None = None,
    ) -> None:
        """Filter trigrams by POS tag patterns.

        :param allowed_patterns: list of (POS1, POS2, POS3) tuples to keep.
            Defaults to ``[("NCMN","NCMN","NCMN")]``.
        :param tagged_words: pre-tagged words as ``[(word, tag), ...]``.
        """
        if allowed_patterns is None:
            allowed_patterns = [("NCMN", "NCMN", "NCMN")]

        if tagged_words is not None:
            word_pos = {w: t for w, t in tagged_words}
        else:
            from pythainlp.tag import pos_tag

            unique_words = list(
                {w for ng in self.ngram_fd for w in ng}
            )
            tags = pos_tag(unique_words)
            word_pos = {w: t for w, t in tags}

        allowed_set = set(allowed_patterns)
        self.ngram_fd = {
            (w1, w2, w3): c
            for (w1, w2, w3), c in self.ngram_fd.items()
            if (
                word_pos.get(w1, ""),
                word_pos.get(w2, ""),
                word_pos.get(w3, ""),
            )
            in allowed_set
        }
        self._recompute_n()

    # ── Scoring ──

    def score_ngrams(
        self,
        measure: Callable[[int, int, int, int], float],
    ) -> list[tuple[tuple[str, str, str], float]]:
        """Score all trigrams with the given association measure.

        Decomposes ``(w1, w2, w3)`` into leading bigram ``(w1, w2)``
        and trailing word ``w3`` for the contingency table.

        :param measure: a static method from TrigramAssocMeasures
        :return: list of ``((w1, w2, w3), score)`` sorted descending
        :rtype: list[tuple[tuple[str, str, str], float]]
        """
        scored = []
        for (w1, w2, w3), n_iii in self.ngram_fd.items():
            n_iix = self.bigram_fd.get((w1, w2), 0)
            n_xxi = self.word_fd.get(w3, 0)
            score = measure(n_iii, n_iix, n_xxi, self.N)
            scored.append(((w1, w2, w3), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def nbest(
        self,
        measure: Callable[[int, int, int, int], float],
        n: int = 10,
    ) -> list[tuple[str, str, str]]:
        """Return the top *n* trigrams by the given measure."""
        return [ng for ng, _ in self.score_ngrams(measure)[:n]]

    def above_score(
        self,
        measure: Callable[[int, int, int, int], float],
        min_score: float,
    ) -> list[tuple[str, str, str]]:
        """Return trigrams scoring at or above *min_score*."""
        return [
            ng
            for ng, score in self.score_ngrams(measure)
            if score >= min_score
        ]
