# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Tier 1: Core statistical readability metrics for Thai text."""

from __future__ import annotations

from functools import lru_cache

from pythainlp.readability import (
    DEFAULT_FREQ_CORPUS,
    DEFAULT_RARE_WORD_TOP_N,
    DEFAULT_SENT_ENGINE,
    DEFAULT_SYLLABLE_ENGINE,
    DEFAULT_WORD_ENGINE,
)

__all__: list[str] = [
    "average_sentence_length",
    "average_syllables_per_word",
    "average_word_length",
    "dead_syllable_ratio",
    "rare_word_ratio",
    "type_token_ratio",
    "word_frequency_score",
]


def _tokenize_words(text: str, engine: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [
        w
        for w in word_tokenize(text, engine=engine)
        if w.strip()
    ]


def _tokenize_sentences(text: str, engine: str) -> list[str]:
    from pythainlp.tokenize import sent_tokenize

    try:
        sents = sent_tokenize(text, engine=engine)
    except (ImportError, ModuleNotFoundError):
        # Fallback: crfcut needs pycrfsuite; use whitespace-based
        sents = sent_tokenize(text, engine="whitespace")
    return [s for s in sents if s.strip()]


def _tokenize_syllables(text: str, engine: str) -> list[str]:
    try:
        from pythainlp.tokenize import syllable_tokenize

        syls = syllable_tokenize(text, engine=engine)
    except (ImportError, ModuleNotFoundError):
        # Fallback: han_solo needs pycrfsuite; use subword tcc
        from pythainlp.tokenize import subword_tokenize

        syls = subword_tokenize(text, engine="tcc")
    return [s for s in syls if s.strip()]


@lru_cache(maxsize=4)
def _load_freq_corpus(corpus: str) -> dict[str, int]:
    if corpus == "ttc":
        from pythainlp.corpus.ttc import unigram_word_freqs

        return unigram_word_freqs()
    else:
        from pythainlp.corpus.tnc import unigram_word_freqs

        return unigram_word_freqs()


@lru_cache(maxsize=4)
def _build_freq_ranks(corpus: str) -> dict[str, int]:
    """Build word → rank mapping (1 = most frequent)."""
    freqs = _load_freq_corpus(corpus)
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    return {word: rank + 1 for rank, (word, _) in enumerate(sorted_words)}


def average_sentence_length(
    text: str = "",
    *,
    words: list[str] | None = None,
    sentences: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    sent_engine: str = DEFAULT_SENT_ENGINE,
) -> float:
    """Average number of words per sentence.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] sentences: pre-segmented sentences (optional)
    :param str word_engine: word tokenizer engine
    :param str sent_engine: sentence tokenizer engine
    :return: average words per sentence
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import average_sentence_length

        average_sentence_length("แมวกินปลา แมวนอนหลับ")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)
    if sentences is None:
        sentences = _tokenize_sentences(text, sent_engine)

    if not sentences:
        return 0.0

    return len(words) / len(sentences)


def average_word_length(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
) -> float:
    """Average number of characters per word.

    Counts only Thai characters (consonants + vowels + tonemarks + signs).

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :return: average characters per word
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import average_word_length

        average_word_length("โรงเรียน")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)

    if not words:
        return 0.0

    return sum(len(w) for w in words) / len(words)


def average_syllables_per_word(
    text: str = "",
    *,
    words: list[str] | None = None,
    syllables: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    syllable_engine: str = DEFAULT_SYLLABLE_ENGINE,
) -> float:
    """Average number of syllables per word.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] syllables: pre-tokenized syllables (optional)
    :param str word_engine: word tokenizer engine
    :param str syllable_engine: syllable tokenizer engine
    :return: average syllables per word
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import average_syllables_per_word

        average_syllables_per_word("สถาบันวิจัย")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)
    if syllables is None:
        syllables = _tokenize_syllables(text, syllable_engine)

    if not words:
        return 0.0

    return len(syllables) / len(words)


def type_token_ratio(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
) -> float:
    """Type-token ratio (lexical diversity).

    Ratio of unique words to total words. Higher values indicate
    more diverse vocabulary, which correlates with difficulty.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :return: TTR in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import type_token_ratio

        type_token_ratio("แมวกินปลา แมวนอนหลับ")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)

    if not words:
        return 0.0

    return len(set(words)) / len(words)


def word_frequency_score(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    freq_corpus: str = DEFAULT_FREQ_CORPUS,
) -> float:
    """Average word frequency rank score.

    For each word, looks up its rank in the frequency corpus.
    Normalizes rank so 0.0 = most common, 1.0 = rarest/unknown.
    Returns the average across all words.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :param str freq_corpus: frequency corpus ("tnc" or "ttc")
    :return: average frequency score in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import word_frequency_score

        word_frequency_score("แมวกินปลา")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)

    if not words:
        return 0.0

    ranks = _build_freq_ranks(freq_corpus)
    total_vocab = len(ranks) if ranks else 1

    scores = []
    for w in words:
        rank = ranks.get(w)
        if rank is None:
            scores.append(1.0)
        else:
            scores.append(min(rank / total_vocab, 1.0))

    return sum(scores) / len(scores)


def rare_word_ratio(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    freq_corpus: str = DEFAULT_FREQ_CORPUS,
    top_n: int = DEFAULT_RARE_WORD_TOP_N,
) -> float:
    """Ratio of rare words (not in top-N frequency list) to total words.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :param str freq_corpus: frequency corpus ("tnc" or "ttc")
    :param int top_n: top N words considered "common"
    :return: ratio of rare words in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import rare_word_ratio

        rare_word_ratio("แมวกินปลา")
    """
    if words is None:
        words = _tokenize_words(text, word_engine)

    if not words:
        return 0.0

    ranks = _build_freq_ranks(freq_corpus)
    rare_count = sum(1 for w in words if ranks.get(w, top_n + 1) > top_n)

    return rare_count / len(words)


def dead_syllable_ratio(
    text: str = "",
    *,
    syllables: list[str] | None = None,
    syllable_engine: str = DEFAULT_SYLLABLE_ENGINE,
) -> float:
    """Ratio of dead syllables to total syllables.

    Dead syllables end in stop consonants or have short vowels,
    creating abrupt sounds. They are disproportionately found in
    Pali/Sanskrit loanwords and formal/academic vocabulary.

    This is a novel Thai-specific readability metric.

    :param str text: Thai text to analyze
    :param list[str] syllables: pre-tokenized syllables (optional)
    :param str syllable_engine: syllable tokenizer engine
    :return: ratio of dead syllables in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import dead_syllable_ratio

        dead_syllable_ratio("มานอนกิน")       # mostly live
        dead_syllable_ratio("ปรัชญาเศรษฐกิจ")  # more dead
    """
    if syllables is None:
        syllables = _tokenize_syllables(text, syllable_engine)

    if not syllables:
        return 0.0

    from pythainlp.util.syllable import sound_syllable

    dead_count = sum(
        1 for s in syllables if sound_syllable(s) == "dead"
    )

    return dead_count / len(syllables)
