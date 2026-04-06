# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Tier 3: Adapted international readability formulas for Thai.

These formulas use original English coefficients. Absolute values are
not calibrated for Thai, but **relative comparisons** between Thai texts
are meaningful (text A harder than text B).

Coleman-Liau and ARI are recommended for Thai because they use character
counts rather than syllable counts (Tongtep et al., 2014, PRICAI).
"""

from __future__ import annotations

from pythainlp.readability import (
    DEFAULT_LONG_WORD_THRESHOLD,
    DEFAULT_SENT_ENGINE,
    DEFAULT_WORD_ENGINE,
)

__all__: list[str] = [
    "automated_readability_index",
    "coleman_liau_index",
    "lix",
]


def automated_readability_index(
    text: str = "",
    *,
    words: list[str] | None = None,
    sentences: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    sent_engine: str = DEFAULT_SENT_ENGINE,
) -> float:
    """Automated Readability Index (ARI) adapted for Thai.

    ``ARI = 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43``

    Uses character counts (not syllable counts), making it more
    suitable for Thai where syllable boundary detection is imperfect.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] sentences: pre-segmented sentences (optional)
    :param str word_engine: word tokenizer engine
    :param str sent_engine: sentence tokenizer engine
    :return: ARI score (higher = harder)
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import automated_readability_index

        automated_readability_index("ฉันไปโรงเรียนทุกวัน")
    """
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(text, engine=word_engine) if w.strip()]
    if sentences is None:
        from pythainlp.tokenize import sent_tokenize

        sentences = [s for s in sent_tokenize(text, engine=sent_engine) if s.strip()]

    n_words = len(words)
    n_sents = len(sentences)

    if n_words == 0 or n_sents == 0:
        return 0.0

    total_chars = sum(len(w) for w in words)

    return 4.71 * (total_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43


def coleman_liau_index(
    text: str = "",
    *,
    words: list[str] | None = None,
    sentences: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    sent_engine: str = DEFAULT_SENT_ENGINE,
) -> float:
    """Coleman-Liau Index adapted for Thai.

    ``CLI = 0.0588 * L - 0.296 * S - 15.8``

    where L = average characters per 100 words,
    S = average sentences per 100 words.

    Character-based formula recommended for Thai by Tongtep et al. (2014).

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] sentences: pre-segmented sentences (optional)
    :param str word_engine: word tokenizer engine
    :param str sent_engine: sentence tokenizer engine
    :return: Coleman-Liau score (higher = harder)
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import coleman_liau_index

        coleman_liau_index("แมวกินปลา")
    """
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(text, engine=word_engine) if w.strip()]
    if sentences is None:
        from pythainlp.tokenize import sent_tokenize

        sentences = [s for s in sent_tokenize(text, engine=sent_engine) if s.strip()]

    n_words = len(words)
    n_sents = len(sentences)

    if n_words == 0:
        return 0.0

    total_chars = sum(len(w) for w in words)

    # L = avg chars per 100 words, S = avg sentences per 100 words
    l_val = (total_chars / n_words) * 100
    s_val = (n_sents / n_words) * 100 if n_words > 0 else 0

    return 0.0588 * l_val - 0.296 * s_val - 15.8


def lix(
    text: str = "",
    *,
    words: list[str] | None = None,
    sentences: list[str] | None = None,
    long_word_threshold: int = DEFAULT_LONG_WORD_THRESHOLD,
    word_engine: str = DEFAULT_WORD_ENGINE,
    sent_engine: str = DEFAULT_SENT_ENGINE,
) -> float:
    """LIX (Läsbarhetsindex) adapted for Thai.

    ``LIX = words/sentences + (long_words * 100) / words``

    Designed to be language-independent. Default long word threshold
    is 6 characters (reasonable for Thai given average word length 3-4).

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] sentences: pre-segmented sentences (optional)
    :param int long_word_threshold: characters above which a word is "long"
    :param str word_engine: word tokenizer engine
    :param str sent_engine: sentence tokenizer engine
    :return: LIX score (higher = harder)
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import lix

        lix("ฉันไปโรงเรียนทุกวัน")
    """
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(text, engine=word_engine) if w.strip()]
    if sentences is None:
        from pythainlp.tokenize import sent_tokenize

        sentences = [s for s in sent_tokenize(text, engine=sent_engine) if s.strip()]

    n_words = len(words)
    n_sents = len(sentences)

    if n_words == 0 or n_sents == 0:
        return 0.0

    long_words = sum(1 for w in words if len(w) > long_word_threshold)

    return (n_words / n_sents) + (long_words * 100 / n_words)
