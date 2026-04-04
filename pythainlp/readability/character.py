# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Tier 2: Thai character complexity metrics for readability."""

from __future__ import annotations

import re

from pythainlp import thai_consonants
from pythainlp.readability import DEFAULT_WORD_ENGINE

__all__: list[str] = [
    "consonant_cluster_ratio",
    "karan_density",
    "tone_mark_density",
]

# Known Thai initial consonant clusters (อักษรควบกล้ำ)
_CLUSTER_PAIRS: frozenset[str] = frozenset(
    {
        "กร", "กล", "กว",
        "ขร", "ขล", "ขว",
        "คร", "คล", "คว",
        "ปร", "ปล",
        "พร", "พล",
        "ตร",
        "ทร",
        "ดร",
        "บร", "บล",
        "ผล",
        "ฝร",
        "ศร",
        "สร",
    }
)

# Karan character: การันต์ (thanthakhat)
_KARAN: str = "\u0e4c"

# Regex to extract first two Thai consonants at start of a word
_CONSONANT_SET: frozenset[str] = frozenset(thai_consonants)


def _has_initial_cluster(word: str) -> bool:
    """Check if a word starts with a known consonant cluster."""
    chars = [c for c in word if c in _CONSONANT_SET]
    if len(chars) < 2:
        return False
    return (chars[0] + chars[1]) in _CLUSTER_PAIRS


def tone_mark_density(
    text: str = "",
    *,
    char_counts: dict[str, int] | None = None,
) -> float:
    """Ratio of tone marks to total Thai characters.

    Higher density indicates more explicit tonal marking,
    often seen in formal or literary text.

    :param str text: Thai text to analyze
    :param dict char_counts: pre-computed result from
        ``count_thai_chars()`` (optional)
    :return: tone mark density in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import tone_mark_density

        tone_mark_density("น้ำตกสวยมาก")
    """
    if char_counts is None:
        from pythainlp.util.thai import count_thai_chars

        char_counts = count_thai_chars(text)

    total_thai = (
        char_counts.get("consonants", 0)
        + char_counts.get("vowels", 0)
        + char_counts.get("tonemarks", 0)
        + char_counts.get("signs", 0)
    )

    if total_thai == 0:
        return 0.0

    return char_counts.get("tonemarks", 0) / total_thai


def consonant_cluster_ratio(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
) -> float:
    """Ratio of words with initial consonant clusters to total words.

    Thai consonant clusters (อักษรควบกล้ำ) like กร-, ปล-, ทร-
    increase decoding difficulty, especially for early readers.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :return: cluster ratio in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import consonant_cluster_ratio

        consonant_cluster_ratio("กรุงเทพเป็นเมืองใหญ่")
    """
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(text, engine=word_engine) if w.strip()]

    if not words:
        return 0.0

    cluster_count = sum(1 for w in words if _has_initial_cluster(w))
    return cluster_count / len(words)


def karan_density(
    text: str = "",
    *,
    words: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
) -> float:
    """Ratio of words containing karan (การันต์, ์) to total words.

    Karan marks silent consonants, typically found in Pali/Sanskrit
    loanwords. Higher density indicates more formal/academic vocabulary.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param str word_engine: word tokenizer engine
    :return: karan density in range [0, 1]
    :rtype: float

    :Example:
    ::

        from pythainlp.readability import karan_density

        karan_density("ภูมิศาสตร์และวิทยาศาสตร์")
    """
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [w for w in word_tokenize(text, engine=word_engine) if w.strip()]

    if not words:
        return 0.0

    karan_count = sum(1 for w in words if _KARAN in w)
    return karan_count / len(words)
