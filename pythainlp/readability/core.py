# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Composite readability scorer for Thai text."""

from __future__ import annotations

from pythainlp.readability import (
    DEFAULT_FREQ_CORPUS,
    DEFAULT_GRADE_THRESHOLDS,
    DEFAULT_RARE_WORD_TOP_N,
    DEFAULT_SENT_ENGINE,
    DEFAULT_SYLLABLE_ENGINE,
    DEFAULT_WEIGHTS,
    DEFAULT_WORD_ENGINE,
    GRADE_LEVEL_DISCLAIMER,
    ReadabilityResult,
)

__all__: list[str] = ["readability_score"]

# Normalization bounds: (raw_easy, raw_hard) → linear map to 0..100
_NORMALIZATION: dict[str, tuple[float, float]] = {
    "avg_sentence_length": (5.0, 40.0),
    "avg_word_length": (2.0, 6.0),
    "avg_syllables_per_word": (1.0, 3.0),
    "type_token_ratio": (0.3, 0.9),
    "word_frequency_score": (0.0, 1.0),
    "rare_word_ratio": (0.0, 0.6),
    "dead_syllable_ratio": (0.1, 0.7),
    "tone_mark_density": (0.0, 0.15),
    "consonant_cluster_ratio": (0.0, 0.3),
    "karan_density": (0.0, 0.2),
    "ari": (-5.0, 20.0),
    "coleman_liau": (-5.0, 20.0),
    "lix": (20.0, 70.0),
}


def _normalize(metric: str, raw: float) -> float:
    """Linear normalization with clamping to [0, 100]."""
    lo, hi = _NORMALIZATION[metric]
    if hi == lo:
        return 0.0
    normalized = (raw - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, normalized))


def _score_to_level(
    score: float,
    thresholds: list[tuple[float, str]],
) -> str:
    """Map a composite score to a grade level label."""
    for cutoff, label in thresholds:
        if score <= cutoff:
            return label
    return thresholds[-1][1]


def readability_score(
    text: str = "",
    *,
    words: list[str] | None = None,
    sentences: list[str] | None = None,
    syllables: list[str] | None = None,
    word_engine: str = DEFAULT_WORD_ENGINE,
    sent_engine: str = DEFAULT_SENT_ENGINE,
    syllable_engine: str = DEFAULT_SYLLABLE_ENGINE,
    freq_corpus: str = DEFAULT_FREQ_CORPUS,
    rare_word_top_n: int = DEFAULT_RARE_WORD_TOP_N,
    weights: dict[str, float] | None = None,
    grade_thresholds: list[tuple[float, str]] | None = None,
) -> ReadabilityResult:
    """Compute composite Thai text readability score.

    Tokenizes text once, then computes all 13 individual metrics and
    combines them into a weighted composite score (0-100, higher = harder).
    Maps the score to a Thai education level with customizable thresholds.

    :param str text: Thai text to analyze
    :param list[str] words: pre-tokenized words (optional)
    :param list[str] sentences: pre-segmented sentences (optional)
    :param list[str] syllables: pre-tokenized syllables (optional)
    :param str word_engine: word tokenizer engine
    :param str sent_engine: sentence tokenizer engine
    :param str syllable_engine: syllable tokenizer engine
    :param str freq_corpus: frequency corpus ("tnc" or "ttc")
    :param int rare_word_top_n: top N words considered "common"
    :param dict weights: custom metric weights (auto-normalized to sum to 1.0)
    :param list grade_thresholds: custom ``[(score, label), ...]`` thresholds
    :return: ReadabilityResult with score, level, disclaimer, and stats
    :rtype: ReadabilityResult

    :Example:
    ::

        from pythainlp.readability import readability_score

        result = readability_score("ฉันไปโรงเรียน")
        print(result["score"])  # e.g., 15.3
        print(result["level"])  # e.g., "ป.1-3"

        # Custom weights
        result = readability_score(
            "ข้อความยาก",
            weights={"dead_syllable_ratio": 0.5, "rare_word_ratio": 0.5},
        )
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if grade_thresholds is None:
        grade_thresholds = DEFAULT_GRADE_THRESHOLDS

    # Normalize weights to sum to 1.0
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        norm_weights = {k: v / weight_sum for k, v in weights.items()}
    else:
        norm_weights = weights

    # ── Tokenize once ──
    if words is None:
        from pythainlp.tokenize import word_tokenize

        words = [
            w
            for w in word_tokenize(text, engine=word_engine)
            if w.strip()
        ]
    if sentences is None:
        from pythainlp.tokenize import sent_tokenize

        try:
            sents = sent_tokenize(text, engine=sent_engine)
        except (ImportError, ModuleNotFoundError):
            sents = sent_tokenize(text, engine="whitespace")
        sentences = [s for s in sents if s.strip()]
    if syllables is None:
        try:
            from pythainlp.tokenize import syllable_tokenize

            syls = syllable_tokenize(text, engine=syllable_engine)
        except (ImportError, ModuleNotFoundError):
            from pythainlp.tokenize import subword_tokenize

            syls = subword_tokenize(text, engine="tcc")
        syllables = [s for s in syls if s.strip()]

    # Early return for empty input
    if not words:
        return ReadabilityResult(
            score=0.0,
            level=grade_thresholds[0][1] if grade_thresholds else "",
            level_disclaimer=GRADE_LEVEL_DISCLAIMER,
            stats={},
        )

    # ── Compute char counts once ──
    from pythainlp.util.thai import count_thai_chars

    char_counts = count_thai_chars(text)

    # ── Compute all 13 metrics ──
    from pythainlp.readability.character import (
        consonant_cluster_ratio,
        karan_density,
        tone_mark_density,
    )
    from pythainlp.readability.formulas import (
        automated_readability_index,
        coleman_liau_index,
        lix,
    )
    from pythainlp.readability.statistics import (
        average_sentence_length,
        average_syllables_per_word,
        average_word_length,
        dead_syllable_ratio,
        rare_word_ratio,
        type_token_ratio,
        word_frequency_score,
    )

    stats: dict[str, float] = {
        "avg_sentence_length": average_sentence_length(
            words=words, sentences=sentences
        ),
        "avg_word_length": average_word_length(words=words),
        "avg_syllables_per_word": average_syllables_per_word(
            words=words, syllables=syllables
        ),
        "type_token_ratio": type_token_ratio(words=words),
        "word_frequency_score": word_frequency_score(
            words=words, freq_corpus=freq_corpus
        ),
        "rare_word_ratio": rare_word_ratio(
            words=words, freq_corpus=freq_corpus, top_n=rare_word_top_n
        ),
        "dead_syllable_ratio": dead_syllable_ratio(syllables=syllables),
        "tone_mark_density": tone_mark_density(char_counts=char_counts),
        "consonant_cluster_ratio": consonant_cluster_ratio(words=words),
        "karan_density": karan_density(words=words),
        "ari": automated_readability_index(
            words=words, sentences=sentences
        ),
        "coleman_liau": coleman_liau_index(
            words=words, sentences=sentences
        ),
        "lix": lix(words=words, sentences=sentences),
    }

    # ── Composite score ──
    composite = 0.0
    for metric, raw in stats.items():
        w = norm_weights.get(metric, 0.0)
        if w > 0 and metric in _NORMALIZATION:
            composite += w * _normalize(metric, raw)

    composite = max(0.0, min(100.0, composite))

    return ReadabilityResult(
        score=round(composite, 2),
        level=_score_to_level(composite, grade_thresholds),
        level_disclaimer=GRADE_LEVEL_DISCLAIMER,
        stats=stats,
    )
