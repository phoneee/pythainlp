# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Thai text readability analysis.

Provides formula-based readability metrics for Thai text with zero
external dependencies. Complements ML-based approaches (e.g., TLTK)
with lightweight, transparent, and customizable scoring.

:Example:
::

    from pythainlp.readability import readability_score

    result = readability_score("ฉันไปโรงเรียน")
    print(result["score"])   # 0-100 (higher = harder)
    print(result["level"])   # e.g., "ป.1-3"
    print(result["stats"])   # dict of all 13 metrics
"""

from __future__ import annotations

from typing import TypedDict

__all__: list[str] = [
    "ReadabilityResult",
    "automated_readability_index",
    "average_sentence_length",
    "average_syllables_per_word",
    "average_word_length",
    "coleman_liau_index",
    "consonant_cluster_ratio",
    "dead_syllable_ratio",
    "karan_density",
    "lix",
    "rare_word_ratio",
    "readability_score",
    "tone_mark_density",
    "type_token_ratio",
    "word_frequency_score",
]

# ── Default engine constants ──
DEFAULT_WORD_ENGINE: str = "newmm"
DEFAULT_SENT_ENGINE: str = "crfcut"
DEFAULT_SYLLABLE_ENGINE: str = "han_solo"
DEFAULT_FREQ_CORPUS: str = "tnc"
DEFAULT_RARE_WORD_TOP_N: int = 5000
DEFAULT_LONG_WORD_THRESHOLD: int = 6

# ── Composite score weights (sum to 1.0) ──
DEFAULT_WEIGHTS: dict[str, float] = {
    "avg_sentence_length": 0.15,
    "avg_word_length": 0.10,
    "avg_syllables_per_word": 0.10,
    "type_token_ratio": 0.05,
    "word_frequency_score": 0.10,
    "rare_word_ratio": 0.10,
    "dead_syllable_ratio": 0.10,
    "tone_mark_density": 0.05,
    "consonant_cluster_ratio": 0.05,
    "karan_density": 0.05,
    "ari": 0.05,
    "coleman_liau": 0.05,
    "lix": 0.05,
}

# ── Grade level thresholds ──
# score ranges: 0 = very easy, 100 = very hard
DEFAULT_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (20.0, "ป.1-3"),
    (40.0, "ป.4-6"),
    (60.0, "ม.1-3"),
    (80.0, "ม.4-6"),
    (100.0, "อุดมศึกษา"),
]

GRADE_LEVEL_DISCLAIMER: str = (
    "Grade level mappings are heuristic estimates based on "
    "formula-based metrics. They have not been validated against "
    "Thai curriculum standards and should be used as rough guidance only."
)


class ReadabilityResult(TypedDict):
    """Result of composite readability analysis.

    - ``score``: float 0-100, higher = harder
    - ``level``: Thai education level (e.g., "ป.1-3")
    - ``level_disclaimer``: disclaimer about heuristic mapping
    - ``stats``: dict of all individual metric raw values
    """

    score: float
    level: str
    level_disclaimer: str
    stats: dict[str, float]


from pythainlp.readability.character import (  # noqa: E402
    consonant_cluster_ratio,
    karan_density,
    tone_mark_density,
)
from pythainlp.readability.core import readability_score  # noqa: E402
from pythainlp.readability.formulas import (  # noqa: E402
    automated_readability_index,
    coleman_liau_index,
    lix,
)
from pythainlp.readability.statistics import (  # noqa: E402
    average_sentence_length,
    average_syllables_per_word,
    average_word_length,
    dead_syllable_ratio,
    rare_word_ratio,
    type_token_ratio,
    word_frequency_score,
)
