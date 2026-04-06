# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Association measures for collocation extraction.

All measures operate on a 2×2 contingency table derived from four values:

- ``n_ii``: observed co-occurrence count of the specific n-gram
- ``n_ix``: total count of n-grams containing word 1 (row marginal)
- ``n_xi``: total count of n-grams containing word 2 (column marginal)
- ``n_xx``: total number of n-gram tokens in the corpus

The contingency table:

+------------+------------+-------------------+
|            | w2 present | w2 absent         |
+============+============+===================+
| w1 present | n_ii       | n_ix - n_ii       |
+------------+------------+-------------------+
| w1 absent  | n_xi - n_ii| n_xx-n_ix-n_xi+n_ii|
+------------+------------+-------------------+
"""

from __future__ import annotations

import math

__all__: list[str] = [
    "BigramAssocMeasures",
    "TrigramAssocMeasures",
]


def _ln_or_zero(x: float) -> float:
    """Safe natural log: returns 0.0 for non-positive values."""
    return math.log(x) if x > 0 else 0.0


class BigramAssocMeasures:
    """Association measures for bigram collocations.

    All methods are static and share the signature
    ``(n_ii, n_ix, n_xi, n_xx) -> float``.

    :Example:
    ::

        from pythainlp.collocation import BigramAssocMeasures

        # bigram "กิน ข้าว" appears 50 times
        # "กิน" appears 200 times, "ข้าว" appears 300 times
        # corpus has 100000 bigrams total
        score = BigramAssocMeasures.pmi(50, 200, 300, 100000)
    """

    @staticmethod
    def raw_freq(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Raw frequency (baseline).

        :param int n_ii: bigram count
        :param int n_ix: row marginal (word 1 count)
        :param int n_xi: column marginal (word 2 count)
        :param int n_xx: total bigram tokens
        :return: raw frequency
        :rtype: float
        """
        if n_xx == 0:
            return 0.0
        return n_ii / n_xx

    @staticmethod
    def pmi(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Pointwise Mutual Information.

        ``PMI = log2( P(w1,w2) / (P(w1) * P(w2)) )``

        Best for finding distinctive, domain-specific collocations
        when combined with a minimum frequency filter.

        :return: PMI score (can be negative)
        :rtype: float
        """
        if n_ii == 0 or n_ix == 0 or n_xi == 0 or n_xx == 0:
            return -float("inf")
        return math.log2((n_ii * n_xx) / (n_ix * n_xi))

    @staticmethod
    def npmi(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Normalized Pointwise Mutual Information.

        ``NPMI = PMI / -log2(P(w1,w2))``

        Range: [-1, +1] where +1 = perfect co-occurrence,
        0 = independence, -1 = never co-occur.
        Comparable across corpora of different sizes.

        :return: NPMI score in range [-1, +1]
        :rtype: float
        """
        if n_ii == 0 or n_ix == 0 or n_xi == 0 or n_xx == 0:
            return -1.0
        pmi_val = math.log2((n_ii * n_xx) / (n_ix * n_xi))
        neg_log_p = -math.log2(n_ii / n_xx)
        if neg_log_p == 0:
            return 0.0
        return pmi_val / neg_log_p

    @staticmethod
    def likelihood_ratio(
        n_ii: int, n_ix: int, n_xi: int, n_xx: int
    ) -> float:
        """Dunning's log-likelihood ratio (G-squared).

        The best general-purpose association measure. Works well for
        both frequent and rare collocations. Statistically well-grounded.

        :return: G-squared score (always >= 0)
        :rtype: float
        """
        if n_xx == 0:
            return 0.0

        n_io = n_ix - n_ii
        n_oi = n_xi - n_ii
        n_oo = n_xx - n_ix - n_xi + n_ii

        # Expected values
        r1 = n_ix
        r2 = n_xx - n_ix
        c1 = n_xi
        c2 = n_xx - n_xi

        e_ii = (r1 * c1) / n_xx if n_xx else 0
        e_io = (r1 * c2) / n_xx if n_xx else 0
        e_oi = (r2 * c1) / n_xx if n_xx else 0
        e_oo = (r2 * c2) / n_xx if n_xx else 0

        g2 = 2.0 * (
            n_ii * _ln_or_zero(n_ii / e_ii if e_ii else 0)
            + n_io * _ln_or_zero(n_io / e_io if e_io else 0)
            + n_oi * _ln_or_zero(n_oi / e_oi if e_oi else 0)
            + n_oo * _ln_or_zero(n_oo / e_oo if e_oo else 0)
        )
        return g2

    @staticmethod
    def chi_sq(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Pearson's chi-squared test.

        ``chi² = N * (ad - bc)² / (R1 * R2 * C1 * C2)``

        Good for medium-to-high frequency collocation detection.

        :return: chi-squared score (always >= 0)
        :rtype: float
        """
        if n_xx == 0:
            return 0.0

        n_io = n_ix - n_ii
        n_oi = n_xi - n_ii
        n_oo = n_xx - n_ix - n_xi + n_ii

        denom = n_ix * (n_xx - n_ix) * n_xi * (n_xx - n_xi)
        if denom == 0:
            return 0.0

        return n_xx * (n_ii * n_oo - n_io * n_oi) ** 2 / denom

    @staticmethod
    def t_score(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Student's t-score.

        ``t = (O - E) / sqrt(O)``

        Favors high-frequency collocations. Good for finding common,
        everyday collocations and language learning applications.

        :return: t-score
        :rtype: float
        """
        if n_ii == 0 or n_xx == 0:
            return 0.0
        expected = (n_ix * n_xi) / n_xx
        return (n_ii - expected) / math.sqrt(n_ii)

    @staticmethod
    def dice(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Dice coefficient.

        ``Dice = 2 * n_ii / (n_ix + n_xi)``

        Range: [0, 1]. Simple and intuitive.

        :return: Dice score in range [0, 1]
        :rtype: float
        """
        denom = n_ix + n_xi
        if denom == 0:
            return 0.0
        return 2 * n_ii / denom

    @staticmethod
    def log_dice(n_ii: int, n_ix: int, n_xi: int, n_xx: int) -> float:
        """Log-Dice (Rychly 2008).

        ``logDice = 14 + log2(2 * n_ii / (n_ix + n_xi))``

        Corpus-size independent. Scores are comparable across corpora.
        Theoretical maximum of 14. Preferred by lexicographers.

        :return: log-Dice score (typically 0-14)
        :rtype: float
        """
        denom = n_ix + n_xi
        if denom == 0 or n_ii == 0:
            return -float("inf")
        dice_val = 2 * n_ii / denom
        return 14 + math.log2(dice_val)


class TrigramAssocMeasures(BigramAssocMeasures):
    """Association measures for trigram collocations.

    Inherits all bigram measures. The trigram finder decomposes
    ``(w1, w2, w3)`` into leading-bigram ``(w1, w2)`` and trailing
    word ``w3`` for scoring with the same contingency table signature.
    """

    pass
