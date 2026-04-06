# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""Thai collocation extraction.

Provides statistical collocation finders and association measures
for identifying significant word co-occurrences in Thai text.

Follows a Build-Filter-Score pattern:

1. **Build** a finder from tokenized words, a corpus, or TNC data
2. **Filter** by frequency, word properties, or POS patterns
3. **Score** with one of 8 association measures

:Example:
::

    from pythainlp.tokenize import word_tokenize
    from pythainlp.collocation import (
        BigramCollocationFinder,
        BigramAssocMeasures,
    )

    words = word_tokenize("ฉันกินข้าวที่โรงเรียนทุกวัน")
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(2)
    top = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
"""

from __future__ import annotations

__all__: list[str] = [
    "BigramAssocMeasures",
    "BigramCollocationFinder",
    "TrigramAssocMeasures",
    "TrigramCollocationFinder",
]

from pythainlp.collocation.core import (
    BigramCollocationFinder,
    TrigramCollocationFinder,
)
from pythainlp.collocation.measures import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
)
