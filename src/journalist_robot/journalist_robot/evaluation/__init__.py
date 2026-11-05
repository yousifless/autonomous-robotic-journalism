"""Internal acceptance score for article drafts (Sec. III-C, Table II, Eq. 4).

    S = 0.35 * S_fact + 0.25 * S_rel + 0.20 * S_cons + 0.20 * S_lang

The signals are model-based and partly reuse models from the perception and
description stages, so S is an internal workflow gate, not an independent
measure of article quality or factual accuracy.
"""
