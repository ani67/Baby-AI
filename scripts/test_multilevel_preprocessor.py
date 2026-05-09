"""Multi-resolution preprocessor smoke test.

Verifies that one paragraph of source text yields items at every
granularity (word / phrase / sentence / paragraph) with the expected
surprise multipliers, and that the multi-level item count is at least
3× the sentence-only count (the practical floor; the spec target is
6-8× and we report whatever we observe).
"""
from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.multilevel_preprocessor import extract_multilevel    # noqa: E402


def main() -> int:
    test = """
Justice is the advantage of the stronger, said Thrasymachus.
Socrates disagreed, arguing that justice benefits all members of society.
The just man, unlike the unjust, does not seek to outdo his equals.
"""

    items = extract_multilevel(test)
    by_level: dict[str, list[str]] = {}
    multipliers: dict[str, float] = {}
    for item in items:
        by_level.setdefault(item.level, []).append(item.text)
        multipliers[item.level] = item.surprise_multiplier

    for level in ("word", "phrase", "sentence", "paragraph"):
        texts = by_level.get(level, [])
        mult = multipliers.get(level, "?")
        print(f"{level}: {len(texts)} items  (multiplier={mult})")
        if texts:
            print(f"  examples: {texts[:3]}")

    sentences = by_level.get("sentence", [])
    ratio = len(items) / max(len(sentences), 1)
    print()
    print(f"total items       = {len(items)}")
    print(f"sentence-only     = {len(sentences)}")
    print(f"ratio vs sentence = {ratio:.1f}x")

    # Spec floor — at least 3× the sentence-only count. Spec target 6-8×.
    assert ratio >= 3.0, f"expected ratio >= 3x, got {ratio:.1f}x"
    # Multipliers must match the locked design.
    assert multipliers.get("word")      == 1.5
    assert multipliers.get("phrase")    == 1.3
    assert multipliers.get("sentence")  == 1.0
    assert multipliers.get("paragraph") == 0.8

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
