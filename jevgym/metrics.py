"""Metrics: accuracy with a 95% Wilson interval, exact McNemar on paired rows, Brier score, log loss, confusion counts."""

from __future__ import annotations

import math
from typing import Sequence

Z95 = 1.959963984540054
# Probability clip for log loss. Jev returns probabilities in whole percents, so "0" means "below 0.5%";
# clipping at half a rounding step treats it that way. The 1e-15 version is
# kept in every summary as log_loss_strict, next to the count of rows with 0 on the true label.
LOG_LOSS_EPS = 0.005
LOG_LOSS_EPS_STRICT = 1e-15


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def normalize(probs: dict[str, float], option_ids: Sequence[str]) -> list[float]:
    """Probabilities in option order, renormalized to sum to 1 (missing options count as 0)."""
    vals = [float(probs.get(o, 0.0)) for o in option_ids]
    s = sum(vals)
    if s <= 0:
        raise ValueError(f"probabilities sum to {s}")
    return [v / s for v in vals]


def brier(prob_rows: Sequence[Sequence[float]], label_idx: Sequence[int]) -> float:
    """Multiclass Brier score: mean over rows of sum over options of (p - y)^2. Range 0..2."""
    total = 0.0
    for ps, y in zip(prob_rows, label_idx):
        total += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(ps))
    return total / len(prob_rows)


def log_loss(prob_rows: Sequence[Sequence[float]], label_idx: Sequence[int], eps: float = LOG_LOSS_EPS) -> float:
    """Mean negative natural log of the probability on the true label, clipped at eps."""
    total = 0.0
    for ps, y in zip(prob_rows, label_idx):
        total += -math.log(max(ps[y], eps))
    return total / len(prob_rows)


def confusion(labels: Sequence[str], preds: Sequence[str], option_ids: Sequence[str]) -> list[list[int]]:
    """Counts [true][predicted] in option order."""
    idx = {o: i for i, o in enumerate(option_ids)}
    m = [[0] * len(option_ids) for _ in option_ids]
    for y, p in zip(labels, preds):
        m[idx[y]][idx[p]] += 1
    return m


def paired_counts(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, int]:
    """The four McNemar cells for two arms scored on the same rows, in the same order.

    both: both right. only_a: A right, B wrong. only_b: A wrong, B right. neither: both wrong.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(f"unpaired: {len(correct_a)} vs {len(correct_b)} rows")
    cells = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
    for a, b in zip(correct_a, correct_b):
        cells["both" if a and b else "only_a" if a else "only_b" if b else "neither"] += 1
    return cells


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    Under "no difference", each discordant row is a fair coin flip between A and B, so
    min(only_a, only_b) ~ Binomial(n, 0.5) with n = only_a + only_b. The p-value doubles the smaller
    tail and caps at 1. Rows where both arms agree carry no information about the difference.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
