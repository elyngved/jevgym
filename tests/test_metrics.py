import math

import pytest

from jevgym import metrics


def test_wilson_known_value():
    lo, hi = metrics.wilson_interval(80, 100)
    assert lo == pytest.approx(0.7112, abs=1e-4)
    assert hi == pytest.approx(0.8666, abs=1e-4)


def test_brier_and_log_loss():
    probs = [[1.0, 0.0], [0.5, 0.5]]
    assert metrics.brier(probs, [0, 1]) == pytest.approx((0 + 0.5) / 2)
    assert metrics.log_loss(probs, [0, 1]) == pytest.approx(math.log(2) / 2)
    assert metrics.log_loss([[1.0, 0.0]], [1]) == pytest.approx(-math.log(metrics.LOG_LOSS_EPS))
    assert metrics.LOG_LOSS_EPS == 0.005 and metrics.LOG_LOSS_EPS_STRICT == 1e-15


def test_normalize_and_confusion():
    assert metrics.normalize({"a": 1, "b": 3}, ["a", "b"]) == [0.25, 0.75]
    assert metrics.confusion(["a", "b", "b"], ["a", "a", "b"], ["a", "b"]) == [[1, 0], [1, 1]]


def test_wilson_edges_hand_values():
    # 0 of 10: centre = (z^2/20)/(1+z^2/10), half equal to centre, so the lower bound is exactly 0.
    lo, hi = metrics.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    # 857/1000 is the S1 B3 run.
    lo, hi = metrics.wilson_interval(857, 1000)
    assert (lo, hi) == (pytest.approx(0.83394, abs=1e-5), pytest.approx(0.87733, abs=1e-5))


def test_paired_counts():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert metrics.paired_counts(a, b) == {"both": 1, "only_a": 2, "only_b": 1, "neither": 1}
    with pytest.raises(ValueError):
        metrics.paired_counts([True], [True, False])


def test_mcnemar_exact_hand_values():
    # n=6, k=1: P(X<=1) = (1+6)/64, two-sided = 14/64.
    assert metrics.mcnemar_exact(1, 5) == pytest.approx(14 / 64)
    assert metrics.mcnemar_exact(5, 1) == pytest.approx(14 / 64)
    # n=10, k=0: 2 * 1/1024.
    assert metrics.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    # n=20, k=5: P(X<=5) = (1+20+190+1140+4845+15504)/2^20 = 21700/1048576.
    assert metrics.mcnemar_exact(5, 15) == pytest.approx(2 * 21700 / 1048576)
    # Balanced and empty cases cap at 1.
    assert metrics.mcnemar_exact(3, 3) == 1.0
    assert metrics.mcnemar_exact(0, 0) == 1.0


def test_brier_multiclass_hand_value():
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14; full-confidence miss = 2.
    assert metrics.brier([[0.7, 0.2, 0.1]], [0]) == pytest.approx(0.14)
    assert metrics.brier([[0.0, 1.0, 0.0]], [0]) == pytest.approx(2.0)


def test_log_loss_clip_is_applied_and_selectable():
    # True label at p=0: cost is -ln(eps); with eps=0.005 that is ln(200).
    assert metrics.log_loss([[0.0, 1.0]], [0], eps=0.005) == pytest.approx(math.log(200))
    assert metrics.log_loss([[0.25, 0.75]], [0], eps=0.005) == pytest.approx(math.log(4))
