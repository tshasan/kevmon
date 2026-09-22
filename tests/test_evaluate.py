"""evaluate.py's win-rate interval. The point of reporting an interval at all is that the default battle counts are
small enough for the interval to be wider than most of the improvements anyone is trying to measure."""
from __future__ import annotations

import pytest

from evaluate import newcombe, wilson
from kev_showdown.player import normalised_entropy


def test_interval_brackets_the_observed_rate() -> None:
    low, high = wilson(50, 100)
    assert low < 0.5 < high


def test_interval_stays_inside_zero_and_one_at_the_extremes() -> None:
    """The normal approximation reports an upper bound above 1.0 for a perfect record, which then gets printed as a
    '104%' confidence bound. Wilson does not."""
    low, high = wilson(100, 100)
    assert 0.0 <= low <= 1.0 and high == pytest.approx(1.0)
    low, high = wilson(0, 100)
    assert low == pytest.approx(0.0) and 0.0 <= high <= 1.0


def test_interval_narrows_as_battles_accumulate() -> None:
    """The headline number in the README: ~+-10 points at 100 battles, and you need roughly 800 to resolve a
    5-point difference."""
    width_100 = wilson(50, 100)[1] - wilson(50, 100)[0]
    width_800 = wilson(400, 800)[1] - wilson(400, 800)[0]
    assert 0.19 < width_100 < 0.21
    assert width_800 < 0.08


def test_no_battles_does_not_divide_by_zero() -> None:
    assert wilson(0, 0) == (0.0, 0.0)


def test_difference_interval_contains_zero_for_equal_models() -> None:
    """Two models with identical records have not been shown to differ, and the interval has to say so -- this is the
    check that stops a run reporting a coin flip as an improvement."""
    low, high = newcombe(400, 800, 400, 800)
    assert low < 0 < high


def test_difference_interval_excludes_zero_for_a_large_real_gap() -> None:
    low, high = newcombe(600, 800, 400, 800)
    assert low > 0, "a 75% vs 50% record over 800 battles each is not ambiguous"


def test_a_five_point_gap_is_not_significant_at_a_hundred_battles() -> None:
    """The reason evaluate.py defaults high and the README pushes 800: at 100 battles a 5-point lead is noise."""
    low, high = newcombe(55, 100, 50, 100)
    assert low < 0 < high


def test_entropy_is_zero_when_the_policy_has_collapsed() -> None:
    assert normalised_entropy({"a": 1.0, "b": 0.0, "c": 0.0}) == 0.0


def test_entropy_is_one_when_the_policy_is_uniform() -> None:
    assert normalised_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)


def test_entropy_is_comparable_across_different_option_counts() -> None:
    """Normalising by log(n) is what makes the number meaningful: a uniform policy over 3 options and one over 12
    are equally undecided, and an un-normalised entropy would rank the 12 as far more so."""
    assert normalised_entropy({k: 1 / 3 for k in "abc"}) == pytest.approx(normalised_entropy({k: 1 / 12 for k in "abcdefghijkl"}))
