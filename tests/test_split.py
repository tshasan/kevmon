"""kev_showdown.split.battle_split: the one guarantee that must hold or a held-out score is meaningless -- every
turn from a battle lands in exactly one split, never more than one."""
from __future__ import annotations

import random

import pytest

from kev_showdown.split import battle_split


def test_split_partitions_every_tag_exactly_once() -> None:
    tags = [f"battle-{i}" for i in range(200)]
    splits = battle_split(tags, {"calibration": 0.1, "development": 0.15}, random.Random(0))
    train, cal, dev = set(splits["train"]), set(splits["calibration"]), set(splits["development"])
    assert train | cal | dev == set(tags)
    assert train & cal == set() and train & dev == set() and cal & dev == set()   # no battle straddles two splits
    assert len(splits["train"]) + len(splits["calibration"]) + len(splits["development"]) == len(tags)


@pytest.mark.parametrize("seed", range(10))
def test_split_never_leaks_across_seeds(seed: int) -> None:
    """Same property, many seeds and sizes: a leakage bug that only shows up for some shuffles would slip past a
    single fixed-seed test."""
    tags = [f"battle-{i}" for i in range(50 + seed)]
    splits = battle_split(tags, {"calibration": 0.1, "development": 0.2}, random.Random(seed))
    all_splits = [set(v) for v in splits.values()]
    assert set().union(*all_splits) == set(tags)
    assert sum(len(s) for s in all_splits) == len(tags)                          # no double-counting -> no overlap


def test_split_respects_fractions_within_rounding() -> None:
    tags = [f"battle-{i}" for i in range(1000)]
    splits = battle_split(tags, {"calibration": 0.1, "development": 0.15}, random.Random(0))
    assert 90 <= len(splits["calibration"]) <= 110
    assert 140 <= len(splits["development"]) <= 160
    assert 740 <= len(splits["train"]) <= 760


def test_split_is_deterministic_given_a_seeded_rng() -> None:
    tags = [f"battle-{i}" for i in range(100)]
    a = battle_split(tags, {"calibration": 0.1, "development": 0.15}, random.Random(42))
    b = battle_split(tags, {"calibration": 0.1, "development": 0.15}, random.Random(42))
    assert a == b


def test_split_works_with_a_single_named_split_too() -> None:
    """The 2-way case (just one named split, e.g. {"heldout": 0.15}) still works -- fractions is a dict, not
    hardcoded to the 3-way train/calibration/development shape."""
    tags = [f"battle-{i}" for i in range(200)]
    splits = battle_split(tags, {"heldout": 0.15}, random.Random(0))
    assert set(splits) == {"train", "heldout"}
    assert set(splits["train"]) & set(splits["heldout"]) == set()


def test_split_rejects_empty_tags() -> None:
    with pytest.raises(ValueError):
        battle_split([], {"development": 0.15}, random.Random(0))


def test_split_rejects_no_fractions() -> None:
    with pytest.raises(ValueError):
        battle_split(["a", "b"], {}, random.Random(0))


def test_split_rejects_train_as_an_explicit_fraction() -> None:
    with pytest.raises(ValueError):
        battle_split(["a", "b"], {"train": 0.5}, random.Random(0))


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_out_of_range_fraction(bad_frac: float) -> None:
    with pytest.raises(ValueError):
        battle_split(["a", "b"], {"development": bad_frac}, random.Random(0))


def test_split_rejects_fractions_summing_to_one_or_more() -> None:
    with pytest.raises(ValueError):
        battle_split(["a", "b", "c"], {"calibration": 0.5, "development": 0.5}, random.Random(0))


def test_split_always_holds_out_at_least_one_battle_per_named_split() -> None:
    """A tiny dataset with a small fraction must not round down to an empty split (silent 100%-train)."""
    splits = battle_split(["only-battle", "another-battle"], {"development": 0.01}, random.Random(0))
    assert len(splits["development"]) == 1
