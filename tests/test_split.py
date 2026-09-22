"""kev_showdown.split.battle_split: the one guarantee that must hold or heldout accuracy is meaningless -- every
turn from a battle lands in exactly one split, never both."""
from __future__ import annotations

import random

import pytest

from kev_showdown.split import battle_split


def test_split_partitions_every_tag_exactly_once() -> None:
    tags = [f"battle-{i}" for i in range(200)]
    splits = battle_split(tags, heldout_frac=0.15, rng=random.Random(0))
    train, heldout = set(splits["train"]), set(splits["heldout"])
    assert train | heldout == set(tags)
    assert train & heldout == set()                       # a battle can never straddle the split
    assert len(splits["train"]) + len(splits["heldout"]) == len(tags)


@pytest.mark.parametrize("seed", range(10))
def test_split_never_leaks_across_seeds(seed: int) -> None:
    """Same property, many seeds and sizes: a leakage bug that only shows up for some shuffles would slip past a
    single fixed-seed test."""
    tags = [f"battle-{i}" for i in range(50 + seed)]
    splits = battle_split(tags, heldout_frac=0.2, rng=random.Random(seed))
    assert set(splits["train"]) & set(splits["heldout"]) == set()
    assert set(splits["train"]) | set(splits["heldout"]) == set(tags)


def test_split_respects_fraction_within_rounding() -> None:
    tags = [f"battle-{i}" for i in range(1000)]
    splits = battle_split(tags, heldout_frac=0.15, rng=random.Random(0))
    assert 140 <= len(splits["heldout"]) <= 160


def test_split_is_deterministic_given_a_seeded_rng() -> None:
    tags = [f"battle-{i}" for i in range(100)]
    a = battle_split(tags, 0.15, random.Random(42))
    b = battle_split(tags, 0.15, random.Random(42))
    assert a == b


def test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        battle_split([], 0.15, random.Random(0))


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_out_of_range_fraction(bad_frac: float) -> None:
    with pytest.raises(ValueError):
        battle_split(["a", "b"], bad_frac, random.Random(0))


def test_split_always_holds_out_at_least_one_battle() -> None:
    """A tiny dataset with a small fraction must not round down to an empty heldout set (silent 100%-train)."""
    splits = battle_split(["only-battle"], heldout_frac=0.01, rng=random.Random(0))
    assert len(splits["heldout"]) == 1
    assert splits["train"] == []
