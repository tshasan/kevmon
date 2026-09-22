"""Battle-level train/heldout partitioning, kept as a pure function so it can be unit-tested without playing a
single battle. This is the one thing that must never leak: every turn from a given battle has to land in the same
split, or heldout accuracy stops measuring generalisation."""
from __future__ import annotations

import random


def battle_split(tags: list[str], heldout_frac: float, rng: random.Random) -> dict[str, list[str]]:
    """Partition battle tags into {"train": [...], "heldout": [...]}, by whole battle, shuffled with `rng`."""
    if not tags:
        raise ValueError("no battle tags to split")
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError("heldout_frac must be between 0 and 1")
    shuffled = list(tags)
    rng.shuffle(shuffled)
    n_held = max(1, int(len(shuffled) * heldout_frac))
    return {"heldout": shuffled[:n_held], "train": shuffled[n_held:]}
