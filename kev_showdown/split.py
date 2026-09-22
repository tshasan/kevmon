"""Battle-level partitioning, kept as a pure function so it can be unit-tested without playing a single battle.
This is the one thing that must never leak: every turn from a given battle has to land in the same split, or a
held-out score stops measuring generalisation.

Splits by whole battle, not by state (kev's own split_data.py groups by exact state hash, which is the right
choice for its independent-record domains, but wrong for ours: a Pokemon battle is a sequential trajectory, so
two different turns of the *same* battle would almost never hash to the same state and could land in different
splits, leaking that battle's trajectory across them)."""
from __future__ import annotations

import random


def battle_split(tags: list[str], fractions: dict[str, float], rng: random.Random) -> dict[str, list[str]]:
    """Partition battle tags by whole battle into named splits sized by `fractions` (e.g.
    {"calibration": 0.1, "development": 0.15}), plus a "train" split holding everything left over. Every named
    split gets at least one battle when `tags` is non-empty."""
    if not tags:
        raise ValueError("no battle tags to split")
    if not fractions:
        raise ValueError("at least one named split fraction is required")
    if "train" in fractions:
        raise ValueError('"train" is the remainder split and cannot be given a fraction')
    if any(not 0.0 < f < 1.0 for f in fractions.values()) or sum(fractions.values()) >= 1.0:
        raise ValueError('split fractions must each be between 0 and 1 and sum to less than 1 (the remainder is "train")')

    shuffled = list(tags)
    rng.shuffle(shuffled)
    out: dict[str, list[str]] = {}
    start = 0
    for name, frac in fractions.items():
        n = min(len(shuffled) - start, max(1, round(len(shuffled) * frac)))
        out[name] = shuffled[start:start + n]
        start += n
    out["train"] = shuffled[start:]
    return out
