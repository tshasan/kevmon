"""Final stats for a Pokemon the opponent has not fully revealed.

poke-engine takes *final* stats, never base stats: `calculate_stats_from_base_stats()` exists in the Rust crate but
is not exposed through PyO3. For our own side none of this is needed -- poke-env hands us real `mon.stats` -- so
everything here applies only to opponents, which bounds the blast radius of a mistake to the opponent model.

The EV/IV rules are NOT "85 everywhere, IV 31". gen9's random-battle generator
(pokemon-showdown/data/random-battles/gen9/teams.ts, ~1546-1600) applies deterministic corrections after it picks a
moveset, and they move speed tiers and bulk -- exactly what a search's decisions turn on:

  * `evs.atk = 0` AND `ivs.atk = 0` when the set has no physical attacking move. This fires on essentially every
    special attacker, so it is the common case. The zeroed IV matters on its own too: it is what makes Foul Play and
    confusion damage small.
  * `evs.spe = 0` and `ivs.spe = 0` on Gyro Ball / Trick Room sets.
  * HP EVs step down 4 at a time to land on Stealth Rock / Sitrus thresholds. Small effect, needs the type chart and
    the item to reproduce; NOT implemented here, and `DEFAULT_EV` is used instead. Recorded as a known approximation.

Nature needs no handling at all: gen9 `randomSet()` never assigns one, and poke-engine defaults to "serious", which
is neutral. That is exact, not an approximation.

Measured fidelity against 40 real gen9 randbats battles (240 Pokemon), comparing this derivation with the stats
Showdown actually assigned to our own side -- the free version of the sampler-fidelity check, since poke-env gives
us both the ground truth and the inputs:

    def 99.6%   spa 99.6%   spd 99.6%   spe 99.6%   atk 99.2%   hp 97.9%   exact

The entire residue is two understood cases, neither worth chasing:
  * HP off by 1-2 on Belly-Drum/Sitrus and Substitute+Endeavor sets -- the unimplemented HP EV step-down.
  * Minior, whose Shields Down formes (Meteor vs Core) carry different base stats, so a forme-resolution mismatch
    swaps its offensive and defensive stats. One species.
Re-run `scratchpad/check_stats.py` after any change here; a regression shows up immediately as a column dropping
off 99%.
"""
from __future__ import annotations

DEFAULT_EV = 85
DEFAULT_IV = 31
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# Physical moves that scale off something other than the user's Attack, so their presence does NOT keep Attack EVs.
# Mirrors teams.ts's `noAttackStatMoves` carve-outs.
_NOT_ATTACK_SCALING = frozenset({"bodypress", "foulplay", "seismictoss", "nightshade", "dragonrage", "sonicboom"})


def hp_stat(base: int, level: int, iv: int = DEFAULT_IV, ev: int = DEFAULT_EV) -> int:
    return ((2 * base + iv + ev // 4) * level) // 100 + level + 10


def other_stat(base: int, level: int, iv: int = DEFAULT_IV, ev: int = DEFAULT_EV) -> int:
    """Neutral nature, so no ±10% multiplier -- see the module docstring on why that is exactly right here."""
    return ((2 * base + iv + ev // 4) * level) // 100 + 5


def _tera_blast_is_physical(species: str, ability: str | None, move_ids: frozenset[str],
                            base_stats: dict[str, int] | None) -> bool:
    """teams.ts treats Tera Blast as physical for Porygon2, Contrary/Defiant, Shift Gear users, and any Pokemon
    whose base Attack exceeds base Special Attack."""
    if species == "porygon2" or (ability or "").lower() in {"contrary", "defiant"}:
        return True
    if "shiftgear" in move_ids:
        return True
    return base_stats is not None and base_stats["atk"] > base_stats["spa"]


def randbats_evs_ivs(move_ids: frozenset[str], categories: dict[str, str], *, species: str = "",
                     ability: str | None = None,
                     base_stats: dict[str, int] | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """The deterministic EV/IV corrections above, given a set's moves and a move-id -> category lookup.

    `categories` maps a move id to "Physical" / "Special" / "Status" (from `GenData.from_gen(9).moves`). Moves we
    cannot classify are treated as non-physical, which errs toward zeroing Attack; that is the direction that
    matches the generator's own default for special attackers.

    `species`/`ability`/`base_stats` are optional but close the last two gaps against the real generator -- without
    them Ditto and Porygon2 come out ~45 Attack short, which validation against real battles catches immediately.
    """
    evs = dict.fromkeys(STAT_KEYS, DEFAULT_EV)
    ivs = dict.fromkeys(STAT_KEYS, DEFAULT_IV)

    physical = {m for m in move_ids
                if categories.get(m) == "Physical" and m not in _NOT_ATTACK_SCALING}
    if "terablast" in move_ids and _tera_blast_is_physical(species, ability, move_ids, base_stats):
        physical.add("terablast")
    # Transform copies the target's stats, so the generator never zeroes Ditto's Attack.
    if not physical and "transform" not in move_ids:
        evs["atk"] = 0
        ivs["atk"] = 0

    if move_ids & {"gyroball", "trickroom"}:
        evs["spe"] = 0
        ivs["spe"] = 0

    return evs, ivs


def final_stats(base_stats: dict[str, int], level: int,
                evs: dict[str, int] | None = None, ivs: dict[str, int] | None = None) -> dict[str, int]:
    """Base stats -> the six final stats poke-engine wants."""
    evs = evs or dict.fromkeys(STAT_KEYS, DEFAULT_EV)
    ivs = ivs or dict.fromkeys(STAT_KEYS, DEFAULT_IV)
    out = {"hp": hp_stat(base_stats["hp"], level, ivs["hp"], evs["hp"])}
    for key in ("atk", "def", "spa", "spd", "spe"):
        out[key] = other_stat(base_stats[key], level, ivs[key], evs[key])
    return out
