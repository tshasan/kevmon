"""poke-env enum names -> poke-engine enum strings, and nothing else.

This module exists because **poke-engine does not validate enum strings when you construct a State**. The Python
constructors store whatever you pass verbatim; the strings are only parsed into Rust enums later, inside
`monte_carlo_tree_search`/`generate_instructions`, where a bad one raises `pyo3_runtime.PanicException` from a Rust
panic rather than a normal Python error. Worse, the two libraries disagree on names in ways that look fine:

    poke-env Status.BRN        -> engine "burn"       ("brn" panics)
    poke-env Weather.SNOWSCAPE -> engine "snow"       ("snowscape" panics)

So every crossing is written down here, once, and checked against the engine's real vocabulary by
`tests/test_engine_vocab.py`. A mapping that silently returned the wrong-but-valid string would be undetectable at
runtime and would quietly corrupt every search, so the tables are exhaustive over the poke-env enums rather than
`dict.get(..., default)` lookups: an unmapped member raises here, at the boundary, naming itself.
"""
from __future__ import annotations

from poke_env.battle import Field, SideCondition, Status, Weather

# Engine vocabularies, transcribed from the Rust `define_enum_with_from_str!` blocks. Weather and Terrain declare no
# `default =`, so an unrecognised string is a panic, not a fallback.
ENGINE_WEATHER = frozenset({"none", "sun", "rain", "sand", "hail", "snow", "harshsun", "heavyrain"})
ENGINE_TERRAIN = frozenset({"none", "electricterrain", "psychicterrain", "mistyterrain", "grassyterrain"})
ENGINE_STATUS = frozenset({"none", "burn", "sleep", "freeze", "paralyze", "poison", "toxic"})

# Weather. poke-env's names are the Showdown move names; the engine's are the effects.
WEATHER: dict[Weather, str] = {
    Weather.SUNNYDAY: "sun",
    Weather.RAINDANCE: "rain",
    Weather.SANDSTORM: "sand",
    Weather.HAIL: "hail",
    Weather.SNOWSCAPE: "snow",
    Weather.DESOLATELAND: "harshsun",
    Weather.PRIMORDIALSEA: "heavyrain",
    # The engine's Weather enum has no Delta Stream. Rayquaza is not in gen9 randbats, so this costs us nothing
    # today, but it must degrade to "no weather" loudly rather than panic mid-search if that ever changes.
    Weather.DELTASTREAM: "none",
    Weather.UNKNOWN: "none",
}

# Terrain. poke-env keeps terrains in the same `Field` enum as Trick Room, Gravity and friends; the engine models
# only the four terrains (Trick Room is its own State field, the rest have no representation at all).
TERRAIN: dict[Field, str] = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}

# Status. Every poke-env abbreviation panics the engine, so this table is the whole reason this module exists.
# FNT has no engine status: a fainted Pokemon is hp=0 with status "none".
STATUS: dict[Status, str] = {
    Status.BRN: "burn",
    Status.SLP: "sleep",
    Status.FRZ: "freeze",
    Status.PAR: "paralyze",
    Status.PSN: "poison",
    Status.TOX: "toxic",
    Status.FNT: "none",
}

# Side conditions -> the `SideConditions` kwarg of the same effect. Members absent here have no engine
# representation (the pledges, every G-Max hazard, UNKNOWN) and are dropped by `side_condition_kwargs`.
SIDE_CONDITIONS: dict[SideCondition, str] = {
    SideCondition.AURORA_VEIL: "aurora_veil",
    SideCondition.CRAFTY_SHIELD: "crafty_shield",
    SideCondition.LIGHT_SCREEN: "light_screen",
    SideCondition.LUCKY_CHANT: "lucky_chant",
    SideCondition.MATBLOCK: "mat_block",
    SideCondition.MIST: "mist",
    SideCondition.QUICK_GUARD: "quick_guard",
    SideCondition.REFLECT: "reflect",
    SideCondition.SAFEGUARD: "safeguard",
    SideCondition.SPIKES: "spikes",
    SideCondition.STEALTH_ROCK: "stealth_rock",
    SideCondition.STICKY_WEB: "sticky_web",
    SideCondition.TAILWIND: "tailwind",
    SideCondition.TOXIC_SPIKES: "toxic_spikes",
    SideCondition.WIDE_GUARD: "wide_guard",
}

# poke-env reports these as a LAYER COUNT; everything else in `battle.side_conditions` is the turn the condition
# started, which has to be converted to turns remaining. Getting this backwards is silent and wrong.
STACKING_CONDITIONS = frozenset({SideCondition.SPIKES, SideCondition.TOXIC_SPIKES})

# Screens last 5 turns, or 8 with Light Clay -- which we cannot know for an opponent. Assume the common case and
# accept a bounded error near expiry rather than pretending to know.
SCREEN_DURATION = 5


def weather(w: Weather | None) -> str:
    if w is None:
        return "none"
    try:
        return WEATHER[w]
    except KeyError:
        raise KeyError(f"no poke-engine weather for {w!r}; add it to vocab.WEATHER") from None


def terrain(fields: dict[Field, int] | None) -> str:
    """The active terrain, or "none". poke-env's `battle.fields` mixes terrains with Trick Room etc."""
    for field in fields or {}:
        if field in TERRAIN:
            return TERRAIN[field]
    return "none"


def status(s: Status | None) -> str:
    if s is None:
        return "none"
    try:
        return STATUS[s]
    except KeyError:
        raise KeyError(f"no poke-engine status for {s!r}; add it to vocab.STATUS") from None


def side_condition_kwargs(conditions: dict[SideCondition, int], turn: int) -> dict[str, int]:
    """`battle.side_conditions` -> `poke_engine.SideConditions(**kwargs)`.

    Stacking hazards carry their layer count straight through. Everything else stores the turn it *started*, and the
    engine wants turns *remaining*, so it is converted and clamped at 0 (an expired screen poke-env has not yet
    cleared must not come through as a negative counter).
    """
    out: dict[str, int] = {}
    for condition, value in (conditions or {}).items():
        name = SIDE_CONDITIONS.get(condition)
        if name is None:
            continue                                  # no engine representation; dropping is the honest option
        if condition in STACKING_CONDITIONS:
            out[name] = int(value)
        else:
            out[name] = max(0, SCREEN_DURATION - (turn - int(value)))
    return out
