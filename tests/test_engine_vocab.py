"""Every string `vocab` can emit must be one the Rust engine actually accepts.

poke-engine validates nothing at construction time -- a bad enum string surfaces much later as a Rust panic inside
the search. So these tests drive the real engine rather than comparing against the transcribed frozensets, which
are themselves just a hand copy of the Rust source and could be wrong in the same direction as the mapping.
"""
from __future__ import annotations

import poke_engine as pe
import pytest
from poke_env.battle import Field, SideCondition, Status, Weather

from kev_showdown.engine import vocab


def _search(**state_kwargs: object) -> object:
    """A minimal legal 1v1 state, searched briefly. Raises PanicException if any enum string is unacceptable."""
    side = lambda species: pe.Side(  # noqa: E731
        pokemon=[pe.Pokemon(id=species, hp=100, maxhp=100, moves=[pe.Move(id="tackle", pp=16)])],
        active_index="0",
    )
    state = pe.State(side_one=side("venusaur"), side_two=side("skarmory"), **state_kwargs)  # type: ignore[arg-type]
    return pe.monte_carlo_tree_search(state, duration_ms=5)


@pytest.mark.parametrize("member", list(Weather))
def test_every_weather_maps_to_a_string_the_engine_accepts(member: Weather) -> None:
    assert _search(weather=vocab.weather(member)).total_visits > 0


@pytest.mark.parametrize("member", list(Status))
def test_every_status_maps_to_a_string_the_engine_accepts(member: Status) -> None:
    mapped = vocab.status(member)
    side = pe.Side(pokemon=[pe.Pokemon(id="venusaur", hp=100, maxhp=100, status=mapped,
                                       moves=[pe.Move(id="tackle", pp=16)])], active_index="0")
    other = pe.Side(pokemon=[pe.Pokemon(id="skarmory", hp=100, maxhp=100,
                                        moves=[pe.Move(id="tackle", pp=16)])], active_index="0")
    assert pe.monte_carlo_tree_search(pe.State(side_one=side, side_two=other), duration_ms=5).total_visits > 0


@pytest.mark.parametrize("member", [f for f in Field if f in vocab.TERRAIN])
def test_every_mapped_terrain_is_accepted(member: Field) -> None:
    assert _search(terrain=vocab.terrain({member: 1})).total_visits > 0


def test_side_condition_kwargs_are_real_constructor_arguments() -> None:
    """A wrong-but-plausible name (reflect vs light_screen) is invisible at runtime, so pin every one of them."""
    every = {c: 1 for c in vocab.SIDE_CONDITIONS}
    kwargs = vocab.side_condition_kwargs(every, turn=1)
    assert kwargs, "mapping produced nothing"
    pe.SideConditions(**kwargs)  # raises TypeError on any name the engine does not have


def test_poke_env_abbreviations_would_have_panicked() -> None:
    """The regression this module exists to prevent: poke-env's own status names are not the engine's."""
    with pytest.raises(BaseException):  # noqa: B017 - it is a Rust PanicException, not an Exception subclass
        side = pe.Side(pokemon=[pe.Pokemon(id="venusaur", hp=100, maxhp=100, status="brn",
                                           moves=[pe.Move(id="tackle", pp=16)])], active_index="0")
        other = pe.Side(pokemon=[pe.Pokemon(id="skarmory", hp=100, maxhp=100,
                                            moves=[pe.Move(id="tackle", pp=16)])], active_index="0")
        pe.monte_carlo_tree_search(pe.State(side_one=side, side_two=other), duration_ms=5)


def test_status_and_weather_tables_are_exhaustive() -> None:
    """Unmapped members must raise at the boundary rather than silently becoming 'none'."""
    assert set(vocab.STATUS) == set(Status)
    assert set(vocab.WEATHER) == set(Weather)


def test_stacking_hazards_pass_layers_through_but_screens_convert_to_turns_remaining() -> None:
    kwargs = vocab.side_condition_kwargs(
        {SideCondition.SPIKES: 3, SideCondition.REFLECT: 2}, turn=4)
    assert kwargs["spikes"] == 3                       # layers, not turns
    assert kwargs["reflect"] == vocab.SCREEN_DURATION - (4 - 2)
    assert vocab.side_condition_kwargs({SideCondition.REFLECT: 1}, turn=99)["reflect"] == 0   # clamped, never negative
