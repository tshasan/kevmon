"""kev_showdown.common: battle -> Kev request. All of these run against a real poke-env Battle (tests/conftest.py),
never a hand-mocked stand-in, so they catch drift in poke-env itself, not just in our glue code."""
from __future__ import annotations

import random

from poke_env.battle import Battle

from kev_showdown.common import build_options, build_request, order_key, render_state


def test_render_state_has_no_label_leaking_fields(battle: Battle) -> None:
    """The state must describe the position only -- never the outcome, the teacher's pick, or anything else that
    would let a model (or a human labeller) shortcut the decision instead of reading the board."""
    state = render_state(battle)
    forbidden = {"action", "label", "win", "won", "result", "teacher", "answer"}
    assert set(state) & forbidden == set()
    assert state["turn"] == 5

    our_active, opponent_active = state["our_active"], state["opponent_active"]
    assert isinstance(our_active, dict) and our_active["species"] == "garchomp"
    assert isinstance(opponent_active, dict) and opponent_active["species"] == "excadrill"


def test_mon_has_a_fixed_shape_even_with_no_active_pokemon(battle_no_active: Battle) -> None:
    """Before any switch-in, battle.active_pokemon is None (verified against poke-env directly -- not a fainting
    scenario). render_state must still produce the same 6 keys for our_active, not a smaller shape."""
    assert battle_no_active.active_pokemon is None
    state = render_state(battle_no_active)
    our_active = state["our_active"]
    assert isinstance(our_active, dict)
    assert set(our_active) == {"species", "types", "hp", "status", "boosts", "terastallized"}


def test_build_options_handles_a_forced_recharge_turn(battle_recharge: Battle) -> None:
    """Regression test: poke-env's fallback stub for the "recharge" pseudo-move (used after Hyper Beam) lacks a
    "priority" key, and move.priority used to raise KeyError here, crashing every collect.py/evaluate.py run that
    reached one. "recharge" itself isn't modeled (no real decision, and no way to describe it safely), but
    switching out -- a genuine legal alternative to recharging -- must still show up as an option."""
    options = build_options(battle_recharge)
    assert "switch to skarmory" in options
    assert not any("recharge" in key for key in options)


def test_build_options_keys_are_unique_and_legal(battle: Battle) -> None:
    """A key collision in build_options would silently drop a legal move from the request -- the model would never
    even see it as an option, which is a much sneakier bug than an outright crash."""
    options = build_options(battle)
    assert len(options) >= 2
    for key, (order, desc) in options.items():
        assert order_key(order) == key
        assert desc  # every option has a non-empty description the model can read


def test_option_order_is_shuffled_per_call(battle: Battle) -> None:
    """build_request must not always hand the model the same option order -- a fixed order lets a model learn
    'pick position N' instead of reading the descriptions, and that shortcut would silently inflate offline eval."""
    options = build_options(battle)
    orderings = set()
    for seed in range(20):
        req = build_request(battle, options, random.Random(seed))
        action = req["questions"]["action"]
        criteria = action["criteria"]
        assert action["type"] == "choice" and isinstance(criteria, dict)
        orderings.add(tuple(criteria))
    assert len(orderings) > 1


def test_build_request_shape(battle: Battle) -> None:
    options = build_options(battle)
    req = build_request(battle, options, random.Random(0), ask_win=True)
    action_criteria = req["questions"]["action"]["criteria"]
    assert req["questions"]["action"]["type"] == "choice"
    assert req["questions"]["win"]["type"] == "noul"
    assert isinstance(action_criteria, dict) and set(action_criteria) == set(options)

    req_no_win = build_request(battle, options, random.Random(0), ask_win=False)
    assert "win" not in req_no_win["questions"]


def test_build_request_is_json_serialisable(battle: Battle) -> None:
    """What actually goes over the wire: must round-trip through json.dumps/loads losslessly (no non-JSON types
    like tuples or Enums sneaking into option text)."""
    import json

    options = build_options(battle)
    req = build_request(battle, options, random.Random(0))
    assert json.loads(json.dumps(req)) == req
