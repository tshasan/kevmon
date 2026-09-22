"""kev_showdown.common: battle -> Kev request. All of these run against a real poke-env Battle (tests/conftest.py),
never a hand-mocked stand-in, so they catch drift in poke-env itself, not just in our glue code."""
from __future__ import annotations

import random

from poke_env.battle import Battle

from kev_showdown.common import build_options, build_request, order_key, render_state
from kev_showdown.types import Choice, Noul


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
        action = req.questions["action"]
        assert isinstance(action, Choice)
        orderings.add(tuple(action.criteria))
    assert len(orderings) > 1


def test_build_request_shape(battle: Battle) -> None:
    options = build_options(battle)
    req = build_request(battle, options, random.Random(0), ask_win=True)
    assert isinstance(req.questions["action"], Choice)
    assert isinstance(req.questions["win"], Noul)
    assert set(req.questions["action"].criteria) == set(options)

    req_no_win = build_request(battle, options, random.Random(0), ask_win=False)
    assert "win" not in req_no_win.questions


def test_build_request_is_json_serialisable_without_none_holes(battle: Battle) -> None:
    """What actually goes over the wire: dumping and re-validating must round-trip losslessly."""
    from kev_showdown.types import SystemOneRequest

    options = build_options(battle)
    req = build_request(battle, options, random.Random(0))
    dumped = req.model_dump(mode="json", exclude_none=True)
    assert SystemOneRequest.model_validate(dumped) == req
