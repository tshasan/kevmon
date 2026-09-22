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
    assert state["speed"] == "unknown" and state["offense"] == "unknown"


def test_state_exposes_the_features_the_teacher_branches_on(battle: Battle) -> None:
    """SimpleHeuristicsPlayer is a deterministic function of the position, and two of its inputs are invisible in a
    types/HP-only rendering: the base-speed comparison in _estimate_matchup, and the atk/def vs spa/spd ratio that
    decides physical or special. Without these the imitation ceiling sits below 100% for reasons no amount of
    training data can fix, so they are rendered explicitly rather than left as arithmetic for a small model."""
    state = render_state(battle)
    speed, offense = state["speed"], state["offense"]
    assert isinstance(speed, str) and isinstance(offense, str)
    # garchomp (base spe 102) is faster than excadrill (base spe 88)
    assert speed == "faster (102 vs 88)"
    # garchomp's 130 atk into excadrill's 60 def beats its 80 spa into excadrill's 65 spd, by a wide margin
    physical, special = (float(part.split()[1].rstrip("x")) for part in offense.split(", "))
    assert offense.startswith("physical ") and physical > special


def test_build_options_handles_a_forced_recharge_turn(battle_recharge: Battle) -> None:
    """Regression test: poke-env's fallback stub for the "recharge" pseudo-move (used after Hyper Beam) lacks a
    "priority" key, and move.priority used to raise KeyError here, crashing every collect.py/evaluate.py run that
    reached one. "recharge" itself isn't modeled (no real decision, and no way to describe it safely), but
    switching out -- a genuine legal alternative to recharging -- must still show up as an option."""
    options = build_options(battle_recharge)
    assert "switch to skarmory" in options
    assert not any("recharge" in key for key in options)


def test_tera_options_describe_the_post_tera_move(battle_tera_changes_stab: Battle) -> None:
    """An option's description has to describe the option, not the position before taking it. Terastallizing to Rock
    grants Stone Edge STAB and turns Tera Blast from a neutral Normal move into a Rock one, which Excadrill
    (Ground/Steel) resists twice over -- so the effectiveness has to be recomputed on the post-tera type too, not
    just the STAB flag. A description written from the pre-tera types misstates both."""
    options = build_options(battle_tera_changes_stab)

    assert "STAB" not in options["use stoneedge"][1]
    assert "STAB" in options["use stoneedge with tera"][1]

    plain_blast, tera_blast = options["use terablast"][1], options["use terablast with tera"][1]
    assert plain_blast.startswith("Normal") and "STAB" not in plain_blast
    assert "0.5x vs excadrill" in plain_blast
    assert tera_blast.startswith("Rock") and "STAB" in tera_blast
    assert "0.25x vs excadrill" in tera_blast


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
