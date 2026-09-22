"""Shared fixtures. `make_battle` builds a real poke-env `Battle` via its own public parsing API
(`parse_request` / `parse_message`, the same calls a live Showdown connection would make) -- not a hand-rolled
stand-in -- so tests that use it exercise real library behaviour (legality, type effectiveness, HP math, ...)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from poke_env.battle import Battle

_OUR_SIDE: dict[str, Any] = {
    "name": "kevbot",
    "id": "p1",
    "pokemon": [
        {
            "ident": "p1: Garchomp", "details": "Garchomp, L50", "condition": "150/150", "active": True,
            "stats": {"atk": 150, "def": 120, "spa": 90, "spd": 110, "spe": 130},
            "moves": ["earthquake", "stoneedge", "swordsdance", "substitute"],
            "baseAbility": "roughskin", "item": "lifeorb", "pokeball": "pokeball",
            "ability": "roughskin", "teraType": "Ground",
        },
        {
            "ident": "p1: Skarmory", "details": "Skarmory, L50", "condition": "140/140", "active": False,
            "stats": {"atk": 90, "def": 150, "spa": 50, "spd": 110, "spe": 95},
            "moves": ["bravebird", "roost", "spikes", "whirlwind"],
            "baseAbility": "sturdy", "item": "leftovers", "pokeball": "pokeball",
            "ability": "sturdy", "teraType": "Steel",
        },
    ],
}
_OUR_ACTIVE_REQUEST = {
    "active": [{
        "moves": [
            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
            {"move": "Stone Edge", "id": "stoneedge", "pp": 8, "maxpp": 8, "target": "normal", "disabled": False},
            {"move": "Swords Dance", "id": "swordsdance", "pp": 32, "maxpp": 32, "target": "self", "disabled": False},
            {"move": "Substitute", "id": "substitute", "pp": 16, "maxpp": 16, "target": "self", "disabled": False},
        ],
        "canTerastallize": "Ground",
    }],
    "side": _OUR_SIDE,
    "rqid": 1,
}


def _make_battle(battle_tag: str = "battle-gen9randombattle-1", opponent_hp: str = "160/160") -> Battle:
    """One turn, mid-battle: our Garchomp (active) + benched Skarmory vs. an opposing Excadrill."""
    battle = Battle(battle_tag=battle_tag, username="kevbot", logger=logging.getLogger("kev_showdown.tests"), gen=9)
    battle.parse_request(_OUR_ACTIVE_REQUEST)
    battle.parse_message(["", "switch", "p2a: Excadrill", "Excadrill, L50", opponent_hp])
    battle.parse_message(["", "turn", "5"])
    return battle


@pytest.fixture
def make_battle() -> Callable[..., Battle]:
    return _make_battle


@pytest.fixture
def battle() -> Battle:
    return _make_battle()


def _make_battle_no_active() -> Battle:
    """A request where every one of our Pokemon has `active: False` -- the real, verified shape of the very first
    request of a battle, before any switch-in has happened yet. `battle.active_pokemon` is `None` here (confirmed
    against poke-env directly), so this is what exercises common.py's None branch, not a fainting scenario."""
    battle = Battle(battle_tag="battle-gen9randombattle-2", username="kevbot", logger=logging.getLogger("kev_showdown.tests"), gen=9)
    request = {
        "active": [{"moves": [
            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
        ]}],
        "side": {"name": "kevbot", "id": "p1", "pokemon": [
            {**p, "active": False} for p in _OUR_SIDE["pokemon"]
        ]},
        "rqid": 1,
    }
    battle.parse_request(request)
    battle.parse_message(["", "switch", "p2a: Excadrill", "Excadrill, L50", "160/160"])
    return battle


@pytest.fixture
def battle_no_active() -> Battle:
    return _make_battle_no_active()


def _make_battle_recharge() -> Battle:
    """Our active must recharge (e.g. after Hyper Beam) -- poke-env's Move.entry stub for "recharge" lacks a
    "priority" key, so move.priority raises KeyError. Switching out remains a real, legal, separately-described
    option; only the undescribable "recharge" pseudo-move itself should be excluded."""
    battle = Battle(battle_tag="battle-gen9randombattle-3", username="kevbot", logger=logging.getLogger("kev_showdown.tests"), gen=9)
    request: dict[str, Any] = {
        "active": [{"moves": [{"move": "Recharge", "id": "recharge", "pp": 1, "maxpp": 1, "target": "self", "disabled": False}]}],
        "side": {"name": "kevbot", "id": "p1", "pokemon": [{**p, "active": i == 0} for i, p in enumerate(_OUR_SIDE["pokemon"])]},
        "rqid": 1,
    }
    battle.parse_request(request)
    battle.parse_message(["", "switch", "p2a: Excadrill", "Excadrill, L50", "160/160"])
    return battle


@pytest.fixture
def battle_recharge() -> Battle:
    return _make_battle_recharge()


def _make_battle_tera_changes_stab() -> Battle:
    """Our Garchomp (Ground/Dragon) can terastallize to Rock -- a type it does not already have. This is the case
    that tells a correct tera description from a wrong one: Stone Edge is not STAB now but is STAB after the tera,
    and Tera Blast stops being Normal and becomes Rock. The main `battle` fixture cannot catch either, because its
    tera type (Ground) is one of Garchomp's own types and nothing about STAB changes."""
    battle = Battle(battle_tag="battle-gen9randombattle-4", username="kevbot", logger=logging.getLogger("kev_showdown.tests"), gen=9)
    moves = [
        {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
        {"move": "Stone Edge", "id": "stoneedge", "pp": 8, "maxpp": 8, "target": "normal", "disabled": False},
        {"move": "Tera Blast", "id": "terablast", "pp": 16, "maxpp": 16, "target": "normal", "disabled": False},
    ]
    active = {**_OUR_SIDE["pokemon"][0], "moves": [m["id"] for m in moves], "teraType": "Rock"}
    request: dict[str, Any] = {
        "active": [{"moves": moves, "canTerastallize": "Rock"}],
        "side": {**_OUR_SIDE, "pokemon": [active, _OUR_SIDE["pokemon"][1]]},
        "rqid": 1,
    }
    battle.parse_request(request)
    battle.parse_message(["", "switch", "p2a: Excadrill", "Excadrill, L50", "160/160"])
    battle.parse_message(["", "turn", "5"])
    return battle


@pytest.fixture
def battle_tera_changes_stab() -> Battle:
    return _make_battle_tera_changes_stab()
