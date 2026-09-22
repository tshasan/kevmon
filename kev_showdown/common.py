"""Shared by the player (inference) and the collector (training data).

The single most important rule of this project: the text Kev sees at train time and at play time must be produced by
the SAME functions. Everything that turns a poke-env Battle into a Kev request lives here.

Design choice (lesson from kev issue #8): small models are bad at arithmetic, so we pre-compute the useful numbers
(type effectiveness, STAB, HP %) and put them in the text instead of hoping the model derives them.
"""
from __future__ import annotations

import random

from poke_env.battle import Battle, Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

from .types import JSONContent, Question, SystemOneRequest

ACTION_INSTRUCTIONS = (
    "You are playing a Pokemon Showdown singles battle. Pick the single best action for this turn. "
    "Prefer super-effective, high-power STAB moves, avoid moves the opponent is immune to, and switch out "
    "when the active Pokemon is badly outmatched or about to faint without doing anything useful."
)
WIN_INSTRUCTIONS = "Given the current battle state, will our side win this battle?"


# ---------------------------------------------------------------------------------------------------- state rendering

def _types(mon: Pokemon) -> str:
    return "/".join(t.name.title() for t in mon.types if t is not None)


def _hp(mon: Pokemon) -> str:
    return f"{round(100 * mon.current_hp_fraction)}%"


def _status(mon: Pokemon) -> str:
    return mon.status.name.lower() if mon.status else "none"


def _boosts(mon: Pokemon) -> str:
    b = {k: v for k, v in mon.boosts.items() if v}
    return ", ".join(f"{k} {v:+d}" for k, v in b.items()) or "none"


def _mon(mon: Pokemon | None) -> dict[str, JSONContent]:
    """Always the same 6 keys (fixed shape), even when nothing has switched in yet -- e.g. the turn right after our
    active Pokemon faints and before a replacement is chosen, where `mon` is None but the request still gets built."""
    if mon is None:
        return {"species": "unknown", "types": "unknown", "hp": "unknown", "status": "unknown",
                "boosts": "unknown", "terastallized": "unknown"}
    return {
        "species": mon.species,
        "types": _types(mon),
        "hp": _hp(mon),
        "status": _status(mon),
        "boosts": _boosts(mon),
        "terastallized": "yes" if mon.is_terastallized else "no",
    }


def _bench(team: dict[str, Pokemon], active: Pokemon | None) -> list[str]:
    out = []
    for mon in team.values():
        if mon is active:
            continue
        state = "fainted" if mon.fainted else f"{_hp(mon)} hp, status {_status(mon)}"
        out.append(f"{mon.species} ({_types(mon)}), {state}")
    return out


def _opponent_team_size(battle: Battle) -> int:
    """poke-env's public `team_size` only reports our own side; the opponent's revealed team size lives on the
    private `_team_size` map. Falls back to 6, correct for every format this project models (randbattle singles)."""
    sizes: dict[str, int] = getattr(battle, "_team_size", {})
    return sizes.get(battle.opponent_role, 6) if battle.opponent_role else 6


def render_state(battle: Battle) -> dict[str, JSONContent]:
    """Battle -> the JSON `state` Kev reads. Kev flattens dicts into 'key: value' text, so keys double as labels."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    alive_me = sum(not m.fainted for m in battle.team.values())
    opp_size = _opponent_team_size(battle)
    alive_opp = opp_size - sum(m.fainted for m in battle.opponent_team.values())
    return {
        "turn": battle.turn,
        "pokemon_remaining": f"us {alive_me}, opponent {alive_opp}",
        "our_active": _mon(me),
        "opponent_active": _mon(opp),
        "our_bench": _bench(battle.team, me) or ["none"],
        "opponent_revealed": _bench(battle.opponent_team, opp) or ["none"],
        "weather": ", ".join(w.name.lower() for w in battle.weather) or "none",
        "field": ", ".join(f.name.lower() for f in battle.fields) or "none",
        "our_side": ", ".join(s.name.lower() for s in battle.side_conditions) or "none",
        "opponent_side": ", ".join(s.name.lower() for s in battle.opponent_side_conditions) or "none",
        "can_terastallize": f"yes, to {_tera_type(battle) or 'unknown type'}" if battle.can_tera else "no",
    }


# ---------------------------------------------------------------------------------------------------- options

def _effectiveness(move: Move, target: Pokemon | None) -> float | None:
    if target is None or move.category.name == "STATUS":
        return None
    try:
        return target.damage_multiplier(move)
    except (KeyError, AttributeError):  # exotic types (e.g. ???/Stellar) missing from the type chart lookup
        return None


def _tera_type(battle: Battle) -> str | None:
    """Our active's tera type. poke-env does not always fill Pokemon.tera_type, so fall back to the raw request."""
    me = battle.active_pokemon
    if me is not None and me.tera_type is not None:
        return me.tera_type.name.title()
    try:
        for p in battle.last_request["side"]["pokemon"]:
            if p.get("active") and p.get("teraType"):
                return str(p["teraType"]).title()
    except (KeyError, TypeError):  # last_request missing "side"/"pokemon", or not set yet (None)
        pass
    return None


def _move_desc(move: Move, battle: Battle, tera: bool, tera_type: str | None) -> str:
    """`tera_type` is the battle's tera type for this turn, computed once by the caller (build_options) instead of
    once per move here -- can_tera is turn-scoped, not move-scoped, so recomputing it per move was pure waste."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    parts = [move.type.name.title(), move.category.name.lower()]
    if move.category.name != "STATUS":
        parts.append(f"power {move.base_power}")
        if opp is not None:
            eff = _effectiveness(move, opp)
            if eff is not None:
                parts.append(f"{eff:g}x vs {opp.species}")
        if me is not None and move.type in me.types:
            parts.append("STAB")
    acc = move.accuracy
    parts.append("never misses" if acc is True else f"accuracy {round(acc * 100)}%")
    if move.priority:
        parts.append(f"priority {move.priority:+d}")
    if move.heal:
        parts.append("heals")
    if move.self_switch:
        parts.append("switches user out")
    if tera:
        parts.append(f"terastallize to {tera_type} first" if tera_type else "terastallize first")
    return ", ".join(parts)


def _switch_desc(mon: Pokemon, battle: Battle) -> str:
    opp = battle.opponent_active_pokemon
    parts = [_types(mon), f"{_hp(mon)} hp", f"status {_status(mon)}"]
    if opp is not None:
        # worst-case damage multiplier the opponent's STAB types deal to this switch-in
        worst = max((mon.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
        parts.append(f"takes {worst:g}x from {opp.species}'s STAB types")
    return ", ".join(parts)


def order_key(order: SingleBattleOrder) -> str | None:
    """Stable, human-readable option name for an order. None for orders we do not model: mega/z/dynamax, and
    forced pseudo-moves (e.g. "recharge" after Hyper Beam) whose poke-env stub entry lacks "priority" -- move.priority
    raises KeyError on those, and there's no real decision to model on a turn with only one legal action anyway."""
    if order.mega or order.z_move or order.dynamax:
        return None
    if isinstance(order.order, Move):
        if "priority" not in order.order.entry:
            return None
        return f"use {order.order.id}" + (" with tera" if order.terastallize else "")
    if isinstance(order.order, Pokemon):
        return f"switch to {order.order.species}"
    return None


def build_options(battle: Battle) -> dict[str, tuple[SingleBattleOrder, str]]:
    """All legal orders this turn -> {option name: (order, description)}. Uses poke-env's own legality check."""
    valid = battle.valid_orders
    valid = valid() if callable(valid) else valid
    tera_type = _tera_type(battle) if battle.can_tera else None   # turn-scoped: compute once, not once per move
    out: dict[str, tuple[SingleBattleOrder, str]] = {}
    for order in valid:
        if not isinstance(order, SingleBattleOrder):
            continue
        key = order_key(order)
        if key is None or key in out:
            continue
        if isinstance(order.order, Move):
            desc = _move_desc(order.order, battle, order.terastallize, tera_type)
        elif isinstance(order.order, Pokemon):
            desc = _switch_desc(order.order, battle)
        else:
            continue
        out[key] = (order, desc)
    return out


def build_request(battle: Battle, options: dict[str, tuple[SingleBattleOrder, str]], rng: random.Random,
                  ask_win: bool = True) -> SystemOneRequest:
    """The /v1/systemone request, as a plain dict. Built once per turn from data this process already computed --
    there is no untrusted input here to justify pydantic validation (unlike the *response*, which is real network
    input and is validated in kev_showdown.types.SystemOneResponse). Option order is shuffled every call: Kev can be
    option-order sensitive, and a fixed order would let it learn 'pick the first switch' instead of reading the text."""
    names = list(options)
    rng.shuffle(names)
    questions: dict[str, Question] = {
        "action": {"type": "choice", "instructions": ACTION_INSTRUCTIONS,
                   "criteria": {n: options[n][1] for n in names}},
    }
    if ask_win:
        questions["win"] = {"type": "noul", "instructions": WIN_INSTRUCTIONS}
    return {"state": render_state(battle), "model": "kev-latest", "questions": questions}
