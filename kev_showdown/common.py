"""Shared by the player (inference) and the collector (training data).

The single most important rule of this project: the text Kev sees at train time and at play time must be produced by
the SAME functions. Everything that turns a poke-env Battle into a Kev request lives here.

Design choice (lesson from kev issue #8): small models are bad at arithmetic, so we pre-compute the useful numbers
(type effectiveness, STAB, HP %) and put them in the text instead of hoping the model derives them.
"""
from __future__ import annotations

import random

from poke_env.battle import Battle, Move, Pokemon, PokemonType
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
    active Pokemon faints and before a replacement is chosen, where `mon` is None but the request still gets built.

    Deliberately no raw base-stat block: at ~22 tokens per side it is the most expensive thing that can go here, and
    `speed` and `offense` already carry the only comparisons anything downstream makes from those numbers. The state
    has a hard 384-token budget (kev.model.MAX_STATE) and over-running it silently drops the record from training."""
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


def _speed(mon: Pokemon | None) -> int | None:
    try:
        return int(mon.base_stats["spe"]) if mon is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def _speed_matchup(me: Pokemon | None, opp: Pokemon | None) -> str:
    """Base-speed comparison only: it ignores items, abilities and paralysis, so it is a proxy for who moves first,
    not a guarantee. Rendered because move-vs-switch decisions hinge on it and the model cannot infer it otherwise."""
    a, b = _speed(me), _speed(opp)
    if a is None or b is None:
        return "unknown"
    return f"{'faster' if a > b else 'slower' if a < b else 'tied'} ({a} vs {b})"


def _effective_stat(mon: Pokemon, stat: str) -> float:
    boost = mon.boosts.get(stat, 0)
    multiplier = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
    return (2 * mon.base_stats[stat] + 36) * multiplier


def _offense(me: Pokemon | None, opp: Pokemon | None) -> str:
    """Attacker-vs-defender stat ratios, boosts included. Whether to attack physically or specially is a division the
    model would otherwise have to do in its head from raw base stats (see the arithmetic note in the module docstring)."""
    if me is None or opp is None:
        return "unknown"
    try:
        physical = _effective_stat(me, "atk") / _effective_stat(opp, "def")
        special = _effective_stat(me, "spa") / _effective_stat(opp, "spd")
    except (KeyError, ZeroDivisionError):
        return "unknown"
    return f"physical {physical:.2f}x, special {special:.2f}x"


def _bench(team: dict[str, Pokemon], active: Pokemon | None) -> list[str]:
    """Terse on purpose: the two bench lists are the largest thing in the state (~128 tokens together against a
    384-token budget), and "status none" on a healthy Pokemon repeated ten times is the single cheapest thing to
    cut. A fainted Pokemon needs no HP or status -- it has no decisions left in it."""
    out = []
    for mon in team.values():
        if mon is active:
            continue
        if mon.fainted:
            out.append(f"{mon.species} fainted")
            continue
        status = "" if mon.status is None else f" {_status(mon)}"
        out.append(f"{mon.species} {_types(mon)} {_hp(mon)}{status}")
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
        "speed": _speed_matchup(me, opp),
        "offense": _offense(me, opp),
        "our_bench": _bench(battle.team, me) or ["none"],
        "opponent_revealed": _bench(battle.opponent_team, opp) or ["none"],
        "weather": ", ".join(w.name.lower() for w in battle.weather) or "none",
        "field": ", ".join(f.name.lower() for f in battle.fields) or "none",
        "our_side": ", ".join(s.name.lower() for s in battle.side_conditions) or "none",
        "opponent_side": ", ".join(s.name.lower() for s in battle.opponent_side_conditions) or "none",
        "can_terastallize": f"yes, to {_tera_display(battle)}" if battle.can_tera else "no",
    }


# ---------------------------------------------------------------------------------------------------- options

def _effectiveness(type_or_move: PokemonType | Move, target: Pokemon | None) -> float | None:
    if target is None:
        return None
    try:
        return target.damage_multiplier(type_or_move)
    except (KeyError, AttributeError):  # exotic types (e.g. ???/Stellar) missing from the type chart lookup
        return None


def _tera_type(battle: Battle) -> PokemonType | None:
    """Our active's tera type. poke-env does not always fill Pokemon.tera_type, so fall back to the raw request."""
    me = battle.active_pokemon
    if me is not None and me.tera_type is not None:
        return me.tera_type
    try:
        for p in battle.last_request["side"]["pokemon"]:
            if p.get("active") and p.get("teraType"):
                return PokemonType.from_name(str(p["teraType"]))
    except (KeyError, TypeError, ValueError):  # last_request missing "side"/"pokemon", not set yet, or an unknown type name
        pass
    return None


def _tera_display(battle: Battle) -> str:
    tera_type = _tera_type(battle)
    return tera_type.name.title() if tera_type is not None else "unknown type"


def _move_desc(move: Move, battle: Battle, tera: bool, tera_type: PokemonType | None) -> str:
    """`tera_type` is the battle's tera type for this turn, computed once by the caller (build_options) instead of
    once per move here -- can_tera is turn-scoped, not move-scoped, so recomputing it per move was pure waste.

    Terastallizing changes two things this description has to respect, or the option text argues against the option:
    Tera Blast becomes the tera type, and STAB is granted on the tera type *in addition to* the original types."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    teraing = tera and tera_type is not None
    attack_type = tera_type if teraing and move.id == "terablast" and tera_type is not None else move.type
    stab_types = {t for t in (me.types if me is not None else ()) if t is not None}
    if teraing and tera_type is not None:
        stab_types.add(tera_type)

    parts = [attack_type.name.title(), move.category.name.lower()]
    if move.category.name != "STATUS":
        parts.append(f"power {move.base_power}")
        if opp is not None:
            eff = _effectiveness(attack_type, opp)
            if eff is not None:
                parts.append(f"{eff:g}x vs {opp.species}")
        if attack_type in stab_types:
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
        parts.append(f"terastallize to {tera_type.name.title()} first" if tera_type else "terastallize first")
    return ", ".join(parts)


def _switch_desc(mon: Pokemon, battle: Battle) -> str:
    opp = battle.opponent_active_pokemon
    parts = [_types(mon), f"{_hp(mon)} hp", f"status {_status(mon)}"]
    if opp is not None:
        # worst-case damage multiplier the opponent's STAB types deal to this switch-in
        worst = max((mon.damage_multiplier(t) for t in opp.types if t is not None), default=1.0)
        parts.append(f"takes {worst:g}x from {opp.species}'s STAB types")
        mine, theirs = _speed(mon), _speed(opp)
        if mine is not None and theirs is not None:
            parts.append("faster" if mine > theirs else "slower" if mine < theirs else "same speed")
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
