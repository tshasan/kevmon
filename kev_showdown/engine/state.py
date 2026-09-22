"""poke-env `Battle` -> `poke_engine.State`.

The opponent's side is passed in rather than read off the battle, because that argument is the seam every
experiment variant plugs into: the true team (self-play, where we control both players), a sampled team, or an
uninformative guess. `to_engine_state` itself is indifferent to which it gets.

Our own side never needs a stat derivation -- poke-env carries the real `mon.stats` -- so `stats.py` only ever runs
on opponents, which keeps a derivation bug out of our half of the board.
"""
from __future__ import annotations

import poke_engine as pe
from poke_env.battle import Battle, Pokemon, SideCondition

from kev_showdown.engine import vocab
from kev_showdown.engine.stats import final_stats, randbats_evs_ivs

MAX_TEAM = 6
_BOOSTS = {"atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
           "spd": "special_defense_boost", "spe": "speed_boost",
           "accuracy": "accuracy_boost", "evasion": "evasion_boost"}


def _types(mon: Pokemon) -> tuple[str, str]:
    """poke-engine always wants two types; the second is "typeless" for single-typed Pokemon."""
    names = [t.name.lower() for t in mon.types if t is not None]
    return (names[0] if names else "typeless", names[1] if len(names) > 1 else "typeless")


def _item(mon: Pokemon) -> str:
    """poke-env uses "unknown_item" for an unrevealed item and "" for a knocked-off one; the engine wants its own
    sentinel for the former, and treats it as inert."""
    if not mon.item:
        return "none"
    return "unknownitem" if mon.item == "unknown_item" else mon.item


def _moves(mon: Pokemon) -> list[pe.Move]:
    return [pe.Move(id=move_id, pp=move.current_pp) for move_id, move in list(mon.moves.items())[:4]]


def to_engine_pokemon(mon: Pokemon, *, known: bool) -> pe.Pokemon:
    """`known=True` when poke-env has this Pokemon's real stats (always true for our own team, and for an
    opponent read out of their own `Player.battles[...]` in self-play). Otherwise stats are derived."""
    if known and mon.stats and mon.stats.get("atk") is not None:
        st = {k: v for k, v in mon.stats.items() if v is not None}
        maxhp = mon.max_hp or st.get("hp") or 1
        hp = mon.current_hp if mon.current_hp is not None else maxhp
    else:
        evs, ivs = randbats_evs_ivs(frozenset(mon.moves), _MOVE_CATEGORIES,
                                    species=mon.species, ability=mon.ability, base_stats=mon.base_stats)
        st = final_stats(mon.base_stats, mon.level, evs, ivs)
        maxhp = st["hp"]
        # An opponent's HP arrives as a percentage, so scale the derived maximum rather than trusting max_hp.
        hp = round(maxhp * (mon.current_hp_fraction if mon.current_hp_fraction is not None else 1.0))

    ability = (mon.ability or (mon.possible_abilities[0] if mon.possible_abilities else "none") or "none")
    return pe.Pokemon(
        id=mon.species, level=mon.level, types=_types(mon), hp=max(0, int(hp)), maxhp=int(maxhp),
        ability=str(ability).replace(" ", "").lower(), item=_item(mon),
        attack=st["atk"], defense=st["def"], special_attack=st["spa"],
        special_defense=st["spd"], speed=st["spe"],
        status=vocab.status(mon.status), moves=_moves(mon),
        terastallized=bool(mon.is_terastallized),
        tera_type=(mon.tera_type.name.lower() if mon.tera_type is not None else "typeless"),
    )


def to_engine_side(team: dict[str, Pokemon], active: Pokemon | None,
                   side_conditions: dict[SideCondition, int], turn: int,
                   *, known: bool) -> pe.Side:
    """A side. The active Pokemon is placed first so `active_index` is stable regardless of dict ordering."""
    mons = list(team.values())
    # Resolve the active by SPECIES, not by object identity. When the caller supplies a team read from another
    # Player's battle (self-play perfect information), `active` is our own limited view of that Pokemon -- a
    # different object, with no revealed moves. Prepending it directly puts a moveless duplicate at index 0 and the
    # engine then rejects the opponent's real move as illegal.
    if active is not None:
        same = next((m for m in mons if m.species == active.species), None)
        chosen = same if same is not None else active
        mons = [chosen] + [m for m in mons if m is not chosen]
    mons = mons[:MAX_TEAM]

    boosts = {}
    if active is not None:
        for key, field in _BOOSTS.items():
            boosts[field] = int(active.boosts.get(key, 0) or 0)

    return pe.Side(
        pokemon=[to_engine_pokemon(m, known=known) for m in mons],
        active_index="0" if active is not None else "0",     # engine wants a STRING index, not an int
        side_conditions=pe.SideConditions(**vocab.side_condition_kwargs(side_conditions, turn)),
        **boosts,
    )


def to_engine_state(battle: Battle, opponent_team: dict[str, Pokemon] | None = None,
                    *, opponent_known: bool = False) -> pe.State:
    """The searchable state. `opponent_team` defaults to what the battle has revealed, which is an incomplete
    side -- fine for smoke tests, not for a real search, which needs a sampled or true team."""
    opponent = opponent_team if opponent_team is not None else battle.opponent_team
    return pe.State(
        side_one=to_engine_side(battle.team, battle.active_pokemon, battle.side_conditions, battle.turn, known=True),
        side_two=to_engine_side(opponent, battle.opponent_active_pokemon, battle.opponent_side_conditions,
                                battle.turn, known=opponent_known),
        weather=vocab.weather(next(iter(battle.weather or {}), None)),
        terrain=vocab.terrain(battle.fields),
        trick_room=any(f.name == "TRICK_ROOM" for f in (battle.fields or {})),
    )


# Move categories, needed by the opponent stat derivation. Loaded once.
def _load_move_categories() -> dict[str, str]:
    from poke_env.data import GenData
    return {k: v.get("category", "") for k, v in GenData.from_gen(9).moves.items()}


_MOVE_CATEGORIES = _load_move_categories()
