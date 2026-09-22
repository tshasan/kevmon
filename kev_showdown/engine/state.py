"""poke-env `Battle` -> `poke_engine.State`.

The opponent's side is passed in as **frozen set data**, never as live poke-env objects. That distinction is the
whole design and it was learned the hard way: a `Battle` is mutable and advances underneath you, so holding a
reference to another Player's battle and reading it later gives you a board from some other turn. In self-play the
two players' battles also advance independently between `await` points, so even reading the peer *during* our own
`choose_move` can catch it a full turn ahead (demonstrated in experiments/trace_turn.py).

So: everything DYNAMIC -- who is active, HP, status, boosts, hazards -- comes from our own `Battle`, which is
consistent by construction because it is the one we were just handed. Everything STATIC -- movesets, items,
abilities, derived stats -- comes from a `KnownSet` map that cannot change once built. That map is the seam every
experiment variant plugs into: true sets (self-play), sampled sets, or an uninformative guess.

Our own side never needs a stat derivation -- poke-env carries the real `mon.stats` -- so `stats.py` only ever runs
on opponents, which keeps a derivation bug out of our half of the board.
"""
from __future__ import annotations

from dataclasses import dataclass

import poke_engine as pe
from poke_env.battle import Battle, Pokemon, SideCondition

from kev_showdown.engine import vocab
from kev_showdown.engine.stats import final_stats, randbats_evs_ivs

MAX_TEAM = 6


@dataclass(frozen=True, slots=True)
class KnownSet:
    """The static half of an opponent Pokemon: everything that does not change during a battle.

    Frozen on purpose. The bug this prevents is subtle and total -- live poke-env objects mutate, so a "snapshot"
    that holds them is not a snapshot at all.
    """
    species: str
    level: int
    types: tuple[str, str]
    stats: dict[str, int]                 # final stats, hp included
    moves: tuple[str, ...]
    ability: str = "none"
    item: str = "none"
    tera_type: str = "typeless"


def known_sets_from_team(team: dict[str, Pokemon]) -> dict[str, KnownSet]:
    """Freeze a team into set data. Used for perfect information in self-play; the sampler will produce the same
    shape from randbats data. Pokemon whose moves have not been populated yet are skipped rather than frozen
    empty -- poke-env fills a team's movesets lazily, so an early call would otherwise bake in a moveless set."""
    out: dict[str, KnownSet] = {}
    for mon in team.values():
        if not mon.moves:
            continue
        if mon.stats and mon.stats.get("atk") is not None:
            st = {k: v for k, v in mon.stats.items() if v is not None}
            st.setdefault("hp", mon.max_hp or 1)
        else:
            evs, ivs = randbats_evs_ivs(frozenset(mon.moves), _MOVE_CATEGORIES, species=mon.species,
                                        ability=mon.ability, base_stats=mon.base_stats)
            st = final_stats(mon.base_stats, mon.level, evs, ivs)
        out[mon.species] = KnownSet(
            species=mon.species, level=mon.level, types=_types(mon), stats=st,
            moves=tuple(mon.moves), ability=str(mon.ability or "none").replace(" ", "").lower(),
            item=_item(mon), tera_type=(mon.tera_type.name.lower() if mon.tera_type is not None else "typeless"),
        )
    return out


def opponent_pokemon(mon: Pokemon, known: KnownSet | None) -> pe.Pokemon:
    """An opponent Pokemon: dynamic state from OUR view of it, static set data from `known` when we have it."""
    if known is not None:
        st, maxhp = known.stats, known.stats["hp"]
        moves = [pe.Move(id=m, pp=16) for m in known.moves[:4]]
        ability, item, types = known.ability, known.item, known.types
        tera = known.tera_type
    else:
        evs, ivs = randbats_evs_ivs(frozenset(mon.moves), _MOVE_CATEGORIES, species=mon.species,
                                    ability=mon.ability, base_stats=mon.base_stats)
        st = final_stats(mon.base_stats, mon.level, evs, ivs)
        maxhp = st["hp"]
        moves = _moves(mon)
        ability = str(mon.ability or (mon.possible_abilities[0] if mon.possible_abilities else "none")).replace(" ", "").lower()
        item, types = _item(mon), _types(mon)
        tera = mon.tera_type.name.lower() if mon.tera_type is not None else "typeless"

    # HP is always ours to observe: an opponent's arrives as a percentage, so scale their real maximum by it.
    frac = mon.current_hp_fraction if mon.current_hp_fraction is not None else 1.0
    return pe.Pokemon(
        id=mon.species, level=mon.level, types=types, hp=max(0, round(maxhp * frac)), maxhp=int(maxhp),
        ability=ability or "none", item=item,
        attack=st["atk"], defense=st["def"], special_attack=st["spa"],
        special_defense=st["spd"], speed=st["spe"],
        status=vocab.status(mon.status), moves=moves,
        terastallized=bool(mon.is_terastallized), tera_type=tera,
    )
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


def _side(mons: list[pe.Pokemon], active_first: bool, boosts: dict[str, int],
          side_conditions: dict[SideCondition, int], turn: int) -> pe.Side:
    return pe.Side(
        pokemon=mons[:MAX_TEAM],
        active_index="0",                                  # the engine wants a STRING index, not an int
        side_conditions=pe.SideConditions(**vocab.side_condition_kwargs(side_conditions, turn)),
        **boosts,
    )


def _boosts(active: Pokemon | None) -> dict[str, int]:
    if active is None:
        return {}
    return {field: int(active.boosts.get(key, 0) or 0) for key, field in _BOOSTS.items()}


def _ordered(team: dict[str, Pokemon], active: Pokemon | None) -> list[Pokemon]:
    """Active first, so `active_index` is always "0" regardless of dict ordering."""
    mons = list(team.values())
    if active is None:
        return mons
    chosen = next((m for m in mons if m.species == active.species), active)
    return [chosen] + [m for m in mons if m is not chosen]


def our_side(battle: Battle) -> pe.Side:
    mons = [to_engine_pokemon(m, known=True) for m in _ordered(battle.team, battle.active_pokemon)]
    return _side(mons, True, _boosts(battle.active_pokemon), battle.side_conditions, battle.turn)


def opponent_side(battle: Battle, sets: dict[str, KnownSet] | None) -> pe.Side:
    """Built from OUR view of the opponent -- who is active, how hurt they look -- plus frozen set data for the
    static half. Never from another Player's live battle: see the module docstring."""
    sets = sets or {}
    active = battle.opponent_active_pokemon
    seen = _ordered(battle.opponent_team, active)
    mons = [opponent_pokemon(m, sets.get(m.species)) for m in seen]

    # Fill the rest of their six from set data we hold but have not seen in play yet, so the engine knows they
    # have switches available. Without this a partially-revealed opponent looks nearly dead and every search is
    # wrong in the same direction.
    for species, known in sets.items():
        if len(mons) >= MAX_TEAM:
            break
        if any(m.id == species for m in mons):
            continue
        mons.append(pe.Pokemon(
            id=known.species, level=known.level, types=known.types,
            hp=known.stats["hp"], maxhp=known.stats["hp"],
            ability=known.ability or "none", item=known.item,
            attack=known.stats["atk"], defense=known.stats["def"],
            special_attack=known.stats["spa"], special_defense=known.stats["spd"],
            speed=known.stats["spe"], status="none",
            moves=[pe.Move(id=m, pp=16) for m in known.moves[:4]], tera_type=known.tera_type,
        ))
    return _side(mons, True, _boosts(active), battle.opponent_side_conditions, battle.turn)


def to_engine_state(battle: Battle, opponent_sets: dict[str, KnownSet] | None = None) -> pe.State:
    """The searchable state. `opponent_sets` is frozen static data (true, sampled, or naive); everything dynamic
    comes from `battle`, which is the only object here guaranteed to be describing the turn we were handed."""
    return pe.State(
        side_one=our_side(battle),
        side_two=opponent_side(battle, opponent_sets),
        weather=vocab.weather(next(iter(battle.weather or {}), None)),
        terrain=vocab.terrain(battle.fields),
        trick_room=any(f.name == "TRICK_ROOM" for f in (battle.fields or {})),
    )


# Move categories, needed by the opponent stat derivation. Loaded once.
def _load_move_categories() -> dict[str, str]:
    from poke_env.data import GenData
    return {k: v.get("category", "") for k, v in GenData.from_gen(9).moves.items()}


_MOVE_CATEGORIES = _load_move_categories()
