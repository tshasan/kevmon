"""poke-env orders <-> poke-engine move-choice strings.

`kev_showdown.common.build_options` stays the single source of truth for what is legal; this module only renames.
Two asymmetries are easy to get wrong and both fail silently -- a mismapped choice just falls back to a random move,
which is invisible in a win rate and shows up only as a fallback-rate spike:

  * a switch is the bare species name going IN (`generate_instructions` wants "skarmory") but comes back OUT of
    MCTS prefixed ("switch skarmory"),
  * tera is a "-tera" suffix on the move, not a separate flag.
"""
from __future__ import annotations

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import SingleBattleOrder

SWITCH_PREFIX = "switch "
TERA_SUFFIX = "-tera"
NO_CHOICE = "none"


def to_engine_choice(order: SingleBattleOrder) -> str | None:
    """None for the orders `common.order_key` also refuses to model, so the two stay in lockstep: an order one of
    them models and the other does not would mean the search and the request see different action sets."""
    if order.mega or order.z_move or order.dynamax:
        return None
    if isinstance(order.order, Move):
        if "priority" not in order.order.entry:        # forced pseudo-moves (recharge); not a real decision
            return None
        return order.order.id + (TERA_SUFFIX if order.terastallize else "")
    if isinstance(order.order, Pokemon):
        return order.order.species
    return None


def from_engine_choice(choice: str,
                       options: dict[str, tuple[SingleBattleOrder, str]]) -> SingleBattleOrder | None:
    """An MCTS `move_choice` back to the legal order it names, or None if it names none of them.

    None is a bug rather than an expected fallback: the engine searched a state built from these very options, so a
    choice that does not map back means the state and the option set disagree. Callers should log it loudly.
    """
    if not choice or choice == NO_CHOICE:
        return None
    wanted = choice[len(SWITCH_PREFIX):] if choice.startswith(SWITCH_PREFIX) else choice
    for order, _desc in options.values():
        if to_engine_choice(order) == wanted:
            return order
    return None
