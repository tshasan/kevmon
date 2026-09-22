"""Hand-verification of single turns: the engine's prediction next to what Showdown actually did.

Aggregate agreement statistics were the wrong place to start -- three harness bugs in a row each moved the headline
number without telling us anything about the engine. This prints one turn at a time, in full, with the raw Showdown
protocol for that turn, so a human can check a handful by eye before any aggregate is trusted.

Usage:  python experiments/trace_turn.py --battles 1 --turns 4
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

import poke_engine as pe
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import SimpleHeuristicsPlayer

from kev_showdown.engine.actions import to_engine_choice
from kev_showdown.engine.state import known_sets_from_team, to_engine_state

LOG: dict[str, list[list[str]]] = defaultdict(list)      # battle_tag -> raw protocol lines, in order
DECIDED: list[dict] = []                                  # one entry per decision point, in order


class Traced(SimpleHeuristicsPlayer):
    def __init__(self, *args, role: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.role = role
        self.peer: Traced | None = None

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        if self.role == "p1":
            for m in split_messages:
                if len(m) > 1:
                    LOG[m[0].split("|")[0] if "|" in m[0] else (self._last_tag(m) or "")].append(m)
        await super()._handle_battle_message(split_messages)

    @staticmethod
    def _last_tag(m: list[str]) -> str:
        return m[0] if m and m[0].startswith("battle-") else ""

    def _sets(self, tag: str, peer_battle) -> dict:
        cached = getattr(self, "_set_cache", None)
        if cached is None:
            cached = self._set_cache = {}
        if tag not in cached and peer_battle is not None:
            frozen = known_sets_from_team(peer_battle.team)
            if len(frozen) == 6:                      # only freeze once every moveset is populated
                cached[tag] = frozen
        return cached.get(tag, {})

    def choose_move(self, battle: AbstractBattle):  # type: ignore[override]
        order = super().choose_move(battle)
        if isinstance(battle, Battle) and battle.active_pokemon is not None:
            peer_battle = self.peer.battles.get(battle.battle_tag) if self.peer else None
            DECIDED.append({
                "role": self.role, "tag": battle.battle_tag, "turn": battle.turn,
                "active": battle.active_pokemon.species,
                "hp": battle.active_pokemon.current_hp,
                "maxhp": battle.active_pokemon.max_hp,
                "force_switch": bool(getattr(battle, "force_switch", False)),
                "choice": to_engine_choice(order),
                "team_hp": {m.species: m.current_hp for m in battle.team.values()},
                "peer_active": (peer_battle.active_pokemon.species
                                if peer_battle and peer_battle.active_pokemon else None),
                "peer_turn": (peer_battle.turn if peer_battle else None),
                # Freeze the peer's sets ONCE, the first time they are populated, and reuse them. Static data
                # cannot race; re-reading the peer's live battle every turn is what did.
                "state": to_engine_state(battle, self._sets(battle.battle_tag, peer_battle)),
            })
        return order


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=1)
    ap.add_argument("--turns", type=int, default=4)
    a = ap.parse_args()

    p1 = Traced(role="p1", battle_format="gen9randombattle", max_concurrent_battles=1)
    p2 = Traced(role="p2", battle_format="gen9randombattle", max_concurrent_battles=1)
    p1.peer, p2.peer = p2, p1
    await p1.battle_against(p2, n_battles=a.battles)

    # Decision points in order, so we can SEE whether turn numbers are adjacent or whether a forced switch
    # inserts an extra decision inside the same turn -- the alignment hypothesis.
    print("=== decision points, in the order they happened ===")
    for d in DECIDED[: a.turns * 4]:
        flag = "  <-- FORCED SWITCH" if d["force_switch"] else ""
        print(f"  {d['role']}  turn {d['turn']:>2}  active {d['active']:<16} hp {d['hp']}/{d['maxhp']}"
              f"  chose {d['choice']}{flag}")

    print("\n=== per-turn comparison (p1's view) ===")
    p1_points = [d for d in DECIDED if d["role"] == "p1"]
    p2_by_turn = {d["turn"]: d for d in DECIDED if d["role"] == "p2"}

    for i, d in enumerate(p1_points[: a.turns]):
        turn = d["turn"]
        other = p2_by_turn.get(turn)
        if other is None:
            print(f"\nturn {turn}: no matching p2 decision -- misalignment"); continue
        nxt = p1_points[i + 1] if i + 1 < len(p1_points) else None
        print(f"\n--- turn {turn}: {d['active']} ({d['hp']}/{d['maxhp']}) vs "
              f"{other['active']} ({other['hp']}/{other['maxhp']})")
        print(f"    p1 chose {d['choice']!r}   p2 chose {other['choice']!r}")
        if d["state"] is None:
            print("    (no peer view)"); continue
        try:
            state = d["state"]
            print(f"    peer battle was on turn {d['peer_turn']} when we snapshotted"
                  f"{'  <-- RACE' if d['peer_turn'] != turn else ''}")
            s1 = state.side_one.pokemon[int(state.side_one.active_index)]
            s2 = state.side_two.pokemon[int(state.side_two.active_index)]
            print(f"    engine sees: {s1.id} {s1.hp}/{s1.maxhp} [{','.join(m.id for m in s1.moves)}]")
            print(f"                 {s2.id} {s2.hp}/{s2.maxhp} [{','.join(m.id for m in s2.moves)}]")
            outs = []
            for si in pe.generate_instructions(state, d["choice"], other["choice"]):
                state.apply_instructions(si)
                outs.append((round(si.percentage, 1),
                             state.side_one.pokemon[int(state.side_one.active_index)].hp,
                             state.side_two.pokemon[int(state.side_two.active_index)].hp))
                state.reverse_instructions(si)
            print(f"    engine predicts (pct, p1hp, p2hp): {outs[:6]}")
        except BaseException as exc:
            print(f"    ENGINE ERROR: {str(exc)[:120]}")
        if nxt:
            print(f"    next p1 decision: turn {nxt['turn']} active {nxt['active']} hp {nxt['hp']}"
                  f"{'  (FORCED SWITCH)' if nxt['force_switch'] else ''}")
            print(f"    actual p1 team hp: {nxt['team_hp']}")


if __name__ == "__main__":
    asyncio.run(main())
