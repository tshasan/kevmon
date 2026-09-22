"""Does poke-engine agree with Showdown about what actually happens?

poke-engine is a separate reimplementation of the battle rules. A search over an engine that disagrees with the
server produces confident nonsense, so this number gates every result built on top of it.

Method: self-play, so we control both players and can read each side's true team off its own `Player` object -- no
sampling, no hidden information, nothing to confound the rules comparison. Each turn we snapshot the engine state
and both chosen moves; once the turn resolves we ask the engine, via `generate_instructions`, for every outcome it
thought possible, and check whether what Showdown actually did is among them.

"Among them" is the right question rather than exact equality: damage rolls are random (85-100%), so the engine
returns a probability-weighted set of outcomes. Agreement means Showdown's result was in that set. A miss means the
two implementations genuinely disagree about the rules.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

import poke_engine as pe
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import Player, SimpleHeuristicsPlayer

from kev_showdown.engine.actions import to_engine_choice
from kev_showdown.engine.state import to_engine_state

# (battle_tag, turn) -> role -> engine move string, and -> the state as seen at that turn
MOVES: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
SNAPSHOT: dict[tuple[str, int], pe.State] = {}
# HP of every Pokemon on both sides as the turn OPENED, keyed by species. Keyed by species rather than by "the
# active slot" on purpose: after a switch or a faint the active slot holds a DIFFERENT Pokemon next turn, so
# comparing slots compares two unrelated Pokemon and reports a disagreement that is entirely our own bug.
ACTUAL: dict[tuple[str, int], tuple[dict[str, int], dict[str, int]]] = {}
ACTIVES: dict[tuple[str, int], tuple[str, str]] = {}   # species active at that turn, to look up afterwards


class Recorder(SimpleHeuristicsPlayer):
    """Plays exactly like SimpleHeuristics, but records what it chose and what the board looked like."""

    def __init__(self, *args, role: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.role = role
        self.peer: Player | None = None

    def choose_move(self, battle: AbstractBattle):  # type: ignore[override]
        order = super().choose_move(battle)
        if isinstance(battle, Battle) and battle.active_pokemon is not None:
            key = (battle.battle_tag, battle.turn)
            choice = to_engine_choice(order)
            if choice:
                MOVES[key][self.role] = choice
            # p1 builds the snapshot, with the opponent's true team read off the peer's own battle object
            if self.role == "p1" and self.peer is not None:
                peer_battle = self.peer.battles.get(battle.battle_tag)
                if peer_battle is not None and peer_battle.active_pokemon is not None:
                    SNAPSHOT[key] = to_engine_state(battle, peer_battle.team, opponent_known=True)
                    ACTUAL[key] = ({m.species: (m.current_hp or 0) for m in battle.team.values()},
                                   {m.species: (m.current_hp or 0) for m in peer_battle.team.values()})
                    ACTIVES[key] = (battle.active_pokemon.species, peer_battle.active_pokemon.species)
        return order


def _hp_of(side: pe.Side, species: str) -> int | None:
    for mon in side.pokemon:
        if mon.id == species:
            return mon.hp
    return None


def _maxhp_of(side: pe.Side, species: str) -> int | None:
    for mon in side.pokemon:
        if mon.id == species:
            return mon.maxhp
    return None


def _outcomes(state: pe.State, m1: str, m2: str, a1: str, a2: str) -> list[tuple[int, int]]:
    """Every (hp of a1, hp of a2) the engine thinks this move pair can produce, looked up by species."""
    out = []
    for si in pe.generate_instructions(state, m1, m2):
        state.apply_instructions(si)
        h1, h2 = _hp_of(state.side_one, a1), _hp_of(state.side_two, a2)
        state.reverse_instructions(si)
        if h1 is not None and h2 is not None:
            out.append((h1, h2))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--format", default="gen9randombattle")
    a = ap.parse_args()

    p1 = Recorder(role="p1", battle_format=a.format, max_concurrent_battles=a.concurrency)
    p2 = Recorder(role="p2", battle_format=a.format, max_concurrent_battles=a.concurrency)
    p1.peer, p2.peer = p2, p1
    await p1.battle_against(p2, n_battles=a.battles)

    errors: list[float] = []
    errored = skipped = 0
    examples: list[str] = []
    for key, state in SNAPSHOT.items():
        tag, turn = key
        moves, nxt = MOVES.get(key, {}), (tag, turn + 1)
        if len(moves) < 2 or nxt not in ACTUAL:
            skipped += 1
            continue
        try:
            possible = _outcomes(state, moves["p1"], moves["p2"], *ACTIVES[key])
        except BaseException as exc:                      # a Rust panic is not an Exception
            errored += 1
            if len(examples) < 4:
                examples.append(f"  ERROR t{turn} {moves['p1']} vs {moves['p2']}: {str(exc)[:70]}")
            continue
        if not possible:
            skipped += 1
            continue
        a1, a2 = ACTIVES[key]
        hp1, hp2 = ACTUAL[nxt]
        if a1 not in hp1 or a2 not in hp2:
            skipped += 1
            continue
        actual = (hp1[a1], hp2[a2])
        # Scale by each side's max HP so the error is comparable across Pokemon, and take the closest outcome the
        # engine offered -- it collapses damage rolls to an average, so exact equality is not the right question.
        m1 = max(1, _maxhp_of(state.side_one, a1) or 1)
        m2 = max(1, _maxhp_of(state.side_two, a2) or 1)
        best = min(max(abs(p[0] - actual[0]) / m1, abs(p[1] - actual[1]) / m2) for p in possible)
        errors.append(best)
        if best > 0.25 and len(examples) < 8:
            examples.append(f"  BIG   t{turn} {moves['p1']} vs {moves['p2']}: actual {actual}, "
                            f"engine {possible[:2]} (err {best:.0%})")

    print(f"battles {a.battles}  turns compared {len(errors)}  (skipped {skipped}, engine errors {errored})")
    if errors:
        errors.sort()
        within = lambda t: 100 * sum(e <= t for e in errors) / len(errors)  # noqa: E731
        print(f"  predicted HP within  2% of actual: {within(0.02):5.1f}%")
        print(f"  predicted HP within  5% of actual: {within(0.05):5.1f}%")
        print(f"  predicted HP within 10% of actual: {within(0.10):5.1f}%")
        print(f"  predicted HP within 25% of actual: {within(0.25):5.1f}%")
        print(f"  median error {errors[len(errors)//2]:.1%}   p90 {errors[int(.9*len(errors))]:.1%}")
    for line in examples:
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
