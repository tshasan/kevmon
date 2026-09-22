"""Does poke-engine agree with Showdown about how much damage a turn deals?

A search over an engine that disagrees with the server produces confident nonsense, so this gates everything.

Three things about the engine's Python API shape this harness, all learned by hand-tracing single turns first
(experiments/trace_turn.py) after aggregate statistics proved useless:

  1. `apply_instructions` does NOT mutate the Python-visible state -- HP reads back unchanged afterwards. State
     diffing is therefore impossible; the `instruction_list` IS the prediction. `PyInstruction` exposes no
     attributes either, only `__repr__` ("Damage SideOne: 174"), so it is parsed as text.
  2. A poke-env `Battle` is live and mutable, and in self-play the two players advance independently between
     await points. Reading the peer's battle for the opponent's side catches it up to a full turn ahead. So the
     opponent's STATIC set data is frozen once and everything DYNAMIC comes from our own battle.
  3. A faint creates a forced-switch decision INSIDE the same turn number, so consecutive decisions are not
     consecutive board states. Those turns are skipped rather than mis-paired -- on BOTH sides. Skipping only our
     own forced switches leaves the opponent's replacement choice overwriting their real move for that turn, which
     reads as "the engine says they switched and dealt no damage" while we visibly took a full hit.

We compare damage to OUR OWN side only. We see our HP exactly, where an opponent's arrives as a percentage, so
this is the half of the comparison that carries no observation error of its own.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from collections import defaultdict

import poke_engine as pe
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import SimpleHeuristicsPlayer

from kev_showdown.engine.actions import to_engine_choice
from kev_showdown.engine.state import known_sets_from_team, to_engine_state

DAMAGE_S1 = re.compile(r"Damage SideOne: (\d+)")
HEAL_S1 = re.compile(r"Heal SideOne: (\d+)")
TURNS: dict[tuple[str, int], dict] = defaultdict(dict)


class Recorder(SimpleHeuristicsPlayer):
    def __init__(self, *args, role: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.role, self.peer, self._sets = role, None, {}

    def _frozen(self, tag: str) -> dict:
        """Freeze the peer's sets once, when every moveset has been populated. poke-env fills them lazily, so an
        earlier freeze would bake in moveless sets and the engine would reject the opponent's real moves."""
        if tag not in self._sets and self.peer is not None:
            peer_battle = self.peer.battles.get(tag)
            if peer_battle is not None:
                frozen = known_sets_from_team(peer_battle.team)
                if len(frozen) == 6:
                    self._sets[tag] = frozen
        return self._sets.get(tag, {})

    def choose_move(self, battle: AbstractBattle):  # type: ignore[override]
        order = super().choose_move(battle)
        if not isinstance(battle, Battle) or battle.active_pokemon is None:
            return order
        key = (battle.battle_tag, battle.turn)
        slot = TURNS[key]
        choice = to_engine_choice(order)
        slot[f"{self.role}_choice"] = choice
        slot[f"{self.role}_forced"] = bool(getattr(battle, "force_switch", False))
        # A switch moves the damage onto the INCOMING Pokemon, so "damage to the mon that was active" is no longer
        # the quantity the engine is reporting. Those turns need different bookkeeping; exclude them.
        slot[f"{self.role}_switch"] = choice is not None and choice in {m.species for m in battle.team.values()}
        if self.role == "p1":
            slot.setdefault("state", to_engine_state(battle, self._frozen(battle.battle_tag)))
            slot.setdefault("active", battle.active_pokemon.species)
            slot.setdefault("hp", battle.active_pokemon.current_hp or 0)
            slot["team_hp"] = {m.species: (m.current_hp or 0) for m in battle.team.values()}
        return order


def predicted_damage(state: pe.State, m1: str, m2: str) -> list[tuple[float, int]]:
    """(probability, damage to side one) for every branch the engine considered."""
    out = []
    for si in pe.generate_instructions(state, m1, m2):
        # Net, not gross: draining and recoil moves emit Heal alongside Damage, and a Drain Punch turn can leave
        # us with MORE hp than we started with. Summing damage alone reports that as a large disagreement.
        reprs = [repr(i) for i in si.instruction_list]
        dmg = sum(int(m) for r in reprs for m in DAMAGE_S1.findall(r))
        heal = sum(int(m) for r in reprs for m in HEAL_S1.findall(r))
        out.append((si.percentage, dmg - heal))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=10)
    a = ap.parse_args()

    p1 = Recorder(role="p1", battle_format="gen9randombattle", max_concurrent_battles=a.concurrency)
    p2 = Recorder(role="p2", battle_format="gen9randombattle", max_concurrent_battles=a.concurrency)
    p1.peer, p2.peer = p2, p1
    await p1.battle_against(p2, n_battles=a.battles)

    exact = close = off = skipped = errored = 0
    examples: list[str] = []
    for (tag, turn), slot in sorted(TURNS.items()):
        nxt = TURNS.get((tag, turn + 1))
        if (not slot.get("state") or not slot.get("p1_choice") or not slot.get("p2_choice")
                or slot.get("p1_forced") or slot.get("p2_forced") or slot.get("p1_switch")
                or nxt is None or not nxt.get("team_hp")
                or slot["active"] not in nxt["team_hp"] or nxt.get("p1_forced")):
            skipped += 1
            continue
        try:
            branches = predicted_damage(slot["state"], slot["p1_choice"], slot["p2_choice"])
        except BaseException as exc:
            errored += 1
            if len(examples) < 3:
                examples.append(f"  ERROR t{turn}: {str(exc)[:70]}")
            continue
        if not branches:
            skipped += 1
            continue
        actual = slot["hp"] - nxt["team_hp"][slot["active"]]
        best = min(abs(d - actual) for _pct, d in branches)
        maxhp = max(1, slot["hp"] or 1)
        if best == 0:
            exact += 1
        elif best / maxhp <= 0.10:          # damage rolls span ~15%, so this is the "same move, different roll" band
            close += 1
        else:
            off += 1
            if len(examples) < 8:
                examples.append(f"  OFF  t{turn} {slot['p1_choice']} vs {slot['p2_choice']}: "
                                f"actual -{actual}, engine {[d for _p, d in branches][:4]}")

    total = exact + close + off
    print(f"battles {a.battles}  turns compared {total}  (skipped {skipped}, engine errors {errored})")
    if total:
        print(f"  exact damage match      {100 * exact / total:5.1f}%  ({exact})")
        print(f"  within a damage roll    {100 * (exact + close) / total:5.1f}%  ({close} more)")
        print(f"  genuinely disagreeing   {100 * off / total:5.1f}%  ({off})")
    for line in examples:
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
