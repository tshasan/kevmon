"""A poke-env Player that picks moves by running poke-engine's MCTS.

Three modes, differing ONLY in how the opponent's six are filled in -- that is the whole point, since the gap
between them is what bounds how much a belief model could ever be worth:

    perfect   the opponent's true sets (cheating; self-play only)
    sampled   K teams drawn from randbats set data, consistent with what has been revealed
    naive     nothing; only what has literally been seen

Compute is equalised across modes on purpose. `sampled` runs K searches and the others run one, so giving each the
same per-search budget would hand `sampled` K times the total thinking and the perfect-vs-sampled gap would measure
a mix of "less information" and "more search", pointing in opposite directions with no way to separate them
afterwards. Every mode gets the same TOTAL iteration count, and iterations rather than wall-clock so the experiment
is reproducible.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import poke_engine as pe
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from kev_showdown.common import build_options
from kev_showdown.engine.actions import from_engine_choice
from kev_showdown.engine.sets import naive_opponent_team, sample_opponent_team
from kev_showdown.engine.state import KnownSet, known_sets_from_team, to_engine_state
from kev_showdown.telemetry import configure, emit, emit_warning

MODES = ("perfect", "sampled", "naive")


class MctsPlayer(Player):
    def __init__(self, *args, mode: str = "sampled", iterations: int = 120_000, samples: int = 8,
                 threads: int = 4, seed: int = 0, log_file: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.iterations, self.samples, self.threads = mode, iterations, samples, threads
        self.rng = random.Random(seed)
        self.peer: Player | None = None                     # only used by mode="perfect"
        self.log = configure(log_file)
        self.fallbacks: dict[str, int] = defaultdict(int)
        self.decisions: dict[str, int] = defaultdict(int)
        self.total_iterations = 0
        self._frozen: dict[str, dict[str, KnownSet]] = {}

    # -- opponent models -------------------------------------------------------------------------------------
    def _true_sets(self, tag: str) -> dict[str, KnownSet]:
        """Frozen once, when every moveset is populated: poke-env fills them lazily, and a live peer battle races
        ours between await points, so re-reading it every turn silently mixes turns."""
        if tag not in self._frozen and self.peer is not None:
            peer_battle = self.peer.battles.get(tag)
            if peer_battle is not None:
                sets = known_sets_from_team(peer_battle.team)
                if len(sets) == 6:
                    self._frozen[tag] = sets
        return self._frozen.get(tag, {})

    def _teams(self, battle: Battle) -> list[dict[str, KnownSet]]:
        if self.mode == "perfect":
            return [self._true_sets(battle.battle_tag)]
        if self.mode == "naive":
            return [naive_opponent_team(battle)]
        return [sample_opponent_team(battle, self.rng) for _ in range(self.samples)]

    # -- play ------------------------------------------------------------------------------------------------
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:  # type: ignore[override]
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        options = build_options(battle)
        if not options:
            return self.choose_random_move(battle)
        if len(options) == 1:
            return next(iter(options.values()))[0]           # forced: nothing to search

        teams = self._teams(battle)
        per_search = max(1, self.iterations // max(1, len(teams)))
        votes: Counter[str] = Counter()
        try:
            for team in teams:
                state = to_engine_state(battle, team)
                result = pe.mcts(state, 0, per_search, self.threads)
                self.total_iterations += result.iteration_count
                for entry in result.s1:
                    votes[entry.move_choice] += entry.visits
        except BaseException as exc:                          # a Rust panic is not an Exception
            return self._fallback(battle, "EngineError", exc)

        if not votes:
            return self._fallback(battle, "NoVotes", None)
        choice = max(votes, key=lambda k: votes[k])
        order = from_engine_choice(choice, options)
        if order is None:
            # Not an expected fallback: the engine searched a state built from these very options, so a choice
            # that does not map back means state and option set disagree.
            return self._fallback(battle, "UnmappableChoice", choice)

        self.decisions[battle.battle_tag] += 1
        emit(self.log, "mcts.decision", battle_tag=battle.battle_tag, turn=battle.turn, mode=self.mode,
             pick=choice, teams=len(teams), iterations=per_search * len(teams))
        return order

    def _fallback(self, battle: Battle, error_type: str, error: object) -> BattleOrder:
        self.fallbacks[battle.battle_tag] += 1
        emit_warning(self.log, "mcts.fallback", battle_tag=battle.battle_tag, turn=battle.turn,
                     mode=self.mode, error_type=error_type, error=str(error)[:200])
        return self.choose_random_move(battle)

    # -- reporting: same shape as KevPlayer.stats so evaluate.play works unchanged ------------------------------
    def stats(self, tags: set[str] | None = None) -> dict[str, object]:
        keys = tags if tags is not None else set(self.decisions) | set(self.fallbacks)
        decisions = sum(self.decisions[t] for t in keys)
        fallbacks = sum(self.fallbacks[t] for t in keys)
        return {"mode": self.mode, "decisions": decisions, "fallbacks": fallbacks,
                "fallback_rate": round(fallbacks / max(decisions + fallbacks, 1), 4),
                "total_iterations": self.total_iterations,
                "iterations_per_decision": self.iterations}
