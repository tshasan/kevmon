"""A poke-env Player whose every decision is one Kev /v1/systemone call.

Start Kev first, e.g.:
    KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import httpx
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import BattleOrder, Player
from pydantic import ValidationError

from .common import build_options, build_request
from .telemetry import configure, emit_warning, emit
from .types import ChoiceAnswer, NoulAnswer, SystemOneResponse

StatsDict = dict[str, "int | float | None"]


@dataclass(slots=True)
class Decision:
    """One answered turn. Kept per battle tag so every statistic can be sliced to the battles played against a single
    opponent, instead of being a single pile averaged over a whole session."""
    latency_ms: float
    confidence: float
    entropy: float          # normalised to [0, 1]: 0 is a collapsed policy, 1 is uniform over the legal options
    p_win: float | None


def normalised_entropy(probabilities: dict[str, float]) -> float:
    """Shannon entropy of the option distribution over log(n), so distributions over different numbers of legal
    actions are comparable. Worth tracking because a policy that has collapsed onto one option scores well on
    accuracy right up until it meets a position its training data never covered."""
    values = [p for p in probabilities.values() if p > 0.0]
    if len(values) < 2:
        return 0.0
    total = sum(values)
    entropy = -sum((p / total) * math.log(p / total) for p in values)
    return entropy / math.log(len(probabilities))


class KevPlayer(Player):
    def __init__(self, *args: object, kev_url: str = "http://127.0.0.1:8009", timeout: float = 15.0,
                 ask_win: bool = True, seed: int = 0, verbose: bool = False, log_file: str | None = None,
                 **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.kev_url = kev_url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.ask_win = ask_win
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.log = configure(log_file)
        self._http: httpx.AsyncClient | None = None
        # Everything is keyed by battle tag so `stats(tags)` can report a single opponent's numbers. Outcomes are
        # joined against each battle's p(win) predictions in _battle_finished_callback: the only way to find out
        # whether the noul head is calibrated for the policy actually being played rather than for the teacher's.
        self.decisions: dict[str, list[Decision]] = defaultdict(list)
        self.fallbacks: Counter[str] = Counter()
        self.outcomes: dict[str, bool] = {}

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle), "kev_showdown only models gen9 singles battles, not doubles"
        options = build_options(battle)
        if not options:
            return self.choose_random_move(battle)
        if len(options) == 1:  # forced: no need to ask
            return next(iter(options.values()))[0]

        req = build_request(battle, options, self.rng, ask_win=self.ask_win)
        t = time.perf_counter()
        try:
            r = await (await self._client()).post(self.kev_url, json=req)
            r.raise_for_status()
            answers = SystemOneResponse.model_validate(r.json())
        except (httpx.HTTPError, ValidationError, KeyError, json.JSONDecodeError) as e:  # never forfeit over Kev being down, slow, or malformed
            self._fallback(battle, type(e).__name__, str(e))
            if self.verbose:
                print(f"[kev] fallback on turn {battle.turn}: {e!r}")
            return self.choose_random_move(battle)
        latency_ms = (time.perf_counter() - t) * 1000

        action = answers.answers.get("action")
        if not isinstance(action, ChoiceAnswer):
            self._fallback(battle, "MissingActionAnswer", "response had no 'action' choice answer")
            return self.choose_random_move(battle)

        win = answers.answers.get("win")
        p_win = win.noul if isinstance(win, NoulAnswer) else None
        entropy = normalised_entropy(action.probabilities)
        self.decisions[battle.battle_tag].append(Decision(latency_ms, action.confidence, entropy, p_win))
        # `probabilities` is the policy's full distribution over the options it was offered. It costs nothing to log
        # (the response already carries it) and it is the one thing a later policy-improvement pass cannot reconstruct
        # after the fact: re-deriving it needs the exact checkpoint, the exact option set and the exact shuffle.
        emit(self.log, "kev.decision", battle_tag=battle.battle_tag, turn=battle.turn, pick=action.choice,
             confidence=action.confidence, p_win=p_win, latency_ms=round(latency_ms, 1), entropy=round(entropy, 4),
             probabilities={k: round(v, 5) for k, v in action.probabilities.items()})
        if self.verbose:
            print(f"[kev] {battle.battle_tag} t{battle.turn}: {action.choice} (conf {action.confidence}, p(win) {p_win})")
        if action.choice in options:
            return options[action.choice][0]
        self._fallback(battle, "UnknownChoice", f"kev returned {action.choice!r}, not one of the offered options")
        return self.choose_random_move(battle)

    def _fallback(self, battle: Battle, error_type: str, error: str) -> None:
        self.fallbacks[battle.battle_tag] += 1
        emit_warning(self.log, "kev.fallback", battle_tag=battle.battle_tag, turn=battle.turn,
                     error_type=error_type, error=error)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        won = bool(battle.won)
        self.outcomes[battle.battle_tag] = won
        decisions = self.decisions.get(battle.battle_tag, [])
        p_wins = [d.p_win for d in decisions if d.p_win is not None]
        emit(self.log, "kev.battle_end", battle_tag=battle.battle_tag, won=won, turns=battle.turn,
             decisions=len(decisions), fallbacks=self.fallbacks[battle.battle_tag],
             mean_p_win=round(statistics.mean(p_wins), 3) if p_wins else None)

    def calibration(self, tags: set[str] | None = None) -> StatsDict:
        """Brier score of p(win) against the outcome of the battle it was predicted in, over every decision of every
        finished battle. Lower is better, but the number only means something next to
        `p_win_brier_if_constant_base_rate`: that is the score for ignoring the board and always answering the base
        rate. A value head that does not clearly beat it is not a value head, whatever its accuracy says."""
        pairs = [(d.p_win, self.outcomes[tag]) for tag, ds in self._selected(tags) if tag in self.outcomes
                 for d in ds if d.p_win is not None]
        if not pairs:
            return {"p_win_predictions": 0, "p_win_brier": None, "p_win_mean": None, "outcome_base_rate": None}
        base_rate = statistics.mean(float(won) for _, won in pairs)
        return {
            "p_win_predictions": len(pairs),
            "p_win_brier": round(statistics.mean((p - won) ** 2 for p, won in pairs), 4),
            "p_win_mean": round(statistics.mean(p for p, _ in pairs), 3),
            "outcome_base_rate": round(base_rate, 3),
            "p_win_brier_if_constant_base_rate": round(base_rate * (1 - base_rate), 4),
        }

    def _selected(self, tags: set[str] | None) -> list[tuple[str, list[Decision]]]:
        return [(tag, ds) for tag, ds in self.decisions.items() if tags is None or tag in tags]

    def stats(self, tags: set[str] | None = None) -> StatsDict:
        """Every recorded statistic, optionally restricted to one opponent's battles. `tags` is the set of battle tags
        that opponent produced; evaluate.py takes the difference of `self.battles` before and after each matchup."""
        selected = self._selected(tags)
        decisions = [d for _, ds in selected for d in ds]
        latencies = sorted(d.latency_ms for d in decisions)
        fallbacks = sum(self.fallbacks[tag] for tag, _ in selected) if tags is not None else sum(self.fallbacks.values())
        turns = [len(ds) for _, ds in selected]
        return {
            "decisions": len(decisions),
            "fallbacks": fallbacks,
            "fallback_rate": round(fallbacks / (len(decisions) + fallbacks), 4) if decisions or fallbacks else None,
            "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
            "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
            "mean_confidence": round(statistics.mean(d.confidence for d in decisions), 3) if decisions else None,
            "mean_policy_entropy": round(statistics.mean(d.entropy for d in decisions), 3) if decisions else None,
            "decisions_per_battle": round(statistics.mean(turns), 1) if turns else None,
            **self.calibration(tags),
        }
