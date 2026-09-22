"""A poke-env Player whose every decision is one Kev /v1/systemone call.

Start Kev first, e.g.:
    KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
"""
from __future__ import annotations

import json
import random
import statistics
import time

import httpx
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import BattleOrder, Player
from pydantic import ValidationError

from .common import build_options, build_request
from .telemetry import configure, emit_warning, emit
from .types import ChoiceAnswer, NoulAnswer, SystemOneResponse

StatsDict = dict[str, "int | float | None"]


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
        # stats for the writeup
        self.latencies_ms: list[float] = []
        self.confidences: list[float] = []
        self.n_decisions = 0
        self.n_fallbacks = 0

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
            self.n_fallbacks += 1
            emit_warning(self.log, "kev.fallback", battle_tag=battle.battle_tag, turn=battle.turn,
                         error_type=type(e).__name__, error=str(e))
            if self.verbose:
                print(f"[kev] fallback on turn {battle.turn}: {e!r}")
            return self.choose_random_move(battle)
        latency_ms = (time.perf_counter() - t) * 1000
        self.latencies_ms.append(latency_ms)
        self.n_decisions += 1

        action = answers.answers.get("action")
        if not isinstance(action, ChoiceAnswer):
            self.n_fallbacks += 1
            emit_warning(self.log, "kev.fallback", battle_tag=battle.battle_tag, turn=battle.turn,
                         error_type="MissingActionAnswer", error="response had no 'action' choice answer")
            return self.choose_random_move(battle)

        self.confidences.append(action.confidence)
        win = answers.answers.get("win")
        p_win = win.noul if isinstance(win, NoulAnswer) else None
        emit(self.log, "kev.decision", battle_tag=battle.battle_tag, turn=battle.turn, pick=action.choice,
             confidence=action.confidence, p_win=p_win, latency_ms=round(latency_ms, 1))
        if self.verbose:
            print(f"[kev] {battle.battle_tag} t{battle.turn}: {action.choice} (conf {action.confidence}, p(win) {p_win})")
        if action.choice in options:
            return options[action.choice][0]
        self.n_fallbacks += 1
        emit_warning(self.log, "kev.fallback", battle_tag=battle.battle_tag, turn=battle.turn,
                     error_type="UnknownChoice", error=f"kev returned {action.choice!r}, not one of the offered options")
        return self.choose_random_move(battle)

    def stats(self) -> StatsDict:
        lat = self.latencies_ms
        return {
            "decisions": self.n_decisions,
            "fallbacks": self.n_fallbacks,
            "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else None,
            "mean_confidence": round(statistics.mean(self.confidences), 3) if self.confidences else None,
        }
