"""Integration tests against a REAL, running `kev.serve` instance -- nothing here is mocked, so this file cannot be
made to pass by faking a good-looking response; it only passes if an actual model answers correctly-shaped,
internally-consistent probabilities for a real battle position.

Needs: KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
(or set KEV_BASE_URL to point elsewhere). Skipped automatically when nothing answers there.

    pytest -m live
"""
from __future__ import annotations

import os
import random

import httpx
import pytest
from poke_env.battle import Battle

from kev_showdown.common import build_options, build_request
from kev_showdown.player import KevPlayer
from kev_showdown.types import ChoiceAnswer, NoulAnswer, SystemOneResponse

KEV_BASE_URL = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8009")


def _server_reachable() -> bool:
    try:
        r = httpx.get(f"{KEV_BASE_URL}/v1/models", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _server_reachable(), reason=f"no kev.serve reachable at {KEV_BASE_URL}")]


@pytest.mark.asyncio
async def test_real_model_answers_a_real_battle_request(battle: Battle) -> None:
    options = build_options(battle)
    req = build_request(battle, options, random.Random(0), ask_win=True)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{KEV_BASE_URL}/v1/systemone", json=req)
    r.raise_for_status()
    resp = SystemOneResponse.model_validate(r.json())

    action = resp.answers["action"]
    assert isinstance(action, ChoiceAnswer)
    assert action.choice in options                                       # picked one of the options we actually sent
    assert sum(action.probabilities.values()) == pytest.approx(1.0, abs=0.05)
    assert set(action.probabilities) == set(options)                      # no phantom or missing options

    win = resp.answers["win"]
    assert isinstance(win, NoulAnswer)
    assert 0.0 <= win.noul <= 1.0


@pytest.mark.asyncio
async def test_option_order_does_not_change_the_set_of_legal_answers(battle: Battle) -> None:
    """The model may weight options differently under different orderings, but it must never invent an option
    that wasn't offered -- a cheap real-model sanity check that a mock would never fail."""
    options = build_options(battle)
    picks = set()
    async with httpx.AsyncClient(timeout=30.0) as client:
        for seed in range(5):
            req = build_request(battle, options, random.Random(seed), ask_win=False)
            r = await client.post(f"{KEV_BASE_URL}/v1/systemone", json=req)
            r.raise_for_status()
            resp = SystemOneResponse.model_validate(r.json())
            action = resp.answers["action"]
            assert isinstance(action, ChoiceAnswer)
            picks.add(action.choice)
    assert picks <= set(options)


@pytest.mark.asyncio
async def test_kev_player_completes_a_real_decision_against_the_live_server(battle: Battle) -> None:
    player = KevPlayer(battle_format="gen9randombattle", kev_url=KEV_BASE_URL, max_concurrent_battles=1)
    order = await player.choose_move(battle)
    assert order is not None
    stats = player.stats()
    assert stats["decisions"] == 1
    assert stats["fallbacks"] == 0                                        # a real, healthy server must not fall back
    # a real model answering a real position spreads some probability around rather than collapsing onto one option
    entropy = stats["mean_policy_entropy"]
    assert isinstance(entropy, float) and 0.0 < entropy <= 1.0
