"""Win rate of KevPlayer against poke-env's baselines, plus latency and fallback stats.

    python evaluate.py --battles 50 --kev http://127.0.0.1:8009

Read the numbers this way: Random is the floor, MaxBasePower is 'knows attacking is good', SimpleHeuristics is the
teacher. A behaviour-cloned Kev should approach ~50% vs SimpleHeuristics; beating it clearly needs more than imitation.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from kev_showdown.player import KevPlayer
from kev_showdown.telemetry import configure, emit


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=50)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--kev", default="http://127.0.0.1:8009")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_replays", default="", help="directory for replays (good for the demo video)")
    ap.add_argument("--log_file", default="", help="also write structured JSONL logs here")
    a = ap.parse_args()

    log = configure(a.log_file or None)
    kev = KevPlayer(battle_format=a.format, kev_url=a.kev, verbose=a.verbose, max_concurrent_battles=1,
                    save_replays=a.save_replays or False, log_file=a.log_file or None)
    results: dict[str, str] = {}
    opponent_classes: list[type[Player]] = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    for opp_cls in opponent_classes:
        opp = opp_cls(battle_format=a.format, max_concurrent_battles=1)
        before_won, before_done = kev.n_won_battles, kev.n_finished_battles
        await kev.battle_against(opp, n_battles=a.battles)
        won = kev.n_won_battles - before_won
        done = kev.n_finished_battles - before_done
        results[opp_cls.__name__] = f"{won}/{done} ({100 * won / max(done, 1):.0f}%)"
        print(f"Kev vs {opp_cls.__name__}: {results[opp_cls.__name__]}")

    summary = {"win_rates": results, **kev.stats()}
    emit(log, "evaluate.session_summary", **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
